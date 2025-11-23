# mcp_embedding_server/services/embedding_service.py
# SPDX-License-Identifier: Apache-2.0

"""
MCP adapter for Corpus Embedding protocol.

This module exposes Corpus `EmbeddingProtocolV1` implementations as
embedding services within MCP servers and workflows, with:

- Seamless integration with MCP tool execution and resource handling
- Support for MCP session context and request tracing
- Context normalization for MCP-specific execution environment
- Framework-agnostic orchestration via `EmbeddingTranslator`
- Async-first design optimized for MCP's async architecture
- Rich error context attachment for observability in MCP workflows

The design follows MCP's protocol patterns while maintaining the
protocol-first Corpus embedding stack.

Service-level limits (MCP-facing; adapter may impose stricter limits):
- Maximum batch size: 1000 texts per request
- Maximum total text size: 1,000,000 characters per request

Resilience (rate limiting, circuit breaking, retries, deadlines, caching)
is expected to be provided by the underlying adapter, which should
typically subclass `BaseEmbeddingAdapter`. The MCP layer stays thin and
only adds MCP-specific semantics and observability.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import wraps, cached_property
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    TypedDict,
    TypeVar,
)

from corpus_sdk.core.context_translation import from_mcp
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.embedding.embedding_base import (
    EmbeddingProtocolV1,
    OperationContext,
    EmbeddingAdapterError,
    ResourceExhausted,
    TransientNetwork,
    Unavailable,
    DeadlineExceeded,
)
from corpus_sdk.embedding.framework_adapters.common.embedding_translation import (
    EmbeddingTranslator,
    BatchConfig,
    TextNormalizationConfig,
    create_embedding_translator,
)
from corpus_sdk.embedding.framework_adapters.common.framework_utils import (
    CoercionErrorCodes,
    coerce_embedding_matrix,
    coerce_embedding_vector,
    warn_if_extreme_batch,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Error codes (service + coercion)
# --------------------------------------------------------------------------- #


class ErrorCodes(CoercionErrorCodes):
    """
    Error code constants for the MCP embedding adapter.

    Extends `CoercionErrorCodes` so that shared coercion utilities can
    produce framework-consistent error codes while we add MCP-specific
    service codes here.
    """

    # Request / validation level
    EMPTY_REQUEST = "EMPTY_REQUEST"
    BATCH_SIZE_EXCEEDED = "BATCH_SIZE_EXCEEDED"
    TEXT_SIZE_EXCEEDED = "TEXT_SIZE_EXCEEDED"
    INVALID_TEXT_TYPE = "INVALID_TEXT_TYPE"

    # Service / adapter level
    EMBEDDING_EXTRACTION_ERROR = "EMBEDDING_EXTRACTION_ERROR"
    REQUEST_TIMEOUT = "REQUEST_TIMEOUT"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"

    # Coercion-level (used by framework_utils)
    INVALID_EMBEDDING_RESULT = "INVALID_EMBEDDING_RESULT"
    EMPTY_EMBEDDING_RESULT = "EMPTY_EMBEDDING_RESULT"
    EMBEDDING_CONVERSION_ERROR = "EMBEDDING_CONVERSION_ERROR"


# --------------------------------------------------------------------------- #
# MCP types & protocols
# --------------------------------------------------------------------------- #


class MCPContext(TypedDict, total=False):
    """Structured type for MCP execution context."""
    session_id: Optional[str]
    request_id: Optional[str]
    tool_name: Optional[str]
    server_id: Optional[str]
    client_id: Optional[str]
    trace_id: Optional[str]


class MCPEmbedder(Protocol):
    """
    Protocol representing the embedder interface for MCP servers.

    This allows type-safe integration with MCP's tool execution system
    without requiring a hard dependency on MCP at type-check time.
    """

    async def embed_documents(
        self,
        texts: List[str],
        *,
        mcp_context: Optional[MCPContext] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Embed multiple documents for MCP tool execution."""
        ...

    async def embed_query(
        self,
        text: str,
        *,
        mcp_context: Optional[MCPContext] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """Embed a single query for MCP retrieval operations."""
        ...


def with_embedding_error_context(
    operation: str,
    **context_kwargs: Any,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator to automatically attach error context to async embedding exceptions.
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                enhanced_context = context_kwargs.copy()
                attach_context(
                    exc,
                    framework="mcp",
                    operation=f"embedding_{operation}",
                    **enhanced_context,
                )
                raise

        return wrapper

    return decorator


class MCPConfig(TypedDict, total=False):
    """Structured configuration for MCP-specific settings."""
    max_concurrent_requests: int
    fallback_to_simple_context: bool
    enable_session_context_propagation: bool
    tool_aware_batching: bool


@dataclass
class MCPEmbeddingResult:
    """Structured result for MCP embedding operations."""
    embeddings: List[List[float]]
    model: str
    request_id: str
    processing_time: float
    total_tokens: Optional[int] = None


class MCPEmbeddingServiceError(Exception):
    """Base exception for MCP embedding service errors."""

    def __init__(self, message: str, code: str, request_id: Optional[str] = None):
        super().__init__(message)
        self.code = code
        self.request_id = request_id
        self.message = message


# --------------------------------------------------------------------------- #
# Main MCP embedding service
# --------------------------------------------------------------------------- #


class CorpusMCPEmbeddings:
    """
    MCP embedding service backed by a Corpus `EmbeddingProtocolV1` adapter.

    Implements the `MCPEmbedder` protocol for seamless integration with
    MCP servers and tool execution workflows.

    Architecture
    ------------
    - This service uses the shared `EmbeddingTranslator` to:
        * Normalize OperationContext from MCP context
        * Build `EmbedSpec` / `BatchEmbedSpec` from raw MCP text inputs
        * Translate results back into framework-level shapes
    - Resilience (deadlines, circuit breaking, rate limiting, caching,
      adapter-level retries) is owned by the embedding adapter itself
      (typically `BaseEmbeddingAdapter`).

    This layer adds MCP-specific:
    - Per-process concurrency limiting (asyncio.Semaphore)
    - MCP-specific request validation (batch/text size)
    - Mapping from adapter errors to MCP service error codes
    - Observability in terms of MCP tool/session IDs
    """

    def __init__(
        self,
        corpus_adapter: EmbeddingProtocolV1,
        model: Optional[str] = None,
        batch_config: Optional[BatchConfig] = None,
        text_normalization_config: Optional[TextNormalizationConfig] = None,
        mcp_config: Optional[MCPConfig] = None,
    ):
        self.corpus_adapter = corpus_adapter
        self.model = model
        self.batch_config = batch_config
        self.text_normalization_config = text_normalization_config
        self.mcp_config = self._validate_mcp_config(mcp_config or {})

        # Service state
        self._active_requests: int = 0
        self._active_requests_lock = asyncio.Lock()
        self._request_semaphore = asyncio.Semaphore(
            self.mcp_config["max_concurrent_requests"]
        )

        # Simple protocol metrics (for health/introspection)
        self._protocol_success_count: int = 0
        self._protocol_error_count: int = 0
        self._protocol_total_latency: float = 0.0  # seconds

        self._protocol_embed_success_count: int = 0
        self._protocol_embed_error_count: int = 0
        self._protocol_embed_total_latency: float = 0.0

        self._protocol_batch_success_count: int = 0
        self._protocol_batch_error_count: int = 0
        self._protocol_batch_total_latency: float = 0.0

        logger.info(
            "CorpusMCPEmbeddings initialized with model=%s, max_concurrent=%d",
            model or "default",
            self.mcp_config["max_concurrent_requests"],
        )

    # ------------------------------------------------------------------ #
    # Translator (framework-agnostic orchestrator)
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> EmbeddingTranslator:
        """
        Lazily construct the `EmbeddingTranslator` for MCP.

        This orchestrator:
        - Normalizes OperationContext
        - Builds EmbedSpec / BatchEmbedSpec
        - Calls the underlying EmbeddingProtocolV1 adapter
        - Translates results to framework-level shapes (dicts)
        """
        translator = create_embedding_translator(
            adapter=self.corpus_adapter,
            framework="mcp",
            translator=None,
            batch_config=self.batch_config,
            text_normalization_config=self.text_normalization_config,
        )
        logger.debug(
            "EmbeddingTranslator initialized for MCP with model=%s",
            self.model or "default",
        )
        return translator

    # ------------------------------------------------------------------ #
    # Configuration validation
    # ------------------------------------------------------------------ #

    def _validate_mcp_config(self, config: MCPConfig) -> MCPConfig:
        """Validate and normalize MCP configuration with sensible defaults."""
        validated: MCPConfig = dict(config)

        validated.setdefault("max_concurrent_requests", 100)
        validated.setdefault("fallback_to_simple_context", True)
        validated.setdefault("enable_session_context_propagation", True)
        validated.setdefault("tool_aware_batching", False)

        try:
            validated["max_concurrent_requests"] = int(
                validated["max_concurrent_requests"]
            )
        except (TypeError, ValueError):
            raise ValueError("max_concurrent_requests must be an integer") from None

        if validated["max_concurrent_requests"] <= 0:
            raise ValueError("max_concurrent_requests must be positive")

        # Bool coercion for robustness
        for key in ("fallback_to_simple_context", "enable_session_context_propagation", "tool_aware_batching"):
            validated[key] = bool(validated[key])

        return validated

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @asynccontextmanager
    async def _track_active_request(self) -> AsyncIterator[None]:
        """Context manager for thread-safe active request tracking."""
        async with self._active_requests_lock:
            self._active_requests += 1
        try:
            yield
        finally:
            async with self._active_requests_lock:
                self._active_requests -= 1

    def _build_contexts(
        self,
        *,
        mcp_context: Optional[MCPContext] = None,
        model: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[Optional[OperationContext], Dict[str, Any]]:
        """
        Build contexts for MCP execution environment with request ID propagation.

        Returns
        -------
        Tuple of:
        - core_ctx: core OperationContext or None
        - framework_ctx: MCP-specific context for translator
        """
        enhanced_mcp_context: Dict[str, Any] = dict(mcp_context) if mcp_context else {}
        if request_id and "request_id" not in enhanced_mcp_context:
            enhanced_mcp_context["request_id"] = request_id

        core_ctx: Optional[OperationContext] = None
        framework_ctx: Dict[str, Any] = {
            "framework": "mcp",
            "config": self.mcp_config,
        }

        if enhanced_mcp_context:
            try:
                core_ctx = from_mcp(enhanced_mcp_context)
                logger.debug(
                    "Created OperationContext for MCP session=%s, tool=%s, request=%s",
                    enhanced_mcp_context.get("session_id", "unknown"),
                    enhanced_mcp_context.get("tool_name", "unknown"),
                    request_id or enhanced_mcp_context.get("request_id", "unknown"),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to create OperationContext from mcp_context: %s",
                    exc,
                )
                if not self.mcp_config["fallback_to_simple_context"]:
                    core_ctx = None

        effective_model = model or self.model
        if effective_model:
            framework_ctx["model"] = effective_model

        if enhanced_mcp_context:
            framework_ctx.update(
                {
                    "session_id": enhanced_mcp_context.get("session_id"),
                    "tool_name": enhanced_mcp_context.get("tool_name"),
                    "server_id": enhanced_mcp_context.get("server_id"),
                    "client_id": enhanced_mcp_context.get("client_id"),
                    "trace_id": enhanced_mcp_context.get("trace_id"),
                    "request_id": enhanced_mcp_context.get("request_id", request_id),
                    **kwargs,
                }
            )

            if (
                self.mcp_config["tool_aware_batching"]
                and enhanced_mcp_context.get("tool_name")
            ):
                framework_ctx["batch_strategy"] = (
                    f"tool_aware_{enhanced_mcp_context['tool_name']}"
                )

        return core_ctx, framework_ctx

    def _get_translation_op_ctx(
        self,
        core_ctx: Optional[OperationContext],
        request_id: str,
    ) -> OperationContext:
        """
        Provide a non-None OperationContext for translator operations.

        If a core OperationContext is already available, it is reused.
        Otherwise, a minimal context is created with the given request ID.
        """
        if core_ctx is not None:
            return core_ctx

        return OperationContext(
            request_id=request_id,
            idempotency_key=None,
            deadline_ms=None,
            traceparent=None,
            tenant=None,
            metrics=None,
            attrs={},
        )

    # ------------------------------------------------------------------ #
    # Error mapping (adapter → MCP)
    # ------------------------------------------------------------------ #

    def _map_adapter_error_to_service_error(
        self,
        exc: Exception,
        request_id: str,
    ) -> MCPEmbeddingServiceError:
        """
        Map adapter-level errors to service-level error codes.

        Prefer the structured `EmbeddingAdapterError` taxonomy when available,
        falling back to a generic mapping otherwise.
        """
        if isinstance(exc, EmbeddingAdapterError):
            code = (exc.code or "").upper()
            message = exc.message or str(exc)

            # Rate / quota / resource exhaustion → SERVICE_UNAVAILABLE (or caller can treat separately)
            if (
                isinstance(exc, ResourceExhausted)
                or code in {"RESOURCE_EXHAUSTED", "RATE_LIMIT", "RATE_LIMIT_EXCEEDED"}
                or exc.resource_scope in {"rate_limit", "quota", "model"}
            ):
                return MCPEmbeddingServiceError(
                    f"Rate or resource limit exceeded: {message}",
                    ErrorCodes.SERVICE_UNAVAILABLE,
                    request_id,
                )

            # Deadline / timeout → REQUEST_TIMEOUT
            if isinstance(exc, DeadlineExceeded) or code == "DEADLINE_EXCEEDED":
                return MCPEmbeddingServiceError(
                    f"Request deadline exceeded: {message}",
                    ErrorCodes.REQUEST_TIMEOUT,
                    request_id,
                )

            # Transient / service unavailability → SERVICE_UNAVAILABLE
            if (
                isinstance(exc, Unavailable)
                or isinstance(exc, TransientNetwork)
                or code in {"UNAVAILABLE", "SERVICE_UNAVAILABLE", "TRANSIENT_NETWORK"}
            ):
                return MCPEmbeddingServiceError(
                    f"Embedding backend unavailable: {message}",
                    ErrorCodes.SERVICE_UNAVAILABLE,
                    request_id,
                )

            # Everything else: treat as generic embedding error
            return MCPEmbeddingServiceError(
                f"Embedding adapter error: {message}",
                ErrorCodes.EMBEDDING_EXTRACTION_ERROR,
                request_id,
            )

        # Non-EmbeddingAdapterError: treat as generic service error
        return MCPEmbeddingServiceError(
            f"Embedding service error: {str(exc)}",
            ErrorCodes.EMBEDDING_EXTRACTION_ERROR,
            request_id,
        )

    # ------------------------------------------------------------------ #
    # Result coercion (shared with other framework adapters)
    # ------------------------------------------------------------------ #

    def _coerce_embedding_matrix(self, result: Any) -> List[List[float]]:
        """
        Coerce translator result into an embedding matrix using the
        shared framework_utils implementation.
        """
        return coerce_embedding_matrix(
            result=result,
            error_codes=ErrorCodes,
            logger=logger,
        )

    def _coerce_embedding_vector(self, result: Any) -> List[float]:
        """
        Coerce translator result into a single embedding vector.

        Delegates to the shared framework_utils implementation, which
        handles dicts like {"embedding": [...]} and matrices alike.
        """
        return coerce_embedding_vector(
            result=result,
            error_codes=ErrorCodes,
            logger=logger,
        )

    def _validate_embedding_request(self, texts: List[str], request_id: str) -> None:
        """Comprehensive request validation (MCP layer)."""
        if not texts:
            raise MCPEmbeddingServiceError(
                "No texts provided",
                ErrorCodes.EMPTY_REQUEST,
                request_id,
            )

        if len(texts) > 1000:
            raise MCPEmbeddingServiceError(
                f"Batch size {len(texts)} exceeds maximum 1000",
                ErrorCodes.BATCH_SIZE_EXCEEDED,
                request_id,
            )

        total_chars = sum(len(text) for text in texts)
        if total_chars > 1_000_000:
            raise MCPEmbeddingServiceError(
                f"Total text size {total_chars} characters exceeds limit",
                ErrorCodes.TEXT_SIZE_EXCEEDED,
                request_id,
            )

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise MCPEmbeddingServiceError(
                    f"Text at index {i} is not a string",
                    ErrorCodes.INVALID_TEXT_TYPE,
                    request_id,
                )

    # ------------------------------------------------------------------ #
    # Core Embedding API (MCP Compatible)
    # ------------------------------------------------------------------ #

    @with_embedding_error_context("documents")
    async def embed_documents(
        self,
        texts: List[str],
        *,
        mcp_context: Optional[MCPContext] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Async embedding for multiple documents.

        Uses the shared `EmbeddingTranslator` for protocol orchestration and
        the underlying EmbeddingProtocolV1 adapter for execution.
        """
        request_id = f"embed_docs_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        self._validate_embedding_request(texts, request_id)

        warn_if_extreme_batch(
            framework="mcp",
            texts=texts,
            op_name="embed_documents",
            batch_config=self.batch_config,
            logger=logger,
        )

        core_ctx, framework_ctx = self._build_contexts(
            mcp_context=mcp_context,
            model=model,
            request_id=request_id,
            **kwargs,
        )

        # Ensure translator has a non-None OperationContext
        translation_ctx = self._get_translation_op_ctx(core_ctx, request_id)

        logger.debug(
            "Embedding %d documents for MCP tool=%s, session=%s, request=%s",
            len(texts),
            mcp_context.get("tool_name", "unknown") if mcp_context else "unknown",
            mcp_context.get("session_id", "unknown") if mcp_context else "unknown",
            request_id,
        )

        async with self._track_active_request():
            async with self._request_semaphore:
                try:
                    protocol_start = time.time()
                    translated = await self._translator.arun_embed(
                        raw_texts=texts,
                        op_ctx=translation_ctx,
                        framework_ctx=framework_ctx,
                    )
                except EmbeddingAdapterError as exc:
                    self._protocol_error_count += 1
                    self._protocol_batch_error_count += 1
                    raise self._map_adapter_error_to_service_error(exc, request_id)
                except asyncio.TimeoutError as exc:
                    # In case underlying stack bubbles raw TimeoutError
                    self._protocol_error_count += 1
                    self._protocol_batch_error_count += 1
                    raise MCPEmbeddingServiceError(
                        "Embedding request timed out",
                        ErrorCodes.REQUEST_TIMEOUT,
                        request_id,
                    ) from exc
                except Exception as exc:  # noqa: BLE001
                    self._protocol_error_count += 1
                    self._protocol_batch_error_count += 1
                    raise self._map_adapter_error_to_service_error(exc, request_id)

                latency = time.time() - protocol_start
                self._protocol_success_count += 1
                self._protocol_total_latency += latency
                self._protocol_batch_success_count += 1
                self._protocol_batch_total_latency += latency

                processing_time = time.time() - start_time
                logger.debug(
                    "Embedding completed in %.3fs for request %s",
                    processing_time,
                    request_id,
                )

                return self._coerce_embedding_matrix(translated)

    @with_embedding_error_context("query")
    async def embed_query(
        self,
        text: str,
        *,
        mcp_context: Optional[MCPContext] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Async embedding for a single query.

        Uses the shared `EmbeddingTranslator` and underlying adapter.
        """
        request_id = f"embed_query_{uuid.uuid4().hex[:8]}"

        # Reuse batch validator with length 1 for consistent limits
        self._validate_embedding_request([text], request_id)

        core_ctx, framework_ctx = self._build_contexts(
            mcp_context=mcp_context,
            model=model,
            request_id=request_id,
            **kwargs,
        )

        translation_ctx = self._get_translation_op_ctx(core_ctx, request_id)

        logger.debug(
            "Embedding query for MCP tool=%s, request=%s",
            mcp_context.get("tool_name", "unknown") if mcp_context else "unknown",
            request_id,
        )

        async with self._track_active_request():
            async with self._request_semaphore:
                try:
                    protocol_start = time.time()
                    translated = await self._translator.arun_embed(
                        raw_texts=text,
                        op_ctx=translation_ctx,
                        framework_ctx=framework_ctx,
                    )
                except EmbeddingAdapterError as exc:
                    self._protocol_error_count += 1
                    self._protocol_embed_error_count += 1
                    raise self._map_adapter_error_to_service_error(exc, request_id)
                except asyncio.TimeoutError as exc:
                    self._protocol_error_count += 1
                    self._protocol_embed_error_count += 1
                    raise MCPEmbeddingServiceError(
                        "Query embedding request timed out",
                        ErrorCodes.REQUEST_TIMEOUT,
                        request_id,
                    ) from exc
                except Exception as exc:  # noqa: BLE001
                    self._protocol_error_count += 1
                    self._protocol_embed_error_count += 1
                    raise self._map_adapter_error_to_service_error(exc, request_id)

                latency = time.time() - protocol_start
                self._protocol_success_count += 1
                self._protocol_total_latency += latency
                self._protocol_embed_success_count += 1
                self._protocol_embed_total_latency += latency

                return self._coerce_embedding_vector(translated)

    # ------------------------------------------------------------------ #
    # Health check / introspection
    # ------------------------------------------------------------------ #

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for MCP server integration."""
        async with self._active_requests_lock:
            active_requests = self._active_requests

        avg_protocol_latency_ms: Optional[float] = None
        if self._protocol_success_count > 0:
            avg_protocol_latency_ms = (
                self._protocol_total_latency / self._protocol_success_count * 1000.0
            )

        avg_embed_latency_ms: Optional[float] = None
        if self._protocol_embed_success_count > 0:
            avg_embed_latency_ms = (
                self._protocol_embed_total_latency
                / self._protocol_embed_success_count
                * 1000.0
            )

        avg_batch_latency_ms: Optional[float] = None
        if self._protocol_batch_success_count > 0:
            avg_batch_latency_ms = (
                self._protocol_batch_total_latency
                / self._protocol_batch_success_count
                * 1000.0
            )

        health_status: Dict[str, Any] = {
            "status": "healthy",
            "active_requests": active_requests,
            "protocol_success_count": self._protocol_success_count,
            "protocol_error_count": self._protocol_error_count,
            "avg_protocol_latency_ms": avg_protocol_latency_ms,
            "protocol_embed_success_count": self._protocol_embed_success_count,
            "protocol_embed_error_count": self._protocol_embed_error_count,
            "avg_protocol_embed_latency_ms": avg_embed_latency_ms,
            "protocol_batch_success_count": self._protocol_batch_success_count,
            "protocol_batch_error_count": self._protocol_batch_error_count,
            "avg_protocol_batch_latency_ms": avg_batch_latency_ms,
            "max_concurrent_requests": self.mcp_config["max_concurrent_requests"],
        }

        # Smoke test with actual embedding to surface adapter-level health
        try:
            test_embedding = await self.embed_query(
                "health_check",
                mcp_context={
                    "tool_name": "health_check",
                    "session_id": "health_check",
                },
            )
            health_status["service_test"] = "passed"
            health_status["embedding_dimension"] = len(test_embedding)
        except Exception as exc:  # noqa: BLE001
            health_status["service_test"] = f"failed: {str(exc)}"
            health_status["status"] = "degraded"

        return health_status


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def create_embedder(
    corpus_adapter: EmbeddingProtocolV1,
    model: Optional[str] = None,
    **kwargs: Any,
) -> MCPEmbedder:
    """
    Create an MCP-compatible embedder for seamless server integration.

    Example:
    ```python
    from mcp_embedding_server.services.embedding_service import create_embedder

    server_embedder = create_embedder(
        corpus_adapter=server_adapter,
        model="text-embedding-3-large",
        mcp_config={
            "max_concurrent_requests": 50,
            "enable_session_context_propagation": True,
        },
    )
    ```
    """
    embedder = CorpusMCPEmbeddings(
        corpus_adapter=corpus_adapter,
        model=model,
        **kwargs,
    )

    logger.info(
        "MCP embedder created successfully with model=%s",
        model or "default",
    )

    return embedder


__all__ = [
    "CorpusMCPEmbeddings",
    "MCPEmbedder",
    "MCPContext",
    "MCPConfig",
    "create_embedder",
    "ErrorCodes",
]
