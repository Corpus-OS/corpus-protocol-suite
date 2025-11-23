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
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    REQUEST_TIMEOUT = "REQUEST_TIMEOUT"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    CIRCUIT_BREAKER_OPEN = "CIRCUIT_BREAKER_OPEN"

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
    max_embedding_retries: int
    fallback_to_simple_context: bool
    enable_session_context_propagation: bool
    tool_aware_batching: bool
    max_concurrent_requests: int
    rate_limit_per_minute: int
    circuit_breaker_failure_threshold: int
    circuit_breaker_reset_timeout: int  # seconds


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
    - It delegates resilience (deadlines, breaker, rate-limit, caching)
      to the underlying adapter, which should typically subclass
      `BaseEmbeddingAdapter`.

    This layer adds MCP-specific:
    - Concurrency limiting (per-process semaphore)
    - Sliding-window rate limiting (requests per minute)
    - Retry loop tuned for EmbeddingAdapterError semantics
    - Circuit breaker at the MCP endpoint level
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
        self._rate_limit_lock = asyncio.Lock()
        self._request_semaphore = asyncio.Semaphore(
            self.mcp_config["max_concurrent_requests"]
        )
        self._rate_limit_tracker: List[float] = []

        # Circuit breaker state (service-level; adapter has its own)
        self._circuit_failure_count: int = 0
        self._circuit_open_until: float = 0.0

        # Protocol-level metrics (approximate, via translator calls)
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
        logger.debug("EmbeddingTranslator initialized for MCP with model=%s", self.model or "default")
        return translator

    # ------------------------------------------------------------------ #
    # Configuration validation
    # ------------------------------------------------------------------ #

    def _validate_mcp_config(self, config: MCPConfig) -> MCPConfig:
        """Validate and normalize MCP configuration with sensible defaults."""
        validated: MCPConfig = dict(config)

        # Defaults
        validated.setdefault("max_embedding_retries", 3)
        validated.setdefault("fallback_to_simple_context", True)
        validated.setdefault("enable_session_context_propagation", True)
        validated.setdefault("tool_aware_batching", False)
        validated.setdefault("max_concurrent_requests", 100)
        validated.setdefault("rate_limit_per_minute", 1000)
        validated.setdefault("circuit_breaker_failure_threshold", 5)
        validated.setdefault("circuit_breaker_reset_timeout", 30)

        # Coerce numeric fields to int and validate
        numeric_keys = [
            "max_embedding_retries",
            "max_concurrent_requests",
            "rate_limit_per_minute",
            "circuit_breaker_failure_threshold",
            "circuit_breaker_reset_timeout",
        ]
        for key in numeric_keys:
            try:
                validated[key] = int(validated[key])
            except (TypeError, ValueError):
                raise ValueError(f"{key} must be an integer") from None

        if validated["max_concurrent_requests"] <= 0:
            raise ValueError("max_concurrent_requests must be positive")
        if validated["rate_limit_per_minute"] <= 0:
            raise ValueError("rate_limit_per_minute must be positive")
        if validated["max_embedding_retries"] < 0:
            raise ValueError("max_embedding_retries cannot be negative")
        if validated["circuit_breaker_failure_threshold"] <= 0:
            raise ValueError("circuit_breaker_failure_threshold must be positive")
        if validated["circuit_breaker_reset_timeout"] <= 0:
            raise ValueError("circuit_breaker_reset_timeout must be positive")

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

    async def _check_rate_limit(self) -> bool:
        """
        Check rate limiting with sliding window and thread-safe locking.

        Returns True if the request is allowed, False if the rate limit is exceeded.
        """
        async with self._rate_limit_lock:
            now = time.time()
            window_start = now - 60  # 1 minute window

            # Clean old requests
            self._rate_limit_tracker = [
                t for t in self._rate_limit_tracker if t > window_start
            ]

            if len(self._rate_limit_tracker) >= self.mcp_config["rate_limit_per_minute"]:
                return False

            self._rate_limit_tracker.append(now)
            return True

    # ------------------------------------------------------------------ #
    # Circuit breaker (service-level)
    # ------------------------------------------------------------------ #

    def _is_circuit_open(self) -> bool:
        """Return True if the circuit breaker is currently open."""
        now = time.time()
        if self._circuit_open_until == 0:
            return False
        if now >= self._circuit_open_until:
            self._circuit_open_until = 0.0
            self._circuit_failure_count = 0
            return False
        return True

    def _record_success(self) -> None:
        """Record a successful protocol call and reset circuit breaker failures."""
        self._circuit_failure_count = 0
        self._circuit_open_until = 0.0

    def _record_failure(self) -> None:
        """Record a failed protocol call and open circuit if threshold reached."""
        self._circuit_failure_count += 1
        threshold = self.mcp_config["circuit_breaker_failure_threshold"]
        if self._circuit_failure_count >= threshold and self._circuit_open_until == 0.0:
            reset_timeout = self.mcp_config["circuit_breaker_reset_timeout"]
            self._circuit_open_until = time.time() + reset_timeout
            logger.error(
                "MCP circuit breaker opened after %d consecutive failures; cooldown=%ds",
                self._circuit_failure_count,
                reset_timeout,
            )

    # ------------------------------------------------------------------ #
    # Retry & timeout logic
    # ------------------------------------------------------------------ #

    async def _execute_with_retries(
        self,
        operation: Callable[[], Awaitable[Any]],
        request_id: str,
        operation_name: str,
    ) -> Any:
        """
        Execute embedding operation with retry logic for transient errors.

        Retries are applied to the provided operation. Circuit breaker state is
        updated based on success/failure outcomes.
        """
        max_retries = self.mcp_config["max_embedding_retries"]
        last_exception: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                result = await operation()
                self._record_success()
                return result
            except Exception as exc:  # noqa: BLE001
                last_exception = exc

                if not self._is_retryable_error(exc):
                    break

                if attempt < max_retries:
                    wait_time = self._get_retry_delay(attempt)
                    logger.warning(
                        "Retryable error in %s (attempt %d/%d), retrying in %.2fs: %s",
                        operation_name,
                        attempt + 1,
                        max_retries + 1,
                        wait_time,
                        str(exc),
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        "All %d retries exhausted for %s (request_id=%s)",
                        max_retries + 1,
                        operation_name,
                        request_id,
                    )

        if last_exception is None:
            self._record_failure()
            raise MCPEmbeddingServiceError(
                "Unknown error in retry loop",
                ErrorCodes.EMBEDDING_EXTRACTION_ERROR,
                request_id,
            )

        self._record_failure()
        raise self._map_adapter_error_to_service_error(last_exception, request_id)

    def _is_retryable_error(self, exc: Exception) -> bool:
        """
        Determine if an error is retryable based on EmbeddingAdapterError semantics
        and, as a fallback, string patterns.
        """
        if isinstance(exc, EmbeddingAdapterError):
            code = (exc.code or "").upper()
            retryable_codes = {
                "RESOURCE_EXHAUSTED",
                "RATE_LIMIT",
                "RATE_LIMIT_EXCEEDED",
                "TRANSIENT_NETWORK",
                "UNAVAILABLE",
                "SERVICE_UNAVAILABLE",
            }
            if code in retryable_codes:
                return True
            # If adapter provides an explicit retry_after_ms hint, treat as retryable
            if exc.retry_after_ms is not None:
                return True
            # Deadline / bad-request style errors are not retryable
            return False

        # Fallback: string pattern matching for non-adapter exceptions
        error_str = str(exc).lower()
        retryable_patterns = [
            "timeout",
            "deadline",
            "rate limit",
            "rate_limit",
            "too many requests",
            "server error",
            "service unavailable",
            "temporary",
            "retry",
            "connection",
            "network",
            "gateway",
        ]
        return any(pattern in error_str for pattern in retryable_patterns)

    def _get_retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter."""
        base_delay = 0.5
        max_delay = 10.0
        delay = min(base_delay * (2**attempt), max_delay)
        jitter = 0.5 + (uuid.uuid4().int % 1000) / 1000.0
        return delay * jitter

    def _map_adapter_error_to_service_error(
        self,
        exc: Exception,
        request_id: str,
    ) -> MCPEmbeddingServiceError:
        """
        Map adapter-level errors to service-level error codes.

        Prefer the structured `EmbeddingAdapterError` taxonomy when available,
        falling back to string pattern matching otherwise.
        """
        if isinstance(exc, EmbeddingAdapterError):
            code = (exc.code or "").upper()
            message = exc.message or str(exc)

            # Rate / quota / resource exhaustion
            if (
                isinstance(exc, ResourceExhausted)
                or code in {"RESOURCE_EXHAUSTED", "RATE_LIMIT", "RATE_LIMIT_EXCEEDED"}
                or exc.resource_scope in {"rate_limit", "quota", "model"}
            ):
                return MCPEmbeddingServiceError(
                    f"Rate or resource limit exceeded: {message}",
                    ErrorCodes.RATE_LIMIT_EXCEEDED,
                    request_id,
                )

            # Deadline / timeout
            if isinstance(exc, DeadlineExceeded) or code == "DEADLINE_EXCEEDED":
                return MCPEmbeddingServiceError(
                    f"Request deadline exceeded: {message}",
                    ErrorCodes.REQUEST_TIMEOUT,
                    request_id,
                )

            # Transient / service unavailability
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

        # Non-EmbeddingAdapterError: string pattern fallback
        error_str = str(exc).lower()

        if any(p in error_str for p in ["rate limit", "rate_limit", "too many requests"]):
            return MCPEmbeddingServiceError(
                f"Rate limit exceeded: {str(exc)}",
                ErrorCodes.RATE_LIMIT_EXCEEDED,
                request_id,
            )
        if any(p in error_str for p in ["timeout", "deadline"]):
            return MCPEmbeddingServiceError(
                f"Request timeout: {str(exc)}",
                ErrorCodes.REQUEST_TIMEOUT,
                request_id,
            )
        if any(p in error_str for p in ["unavailable", "down", "maintenance"]):
            return MCPEmbeddingServiceError(
                f"Service unavailable: {str(exc)}",
                ErrorCodes.SERVICE_UNAVAILABLE,
                request_id,
            )

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

        if self._is_circuit_open():
            raise MCPEmbeddingServiceError(
                "Embedding circuit breaker is open",
                ErrorCodes.CIRCUIT_BREAKER_OPEN,
                request_id,
            )

        if not await self._check_rate_limit():
            raise MCPEmbeddingServiceError(
                f"Rate limit exceeded: {self.mcp_config['rate_limit_per_minute']} "
                "requests per minute",
                ErrorCodes.RATE_LIMIT_EXCEEDED,
                request_id,
            )

        self._validate_embedding_request(texts, request_id)

        # Batch size observability
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

        if core_ctx is not None and self.mcp_config["enable_session_context_propagation"]:
            framework_ctx["_operation_context"] = core_ctx

        logger.debug(
            "Embedding %d documents for MCP tool=%s, session=%s, request=%s",
            len(texts),
            mcp_context.get("tool_name", "unknown") if mcp_context else "unknown",
            mcp_context.get("session_id", "unknown") if mcp_context else "unknown",
            request_id,
        )

        timeout: Optional[float] = None
        if core_ctx is not None and getattr(core_ctx, "deadline_ms", None):
            timeout = core_ctx.deadline_ms / 1000.0

        translation_ctx = self._get_translation_op_ctx(core_ctx, request_id)

        async with self._track_active_request():
            try:
                async with self._request_semaphore:

                    async def embed_operation() -> Any:
                        """
                        Execute batch embedding via the EmbeddingTranslator.

                        The translator decides between embed vs batch_embed;
                        for a list of texts, it routes to batch_embed and
                        returns a dict with "embeddings".
                        """
                        protocol_start = time.time()
                        try:
                            translated = await self._translator.arun_embed(
                                raw_texts=texts,
                                op_ctx=translation_ctx,
                                framework_ctx=framework_ctx,
                            )
                        except Exception:
                            self._protocol_error_count += 1
                            self._protocol_batch_error_count += 1
                            raise
                        else:
                            latency = time.time() - protocol_start
                            self._protocol_success_count += 1
                            self._protocol_total_latency += latency
                            self._protocol_batch_success_count += 1
                            self._protocol_batch_total_latency += latency
                            return translated

                    if timeout is not None:
                        translated = await asyncio.wait_for(
                            self._execute_with_retries(
                                embed_operation,
                                request_id,
                                "embed_documents",
                            ),
                            timeout=timeout,
                        )
                    else:
                        translated = await self._execute_with_retries(
                            embed_operation,
                            request_id,
                            "embed_documents",
                        )

                    processing_time = time.time() - start_time
                    logger.debug(
                        "Embedding completed in %.3fs for request %s",
                        processing_time,
                        request_id,
                    )

                    return self._coerce_embedding_matrix(translated)

            except asyncio.TimeoutError:
                raise MCPEmbeddingServiceError(
                    "Embedding request timed out",
                    ErrorCodes.REQUEST_TIMEOUT,
                    request_id,
                ) from None

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

        if self._is_circuit_open():
            raise MCPEmbeddingServiceError(
                "Embedding circuit breaker is open",
                ErrorCodes.CIRCUIT_BREAKER_OPEN,
                request_id,
            )

        if not await self._check_rate_limit():
            raise MCPEmbeddingServiceError(
                f"Rate limit exceeded: {self.mcp_config['rate_limit_per_minute']} "
                "requests per minute",
                ErrorCodes.RATE_LIMIT_EXCEEDED,
                request_id,
            )

        self._validate_embedding_request([text], request_id)

        core_ctx, framework_ctx = self._build_contexts(
            mcp_context=mcp_context,
            model=model,
            request_id=request_id,
            **kwargs,
        )

        if core_ctx is not None and self.mcp_config["enable_session_context_propagation"]:
            framework_ctx["_operation_context"] = core_ctx

        logger.debug(
            "Embedding query for MCP tool=%s, request=%s",
            mcp_context.get("tool_name", "unknown") if mcp_context else "unknown",
            request_id,
        )

        timeout: Optional[float] = None
        if core_ctx is not None and getattr(core_ctx, "deadline_ms", None):
            timeout = core_ctx.deadline_ms / 1000.0

        translation_ctx = self._get_translation_op_ctx(core_ctx, request_id)

        async with self._track_active_request():
            try:
                async with self._request_semaphore:

                    async def embed_operation() -> Any:
                        """
                        Execute single-text embedding via the EmbeddingTranslator.
                        """
                        protocol_start = time.time()
                        try:
                            translated = await self._translator.arun_embed(
                                raw_texts=text,
                                op_ctx=translation_ctx,
                                framework_ctx=framework_ctx,
                            )
                        except Exception:
                            self._protocol_error_count += 1
                            self._protocol_embed_error_count += 1
                            raise
                        else:
                            latency = time.time() - protocol_start
                            self._protocol_success_count += 1
                            self._protocol_total_latency += latency
                            self._protocol_embed_success_count += 1
                            self._protocol_embed_total_latency += latency
                            return translated

                    if timeout is not None:
                        translated = await asyncio.wait_for(
                            self._execute_with_retries(
                                embed_operation,
                                request_id,
                                "embed_query",
                            ),
                            timeout=timeout,
                        )
                    else:
                        translated = await self._execute_with_retries(
                            embed_operation,
                            request_id,
                            "embed_query",
                        )

                    return self._coerce_embedding_vector(translated)

            except asyncio.TimeoutError:
                raise MCPEmbeddingServiceError(
                    "Query embedding request timed out",
                    ErrorCodes.REQUEST_TIMEOUT,
                    request_id,
                ) from None

    # ------------------------------------------------------------------ #
    # Health check / introspection
    # ------------------------------------------------------------------ #

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for MCP server integration."""
        async with self._active_requests_lock:
            active_requests = self._active_requests

        rate_limit_remaining = (
            self.mcp_config["rate_limit_per_minute"] - len(self._rate_limit_tracker)
        )

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
            "rate_limit_remaining": rate_limit_remaining,
            "protocol_success_count": self._protocol_success_count,
            "protocol_error_count": self._protocol_error_count,
            "avg_protocol_latency_ms": avg_protocol_latency_ms,
            "protocol_embed_success_count": self._protocol_embed_success_count,
            "protocol_embed_error_count": self._protocol_embed_error_count,
            "avg_protocol_embed_latency_ms": avg_embed_latency_ms,
            "protocol_batch_success_count": self._protocol_batch_success_count,
            "protocol_batch_error_count": self._protocol_batch_error_count,
            "avg_protocol_batch_latency_ms": avg_batch_latency_ms,
            "circuit_open": self._is_circuit_open(),
            "circuit_failure_count": self._circuit_failure_count,
        }

        # Smoke test with actual embedding
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
            "tool_aware_batching": True,
            "max_embedding_retries": 3,
            "max_concurrent_requests": 50,
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
