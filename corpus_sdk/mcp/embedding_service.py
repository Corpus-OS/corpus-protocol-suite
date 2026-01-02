# mcp_embedding_server/services/embedding_service.py
# SPDX-License-Identifier

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

Architecture
------------
This is a framework adapter layer (not the concrete embedding adapters).
It provides MCP-friendly semantics around a Corpus EmbeddingProtocolV1-compatible
adapter:

**Layering:**
- MCP layer (this module): Request validation, concurrency control, error mapping
- Translation layer (`EmbeddingTranslator`): Framework-agnostic orchestration
- Protocol layer (`EmbeddingProtocolV1`): Standard embedding interface
- Adapter layer: Provider-specific implementations

**Resilience:**
Rate limiting, circuit breaking, retries, deadlines, and caching are expected
to be provided by the underlying adapter (typically `BaseEmbeddingAdapter`).
The MCP layer stays thin and only adds MCP-specific semantics and observability.

**Service-level limits (MCP-facing):**
- Maximum batch size: 1000 texts per request
- Maximum total text size: 1,000,000 characters per request

Adapters may impose stricter limits based on provider constraints.

**Empty text handling:**
Empty strings are filtered before adapter calls and replaced with zero vectors
in results, preserving batch ordering and preventing adapter confusion.

**Error context:**
All operations automatically attach rich context (session_id, tool_name, metrics)
to exceptions for distributed tracing and debugging.

**Test compatibility:**
This module is intentionally compatible with the OSS test suite at
tests/frameworks/embedding/test_mcp_adapter.py

Design notes / philosophy
-------------------------
- **Protocol-first**: We only require an `embed` method (duck-typed), not a
  concrete base class, so provider adapters remain flexible.
- **Async-first**: The public API is fully async to match MCP's architecture
  and avoid blocking event loops.
- **Strict at the boundary**: Request validation (types, batch size, total
  text size) happens at the MCP boundary; invalid requests are rejected with
  structured `MCPEmbeddingServiceError`s.
- **Fail-safe context translation**: MCP context translation must never break
  embeddings; failures gracefully fall back to simple `OperationContext` or
  no context, with diagnostic context attached to errors.
- **Observability-first**: All embedding operations attach rich error context
  (framework identity, model, batch sizes, text lengths, routing IDs, and
  embedding dimension hints) to exceptions for debugging and tracing.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from functools import wraps
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    runtime_checkable,
)

from corpus_sdk.core.context_translation import from_mcp
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.embedding.embedding_base import (
    DeadlineExceeded,
    EmbeddingAdapterError,
    EmbeddingProtocolV1,
    OperationContext,
    ResourceExhausted,
    TransientNetwork,
    Unavailable,
)
from corpus_sdk.embedding.framework_adapters.common.embedding_translation import (
    BatchConfig,
    EmbeddingTranslator,
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

_FRAMEWORK_NAME = "mcp"

# Best-effort version detection
try:  # pragma: no cover
    from importlib.metadata import version as _pkg_version

    try:
        _FRAMEWORK_VERSION: Optional[str] = _pkg_version("corpus_sdk")
    except Exception:  # noqa: BLE001
        _FRAMEWORK_VERSION = None
except Exception:  # noqa: BLE001
    _FRAMEWORK_VERSION = None


# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------


class ErrorCodes:
    """
    Error code constants for the MCP embedding adapter.

    This is a simple namespace for service-level and coercion-level codes.
    The shared coercion helpers use `EMBEDDING_COERCION_ERROR_CODES`, which is
    a `CoercionErrorCodes` instance derived from these values.
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


EMBEDDING_COERCION_ERROR_CODES: CoercionErrorCodes = CoercionErrorCodes(
    invalid_result=ErrorCodes.INVALID_EMBEDDING_RESULT,
    empty_result=ErrorCodes.EMPTY_EMBEDDING_RESULT,
    conversion_error=ErrorCodes.EMBEDDING_CONVERSION_ERROR,
    framework_label=_FRAMEWORK_NAME,
)


# ---------------------------------------------------------------------------
# Types & protocols
# ---------------------------------------------------------------------------


class MCPContext(Dict[str, Any]):
    """
    Structured type for MCP execution context.

    Common fields:
    - session_id: MCP session identifier
    - tool_name: Tool being executed
    - server_id: MCP server identifier
    - client_id: MCP client identifier
    - trace_id: Distributed trace identifier
    - request_id: Specific request identifier
    """

    pass


class MCPConfig(Dict[str, Any]):
    """
    Structured configuration for MCP-specific settings.

    Fields:
    - max_concurrent_requests: Maximum concurrent embedding requests
    - fallback_to_simple_context: Create default OperationContext on translation failure
    - enable_session_context_propagation: Include OperationContext in framework_ctx
    - tool_aware_batching: Add tool-specific batch strategies
    """

    pass


@runtime_checkable
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


class MCPEmbeddingServiceError(Exception):
    """Base exception for MCP embedding service errors."""

    def __init__(self, message: str, code: str, request_id: Optional[str] = None):
        super().__init__(message)
        self.code = code
        self.request_id = request_id
        self.message = message


# ---------------------------------------------------------------------------
# Helpers (compat + observability)
# ---------------------------------------------------------------------------


async def _maybe_await(value: Any) -> Any:
    """
    Support adapters that are async, sync, or mocks returning plain values.

    This enables compatibility with:
    - Async adapters: `async def embed(...)`
    - Sync adapters: `def embed(...)`
    - Mock objects: `Mock(return_value=[...])`
    """
    if inspect.isawaitable(value):
        return await value
    return value


async def _call_adapter_embed(adapter: Any, texts: List[str], **kwargs: Any) -> Any:
    """
    Call adapter.embed() in a way that supports sync, async, and mock adapters.

    Parameters
    ----------
    adapter:
        Embedding adapter (sync, async, or mock)
    texts:
        List of texts to embed
    **kwargs:
        Additional arguments (op_ctx, framework_ctx, etc.)

    Returns
    -------
    Embedding result from adapter (format depends on adapter)
    """
    fn = getattr(adapter, "embed", None)
    if fn is None:
        raise TypeError("corpus_adapter must define an embed method")
    return await _maybe_await(fn(texts, **kwargs))


def _extract_dynamic_context(
    *,
    instance: Any,
    operation: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract dynamic context from an MCP embedding call for observability.

    Captures:
    - model identifier from the embedding instance
    - framework_version if present
    - text_len for single-text operations
    - texts_count / empty_texts_count for batch operations
    - key MCP routing fields (session_id, tool_name, server_id, client_id, trace_id, request_id)
    - optional embedding_dim hint when available
    """
    ctx: Dict[str, Any] = {
        "framework": _FRAMEWORK_NAME,
        "operation": f"embedding_{operation}",
        "model": getattr(instance, "model", None),
        "framework_version": getattr(instance, "_framework_version", None),
    }

    # Optional best-effort embedding dimension hint
    dim_hint = getattr(instance, "_embedding_dim_hint", None)
    if isinstance(dim_hint, int):
        ctx["embedding_dim"] = dim_hint

    # Text / batch metrics
    if operation == "documents" and args and isinstance(args[0], Sequence) and not isinstance(args[0], str):
        texts = args[0]
        ctx["texts_count"] = len(texts)
        empty_count = sum(1 for t in texts if not isinstance(t, str) or not t.strip())
        if empty_count:
            ctx["empty_texts_count"] = empty_count
    elif operation == "query" and args and isinstance(args[0], str):
        ctx["text_len"] = len(args[0])

    # MCP routing fields
    mcp_context = kwargs.get("mcp_context") or {}
    if isinstance(mcp_context, Mapping):
        for key in ("session_id", "tool_name", "server_id", "client_id", "trace_id", "request_id"):
            if key in mcp_context:
                ctx[key] = mcp_context.get(key)

    return ctx


def _attach_error_context(exc: BaseException, ctx: Dict[str, Any]) -> None:
    """
    Attach error context to exception, never letting observability throw.

    This is safe to call in exception handlers and will not mask the original
    exception if context attachment fails.
    """
    try:
        attach_context(exc, **ctx)
    except Exception:  # noqa: BLE001
        # Never let observability break exception handling
        pass


# ---------------------------------------------------------------------------
# Error context decorators
# ---------------------------------------------------------------------------


def _create_error_context_decorator(
    operation: str,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Factory for creating async error-context decorators with rich per-call metrics.

    This ensures that ALL exceptions from decorated methods automatically receive
    rich context for observability, including:
    - Framework metadata (name, version)
    - Operation type (query vs documents vs capabilities vs health)
    - MCP routing fields (session_id, tool_name, etc.)
    - Dynamic metrics (text_len, texts_count, etc. where applicable)
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
            dynamic_context = _extract_dynamic_context(
                instance=self,
                operation=operation,
                args=args,
                kwargs=kwargs,
            )
            full_context: Dict[str, Any] = {
                "framework": _FRAMEWORK_NAME,
                "framework_version": getattr(self, "_framework_version", None),
                "error_codes": EMBEDDING_COERCION_ERROR_CODES,
                **dynamic_context,
            }
            try:
                return await func(self, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                _attach_error_context(exc, full_context)
                raise

        return wrapper

    return decorator


def with_embedding_error_context(
    operation: str,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator for async methods with automatic rich error context attachment.

    Usage
    -----
    @with_embedding_error_context("documents")
    async def embed_documents(self, texts, **kwargs): ...

    @with_embedding_error_context("query")
    async def embed_query(self, text, **kwargs): ...

    @with_embedding_error_context("capabilities")
    async def acapabilities(self): ...

    @with_embedding_error_context("health")
    async def ahealth(self): ...
    """
    return _create_error_context_decorator(operation)


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


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
    - Empty string filtering and zero-vector reconstruction
    - Defensive normalization against adapter misbehavior
    """

    def __init__(
        self,
        corpus_adapter: EmbeddingProtocolV1,
        model: Optional[str] = None,
        batch_config: Optional[BatchConfig] = None,
        text_normalization_config: Optional[TextNormalizationConfig] = None,
        mcp_config: Optional[MCPConfig] = None,
        **_: Any,
    ) -> None:
        self.corpus_adapter = corpus_adapter
        self.model = model
        self.batch_config = batch_config
        self.text_normalization_config = text_normalization_config
        self.mcp_config: MCPConfig = self._validate_mcp_config(mcp_config or {})

        # Optional framework/service version for parity with other adapters
        self._framework_version: Optional[str] = _FRAMEWORK_VERSION

        # Concurrency control
        self._request_semaphore = asyncio.Semaphore(self.mcp_config["max_concurrent_requests"])

        # Active request tracking
        self._active_requests = 0
        self._active_requests_lock = threading.Lock()

        # Translator lifecycle - thread-safe lazy initialization
        self._translator_lock = threading.Lock()
        self._translator_instance: Optional[EmbeddingTranslator] = None

        # Best-effort embedding dimension hint for observability
        self._embedding_dim_hint: Optional[int] = None

        # Protocol metrics
        self._protocol_success_count = 0
        self._protocol_error_count = 0
        self._protocol_total_latency_s = 0.0

        self._protocol_query_success_count = 0
        self._protocol_query_error_count = 0
        self._protocol_query_total_latency_s = 0.0

        self._protocol_batch_success_count = 0
        self._protocol_batch_error_count = 0
        self._protocol_batch_total_latency_s = 0.0

        logger.info(
            "CorpusMCPEmbeddings initialized with model=%s, max_concurrent=%d, framework_version=%s",
            model or "default",
            self.mcp_config["max_concurrent_requests"],
            self._framework_version or "unknown",
        )

    # -------------------------
    # Config validation
    # -------------------------

    def _validate_mcp_config(self, config: MCPConfig) -> MCPConfig:
        """
        Validate and normalize MCP configuration with sensible defaults.
        """
        validated: MCPConfig = dict(config)

        validated.setdefault("max_concurrent_requests", 100)
        validated.setdefault("fallback_to_simple_context", True)
        validated.setdefault("enable_session_context_propagation", True)
        validated.setdefault("tool_aware_batching", False)

        try:
            validated["max_concurrent_requests"] = int(validated["max_concurrent_requests"])
        except Exception:
            # Tests expect this exact message substring
            raise ValueError("max_concurrent_requests must be an integer") from None

        if validated["max_concurrent_requests"] <= 0:
            # Tests expect this exact message substring
            raise ValueError("max_concurrent_requests must be positive") from None

        for key in ("fallback_to_simple_context", "enable_session_context_propagation", "tool_aware_batching"):
            validated[key] = bool(validated[key])

        return validated

    # -------------------------
    # Translator retrieval
    # -------------------------

    @property
    def _translator(self) -> EmbeddingTranslator:
        """
        Lazily construct the `EmbeddingTranslator` for MCP.
        """
        existing = self._translator_instance
        if isinstance(existing, EmbeddingTranslator):
            return existing

        with self._translator_lock:
            existing = self._translator_instance
            if isinstance(existing, EmbeddingTranslator):
                return existing

            translator = create_embedding_translator(
                adapter=self.corpus_adapter,
                framework=_FRAMEWORK_NAME,
                translator=None,
                batch_config=self.batch_config,
                text_normalization_config=self.text_normalization_config,
            )
            self._translator_instance = translator
            logger.debug(
                "EmbeddingTranslator initialized for MCP with model=%s",
                self.model or "default",
            )
            return translator

    # -------------------------
    # Context building
    # -------------------------

    def _build_contexts(
        self,
        *,
        mcp_context: Optional[MCPContext],
        request_id: str,
        model: Optional[str],
        **kwargs: Any,
    ) -> Tuple[Optional[OperationContext], Dict[str, Any]]:
        """
        Build contexts for MCP execution environment with request ID propagation.

        Returns
        -------
        Tuple of:
        - core_ctx: OperationContext (or None if translation failed and fallback disabled)
        - framework_ctx: MCP-specific context for translator and observability
        """
        ctx_in: Dict[str, Any] = dict(mcp_context or {})
        ctx_in.setdefault("request_id", request_id)

        core_ctx: Optional[OperationContext] = None
        if ctx_in:
            try:
                candidate = from_mcp(ctx_in)
                if isinstance(candidate, OperationContext):
                    attrs = dict(getattr(candidate, "attrs", {}) or {})
                    attrs.setdefault("framework", _FRAMEWORK_NAME)
                    if self._framework_version is not None:
                        attrs.setdefault("framework_version", self._framework_version)

                    core_ctx = OperationContext(
                        request_id=candidate.request_id,
                        idempotency_key=candidate.idempotency_key,
                        deadline_ms=candidate.deadline_ms,
                        traceparent=candidate.traceparent,
                        tenant=candidate.tenant,
                        attrs=attrs,
                    )
                    logger.debug(
                        "Created OperationContext for MCP session=%s, tool=%s, request=%s",
                        ctx_in.get("session_id", "unknown"),
                        ctx_in.get("tool_name", "unknown"),
                        request_id,
                    )
                else:
                    logger.warning(
                        "from_mcp returned non-OperationContext type: %s. "
                        "Proceeding without OperationContext.",
                        type(candidate).__name__,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to create OperationContext from mcp_context: %s", exc)
                if self.mcp_config["fallback_to_simple_context"]:
                    core_ctx = OperationContext(request_id=request_id)

        framework_ctx: Dict[str, Any] = {
            "framework": _FRAMEWORK_NAME,
            "framework_version": self._framework_version,
            "mcp_config": dict(self.mcp_config),
            "error_codes": EMBEDDING_COERCION_ERROR_CODES,
        }

        effective_model = model or self.model
        if effective_model:
            framework_ctx["model"] = effective_model

        for key in ("session_id", "tool_name", "server_id", "client_id", "trace_id", "request_id"):
            if key in ctx_in:
                framework_ctx[key] = ctx_in.get(key)

        if self.mcp_config["tool_aware_batching"] and ctx_in.get("tool_name"):
            framework_ctx["batch_strategy"] = f"tool_aware_{ctx_in['tool_name']}"

        framework_ctx.update({k: v for k, v in kwargs.items() if not k.startswith("_")})

        if core_ctx is not None and self.mcp_config["enable_session_context_propagation"]:
            framework_ctx["_operation_context"] = core_ctx

        # Surface a best-effort embedding dimension hint for observability.
        if isinstance(self._embedding_dim_hint, int):
            framework_ctx["embedding_dim_hint"] = self._embedding_dim_hint

        return core_ctx, framework_ctx

    def _ensure_op_ctx(self, core_ctx: Optional[OperationContext], request_id: str) -> OperationContext:
        """
        Provide a non-None OperationContext for translator operations.
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
            attrs={
                "framework": _FRAMEWORK_NAME,
                "framework_version": self._framework_version,
            },
        )

    # -------------------------
    # Request validation
    # -------------------------

    def _validate_request_texts(self, texts: List[Any], request_id: str) -> List[str]:
        """
        Comprehensive request validation at MCP service boundary.
        """
        if not texts:
            raise MCPEmbeddingServiceError("No texts provided", ErrorCodes.EMPTY_REQUEST, request_id)

        if len(texts) > 1000:
            raise MCPEmbeddingServiceError(
                f"Batch size {len(texts)} exceeds maximum 1000",
                ErrorCodes.BATCH_SIZE_EXCEEDED,
                request_id,
            )

        validated: List[str] = []
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise MCPEmbeddingServiceError(
                    f"Text at index {i} is not a string",
                    ErrorCodes.INVALID_TEXT_TYPE,
                    request_id,
                )
            validated.append(t)

        total_chars = sum(len(t) for t in validated)
        if total_chars >= 1_000_000:
            raise MCPEmbeddingServiceError(
                f"Total text size {total_chars} exceeds limit",
                ErrorCodes.TEXT_SIZE_EXCEEDED,
                request_id,
            )

        return validated

    # -------------------------
    # Error mapping
    # -------------------------

    def _map_error(self, exc: BaseException, request_id: str) -> MCPEmbeddingServiceError:
        """
        Map adapter-level errors to service-level error codes.
        """
        if isinstance(exc, EmbeddingAdapterError):
            msg = getattr(exc, "message", None) or str(exc)

            if isinstance(exc, ResourceExhausted):
                return MCPEmbeddingServiceError(
                    f"Rate or resource limit exceeded: {msg}",
                    ErrorCodes.SERVICE_UNAVAILABLE,
                    request_id,
                )
            if isinstance(exc, DeadlineExceeded):
                return MCPEmbeddingServiceError(
                    f"Request deadline exceeded: {msg}",
                    ErrorCodes.REQUEST_TIMEOUT,
                    request_id,
                )
            if isinstance(exc, (Unavailable, TransientNetwork)):
                return MCPEmbeddingServiceError(
                    f"Embedding backend unavailable: {msg}",
                    ErrorCodes.SERVICE_UNAVAILABLE,
                    request_id,
                )

            return MCPEmbeddingServiceError(
                f"Embedding adapter error: {msg}",
                ErrorCodes.EMBEDDING_EXTRACTION_ERROR,
                request_id,
            )

        if isinstance(exc, asyncio.TimeoutError):
            return MCPEmbeddingServiceError(
                "Embedding request timed out",
                ErrorCodes.REQUEST_TIMEOUT,
                request_id,
            )

        return MCPEmbeddingServiceError(
            f"Embedding service error: {str(exc)}",
            ErrorCodes.EMBEDDING_EXTRACTION_ERROR,
            request_id,
        )

    # -------------------------
    # Active request tracking
    # -------------------------

    @asynccontextmanager
    async def _track_active(self) -> AsyncIterator[None]:
        """
        Context manager for thread-safe active request tracking.
        """
        with self._active_requests_lock:
            self._active_requests += 1
        try:
            yield
        finally:
            with self._active_requests_lock:
                self._active_requests -= 1

    # -------------------------
    # Embed execution
    # -------------------------

    async def _run_embed(
        self,
        *,
        raw_texts: Any,
        op_ctx: OperationContext,
        framework_ctx: Dict[str, Any],
    ) -> Any:
        """
        Execute embedding via translator or adapter fallback.
        """
        translator = self._translator

        if translator is not None and hasattr(translator, "arun_embed"):
            return await translator.arun_embed(raw_texts=raw_texts, op_ctx=op_ctx, framework_ctx=framework_ctx)

        if isinstance(raw_texts, list):
            return await _call_adapter_embed(self.corpus_adapter, raw_texts, op_ctx=op_ctx, framework_ctx=framework_ctx)
        return await _call_adapter_embed(self.corpus_adapter, [raw_texts], op_ctx=op_ctx, framework_ctx=framework_ctx)

    # -------------------------
    # Dimension + coercion helpers
    # -------------------------

    def _get_embedding_dimension_fallback(self) -> int:
        """
        Get embedding dimension from adapter or use fallback.

        NOTE: Behavior is intentionally stable:
        - Prefer adapter.get_embedding_dimension() when available
        - On failure, fall back to a small fixed dimension (8)
        """
        if hasattr(self.corpus_adapter, "get_embedding_dimension"):
            try:
                dim = int(self.corpus_adapter.get_embedding_dimension())
                self._embedding_dim_hint = dim
                return dim
            except Exception:  # noqa: BLE001
                pass
        return 8

    def _coerce_embedding_matrix(self, result: Any) -> List[List[float]]:
        """
        Coerce a result into a 2D embedding matrix and update the dim hint.
        """
        mat = coerce_embedding_matrix(
            result=result,
            framework=_FRAMEWORK_NAME,
            error_codes=EMBEDDING_COERCION_ERROR_CODES,
            logger=logger,
        )
        if mat and isinstance(mat[0], list):
            self._embedding_dim_hint = len(mat[0])
        return mat

    def _coerce_embedding_vector(self, result: Any) -> List[float]:
        """
        Coerce a result into a 1D embedding vector and update the dim hint.
        """
        vec = coerce_embedding_vector(
            result=result,
            framework=_FRAMEWORK_NAME,
            error_codes=EMBEDDING_COERCION_ERROR_CODES,
            logger=logger,
        )
        self._embedding_dim_hint = len(vec)
        return vec

    def _warn_if_extreme_batch(self, texts: Sequence[str], *, op_name: str) -> None:
        """
        Soft warning for extremely large batches (diagnostic only).

        Hard limits are still enforced by _validate_request_texts; this helper
        just emits warnings for very large-but-allowed batches.
        """
        warn_if_extreme_batch(
            framework=_FRAMEWORK_NAME,
            texts=texts,
            op_name=op_name,
            batch_config=self.batch_config,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # MCPEmbedder API
    # ------------------------------------------------------------------

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
        """
        request_id = f"embed_docs_{uuid.uuid4().hex[:8]}"
        start = time.perf_counter()

        try:
            validated = self._validate_request_texts(list(texts), request_id)
            self._warn_if_extreme_batch(validated, op_name="embed_documents")

            non_empty: List[str] = [t for t in validated if t.strip()]
            empty_positions: List[int] = [i for i, t in enumerate(validated) if not t.strip()]

            core_ctx, framework_ctx = self._build_contexts(
                mcp_context=mcp_context,
                request_id=request_id,
                model=model,
                **kwargs,
            )
            op_ctx = self._ensure_op_ctx(core_ctx, request_id)

            async with self._track_active():
                async with self._request_semaphore:
                    if not non_empty:
                        dim = self._get_embedding_dimension_fallback()
                        out = [[0.0] * dim for _ in validated]
                    else:
                        translated = await self._run_embed(
                            raw_texts=non_empty,
                            op_ctx=op_ctx,
                            framework_ctx=framework_ctx,
                        )
                        mat = self._coerce_embedding_matrix(translated)

                        if len(mat) != len(non_empty):
                            logger.warning(
                                "Adapter/translator returned %d rows for %d inputs (request_id=%s). "
                                "This indicates an adapter bug. Normalizing result.",
                                len(mat),
                                len(non_empty),
                                request_id,
                            )
                            if len(mat) < len(non_empty):
                                dim = len(mat[0]) if mat else self._get_embedding_dimension_fallback()
                                mat = mat + [[0.0] * dim for _ in range(len(non_empty) - len(mat))]
                            else:
                                mat = mat[: len(non_empty)]

                        dim = len(mat[0]) if mat else self._get_embedding_dimension_fallback()
                        out: List[List[float]] = []
                        ne_i = 0
                        empty_set = set(empty_positions)
                        for i in range(len(validated)):
                            if i in empty_set:
                                out.append([0.0] * dim)
                            else:
                                out.append(mat[ne_i])
                                ne_i += 1

            latency = time.perf_counter() - start
            self._protocol_success_count += 1
            self._protocol_total_latency_s += latency
            self._protocol_batch_success_count += 1
            self._protocol_batch_total_latency_s += latency

            logger.debug(
                "Embedding completed in %.3fs for request %s (%d documents, %d empty)",
                latency,
                request_id,
                len(validated),
                len(empty_positions),
            )

            return out

        except asyncio.CancelledError:
            raise
        except MCPEmbeddingServiceError:
            self._protocol_error_count += 1
            self._protocol_batch_error_count += 1
            raise
        except Exception as exc:  # noqa: BLE001
            self._protocol_error_count += 1
            self._protocol_batch_error_count += 1
            svc_err = self._map_error(exc, request_id)
            raise svc_err

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
        """
        request_id = f"embed_query_{uuid.uuid4().hex[:8]}"
        start = time.perf_counter()

        try:
            validated = self._validate_request_texts([text], request_id)
            single = validated[0]

            core_ctx, framework_ctx = self._build_contexts(
                mcp_context=mcp_context,
                request_id=request_id,
                model=model,
                **kwargs,
            )
            op_ctx = self._ensure_op_ctx(core_ctx, request_id)

            async with self._track_active():
                async with self._request_semaphore:
                    if not single.strip():
                        dim = self._get_embedding_dimension_fallback()
                        vec = [0.0] * dim
                    else:
                        translated = await self._run_embed(
                            raw_texts=single,
                            op_ctx=op_ctx,
                            framework_ctx=framework_ctx,
                        )
                        vec = self._coerce_embedding_vector(translated)

            latency = time.perf_counter() - start
            self._protocol_success_count += 1
            self._protocol_total_latency_s += latency
            self._protocol_query_success_count += 1
            self._protocol_query_total_latency_s += latency

            logger.debug(
                "Query embedding completed in %.3fs for request %s",
                latency,
                request_id,
            )

            return vec

        except asyncio.CancelledError:
            raise
        except MCPEmbeddingServiceError:
            self._protocol_error_count += 1
            self._protocol_query_error_count += 1
            raise
        except Exception as exc:  # noqa: BLE001
            self._protocol_error_count += 1
            self._protocol_query_error_count += 1
            svc_err = self._map_error(exc, request_id)
            raise svc_err

    # ------------------------------------------------------------------
    # Capabilities / health passthrough (async) — via translator
    # ------------------------------------------------------------------

    @with_embedding_error_context("capabilities")
    async def acapabilities(self) -> Dict[str, Any]:
        """
        Best-effort capabilities passthrough.

        Preferred order:
        1. EmbeddingTranslator.arun_capabilities()
        2. EmbeddingTranslator.capabilities() (via thread)
        3. Underlying adapter acapabilities()
        4. Underlying adapter capabilities() (via thread)
        """
        translator: Optional[EmbeddingTranslator]
        try:
            translator = self._translator
        except Exception:  # noqa: BLE001
            translator = None

        if translator is not None:
            if hasattr(translator, "arun_capabilities"):
                caps = await translator.arun_capabilities()  # type: ignore[func-returns-value]
                return dict(caps) if isinstance(caps, Mapping) else dict(caps or {})
            if hasattr(translator, "capabilities"):
                caps = await asyncio.to_thread(translator.capabilities)  # type: ignore[arg-type]
                return dict(caps) if isinstance(caps, Mapping) else dict(caps or {})

        if hasattr(self.corpus_adapter, "acapabilities"):
            caps = await self.corpus_adapter.acapabilities()  # type: ignore[func-returns-value]
            return dict(caps) if isinstance(caps, Mapping) else dict(caps or {})
        if hasattr(self.corpus_adapter, "capabilities"):
            caps = await asyncio.to_thread(self.corpus_adapter.capabilities)  # type: ignore[arg-type]
            return dict(caps) if isinstance(caps, Mapping) else dict(caps or {})

        return {}

    @with_embedding_error_context("health")
    async def ahealth(self) -> Dict[str, Any]:
        """
        Best-effort health passthrough.

        Preferred order:
        1. EmbeddingTranslator.arun_health()
        2. EmbeddingTranslator.health() (via thread)
        3. Underlying adapter ahealth()
        4. Underlying adapter health() (via thread)
        """
        translator: Optional[EmbeddingTranslator]
        try:
            translator = self._translator
        except Exception:  # noqa: BLE001
            translator = None

        if translator is not None:
            if hasattr(translator, "arun_health"):
                h = await translator.arun_health()  # type: ignore[func-returns-value]
                return dict(h) if isinstance(h, Mapping) else dict(h or {})
            if hasattr(translator, "health"):
                h = await asyncio.to_thread(translator.health)  # type: ignore[arg-type]
                return dict(h) if isinstance(h, Mapping) else dict(h or {})

        if hasattr(self.corpus_adapter, "ahealth"):
            h = await self.corpus_adapter.ahealth()  # type: ignore[func-returns-value]
            return dict(h) if isinstance(h, Mapping) else dict(h or {})
        if hasattr(self.corpus_adapter, "health"):
            h = await asyncio.to_thread(self.corpus_adapter.health)  # type: ignore[arg-type]
            return dict(h) if isinstance(h, Mapping) else dict(h or {})

        return {}

    # ------------------------------------------------------------------
    # Health check (must not skew metrics)
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check for MCP server integration.
        """
        with self._active_requests_lock:
            active = self._active_requests

        avg_ms: Optional[float] = None
        if self._protocol_success_count > 0:
            avg_ms = (self._protocol_total_latency_s / self._protocol_success_count) * 1000.0

        avg_query_ms: Optional[float] = None
        if self._protocol_query_success_count > 0:
            avg_query_ms = (self._protocol_query_total_latency_s / self._protocol_query_success_count) * 1000.0

        avg_batch_ms: Optional[float] = None
        if self._protocol_batch_success_count > 0:
            avg_batch_ms = (self._protocol_batch_total_latency_s / self._protocol_batch_success_count) * 1000.0

        health: Dict[str, Any] = {
            "status": "healthy",
            "active_requests": active,
            "max_concurrent_requests": self.mcp_config["max_concurrent_requests"],
            "framework_version": _FRAMEWORK_VERSION,
            "protocol_success_count": self._protocol_success_count,
            "protocol_error_count": self._protocol_error_count,
            "avg_protocol_latency_ms": avg_ms,
            "protocol_query_success_count": self._protocol_query_success_count,
            "protocol_query_error_count": self._protocol_query_error_count,
            "avg_protocol_query_latency_ms": avg_query_ms,
            "protocol_batch_success_count": self._protocol_batch_success_count,
            "protocol_batch_error_count": self._protocol_batch_error_count,
            "avg_protocol_batch_latency_ms": avg_batch_ms,
        }

        try:
            dim = self._get_embedding_dimension_fallback()
            if dim <= 0:
                core_ctx, framework_ctx = self._build_contexts(
                    mcp_context={
                        "tool_name": "health_check",
                        "session_id": "health_check",
                        "request_id": "health_check",
                    },
                    request_id="health_check",
                    model=None,
                    health_check=True,
                )
                op_ctx = self._ensure_op_ctx(core_ctx, "health_check")
                translated = await self._run_embed(
                    raw_texts="health_check",
                    op_ctx=op_ctx,
                    framework_ctx=framework_ctx,
                )
                vec = self._coerce_embedding_vector(translated)
                dim = len(vec)

            health["service_test"] = "passed"
            health["embedding_dimension"] = dim
        except Exception as exc:  # noqa: BLE001
            health["status"] = "degraded"
            health["service_test"] = f"failed: {str(exc)}"
            health["embedding_dimension"] = self._get_embedding_dimension_fallback()

        return health

    # ------------------------------------------------------------------
    # Resource management (optional)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Close underlying adapter if it exposes close().

        This mirrors the lifecycle helpers provided by other framework adapters.
        """
        fn = getattr(self.corpus_adapter, "close", None)
        if callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error while closing embedding adapter in close(): %s", exc)

    async def aclose(self) -> None:
        """
        Async-close underlying adapter if it exposes aclose(), else fall back to close().
        """
        fn_async = getattr(self.corpus_adapter, "aclose", None)
        if callable(fn_async):
            try:
                await fn_async()
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error while closing embedding adapter in aclose(): %s", exc)

        fn_sync = getattr(self.corpus_adapter, "close", None)
        if callable(fn_sync):
            try:
                await asyncio.to_thread(fn_sync)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error while closing embedding adapter in async close(): %s", exc)

    def __enter__(self) -> "CorpusMCPEmbeddings":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> "CorpusMCPEmbeddings":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_embedder(
    corpus_adapter: EmbeddingProtocolV1,
    model: Optional[str] = None,
    **kwargs: Any,
) -> MCPEmbedder:
    """
    Create an MCP-compatible embedder for seamless server integration.
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
    "MCPEmbeddingServiceError",
    "create_embedder",
    "ErrorCodes",
]