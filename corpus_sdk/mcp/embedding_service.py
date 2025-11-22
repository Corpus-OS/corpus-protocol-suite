# mcp_embedding_server/services/embedding_service.py
# SPDX-License-Identifier: Apache-2.0

"""
MCP adapter for Corpus Embedding protocol.

This module exposes Corpus `EmbeddingProtocolV1` implementations as
embedding services within MCP servers and workflows, with:

- Seamless integration with MCP tool execution and resource handling
- Support for MCP session context and request tracing
- Context normalization for MCP-specific execution environment
- Framework-agnostic translation via an `EmbeddingFrameworkTranslator`
- Async-first design optimized for MCP's async architecture
- Rich error context attachment for observability in MCP workflows

The design follows MCP's protocol patterns while maintaining the
protocol-first Corpus embedding stack.

Limits:
- Maximum batch size: 1000 texts per request
- Maximum total text size: 1,000,000 characters per request
- Default rate limit: 1000 requests per minute
- Default concurrency: 100 simultaneous requests
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import wraps
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
    EmbedSpec,
    EmbedResult,
    BatchEmbedSpec,
    BatchEmbedResult,
)
from corpus_sdk.embedding.framework_adapters.common.embedding_translation import (
    BatchConfig,
    TextNormalizationConfig,
    TextNormalizer,
    EmbeddingFrameworkTranslator,
    DefaultEmbeddingFrameworkTranslator,
    get_embedding_translator_factory,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# Error code constants
class ErrorCodes:
    EMPTY_REQUEST = "EMPTY_REQUEST"
    BATCH_SIZE_EXCEEDED = "BATCH_SIZE_EXCEEDED"
    TEXT_SIZE_EXCEEDED = "TEXT_SIZE_EXCEEDED"
    INVALID_TEXT_TYPE = "INVALID_TEXT_TYPE"
    EMBEDDING_EXTRACTION_ERROR = "EMBEDDING_EXTRACTION_ERROR"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    REQUEST_TIMEOUT = "REQUEST_TIMEOUT"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    CIRCUIT_BREAKER_OPEN = "CIRCUIT_BREAKER_OPEN"


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
            except Exception as exc:
                enhanced_context = context_kwargs.copy()
                attach_context(
                    exc,
                    framework="mcp",
                    embedding_operation=operation,
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


class CorpusMCPEmbeddings:
    """
    MCP embedding service backed by a Corpus `EmbeddingProtocolV1` adapter.

    Implements the `MCPEmbedder` protocol for seamless integration with
    MCP servers and tool execution workflows.

    Architecture
    ------------
    - This service uses a per-framework `EmbeddingFrameworkTranslator` to:
        * Build `EmbedSpec` / `BatchEmbedSpec` from raw MCP text inputs
        * Translate `EmbedResult` / `BatchEmbedResult` back into framework-level outputs
    - It calls the underlying Corpus embedding protocol (`EmbeddingProtocolV1`)
      **explicitly**, preserving protocol layering:

        spec = translator.build_embed_spec(...)
        result = corpus_adapter.embed(spec, ctx=core_ctx)
        output = translator.translate_embed_result(result, op_ctx=core_ctx, ...)

    - This pattern matches other framework integrations (e.g., CrewAI) and keeps
      the protocol boundary clear and testable.

    Resilience
    ----------
    - Rate limiting with a sliding window
    - Bounded concurrency via an asyncio.Semaphore
    - Exponential backoff retries for transient failures
    - Circuit breaker for repeated failures
    - Deadline-aware timeouts propagated from `OperationContext.deadline_ms`
    - Rich error context attachment for observability

    Metrics & Observability
    -----------------------
    - Tracks global protocol metrics and per-operation metrics:
        * embed vs batch_embed success/error counts
        * latency for embed and batch_embed separately
    - Exposed via `health_check` for monitoring integration.

    Attributes
    ----------
    corpus_adapter: Underlying Corpus embedding protocol adapter
    model: Optional default model identifier
    batch_config: Optional batching configuration (stored for external orchestrators)
    text_normalization_config: Optional text normalization settings for translator
    mcp_config: MCP-specific configuration with validation and circuit breaker settings
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
        self.batch_config = self._validate_batch_config(batch_config)
        self.text_normalization_config = self._validate_text_normalization_config(
            text_normalization_config
        )
        self.mcp_config = self._validate_mcp_config(mcp_config or {})

        # Service state
        self._translator: Optional[EmbeddingFrameworkTranslator] = None
        self._active_requests: int = 0
        self._active_requests_lock = asyncio.Lock()
        self._rate_limit_lock = asyncio.Lock()
        self._request_semaphore = asyncio.Semaphore(
            self.mcp_config["max_concurrent_requests"]
        )
        self._rate_limit_tracker: List[float] = []

        # Circuit breaker state
        self._circuit_failure_count: int = 0
        self._circuit_open_until: float = 0.0

        # Global protocol-level metrics
        self._protocol_success_count: int = 0
        self._protocol_error_count: int = 0
        self._protocol_total_latency: float = 0.0  # seconds

        # Per-operation protocol metrics (embed vs batch_embed)
        self._protocol_embed_success_count: int = 0
        self._protocol_embed_error_count: int = 0
        self._protocol_embed_total_latency: float = 0.0

        self._protocol_batch_success_count: int = 0
        self._protocol_batch_error_count: int = 0
        self._protocol_batch_total_latency: float = 0.0

        logger.info(
            "CorpusMCPEmbeddings initialized with model: %s, max_concurrent: %d",
            model or "default",
            self.mcp_config["max_concurrent_requests"],
        )

    # ------------------------------------------------------------------ #
    # Configuration validation
    # ------------------------------------------------------------------ #

    def _validate_mcp_config(self, config: MCPConfig) -> MCPConfig:
        """Validate and normalize MCP configuration with sensible defaults."""
        validated: MCPConfig = config.copy()

        # Set defaults for missing values
        if "max_embedding_retries" not in validated:
            validated["max_embedding_retries"] = 3
        if "fallback_to_simple_context" not in validated:
            validated["fallback_to_simple_context"] = True
        if "enable_session_context_propagation" not in validated:
            validated["enable_session_context_propagation"] = True
        if "tool_aware_batching" not in validated:
            validated["tool_aware_batching"] = False
        if "max_concurrent_requests" not in validated:
            validated["max_concurrent_requests"] = 100
        if "rate_limit_per_minute" not in validated:
            validated["rate_limit_per_minute"] = 1000
        if "circuit_breaker_failure_threshold" not in validated:
            validated["circuit_breaker_failure_threshold"] = 5
        if "circuit_breaker_reset_timeout" not in validated:
            validated["circuit_breaker_reset_timeout"] = 30

        # Validate numeric ranges
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

    def _validate_batch_config(
        self, batch_config: Optional[BatchConfig]
    ) -> Optional[BatchConfig]:
        """
        Validate batch configuration type.

        The service stores this config so that external orchestrators or adapters
        can inspect or reuse it; it does not perform automatic batch splitting.
        """
        if batch_config is None:
            return None
        if not isinstance(batch_config, BatchConfig):
            raise TypeError(
                f"batch_config must be a BatchConfig instance or None, "
                f"got {type(batch_config).__name__}"
            )
        return batch_config

    def _validate_text_normalization_config(
        self, config: Optional[TextNormalizationConfig]
    ) -> Optional[TextNormalizationConfig]:
        """
        Validate text normalization configuration type.

        When provided, it will be used to construct a TextNormalizer passed
        into the framework translator.
        """
        if config is None:
            return None
        if not isinstance(config, TextNormalizationConfig):
            raise TypeError(
                "text_normalization_config must be a TextNormalizationConfig "
                f"instance or None, got {type(config).__name__}"
            )
        # TextNormalizationConfig validates itself in __post_init__
        return config

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @property
    def translator(self) -> EmbeddingFrameworkTranslator:
        """
        Lazily construct and cache the framework-level translator.

        This translator is responsible purely for:
          - Building protocol specs from raw texts
          - Translating protocol results into framework-facing shapes

        Protocol calls (`embed`, `batch_embed`) are invoked explicitly
        by this service via `self.corpus_adapter`.
        """
        if self._translator is None:
            factory = get_embedding_translator_factory("mcp")
            if factory is not None:
                self._translator = factory(self.corpus_adapter)
            else:
                text_normalizer: Optional[TextNormalizer] = None
                if self.text_normalization_config is not None:
                    text_normalizer = TextNormalizer(self.text_normalization_config)
                self._translator = DefaultEmbeddingFrameworkTranslator(
                    text_normalizer=text_normalizer
                )
        return self._translator

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
        # Start with MCP context and inject request_id for full-stack tracing
        enhanced_mcp_context: Dict[str, Any] = dict(mcp_context) if mcp_context else {}
        if request_id and "request_id" not in enhanced_mcp_context:
            enhanced_mcp_context["request_id"] = request_id

        core_ctx: Optional[OperationContext] = None
        framework_ctx: Dict[str, Any] = {
            "framework": "mcp",
            "config": self.mcp_config,
        }

        # Convert MCP context to core OperationContext with request_id propagation
        if enhanced_mcp_context:
            try:
                core_ctx = from_mcp(enhanced_mcp_context)

                # Ensure request_id flows through the entire stack
                if request_id and hasattr(core_ctx, "attrs"):
                    core_ctx.attrs["request_id"] = request_id

                logger.debug(
                    "Created OperationContext for MCP session: %s, tool: %s, request: %s",
                    enhanced_mcp_context.get("session_id", "unknown"),
                    enhanced_mcp_context.get("tool_name", "unknown"),
                    request_id or "unknown",
                )
            except Exception as e:
                logger.warning(
                    "Failed to create OperationContext from mcp_context: %s",
                    e,
                )
                if self.mcp_config["fallback_to_simple_context"]:
                    core_ctx = None

        # Framework-level context for MCP-specific optimizations
        effective_model = model or self.model
        if effective_model:
            framework_ctx["model"] = effective_model

        # Add MCP-specific context for observability
        if enhanced_mcp_context:
            framework_ctx.update(
                {
                    "session_id": enhanced_mcp_context.get("session_id"),
                    "tool_name": enhanced_mcp_context.get("tool_name"),
                    "server_id": enhanced_mcp_context.get("server_id"),
                    "client_id": enhanced_mcp_context.get("client_id"),
                    "trace_id": enhanced_mcp_context.get("trace_id"),
                    "request_id": request_id,
                    **kwargs,
                }
            )

            # Enable tool-aware batching if configured
            if self.mcp_config["tool_aware_batching"] and enhanced_mcp_context.get(
                "tool_name"
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
        Helper to provide a non-None OperationContext for translator operations.

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

            # Check if under limit
            if len(self._rate_limit_tracker) >= self.mcp_config["rate_limit_per_minute"]:
                return False

            # Add current request timestamp
            self._rate_limit_tracker.append(now)
            return True

    # ------------------------------------------------------------------ #
    # Circuit breaker
    # ------------------------------------------------------------------ #

    def _is_circuit_open(self) -> bool:
        """Return True if the circuit breaker is currently open."""
        now = time.time()
        if self._circuit_open_until == 0:
            return False
        if now >= self._circuit_open_until:
            # Reset circuit after cooldown
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
                "Circuit breaker opened after %d consecutive failures; "
                "cooldown: %ds",
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

            except Exception as exc:
                last_exception = exc

                # Check if this is a retryable error
                if not self._is_retryable_error(exc):
                    break

                if attempt < max_retries:
                    wait_time = self._get_retry_delay(attempt)
                    logger.warning(
                        "Retryable error in %s (attempt %d/%d), retrying in %.1fs: %s",
                        operation_name,
                        attempt + 1,
                        max_retries + 1,
                        wait_time,
                        str(exc),
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        "All %d retries exhausted for %s",
                        max_retries + 1,
                        operation_name,
                    )

        # If we get here, all retries failed or error is non-retryable
        if last_exception is None:
            # Should never happen, but for type safety
            self._record_failure()
            raise MCPEmbeddingServiceError(
                "Unknown error in retry loop",
                ErrorCodes.EMBEDDING_EXTRACTION_ERROR,
                request_id,
            )

        self._record_failure()
        raise self._map_adapter_error_to_service_error(last_exception, request_id)

    def _is_retryable_error(self, exc: Exception) -> bool:
        """Determine if an error is retryable based on type and content."""
        # TODO: Import and check for EmbeddingAdapterError when available
        # if isinstance(exc, EmbeddingAdapterError):
        #     return exc.code in ['RATE_LIMIT_EXCEEDED', 'TIMEOUT', 'SERVICE_UNAVAILABLE']

        # Fall back to string pattern matching for now
        error_str = str(exc).lower()

        # Common retryable patterns
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
        # Jitter based on UUID to reduce thundering herd
        jitter = 0.5 + (uuid.uuid4().int % 1000) / 1000.0
        return delay * jitter

    def _map_adapter_error_to_service_error(
        self,
        exc: Exception,
        request_id: str,
    ) -> MCPEmbeddingServiceError:
        """Map adapter-level errors to service-level error codes."""
        # TODO: Import and use EmbeddingAdapterError when available
        # if isinstance(exc, EmbeddingAdapterError):
        #     code_map = {
        #         'RATE_LIMIT_EXCEEDED': ErrorCodes.RATE_LIMIT_EXCEEDED,
        #         'TIMEOUT': ErrorCodes.REQUEST_TIMEOUT,
        #         'SERVICE_UNAVAILABLE': ErrorCodes.SERVICE_UNAVAILABLE,
        #     }
        #     return MCPEmbeddingServiceError(
        #         str(exc),
        #         code_map.get(exc.code, ErrorCodes.EMBEDDING_EXTRACTION_ERROR),
        #         request_id
        #     )

        # Fall back to string pattern matching
        error_str = str(exc).lower()

        if any(pattern in error_str for pattern in ["rate limit", "rate_limit", "too many requests"]):
            return MCPEmbeddingServiceError(
                f"Rate limit exceeded: {str(exc)}",
                ErrorCodes.RATE_LIMIT_EXCEEDED,
                request_id,
            )
        if any(pattern in error_str for pattern in ["timeout", "deadline"]):
            return MCPEmbeddingServiceError(
                f"Request timeout: {str(exc)}",
                ErrorCodes.REQUEST_TIMEOUT,
                request_id,
            )
        if any(pattern in error_str for pattern in ["unavailable", "down", "maintenance"]):
            return MCPEmbeddingServiceError(
                f"Service unavailable: {str(exc)}",
                ErrorCodes.SERVICE_UNAVAILABLE,
                request_id,
            )

        # Generic service error for non-retryable cases
        return MCPEmbeddingServiceError(
            f"Embedding service error: {str(exc)}",
            ErrorCodes.EMBEDDING_EXTRACTION_ERROR,
            request_id,
        )

    # ------------------------------------------------------------------ #
    # Protocol result coercion
    # ------------------------------------------------------------------ #

    def _coerce_embedding_matrix(self, result: Any) -> List[List[float]]:
        """
        Coerce translator result into embedding matrix with validation.

        Handles both single embedding results and batch embedding results.
        """
        embeddings_obj: Any = None

        if isinstance(result, dict):
            if "embeddings" in result:
                embeddings_obj = result["embeddings"]
            elif "batch_embeddings" in result:
                # Flatten batches into a single list of rows
                batches = result["batch_embeddings"]
                if not isinstance(batches, (list, tuple)):
                    raise MCPEmbeddingServiceError(
                        "batch_embeddings must be a list of batches",
                        ErrorCodes.EMBEDDING_EXTRACTION_ERROR,
                    )
                flattened: List[List[float]] = []
                for b_idx, batch in enumerate(batches):
                    if not isinstance(batch, (list, tuple)):
                        raise MCPEmbeddingServiceError(
                            f"Expected batch {b_idx} to be sequence, got {type(batch).__name__}",
                            ErrorCodes.EMBEDDING_EXTRACTION_ERROR,
                        )
                    flattened.extend(batch)
                embeddings_obj = flattened
            else:
                embeddings_obj = result
        elif hasattr(result, "embeddings"):
            embeddings_obj = getattr(result, "embeddings")
        elif hasattr(result, "batch_embeddings"):
            # Handle batch result objects
            batches2 = getattr(result, "batch_embeddings")
            if not isinstance(batches2, (list, tuple)):
                raise MCPEmbeddingServiceError(
                    "batch_embeddings must be a list of batches",
                    ErrorCodes.EMBEDDING_EXTRACTION_ERROR,
                )
            flattened2: List[List[float]] = []
            for b_idx, batch in enumerate(batches2):
                if not isinstance(batch, (list, tuple)):
                    raise MCPEmbeddingServiceError(
                        f"Expected batch {b_idx} to be sequence, got {type(batch).__name__}",
                        ErrorCodes.EMBEDDING_EXTRACTION_ERROR,
                    )
                flattened2.extend(batch)
            embeddings_obj = flattened2
        else:
            embeddings_obj = result

        if not isinstance(embeddings_obj, (list, tuple)):
            raise MCPEmbeddingServiceError(
                f"Translator result does not contain valid embeddings sequence: {type(embeddings_obj).__name__}",
                ErrorCodes.EMBEDDING_EXTRACTION_ERROR,
            )

        matrix: List[List[float]] = []
        for i, row in enumerate(embeddings_obj):
            if not isinstance(row, (list, tuple)):
                raise MCPEmbeddingServiceError(
                    f"Expected embedding row to be sequence, got {type(row).__name__} at index {i}",
                    ErrorCodes.EMBEDDING_EXTRACTION_ERROR,
                )

            if len(row) == 0:
                logger.warning("Empty embedding row at index %d, skipping", i)
                continue

            try:
                embedding_vector = [float(x) for x in row]
                matrix.append(embedding_vector)
            except (TypeError, ValueError) as e:
                raise MCPEmbeddingServiceError(
                    f"Failed to convert embedding values to float at row {i}: {e}",
                    ErrorCodes.EMBEDDING_EXTRACTION_ERROR,
                ) from e

        if not matrix:
            raise MCPEmbeddingServiceError(
                "Translator returned no valid embedding rows",
                ErrorCodes.EMBEDDING_EXTRACTION_ERROR,
            )

        return matrix

    def _coerce_embedding_vector(self, result: Any) -> List[float]:
        """Coerce translator result for single-text embed with validation."""
        matrix = self._coerce_embedding_matrix(result)

        if len(matrix) > 1:
            logger.warning(
                "Expected single embedding for query, got %d rows; using first row",
                len(matrix),
            )

        return matrix[0]

    def _validate_embedding_request(self, texts: List[str], request_id: str) -> None:
        """Comprehensive request validation."""
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
        if total_chars > 1_000_000:  # ~1MB total text
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

        Uses batch embedding when configured for optimal performance in MCP tool execution.

        Parameters
        ----------
        texts: List of document texts to embed
        mcp_context: Optional MCP execution context for session/tool awareness
        model: Optional model override for this specific call
        **kwargs: Additional framework-specific parameters

        Returns
        -------
        List of embedding vectors, one per input text

        Raises
        ------
        MCPEmbeddingServiceError: For service-level errors
        Exception: Any underlying embedding errors with enriched context
        """
        request_id = f"embed_docs_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        # Circuit breaker check
        if self._is_circuit_open():
            raise MCPEmbeddingServiceError(
                "Embedding circuit breaker is open",
                ErrorCodes.CIRCUIT_BREAKER_OPEN,
                request_id,
            )

        # Rate limiting check
        if not await self._check_rate_limit():
            raise MCPEmbeddingServiceError(
                f"Rate limit exceeded: {self.mcp_config['rate_limit_per_minute']} requests per minute",
                ErrorCodes.RATE_LIMIT_EXCEEDED,
                request_id,
            )

        # Request validation
        self._validate_embedding_request(texts, request_id)

        # Batch size monitoring
        if len(texts) > 100:
            logger.info(
                "Large batch size %d for MCP tool: %s",
                len(texts),
                mcp_context.get("tool_name", "unknown") if mcp_context else "unknown",
            )

        core_ctx, framework_ctx = self._build_contexts(
            mcp_context=mcp_context,
            model=model,
            request_id=request_id,
            **kwargs,
        )

        # Ensure context propagation for MCP sessions
        if core_ctx is not None and self.mcp_config["enable_session_context_propagation"]:
            framework_ctx["_operation_context"] = core_ctx

        logger.debug(
            "Embedding %d documents for MCP tool: %s, session: %s, request: %s",
            len(texts),
            mcp_context.get("tool_name", "unknown") if mcp_context else "unknown",
            mcp_context.get("session_id", "unknown") if mcp_context else "unknown",
            request_id,
        )

        # Derive timeout (seconds) from OperationContext.deadline_ms if present
        timeout: Optional[float] = None
        if core_ctx is not None and getattr(core_ctx, "deadline_ms", None):
            timeout = core_ctx.deadline_ms / 1000.0

        translation_ctx = self._get_translation_op_ctx(core_ctx, request_id)

        async with self._track_active_request():
            try:
                async with self._request_semaphore:

                    async def embed_operation() -> Any:
                        """
                        Invoke the embedding protocol explicitly, using the
                        framework translator only for spec construction and result translation.
                        """
                        # Choose batch vs single-embed protocol path
                        if len(texts) > 1 and self.mcp_config["tool_aware_batching"]:
                            # Build BatchEmbedSpec from raw texts
                            batch_spec: BatchEmbedSpec = self.translator.build_batch_embed_spec(
                                raw_batch=[texts],  # Single batch for now
                                op_ctx=translation_ctx,
                                framework_ctx=framework_ctx,
                            )

                            # Protocol call with metrics & type validation
                            protocol_start = time.time()
                            try:
                                batch_result = await self.corpus_adapter.batch_embed(
                                    batch_spec,
                                    ctx=core_ctx,
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

                            if not isinstance(batch_result, BatchEmbedResult):
                                raise MCPEmbeddingServiceError(
                                    f"Protocol batch_embed returned unsupported type: "
                                    f"{type(batch_result).__name__}",
                                    ErrorCodes.EMBEDDING_EXTRACTION_ERROR,
                                    request_id,
                                )

                            # Translate protocol result back to framework-level shape
                            return self.translator.translate_batch_embed_result(
                                batch_result,
                                op_ctx=translation_ctx,
                                framework_ctx=framework_ctx,
                            )

                        # Non-batch path: build single EmbedSpec for list of texts
                        embed_spec: EmbedSpec = self.translator.build_embed_spec(
                            raw_texts=texts,
                            op_ctx=translation_ctx,
                            framework_ctx=framework_ctx,
                            stream=False,
                        )

                        protocol_start = time.time()
                        try:
                            result = await self.corpus_adapter.embed(
                                embed_spec,
                                ctx=core_ctx,
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

                        if not isinstance(result, EmbedResult):
                            raise MCPEmbeddingServiceError(
                                f"Protocol embed returned unsupported type: "
                                f"{type(result).__name__}",
                                ErrorCodes.EMBEDDING_EXTRACTION_ERROR,
                                request_id,
                            )

                        return self.translator.translate_embed_result(
                            result,
                            op_ctx=translation_ctx,
                            framework_ctx=framework_ctx,
                        )

                    # Execute with retry logic and propagate OperationContext timeout
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
                )

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

        Used by MCP for query understanding and retrieval in tool workflows.

        Parameters
        ----------
        text: Query text to embed
        mcp_context: Optional MCP execution context
        model: Optional model override
        **kwargs: Additional parameters

        Returns
        -------
        Single embedding vector for the query text
        """
        request_id = f"embed_query_{uuid.uuid4().hex[:8]}"

        # Circuit breaker check
        if self._is_circuit_open():
            raise MCPEmbeddingServiceError(
                "Embedding circuit breaker is open",
                ErrorCodes.CIRCUIT_BREAKER_OPEN,
                request_id,
            )

        # Rate limiting check
        if not await self._check_rate_limit():
            raise MCPEmbeddingServiceError(
                f"Rate limit exceeded: {self.mcp_config['rate_limit_per_minute']} requests per minute",
                ErrorCodes.RATE_LIMIT_EXCEEDED,
                request_id,
            )

        # Query is a single string; reuse validation logic
        self._validate_embedding_request([text], request_id)

        core_ctx, framework_ctx = self._build_contexts(
            mcp_context=mcp_context,
            model=model,
            request_id=request_id,
            **kwargs,
        )

        # Ensure context propagation for query understanding
        if core_ctx is not None and self.mcp_config["enable_session_context_propagation"]:
            framework_ctx["_operation_context"] = core_ctx

        logger.debug(
            "Embedding query for MCP tool: %s, request: %s",
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
                        Use the framework translator and protocol adapter for
                        single-text embedding.
                        """
                        embed_spec: EmbedSpec = self.translator.build_embed_spec(
                            raw_texts=text,
                            op_ctx=translation_ctx,
                            framework_ctx=framework_ctx,
                            stream=False,
                        )

                        protocol_start = time.time()
                        try:
                            result = await self.corpus_adapter.embed(
                                embed_spec,
                                ctx=core_ctx,
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

                        if not isinstance(result, EmbedResult):
                            raise MCPEmbeddingServiceError(
                                f"Protocol embed returned unsupported type: "
                                f"{type(result).__name__}",
                                ErrorCodes.EMBEDDING_EXTRACTION_ERROR,
                                request_id,
                            )

                        return self.translator.translate_embed_result(
                            result,
                            op_ctx=translation_ctx,
                            framework_ctx=framework_ctx,
                        )

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
                )

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for MCP server integration."""
        async with self._active_requests_lock:
            active_requests = self._active_requests

        # Reading rate_limit_tracker without lock is acceptable for a health snapshot.
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
        ]

        # Test with actual embedding
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
        except Exception as e:
            health_status["service_test"] = f"failed: {str(e)}"
            health_status["status"] = "degraded"

        return health_status


def create_embedder(
    corpus_adapter: EmbeddingProtocolV1,
    model: Optional[str] = None,
    **kwargs: Any,
) -> MCPEmbedder:
    """
    Create an MCP-compatible embedder for seamless server integration.

    Example:
    ```python
    import mcp
    from mcp_embedding_server.services.embedding_service import create_embedder

    # Create optimized embedder for MCP server
    server_embedder = create_embedder(
        corpus_adapter=server_adapter,
        model="text-embedding-3-large",
        mcp_config={
            "tool_aware_batching": True,
            "max_embedding_retries": 3,
            "max_concurrent_requests": 50
        }
    )

    # Create different embedder for high-volume tools
    batch_embedder = create_embedder(
        corpus_adapter=batch_adapter,
        model="text-embedding-3-small",
        mcp_config={
            "max_concurrent_requests": 200,
            "rate_limit_per_minute": 5000
        }
    )
    ```

    Parameters
    ----------
    corpus_adapter: Corpus embedding protocol adapter
    model: Model identifier for embedding operations
    **kwargs: Additional arguments for CorpusMCPEmbeddings

    Returns
    -------
    MCPEmbedder compatible embedder instance optimized for MCP server workflows
    """
    embedder = CorpusMCPEmbeddings(
        corpus_adapter=corpus_adapter,
        model=model,
        **kwargs,
    )

    logger.info(
        "MCP embedder created successfully with model: %s",
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