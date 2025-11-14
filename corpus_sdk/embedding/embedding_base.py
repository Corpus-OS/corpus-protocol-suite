# corpus_sdk/embedding/embedding_base.py
# SPDX-License-Identifier: Apache-2.0
"""
Adapter SDK — Embedding Protocol V1 (public contract + production-grade base)

Purpose
-------
A stable, vendor-neutral API for text embedding generation — with structured errors,
caching strategies, rate limiting, and production observability.

This protocol enables seamless integration with any embedding model provider while
maintaining production-grade security, performance monitoring, and operational rigor
for high-scale text embedding workloads.

Design Philosophy
-----------------
- Minimal surface area: Core embedding operations only, no vendor-specific extensions
- Async-first: All operations are non-blocking for high-concurrency environments
- Production hardened: Built-in caching, circuit breaking, backpressure, and metrics
- Extensible: Capability discovery allows for model-specific features
- Performance optimized: Built-in caching strategies for embedding generation
- Wire-first: Code-level contracts map cleanly onto a canonical JSON envelope

Deliberate Non-Goals
--------------------
- No text preprocessing, tokenization, or chunking strategies
- No model training, fine-tuning, or version management
- No vendor-specific model architectures or optimizations
- No client-side embedding post-processing or normalization

Those behaviors live in the text processing and model management layers.

Mode Strategy
-------------
Two operating modes ensure clean composition with external control planes while
offering a safe "batteries included" option for direct use:

- mode: "thin" (default)
    For composition under an external manager/router. All policies are no-ops:
    no caching, no rate limiting, no circuit breaker, no deadline enforcement.
    Use this when your closed-source layer provides resiliency & control.

- mode: "standalone"
    For direct use. Enables:
      - basic deadline enforcement
      - a small circuit breaker
      - an in-memory TTL cache (read paths)
      - a simple token-bucket rate limiter
    Suitable for development and light production. Not a replacement for a
    full-blown distributed control plane.

Versioning
----------
Follow SemVer against EMBEDDING_PROTOCOL_VERSION. Minor versions are strictly additive.
- Patch (x.y.Z): Editorial clarifications, non-breaking fixes
- Minor (x.Y.z): New optional parameters, capabilities, or methods
- Major (X.y.z): Breaking changes to signatures or behavior

Wire Contract (Canonical Interface)
-----------------------------------
The canonical interoperability surface for this protocol is the JSON wire envelope.
This module defines a code-level interface (EmbeddingProtocolV1 / BaseEmbeddingAdapter)
plus a thin wire adapter (WireEmbeddingHandler) that maps envelopes ⇄ typed methods.

All requests MUST use the following envelope shape:

    {
        "op": "embedding.<operation>",
        "ctx": {
            "request_id": "...",
            "idempotency_key": "...",
            "deadline_ms": 1234567890,
            "traceparent": "...",
            "tenant": "...",
            "attrs": { ... }
        },
        "args": { ... }  # operation-specific
    }

Unary Responses (success):

    {
        "ok": true,
        "code": "OK",
        "ms": <float>,          # elapsed milliseconds (best-effort)
        "result": { ... }       # operation-specific payload
    }

Unary Responses (error):

    {
        "ok": false,
        "code": "<UPPER_SNAKE_CASE>",   # e.g. BAD_REQUEST, UNAVAILABLE
        "error": "<ErrorClassName>",    # e.g. BadRequest
        "message": "<human readable>",
        "retry_after_ms": <int|null>,
        "details": { ... } | null,
        "ms": <float>
    }

The WireEmbeddingHandler in this file is the reference adapter for this contract and is
intentionally transport-agnostic (HTTP, gRPC, WebSocket, etc.).
"""

from __future__ import annotations

import asyncio
import time
import hashlib
import math
import logging
from dataclasses import dataclass, asdict
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

EMBEDDING_PROTOCOL_VERSION = "1.0.0"
EMBEDDING_PROTOCOL_ID = "embedding/v1.0"
LOG = logging.getLogger(__name__)

# =============================================================================
# Core Type Definitions
# =============================================================================

@dataclass(frozen=True)
class EmbeddingVector:
    """
    A single embedding vector with metadata.

    Attributes:
        vector: The embedding vector as a list of floats
        text: The source text that was embedded
        model: Model used to generate the embedding
        dimensions: Vector dimensions
    """
    vector: List[float]
    text: str
    model: str
    dimensions: int


@dataclass(frozen=True)
class EmbeddingResult:
    """
    Result from embedding generation operations.

    Attributes:
        embeddings: List of generated embedding vectors
        model: Model used for generation
        total_tokens: Total tokens processed (if available)
        processing_time_ms: Time taken to generate embeddings
    """
    embeddings: List[EmbeddingVector]
    model: str
    total_tokens: Optional[int] = None
    processing_time_ms: Optional[float] = None


@dataclass(frozen=True)
class EmbeddingBatch:
    """
    Batch of texts for embedding generation.

    Attributes:
        texts: List of texts to embed
        model: Target model for embedding generation
        truncate: Whether to truncate long texts (True) or error (False)
        normalize: Whether to normalize output vectors to unit length
    """
    texts: List[str]
    model: str
    truncate: bool = True
    normalize: bool = False


# =============================================================================
# Normalized Errors (with retry hints and operational guidance)
# =============================================================================

class EmbeddingAdapterError(Exception):
    """
    Base exception for all embedding adapter errors.

    Provides structured error information including retry guidance, resource limits,
    and operational suggestions for callers to handle failures gracefully.

    Attributes:
        message: Human-readable error description
        code: Machine-readable error code for programmatic handling
        retry_after_ms: Suggested delay before retry (None if not retryable)
        resource_scope: Scope of resource limitation ("model", "token_limit", "rate_limit")
        suggested_batch_reduction: Percentage reduction suggestion for batch size
        details: Additional context-specific error details (SIEM-safe, JSON-serializable)
    """
    def __init__(
        self,
        message: str = "",
        *,
        code: Optional[str] = None,
        retry_after_ms: Optional[int] = None,
        resource_scope: Optional[str] = None,
        suggested_batch_reduction: Optional[int] = None,
        details: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.retry_after_ms = retry_after_ms
        self.resource_scope = resource_scope
        self.suggested_batch_reduction = suggested_batch_reduction
        # Ensure JSON-serializable, shallow mapping for SIEM safety
        self.details = dict(details or {})

    def asdict(self) -> Dict[str, Any]:
        """
        Convert error to dictionary for serialization and logging.

        Only includes SIEM-safe, low-cardinality data.
        """
        return {
            "message": self.message,
            "code": self.code,
            "retry_after_ms": self.retry_after_ms,
            "resource_scope": self.resource_scope,
            "suggested_batch_reduction": self.suggested_batch_reduction,
            "details": {k: self.details[k] for k in sorted(self.details)},
        }

    def __str__(self) -> str:
        base = self.message or self.__class__.__name__
        if self.code:
            base += f" [code={self.code}]"
        if self.retry_after_ms is not None:
            base += f" retry_after_ms={self.retry_after_ms}"
        if self.resource_scope:
            base += f" resource_scope={self.resource_scope}"
        if self.suggested_batch_reduction is not None:
            base += f" suggested_batch_reduction={self.suggested_batch_reduction}%"
        if self.details:
            base += f" details={self.details}"
        return base


class BadRequest(EmbeddingAdapterError):
    """Client sent an invalid request (malformed texts, invalid parameters)."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "BAD_REQUEST")
        super().__init__(message, **kwargs)


class AuthError(EmbeddingAdapterError):
    """Authentication or authorization failed (invalid credentials, permissions)."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "AUTH_ERROR")
        super().__init__(message, **kwargs)


class ResourceExhausted(EmbeddingAdapterError):
    """Quota, rate limit, or resource constraints exceeded."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "RESOURCE_EXHAUSTED")
        super().__init__(message, **kwargs)


class TextTooLong(EmbeddingAdapterError):
    """Input text exceeds model's maximum context length."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "TEXT_TOO_LONG")
        super().__init__(message, **kwargs)


class ModelNotAvailable(EmbeddingAdapterError):
    """Requested embedding model is not available."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "MODEL_NOT_AVAILABLE")
        super().__init__(message, **kwargs)


class TransientNetwork(EmbeddingAdapterError):
    """Transient network failure that may succeed on retry."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "TRANSIENT_NETWORK")
        super().__init__(message, **kwargs)


class Unavailable(EmbeddingAdapterError):
    """Service is temporarily unavailable or overloaded."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "UNAVAILABLE")
        super().__init__(message, **kwargs)


class NotSupported(EmbeddingAdapterError):
    """Requested operation or parameter is not supported."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "NOT_SUPPORTED")
        super().__init__(message, **kwargs)


class DeadlineExceeded(EmbeddingAdapterError):
    """Operation exceeded ctx.deadline_ms budget."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "DEADLINE_EXCEEDED")
        super().__init__(message, **kwargs)


# =============================================================================
# Context (used for deadlines, identity, SIEM-safe metrics)
# =============================================================================

@dataclass(frozen=True)
class OperationContext:
    """
    Context for embedding operations providing tracing, deadlines, and multi-tenant isolation.

    All context information is propagated through the call chain and used for
    observability, security, and operational control without exposing sensitive data.

    Attributes:
        request_id: Unique identifier for the request chain (correlation ID)
        idempotency_key: Key for ensuring idempotent operations (when supported)
        deadline_ms: Absolute epoch milliseconds when operation should timeout
        traceparent: W3C Trace Context header for distributed tracing
        tenant: Multi-tenant isolation scope (NEVER logged or exposed in metrics)
        attrs: Additional operation attributes for extensibility and middleware
    """
    request_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    deadline_ms: Optional[int] = None
    traceparent: Optional[str] = None
    tenant: Optional[str] = None
    attrs: Mapping[str, Any] = None

    def __post_init__(self) -> None:
        """Ensure attrs is always a valid dictionary."""
        if self.attrs is None:
            object.__setattr__(self, "attrs", {})

    def remaining_ms(self) -> Optional[int]:
        """
        Return remaining milliseconds until deadline, or None if no deadline set.

        Always non-negative (0 if expired).
        """
        if self.deadline_ms is None:
            return None
        now_ms = int(time.time() * 1000)
        return max(0, self.deadline_ms - now_ms)


# =============================================================================
# Metrics Interface (SIEM-safe, low-cardinality)
# =============================================================================

class MetricsSink(Protocol):
    """
    Protocol for metrics collection implementations.

    Used for operational monitoring without exposing sensitive information.
    All metrics must be low-cardinality and never include PII or tenant identifiers.
    """
    def observe(
        self,
        *,
        component: str,
        op: str,
        ms: float,
        ok: bool,
        code: str = "OK",
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Record operation timing and status.

        Args:
            component: Component name (e.g., "embedding")
            op: Operation name (e.g., "embed", "embed_batch")
            ms: Operation duration in milliseconds
            ok: Whether operation succeeded
            code: Status code (error class name or "OK")
            extra: Additional low-cardinality dimensions
        """
        ...

    def counter(
        self,
        *,
        component: str,
        name: str,
        value: int = 1,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Increment a counter metric.

        Args:
            component: Component name (e.g., "embedding")
            name: Counter name (e.g., "texts_embedded", "tokens_processed")
            value: Increment value
            extra: Additional low-cardinality dimensions
        """
        ...


class NoopMetrics:
    """No-operation metrics sink for testing or when metrics are disabled."""
    def observe(self, **_: Any) -> None:
        ...
    def counter(self, **_: Any) -> None:
        ...


# =============================================================================
# Pluggable policy interfaces (deadlines, truncation, normalization, CB, cache)
# =============================================================================

class DeadlinePolicy(Protocol):
    """Strategy to apply time budgets (ctx.deadline_ms) to awaits."""
    async def wrap(self, awaitable, ctx: Optional[OperationContext]) -> Any:
        ...


class TruncationPolicy(Protocol):
    """Strategy to deterministically truncate text when allowed."""
    def apply(self, text: str, max_len: Optional[int], allow: bool) -> Tuple[str, bool]:
        ...


class NormalizationPolicy(Protocol):
    """Strategy to normalize vectors when requested."""
    def normalize(self, vec: List[float]) -> List[float]:
        ...


class CircuitBreaker(Protocol):
    """Minimal circuit breaker interface."""
    def allow(self) -> bool:
        ...
    def on_success(self) -> None:
        ...
    def on_error(self, err: Exception) -> None:
        ...


class Cache(Protocol):
    """Minimal async cache interface."""
    async def get(self, key: str) -> Optional[Any]:
        ...
    async def set(self, key: str, value: Any, ttl_s: int) -> None:
        ...


class RateLimiter(Protocol):
    """Minimal rate limiter interface."""
    async def acquire(self) -> None:
        ...
    def release(self) -> None:
        ...


# ---- No-op / simple policies ----

class NoopDeadline:
    """No-op deadline policy (no timing/timeout behavior)."""
    async def wrap(self, awaitable, ctx: Optional[OperationContext]) -> Any:
        return await awaitable


class EnforcingDeadline:
    """
    Deadline policy that enforces ctx.deadline_ms using asyncio.wait_for.

    If deadline is already expired, fail fast with DeadlineExceeded.
    """
    async def wrap(self, awaitable, ctx: Optional[OperationContext]) -> Any:
        if ctx is None or ctx.deadline_ms is None:
            return await awaitable
        remaining = ctx.remaining_ms()
        if remaining is not None and remaining <= 0:
            raise DeadlineExceeded("deadline already exceeded")
        try:
            return await asyncio.wait_for(
                awaitable,
                timeout=(remaining / 1000.0 if remaining is not None else None),
            )
        except asyncio.TimeoutError as e:
            raise DeadlineExceeded("operation timed out") from e


class SimpleCharTruncation:
    """Deterministic truncation policy based on max character length."""
    def apply(self, text: str, max_len: Optional[int], allow: bool) -> Tuple[str, bool]:
        if not max_len or len(text) <= max_len:
            return text, False
        if not allow:
            raise TextTooLong(f"text exceeds maximum length of {max_len}")
        return text[:max_len], True


class L2Normalization:
    """L2-normalize embedding vectors when requested."""
    def normalize(self, vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class NoopBreaker:
    def allow(self) -> bool:
        return True
    def on_success(self) -> None:
        ...
    def on_error(self, err: Exception) -> None:
        ...


class SimpleCircuitBreaker:
    """
    Extremely small circuit breaker:
      - Opens after `failure_threshold` consecutive errors.
      - Half-opens after `cooldown_s`.
      - Closes on the first success while half-open.

    Intended for standalone/dev use only (not distributed).
    """
    def __init__(self, failure_threshold: int = 5, cooldown_s: float = 5.0) -> None:
        self._failure_threshold = max(1, int(failure_threshold))
        self._cooldown_s = max(0.1, float(cooldown_s))
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._half_open = False

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        elapsed = time.monotonic() - self._opened_at
        if elapsed >= self._cooldown_s:
            # Half-open: allow a probe request.
            self._half_open = True
            return True
        return False

    def on_success(self) -> None:
        # Close on successful probe; reset failures.
        self._failures = 0
        self._opened_at = None
        self._half_open = False

    def on_error(self, _err: Exception) -> None:
        # Count consecutive failures and open when threshold exceeded.
        self._failures += 1
        if self._failures >= self._failure_threshold:
            self._opened_at = time.monotonic()
            self._failures = 0
            self._half_open = False


class NoopCache:
    """No-op cache used in thin/composed mode."""
    async def get(self, key: str) -> Optional[Any]:
        return None
    async def set(self, key: str, value: Any, ttl_s: int) -> None:
        return None


class InMemoryTTLCache:
    """
    Tiny in-memory cache with TTL. Suitable for demos/tests only.

    Not process-safe or distributed.
    """
    def __init__(self) -> None:
        self._store: Dict[str, Tuple[float, Any]] = {}

    async def get(self, key: str) -> Optional[Any]:
        now = time.monotonic()
        item = self._store.get(key)
        if not item:
            return None
        exp, val = item
        if exp < now:
            self._store.pop(key, None)
            return None
        return val

    async def set(self, key: str, value: Any, ttl_s: int) -> None:
        ttl = max(0, int(ttl_s))
        self._store[key] = (time.monotonic() + ttl, value)


class NoopLimiter:
    """No-op limiter used in thin/composed mode."""
    async def acquire(self) -> None:
        return None
    def release(self) -> None:
        return None


class TokenBucketLimiter:
    """
    Very simple token bucket limiter.

    Args:
        rate: tokens per second
        burst: max bucket size

    Intended for standalone/dev; avoids impacting main path on internal errors.
    """
    def __init__(self, rate: float = 50.0, burst: int = 50) -> None:
        self._rate = max(0.1, float(rate))
        self._burst = max(1, int(burst))
        self._tokens = float(self._burst)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last
                if elapsed > 0:
                    self._tokens = min(
                        self._burst,
                        self._tokens + elapsed * self._rate,
                    )
                    self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                needed = (1.0 - self._tokens) / self._rate
                await asyncio.sleep(max(needed, 0.001))

    def release(self) -> None:
        # Classic token bucket: only charge on acquire.
        return None


# =============================================================================
# Capabilities (dynamic discovery for routing and planning)
# =============================================================================

@dataclass(frozen=True)
class EmbeddingCapabilities:
    """
    Describes the capabilities and limitations of an embedding adapter implementation.

    Used by routing layers for intelligent model selection, request planning,
    and feature compatibility checking across different embedding providers.

    Attributes:
        server: Backend server identifier (e.g., "openai", "cohere", "huggingface")
        version: Backend server version string
        supported_models: Supported embedding model names
        max_batch_size: Maximum texts per batch operation
        max_text_length: Maximum characters per text input
        max_dimensions: Maximum vector dimensions supported
        supports_normalization: Whether vector normalization is supported
        supports_truncation: Whether text truncation is supported
        supports_token_counting: Whether token counting is available
        idempotent_operations: Whether operations are idempotent with idempotency_key
        supports_multi_tenant: Whether multi-tenant isolation is supported
        normalizes_at_source: Whether adapter normalizes vectors at source when requested
        truncation_mode: "base" or "adapter" to signal where truncation is applied
        supports_deadline: Whether adapter cooperates with deadline cancellation
    """
    server: str
    version: str
    supported_models: Tuple[str, ...]
    max_batch_size: Optional[int] = None
    max_text_length: Optional[int] = None
    max_dimensions: Optional[int] = None
    supports_normalization: bool = False
    supports_truncation: bool = True
    supports_token_counting: bool = False
    idempotent_operations: bool = False
    supports_multi_tenant: bool = False
    normalizes_at_source: bool = False
    truncation_mode: str = "base"
    supports_deadline: bool = True


# =============================================================================
# Operation Specifications
# =============================================================================

@dataclass(frozen=True)
class EmbedSpec:
    """
    Specification for single text embedding generation.

    Attributes:
        text: Text to convert to embedding
        model: Target model for embedding generation
        truncate: Whether to truncate long texts (True) or error (False)
        normalize: Whether to normalize output vector to unit length
    """
    text: str
    model: str
    truncate: bool = True
    normalize: bool = False


@dataclass(frozen=True)
class BatchEmbedSpec:
    """
    Specification for batch text embedding generation.

    Attributes:
        texts: List of texts to embed
        model: Target model for embedding generation
        truncate: Whether to truncate long texts (True) or error (False)
        normalize: Whether to normalize output vectors to unit length
    """
    texts: List[str]
    model: str
    truncate: bool = True
    normalize: bool = False


# =============================================================================
# Operation Results
# =============================================================================

@dataclass
class EmbedResult:
    """
    Result from single text embedding generation.

    Attributes:
        embedding: Generated embedding vector
        model: Model used for generation
        text: Original input text
        tokens_used: Number of tokens processed (if available)
        truncated: Whether input text was truncated
    """
    embedding: EmbeddingVector
    model: str
    text: str
    tokens_used: Optional[int] = None
    truncated: bool = False


@dataclass
class BatchEmbedResult:
    """
    Result from batch text embedding generation.

    Attributes:
        embeddings: List of generated embedding vectors
        model: Model used for generation
        total_texts: Total number of texts processed
        total_tokens: Total tokens processed (if available)
        failed_texts: List of per-text failure details:
            - index: index in input batch
            - text: original text
            - error: error class name
            - code: stable error code (if available)
            - message: human-readable error
    """
    embeddings: List[EmbeddingVector]
    model: str
    total_texts: int
    total_tokens: Optional[int] = None
    failed_texts: List[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.failed_texts is None:
            self.failed_texts = []


# =============================================================================
# Stable Protocol Interface (async, versioned contract)
# =============================================================================

@runtime_checkable
class EmbeddingProtocolV1(Protocol):
    """
    Protocol defining the Embedding Protocol V1 interface.

    Implement this protocol to create compatible embedding adapters. All methods are async
    and designed for high-concurrency environments. The protocol is runtime-checkable
    for dynamic adapter validation.
    """

    async def capabilities(self) -> EmbeddingCapabilities:
        """
        Get the capabilities of this embedding adapter.

        Returns:
            EmbeddingCapabilities: Description of supported features and limitations
        """
        ...

    async def embed(
        self,
        spec: EmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> EmbedResult:
        """Generate embedding for a single text."""
        ...

    async def embed_batch(
        self,
        spec: BatchEmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchEmbedResult:
        """Generate embeddings for multiple texts in batch."""
        ...

    async def count_tokens(
        self,
        text: str,
        model: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> int:
        """Count tokens in text for a specific model."""
        ...

    async def health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """Check the health status of the embedding backend."""
        ...


# =============================================================================
# Base Instrumented Adapter (validation, metrics, error handling)
# =============================================================================

class BaseEmbeddingAdapter(EmbeddingProtocolV1):
    """
    Base class for implementing Embedding Protocol V1 adapters.

    Provides common validation, metrics instrumentation, error handling, and
    SIEM-safe observability. Implementers override the `_do_*` methods to provide
    backend-specific functionality while inheriting:

      - Normalized error taxonomy
      - Deadline enforcement via DeadlinePolicy
      - Circuit breaker integration
      - Optional in-memory caching (standalone)
      - Rate limiting via token bucket (standalone)
      - Canonical metrics emission for ops & latency

    Mode Strategy
    -------------
    - "thin" (default): For composition under external control planes.
      All infra (deadline, breaker, limiter, cache) defaults to no-op.

    - "standalone": For direct use. Turns on:
        - EnforcingDeadline
        - SimpleCircuitBreaker
        - InMemoryTTLCache
        - TokenBucketLimiter
      Intended for development and light production.
    """

    _component = "embedding"

    def __init__(
        self,
        *,
        mode: str = "thin",
        metrics: Optional[MetricsSink] = None,
        deadline_policy: Optional[DeadlinePolicy] = None,
        truncation: Optional[TruncationPolicy] = None,
        normalization: Optional[NormalizationPolicy] = None,
        breaker: Optional[CircuitBreaker] = None,
        cache: Optional[Cache] = None,
        limiter: Optional[RateLimiter] = None,
        tag_model_in_metrics: Optional[bool] = None,
        cache_embed_ttl_s: int = 60,
        cache_caps_ttl_s: int = 30,
    ) -> None:
        """
        Initialize the embedding adapter with metrics instrumentation and optional policies.

        Args:
            mode: "thin" (no-op infra; meant to be wrapped by a provider) or
                  "standalone" (turn on basic demo policies).
            metrics: Metrics sink for operational monitoring. Uses NoopMetrics if None.
            deadline_policy: Optional deadline policy to enforce ctx.deadline_ms.
            truncation: Optional truncation policy; defaults vary by mode.
            normalization: Optional normalization policy; defaults vary by mode.
            breaker: Optional circuit breaker; defaults vary by mode.
            cache: Optional async cache; defaults vary by mode.
            limiter: Optional rate limiter; defaults vary by mode.
            tag_model_in_metrics: Whether to include 'model' as a metric tag.
            cache_embed_ttl_s: TTL for embed() cache entries when using in-memory cache.
            cache_caps_ttl_s: TTL for capabilities() cache entries in standalone mode.
        """
        m = (mode or "thin").strip().lower()
        if m not in {"thin", "standalone"}:
            m = "thin"
        self._mode = m

        self._metrics: MetricsSink = metrics or NoopMetrics()

        # Warn if standalone without metrics sink
        if self._mode == "standalone" and isinstance(self._metrics, NoopMetrics):
            LOG.warning(
                "Using standalone mode without metrics - "
                "consider providing a MetricsSink for production use"
            )

        # Policies/infra defaults by mode (overridable)
        if self._mode == "thin":
            self._deadline: DeadlinePolicy = deadline_policy or NoopDeadline()
            self._trunc: TruncationPolicy = truncation or SimpleCharTruncation()
            self._norm: NormalizationPolicy = normalization or L2Normalization()
            self._breaker: CircuitBreaker = breaker or NoopBreaker()
            self._cache: Cache = cache or NoopCache()
            self._limiter: RateLimiter = limiter or NoopLimiter()
            self._tag_model_in_metrics: bool = (
                bool(tag_model_in_metrics) if tag_model_in_metrics is not None else False
            )
        else:  # "standalone"
            self._deadline: DeadlinePolicy = deadline_policy or EnforcingDeadline()
            self._trunc: TruncationPolicy = truncation or SimpleCharTruncation()
            self._norm: NormalizationPolicy = normalization or L2Normalization()
            self._breaker: CircuitBreaker = breaker or SimpleCircuitBreaker()
            self._cache: Cache = cache or InMemoryTTLCache()
            self._limiter: RateLimiter = limiter or TokenBucketLimiter()
            self._tag_model_in_metrics: bool = (
                bool(tag_model_in_metrics) if tag_model_in_metrics is not None else True
            )

        self._cache_embed_ttl_s: int = max(1, int(cache_embed_ttl_s))
        self._cache_caps_ttl_s: int = max(1, int(cache_caps_ttl_s))

    # --- internal helpers (validation, hashing, metrics, deadlines) ---

    @staticmethod
    def _require_non_empty(name: str, value: str) -> None:
        """Validate that a string value is non-empty."""
        if not isinstance(value, str) or not value.strip():
            raise BadRequest(f"{name} must be a non-empty string")

    @staticmethod
    def _tenant_hash(tenant: Optional[str]) -> Optional[str]:
        """
        Create privacy-preserving hash of tenant identifier for metrics.

        Raw tenant IDs MUST NOT appear directly in metrics.
        """
        if not tenant:
            return None
        return hashlib.sha256(tenant.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _safe_model_tag(model: str) -> Optional[str]:
        """
        Normalize model name for metrics to avoid cardinality explosions.

        - Empty/None → None
        - Length > 100 → "unknown"
        """
        if not model:
            return None
        m = str(model)
        if len(m) > 100:
            return "unknown"
        return m

    def _record(
        self,
        op: str,
        t0: float,
        ok: bool,
        *,
        code: str = "OK",
        ctx: Optional[OperationContext] = None,
        **extra: Any,
    ) -> None:
        """
        Record operation metrics with context and tenant hashing.

        Never lets metrics failures impact the main control path.
        """
        try:
            ms = (time.monotonic() - t0) * 1000.0
            x = dict(extra or {})
            if ctx:
                tenant_h = self._tenant_hash(ctx.tenant)
                if tenant_h:
                    x.setdefault("tenant_hash", tenant_h)
                rem = ctx.remaining_ms()
                if rem is not None:
                    if rem < 1000:
                        x["deadline_bucket"] = "<1s"
                    elif rem < 5000:
                        x["deadline_bucket"] = "<5s"
                    elif rem < 15000:
                        x["deadline_bucket"] = "<15s"
                    elif rem < 60000:
                        x["deadline_bucket"] = "<60s"
                    else:
                        x["deadline_bucket"] = ">=60s"
            self._metrics.observe(
                component=self._component,
                op=op,
                ms=ms,
                ok=ok,
                code=code,
                extra=x or None,
            )
            if not ok:
                self._metrics.counter(
                    component=self._component,
                    name="errors_total",
                    value=1,
                    extra={"code": code},
                )
        except Exception:
            # Metrics failures MUST NOT affect request path.
            pass

    async def _apply_deadline(self, awaitable, ctx: Optional[OperationContext]) -> Any:
        """
        Apply the configured deadline policy to an awaitable.

        Any asyncio.TimeoutError not handled by the policy is normalized into DeadlineExceeded.
        """
        try:
            return await self._deadline.wrap(awaitable, ctx)
        except DeadlineExceeded:
            raise
        except asyncio.TimeoutError as e:
            raise DeadlineExceeded("operation timed out") from e

    def _fail_if_expired(self, ctx: Optional[OperationContext]) -> None:
        """
        Fail fast if ctx.deadline_ms is already expired.

        Avoids wasting backend capacity on doomed requests.
        """
        if ctx is None or ctx.deadline_ms is None:
            return
        remaining = ctx.remaining_ms()
        if remaining is not None and remaining <= 0:
            raise DeadlineExceeded("deadline already exceeded")

    @staticmethod
    def _caps_cache_key() -> str:
        """Cache key for capabilities() when cached."""
        return "embedding:capabilities"

    def _embed_cache_key(
        self,
        model: str,
        normalize: bool,
        text: str,
        ctx: Optional[OperationContext],
    ) -> str:
        """
        Construct a cache key for embed() that:

          - Avoids leaking raw text (uses SHA-256 digest).
          - Is isolated per tenant via tenant hash (or 'global' if none).
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        tenant_hash = self._tenant_hash(ctx.tenant) if ctx and ctx.tenant else "global"
        return (
            f"embedding:embed:"
            f"tenant={tenant_hash}:"
            f"model={model}:"
            f"norm={int(normalize)}:"
            f"text={digest}"
        )

    async def _with_gates_unary(
        self,
        *,
        op: str,
        ctx: Optional[OperationContext],
        call_coro,
        metric_extra: Optional[Mapping[str, Any]] = None,
        error_extra: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """
        Shared unary gate wrapper:

          - deadline preflight
          - circuit breaker allow
          - rate limiter acquire/release
          - deadline enforcement
          - metrics
          - breaker success/error wiring
        """
        self._fail_if_expired(ctx)

        # Circuit breaker gate
        if not self._breaker.allow():
            raise Unavailable("circuit open")

        # Rate limit acquire
        await self._limiter.acquire()

        t0 = time.monotonic()
        try:
            result = await self._apply_deadline(call_coro, ctx)
            self._record(
                op,
                t0,
                True,
                ctx=ctx,
                **(metric_extra or {}),
            )
            self._breaker.on_success()
            return result
        except EmbeddingAdapterError as e:
            code = e.code or type(e).__name__
            extra = dict(error_extra or {})
            self._record(
                op,
                t0,
                False,
                code=code,
                ctx=ctx,
                **extra,
            )
            self._breaker.on_error(e)
            raise
        except Exception as e:
            extra = dict(error_extra or {})
            self._record(
                op,
                t0,
                False,
                code="UnhandledException",
                ctx=ctx,
                **extra,
            )
            self._breaker.on_error(e)
            raise
        finally:
            self._limiter.release()

    # --- public APIs (use helpers + backend hooks) ---

    async def capabilities(self) -> EmbeddingCapabilities:
        """
        Get the capabilities of this embedding adapter (with optional caching).

        In standalone mode with an in-memory TTL cache, caches capabilities
        using the configured cache_caps_ttl_s.
        """
        t0 = time.monotonic()
        try:
            if self._mode == "standalone" and isinstance(self._cache, InMemoryTTLCache):
                cached = await self._cache.get(self._caps_cache_key())
                if cached:
                    self._metrics.counter(
                        component=self._component,
                        name="cache_hits",
                        value=1,
                        extra={"op": "capabilities"},
                    )
                    self._record("capabilities", t0, True)
                    return cached

            caps = await self._apply_deadline(self._do_capabilities(), ctx=None)

            if self._mode == "standalone" and isinstance(self._cache, InMemoryTTLCache):
                try:
                    await self._cache.set(
                        self._caps_cache_key(),
                        caps,
                        ttl_s=self._cache_caps_ttl_s,
                    )
                except Exception:
                    # Cache failures are non-fatal.
                    pass

            self._record("capabilities", t0, True)
            return caps
        except EmbeddingAdapterError as e:
            code = e.code or type(e).__name__
            self._record("capabilities", t0, False, code=code)
            raise
        except Exception as e:
            self._record("capabilities", t0, False, code="UNAVAILABLE")
            raise Unavailable("capabilities fetch failed") from e

    async def embed(
        self,
        spec: EmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> EmbedResult:
        """
        Generate embedding for a single text with validation, gates, and metrics.

        See EmbeddingProtocolV1.embed for full documentation.
        """
        self._require_non_empty("text", spec.text)
        self._require_non_empty("model", spec.model)

        async def _run() -> EmbedResult:
            # Capabilities for validation and limits.
            caps = await self._do_capabilities()
            if spec.model not in caps.supported_models:
                raise ModelNotAvailable(f"Model '{spec.model}' is not supported")

            # Deterministic truncation if needed.
            text = spec.text
            truncated = False
            if caps.max_text_length:
                text, truncated = self._trunc.apply(
                    text,
                    caps.max_text_length,
                    spec.truncate,
                )

            eff_spec = EmbedSpec(
                text=text,
                model=spec.model,
                truncate=spec.truncate,
                normalize=spec.normalize,
            )

            # Optional cache: isolated per tenant via _embed_cache_key.
            result: Optional[EmbedResult] = None
            cache_key: Optional[str] = None
            if isinstance(self._cache, InMemoryTTLCache):
                cache_key = self._embed_cache_key(
                    eff_spec.model,
                    eff_spec.normalize,
                    eff_spec.text,
                    ctx,
                )
                cached = await self._cache.get(cache_key)
                if cached:
                    self._metrics.counter(
                        component=self._component,
                        name="cache_hits",
                        value=1,
                        extra={"op": "embed"},
                    )
                    result = cached

            if result is None:
                # Provider-specific embed.
                result = await self._do_embed(eff_spec, ctx=ctx)

                # Post-processing: normalization if requested and not handled at source.
                if eff_spec.normalize:
                    if not caps.supports_normalization:
                        raise NotSupported("normalization not supported for this adapter")
                    if not caps.normalizes_at_source:
                        vec = self._norm.normalize(result.embedding.vector)
                        result.embedding = EmbeddingVector(
                            vector=vec,
                            text=result.text,
                            model=result.model,
                            dimensions=len(vec),
                        )

                # Mark truncation result flag.
                result.truncated = bool(truncated)

                # Cache set (best-effort).
                if cache_key is not None:
                    try:
                        await self._cache.set(
                            cache_key,
                            result,
                            ttl_s=self._cache_embed_ttl_s,
                        )
                    except Exception:
                        pass

            # Per-op counters.
            self._metrics.counter(
                component=self._component,
                name="texts_embedded",
                value=1,
            )
            if result.tokens_used is not None:
                self._metrics.counter(
                    component=self._component,
                    name="tokens_processed",
                    value=int(result.tokens_used),
                )

            return result

        metric_extra: Dict[str, Any] = {}
        error_extra: Dict[str, Any] = {}

        if self._tag_model_in_metrics:
            model_tag = self._safe_model_tag(spec.model)
            if model_tag:
                metric_extra["model"] = model_tag
                error_extra["model"] = model_tag

        return await self._with_gates_unary(
            op="embed",
            ctx=ctx,
            call_coro=_run(),
            metric_extra=metric_extra,
            error_extra=error_extra,
        )

    async def embed_batch(
        self,
        spec: BatchEmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchEmbedResult:
        """
        Generate embeddings for multiple texts with validation, gates, and metrics.

        Supports:
          - Provider-native batch behavior via _do_embed_batch.
          - Optional partial-success fallback when batch is not supported:
            falls back to per-item _do_embed calls and populates failed_texts.
        """
        self._require_non_empty("model", spec.model)
        if not spec.texts:
            raise BadRequest("texts must not be empty")

        async def _run() -> BatchEmbedResult:
            caps = await self._do_capabilities()
            if spec.model not in caps.supported_models:
                raise ModelNotAvailable(f"Model '{spec.model}' is not supported")

            if caps.max_batch_size and len(spec.texts) > caps.max_batch_size:
                raise BadRequest(
                    f"Batch size {len(spec.texts)} exceeds maximum of {caps.max_batch_size}",
                    details={"max_batch_size": caps.max_batch_size},
                )

            # Validate and possibly truncate each text deterministically.
            eff_texts: List[str] = []
            for text in spec.texts:
                self._require_non_empty("text", text)
                if caps.max_text_length:
                    new_text, _ = self._trunc.apply(
                        text,
                        caps.max_text_length,
                        spec.truncate,
                    )
                    eff_texts.append(new_text)
                else:
                    eff_texts.append(text)

            eff_spec = BatchEmbedSpec(
                texts=eff_texts,
                model=spec.model,
                truncate=spec.truncate,
                normalize=spec.normalize,
            )

            try:
                # Primary path: provider-specific batch implementation.
                result = await self._do_embed_batch(eff_spec, ctx=ctx)
            except NotSupported:
                # Fallback path: partial-success aware per-item embedding.
                embeddings: List[EmbeddingVector] = []
                failed: List[Dict[str, Any]] = []

                for idx, text in enumerate(eff_spec.texts):
                    try:
                        single_spec = EmbedSpec(
                            text=text,
                            model=eff_spec.model,
                            truncate=eff_spec.truncate,
                            normalize=False,  # normalization applied uniformly below
                        )
                        single = await self._do_embed(single_spec, ctx=ctx)
                        ev = single.embedding

                        if eff_spec.normalize:
                            if not caps.supports_normalization:
                                raise NotSupported(
                                    "normalization not supported for this adapter"
                                )
                            if not caps.normalizes_at_source:
                                vec = self._norm.normalize(ev.vector)
                                ev = EmbeddingVector(
                                    vector=vec,
                                    text=ev.text,
                                    model=ev.model,
                                    dimensions=len(vec),
                                )

                        embeddings.append(ev)
                    except EmbeddingAdapterError as item_err:
                        failed.append(
                            {
                                "index": idx,
                                "text": text,
                                "error": type(item_err).__name__,
                                "code": item_err.code or type(item_err).__name__,
                                "message": item_err.message,
                            }
                        )
                    except Exception as item_err:
                        failed.append(
                            {
                                "index": idx,
                                "text": text,
                                "error": type(item_err).__name__,
                                "code": "UNAVAILABLE",
                                "message": str(item_err) or "internal error",
                            }
                        )

                result = BatchEmbedResult(
                    embeddings=embeddings,
                    model=eff_spec.model,
                    total_texts=len(spec.texts),
                    total_tokens=None,
                    failed_texts=failed,
                )

            # Post-processing: normalization if requested and not handled at source,
            # for providers that implement _do_embed_batch directly.
            if eff_spec.normalize:
                if not caps.supports_normalization:
                    raise NotSupported("normalization not supported for this adapter")
                if not caps.normalizes_at_source:
                    for i, ev in enumerate(result.embeddings):
                        vec = self._norm.normalize(ev.vector)
                        result.embeddings[i] = EmbeddingVector(
                            vector=vec,
                            text=ev.text,
                            model=ev.model,
                            dimensions=len(vec),
                        )

            # Per-op counters.
            self._metrics.counter(
                component=self._component,
                name="texts_embedded",
                value=len(result.embeddings),
            )
            if result.total_tokens is not None:
                self._metrics.counter(
                    component=self._component,
                    name="tokens_processed",
                    value=int(result.total_tokens),
                )

            return result

        metric_extra: Dict[str, Any] = {
            "batch_size": len(spec.texts),
        }
        error_extra: Dict[str, Any] = {
            "batch_size": len(spec.texts),
        }

        if self._tag_model_in_metrics:
            model_tag = self._safe_model_tag(spec.model)
            if model_tag:
                metric_extra["model"] = model_tag
                error_extra["model"] = model_tag

        return await self._with_gates_unary(
            op="embed_batch",
            ctx=ctx,
            call_coro=_run(),
            metric_extra=metric_extra,
            error_extra=error_extra,
        )

    async def count_tokens(
        self,
        text: str,
        model: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> int:
        """
        Count tokens in text with validation, gates, and metrics.
        """
        # Allow empty string; just ensure it's a string
        if not isinstance(text, str):
            raise BadRequest("text must be a string")
        self._require_non_empty("model", model)

        async def _run() -> int:
            caps = await self._do_capabilities()
            if model not in caps.supported_models:
                raise ModelNotAvailable(f"Model '{model}' is not supported")
            if not caps.supports_token_counting:
                raise NotSupported("count_tokens is not supported by this adapter")
            return await self._do_count_tokens(text, model, ctx=ctx)

        metric_extra: Dict[str, Any] = {
            "text_length": len(text),
        }
        error_extra: Dict[str, Any] = {}

        if self._tag_model_in_metrics:
            model_tag = self._safe_model_tag(model)
            if model_tag:
                metric_extra["model"] = model_tag
                error_extra["model"] = model_tag

        result = await self._with_gates_unary(
            op="count_tokens",
            ctx=ctx,
            call_coro=_run(),
            metric_extra=metric_extra,
            error_extra=error_extra,
        )

        # Successful call counter (non-critical).
        self._metrics.counter(
            component=self._component,
            name="count_tokens_calls",
            value=1,
        )
        return int(result)

    async def health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """
        Check health status with metrics instrumentation.

        Returns a small mapping; unknown keys from backends are informational only.
        """

        async def _run() -> Dict[str, Any]:
            h = await self._do_health(ctx=ctx)
            return {
                "ok": bool(h.get("ok", True)),
                "server": str(h.get("server", "")),
                "version": str(h.get("version", "")),
                "models": h.get("models", {}),
            }

        return await self._with_gates_unary(
            op="health",
            ctx=ctx,
            call_coro=_run(),
        )

    # --- async context manager (resource cleanup hook) ---

    async def __aenter__(self) -> "BaseEmbeddingAdapter":
        """Allow use as an async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """On context exit, trigger close() for backend cleanup."""
        await self.close()

    async def close(self) -> None:
        """
        Clean up resources (override in backend implementations).

        Default implementation is a no-op.
        """
        return None

    # --- hooks to implement per backend (override these) ---

    async def _do_capabilities(self) -> EmbeddingCapabilities:
        """Implement to return adapter-specific capabilities."""
        raise NotImplementedError

    async def _do_embed(
        self,
        spec: EmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> EmbedResult:
        """Implement single text embedding with validated inputs."""
        raise NotImplementedError

    async def _do_embed_batch(
        self,
        spec: BatchEmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchEmbedResult:
        """
        Implement batch text embedding with validated inputs.

        Implementers MAY populate failed_texts for partial success semantics.
        If this method raises NotSupported, BaseEmbeddingAdapter falls back to
        a per-item _do_embed()-based implementation with partial results.
        """
        raise NotImplementedError

    async def _do_count_tokens(
        self,
        text: str,
        model: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> int:
        """Implement token counting for the specified model."""
        raise NotImplementedError

    async def _do_health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """Implement health check for the embedding backend."""
        raise NotImplementedError



# =============================================================================
# Wire-Level Helpers (canonical envelopes)
# =============================================================================

def _ctx_from_wire(ctx_dict: Mapping[str, Any]) -> OperationContext:
    """
    Convert a wire-level ctx dict into an OperationContext.

    Unknown keys are ignored per protocol rules (forward compatible).
    """
    if ctx_dict is None:
        return OperationContext()
    return OperationContext(
        request_id=ctx_dict.get("request_id"),
        idempotency_key=ctx_dict.get("idempotency_key"),
        deadline_ms=ctx_dict.get("deadline_ms"),
        traceparent=ctx_dict.get("traceparent"),
        tenant=ctx_dict.get("tenant"),
        attrs=ctx_dict.get("attrs") or {},
    )


def _error_to_wire(e: Exception, ms: float) -> Dict[str, Any]:
    """
    Map EmbeddingAdapterError (or unexpected Exception) to canonical error envelope.

    Single source of truth for wire-level error normalization.
    """
    if isinstance(e, EmbeddingAdapterError):
        payload = e.asdict()
        return {
            "ok": False,
            "code": payload.get("code") or type(e).__name__.upper(),
            "error": type(e).__name__,
            "message": payload.get("message", ""),
            "retry_after_ms": payload.get("retry_after_ms"),
            "details": payload.get("details") or None,
            "ms": ms,
        }
    # Fallback: treat as UNAVAILABLE/INTERNAL for unknown exceptions.
    return {
        "ok": False,
        "code": "UNAVAILABLE",
        "error": type(e).__name__,
        "message": str(e) or "internal error",
        "retry_after_ms": None,
        "details": None,
        "ms": ms,
    }


def _success_to_wire(result: Any, ms: float) -> Dict[str, Any]:
    """
    Map typed result objects or primitives to canonical success envelope.

    Uses dataclasses.asdict() where applicable, else passes through.
    """
    if hasattr(result, "__dataclass_fields__"):
        payload = asdict(result)
    else:
        payload = result
    return {
        "ok": True,
        "code": "OK",
        "ms": ms,
        "result": payload,
    }


class WireEmbeddingHandler:
    """
    Thin wire-level adapter that exposes an EmbeddingProtocolV1 implementation using
    the canonical JSON envelope contract:

        { "op": "embedding.embed", "ctx": {...}, "args": {...} } -> { ... }

    Transport-agnostic: can be wrapped by HTTP, gRPC, WebSockets, etc.
    """

    def __init__(self, adapter: EmbeddingProtocolV1):
        self._adapter = adapter

    async def handle(self, envelope: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Handle a single unary request envelope and return a response envelope.

        Supports:
            - embedding.capabilities
            - embedding.embed
            - embedding.embed_batch
            - embedding.count_tokens
            - embedding.health
        """
        t0 = time.monotonic()
        try:
            op = envelope.get("op")
            if not isinstance(op, str):
                raise BadRequest("missing or invalid 'op'")

            ctx = _ctx_from_wire(envelope.get("ctx") or {})
            args = envelope.get("args") or {}

            if op == "embedding.capabilities":
                res = await self._adapter.capabilities()
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "embedding.embed":
                spec = EmbedSpec(
                    text=args.get("text", ""),
                    model=args.get("model", ""),
                    truncate=bool(args.get("truncate", True)),
                    normalize=bool(args.get("normalize", False)),
                )
                res = await self._adapter.embed(spec, ctx=ctx)
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "embedding.embed_batch":
                texts = args.get("texts")
                if not isinstance(texts, list):
                    raise BadRequest("texts must be provided as a list of strings")
                spec = BatchEmbedSpec(
                    texts=[str(t) for t in texts],
                    model=args.get("model", ""),
                    truncate=bool(args.get("truncate", True)),
                    normalize=bool(args.get("normalize", False)),
                )
                res = await self._adapter.embed_batch(spec, ctx=ctx)
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "embedding.count_tokens":
                text = args.get("text")
                model = args.get("model")
                if not isinstance(text, str):
                    raise BadRequest("text must be a string")
                if not isinstance(model, str):
                    raise BadRequest("model must be a string")
                res = await self._adapter.count_tokens(text=text, model=model, ctx=ctx)
                return _success_to_wire(int(res), (time.monotonic() - t0) * 1000.0)

            if op == "embedding.health":
                res = await self._adapter.health(ctx=ctx)
                return _success_to_wire(res, (time.monotonic() - t0) * 1000.0)

            raise NotSupported(f"unknown operation '{op}'")

        except Exception as e:
            ms = (time.monotonic() - t0) * 1000.0
            return _error_to_wire(e, ms)


# =============================================================================
# Public Exports
# =============================================================================

__all__ = [
    "EMBEDDING_PROTOCOL_VERSION",
    "EMBEDDING_PROTOCOL_ID",
    "EmbeddingVector",
    "EmbeddingResult",
    "EmbeddingBatch",
    "EmbeddingAdapterError",
    "BadRequest",
    "AuthError",
    "ResourceExhausted",
    "TextTooLong",
    "ModelNotAvailable",
    "TransientNetwork",
    "Unavailable",
    "NotSupported",
    "DeadlineExceeded",
    "OperationContext",
    "EmbedSpec",
    "BatchEmbedSpec",
    "EmbedResult",
    "BatchEmbedResult",
    "EmbeddingCapabilities",
    "EmbeddingProtocolV1",
    "BaseEmbeddingAdapter",
    # policy & infra extension points
    "DeadlinePolicy",
    "TruncationPolicy",
    "NormalizationPolicy",
    "CircuitBreaker",
    "Cache",
    "RateLimiter",
    "NoopDeadline",
    "EnforcingDeadline",
    "SimpleCharTruncation",
    "L2Normalization",
    "NoopBreaker",
    "SimpleCircuitBreaker",
    "NoopCache",
    "InMemoryTTLCache",
    "NoopLimiter",
    "TokenBucketLimiter",
    "MetricsSink",
    "NoopMetrics",
    # wire helpers
    "WireEmbeddingHandler",
    "_ctx_from_wire",
    "_error_to_wire",
    "_success_to_wire",
]
