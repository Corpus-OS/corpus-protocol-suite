# adapter_sdk/vector_base.py
# SPDX-License-Identifier: Apache-2.0
"""
Adapter SDK — Vector Protocol V1.0

Purpose
-------
A stable, vendor-neutral API for vector similarity search and operations — with
structured errors, caching strategies, rate limiting, and production observability.

This module serves as a reference implementation and SDK mapping for the
Vector Protocol V1.0. The canonical interface for interoperability is the
wire-level contract:

    Request:
        {
            "op": "vector.<operation>",
            "ctx": { ... },
            "args": { ... }
        }

    Response (success):
        {
            "ok": true,
            "code": "OK",
            "ms": <float>,
            "result": { ... }
        }

    Response (error):
        {
            "ok": false,
            "code": "<UPPER_SNAKE_CASE>",
            "error": "<ErrorClassName>",
            "message": "<human readable>",
            "retry_after_ms": <int|null>,
            "details": { ... }
        }

This file provides:

- Typed Python contracts that mirror the protocol data shapes
- A production-ready BaseVectorAdapter with validation, metrics, and backpressure
- A thin WireVectorHandler that converts wire envelopes <-> typed API

Design Philosophy
-----------------
- Minimal surface area: Core vector operations only, no vendor-specific extensions
- Async-first: All operations are non-blocking for high-concurrency environments
- Production hardened: Built-in caching, circuit breaking, backpressure, and metrics
- Extensible: Capability discovery allows for database-specific vector features
- Performance optimized: Built-in caching strategies for vector similarity search
- Wire-first: Types and helpers exist to faithfully implement the canonical JSON contract

Deliberate Non-Goals
--------------------
- No embedding model management or text-to-vector transformations
- No vector index tuning or optimization strategies
- No provider-specific algorithms beyond capabilities
- No client-side result re-ranking or post-processing

Those behaviors live in embedding services and upper application layers.

Mode Strategy
-------------
Two operating modes ensure clean composition with external control planes while
offering a safe "batteries included" option for direct use:

- mode: "thin" (default) — For composition under an external manager/router.
  All policies are no-ops: no caching, no rate limiting, no circuit breaker,
  no deadline enforcement. Use this when your closed-source layer provides
  resiliency, concurrency control, and caching.

- mode: "standalone" — For direct use. Enables basic deadline enforcement,
  a small circuit breaker, an in-memory TTL cache (read paths), and a simple
  token-bucket rate limiter. Suitable for development and light production.

Versioning
----------
Follow SemVer against VECTOR_PROTOCOL_VERSION. Minor versions are strictly additive.
- Patch (x.y.Z): Editorial clarifications, non-breaking fixes
- Minor (x.Y.z): New optional parameters, capabilities, or methods
- Major (X.y.z): Breaking changes to signatures or behavior
"""

from __future__ import annotations

import asyncio
import time
import hashlib
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
    Iterable,
    runtime_checkable,
    AsyncIterator,
    Union,
    NewType,
)

VECTOR_PROTOCOL_VERSION = "1.0.0"
VECTOR_PROTOCOL_ID = "vector/v1.0"
LOG = logging.getLogger(__name__)

# =============================================================================
# Core Type Definitions
# =============================================================================

VectorID = NewType('VectorID', str)
"""Type alias for vector identifiers providing explicit type safety."""

@dataclass(frozen=True)
class Vector:
    """
    A vector with optional metadata and identifier.
    
    Attributes:
        id: Unique identifier for the vector
        vector: The vector embedding as a list of floats
        metadata: Optional key-value pairs for filtering and retrieval
        namespace: Optional namespace/collection for multi-tenant isolation
    """
    id: VectorID
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None
    namespace: Optional[str] = None

@dataclass(frozen=True)
class VectorMatch:
    """
    A single vector match with similarity score.
    
    Attributes:
        vector: The matching vector
        score: Similarity score (higher = more similar)
        distance: Raw distance metric value (lower = more similar)
    """
    vector: Vector
    score: float
    distance: float

@dataclass(frozen=True)
class QueryResult:
    """
    Result from a vector similarity search query.
    
    Attributes:
        matches: List of matching vectors with similarity scores
        query_vector: The original query vector used for search
        namespace: Namespace/collection where search was performed
        total_matches: Total number of matches found (before limiting)
    """
    matches: List[VectorMatch]
    query_vector: List[float]
    namespace: str
    total_matches: int

# =============================================================================
# Normalized Errors (with retry hints and operational guidance)
# =============================================================================

class VectorAdapterError(Exception):
    """
    Base exception for all vector adapter errors.
    
    Provides structured error information including retry guidance, resource limits,
    and operational suggestions for callers to handle failures gracefully.
    
    Attributes:
        message: Human-readable error description
        code: Machine-readable error code (UPPER_SNAKE_CASE where possible)
        retry_after_ms: Suggested delay before retry (None if not retryable)
        resource_scope: Scope of resource limitation ("index", "memory", "compute", etc.)
        suggested_batch_reduction: Percentage reduction suggestion for batch size
        details: Additional context-specific error details (JSON-serializable, SIEM-safe)
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
        """Convert error to dictionary for serialization and logging."""
        return {
            "message": self.message,
            "code": self.code,
            "retry_after_ms": self.retry_after_ms,
            "resource_scope": self.resource_scope,
            "suggested_batch_reduction": self.suggested_batch_reduction,
            "details": {k: self.details[k] for k in sorted(self.details)},
        }

# Subclasses set default `code` in UPPER_SNAKE_CASE where not explicitly provided.

class BadRequest(VectorAdapterError):
    """Client sent an invalid request (malformed vectors, invalid parameters)."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "BAD_REQUEST")
        super().__init__(message, **kwargs)

class AuthError(VectorAdapterError):
    """Authentication or authorization failed (invalid credentials, permissions)."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "AUTH_ERROR")
        super().__init__(message, **kwargs)

class ResourceExhausted(VectorAdapterError):
    """Quota, rate limit, or resource constraints exceeded."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "RESOURCE_EXHAUSTED")
        super().__init__(message, **kwargs)

class DimensionMismatch(VectorAdapterError):
    """Vector dimensions do not match expected schema."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "DIMENSION_MISMATCH")
        super().__init__(message, **kwargs)

class IndexNotReady(VectorAdapterError):
    """Vector index is not built or ready for queries."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "INDEX_NOT_READY")
        super().__init__(message, **kwargs)

class TransientNetwork(VectorAdapterError):
    """Transient network failure that may succeed on retry."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "TRANSIENT_NETWORK")
        super().__init__(message, **kwargs)

class Unavailable(VectorAdapterError):
    """Service is temporarily unavailable or overloaded."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "UNAVAILABLE")
        super().__init__(message, **kwargs)

class NotSupported(VectorAdapterError):
    """Requested operation or parameter is not supported."""
    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "NOT_SUPPORTED")
        super().__init__(message, **kwargs)

class DeadlineExceeded(VectorAdapterError):
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
    Context for vector operations providing tracing, deadlines, and multi-tenant isolation.
    
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
        Non-negative (0 if expired).
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
    All metrics must be low-cardinality and never include PII or raw tenant identifiers.
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
        """
        ...

class NoopMetrics:
    """No-operation metrics sink for testing or when metrics are disabled."""
    def observe(self, **_: Any) -> None: ...
    def counter(self, **_: Any) -> None: ...

# =============================================================================
# Policy & Infra Extension Points (deadline, breaker, cache, limiter)
# =============================================================================

class DeadlinePolicy(Protocol):
    """Strategy to apply time budgets (ctx.deadline_ms) to awaits."""
    async def wrap(self, coro, ctx: Optional[OperationContext]): ...

class CircuitBreaker(Protocol):
    """Minimal circuit breaker interface."""
    def allow(self) -> bool: ...
    def on_success(self) -> None: ...
    def on_error(self, err: Exception) -> None: ...

class Cache(Protocol):
    """Minimal async cache interface."""
    async def get(self, key: str) -> Optional[Any]: ...
    async def set(self, key: str, value: Any, ttl_s: int) -> None: ...

class RateLimiter(Protocol):
    """Minimal rate limiter interface."""
    async def acquire(self) -> None: ...
    def release(self) -> None: ...

# ---- No-op / simple policies ----

class NoopDeadline:
    async def wrap(self, coro, ctx: Optional[OperationContext]):
        return await coro

class SimpleDeadline:
    """
    Enforces ctx.deadline_ms using asyncio.wait_for.
    Maps asyncio.TimeoutError -> DeadlineExceeded.
    """
    async def wrap(self, coro, ctx: Optional[OperationContext]):
        if ctx is None or ctx.deadline_ms is None:
            return await coro
        rem = OperationContext(deadline_ms=ctx.deadline_ms).remaining_ms()
        if rem is not None and rem <= 0:
            raise DeadlineExceeded("operation timed out (preflight)", details={"preflight": True})
        try:
            return await asyncio.wait_for(coro, timeout=(rem / 1000.0 if rem is not None else None))
        except asyncio.TimeoutError:
            raise DeadlineExceeded("operation timed out")

class NoopBreaker:
    def allow(self) -> bool: return True
    def on_success(self) -> None: ...
    def on_error(self, err: Exception) -> None: ...

class SimpleCircuitBreaker:
    """
    Tiny counter-based breaker:
    - Opens after consecutive_failure_threshold errors.
    - Half-opens after reset_timeout_s; first success closes it.
    """
    def __init__(self, consecutive_failure_threshold: int = 5, reset_timeout_s: float = 10.0) -> None:
        self._threshold = max(1, consecutive_failure_threshold)
        self._reset_timeout_s = max(1.0, float(reset_timeout_s))
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._half_open = False

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        elapsed = time.monotonic() - self._opened_at
        if elapsed >= self._reset_timeout_s:
            # half-open probe
            self._half_open = True
            return True
        return False

    def on_success(self) -> None:
        if self._half_open:
            # close on successful probe
            self._opened_at = None
            self._half_open = False
            self._failures = 0
        else:
            self._failures = 0

    def on_error(self, err: Exception) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = time.monotonic()
            self._failures = 0
            self._half_open = False

class NoopCache:
    async def get(self, key: str) -> Optional[Any]: return None
    async def set(self, key: str, value: Any, ttl_s: int) -> None: ...

class InMemoryTTLCache:
    """Very small, in-memory TTL cache (not for large workloads)."""
    def __init__(self) -> None:
        self._store: Dict[str, Tuple[float, Any]] = {}

    async def get(self, key: str) -> Optional[Any]:
        now = time.monotonic()
        item = self._store.get(key)
        if not item:
            return None
        expires_at, value = item
        if now >= expires_at:
            self._store.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: Any, ttl_s: int) -> None:
        ttl_s = max(0, int(ttl_s))
        self._store[key] = (time.monotonic() + ttl_s, value)

class NoopLimiter:
    async def acquire(self) -> None: ...
    def release(self) -> None: ...

class SimpleTokenBucketLimiter:
    """
    Simple token-bucket limiter with second-level refill granularity.
    Fail-open on internal errors to avoid deadlocks.
    """
    def __init__(self, rate_per_sec: int = 50, burst: int = 100) -> None:
        self._capacity = max(1, int(burst))
        self._rate = max(1, int(rate_per_sec))
        self._tokens = self._capacity
        self._last = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        delta = now - self._last
        if delta <= 0:
            return
        add = int(delta * self._rate)
        if add > 0:
            self._tokens = min(self._capacity, self._tokens + add)
            self._last = now

    async def acquire(self) -> None:
        try:
            while True:
                self._refill()
                if self._tokens > 0:
                    self._tokens -= 1
                    return
                # back off in small steps
                await asyncio.sleep(0.02)
        except Exception:
            # fail-open
            return

    def release(self) -> None:
        try:
            self._refill()
            if self._tokens < self._capacity:
                self._tokens += 1
        except Exception:
            # fail-open
            return

# =============================================================================
# Capabilities (dynamic discovery for routing and planning)
# =============================================================================

@dataclass(frozen=True)
class VectorCapabilities:
    """
    Describes the capabilities and limitations of a vector adapter implementation.
    
    Used by routing layers for intelligent database selection, query planning,
    and feature compatibility checking across different vector database backends.
    
    Attributes:
        server: Backend server identifier (e.g., "pinecone", "qdrant", "weaviate")
        version: Backend server version string
        protocol: Protocol identifier (e.g., "vector/v1.0")
        max_dimensions: Maximum vector dimensions supported
        supported_metrics: Supported distance metrics ("cosine", "euclidean", "dotproduct")
        supports_namespaces: Whether namespaces/collections are supported
        supports_metadata_filtering: Whether metadata filtering is supported
        supports_batch_operations: Whether batch upsert/delete are supported
        max_batch_size: Maximum vectors per batch operation
        supports_index_management: Whether index creation/deletion is supported
        idempotent_writes: Whether write operations are idempotent with idempotency_key
        supports_multi_tenant: Whether multi-tenant isolation is supported
        supports_deadline: Whether adapter cooperates with deadline cancellation
        max_top_k: Optional upper bound on top_k per query (None means unspecified)
        max_filter_terms: Optional guideline for filter complexity
    """
    server: str
    version: str
    protocol: str = VECTOR_PROTOCOL_ID
    max_dimensions: int = 0
    supported_metrics: Tuple[str, ...] = ("cosine", "euclidean", "dotproduct")
    supports_namespaces: bool = True
    supports_metadata_filtering: bool = True
    supports_batch_operations: bool = True
    max_batch_size: Optional[int] = None
    supports_index_management: bool = False
    idempotent_writes: bool = False
    supports_multi_tenant: bool = False
    supports_deadline: bool = True
    max_top_k: Optional[int] = None
    max_filter_terms: Optional[int] = None

# =============================================================================
# Query and Operation Specifications
# =============================================================================

@dataclass(frozen=True)
class QuerySpec:
    """
    Specification for vector similarity search queries.
    
    Attributes:
        vector: Query vector for similarity search
        top_k: Number of top results to return
        namespace: Target namespace/collection for the query
        filter: Optional metadata filters for pre-search filtering
        include_metadata: Whether to include metadata in results
        include_vectors: Whether to include vector data in results
    """
    vector: List[float]
    top_k: int
    namespace: str = "default"
    filter: Optional[Dict[str, Any]] = None
    include_metadata: bool = True
    include_vectors: bool = False

@dataclass(frozen=True)
class UpsertSpec:
    """
    Specification for vector upsert operations.
    
    Attributes:
        vectors: List of vectors to upsert
        namespace: Target namespace/collection for the operation
    """
    vectors: List[Vector]
    namespace: str = "default"

@dataclass(frozen=True)
class DeleteSpec:
    """
    Specification for vector deletion operations.
    
    Attributes:
        ids: List of vector IDs to delete
        namespace: Target namespace/collection for the operation
        filter: Optional metadata filter for bulk deletion
    """
    ids: List[VectorID]
    namespace: str = "default"
    filter: Optional[Dict[str, Any]] = None

@dataclass(frozen=True)
class NamespaceSpec:
    """
    Specification for namespace/collection management operations.
    
    Attributes:
        namespace: Namespace/collection name
        dimensions: Vector dimensions for this namespace
        distance_metric: Distance metric to use ("cosine", "euclidean", "dotproduct")
    """
    namespace: str
    dimensions: int
    distance_metric: str = "cosine"

# =============================================================================
# Operation Results
# =============================================================================

@dataclass
class UpsertResult:
    """
    Result from vector upsert operations.
    
    Attributes:
        upserted_count: Number of vectors successfully upserted
        failed_count: Number of vectors that failed to upsert
        failures: List of individual failure details
    """
    upserted_count: int
    failed_count: int
    failures: List[Dict[str, Any]]

@dataclass
class DeleteResult:
    """
    Result from vector deletion operations.
    
    Attributes:
        deleted_count: Number of vectors successfully deleted
        failed_count: Number of vectors that failed to delete
        failures: List of individual failure details
    """
    deleted_count: int
    failed_count: int
    failures: List[Dict[str, Any]]

@dataclass
class NamespaceResult:
    """
    Result from namespace management operations.
    
    Attributes:
        success: Whether the operation completed successfully
        namespace: The namespace that was operated on
        details: Additional operation-specific details
    """
    success: bool
    namespace: str
    details: Dict[str, Any]

# =============================================================================
# Stable Protocol Interface (async, versioned contract)
# =============================================================================

@runtime_checkable
class VectorProtocolV1(Protocol):
    """
    Protocol defining the Vector Protocol V1.0 interface.
    
    Implement this protocol to create compatible vector adapters. All methods are async
    and designed for high-concurrency environments. This protocol is language-level,
    while the canonical wire contract is defined by the JSON envelopes.
    """

    async def capabilities(self) -> VectorCapabilities: ...

    async def query(
        self,
        spec: QuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult: ...

    async def upsert(
        self,
        spec: UpsertSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult: ...

    async def delete(
        self,
        spec: DeleteSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult: ...

    async def create_namespace(
        self,
        spec: NamespaceSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult: ...

    async def delete_namespace(
        self,
        namespace: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult: ...

    async def health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]: ...

# =============================================================================
# Base Instrumented Adapter (validation, metrics, error handling)
# =============================================================================

class BaseVectorAdapter(VectorProtocolV1):
    """
    Base class for implementing Vector Protocol V1.0 adapters.
    
    Provides common validation, metrics instrumentation, error handling, and
    SIEM-safe observability. Implementers should override the `_do_*` methods
    to provide backend-specific functionality while getting production-ready
    infrastructure for free.
    
    Example:
        class PineconeAdapter(BaseVectorAdapter):
            async def _do_query(self, spec: QuerySpec, *, ctx: Optional[OperationContext]) -> QueryResult:
                # Pinecone-specific implementation
                ...
    """

    _component = "vector"

    def __init__(
        self,
        *,
        metrics: Optional[MetricsSink] = None,
        mode: str = "thin",
        # Optional explicit policy overrides (advanced)
        deadline_policy: Optional[DeadlinePolicy] = None,
        breaker: Optional[CircuitBreaker] = None,
        cache: Optional[Cache] = None,
        limiter: Optional[RateLimiter] = None,
        cache_query_ttl_s: int = 60,
        cache_caps_ttl_s: int = 30,
        warn_on_standalone_no_metrics: bool = True,
    ) -> None:
        """
        Initialize the vector adapter with metrics instrumentation and optional policies.
        """
        self._metrics: MetricsSink = metrics or NoopMetrics()

        # Resolve policies by mode; explicit overrides always win.
        mode = (mode or "thin").strip().lower()
        if mode not in ("thin", "standalone"):
            mode = "thin"

        if mode == "thin":
            self._deadline: DeadlinePolicy = deadline_policy or NoopDeadline()
            self._breaker: CircuitBreaker = breaker or NoopBreaker()
            self._cache: Cache = cache or NoopCache()
            self._limiter: RateLimiter = limiter or NoopLimiter()
        else:
            self._deadline = deadline_policy or SimpleDeadline()
            self._breaker = breaker or SimpleCircuitBreaker()
            self._cache = cache or InMemoryTTLCache()
            self._limiter = limiter or SimpleTokenBucketLimiter()
            if warn_on_standalone_no_metrics and isinstance(self._metrics, NoopMetrics):
                LOG.warning(
                    "Using standalone mode without metrics — provide a MetricsSink for production use"
                )

        self._mode = mode
        self._cache_query_ttl_s = int(max(1, cache_query_ttl_s))
        self._cache_caps_ttl_s = int(max(1, cache_caps_ttl_s))

    # --- internal helpers (validation and instrumentation) ---

    @staticmethod
    def _require_non_empty(name: str, value: str) -> None:
        """
        Validate that a string value is non-empty.
        """
        if not isinstance(value, str) or not value.strip():
            raise BadRequest(f"{name} must be a non-empty string")

    @staticmethod
    def _validate_vector(vector: List[float]) -> None:
        """
        Validate that a vector is properly formed.
        """
        if not vector or not isinstance(vector, list):
            raise BadRequest("vector must be a non-empty list of floats")
        if not all(isinstance(x, (int, float)) for x in vector):
            raise BadRequest("vector must contain only numeric values")

    @staticmethod
    def _tenant_hash(tenant: Optional[str]) -> Optional[str]:
        """
        Create privacy-preserving hash of tenant identifier for metrics.
        """
        if not tenant: 
            return None
        return hashlib.sha256(tenant.encode()).hexdigest()[:12]

    def _record(
        self, 
        op: str, 
        t0: float, 
        ok: bool, 
        *, 
        code: str = "OK", 
        ctx: Optional[OperationContext] = None, 
        **extra: Any
    ) -> None:
        """
        Record operation metrics with context and tenant hashing.
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
                    if rem < 1000: x["deadline_bucket"] = "<1s"
                    elif rem < 5000: x["deadline_bucket"] = "<5s"
                    elif rem < 15000: x["deadline_bucket"] = "<15s"
                    elif rem < 60000: x["deadline_bucket"] = "<60s"
                    else: x["deadline_bucket"] = ">=60s"
            self._metrics.observe(
                component=self._component, 
                op=op, 
                ms=ms, 
                ok=ok, 
                code=code, 
                extra=x or None
            )
        except Exception:
            # Never let metrics recording break the operation
            pass

    async def _apply_deadline(self, coro, ctx: Optional[OperationContext]):
        """
        Apply the configured deadline policy to awaitable; map timeouts to DeadlineExceeded.
        """
        try:
            return await self._deadline.wrap(coro, ctx)
        except DeadlineExceeded:
            raise
        except asyncio.TimeoutError:
            raise DeadlineExceeded("operation timed out")

    def _fail_if_expired(self, ctx: Optional[OperationContext]) -> None:
        """
        Fail fast if ctx.deadline_ms is already expired.
        """
        if ctx is None or ctx.deadline_ms is None:
            return
        if ctx.remaining_ms() == 0:
            raise DeadlineExceeded("operation timed out (preflight)", details={"preflight": True})

    # --- cache keys (read paths only) ---

    @staticmethod
    def _hash_obj(obj: Any) -> str:
        raw = repr(obj).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _query_cache_key(self, spec: QuerySpec, caps: Optional[VectorCapabilities], ctx: Optional[OperationContext]) -> str:
        """
        Compose a cache key for query() that avoids cross-hit pollution.
        Includes vector hash, namespace, top_k, filter, flags, protocol/backend info, and tenant hash.
        """
        tenant_h = self._tenant_hash(ctx.tenant) if ctx else None
        caps_part = f"{caps.server}:{caps.version}" if caps else "unknown"
        return (
            f"v1:query:"
            f"{caps_part}:"
            f"ns={spec.namespace}:"
            f"topk={spec.top_k}:"
            f"im={int(bool(spec.include_metadata))}:iv={int(bool(spec.include_vectors))}:"
            f"vec={self._hash_obj(spec.vector)}:"
            f"flt={self._hash_obj(spec.filter)}:"
            f"tenant={tenant_h}"
        )

    @staticmethod
    def _caps_cache_key() -> str:
        return "v1:capabilities"

    # --- final public APIs (validation + instrumentation) ---

    async def capabilities(self) -> VectorCapabilities:
        """Get the capabilities of this vector adapter."""
        t0 = time.monotonic()
        try:
            caps: Optional[VectorCapabilities] = None
            # standalone: try cache
            if self._mode == "standalone":
                cached = await self._cache.get(self._caps_cache_key())
                if cached:
                    self._metrics.counter(component=self._component, name="cache_hits", value=1, extra={"op": "capabilities"})
                    self._record("capabilities", t0, True)
                    return cached

            caps = await self._apply_deadline(self._do_capabilities(), ctx=None)
            # standalone: set cache
            if self._mode == "standalone":
                await self._cache.set(self._caps_cache_key(), caps, ttl_s=self._cache_caps_ttl_s)

            self._record("capabilities", t0, True)
            return caps
        except VectorAdapterError as e:
            self._record("capabilities", t0, False, code=e.code or type(e).__name__)
            raise
        except Exception as e:
            self._record("capabilities", t0, False, code="UNAVAILABLE")
            # Normalize unexpected exceptions as Unavailable for callers
            raise Unavailable("capabilities fetch failed") from e

    async def query(
        self,
        spec: QuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        """
        Execute a vector similarity search query with validation and metrics.
        
        See VectorProtocolV1.query for full documentation.
        """
        # Preflight validation
        self._validate_vector(spec.vector)
        self._require_non_empty("namespace", spec.namespace)
        if not isinstance(spec.top_k, int) or spec.top_k <= 0:
            raise BadRequest("top_k must be a positive integer")
        if spec.filter is not None and not isinstance(spec.filter, Mapping):
            raise BadRequest("filter must be a mapping (dict) when provided")
        if not isinstance(spec.include_metadata, bool) or not isinstance(spec.include_vectors, bool):
            raise BadRequest("include_metadata/include_vectors must be booleans")

        # Deadline preflight
        self._fail_if_expired(ctx)

        # Breaker & limiter gates
        if not self._breaker.allow():
            raise Unavailable("circuit open")
        await self._limiter.acquire()

        t0 = time.monotonic()
        try:
            # Capability gating
            caps = await self.capabilities()
            if caps.max_dimensions and len(spec.vector) > int(caps.max_dimensions):
                raise DimensionMismatch(
                    f"vector dimension {len(spec.vector)} exceeds max {caps.max_dimensions}",
                    details={"provided": len(spec.vector), "max": int(caps.max_dimensions)}
                )
            if caps.max_top_k is not None and spec.top_k > caps.max_top_k:
                raise BadRequest(
                    f"top_k {spec.top_k} exceeds maximum of {caps.max_top_k}",
                    details={"max_top_k": caps.max_top_k}
                )
            if spec.filter and not caps.supports_metadata_filtering:
                raise NotSupported("metadata filtering is not supported by this adapter")

            # Read-path cache (standalone only)
            if self._mode == "standalone":
                ck = self._query_cache_key(spec, caps, ctx)
                cached = await self._cache.get(ck)
                if cached:
                    self._metrics.counter(component=self._component, name="cache_hits", value=1, extra={"op": "query"})
                    self._record("query", t0, True, ctx=ctx, namespace=spec.namespace, top_k=spec.top_k, cached=1, matches=len(cached.matches))
                    self._breaker.on_success()
                    self._limiter.release()
                    return cached

            # Execute with deadline
            result = await self._apply_deadline(self._do_query(spec, ctx=ctx), ctx)

            # Cache the result if eligible
            if self._mode == "standalone":
                ck = self._query_cache_key(spec, caps, ctx)
                await self._cache.set(ck, result, ttl_s=self._cache_query_ttl_s)

            # Metrics
            self._record(
                "query", t0, True, ctx=ctx,
                namespace=spec.namespace, top_k=spec.top_k, matches=len(result.matches)
            )
            self._metrics.counter(component=self._component, name="queries", value=1)
            self._breaker.on_success()
            return result

        except VectorAdapterError as e:
            self._record("query", t0, False, code=e.code or type(e).__name__, ctx=ctx, namespace=spec.namespace, top_k=spec.top_k)
            self._breaker.on_error(e)
            raise
        except Exception as e:
            self._record("query", t0, False, code="UNAVAILABLE", ctx=ctx, namespace=spec.namespace, top_k=spec.top_k)
            self._breaker.on_error(e)
            raise
        finally:
            self._limiter.release()

    async def upsert(
        self,
        spec: UpsertSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        """
        Upsert vectors into the vector store with validation and metrics.
        
        See VectorProtocolV1.upsert for full documentation.
        """
        self._require_non_empty("namespace", spec.namespace)
        if not spec.vectors:
            raise BadRequest("vectors must not be empty")
        for vector in spec.vectors:
            self._validate_vector(vector.vector)

        # Deadline preflight
        self._fail_if_expired(ctx)

        # Breaker & limiter gates
        if not self._breaker.allow():
            raise Unavailable("circuit open")
        await self._limiter.acquire()

        t0 = time.monotonic()
        try:
            # Capability gating
            caps = await self.capabilities()
            if caps.max_batch_size is not None and len(spec.vectors) > caps.max_batch_size:
                raise BadRequest(
                    f"batch size {len(spec.vectors)} exceeds maximum of {caps.max_batch_size}",
                    details={"max_batch_size": caps.max_batch_size},
                    suggested_batch_reduction=int(100 * (len(spec.vectors) - caps.max_batch_size) / len(spec.vectors)) if len(spec.vectors) else None,
                )
            if caps.max_dimensions:
                for v in spec.vectors:
                    if len(v.vector) > caps.max_dimensions:
                        raise DimensionMismatch(
                            f"vector dimension {len(v.vector)} exceeds max {caps.max_dimensions}",
                            details={"provided": len(v.vector), "max": int(caps.max_dimensions)}
                        )

            # Execute with deadline
            result = await self._apply_deadline(self._do_upsert(spec, ctx=ctx), ctx)

            # Metrics
            self._record("upsert", t0, True, ctx=ctx, namespace=spec.namespace, vectors_processed=len(spec.vectors))
            self._metrics.counter(component=self._component, name="vectors_upserted", value=int(result.upserted_count))
            self._breaker.on_success()
            return result

        except VectorAdapterError as e:
            self._record("upsert", t0, False, code=e.code or type(e).__name__, ctx=ctx, namespace=spec.namespace)
            self._breaker.on_error(e)
            raise
        except Exception as e:
            self._record("upsert", t0, False, code="UNAVAILABLE", ctx=ctx, namespace=spec.namespace)
            self._breaker.on_error(e)
            raise
        finally:
            self._limiter.release()

    async def delete(
        self,
        spec: DeleteSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        """
        Delete vectors from the vector store with validation and metrics.
        
        See VectorProtocolV1.delete for full documentation.
        """
        self._require_non_empty("namespace", spec.namespace)
        if not spec.ids and not spec.filter:
            raise BadRequest("must provide either ids or filter for deletion")

        # Deadline preflight
        self._fail_if_expired(ctx)

        # Breaker & limiter gates
        if not self._breaker.allow():
            raise Unavailable("circuit open")
        await self._limiter.acquire()

        t0 = time.monotonic()
        try:
            # Capability gating
            caps = await self.capabilities()
            if caps.max_batch_size is not None and spec.ids and len(spec.ids) > caps.max_batch_size:
                raise BadRequest(
                    f"batch size {len(spec.ids)} exceeds maximum of {caps.max_batch_size}",
                    details={"max_batch_size": caps.max_batch_size},
                    suggested_batch_reduction=int(100 * (len(spec.ids) - caps.max_batch_size) / len(spec.ids)) if len(spec.ids) else None,
                )
            if spec.filter and not caps.supports_metadata_filtering:
                raise NotSupported("metadata filtering is not supported by this adapter")

            # Execute with deadline
            result = await self._apply_deadline(self._do_delete(spec, ctx=ctx), ctx)

            # Metrics
            self._record(
                "delete", t0, True, ctx=ctx, namespace=spec.namespace,
                vectors_targeted=len(spec.ids) if spec.ids else 0
            )
            self._metrics.counter(component=self._component, name="vectors_deleted", value=int(result.deleted_count))
            self._breaker.on_success()
            return result

        except VectorAdapterError as e:
            self._record("delete", t0, False, code=e.code or type(e).__name__, ctx=ctx, namespace=spec.namespace)
            self._breaker.on_error(e)
            raise
        except Exception as e:
            self._record("delete", t0, False, code="UNAVAILABLE", ctx=ctx, namespace=spec.namespace)
            self._breaker.on_error(e)
            raise
        finally:
            self._limiter.release()

    async def create_namespace(
        self,
        spec: NamespaceSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        """
        Create a new namespace/collection with validation and metrics.
        
        See VectorProtocolV1.create_namespace for full documentation.
        """
        self._require_non_empty("namespace", spec.namespace)
        if spec.dimensions <= 0:
            raise BadRequest("dimensions must be positive")
        if spec.distance_metric not in ("cosine", "euclidean", "dotproduct"):
            raise BadRequest("distance_metric must be one of: cosine, euclidean, dotproduct")

        # Deadline preflight
        self._fail_if_expired(ctx)

        # Breaker & limiter gates
        if not self._breaker.allow():
            raise Unavailable("circuit open")
        await self._limiter.acquire()

        t0 = time.monotonic()
        try:
            # Capability gating
            caps = await self.capabilities()
            if caps.max_dimensions and spec.dimensions > caps.max_dimensions:
                raise BadRequest(
                    f"dimensions {spec.dimensions} exceed maximum of {caps.max_dimensions}",
                    details={"max_dimensions": caps.max_dimensions}
                )
            if spec.distance_metric not in caps.supported_metrics:
                raise NotSupported(
                    f"distance_metric '{spec.distance_metric}' not supported",
                    details={"supported_metrics": caps.supported_metrics}
                )

            # Execute with deadline
            result = await self._apply_deadline(self._do_create_namespace(spec, ctx=ctx), ctx)

            self._record("create_namespace", t0, True, ctx=ctx, namespace=spec.namespace)
            self._breaker.on_success()
            return result

        except VectorAdapterError as e:
            self._record("create_namespace", t0, False, code=e.code or type(e).__name__, ctx=ctx, namespace=spec.namespace)
            self._breaker.on_error(e)
            raise
        except Exception as e:
            self._record("create_namespace", t0, False, code="UNAVAILABLE", ctx=ctx, namespace=spec.namespace)
            self._breaker.on_error(e)
            raise
        finally:
            self._limiter.release()

    async def delete_namespace(
        self,
        namespace: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        """
        Delete a namespace/collection with validation and metrics.
        
        See VectorProtocolV1.delete_namespace for full documentation.
        """
        self._require_non_empty("namespace", namespace)

        # Deadline preflight
        self._fail_if_expired(ctx)

        # Breaker & limiter gates
        if not self._breaker.allow():
            raise Unavailable("circuit open")
        await self._limiter.acquire()

        t0 = time.monotonic()
        try:
            result = await self._apply_deadline(self._do_delete_namespace(namespace, ctx=ctx), ctx)
            self._record("delete_namespace", t0, True, ctx=ctx, namespace=namespace)
            self._breaker.on_success()
            return result

        except VectorAdapterError as e:
            self._record("delete_namespace", t0, False, code=e.code or type(e).__name__, ctx=ctx, namespace=namespace)
            self._breaker.on_error(e)
            raise
        except Exception as e:
            self._record("delete_namespace", t0, False, code="UNAVAILABLE", ctx=ctx, namespace=namespace)
            self._breaker.on_error(e)
            raise
        finally:
            self._limiter.release()

    async def health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """Check health status with metrics instrumentation."""
        # Deadline preflight
        self._fail_if_expired(ctx)

        t0 = time.monotonic()
        try:
            h = await self._apply_deadline(self._do_health(ctx=ctx), ctx)
            self._record("health", t0, True, ctx=ctx)
            return {
                "ok": bool(h.get("ok", True)),
                "server": str(h.get("server", "")),
                "version": str(h.get("version", "")),
                "namespaces": h.get("namespaces", {}),
            }
        except VectorAdapterError as e:
            self._record("health", t0, False, code=e.code or type(e).__name__, ctx=ctx)
            raise
        except asyncio.TimeoutError:
            self._record("health", t0, False, code="DEADLINE_EXCEEDED", ctx=ctx)
            raise DeadlineExceeded("operation timed out")
        except Exception as e:
            self._record("health", t0, False, code="UNAVAILABLE", ctx=ctx)
            raise Unavailable("health check failed") from e

    # --- hooks to implement per backend (override these) ---

    async def _do_capabilities(self) -> VectorCapabilities:
        """Implement to return adapter-specific capabilities."""
        raise NotImplementedError

    async def _do_query(
        self,
        spec: QuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        """Implement vector similarity search with validated inputs."""
        raise NotImplementedError

    async def _do_upsert(
        self,
        spec: UpsertSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        """Implement vector upsert operations with validated inputs."""
        raise NotImplementedError

    async def _do_delete(
        self,
        spec: DeleteSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        """Implement vector deletion operations with validated inputs."""
        raise NotImplementedError

    async def _do_create_namespace(
        self,
        spec: NamespaceSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        """Implement namespace creation with validated inputs."""
        raise NotImplementedError

    async def _do_delete_namespace(
        self,
        namespace: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        """Implement namespace deletion."""
        raise NotImplementedError

    async def _do_health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """Implement health check for the vector backend."""
        raise NotImplementedError

# =============================================================================
# Wire-Level Helpers (canonical envelopes)
# =============================================================================

def _ctx_from_wire(ctx_dict: Mapping[str, Any]) -> OperationContext:
    """
    Convert a wire-level ctx dict into an OperationContext.
    Unknown keys are ignored, per protocol rules.
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
    Map VectorAdapterError (or unexpected Exception) to canonical error envelope.
    """
    if isinstance(e, VectorAdapterError):
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
    # Fallback: treat as UNAVAILABLE/INTERNAL
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
    Map typed result objects to canonical success envelope.
    Uses dataclasses.asdict() where applicable, else passes through.
    """
    if hasattr(result, "__dataclass_fields__"):
        result_payload = asdict(result)
    else:
        result_payload = result
    return {
        "ok": True,
        "code": "OK",
        "ms": ms,
        "result": result_payload,
    }

class WireVectorHandler:
    """
    Thin wire-level adapter that exposes a VectorProtocolV1 implementation using
    the canonical JSON envelope contract:

        { "op": "vector.query", "ctx": {...}, "args": {...} } -> { ... }

    This makes the protocol code-agnostic and transport-agnostic: you can plug
    this into HTTP, gRPC, WebSockets, etc.
    """

    def __init__(self, adapter: VectorProtocolV1):
        self._adapter = adapter

    async def handle(self, envelope: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Handle a single request envelope and return a response envelope.

        Expects:
            op: "vector.<operation>"
            ctx: { ... }  (optional)
            args: { ... } (operation-specific)
        """
        t0 = time.monotonic()
        try:
            op = envelope.get("op")
            if not isinstance(op, str):
                raise BadRequest("missing or invalid 'op'")

            ctx = _ctx_from_wire(envelope.get("ctx") or {})
            args = envelope.get("args") or {}

            if op == "vector.capabilities":
                res = await self._adapter.capabilities()
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "vector.query":
                spec = QuerySpec(**args)
                res = await self._adapter.query(spec, ctx=ctx)
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "vector.upsert":
                vectors = [Vector(**v) for v in args.get("vectors", [])]
                spec = UpsertSpec(vectors=vectors, namespace=args.get("namespace", "default"))
                res = await self._adapter.upsert(spec, ctx=ctx)
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "vector.delete":
                ids = [VectorID(v) for v in args.get("ids", [])]
                spec = DeleteSpec(ids=ids, namespace=args.get("namespace", "default"), filter=args.get("filter"))
                res = await self._adapter.delete(spec, ctx=ctx)
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "vector.create_namespace":
                spec = NamespaceSpec(**args)
                res = await self._adapter.create_namespace(spec, ctx=ctx)
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "vector.delete_namespace":
                namespace = args.get("namespace")
                if not isinstance(namespace, str):
                    raise BadRequest("namespace must be provided as a string")
                res = await self._adapter.delete_namespace(namespace, ctx=ctx)
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "vector.health":
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
    "VECTOR_PROTOCOL_VERSION",
    "VECTOR_PROTOCOL_ID",
    "VectorID",
    "Vector",
    "QueryResult", 
    "VectorMatch",
    "VectorAdapterError",
    "BadRequest",
    "AuthError",
    "ResourceExhausted",
    "DimensionMismatch",
    "IndexNotReady",
    "TransientNetwork",
    "Unavailable",
    "NotSupported",
    "DeadlineExceeded",
    "OperationContext",
    "QuerySpec",
    "UpsertSpec",
    "DeleteSpec",
    "NamespaceSpec",
    "UpsertResult",
    "DeleteResult",
    "NamespaceResult",
    "VectorCapabilities",
    "VectorProtocolV1",
    "BaseVectorAdapter",
    # policy & infra extension points
    "DeadlinePolicy",
    "CircuitBreaker",
    "Cache",
    "RateLimiter",
    "NoopDeadline",
    "SimpleDeadline",
    "NoopBreaker",
    "SimpleCircuitBreaker",
    "NoopCache",
    "InMemoryTTLCache",
    "NoopLimiter",
    "SimpleTokenBucketLimiter",
    # wire helpers
    "WireVectorHandler",
    "_ctx_from_wire",
    "_error_to_wire",
    "_success_to_wire",
]
