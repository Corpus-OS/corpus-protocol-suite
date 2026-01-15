# corpus_sdk/vector/vector_base.py
# SPDX-License-Identifier: Apache-2.0
"""
Purpose
-------
A stable, vendor-neutral API for vector similarity search and operations — with
structured errors, caching strategies, rate limiting, and production observability.

This module serves as the reference implementation and SDK mapping for the
Vector Protocol V1.0. It defines:

- Typed Python contracts mirroring the protocol data shapes
- A production-ready BaseVectorAdapter with validation, metrics, and backpressure
- A thin WireVectorHandler that converts wire envelopes ⇄ typed API
- First-class text storage support via DocStore integration
- Optional batch query operations for performance optimization
- Automatic vector normalization for cosine similarity optimization

Design Philosophy
-----------------
- Minimal surface area: Core vector operations only, no vendor-specific extensions
- Async-first: All operations are non-blocking for high-concurrency environments
- Production hardened: Built-in caching, circuit breaking, backpressure, and metrics
- Extensible: Capability discovery allows for database-specific vector features
- Performance-optimized: Read-path caching for similarity queries (standalone mode)
- Wire-first: Types and helpers exist to faithfully implement the canonical JSON contract
- Text-aware: First-class support for source text storage and retrieval
- Batch-friendly: Optional batch operations for efficiency where supported

Deliberate Non-Goals
--------------------
- No embedding model management or text-to-vector transformations
- No vector index tuning or optimization strategies
- No provider-specific algorithms beyond capabilities
- No client-side result re-ranking or post-processing

Those behaviors live in embedding services and upper application layers.

Text Storage Strategy
---------------------
The protocol supports three text storage strategies:

1. "metadata" (default): Store text in vector metadata (simple but expensive for large texts)
2. "docstore": Store text in separate document store (cost-optimized for production)
3. "none": Text storage not supported

When using docstore, text is automatically stored/retrieved during upsert/query operations.
Docstore failures during upsert cause the entire operation to fail (atomicity).
Docstore failures during query cause text to be missing but don't fail the query (graceful degradation).

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

Auto-Normalization Feature
--------------------------
When enabled (auto_normalize=True), vectors are automatically normalized to unit
length (L2 norm) during upsert and query operations. This is particularly useful
for cosine similarity searches where normalized vectors provide more consistent
results. The feature includes optimizations to avoid unnecessary normalization
when vectors are already approximately unit length.

Threading & Processes
---------------------
- In-memory components (cache, breaker, limiter) are **per-process only**.
- They are **not thread-safe** and are intended for a single-threaded async
  event loop. For multi-threaded or multi-process deployments, use external,
  distributed implementations of cache / limiter / breaker instead.

Versioning
----------
Follow SemVer against VECTOR_PROTOCOL_VERSION. Minor versions are strictly additive.
- Patch (x.y.Z): Editorial clarifications, non-breaking fixes
- Minor (x.Y.z): New optional parameters, capabilities, or methods
- Major (X.y.z): Breaking changes to signatures or behavior

Wire Contract (Canonical Interface)
-----------------------------------
The canonical interoperability surface for this protocol is the JSON wire envelope.
This module defines a code-level interface (VectorProtocolV1 / BaseVectorAdapter)
plus a thin wire adapter (WireVectorHandler) that maps envelopes ⇄ typed methods.

All requests MUST use the following envelope shape (ctx and args are REQUIRED):

    {
        "op": "vector.<operation>",
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

The WireVectorHandler in this file is the reference adapter for this contract and is
intentionally transport-agnostic (HTTP, gRPC, WebSocket, etc.).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, asdict
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    NewType,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

VECTOR_PROTOCOL_VERSION = "1.0.0"
VECTOR_PROTOCOL_ID = "vector/v1.0"
LOG = logging.getLogger(__name__)

# =============================================================================
# Core Type Definitions
# =============================================================================

VectorID = NewType("VectorID", str)
"""Type alias for vector identifiers providing explicit type safety."""


@dataclass(frozen=True)
class Vector:
    """
    A vector with optional metadata, identifier, and source text.

    Attributes:
        id: Unique identifier for the vector
        vector: The vector embedding as a list of floats
        metadata: Optional key-value pairs for filtering and retrieval
        namespace: Optional namespace/collection for multi-tenant isolation
        text: Optional source text that generated this vector
    """
    id: VectorID
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None
    namespace: Optional[str] = None
    text: Optional[str] = None


@dataclass(frozen=True)
class VectorMatch:
    """
    A single vector match with similarity score and distance.

    Attributes:
        vector: The matching vector
        score: Similarity score (higher = more similar)
        distance: Raw distance metric value (lower = more similar)
    """
    vector: Vector
    score: float
    distance: float


@dataclass
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
# Document Storage (for text associated with vectors)
# =============================================================================

@dataclass(frozen=True)
class Document:
    """Document with text content and metadata.

    Note:
        Metadata stored here is usually a *copy* of vector metadata and may
        diverge over time. The vector backend remains the source of truth for
        vector-level metadata; the docstore metadata is primarily for
        doc-centric access patterns or convenience.
    """
    id: str
    text: str
    metadata: Dict[str, Any]


class DocStore(Protocol):
    """
    Document storage interface for vector source text.

    Used when vectors have associated text that should not be stored
    in vector database metadata (for cost/performance reasons).
    """

    async def get(self, doc_id: str) -> Optional[Document]:
        """Retrieve document by ID"""
        ...

    async def put(self, doc: Document) -> None:
        """Store document"""
        ...

    async def batch_get(self, doc_ids: List[str]) -> Dict[str, Document]:
        """Retrieve multiple documents (default: calls get() in loop)"""
        ...

    async def batch_put(self, docs: List[Document]) -> None:
        """Store multiple documents (default: calls put() in loop)"""
        ...

    async def delete(self, doc_id: str) -> None:
        """Delete document"""
        ...

    async def batch_delete(self, doc_ids: List[str]) -> None:
        """Delete multiple documents (required implementation)"""
        ...


class InMemoryDocStore:
    """In-memory document store for testing and development"""

    def __init__(self):
        self._store: Dict[str, Document] = {}

    async def get(self, doc_id: str) -> Optional[Document]:
        return self._store.get(doc_id)

    async def put(self, doc: Document) -> None:
        self._store[doc.id] = doc

    async def batch_get(self, doc_ids: List[str]) -> Dict[str, Document]:
        return {doc_id: doc for doc_id in doc_ids if (doc := self._store.get(doc_id))}

    async def batch_put(self, docs: List[Document]) -> None:
        for doc in docs:
            self._store[doc.id] = doc

    async def delete(self, doc_id: str) -> None:
        self._store.pop(doc_id, None)

    async def batch_delete(self, doc_ids: List[str]) -> None:
        """Required batch delete implementation."""
        for doc_id in doc_ids:
            self._store.pop(doc_id, None)


class RedisDocStore:
    """Redis-backed document store for production.

    Note:
        Metadata stored here mirrors vector metadata at write time but is not
        automatically kept in sync with later vector metadata changes.
    """

    def __init__(self, redis_client, key_prefix: str = "corpus:doc:"):
        self._redis = redis_client
        self._prefix = key_prefix

    def _key(self, doc_id: str) -> str:
        return f"{self._prefix}{doc_id}"

    async def get(self, doc_id: str) -> Optional[Document]:
        import msgpack
        raw = await self._redis.get(self._key(doc_id))
        if raw is None:
            return None
        data = msgpack.unpackb(raw, raw=False)
        return Document(
            id=data["id"],
            text=data["text"],
            metadata=data.get("metadata", {})
        )

    async def put(self, doc: Document) -> None:
        import msgpack
        data = {"id": doc.id, "text": doc.text, "metadata": doc.metadata}
        await self._redis.set(
            self._key(doc.id),
            msgpack.packb(data),
            ex=86400 * 30  # 30 day TTL
        )

    async def batch_get(self, doc_ids: List[str]) -> Dict[str, Document]:
        import msgpack
        if not doc_ids:
            return {}
        keys = [self._key(doc_id) for doc_id in doc_ids]
        values = await self._redis.mget(keys)
        results = {}
        for doc_id, raw in zip(doc_ids, values):
            if raw is not None:
                data = msgpack.unpackb(raw, raw=False)
                results[doc_id] = Document(
                    id=data["id"],
                    text=data["text"],
                    metadata=data.get("metadata", {})
                )
        return results

    async def batch_put(self, docs: List[Document]) -> None:
        import msgpack
        if not docs:
            return
        pipe = self._redis.pipeline()
        for doc in docs:
            data = {"id": doc.id, "text": doc.text, "metadata": doc.metadata}
            pipe.set(self._key(doc.id), msgpack.packb(data), ex=86400 * 30)
        await pipe.execute()

    async def delete(self, doc_id: str) -> None:
        await self._redis.delete(self._key(doc_id))

    async def batch_delete(self, doc_ids: List[str]) -> None:
        """Required batch delete implementation."""
        if not doc_ids:
            return
        keys = [self._key(doc_id) for doc_id in doc_ids]
        await self._redis.delete(*keys)


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
    attrs: Optional[Mapping[str, Any]] = None

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
        """Record operation timing and status."""
        ...

    def counter(
        self,
        *,
        component: str,
        name: str,
        value: int = 1,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Increment a counter metric."""
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
    async def wrap(self, awaitable: Awaitable[Any], ctx: Optional[OperationContext]) -> Any: ...


class CircuitBreaker(Protocol):
    """Minimal circuit breaker interface."""
    def allow(self) -> bool: ...
    def on_success(self) -> None: ...
    def on_error(self, err: Exception) -> None: ...


class Cache(Protocol):
    """
    Minimal async cache interface.

    Implementations MAY store arbitrary Python objects. The in-memory default
    simply holds references and does not serialize. If you plug in a distributed
    cache (e.g., Redis), you are responsible for serializing/deserializing values
    (JSON, msgpack, etc.) safely.

    Note: cache implementations MAY optionally expose an async
    `invalidate_namespace(namespace: str)` method. If present, BaseVectorAdapter
    will call it for namespace-scoped invalidation instead of relying on TTL.
    """
    async def get(self, key: str) -> Optional[Any]: ...
    async def set(self, key: str, value: Any, ttl_s: int) -> None: ...


class RateLimiter(Protocol):
    """Minimal rate limiter interface."""
    async def acquire(self) -> None: ...
    def release(self) -> None: ...


class NoopDeadline:
    """No-op deadline policy (no timing/timeout behavior)."""
    async def wrap(self, awaitable: Awaitable[Any], ctx: Optional[OperationContext]) -> Any:
        return await awaitable


class SimpleDeadline:
    """
    Enforces ctx.deadline_ms using asyncio.wait_for.

    Maps asyncio.TimeoutError → DeadlineExceeded.
    """
    async def wrap(self, awaitable: Awaitable[Any], ctx: Optional[OperationContext]) -> Any:
        if ctx is None or ctx.deadline_ms is None:
            return await awaitable
        tmp = OperationContext(deadline_ms=ctx.deadline_ms)
        rem = tmp.remaining_ms()
        if rem is not None and rem <= 0:
            # Align with LLM adapter semantics: explicit "deadline already exceeded"
            raise DeadlineExceeded(
                "deadline already exceeded",
                details={"remaining_ms": 0, "preflight": True},
            )
        try:
            return await asyncio.wait_for(
                awaitable,
                timeout=(rem / 1000.0 if rem is not None else None),
            )
        except asyncio.TimeoutError:
            raise DeadlineExceeded(
                "operation timed out",
                details={"remaining_ms": 0},
            )


class NoopBreaker:
    """No-op circuit breaker that always allows operations."""
    def allow(self) -> bool:
        return True

    def on_success(self) -> None:
        pass

    def on_error(self, err: Exception) -> None:
        pass


class SimpleCircuitBreaker:
    """
    Tiny counter-based breaker:
      - Opens after `consecutive_failure_threshold` errors.
      - Half-opens after `reset_timeout_s`; first success closes it.

    Intended for standalone/dev use only (not distributed).
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
            # Close on successful probe
            self._opened_at = None
            self._half_open = False
            self._failures = 0
        else:
            self._failures = 0

    def on_error(self, _err: Exception) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = time.monotonic()
            self._failures = 0
            self._half_open = False


class NoopCache:
    """No-op cache used in thin/composed mode."""
    async def get(self, key: str) -> Optional[Any]:
        return None

    async def set(self, key: str, value: Any, ttl_s: int) -> None:
        pass


class InMemoryTTLCache:
    """
    Very small, in-memory TTL cache (not for large workloads).

    Characteristics:
        - Per-process only; NOT multi-process or distributed.
        - Not thread-safe; intended for use in a single-threaded async event loop.
        - Stores Python objects by reference; no serialization is performed.
          If you need cross-process or cross-host caching, use a different Cache
          implementation with explicit serialization.

    This implementation also exposes an optional `invalidate_namespace(namespace: str)`
    method used by BaseVectorAdapter for namespace-scoped invalidation.
    """
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
        if ttl_s == 0:
            # Treat TTL=0 as "do not cache" (alignment with other protocol bases).
            return
        self._store[key] = (time.monotonic() + ttl_s, value)

    async def invalidate_namespace(self, namespace: str) -> None:
        """
        Best-effort namespace invalidation based on the standard cache key
        pattern used by BaseVectorAdapter (`ns=<namespace>:`).
        """
        if not namespace:
            return
        pattern = f"ns={namespace}:"
        keys_to_remove = [k for k in list(self._store.keys()) if pattern in k]
        for k in keys_to_remove:
            self._store.pop(k, None)


class NoopLimiter:
    """No-op limiter used in thin/composed mode."""
    async def acquire(self) -> None:
        pass

    def release(self) -> None:
        pass


class SimpleTokenBucketLimiter:
    """
    Simple token-bucket limiter with second-level refill.

    Characteristics:
        - Per-process only; not distributed.
        - Not thread-safe; intended for a single-threaded async event loop.
        - Fail-open on internal errors to avoid deadlocks in critical paths.

    Intended for standalone/dev use. For production backpressure across
    multiple workers, use a distributed rate limiter.

    Note:
        release() is intentionally a no-op to align with LLM/Graph token bucket
        semantics: tokens are charged on acquire() only.
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
                # Back off in small steps
                await asyncio.sleep(0.02)
        except Exception:
            # Fail-open on limiter errors
            return

    def release(self) -> None:
        # No-op to align with LLM/Graph token bucket semantics (charge on acquire only).
        return


# =============================================================================
# Capabilities (dynamic discovery for routing and planning)
# =============================================================================

@dataclass(frozen=True)
class VectorCapabilities:
    """
    Describes the capabilities and limitations of a vector adapter implementation.

    Used by routing layers for intelligent database selection, query planning,
    and feature compatibility checking across different vector backends.

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
        text_storage_strategy: How text is stored ("metadata", "docstore", "none")
        max_text_length: Maximum text length supported (None means unlimited)
        supports_batch_queries: Whether batch query operations are supported
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
    text_storage_strategy: str = "metadata"  # "metadata", "docstore", "none"
    max_text_length: Optional[int] = None
    supports_batch_queries: bool = False


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
class BatchQuerySpec:
    """
    Specification for batch vector similarity search queries.

    Attributes:
        queries: List of individual query specifications
        namespace: Target namespace/collection for all queries
    """
    queries: List[QuerySpec]
    namespace: str = "default"


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
    and designed for high-concurrency environments. This protocol is language-level;
    the canonical wire contract is defined by the JSON envelopes.

    Note: batch_query is a V1.0 addition. Adapters that don't support batch queries
    should raise NotSupported when this method is called.
    """

    async def capabilities(self) -> VectorCapabilities:
        """Get the capabilities of this vector adapter."""
        ...

    async def query(
        self,
        spec: QuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        """Execute a vector similarity search query."""
        ...

    async def batch_query(
        self,
        spec: BatchQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> List[QueryResult]:
        """
        Execute multiple vector similarity search queries in batch.

        This is a V1.0 addition. Adapters that don't support batch queries
        should raise NotSupported.
        """
        ...

    async def upsert(
        self,
        spec: UpsertSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        """Upsert vectors into the vector store."""
        ...

    async def delete(
        self,
        spec: DeleteSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        """Delete vectors from the vector store."""
        ...

    async def create_namespace(
        self,
        spec: NamespaceSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        """Create a new namespace/collection."""
        ...

    async def delete_namespace(
        self,
        namespace: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        """Delete a namespace/collection."""
        ...

    async def health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """Check backend health status."""
        ...


# =============================================================================
# Structured Configuration
# =============================================================================

@dataclass(frozen=True)
class VectorAdapterConfig:
    """
    Structured configuration for BaseVectorAdapter policies.

    This allows callers to configure core behavior (breaker, limiter, cache TTLs)
    in a single place, instead of ad-hoc kwargs.

    Notes:
        - `cache_*_ttl_s` are used as defaults when explicit constructor args
          are not provided. Use 0 to disable caching for that operation.
        - `limiter_*` and `breaker_*` are used only when BaseVectorAdapter is
          constructing its own SimpleTokenBucketLimiter / SimpleCircuitBreaker.
        - `auto_normalize`: If True, vectors in upsert/query operations are
          normalized to unit length (L2) automatically. This is particularly
          useful for cosine similarity searches.
    """
    cache_query_ttl_s: int = 60
    cache_caps_ttl_s: int = 30
    breaker_failure_threshold: int = 5
    breaker_reset_timeout_s: float = 10.0
    limiter_rate_per_sec: int = 50
    limiter_burst: int = 100
    auto_normalize: bool = False


# =============================================================================
# Base Instrumented Adapter (validation, metrics, error handling)
# =============================================================================

class BaseVectorAdapter(VectorProtocolV1):
    """
    Base class for implementing Vector Protocol V1.0 adapters.

    Provides common validation, metrics instrumentation, error handling, and
    SIEM-safe observability. Implementers override the `_do_*` methods to plug in
    backend-specific behavior while inheriting:

      - Normalized error taxonomy
      - Simple deadline enforcement (if configured)
      - Circuit breaker integration
      - Read-path caching for queries (standalone mode)
      - Rate limiting via token bucket (standalone mode)
      - Canonical metrics emission for ops & latency
      - Automatic text storage/retrieval via DocStore
      - Cache invalidation on write operations
      - Optional automatic vector normalization for cosine similarity

    Text Storage Behavior:
      - Upsert: Docstore failures cause the entire operation to fail (atomicity)
      - Query: Docstore failures cause missing text but don't fail the query (graceful degradation)

    Auto-Normalization Behavior:
      - When enabled (auto_normalize=True), vectors are automatically normalized to unit length
      - Particularly useful for cosine similarity searches
      - Includes optimization to skip normalization if vector is already unit length
      - Applied consistently to query, batch_query, and upsert operations

    Threading:
        - In-memory infra (cache, breaker, limiter) is not thread-safe.
        - Intended for single-threaded async event loops. Use external distributed
          infra for multi-threaded or multi-process deployments.

    Backpressure:
        - The SimpleTokenBucketLimiter provides per-process QPS-style backpressure.
        - For high-volume or large-batch workloads, callers should also respect
          capabilities.max_batch_size and manually chunk work into smaller requests.
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
        docstore: Optional[DocStore] = None,
        cache_query_ttl_s: Optional[int] = None,
        cache_caps_ttl_s: Optional[int] = None,
        warn_on_standalone_no_metrics: bool = True,
        config: Optional[VectorAdapterConfig] = None,
    ) -> None:
        """
        Initialize the vector adapter with metrics instrumentation and optional policies.

        Args:
            metrics: Metrics sink. Uses NoopMetrics if None.
            mode: "thin" or "standalone" (see module docs).
            deadline_policy: Optional deadline policy override.
            breaker: Optional circuit breaker override.
            cache: Optional cache override (read paths).
            limiter: Optional rate limiter override.
            docstore: Optional document store for text storage.
            cache_query_ttl_s: TTL for query cache entries in standalone mode. Use 0 to disable.
            cache_caps_ttl_s: TTL for capabilities cache entries in standalone mode. Use 0 to disable.
            warn_on_standalone_no_metrics: Warn if standalone is used without metrics.
            config: Optional VectorAdapterConfig to centralize policy configuration.
                    Explicit constructor args take precedence over config fields.
        """
        self._metrics: MetricsSink = metrics or NoopMetrics()

        # Structured config defaults
        cfg = config or VectorAdapterConfig()

        # Effective TTLs: explicit args win, config is fallback. Allow 0 to disable.
        self._cache_query_ttl_s = (
            cache_query_ttl_s if cache_query_ttl_s is not None
            else cfg.cache_query_ttl_s
        )
        self._cache_caps_ttl_s = (
            cache_caps_ttl_s if cache_caps_ttl_s is not None
            else cfg.cache_caps_ttl_s
        )
        self._auto_normalize = cfg.auto_normalize

        m = (mode or "thin").strip().lower()
        if m not in {"thin", "standalone"}:
            m = "thin"
        self._mode = m

        if self._mode == "thin":
            self._deadline: DeadlinePolicy = deadline_policy or NoopDeadline()
            self._breaker: CircuitBreaker = breaker or NoopBreaker()
            self._cache: Cache = cache or NoopCache()
            self._limiter: RateLimiter = limiter or NoopLimiter()
        else:
            self._deadline = deadline_policy or SimpleDeadline()
            self._breaker = breaker or SimpleCircuitBreaker(
                consecutive_failure_threshold=cfg.breaker_failure_threshold,
                reset_timeout_s=cfg.breaker_reset_timeout_s,
            )
            self._cache = cache or InMemoryTTLCache()
            self._limiter = limiter or SimpleTokenBucketLimiter(
                rate_per_sec=cfg.limiter_rate_per_sec,
                burst=cfg.limiter_burst,
            )
            if warn_on_standalone_no_metrics and isinstance(self._metrics, NoopMetrics):
                LOG.warning(
                    "Using standalone mode without metrics — provide a MetricsSink for production use"
                )

        # Document store for text storage
        self._docstore = docstore

        # Capabilities cache key is namespaced per adapter instance (module/class/id)
        self._caps_cache_key = (
            f"{VECTOR_PROTOCOL_VERSION}:capabilities:"
            f"{self.__class__.__module__}.{self.__class__.__qualname__}:{id(self)}"
        )

    # --- async context management & cleanup hooks ----------------------------

    async def __aenter__(self) -> "BaseVectorAdapter":
        """
        Async context manager entry.

        Implementers may override `close()` to ensure resources are released when
        used via `async with BaseVectorAdapter(...):`.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Async context manager exit.

        Delegates to `close()`; errors during cleanup SHOULD NOT raise unless
        they represent critical invariants.
        """
        await self.close()

    async def close(self) -> None:
        """
        Clean up resources (connections, pools, clients).

        Override in backend implementations that maintain external resources.
        Default implementation is a no-op.

        Note: In-memory components (cache, breaker, limiter) do not require
        explicit cleanup as they have no external resource handles.
        """
        pass

    # --- internal helpers (validation and instrumentation) -------------------

    @staticmethod
    def _require_non_empty(name: str, value: str) -> None:
        """Validate that a string value is non-empty."""
        if not isinstance(value, str) or not value.strip():
            raise BadRequest(f"{name} must be a non-empty string")

    def _validate_vector(self, vector: List[float], normalize: bool = False) -> List[float]:
        """
        Validate that a vector is properly formed and optionally normalize it.

        Requirements:
            - non-empty list
            - all elements numeric (int/float)

        Args:
            vector: The input vector list.
            normalize: If True, returns the L2 normalized version of the vector.
                       Throws BadRequest if vector length is zero.

        Returns:
            The validated (and potentially normalized) vector.
        """
        if not isinstance(vector, list) or not vector:
            raise BadRequest("vector must be a non-empty list of floats")
        if not all(isinstance(x, (int, float)) for x in vector):
            raise BadRequest("vector must contain only numeric values")

        if normalize:
            norm = math.sqrt(sum(x * x for x in vector))
            if norm == 0:
                raise BadRequest("cannot normalize zero-length vector")
            # Optimization: if already close to 1.0, return as-is
            if abs(norm - 1.0) < 1e-6:
                return vector
            return [x / norm for x in vector]

        return vector

    @staticmethod
    def _ensure_json_serializable(value: Any, label: str) -> None:
        """
        Ensure a value is JSON-serializable.

        Used to fail fast on invalid filter/metadata payloads that must traverse
        wire contracts or caches. Error messages are SIEM-safe and low detail.
        """
        if value is None:
            return
        try:
            json.dumps(value)
        except (TypeError, ValueError) as e:
            raise BadRequest(f"{label} must be JSON-serializable: {e}")

    @staticmethod
    def _tenant_hash(tenant: Optional[str]) -> Optional[str]:
        """
        Create privacy-preserving hash of tenant identifier for metrics.

        Raw tenant IDs MUST NOT appear directly in metrics.
        Uses a 12-character prefix of SHA256 for low-cardinality labeling.
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
        except Exception:
            pass

    async def _apply_deadline(self, awaitable: Awaitable[Any], ctx: Optional[OperationContext]) -> Any:
        """
        Apply the configured deadline policy to an awaitable.

        Any asyncio.TimeoutError is normalized into DeadlineExceeded.
        """
        try:
            return await self._deadline.wrap(awaitable, ctx)
        except DeadlineExceeded:
            # Propagate adapter-specific deadline errors
            raise
        except asyncio.TimeoutError:
            raise DeadlineExceeded("operation timed out", details={"remaining_ms": 0})

    def _fail_if_expired(self, ctx: Optional[OperationContext]) -> None:
        """
        Fail fast if ctx.deadline_ms is already expired.

        Avoids wasting backend capacity on doomed requests.
        """
        if ctx is None or ctx.deadline_ms is None:
            return
        if ctx.remaining_ms() == 0:
            raise DeadlineExceeded(
                "operation timed out (preflight)",
                details={"preflight": True, "remaining_ms": 0},
            )

    # --- cache keys (read paths only) ----------------------------------------

    @staticmethod
    def _hash_obj(obj: Any) -> str:
        """
        Stable hash for JSON-serializable objects.

        Uses json.dumps(..., sort_keys=True) with type prefix to avoid collisions
        for different types that serialize to identical JSON.
        """
        if isinstance(obj, (list, dict)):
            # Stable JSON with type prefix
            payload = json.dumps(obj, sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(f"j:{payload}".encode()).hexdigest()
        else:
            # Use repr for non-JSON types with prefix
            return hashlib.sha256(f"p:{repr(obj)}".encode()).hexdigest()

    def _query_cache_key(
        self,
        spec: QuerySpec,
        caps: Optional[VectorCapabilities],
        ctx: Optional[OperationContext],
    ) -> str:
        """
        Compose a cache key for query() that avoids cross-hit pollution.

        Includes:
            - vector hash
            - namespace, top_k, filter, include_* flags
            - backend identity (server/version)
            - tenant hash (if present)
            - text storage strategy
            - protocol version for cache safety
        """
        tenant_h = self._tenant_hash(ctx.tenant) if ctx else None
        caps_part = f"{caps.server}:{caps.version}" if caps else "unknown"
        text_strategy = caps.text_storage_strategy if caps else "unknown"
        return (
            f"{VECTOR_PROTOCOL_VERSION}:query:"
            f"{caps_part}:"
            f"ns={spec.namespace}:"
            f"topk={spec.top_k}:"
            f"im={int(bool(spec.include_metadata))}:iv={int(bool(spec.include_vectors))}:"
            f"vec={self._hash_obj(spec.vector)}:"
            f"flt={self._hash_obj(spec.filter)}:"
            f"tenant={tenant_h}:"
            f"text={text_strategy}"
        )

    def _batch_query_cache_key(
        self,
        spec: BatchQuerySpec,
        caps: Optional[VectorCapabilities],
        ctx: Optional[OperationContext],
    ) -> str:
        """
        Compose a cache key for batch_query().
        """
        tenant_h = self._tenant_hash(ctx.tenant) if ctx else None
        caps_part = f"{caps.server}:{caps.version}" if caps else "unknown"
        text_strategy = caps.text_storage_strategy if caps else "unknown"

        # Hash all queries for the cache key
        queries_hash = self._hash_obj([
            {
                "vector": q.vector,
                "top_k": q.top_k,
                "filter": q.filter,
                "include_metadata": q.include_metadata,
                "include_vectors": q.include_vectors
            }
            for q in spec.queries
        ])

        return (
            f"{VECTOR_PROTOCOL_VERSION}:batch_query:"
            f"{caps_part}:"
            f"ns={spec.namespace}:"
            f"queries={queries_hash}:"
            f"tenant={tenant_h}:"
            f"text={text_strategy}"
        )

    def _namespace_cache_pattern(self, namespace: str) -> str:
        """
        Return a string pattern for cache keys belonging to a specific namespace.
        Used for cache invalidation on write operations.

        Note: This uses simple substring matching for in-memory cache.
        For distributed caches, cache implementations can provide their own
        `invalidate_namespace(namespace: str)` hook.
        """
        return f"ns={namespace}:"

    async def _invalidate_namespace_cache(self, namespace: str) -> None:
        """
        Best-effort namespace cache invalidation.

        Behavior:
            - If the underlying cache exposes an async `invalidate_namespace(namespace)`
              method, it is called directly.
            - Otherwise, this is a no-op and TTL-based expiry is relied upon.

        This avoids reaching into cache internals (e.g., _store) and allows
        distributed caches to provide their own invalidation semantics.
        """
        if isinstance(self._cache, NoopCache):
            return

        try:
            invalidate = getattr(self._cache, "invalidate_namespace", None)
            if callable(invalidate):
                await invalidate(namespace)
        except Exception:
            # Never let cache invalidation break the main operation
            LOG.debug("Cache invalidation failed for namespace %s", namespace)

    # --- unified unary gate wrapper (for data-path ops) ----------------------

    async def _with_gates_unary(
        self,
        *,
        op: str,
        ctx: Optional[OperationContext],
        call: Callable[[], Awaitable[Any]],
        metric_extra: Optional[Mapping[str, Any]] = None,
        on_result: Optional[Callable[[Any], Mapping[str, Any]]] = None,
    ) -> Any:
        """
        DRY wrapper for unary (non-streaming) data-path operations.

        Responsibilities:
            - Preflight deadline check
            - Circuit breaker gate
            - Rate limiter acquire/release
            - DeadlinePolicy enforcement
            - Metrics emission (success/failure) with optional result-derived fields

        Behavior is aligned with the explicit per-method logic:
            - Same error types and codes
            - Same breaker semantics
            - Same limiter semantics
        """
        extras = dict(metric_extra or {})
        self._fail_if_expired(ctx)

        if not self._breaker.allow():
            e = Unavailable("circuit open")
            # Record immediately to preserve observability for hard rejections
            self._record(op, time.monotonic(), False, code=e.code or type(e).__name__, ctx=ctx, **extras)
            raise e

        await self._limiter.acquire()
        t0 = time.monotonic()
        try:
            result = await self._apply_deadline(call(), ctx)

            # Allow per-op enrichment based on the result (e.g., matches, counts, cache hit)
            if on_result is not None:
                try:
                    res_extras = on_result(result) or {}
                    extras.update(res_extras)
                except Exception:
                    # Never let metrics enrichment break the main path
                    pass

            self._record(op, t0, True, ctx=ctx, **extras)
            self._breaker.on_success()
            return result
        except VectorAdapterError as e:
            self._record(
                op,
                t0,
                False,
                code=e.code or type(e).__name__,
                ctx=ctx,
                **extras,
            )
            self._breaker.on_error(e)
            raise
        except Exception as e:
            self._record(
                op,
                t0,
                False,
                code="UNAVAILABLE",
                ctx=ctx,
                **extras,
            )
            self._breaker.on_error(e)
            raise
        finally:
            self._limiter.release()

    # --- final public APIs (validation + instrumentation) --------------------

    async def capabilities(self) -> VectorCapabilities:
        """
        Get the capabilities of this vector adapter (with optional caching).

        Standalone mode:
            - Results may be cached briefly to avoid hot path calls.
            - Callers MUST treat capabilities as advisory and refresh periodically.
        """
        t0 = time.monotonic()
        try:
            # Check cache if TTL is non-zero
            if self._mode == "standalone" and self._cache_caps_ttl_s > 0:
                cached = await self._cache.get(self._caps_cache_key)
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

            # Cache if TTL is non-zero
            if self._mode == "standalone" and self._cache_caps_ttl_s > 0:
                try:
                    await self._cache.set(self._caps_cache_key, caps, ttl_s=self._cache_caps_ttl_s)
                except Exception:
                    pass

            self._record("capabilities", t0, True)
            return caps
        except VectorAdapterError as e:
            self._record("capabilities", t0, False, code=e.code or type(e).__name__)
            raise
        except Exception as e:
            self._record("capabilities", t0, False, code="UNAVAILABLE")
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

        Docstore Behavior:
            - If docstore is configured and text hydration fails, the query continues
              but vectors will have text=None (graceful degradation).

        Auto-Normalization:
            - If auto_normalize is enabled, query vectors are normalized to unit length
            - Particularly useful for cosine similarity searches

        Backpressure note:
            - For high top_k values or very hot namespaces, consider using a
              distributed rate limiter in addition to the built-in per-process
              token bucket to avoid overload in large clusters.
        """
        # Structural validation and optional normalization
        spec_vector = self._validate_vector(spec.vector, normalize=self._auto_normalize)

        # Track if normalization occurred for metrics
        normalization_occurred = self._auto_normalize and spec_vector is not spec.vector

        self._require_non_empty("namespace", spec.namespace)
        if not isinstance(spec.top_k, int) or spec.top_k <= 0:
            raise BadRequest("top_k must be a positive integer")
        if spec.filter is not None and not isinstance(spec.filter, Mapping):
            raise BadRequest("filter must be a mapping (dict) when provided")
        if not isinstance(spec.include_metadata, bool) or not isinstance(spec.include_vectors, bool):
            raise BadRequest("include_metadata/include_vectors must be booleans")

        # If normalization changed the vector, create new spec
        if normalization_occurred:
            spec = QuerySpec(
                vector=spec_vector,
                top_k=spec.top_k,
                namespace=spec.namespace,
                filter=spec.filter,
                include_metadata=spec.include_metadata,
                include_vectors=spec.include_vectors
            )

        # JSON-serializability validation for filters (wire/caching safety)
        if spec.filter is not None:
            self._ensure_json_serializable(spec.filter, "filter")

        async def _call() -> QueryResult:
            caps = await self.capabilities()

            # Capability gating
            if caps.max_dimensions and len(spec.vector) > int(caps.max_dimensions):
                raise DimensionMismatch(
                    f"vector dimension {len(spec.vector)} exceeds max {caps.max_dimensions}",
                    details={"provided": len(spec.vector), "max": int(caps.max_dimensions)},
                )
            if caps.max_top_k is not None and spec.top_k > caps.max_top_k:
                raise BadRequest(
                    f"top_k {spec.top_k} exceeds maximum of {caps.max_top_k}",
                    details={"max_top_k": caps.max_top_k},
                )
            if spec.filter and not caps.supports_metadata_filtering:
                raise NotSupported("metadata filtering is not supported by this adapter")

            # Read-path cache (standalone only with non-zero TTL)
            cache_hit = False
            if self._mode == "standalone" and self._cache_query_ttl_s > 0:
                ck = self._query_cache_key(spec, caps, ctx)
                cached = await self._cache.get(ck)
                if cached:
                    cache_hit = True
                    self._metrics.counter(
                        component=self._component,
                        name="cache_hits",
                        value=1,
                        extra={"op": "query"},
                    )
                    return cached
                else:
                    self._metrics.counter(
                        component=self._component,
                        name="cache_misses",
                        value=1,
                        extra={"op": "query"},
                    )

            # Execute backend query
            result = await self._do_query(spec, ctx=ctx)

            # Hydrate text from docstore if configured (graceful degradation on failure)
            if self._docstore is not None and result.matches:
                try:
                    doc_ids = [str(match.vector.id) for match in result.matches]
                    docs = await self._docstore.batch_get(doc_ids)

                    # Rebuild matches with hydrated text
                    hydrated_matches = []
                    for match in result.matches:
                        doc = docs.get(str(match.vector.id))
                        if doc is not None:
                            # Create new vector with text populated
                            hydrated_vector = Vector(
                                id=match.vector.id,
                                vector=match.vector.vector,
                                metadata=match.vector.metadata,
                                namespace=match.vector.namespace,
                                text=doc.text  # Hydrated from docstore
                            )
                            hydrated_matches.append(VectorMatch(
                                vector=hydrated_vector,
                                score=match.score,
                                distance=match.distance
                            ))
                        else:
                            # No text found, pass through as-is
                            hydrated_matches.append(match)

                    # Replace matches with hydrated version
                    result = QueryResult(
                        matches=hydrated_matches,
                        query_vector=result.query_vector,
                        namespace=result.namespace,
                        total_matches=result.total_matches
                    )
                except Exception as e:
                    # Graceful degradation: log but don't fail the query
                    LOG.debug(
                        "Docstore hydration failed for query in namespace %s: %r",
                        result.namespace,
                        e,
                    )
                    self._metrics.counter(
                        component=self._component,
                        name="docstore_hydration_errors",
                        value=1,
                        extra={"op": "query"},
                    )
                    # Continue with original result (text will be None)

            # Populate cache if eligible (non-zero TTL and not already cached)
            if self._mode == "standalone" and self._cache_query_ttl_s > 0 and not cache_hit:
                try:
                    ck = self._query_cache_key(spec, caps, ctx)
                    await self._cache.set(ck, result, ttl_s=self._cache_query_ttl_s)
                except Exception:
                    pass

            return result

        def _on_result(res: QueryResult) -> Mapping[str, Any]:
            extra: Dict[str, Any] = {
                "namespace": spec.namespace,
                "top_k": spec.top_k,
            }
            try:
                extra["matches"] = len(res.matches)
            except Exception:
                pass
            # Record normalization metrics if it occurred
            if normalization_occurred:
                extra["vector_normalized"] = True
            return extra

        result = await self._with_gates_unary(
            op="query",
            ctx=ctx,
            call=_call,
            on_result=_on_result,
        )

        # Record normalization metric if it occurred
        if normalization_occurred:
            self._metrics.counter(
                component=self._component,
                name="vectors_normalized",
                value=1,
                extra={"op": "query"},
            )

        # Request counters (keep legacy name and add cross-component-friendly name)
        self._metrics.counter(
            component=self._component,
            name="queries",
            value=1,
        )
        self._metrics.counter(
            component=self._component,
            name="requests_total",
            value=1,
            extra={"op": "query"},
        )

        return result

    async def batch_query(
        self,
        spec: BatchQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> List[QueryResult]:
        """
        Execute multiple vector similarity search queries in batch.

        This can be more efficient than individual queries for some backends.
        Adapters that don't support batch queries should raise NotSupported.

        Docstore Behavior:
            - If docstore is configured and text hydration fails, the batch query continues
              but vectors will have text=None (graceful degradation).

        Auto-Normalization:
            - If auto_normalize is enabled, all query vectors are normalized to unit length
        """
        self._require_non_empty("namespace", spec.namespace)
        if not spec.queries:
            raise BadRequest("queries must not be empty")

        # Validate each query and track normalization
        normalized_queries = []
        queries_changed = False
        normalization_count = 0

        for i, query in enumerate(spec.queries):
            norm_vec = self._validate_vector(query.vector, normalize=self._auto_normalize)

            # Track if this vector was normalized
            if self._auto_normalize and norm_vec is not query.vector:
                queries_changed = True
                normalization_count += 1

            if not isinstance(query.top_k, int) or query.top_k <= 0:
                raise BadRequest(f"query[{i}].top_k must be a positive integer")
            if query.filter is not None and not isinstance(query.filter, Mapping):
                raise BadRequest(f"query[{i}].filter must be a mapping (dict) when provided")
            if query.filter is not None:
                self._ensure_json_serializable(query.filter, f"query[{i}].filter")

            if self._auto_normalize and norm_vec is not query.vector:
                # Create updated query spec
                normalized_queries.append(QuerySpec(
                    vector=norm_vec,
                    top_k=query.top_k,
                    namespace=query.namespace,
                    filter=query.filter,
                    include_metadata=query.include_metadata,
                    include_vectors=query.include_vectors
                ))
            else:
                normalized_queries.append(query)

        # Update spec if normalization occurred
        if queries_changed:
            spec = BatchQuerySpec(queries=normalized_queries, namespace=spec.namespace)

        async def _call() -> List[QueryResult]:
            caps = await self.capabilities()

            if not caps.supports_batch_queries:
                raise NotSupported("batch queries are not supported by this adapter")

            # Validate capabilities for all queries
            for i, query in enumerate(spec.queries):
                if caps.max_dimensions and len(query.vector) > int(caps.max_dimensions):
                    raise DimensionMismatch(
                        f"query[{i}] vector dimension {len(query.vector)} exceeds max {caps.max_dimensions}",
                        details={"provided": len(query.vector), "max": int(caps.max_dimensions)},
                    )
                if caps.max_top_k is not None and query.top_k > caps.max_top_k:
                    raise BadRequest(
                        f"query[{i}] top_k {query.top_k} exceeds maximum of {caps.max_top_k}",
                        details={"max_top_k": caps.max_top_k},
                    )
                if query.filter and not caps.supports_metadata_filtering:
                    raise NotSupported(f"query[{i}] metadata filtering is not supported by this adapter")

            # Read-path cache (standalone only with non-zero TTL)
            cache_hit = False
            if self._mode == "standalone" and self._cache_query_ttl_s > 0:
                ck = self._batch_query_cache_key(spec, caps, ctx)
                cached = await self._cache.get(ck)
                if cached:
                    cache_hit = True
                    self._metrics.counter(
                        component=self._component,
                        name="cache_hits",
                        value=1,
                        extra={"op": "batch_query"},
                    )
                    return cached
                else:
                    self._metrics.counter(
                        component=self._component,
                        name="cache_misses",
                        value=1,
                        extra={"op": "batch_query"},
                    )

            # Execute backend batch query
            results = await self._do_batch_query(spec, ctx=ctx)

            # Hydrate text from docstore if configured (graceful degradation on failure)
            if self._docstore is not None:
                try:
                    # Collect all document IDs from all results
                    all_doc_ids = []
                    for result in results:
                        for match in result.matches:
                            all_doc_ids.append(str(match.vector.id))

                    if all_doc_ids:
                        docs = await self._docstore.batch_get(all_doc_ids)

                        # Rebuild all results with hydrated text
                        hydrated_results = []
                        for result in results:
                            hydrated_matches = []
                            for match in result.matches:
                                doc = docs.get(str(match.vector.id))
                                if doc is not None:
                                    hydrated_vector = Vector(
                                        id=match.vector.id,
                                        vector=match.vector.vector,
                                        metadata=match.vector.metadata,
                                        namespace=match.vector.namespace,
                                        text=doc.text
                                    )
                                    hydrated_matches.append(VectorMatch(
                                        vector=hydrated_vector,
                                        score=match.score,
                                        distance=match.distance
                                    ))
                                else:
                                    hydrated_matches.append(match)

                            hydrated_results.append(QueryResult(
                                matches=hydrated_matches,
                                query_vector=result.query_vector,
                                namespace=result.namespace,
                                total_matches=result.total_matches
                            ))

                        results = hydrated_results
                except Exception as e:
                    # Graceful degradation: log but don't fail the batch query
                    LOG.debug("Docstore hydration failed for batch query in namespace %s: %r", spec.namespace, e)
                    self._metrics.counter(
                        component=self._component,
                        name="docstore_hydration_errors",
                        value=1,
                        extra={"op": "batch_query"},
                    )
                    # Continue with original results (text will be None)

            # Populate cache if eligible
            if self._mode == "standalone" and self._cache_query_ttl_s > 0 and not cache_hit:
                try:
                    ck = self._batch_query_cache_key(spec, caps, ctx)
                    await self._cache.set(ck, results, ttl_s=self._cache_query_ttl_s)
                except Exception:
                    pass

            return results

        def _on_result(results: List[QueryResult]) -> Mapping[str, Any]:
            total_matches = sum(len(result.matches) for result in results)
            extra = {
                "namespace": spec.namespace,
                "query_count": len(spec.queries),
                "total_matches": total_matches,
            }
            # Record normalization metrics if any occurred
            if normalization_count > 0:
                extra["vectors_normalized"] = normalization_count
            return extra

        results = await self._with_gates_unary(
            op="batch_query",
            ctx=ctx,
            call=_call,
            on_result=_on_result,
        )

        # Record normalization metrics if any occurred
        if normalization_count > 0:
            self._metrics.counter(
                component=self._component,
                name="vectors_normalized",
                value=normalization_count,
                extra={"op": "batch_query"},
            )

        self._metrics.counter(
            component=self._component,
            name="batch_queries",
            value=1,
        )
        self._metrics.counter(
            component=self._component,
            name="queries_in_batch",
            value=len(spec.queries),
        )
        self._metrics.counter(
            component=self._component,
            name="requests_total",
            value=1,
            extra={"op": "batch_query"},
        )

        return results

    async def upsert(
        self,
        spec: UpsertSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        """
        Upsert vectors into the vector store with validation and metrics.

        See VectorProtocolV1.upsert for full documentation.

        Docstore Behavior:
            - If docstore is configured and text storage fails, the entire upsert
              operation fails (atomicity).

        Auto-Normalization:
            - If auto_normalize is enabled, all vectors are normalized to unit length
              before storage

        Guidance:
            - For very large batches, respect capabilities.max_batch_size and
              chunk requests accordingly to avoid memory pressure.
        """
        self._require_non_empty("namespace", spec.namespace)
        if not spec.vectors:
            raise BadRequest("vectors must not be empty")

        # Validate each vector, id, and its metadata
        # Normalize vectors if auto_normalize is enabled
        validated_vectors = []
        vectors_changed = False
        normalization_count = 0

        for v in spec.vectors:
            self._require_non_empty("vector.id", str(v.id))
            norm_vec = self._validate_vector(v.vector, normalize=self._auto_normalize)

            # Track if this vector was normalized
            if self._auto_normalize and norm_vec is not v.vector:
                vectors_changed = True
                normalization_count += 1

            if v.metadata is not None and not isinstance(v.metadata, Mapping):
                raise BadRequest("metadata must be a mapping (dict) when provided")
            if v.metadata is not None:
                self._ensure_json_serializable(v.metadata, "metadata")

            if self._auto_normalize and norm_vec is not v.vector:
                validated_vectors.append(Vector(
                    id=v.id,
                    vector=norm_vec,
                    metadata=v.metadata,
                    namespace=v.namespace,
                    text=v.text
                ))
            else:
                validated_vectors.append(v)

        if vectors_changed:
            spec = UpsertSpec(vectors=validated_vectors, namespace=spec.namespace)

        async def _call() -> UpsertResult:
            caps = await self.capabilities()

            # Enforce batch operation support
            if not caps.supports_batch_operations and len(spec.vectors) > 1:
                raise NotSupported(
                    "batch upsert is not supported by this adapter",
                    details={"requested": len(spec.vectors)},
                )

            # Validate text storage capabilities
            texts_present = any(v.text for v in spec.vectors)
            if texts_present:
                if caps.text_storage_strategy == "none":
                    raise NotSupported("Text storage is not supported by this adapter")
                if caps.max_text_length:
                    for v in spec.vectors:
                        if v.text and len(v.text) > caps.max_text_length:
                            raise BadRequest(
                                f"Text length {len(v.text)} exceeds maximum {caps.max_text_length}",
                                details={"max_text_length": caps.max_text_length}
                            )

            if caps.max_batch_size is not None and len(spec.vectors) > caps.max_batch_size:
                suggested = (
                    int(100 * (len(spec.vectors) - caps.max_batch_size) / len(spec.vectors))
                    if spec.vectors else None
                )
                raise BadRequest(
                    f"batch size {len(spec.vectors)} exceeds maximum of {caps.max_batch_size}",
                    details={"max_batch_size": caps.max_batch_size},
                    suggested_batch_reduction=suggested,
                )
            if caps.max_dimensions:
                for v in spec.vectors:
                    if len(v.vector) > caps.max_dimensions:
                        raise DimensionMismatch(
                            f"vector dimension {len(v.vector)} exceeds max {caps.max_dimensions}",
                            details={"provided": len(v.vector), "max": int(caps.max_dimensions)},
                        )

            # Handle text storage if docstore is configured (atomic operation)
            backend_vectors = spec.vectors
            if self._docstore is not None and texts_present:
                # Store texts in docstore and create cleaned vectors for backend
                texts_to_store = []
                cleaned_vectors = []

                for v in spec.vectors:
                    if v.text is not None:
                        # Store in docstore
                        texts_to_store.append(Document(
                            id=str(v.id),
                            text=v.text,
                            metadata=v.metadata or {}
                        ))

                        # Create cleaned vector without text
                        cleaned_vectors.append(Vector(
                            id=v.id,
                            vector=v.vector,
                            metadata=v.metadata,
                            namespace=v.namespace,
                            text=None  # Removed
                        ))
                    else:
                        # No text, pass through as-is
                        cleaned_vectors.append(v)

                # Batch store texts (failure here fails the entire upsert)
                if texts_to_store:
                    await self._docstore.batch_put(texts_to_store)

                backend_vectors = cleaned_vectors

            # Call backend with cleaned vectors (no text in metadata/vector)
            result = await self._do_upsert(
                UpsertSpec(vectors=backend_vectors, namespace=spec.namespace),
                ctx=ctx
            )

            # Invalidate cache for this namespace on successful upsert
            if result.upserted_count > 0:
                try:
                    await self._invalidate_namespace_cache(spec.namespace)
                except Exception:
                    # Never let cache invalidation break the operation
                    pass

            return result

        def _on_result(res: UpsertResult) -> Mapping[str, Any]:
            extra = {
                "namespace": spec.namespace,
                "vectors_processed": len(spec.vectors),
                "upserted_count": res.upserted_count,
            }
            # Record normalization metrics if any occurred
            if normalization_count > 0:
                extra["vectors_normalized"] = normalization_count
            return extra

        result = await self._with_gates_unary(
            op="upsert",
            ctx=ctx,
            call=_call,
            on_result=_on_result,
        )

        # Record normalization metrics if any occurred
        if normalization_count > 0:
            self._metrics.counter(
                component=self._component,
                name="vectors_normalized",
                value=normalization_count,
                extra={"op": "upsert"},
            )

        # Token-style counters for upserted vectors and requests
        self._metrics.counter(
            component=self._component,
            name="vectors_upserted",
            value=int(result.upserted_count),
        )
        self._metrics.counter(
            component=self._component,
            name="upsert_batches",
            value=1,
        )
        self._metrics.counter(
            component=self._component,
            name="requests_total",
            value=1,
            extra={"op": "upsert"},
        )

        return result

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
        if spec.filter is not None and not isinstance(spec.filter, Mapping):
            raise BadRequest("filter must be a mapping (dict) when provided")
        if spec.filter is not None:
            self._ensure_json_serializable(spec.filter, "filter")
        if spec.ids:
            for vid in spec.ids:
                self._require_non_empty("id", str(vid))

        async def _call() -> DeleteResult:
            caps = await self.capabilities()

            # Enforce batch operation support for multi-id deletes
            if not caps.supports_batch_operations and spec.ids and len(spec.ids) > 1:
                raise NotSupported(
                    "batch delete is not supported by this adapter",
                    details={"requested": len(spec.ids)},
                )

            if caps.max_batch_size is not None and spec.ids and len(spec.ids) > caps.max_batch_size:
                suggested = (
                    int(100 * (len(spec.ids) - caps.max_batch_size) / len(spec.ids))
                    if spec.ids else None
                )
                raise BadRequest(
                    f"batch size {len(spec.ids)} exceeds maximum of {caps.max_batch_size}",
                    details={"max_batch_size": caps.max_batch_size},
                    suggested_batch_reduction=suggested,
                )
            if spec.filter and not caps.supports_metadata_filtering:
                raise NotSupported("metadata filtering is not supported by this adapter")

            result = await self._do_delete(spec, ctx=ctx)

            # Clean up docstore entries and cache on successful delete
            if result.deleted_count > 0:
                # Clean up docstore entries (using efficient batch_delete)
                if self._docstore is not None and spec.ids:
                    try:
                        doc_ids = [str(doc_id) for doc_id in spec.ids]
                        await self._docstore.batch_delete(doc_ids)
                    except Exception:
                        # Log but don't fail the operation
                        LOG.debug("Failed to clean up docstore entries after delete")

                # Invalidate cache for this namespace
                try:
                    await self._invalidate_namespace_cache(spec.namespace)
                except Exception:
                    pass

            return result

        def _on_result(res: DeleteResult) -> Mapping[str, Any]:
            targeted = len(spec.ids) if spec.ids else 0
            return {
                "namespace": spec.namespace,
                "vectors_targeted": targeted,
                "deleted_count": res.deleted_count,
            }

        result = await self._with_gates_unary(
            op="delete",
            ctx=ctx,
            call=_call,
            on_result=_on_result,
        )

        self._metrics.counter(
            component=self._component,
            name="vectors_deleted",
            value=int(result.deleted_count),
        )
        self._metrics.counter(
            component=self._component,
            name="delete_batches",
            value=1,
        )
        self._metrics.counter(
            component=self._component,
            name="requests_total",
            value=1,
            extra={"op": "delete"},
        )

        return result

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

        async def _call() -> NamespaceResult:
            caps = await self.capabilities()

            if caps.max_dimensions and spec.dimensions > caps.max_dimensions:
                raise BadRequest(
                    f"dimensions {spec.dimensions} exceed maximum of {caps.max_dimensions}",
                    details={"max_dimensions": caps.max_dimensions},
                )
            if spec.distance_metric not in caps.supported_metrics:
                raise NotSupported(
                    f"distance_metric '{spec.distance_metric}' not supported",
                    details={"supported_metrics": caps.supported_metrics},
                )

            return await self._do_create_namespace(spec, ctx=ctx)

        def _on_result(_: NamespaceResult) -> Mapping[str, Any]:
            return {"namespace": spec.namespace}

        result = await self._with_gates_unary(
            op="create_namespace",
            ctx=ctx,
            call=_call,
            on_result=_on_result,
        )

        self._metrics.counter(
            component=self._component,
            name="requests_total",
            value=1,
            extra={"op": "create_namespace"},
        )

        return result

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

        async def _call() -> NamespaceResult:
            result = await self._do_delete_namespace(namespace, ctx=ctx)

            # Invalidate all cache entries for this namespace
            if result.success:
                try:
                    await self._invalidate_namespace_cache(namespace)
                except Exception:
                    pass

            return result

        def _on_result(_: NamespaceResult) -> Mapping[str, Any]:
            return {"namespace": namespace}

        result = await self._with_gates_unary(
            op="delete_namespace",
            ctx=ctx,
            call=_call,
            on_result=_on_result,
        )

        self._metrics.counter(
            component=self._component,
            name="requests_total",
            value=1,
            extra={"op": "delete_namespace"},
        )

        return result

    async def health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """
        Check health status with metrics instrumentation.

        Returns a small mapping; adapters may include extra fields, but callers
        must treat unknown keys as informational only.
        """
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
        except Exception as e:
            self._record("health", t0, False, code="UNAVAILABLE", ctx=ctx)
            raise Unavailable("health check failed") from e

    # --- hooks to implement per backend (override these) ---------------------

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

    async def _do_batch_query(
        self,
        spec: BatchQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> List[QueryResult]:
        """Implement batch vector similarity search with validated inputs."""
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
    Map VectorAdapterError (or unexpected Exception) to canonical error envelope.

    Single source of truth for wire-level error normalization.
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

    Transport-agnostic: can be wrapped by HTTP, gRPC, WebSockets, etc.

    Wire strictness:
        - Envelopes MUST contain all three top-level keys: op, ctx, args.
        - ctx and args MUST be JSON objects (mappings). Callers may send empty
          objects, but the keys must be present for protocol conformance.
    """

    def __init__(self, adapter: VectorProtocolV1):
        self._adapter = adapter

    @staticmethod
    def _require_mapping_field(envelope: Mapping[str, Any], field: str) -> Mapping[str, Any]:
        """
        Enforce that envelope[field] exists and is a mapping.

        This makes ctx/args strict at the wire handler boundary to match the
        canonical envelope contract.
        """
        if field not in envelope:
            raise BadRequest(f"missing required '{field}'")
        v = envelope.get(field)
        if not isinstance(v, Mapping):
            raise BadRequest(
                f"'{field}' must be an object",
                details={"field": field, "type": type(v).__name__},
            )
        return v

    @staticmethod
    def _require_mapping_list(name: str, value: Any) -> List[Mapping[str, Any]]:
        """
        Defensive parsing helper for wire lists.

        Ensures list elements are mappings so malformed input becomes BadRequest
        (not a generic UNAVAILABLE).
        """
        if value is None:
            return []
        if not isinstance(value, list):
            raise BadRequest(
                f"{name} must be a list",
                details={"field": name, "type": type(value).__name__},
            )
        out: List[Mapping[str, Any]] = []
        for i, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise BadRequest(
                    f"{name}[{i}] must be an object",
                    details={"field": f"{name}[{i}]", "type": type(item).__name__},
                )
            out.append(item)
        return out

    @staticmethod
    def _require_string_list(name: str, value: Any) -> List[str]:
        """Strictly require a list of strings for wire ID lists."""
        if value is None:
            return []
        if not isinstance(value, list):
            raise BadRequest(
                f"{name} must be a list",
                details={"field": name, "type": type(value).__name__},
            )
        out: List[str] = []
        for i, item in enumerate(value):
            if not isinstance(item, str):
                raise BadRequest(
                    f"{name}[{i}] must be a string",
                    details={"field": f"{name}[{i}]", "type": type(item).__name__},
                )
            out.append(item)
        return out

    async def handle(self, envelope: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Handle a single unary request envelope and return a response envelope.

        Supports:
            - vector.capabilities
            - vector.query
            - vector.batch_query (V1.0+)
            - vector.upsert
            - vector.delete
            - vector.create_namespace
            - vector.delete_namespace
            - vector.health
        """
        t0 = time.monotonic()
        try:
            if not isinstance(envelope, Mapping):
                raise BadRequest("envelope must be an object")

            op = envelope.get("op")
            if not isinstance(op, str):
                raise BadRequest("missing or invalid 'op'")

            # REQUIRED fields (ctx + args must be present; may be empty objects)
            ctx_map = self._require_mapping_field(envelope, "ctx")
            args = self._require_mapping_field(envelope, "args")
            ctx = _ctx_from_wire(ctx_map)

            if op == "vector.capabilities":
                res = await self._adapter.capabilities()
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "vector.query":
                try:
                    spec = QuerySpec(**dict(args))
                except TypeError as te:
                    raise BadRequest(f"missing required field(s): {str(te)}") from te
                res = await self._adapter.query(spec, ctx=ctx)
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "vector.batch_query":
                queries_raw = self._require_mapping_list("queries", args.get("queries"))
                if not queries_raw:
                    raise BadRequest("queries must not be empty")
                queries: List[QuerySpec] = []
                for i, q in enumerate(queries_raw):
                    try:
                        queries.append(QuerySpec(**dict(q)))
                    except TypeError as te:
                        raise BadRequest(
                            f"Invalid query spec at queries[{i}]: {te}",
                            details={"index": i},
                        ) from te
                    except Exception as e:
                        raise BadRequest(
                            f"Invalid query spec at queries[{i}]: {e}",
                            details={"index": i, "type": type(e).__name__},
                        ) from e
                spec = BatchQuerySpec(
                    queries=queries,
                    namespace=str(args.get("namespace", "default")),
                )
                res = await self._adapter.batch_query(spec, ctx=ctx)
                return _success_to_wire([asdict(r) for r in res], (time.monotonic() - t0) * 1000.0)

            if op == "vector.upsert":
                vectors_raw = self._require_mapping_list("vectors", args.get("vectors"))
                vectors: List[Vector] = []
                for i, v in enumerate(vectors_raw):
                    try:
                        vectors.append(Vector(**dict(v)))
                    except TypeError as te:
                        raise BadRequest(
                            f"Invalid vector at vectors[{i}]: {te}",
                            details={"index": i},
                        ) from te
                    except Exception as e:
                        raise BadRequest(
                            f"Invalid vector at vectors[{i}]: {e}",
                            details={"index": i, "type": type(e).__name__},
                        ) from e
                spec = UpsertSpec(
                    vectors=vectors,
                    namespace=str(args.get("namespace", "default")),
                )
                res = await self._adapter.upsert(spec, ctx=ctx)
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "vector.delete":
                ids_raw = self._require_string_list("ids", args.get("ids"))
                ids = [VectorID(v) for v in ids_raw]
                spec = DeleteSpec(
                    ids=ids,
                    namespace=str(args.get("namespace", "default")),
                    filter=args.get("filter"),
                )
                res = await self._adapter.delete(spec, ctx=ctx)
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "vector.create_namespace":
                try:
                    spec = NamespaceSpec(**dict(args))
                except TypeError as te:
                    raise BadRequest(f"Invalid namespace spec: {te}") from te
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
    "VectorMatch",
    "QueryResult",
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
    "BatchQuerySpec",
    "UpsertSpec",
    "DeleteSpec",
    "NamespaceSpec",
    "UpsertResult",
    "DeleteResult",
    "NamespaceResult",
    "VectorCapabilities",
    "VectorProtocolV1",
    "VectorAdapterConfig",
    "BaseVectorAdapter",
    # document storage
    "Document",
    "DocStore",
    "InMemoryDocStore",
    "RedisDocStore",
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
