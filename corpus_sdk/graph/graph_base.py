# adapter_sdk/graph_base.py
# SPDX-License-Identifier: Apache-2.0
"""
Adapter SDK — Graph Protocol V1 (public contract + production-grade base)

Purpose
-------
A minimal, production-quality surface for building graph adapters that plug into a
control plane. This protocol enables seamless integration with any graph database
while maintaining production-grade observability, security, and operational rigor.

Design Philosophy
-----------------
- Minimal surface area: Core graph operations only, no vendor-specific extensions
- Async-first: All operations are non-blocking for high-concurrency environments
- Production hardened: Built-in metrics, error taxonomy, and context propagation
- Extensible: Capability discovery allows for database-specific features
- Query agnostic: Supports multiple graph query dialects through unified interface

Deliberate Non-Goals
--------------------
- No retries, hedging, circuit breakers, failover, pooling, or dynamic config.
- No secret fetching or policy evaluation.
- No backend-specific optimizations beyond validated calls and timing.
- No query planning, optimization, or result post-processing.

Those behaviors live in the control plane and upper routing layers.

Mode Strategy
-------------
mode="thin" (default) — Composition mode. All resiliency hooks are no-ops and the
base performs only validation, timing, and SIEM-safe metrics. Use this when your
external router/manager provides rate limiting, circuit breaking, scheduling, and caching.

mode="standalone" — Self-contained mode for development and light production. Enables
basic deadline enforcement (ctx.deadline_ms), a tiny circuit breaker, a simple token
bucket rate limiter, and in-memory TTL caches for safe read paths (capabilities(),
get_schema(), query()). This keeps behavior deterministic without duplicating any
closed-source control-plane logic.

Versioning
----------
Follow SemVer against GRAPH_PROTOCOL_VERSION. Minor versions are strictly additive.
- Patch (x.y.Z): Editorial clarifications, non-breaking fixes
- Minor (x.Y.z): New optional parameters, capabilities, or methods
- Major (X.y.z): Breaking changes to signatures or behavior
"""

from __future__ import annotations
import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import (
    Any, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple,
    runtime_checkable, AsyncIterator, NewType, Callable
)

LOG = logging.getLogger(__name__)

GRAPH_PROTOCOL_VERSION = "1.1.0"  # minor bump: added capabilities fields & infra hooks
KNOWN_DIALECTS: Tuple[str, ...] = ("cypher", "opencypher", "gremlin", "gql")

# =============================================================================
# Core Type Definitions
# =============================================================================

GraphID = NewType('GraphID', str)
"""
Type alias for graph identifiers providing explicit type safety.

Using GraphID instead of raw string enhances protocol clarity and enables
better IDE support and type checking while maintaining string compatibility.
"""

class HealthStatus:
    """Standard health status constants for consistent health reporting."""
    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    READ_ONLY = "read_only"

# =============================================================================
# Normalized Errors (with retry hints and operational guidance)
# =============================================================================

class AdapterError(Exception):
    """
    Base exception for all graph adapter errors.

    Provides structured error information including retry guidance, throttling context,
    and operational suggestions for callers to handle failures gracefully.

    Attributes:
        message: Human-readable error description
        code: Machine-readable error code for programmatic handling
        retry_after_ms: Suggested delay before retry (None if not retryable)
        throttle_scope: Scope of throttling ("tenant", "cluster", "query_complexity")
        suggested_batch_reduction: Percentage reduction suggestion for batch size
        details: Additional context-specific error details
        operation: Operation that failed (for debugging and metrics)
        dialect: Query dialect in use during failure (if applicable)
    """
    def __init__(
        self,
        message: str = "",
        *,
        code: Optional[str] = None,
        retry_after_ms: Optional[int] = None,
        throttle_scope: Optional[str] = None,
        suggested_batch_reduction: Optional[int] = None,
        details: Optional[Mapping[str, Any]] = None,
        operation: Optional[str] = None,
        dialect: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.retry_after_ms = retry_after_ms
        self.throttle_scope = throttle_scope
        self.suggested_batch_reduction = suggested_batch_reduction
        self.details = dict(details or {})
        self.operation = operation
        self.dialect = dialect

    def asdict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization and logging."""
        result = {
            "message": self.message,
            "code": self.code,
            "retry_after_ms": self.retry_after_ms,
            "throttle_scope": self.throttle_scope,
            "suggested_batch_reduction": self.suggested_batch_reduction,
            "details": {k: self.details[k] for k in sorted(self.details)},
        }
        if self.operation:
            result["operation"] = self.operation
        if self.dialect:
            result["dialect"] = self.dialect
        return result

class BadRequest(AdapterError):
    """Client sent an invalid request (malformed parameters, invalid queries)."""
    pass

class AuthError(AdapterError):
    """Authentication or authorization failed (invalid credentials, permissions)."""
    pass

class ResourceExhausted(AdapterError):
    """Quota, rate limit, or resource constraints exceeded."""
    pass

class TransientNetwork(AdapterError):
    """Transient network failure that may succeed on retry."""
    pass

class Unavailable(AdapterError):
    """Service is temporarily unavailable or overloaded."""
    pass

class NotSupported(AdapterError):
    """Requested operation, dialect, or parameter is not supported."""
    pass

class DeadlineExceeded(AdapterError):
    """Operation exceeded ctx.deadline_ms budget."""
    pass

# =============================================================================
# Context (used for deadlines, identity, SIEM-safe metrics)
# =============================================================================

@dataclass(frozen=True)
class OperationContext:
    """
    Context for graph operations providing tracing, deadlines, and multi-tenant isolation.

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

    @property
    def is_timed_out(self) -> bool:
        """Check if the context deadline has already elapsed."""
        if self.deadline_ms is None:
            return False
        now_ms = int(time.time() * 1000.0)
        return now_ms >= int(self.deadline_ms)

    def remaining_ms(self) -> Optional[int]:
        """Return remaining time budget in milliseconds, if any."""
        if self.deadline_ms is None:
            return None
        now_ms = int(time.time() * 1000.0)
        return max(0, int(self.deadline_ms) - now_ms)

    def with_attrs(self, **new_attrs: Any) -> OperationContext:
        """
        Return new context with additional attributes.

        Args:
            **new_attrs: Additional attributes to merge into existing attrs

        Returns:
            OperationContext: New context with merged attributes
        """
        return OperationContext(
            request_id=self.request_id,
            idempotency_key=self.idempotency_key,
            deadline_ms=self.deadline_ms,
            traceparent=self.traceparent,
            tenant=self.tenant,
            attrs={**self.attrs, **new_attrs}
        )

# =============================================================================
# Observability Interfaces (SIEM-safe, low-cardinality)
# =============================================================================

class LogSink(Protocol):
    """
    Protocol for logging implementations.

    Used for structured logging without exposing sensitive information.
    All log data must avoid PII and use hashed tenant identifiers.
    """
    def debug(self, message: str, *, extra: Optional[Mapping[str, Any]] = None) -> None: ...
    def info(self, message: str, *, extra: Optional[Mapping[str, Any]] = None) -> None: ...
    def warning(self, message: str, *, extra: Optional[Mapping[str, Any]] = None) -> None: ...
    def error(self, message: str, *, extra: Optional[Mapping[str, Any]] = None) -> None: ...

class NoopLogSink:
    """No-operation log sink for testing or when logging is disabled."""
    def debug(self, message: str, **_: Any) -> None: ...
    def info(self, message: str, **_: Any) -> None: ...
    def warning(self, message: str, **_: Any) -> None: ...
    def error(self, message: str, **_: Any) -> None: ...

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
    ) -> None: ...
    def counter(
        self,
        *,
        component: str,
        name: str,
        value: int = 1,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None: ...

class NoopMetrics:
    """No-operation metrics sink for testing or when metrics are disabled."""
    def observe(self, **_: Any) -> None: ...
    def counter(self, **_: Any) -> None: ...

# =============================================================================
# Pluggable policies (deadline, breaker, limiter, cache)
# =============================================================================

class DeadlinePolicy(Protocol):
    async def wrap(self, coro, ctx: Optional[OperationContext]) -> Any: ...

class NoopDeadline(DeadlinePolicy):
    async def wrap(self, coro, ctx: Optional[OperationContext]) -> Any:
        return await coro

class CtxDeadline(DeadlinePolicy):
    """Enforce ctx.deadline_ms via asyncio.wait_for; map timeout to DeadlineExceeded."""
    async def wrap(self, coro, ctx: Optional[OperationContext]) -> Any:
        if ctx is None or ctx.deadline_ms is None:
            return await coro
        remaining = OperationContext(deadline_ms=ctx.deadline_ms).remaining_ms()
        if remaining is None:
            return await coro
        if remaining <= 0:
            raise DeadlineExceeded("deadline expired before operation start", code="DEADLINE")
        try:
            return await asyncio.wait_for(coro, timeout=float(remaining) / 1000.0)
        except asyncio.TimeoutError as e:
            raise DeadlineExceeded("operation timed out", code="DEADLINE") from e

class CircuitBreaker(Protocol):
    def allow(self) -> bool: ...
    def on_success(self) -> None: ...
    def on_error(self, err: Exception) -> None: ...

class NoopBreaker(CircuitBreaker):
    def allow(self) -> bool: return True
    def on_success(self) -> None: ...
    def on_error(self, err: Exception) -> None: ...

class SimpleCircuitBreaker(CircuitBreaker):
    """
    Minimal circuit breaker with counts. Not intended to leak closed-source behavior.
    States: closed -> open after error_threshold within window.
            open -> half-open once cool_down passes; first success closes, error re-opens.
    """
    def __init__(self, *, error_threshold: int = 5, cool_down_s: float = 5.0) -> None:
        self._error_threshold = max(1, int(error_threshold))
        self._cool_down_s = max(0.5, float(cool_down_s))
        self._state = "closed"   # "closed" | "open" | "half"
        self._errors = 0
        self._opened_at = 0.0

    def allow(self) -> bool:
        if self._state == "open":
            if (time.monotonic() - self._opened_at) >= self._cool_down_s:
                self._state = "half"
                return True
            return False
        return True

    def on_success(self) -> None:
        if self._state in ("half", "closed"):
            self._state = "closed"
            self._errors = 0

    def on_error(self, err: Exception) -> None:
        self._errors += 1
        if self._errors >= self._error_threshold:
            self._state = "open"
            self._opened_at = time.monotonic()

class RateLimiter(Protocol):
    async def acquire(self) -> None: ...
    def release(self) -> None: ...

class NoopLimiter(RateLimiter):
    async def acquire(self) -> None: ...
    def release(self) -> None: ...

class TokenBucketLimiter(RateLimiter):
    """
    Simple token bucket limiter (coarse-grained). Tokens refill linearly.
    Designed for safety; failures fail-open to avoid breaking callers.
    """
    def __init__(self, *, rate_per_sec: float = 100.0, burst: int = 100) -> None:
        self._rate = max(0.1, float(rate_per_sec))
        self._capacity = max(1, int(burst))
        self._tokens = float(self._capacity)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        delta = now - self._last
        self._last = now
        self._tokens = min(self._capacity, self._tokens + delta * self._rate)

    async def acquire(self) -> None:
        try:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            # brief wait loop until a token is available
            while True:
                await asyncio.sleep(0.01)
                async with self._lock:
                    self._refill()
                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        return
        except Exception:
            # fail-open
            return

    def release(self) -> None:
        try:
            with (self._lock if hasattr(self, "_lock") else None):  # type: ignore
                self._tokens = min(self._capacity, self._tokens + 0.0)
        except Exception:
            pass

class Cache(Protocol):
    async def get(self, key: str) -> Optional[Any]: ...
    async def set(self, key: str, value: Any, ttl_s: int) -> None: ...

class NoopCache(Cache):
    async def get(self, key: str) -> Optional[Any]: return None
    async def set(self, key: str, value: Any, ttl_s: int) -> None: ...

class InMemoryTTLCache(Cache):
    """Tiny async in-memory TTL cache. Safe for read-path hints."""
    def __init__(self) -> None:
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            ent = self._store.get(key)
            if not ent:
                return None
            exp, val = ent
            if time.monotonic() > exp:
                self._store.pop(key, None)
                return None
            return val

    async def set(self, key: str, value: Any, ttl_s: int) -> None:
        ttl = max(1, int(ttl_s))
        async with self._lock:
            self._store[key] = (time.monotonic() + ttl, value)

# =============================================================================
# Capabilities (dynamic discovery for routing and planning)
# =============================================================================

@dataclass(frozen=True)
class GraphCapabilities:
    """
    Describes the capabilities and limitations of a graph adapter implementation.

    Used by routing layers for intelligent database selection, query planning,
    and feature compatibility checking across different graph database backends.

    Attributes:
        server: Backend server identifier (e.g., "neo4j", "janusgraph", "tigergraph")
        version: Backend server version string
        dialects: Supported query dialects ("cypher", "gremlin", "gql", etc.)
        supports_txn: Whether ACID transactions are supported
        supports_schema_ops: Whether schema operations are supported
        max_batch_ops: Maximum operations per batch (None for unlimited)
        retryable_codes: Which error codes are retryable
        rate_limit_unit: Unit for rate limiting ("requests_per_second", "tokens_per_minute")
        max_qps: Maximum queries per second (None for unlimited)
        idempotent_writes: Whether write operations are idempotent with idempotency_key
        supports_multi_tenant: Whether multi-tenant isolation is supported
        supports_streaming: Whether streaming queries are supported
        supports_bulk_ops: Whether bulk operations are supported
        supports_deadline: Whether adapter cooperates with deadline cancellation
    """
    server: str
    version: str
    dialects: Tuple[str, ...] = ("cypher",)
    supports_txn: bool = True
    supports_schema_ops: bool = True
    max_batch_ops: Optional[int] = None
    retryable_codes: Tuple[str, ...] = ()
    rate_limit_unit: str = "requests_per_second"
    max_qps: Optional[int] = None
    idempotent_writes: bool = False
    supports_multi_tenant: bool = False
    supports_streaming: bool = False
    supports_bulk_ops: bool = False
    supports_deadline: bool = True  # NEW: parity with embedding/llm

# =============================================================================
# Helper Classes (utilities for common operations)
# =============================================================================

class BatchOperations:
    """
    Helper methods for constructing batch operations.

    Provides type-safe utilities for creating batch operation dictionaries
    that can be passed to the batch() method.
    """

    @staticmethod
    def create_vertex_op(label: str, props: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Create a vertex creation batch operation.

        Args:
            label: Vertex type/label
            props: Vertex properties as key-value pairs

        Returns:
            Dictionary representing a create_vertex batch operation
        """
        return {"type": "create_vertex", "label": label, "props": dict(props)}

    @staticmethod
    def create_edge_op(label: str, from_id: GraphID, to_id: GraphID, props: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Create an edge creation batch operation.

        Args:
            label: Edge type/label
            from_id: Source vertex identifier
            to_id: Target vertex identifier
            props: Edge properties as key-value pairs

        Returns:
            Dictionary representing a create_edge batch operation
        """
        return {
            "type": "create_edge",
            "label": label,
            "from_id": str(from_id),
            "to_id": str(to_id),
            "props": dict(props)
        }

    @staticmethod
    def query_op(dialect: str, text: str, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a query batch operation.

        Args:
            dialect: Query dialect ("cypher", "gremlin", etc.)
            text: Query text
            params: Query parameters

        Returns:
            Dictionary representing a query batch operation
        """
        return {"type": "query", "dialect": dialect, "text": text, "params": dict(params or {})}

class ProtocolVersion:
    """
    Helper for protocol version compatibility checks.

    Provides semantic version parsing and compatibility checking
    to ensure adapter and caller version alignment.
    """

    def __init__(self, version_str: str):
        """
        Initialize with a semantic version string.

        Args:
            version_str: Semantic version string (e.g., "1.0.0")
        """
        self.major, self.minor, self.patch = map(int, version_str.split('.'))

    def is_compatible_with(self, other: str) -> bool:
        """
        Check if this version is compatible with another version.

        Compatibility rules:
        - Major versions must match exactly
        - Minor version must be >= the other minor version
        - Patch versions don't affect compatibility

        Args:
            other: Other version string to check compatibility with

        Returns:
            bool: True if versions are compatible
        """
        other_ver = ProtocolVersion(other)
        return self.major == other_ver.major and self.minor >= other_ver.minor

# =============================================================================
# Stable Protocol Interface (async, versioned contract)
# =============================================================================

@runtime_checkable
class GraphProtocolV1(Protocol):
    """
    Protocol defining the Graph Protocol V1 interface.

    Implement this protocol to create compatible graph adapters. All methods are async
    and designed for high-concurrency environments. The protocol is runtime-checkable
    for dynamic adapter validation.
    """

    async def capabilities(self) -> GraphCapabilities:
        """
        Get the capabilities of this graph adapter.

        Returns:
            GraphCapabilities: Description of supported features and limitations

        Note:
            This method is async to support dynamic capability discovery in
            distributed systems where capabilities may change or require
            network calls to determine.
        """
        ...

    async def create_vertex(
        self, label: str, props: Mapping[str, Any], *, ctx: Optional[OperationContext] = None
    ) -> GraphID:
        """
        Create a new vertex with the given label and properties.

        Args:
            label: The vertex type/label (must be non-empty string)
            props: Vertex properties as key-value pairs
            ctx: Operation context for tracing, deadlines, and multi-tenancy

        Returns:
            GraphID: The unique identifier for the created vertex

        Raises:
            BadRequest: For invalid arguments or malformed parameters
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            NotSupported: If vertex creation is not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...

    async def create_edge(
        self, label: str, from_id: GraphID, to_id: GraphID, props: Mapping[str, Any], *, ctx: Optional[OperationContext] = None
    ) -> GraphID:
        """
        Create a new edge between two vertices.

        Args:
            label: The edge type/label (must be non-empty string)
            from_id: Source vertex identifier (must exist)
            to_id: Target vertex identifier (must exist)
            props: Edge properties as key-value pairs
            ctx: Operation context for tracing, deadlines, and multi-tenancy

        Returns:
            GraphID: The unique identifier for the created edge

        Raises:
            BadRequest: For invalid arguments or non-existent vertices
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            NotSupported: If edge creation is not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...

    async def delete_vertex(self, vertex_id: GraphID, *, ctx: Optional[OperationContext] = None) -> None:
        """
        Delete a vertex by its identifier.

        Args:
            vertex_id: Vertex identifier to delete
            ctx: Operation context for tracing and multi-tenancy

        Raises:
            BadRequest: For invalid vertex identifier
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            NotSupported: If vertex deletion is not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...

    async def delete_edge(self, edge_id: GraphID, *, ctx: Optional[OperationContext] = None) -> None:
        """
        Delete an edge by its identifier.

        Args:
            edge_id: Edge identifier to delete
            ctx: Operation context for tracing and multi-tenancy

        Raises:
            BadRequest: For invalid edge identifier
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            NotSupported: If edge deletion is not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...

    async def query(
        self,
        *,
        dialect: str,
        text: str,
        params: Optional[Mapping[str, Any]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> List[Mapping[str, Any]]:
        """
        Execute a query and return all results.

        Args:
            dialect: Query dialect ("cypher", "gremlin", "gql", etc.)
            text: Query text in the specified dialect
            params: Query parameters as key-value pairs
            ctx: Operation context for tracing, deadlines, and multi-tenancy

        Returns:
            List of result mappings. Each result is a dictionary representing
            a node, edge, or projection from the query.

        Raises:
            BadRequest: For invalid query, dialect, or parameters
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            NotSupported: If the dialect or query type is not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...

    async def stream_query(
        self,
        *,
        dialect: str,
        text: str,
        params: Optional[Mapping[str, Any]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[Mapping[str, Any]]:
        """
        Execute a query and stream results as they arrive.

        Args:
            dialect: Query dialect ("cypher", "gremlin", "gql", etc.)
            text: Query text in the specified dialect
            params: Query parameters as key-value pairs
            ctx: Operation context for tracing, deadlines, and multi-tenancy

        Yields:
            Result mappings as they become available. Each result is a dictionary
            representing a node, edge, or projection from the query.

        Raises:
            BadRequest: For invalid query, dialect, or parameters
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            NotSupported: If streaming or the dialect is not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...

    async def bulk_vertices(
        self, vertices: Iterable[Tuple[str, Mapping[str, Any]]], *, ctx: Optional[OperationContext] = None
    ) -> List[GraphID]:
        """
        Create multiple vertices in a single operation.

        Args:
            vertices: Iterable of (label, properties) tuples for vertices to create
            ctx: Operation context for tracing, deadlines, and multi-tenancy

        Returns:
            List of GraphIDs for the created vertices in the same order as input

        Raises:
            BadRequest: For invalid labels or properties
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            NotSupported: If bulk operations are not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...

    async def batch(
        self,
        ops: Iterable[Mapping[str, Any]],
        *,
        ctx: Optional[OperationContext] = None,
    ) -> List[Mapping[str, Any]]:
        """
        Execute multiple operations in a single batch.

        Args:
            ops: Iterable of operation dictionaries (use BatchOperations helpers)
            ctx: Operation context for tracing, deadlines, and multi-tenancy

        Returns:
            List of results corresponding to each operation in the input batch

        Raises:
            BadRequest: For invalid operations or batch size exceeded
            AuthError: For authentication or authorization failures
            ResourceExhausted: For quota or rate limit exceeded
            NotSupported: If batching is not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...

    async def get_schema(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """
        Get the graph schema information.

        Args:
            ctx: Operation context for tracing, deadlines, and multi-tenancy

        Returns:
            Dictionary containing schema information (structure varies by backend)

        Raises:
            AuthError: For authentication or authorization failures
            NotSupported: If schema operations are not supported
            TransientNetwork: For retryable network failures
            Unavailable: For service unavailable errors
        """
        ...

    async def health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """
        Check the health status of the graph backend.

        Args:
            ctx: Operation context for tracing and multi-tenancy

        Returns:
            Dictionary with health information including:
            - status: Overall status (HealthStatus constants)
            - read_only: Whether backend is in read-only mode
            - degraded: Whether backend is degraded but operational
            - version: Backend version information
            - server: Backend server identifier
            - details: Additional backend-specific health details

        Raises:
            Unavailable: If the health check fails or backend is unreachable
        """
        ...

# =============================================================================
# Base Instrumented Adapter (validation, metrics, error handling)
# =============================================================================

class _Base:
    """
    Base functionality shared by graph adapters.

    Provides common validation, metrics recording, utility methods, and
    SIEM-safe observability. This class contains all shared infrastructure
    without implementing the protocol methods.
    """
    _component = "graph"
    _noop_log_sink = NoopLogSink()

    def __init__(
        self,
        *,
        metrics: Optional[MetricsSink] = None,
        logs: Optional[LogSink] = None,
        mode: str = "thin",
        deadline_policy: Optional[DeadlinePolicy] = None,
        breaker: Optional[CircuitBreaker] = None,
        limiter: Optional[RateLimiter] = None,
        cache: Optional[Cache] = None,
    ) -> None:
        """
        Initialize the graph adapter with observability instrumentation and optional policies.

        Args:
            metrics: Metrics sink for operational monitoring. Uses NoopMetrics if None.
            logs: Log sink for structured logging. Uses NoopLogSink if None.
            mode: "thin" (default) or "standalone" to toggle infra hooks.
            deadline_policy: Optional deadline policy to enforce ctx.deadline_ms.
            breaker: Optional circuit breaker.
            limiter: Optional rate limiter.
            cache: Optional async cache for safe read paths.
        """
        self._metrics: MetricsSink = metrics or NoopMetrics()
        self._logs: LogSink = logs or self._noop_log_sink

        # Mode wiring with explicit defaults; caller's overrides always win.
        m = (mode or "thin").lower().strip()
        self._mode = m if m in {"thin", "standalone"} else "thin"

        if self._mode == "thin":
            self._deadline: DeadlinePolicy = deadline_policy or NoopDeadline()
            self._breaker: CircuitBreaker = breaker or NoopBreaker()
            self._limiter: RateLimiter = limiter or NoopLimiter()
            self._cache: Cache = cache or NoopCache()
        else:
            # standalone defaults are conservative, and only for dev/light production
            self._deadline = deadline_policy or CtxDeadline()
            self._breaker = breaker or SimpleCircuitBreaker(error_threshold=5, cool_down_s=5.0)
            self._limiter = limiter or TokenBucketLimiter(rate_per_sec=100.0, burst=100)
            self._cache = cache or InMemoryTTLCache()
            if isinstance(self._metrics, NoopMetrics):
                LOG.warning("Using standalone mode without metrics - consider providing a metrics sink for production use")

    # ---------------------- validation helpers ----------------------

    @staticmethod
    def _require_non_empty(name: str, value: str) -> None:
        """
        Validate that a string value is non-empty.

        Args:
            name: Parameter name for error messages
            value: Value to validate

        Raises:
            BadRequest: If value is empty or not a string
        """
        if not isinstance(value, str) or not value.strip():
            raise BadRequest(f"{name} must be a non-empty string")

    def _validate_dialect(self, dialect: str) -> None:
        """
        Validate that a dialect is known and supported.

        Args:
            dialect: Dialect to validate

        Raises:
            BadRequest: If dialect is empty
            NotSupported: If dialect is not in KNOWN_DIALECTS
        """
        self._require_non_empty("dialect", dialect)
        if dialect not in KNOWN_DIALECTS:
            raise NotSupported(
                f"Dialect '{dialect}' not supported. Known dialects: {KNOWN_DIALECTS}",
                dialect=dialect
            )

    def _validate_properties(self, props: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Convert and validate property types for graph compatibility.

        Args:
            props: Properties to validate and convert

        Returns:
            Validated properties dictionary with string keys
        """
        if props is None:
            return {}
        return {str(k): v for k, v in props.items()}

    @staticmethod
    def _bucket_ms(ms: Optional[int]) -> Optional[str]:
        """
        Bucket milliseconds for metrics categorization.

        Args:
            ms: Milliseconds to bucket

        Returns:
            Bucketed time range string or None
        """
        if ms is None or ms < 0: return None
        if ms < 1000: return "<1s"
        if ms < 5000: return "<5s"
        if ms < 15000: return "<15s"
        if ms < 60000: return "<60s"
        return ">=60s"

    @staticmethod
    def _tenant_hash(tenant: Optional[str]) -> Optional[str]:
        """
        Create privacy-preserving hash of tenant identifier for metrics.

        Args:
            tenant: Raw tenant identifier

        Returns:
            Hashed tenant identifier (first 12 chars of SHA256) or None
        """
        if not tenant: return None
        return hashlib.sha256(tenant.encode("utf-8")).hexdigest()[:12]

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

        Never exposes raw tenant identifiers in metrics. Safe for SIEM systems.

        Args:
            op: Operation name
            t0: Start time from time.monotonic()
            ok: Whether operation succeeded
            code: Status code for metrics
            ctx: Operation context for tenant and deadline information
            **extra: Additional metric dimensions
        """
        try:
            dt_ms = (time.monotonic() - t0) * 1000.0
            x = dict(extra or {})
            if ctx:
                x["deadline_bucket"] = self._bucket_ms(ctx.deadline_ms)
                x["tenant"] = self._tenant_hash(ctx.tenant)
            self._metrics.observe(
                component=self._component, op=op, ms=dt_ms, ok=ok, code=code, extra=x or None
            )
            # light counters for visibility
            name = f"{op}_total" if ok else f"{op}_errors"
            self._metrics.counter(component=self._component, name=name, value=1)
        except Exception:
            # Never let metrics recording break the operation
            pass

    # ---------------------- policy helpers ----------------------

    def _preflight_deadline(self, ctx: Optional[OperationContext]) -> None:
        if ctx is not None and ctx.is_timed_out:
            raise DeadlineExceeded("deadline expired before operation start", code="DEADLINE")

    async def _apply_deadline(self, coro, ctx: Optional[OperationContext]) -> Any:
        try:
            return await self._deadline.wrap(coro, ctx)
        except DeadlineExceeded:
            raise
        except asyncio.TimeoutError as e:
            raise DeadlineExceeded("operation timed out", code="DEADLINE") from e

    async def _with_gates(self, op: str, ctx: Optional[OperationContext], fn: Callable[[], Any]) -> Any:
        """
        Apply breaker + limiter admission and ensure limiter release & breaker bookkeeping.
        """
        self._preflight_deadline(ctx)
        if not self._breaker.allow():
            raise Unavailable("circuit open", code="CIRCUIT_OPEN")
        await self._limiter.acquire()
        try:
            return await self._apply_deadline(fn(), ctx)
        except Exception as e:
            try:
                self._breaker.on_error(e)
            finally:
                raise
        finally:
            try:
                self._limiter.release()
            except Exception:
                pass

    # ---------------------- caching helpers ----------------------

    @staticmethod
    def _stable_json(obj: Any) -> str:
        # A stable representation for cache keys without importing json to avoid heavy deps:
        # We will use a simple deterministic flattener.
        if obj is None:
            return "null"
        if isinstance(obj, (str, int, float, bool)):
            return repr(obj)
        if isinstance(obj, Mapping):
            items = ",".join(f"{repr(str(k))}:{_Base._stable_json(v)}" for k, v in sorted(obj.items(), key=lambda kv: str(kv[0])))
            return "{" + items + "}"
        if isinstance(obj, (list, tuple)):
            return "[" + ",".join(_Base._stable_json(v) for v in obj) + "]"
        return repr(str(obj))

    def _query_cache_key(
        self,
        *,
        server: Optional[str],
        dialect: str,
        text: str,
        params: Mapping[str, Any],
        tenant_hash: Optional[str],
    ) -> str:
        base = {
            "server": server or "unknown",
            "dialect": dialect,
            "text": text,
            "params": params,
            "tenant": tenant_hash or "anon",
            "version": GRAPH_PROTOCOL_VERSION,
        }
        s = self._stable_json(base).encode("utf-8")
        return "graph:query:" + hashlib.blake2b(s, digest_size=20).hexdigest()

    def _schema_cache_key(self, server: Optional[str], tenant_hash: Optional[str]) -> str:
        s = f"{server or 'unknown'}:{tenant_hash or 'anon'}:{GRAPH_PROTOCOL_VERSION}".encode("utf-8")
        return "graph:schema:" + hashlib.blake2b(s, digest_size=20).hexdigest()

    def _caps_cache_key(self) -> str:
        return "graph:caps:" + GRAPH_PROTOCOL_VERSION

class BaseGraphAdapter(_Base, GraphProtocolV1):
    """
    Base class for implementing Graph Protocol V1 adapters.

    Provides common validation, metrics instrumentation, error handling, and
    SIEM-safe observability. Implementers should override the `_do_*` methods
    to provide backend-specific functionality while getting production-ready
    infrastructure for free.

    Example:
        class Neo4jAdapter(BaseGraphAdapter):
            async def _do_create_vertex(self, label: str, props: Dict[str, Any], *, ctx: Optional[OperationContext]) -> GraphID:
                # Neo4j-specific implementation using driver sessions
                async with self._driver.session() as session:
                    result = await session.run(
                        "CREATE (v:$label) SET v = $props RETURN id(v) as id",
                        label=label, props=props
                    )
                    record = await result.single()
                    return GraphID(str(record["id"]))
    """

    # ---------------------- public API (with hardening) ----------------------

    async def capabilities(self) -> GraphCapabilities:
        """Get the capabilities of this graph adapter (with optional caching)."""
        t0 = time.monotonic()
        try:
            async def _call():
                # Read-path cache only in standalone
                if self._mode == "standalone":
                    key = self._caps_cache_key()
                    cached = await self._cache.get(key)
                    if cached is not None:
                        self._metrics.counter(component=self._component, name="cache_hits", value=1)
                        return cached
                    caps = await self._do_capabilities()
                    await self._cache.set(key, caps, ttl_s=30)
                    return caps
                # thin: no cache
                return await self._do_capabilities()

            caps = await _call()
            self._record("capabilities", t0, True, ctx=None)
            return caps
        except AdapterError as e:
            self._record("capabilities", t0, False, code=type(e).__name__, ctx=None)
            raise

    async def create_vertex(
        self, label: str, props: Mapping[str, Any], *, ctx: Optional[OperationContext] = None
    ) -> GraphID:
        """
        Create a new vertex with the given label and properties.

        See GraphProtocolV1.create_vertex for full documentation.
        """
        self._require_non_empty("label", label)
        validated_props = self._validate_properties(props)
        t0 = time.monotonic()
        try:
            # gates + deadline (no caching for mutations)
            async def _fn():
                return await self._do_create_vertex(label, validated_props, ctx=ctx)

            self._preflight_deadline(ctx)
            if not self._breaker.allow():
                raise Unavailable("circuit open", code="CIRCUIT_OPEN")
            await self._limiter.acquire()
            try:
                vid = await self._apply_deadline(_fn(), ctx)
                self._breaker.on_success()
            except Exception as e:
                self._breaker.on_error(e)
                raise
            finally:
                self._limiter.release()

            self._record("create_vertex", t0, True, ctx=ctx)
            return GraphID(str(vid))  # Explicit type conversion
        except AdapterError as e:
            self._record("create_vertex", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    async def create_edge(
        self, label: str, from_id: GraphID, to_id: GraphID, props: Mapping[str, Any], *, ctx: Optional[OperationContext] = None
    ) -> GraphID:
        """
        Create a new edge between two vertices.

        See GraphProtocolV1.create_edge for full documentation.
        """
        for n, v in (("label", label), ("from_id", from_id), ("to_id", to_id)):
            self._require_non_empty(n, str(v))
        validated_props = self._validate_properties(props)
        t0 = time.monotonic()
        try:
            async def _fn():
                return await self._do_create_edge(label, str(from_id), str(to_id), validated_props, ctx=ctx)

            self._preflight_deadline(ctx)
            if not self._breaker.allow():
                raise Unavailable("circuit open", code="CIRCUIT_OPEN")
            await self._limiter.acquire()
            try:
                eid = await self._apply_deadline(_fn(), ctx)
                self._breaker.on_success()
            except Exception as e:
                self._breaker.on_error(e)
                raise
            finally:
                self._limiter.release()

            self._record("create_edge", t0, True, ctx=ctx)
            return GraphID(str(eid))  # Explicit type conversion
        except AdapterError as e:
            self._record("create_edge", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    async def delete_vertex(self, vertex_id: GraphID, *, ctx: Optional[OperationContext] = None) -> None:
        """Delete a vertex by its identifier."""
        self._require_non_empty("vertex_id", str(vertex_id))
        t0 = time.monotonic()
        try:
            async def _fn():
                return await self._do_delete_vertex(str(vertex_id), ctx=ctx)

            self._preflight_deadline(ctx)
            if not self._breaker.allow():
                raise Unavailable("circuit open", code="CIRCUIT_OPEN")
            await self._limiter.acquire()
            try:
                await self._apply_deadline(_fn(), ctx)
                self._breaker.on_success()
            except Exception as e:
                self._breaker.on_error(e)
                raise
            finally:
                self._limiter.release()

            self._record("delete_vertex", t0, True, ctx=ctx)
        except AdapterError as e:
            self._record("delete_vertex", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    async def delete_edge(self, edge_id: GraphID, *, ctx: Optional[OperationContext] = None) -> None:
        """Delete an edge by its identifier."""
        self._require_non_empty("edge_id", str(edge_id))
        t0 = time.monotonic()
        try:
            async def _fn():
                return await self._do_delete_edge(str(edge_id), ctx=ctx)

            self._preflight_deadline(ctx)
            if not self._breaker.allow():
                raise Unavailable("circuit open", code="CIRCUIT_OPEN")
            await self._limiter.acquire()
            try:
                await self._apply_deadline(_fn(), ctx)
                self._breaker.on_success()
            except Exception as e:
                self._breaker.on_error(e)
                raise
            finally:
                self._limiter.release()

            self._record("delete_edge", t0, True, ctx=ctx)
        except AdapterError as e:
            self._record("delete_edge", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    async def query(
        self,
        *,
        dialect: str,
        text: str,
        params: Optional[Mapping[str, Any]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> List[Mapping[str, Any]]:
        """Execute a query and return all results."""
        self._validate_dialect(dialect)
        self._require_non_empty("text", text)
        validated_params = self._validate_properties(params or {})
        t0 = time.monotonic()
        try:
            # Capabilities gating (dialect & optional batch limits for internal ops)
            caps = await self._do_capabilities()
            if dialect not in caps.dialects:
                raise NotSupported(f"Dialect '{dialect}' not supported by backend", dialect=dialect)

            tenant_hash = self._tenant_hash(ctx.tenant) if ctx else None
            server = getattr(caps, "server", None)
            cache_key = self._query_cache_key(server=server, dialect=dialect, text=text, params=validated_params, tenant_hash=tenant_hash)

            async def _fn():
                # Optional cache in standalone
                if self._mode == "standalone":
                    cached = await self._cache.get(cache_key)
                    if cached is not None:
                        self._metrics.counter(component=self._component, name="cache_hits", value=1)
                        return cached
                    res = await self._do_query(dialect=dialect, text=text, params=validated_params, ctx=ctx)
                    await self._cache.set(cache_key, res, ttl_s=60)
                    return res
                return await self._do_query(dialect=dialect, text=text, params=validated_params, ctx=ctx)

            self._preflight_deadline(ctx)
            if not self._breaker.allow():
                raise Unavailable("circuit open", code="CIRCUIT_OPEN")
            await self._limiter.acquire()
            try:
                res = await self._apply_deadline(_fn(), ctx)
                self._breaker.on_success()
            except Exception as e:
                self._breaker.on_error(e)
                raise
            finally:
                self._limiter.release()

            self._record("query", t0, True, ctx=ctx, dialect=dialect, rows=len(res))
            return res
        except AdapterError as e:
            self._record("query", t0, False, code=type(e).__name__, ctx=ctx, dialect=dialect)
            raise

    async def stream_query(
        self,
        *,
        dialect: str,
        text: str,
        params: Optional[Mapping[str, Any]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[Mapping[str, Any]]:
        """Execute a query and stream results as they arrive."""
        self._validate_dialect(dialect)
        self._require_non_empty("text", text)
        validated_params = self._validate_properties(params or {})
        t0 = time.monotonic()
        ok = False
        try:
            caps = await self._do_capabilities()
            if dialect not in caps.dialects:
                raise NotSupported(f"Dialect '{dialect}' not supported by backend", dialect=dialect)

            self._preflight_deadline(ctx)
            if not self._breaker.allow():
                raise Unavailable("circuit open", code="CIRCUIT_OPEN")
            await self._limiter.acquire()

            # Wrap the underlying async generator with deadline checks.
            async def _gen():
                # Note: we intentionally don't cache streams.
                agen = self._do_stream_query(dialect=dialect, text=text, params=validated_params, ctx=ctx)
                try:
                    async for row in agen:
                        # Per-chunk deadline check:
                        if ctx and ctx.is_timed_out:
                            raise DeadlineExceeded("stream exceeded deadline", code="DEADLINE")
                        yield row
                finally:
                    # Ensure underlying generator is closed if caller stops early.
                    try:
                        await agen.aclose()  # type: ignore[attr-defined]
                    except Exception:
                        pass

            count = 0
            async for row in self._apply_deadline(_gen(), ctx):  # type: ignore[arg-type]
                count += 1
                yield row
            ok = True
            self._breaker.on_success()
            self._record("stream_query", t0, True, ctx=ctx, dialect=dialect, rows=count)
        except AdapterError as e:
            self._breaker.on_error(e)
            self._record("stream_query", t0, False, code=type(e).__name__, ctx=ctx, dialect=dialect)
            raise
        except Exception as e:
            self._breaker.on_error(e)
            self._record("stream_query", t0, False, code=type(e).__name__, ctx=ctx, dialect=dialect)
            raise
        finally:
            try:
                self._limiter.release()
            except Exception:
                pass
            if not ok:
                # no-op, final accounting handled above
                pass

    async def bulk_vertices(
        self, vertices: Iterable[Tuple[str, Mapping[str, Any]]], *, ctx: Optional[OperationContext] = None
    ) -> List[GraphID]:
        """Create multiple vertices in a single operation."""
        vertex_list = list(vertices)
        for label, props in vertex_list:
            self._require_non_empty("label", label)

        # Capabilities gating for batch size if provided
        caps = await self._do_capabilities()
        if caps.max_batch_ops is not None and len(vertex_list) > int(caps.max_batch_ops):
            raise BadRequest(
                f"batch size {len(vertex_list)} exceeds max_batch_ops {caps.max_batch_ops}",
                suggested_batch_reduction=100 * (len(vertex_list) - int(caps.max_batch_ops)) // len(vertex_list),
                details={"max_batch_ops": int(caps.max_batch_ops)},
                operation="bulk_vertices",
            )

        validated_vertices = [(label, self._validate_properties(props)) for label, props in vertex_list]
        t0 = time.monotonic()
        try:
            async def _fn():
                return await self._do_bulk_vertices(validated_vertices, ctx=ctx)

            self._preflight_deadline(ctx)
            if not self._breaker.allow():
                raise Unavailable("circuit open", code="CIRCUIT_OPEN")
            await self._limiter.acquire()
            try:
                ids = await self._apply_deadline(_fn(), ctx)
                self._breaker.on_success()
            except Exception as e:
                self._breaker.on_error(e)
                raise
            finally:
                self._limiter.release()

            self._record("bulk_vertices", t0, True, ctx=ctx, count=len(vertex_list))
            return [GraphID(str(id)) for id in ids]  # Explicit type conversion
        except AdapterError as e:
            self._record("bulk_vertices", t0, False, code=type(e).__name__, ctx=ctx, count=len(vertex_list))
            raise

    async def batch(
        self,
        ops: Iterable[Mapping[str, Any]],
        *,
        ctx: Optional[OperationContext] = None,
    ) -> List[Mapping[str, Any]]:
        """Execute multiple operations in a single batch."""
        op_list = list(ops)

        caps = await self._do_capabilities()
        if caps.max_batch_ops is not None and len(op_list) > int(caps.max_batch_ops):
            raise BadRequest(
                f"batch size {len(op_list)} exceeds max_batch_ops {caps.max_batch_ops}",
                suggested_batch_reduction=100 * (len(op_list) - int(caps.max_batch_ops)) // len(op_list),
                details={"max_batch_ops": int(caps.max_batch_ops)},
                operation="batch",
            )

        t0 = time.monotonic()
        try:
            async def _fn():
                return await self._do_batch(op_list, ctx=ctx)

            self._preflight_deadline(ctx)
            if not self._breaker.allow():
                raise Unavailable("circuit open", code="CIRCUIT_OPEN")
            await self._limiter.acquire()
            try:
                res = await self._apply_deadline(_fn(), ctx)
                self._breaker.on_success()
            except Exception as e:
                self._breaker.on_error(e)
                raise
            finally:
                self._limiter.release()

            self._record("batch", t0, True, ctx=ctx, ops=len(op_list))
            return res
        except AdapterError as e:
            self._record("batch", t0, False, code=type(e).__name__, ctx=ctx, ops=len(op_list))
            raise

    async def get_schema(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """Get the graph schema information."""
        t0 = time.monotonic()
        try:
            caps = await self._do_capabilities()
            tenant_hash = self._tenant_hash(ctx.tenant) if ctx else None
            key = self._schema_cache_key(getattr(caps, "server", None), tenant_hash)

            async def _fn():
                if self._mode == "standalone":
                    cached = await self._cache.get(key)
                    if cached is not None:
                        self._metrics.counter(component=self._component, name="cache_hits", value=1)
                        return cached
                    schema = await self._do_get_schema(ctx=ctx)
                    schema = dict(schema)
                    await self._cache.set(key, schema, ttl_s=45)
                    return schema
                return dict(await self._do_get_schema(ctx=ctx))

            self._preflight_deadline(ctx)
            if not self._breaker.allow():
                raise Unavailable("circuit open", code="CIRCUIT_OPEN")
            await self._limiter.acquire()
            try:
                schema = await self._apply_deadline(_fn(), ctx)
                self._breaker.on_success()
            except Exception as e:
                self._breaker.on_error(e)
                raise
            finally:
                self._limiter.release()

            self._record("get_schema", t0, True, ctx=ctx)
            return dict(schema)
        except AdapterError as e:
            self._record("get_schema", t0, False, code=type(e).__name__, ctx=ctx)
            raise

    async def health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        """Check the health status of the graph backend."""
        t0 = time.monotonic()
        try:
            async def _fn():
                return await self._do_health(ctx=ctx)

            # health still respects deadlines
            h = await self._apply_deadline(_fn(), ctx)
            self._record("health", t0, True, ctx=ctx)
            return {
                "status": h.get("status", HealthStatus.OK),
                "read_only": bool(h.get("read_only", False)),
                "degraded": bool(h.get("degraded", False)),
                "version": str(h.get("version", "")),
                "server": str(h.get("server", "")),
                "details": dict(h.get("details", {})),
            }
        except AdapterError as e:
            self._record("health", t0, False, code=type(e).__name__, ctx=ctx)
            raise
        except Exception as e:
            self._record("health", t0, False, code=type(e).__name__, ctx=ctx)
            # Normalize unexpected exceptions as Unavailable for callers
            raise Unavailable("health check failed") from e

    # ============ ABSTRACT METHODS TO BE IMPLEMENTED ============

    async def _do_capabilities(self) -> GraphCapabilities:
        """Implement to return adapter-specific capabilities."""
        raise NotImplementedError

    async def _do_create_vertex(self, label: str, props: Dict[str, Any], *, ctx: Optional[OperationContext]) -> GraphID:
        """Implement vertex creation with validated inputs."""
        raise NotImplementedError

    async def _do_create_edge(
        self, label: str, from_id: str, to_id: str, props: Dict[str, Any], *, ctx: Optional[OperationContext]
    ) -> GraphID:
        """Implement edge creation with validated inputs."""
        raise NotImplementedError

    async def _do_delete_vertex(self, vertex_id: str, *, ctx: Optional[OperationContext]) -> None:
        """Implement vertex deletion."""
        raise NotImplementedError

    async def _do_delete_edge(self, edge_id: str, *, ctx: Optional[OperationContext]) -> None:
        """Implement edge deletion."""
        raise NotImplementedError

    async def _do_query(
        self, *, dialect: str, text: str, params: Mapping[str, Any], ctx: Optional[OperationContext]
    ) -> List[Mapping[str, Any]]:
        """Implement query execution."""
        raise NotImplementedError

    async def _do_stream_query(
        self, *, dialect: str, text: str, params: Mapping[str, Any], ctx: Optional[OperationContext]
    ) -> AsyncIterator[Mapping[str, Any]]:
        """Implement streaming query execution."""
        raise NotImplementedError

    async def _do_bulk_vertices(
        self, vertices: List[Tuple[str, Mapping[str, Any]]], *, ctx: Optional[OperationContext]
    ) -> List[GraphID]:
        """Implement bulk vertex creation."""
        raise NotImplementedError

    async def _do_batch(self, ops: List[Mapping[str, Any]], *, ctx: Optional[OperationContext]) -> List[Mapping[str, Any]]:
        """Implement batch operation execution."""
        raise NotImplementedError

    async def _do_get_schema(self, *, ctx: Optional[OperationContext]) -> Dict[str, Any]:
        """Implement schema retrieval."""
        raise NotImplementedError

    async def _do_health(self, *, ctx: Optional[OperationContext]) -> Dict[str, Any]:
        """Implement health check."""
        raise NotImplementedError

__all__ = [
    "GRAPH_PROTOCOL_VERSION", "KNOWN_DIALECTS", "GraphID", "AdapterError",
    "BadRequest", "AuthError", "ResourceExhausted", "TransientNetwork",
    "Unavailable", "NotSupported", "DeadlineExceeded", "OperationContext", "LogSink",
    "NoopLogSink", "MetricsSink", "NoopMetrics", "GraphCapabilities",
    "GraphProtocolV1", "BaseGraphAdapter", "BatchOperations", "HealthStatus",
    "ProtocolVersion", "DeadlinePolicy", "NoopDeadline", "CtxDeadline",
    "CircuitBreaker", "NoopBreaker", "SimpleCircuitBreaker",
    "RateLimiter", "NoopLimiter", "TokenBucketLimiter",
    "Cache", "NoopCache", "InMemoryTTLCache",
]
