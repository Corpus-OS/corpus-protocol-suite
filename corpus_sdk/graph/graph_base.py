# adapter_sdk/graph_base.py
# SPDX-License-Identifier: Apache-2.0
"""
Adapter SDK — Graph Protocol V1.0

Purpose
-------
A stable, vendor-neutral API for graph query and mutation operations, with:

- Structured, normalized error taxonomy (SIEM-safe, machine-actionable)
- Async-first, high-concurrency friendly interface
- Backpressure: circuit breaker + rate limiting + deadline propagation
- Optional read-path caching and capability discovery
- Wire-level handler for canonical JSON envelopes (transport-agnostic)

Design Philosophy
-----------------
- Minimal core surface: query, stream_query, upsert/delete nodes & edges, namespaces.
- Keep existing power-user APIs: streaming, bulk vertices, batch operations, schema introspection.
- No vendor- or dialect-specific helpers in the base.
- Async-only: suitable for servers and routers; sync wrappers live elsewhere.
- DRY infra: shared gate wrapper for breaker / limiter / deadlines / metrics.

Deliberate Non-Goals
--------------------
- No routing, retries, hedging, or policy enforcement.
- No automatic dialect rewriting (Cypher ⇄ Gremlin, etc.).
- No client-side schema management or migrations.
- No embedding/LLM integration (belongs in higher layers).

Those behaviors live in your control-plane / router layers.

Mode Strategy
-------------
As with LLM and Vector:

mode: "thin" (default)
    - For composition under an external manager/router.
    - All policies default to no-op: no caching, no breaker, no rate limiter.
    - Use when your infra already handles concurrency and resilience.

mode: "standalone"
    - For direct use in services.
    - Enables:
        * SimpleDeadline (ctx.deadline_ms)
        * SimpleCircuitBreaker
        * InMemoryTTLCache for read paths
          (query / capabilities / bulk_vertices / get_schema)
        * SimpleTokenBucketLimiter
    - Intended for development / light production; NOT a full distributed control plane.

Versioning
----------
Follow SemVer against GRAPH_PROTOCOL_VERSION (wire & type contract).

- Patch (x.y.Z): Documentation and strictly non-breaking edits.
- Minor (x.Y.z): Additive fields, methods, or capabilities.
- Major (X.y.z): Breaking changes only (avoid in base; prefer additive).

Wire Contract (Canonical Interface)
-----------------------------------
The canonical interop surface is JSON envelopes; this module provides:

- GraphProtocolV1 / BaseGraphAdapter: typed in-process contract
- WireGraphHandler: reference transport-agnostic wire adapter

Envelopes:

    Request:
        {
            "op": "graph.<operation>",
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

    Success:
        {
            "ok": true,
            "code": "OK",
            "ms": <float>,          # elapsed milliseconds (best-effort)
            "result": { ... }       # operation-specific payload
        }

    Error:
        {
            "ok": false,
            "code": "<UPPER_SNAKE_CASE>",
            "error": "<ErrorClassName>",
            "message": "<human readable>",
            "retry_after_ms": <int|null>,
            "details": { ... } | null,
            "ms": <float>
        }

Streaming queries (graph.stream_query) follow the same pattern as llm.stream:

    - Request: single envelope with op="graph.stream_query"
    - Response: stream of { ok, code, ms, chunk: { ... } } or terminal error envelope.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    NewType,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

LOG = logging.getLogger(__name__)

GRAPH_PROTOCOL_VERSION = "1.0.0"
GRAPH_PROTOCOL_ID = "graph/v1.0"

# =============================================================================
# Core ID / Model Types
# =============================================================================

GraphID = NewType("GraphID", str)
"""Opaque identifier for nodes/edges (backends may layer structure on top)."""


@dataclass(frozen=True)
class Node:
    """
    Graph node representation.

    Attributes:
        id: Stable node identifier (GraphID or backend-native ID)
        labels: Optional set/list of labels / types / kinds
        properties: Arbitrary JSON-serializable property map
        namespace: Optional logical graph / tenant / dataset
    """
    id: GraphID
    labels: Tuple[str, ...] = ()
    properties: Mapping[str, Any] = None
    namespace: Optional[str] = None

    def __post_init__(self) -> None:
        if self.properties is None:
            object.__setattr__(self, "properties", {})


@dataclass(frozen=True)
class Edge:
    """
    Graph edge representation.

    Attributes:
        id: Stable edge identifier (GraphID or backend-native ID)
        src: Source node ID
        dst: Target node ID
        label: Relationship / predicate / type
        properties: Arbitrary JSON-serializable property map
        namespace: Optional logical graph / tenant / dataset
    """
    id: GraphID
    src: GraphID
    dst: GraphID
    label: str
    properties: Mapping[str, Any] = None
    namespace: Optional[str] = None

    def __post_init__(self) -> None:
        if self.properties is None:
            object.__setattr__(self, "properties", {})


@dataclass(frozen=True)
class GraphQuerySpec:
    """
    Specification for graph queries.

    Mirrors the wire shape to keep routing trivial.

    Attributes:
        text: The query text (e.g., Cypher/Gremlin/GQL/SQL/JSON-dsl/etc).
        dialect: Optional query dialect identifier:
                 E.g. "cypher", "gremlin", "gql", "sql", "sparql", "native".
        params: Optional parameter map for bind variables.
        namespace: Logical graph / dataset / tenant this query targets.
        timeout_ms: Optional query-level timeout hint (advisory).
        stream: If True, caller prefers streaming (used by routers and adapters).
    """
    text: str
    dialect: Optional[str] = None
    params: Mapping[str, Any] = None
    namespace: Optional[str] = None
    timeout_ms: Optional[int] = None
    stream: bool = False

    def __post_init__(self) -> None:
        if self.params is None:
            object.__setattr__(self, "params", {})


@dataclass(frozen=True)
class UpsertNodesSpec:
    """
    Batch upsert specification for nodes.

    Attributes:
        nodes: Node objects to upsert
        namespace: Optional override namespace (per-node namespace wins if set)
    """
    nodes: List[Node]
    namespace: Optional[str] = None


@dataclass(frozen=True)
class UpsertEdgesSpec:
    """
    Batch upsert specification for edges.

    Attributes:
        edges: Edge objects to upsert
        namespace: Optional override namespace (per-edge namespace wins if set)
    """
    edges: List[Edge]
    namespace: Optional[str] = None


@dataclass(frozen=True)
class DeleteNodesSpec:
    """
    Batch delete specification for nodes.

    Attributes:
        ids: Node IDs to delete
        namespace: Optional namespace / graph
        filter: Optional property/label filter for bulk deletes
    """
    ids: List[GraphID]
    namespace: Optional[str] = None
    filter: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class DeleteEdgesSpec:
    """
    Batch delete specification for edges.

    Attributes:
        ids: Edge IDs to delete
        namespace: Optional namespace / graph
        filter: Optional property filter for bulk deletes
    """
    ids: List[GraphID]
    namespace: Optional[str] = None
    filter: Optional[Mapping[str, Any]] = None


# ---- Bulk / Batch specs (restored) ------------------------------------------

@dataclass(frozen=True)
class BulkVerticesSpec:
    """
    Specification for scanning / listing vertices in bulk.

    Useful for offline sync, backfills, and migrations.

    Attributes:
        namespace: Namespace / graph to scan (None = adapter default).
        limit: Max number of nodes to return in this page.
        cursor: Opaque cursor for pagination (adapter-defined).
        filter: Optional metadata/label filter (adapter-defined semantics).
    """
    namespace: Optional[str] = None
    limit: int = 100
    cursor: Optional[str] = None
    filter: Optional[Mapping[str, Any]] = None


@dataclass
class BulkVerticesResult:
    """
    Result for bulk_vertices operations.

    Attributes:
        nodes: Nodes in this page.
        next_cursor: Cursor for the next page (None if no more).
        has_more: True if additional pages are available.
    """
    nodes: List[Node]
    next_cursor: Optional[str]
    has_more: bool


@dataclass(frozen=True)
class BatchOperation:
    """
    Opaque batched graph operation.

    Intentionally generic; routers and adapters agree out-of-band
    on supported shapes. Common examples:

        {"op": "upsert_nodes", "args": {...}}
        {"op": "upsert_edges", "args": {...}}
        {"op": "delete_nodes", "args": {...}}
        {"op": "query", "args": {...}}
    """
    op: str
    args: Mapping[str, Any]


@dataclass
class BatchResult:
    """
    Result for batch().

    Attributes:
        results: Per-operation results (success or error payloads).
    """
    results: List[Any]


# ---- Schema Introspection (restored) ----------------------------------------

@dataclass
class GraphSchema:
    """
    Logical graph schema description for introspection and tooling.

    Structure is adapter-defined but MUST be JSON-serializable and stable enough
    for:

        - Query builders and IDEs
        - Documentation generators
        - Router query planning
        - Migration tools
        - UI schema explorers
    """
    nodes: Mapping[str, Any]
    edges: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.nodes is None:
            object.__setattr__(self, "nodes", {})
        if self.edges is None:
            object.__setattr__(self, "edges", {})
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


# =============================================================================
# Results (strongly-typed, JSON-safe)
# =============================================================================

@dataclass
class QueryResult:
    """
    Result for non-streaming graph queries.

    Attributes:
        records: Backend-defined rows/tuples/paths; JSON-serializable.
        summary: Optional metadata, e.g. stats, plan, consumed capacity.
        dialect: Effective dialect used to run the query.
        namespace: Target namespace / graph.
    """
    records: List[Any]
    summary: Mapping[str, Any]
    dialect: Optional[str] = None
    namespace: Optional[str] = None


@dataclass
class QueryChunk:
    """
    Chunk for streaming graph queries.

    Attributes:
        records: Partial records for this chunk.
        is_final: True if this is the last chunk in the stream.
        summary: Optional final summary when is_final is True.
    """
    records: List[Any]
    is_final: bool = False
    summary: Optional[Mapping[str, Any]] = None


@dataclass
class UpsertResult:
    """
    Result for batch upsert operations.

    Attributes:
        upserted_count: Number of items successfully upserted.
        failed_count: Number of items that failed.
        failures: List of per-item failure details (id, message, code).
    """
    upserted_count: int
    failed_count: int
    failures: List[Mapping[str, Any]]


@dataclass
class DeleteResult:
    """
    Result for batch delete operations.

    Attributes:
        deleted_count: Number of items successfully deleted.
        failed_count: Number of delete failures.
        failures: List of per-item failure details.
    """
    deleted_count: int
    failed_count: int
    failures: List[Mapping[str, Any]]


# =============================================================================
# Normalized Errors
# =============================================================================

class GraphAdapterError(Exception):
    """
    Base exception for graph adapter errors.

    Attributes:
        message: Human-readable description.
        code: Machine-readable, UPPER_SNAKE_CASE error code.
        retry_after_ms: Suggested client backoff (if applicable).
        details: Additional, SIEM-safe machine context (no PII).
    """
    def __init__(
        self,
        message: str = "",
        *,
        code: Optional[str] = None,
        retry_after_ms: Optional[int] = None,
        details: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.retry_after_ms = retry_after_ms
        self.details = dict(details or {})

    def __str__(self) -> str:
        base = self.message or self.__class__.__name__
        if self.code:
            base += f" [code={self.code}]"
        if self.retry_after_ms is not None:
            base += f" retry_after_ms={self.retry_after_ms}"
        if self.details:
            base += f" details={self.details}"
        return base


class BadRequest(GraphAdapterError):
    """Client error: invalid query/spec/parameters."""
    def __init__(self, message: str, **kw: Any):
        kw.setdefault("code", "BAD_REQUEST")
        super().__init__(message, **kw)


class AuthError(GraphAdapterError):
    """Authentication / authorization failure."""
    def __init__(self, message: str, **kw: Any):
        kw.setdefault("code", "AUTH_ERROR")
        super().__init__(message, **kw)


class ResourceExhausted(GraphAdapterError):
    """Quota, rate limit, or capacity exhausted."""
    def __init__(self, message: str, **kw: Any):
        kw.setdefault("code", "RESOURCE_EXHAUSTED")
        super().__init__(message, **kw)


class TransientNetwork(GraphAdapterError):
    """Retryable network failure."""
    def __init__(self, message: str, **kw: Any):
        kw.setdefault("code", "TRANSIENT_NETWORK")
        super().__init__(message, **kw)


class Unavailable(GraphAdapterError):
    """Backend unavailable / overloaded."""
    def __init__(self, message: str, **kw: Any):
        kw.setdefault("code", "UNAVAILABLE")
        super().__init__(message, **kw)


class NotSupported(GraphAdapterError):
    """Unsupported feature / dialect / parameter."""
    def __init__(self, message: str, **kw: Any):
        kw.setdefault("code", "NOT_SUPPORTED")
        super().__init__(message, **kw)


class DeadlineExceeded(GraphAdapterError):
    """Operation exceeded ctx.deadline_ms."""
    def __init__(self, message: str, **kw: Any):
        kw.setdefault("code", "DEADLINE_EXCEEDED")
        super().__init__(message, **kw)


# =============================================================================
# Context + Metrics
# =============================================================================

@dataclass(frozen=True)
class OperationContext:
    """
    Context for graph operations.

    Attributes:
        request_id: Correlation ID for tracing.
        idempotency_key: For idempotent writes (when supported).
        deadline_ms: Absolute epoch ms for operation timeout.
        traceparent: W3C traceparent header.
        tenant: Tenant / customer / app identifier (never logged raw).
        attrs: Extra attributes for middleware / routing (SIEM-safe).
    """
    request_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    deadline_ms: Optional[int] = None
    traceparent: Optional[str] = None
    tenant: Optional[str] = None
    attrs: Mapping[str, Any] = None

    def __post_init__(self) -> None:
        if self.attrs is None:
            object.__setattr__(self, "attrs", {})

    def remaining_ms(self) -> Optional[int]:
        """Return non-negative ms remaining until deadline, or None."""
        if self.deadline_ms is None:
            return None
        now = int(time.time() * 1000)
        return max(0, self.deadline_ms - now)


class MetricsSink(Protocol):
    """
    Metrics collection protocol (low-cardinality; SIEM-safe).
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
        ...

    def counter(
        self,
        *,
        component: str,
        name: str,
        value: int = 1,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        ...


class NoopMetrics:
    def observe(self, **_: Any) -> None:
        ...
    def counter(self, **_: Any) -> None:
        ...


# =============================================================================
# Policy / Infra Extension Points
# =============================================================================

class DeadlinePolicy(Protocol):
    async def wrap(self, awaitable: Awaitable[Any], ctx: Optional[OperationContext]) -> Any:
        ...


class CircuitBreaker(Protocol):
    def allow(self) -> bool:
        ...
    def on_success(self) -> None:
        ...
    def on_error(self, err: Exception) -> None:
        ...


class Cache(Protocol):
    async def get(self, key: str) -> Optional[Any]:
        ...
    async def set(self, key: str, value: Any, ttl_s: int) -> None:
        ...


class RateLimiter(Protocol):
    async def acquire(self) -> None:
        ...
    def release(self) -> None:
        ...


class NoopDeadline:
    async def wrap(self, awaitable: Awaitable[Any], ctx: Optional[OperationContext]) -> Any:
        return await awaitable


class SimpleDeadline:
    """Enforce ctx.deadline_ms using asyncio.wait_for."""
    async def wrap(self, awaitable: Awaitable[Any], ctx: Optional[OperationContext]) -> Any:
        if ctx is None or ctx.deadline_ms is None:
            return await awaitable
        rem = ctx.remaining_ms()
        if rem is not None and rem <= 0:
            raise DeadlineExceeded("deadline already exceeded", details={"remaining_ms": 0})
        try:
            return await asyncio.wait_for(
                awaitable, timeout=(rem / 1000.0 if rem is not None else None)
            )
        except asyncio.TimeoutError as e:
            raise DeadlineExceeded("operation timed out", details={"remaining_ms": 0}) from e


class NoopBreaker:
    def allow(self) -> bool:
        return True
    def on_success(self) -> None:
        ...
    def on_error(self, err: Exception) -> None:
        ...


class SimpleCircuitBreaker:
    """
    Tiny counter-based breaker; per-process only.
    Opens after N consecutive failures; half-open after cool-down.
    """
    def __init__(self, *, failure_threshold: int = 5, recovery_after_s: float = 10.0) -> None:
        self._threshold = max(1, failure_threshold)
        self._recovery_after_s = max(0.1, float(recovery_after_s))
        self._failures = 0
        self._opened_at: Optional[float] = None

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        if (time.monotonic() - self._opened_at) >= self._recovery_after_s:
            # allow one probe (half-open)
            return True
        return False

    def on_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def on_error(self, _err: Exception) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = time.monotonic()


class NoopCache:
    async def get(self, key: str) -> Optional[Any]:
        return None
    async def set(self, key: str, value: Any, ttl_s: int) -> None:
        ...


class InMemoryTTLCache:
    """
    Small in-memory TTL cache for read paths (standalone mode only).
    """
    def __init__(self) -> None:
        self._store: Dict[str, Tuple[float, Any]] = {}

    async def get(self, key: str) -> Optional[Any]:
        now = time.monotonic()
        item = self._store.get(key)
        if not item:
            return None
        exp, val = item
        if now >= exp:
            self._store.pop(key, None)
            return None
        return val

    async def set(self, key: str, value: Any, ttl_s: int) -> None:
        ttl = max(1, int(ttl_s))
        self._store[key] = (time.monotonic() + ttl, value)


class NoopLimiter:
    async def acquire(self) -> None:
        ...
    def release(self) -> None:
        ...


class SimpleTokenBucketLimiter:
    """
    Simple token-bucket limiter; per-process; fail-open on internal error.
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
                await asyncio.sleep(0.02)
        except Exception:
            return  # fail-open

    def release(self) -> None:
        try:
            self._refill()
            if self._tokens < self._capacity:
                self._tokens += 1
        except Exception:
            return  # fail-open


# =============================================================================
# Capabilities (with dialect + streaming + batch + schema flags)
# =============================================================================

@dataclass(frozen=True)
class GraphCapabilities:
    """
    Describes backend capabilities for routing and validation.

    Attributes:
        server: Backend identifier ("neo4j", "janusgraph", "dgraph", etc.)
        version: Backend/server version string.
        protocol: Protocol identifier ("graph/v1.0").
        supports_stream_query: Whether stream_query is supported.
        supported_query_dialects: Allowed dialects for GraphQuerySpec.dialect.
                                  Empty tuple means "adapter-defined" / opaque.
        supports_namespaces: Whether namespace scoping is supported.
        supports_property_filters: Whether Delete*Spec.filter is honored.
        supports_bulk_vertices: Whether bulk_vertices is supported.
        supports_batch: Whether batch() is supported.
        supports_schema: Whether get_schema() is supported.
        idempotent_writes: Whether idempotency_key is honored for writes.
        supports_multi_tenant: Whether tenant-aware isolation is supported.
        supports_deadline: Whether ctx.deadline_ms is respected.
    """
    server: str
    version: str
    protocol: str = GRAPH_PROTOCOL_ID
    supports_stream_query: bool = True
    supported_query_dialects: Tuple[str, ...] = ()
    supports_namespaces: bool = True
    supports_property_filters: bool = True
    supports_bulk_vertices: bool = False
    supports_batch: bool = False
    supports_schema: bool = False
    idempotent_writes: bool = False
    supports_multi_tenant: bool = False
    supports_deadline: bool = True


# =============================================================================
# Stable Protocol Interface
# =============================================================================

@runtime_checkable
class GraphProtocolV1(Protocol):
    """
    Language-level contract for graph adapters.

    WireGraphHandler is built strictly on top of this interface.
    """

    async def capabilities(self) -> GraphCapabilities:
        ...

    async def query(
        self,
        spec: GraphQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        ...

    async def stream_query(
        self,
        spec: GraphQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[QueryChunk]:
        ...

    async def upsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        ...

    async def upsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        ...

    async def delete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        ...

    async def delete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        ...

    async def bulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BulkVerticesResult:
        ...

    async def batch(
        self,
        ops: List[BatchOperation],
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchResult:
        ...

    async def get_schema(
        self,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> GraphSchema:
        ...

    async def health(
        self,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> Mapping[str, Any]:
        ...


# =============================================================================
# Base Instrumented Adapter (DRY gates via _with_gates)
# =============================================================================

class BaseGraphAdapter(GraphProtocolV1):
    """
    Base implementation of GraphProtocolV1.

    Responsibilities:
        - Input validation (specs, dialects, namespaces).
        - Deadline enforcement via DeadlinePolicy.
        - Circuit breaker + rate limiter gates.
        - SIEM-safe metrics.
        - Optional read-path caching in standalone mode.
        - Wire-agnostic; used by WireGraphHandler.

    Backend implementers override `_do_*` methods only.
    """

    _component = "graph"

    def __init__(
        self,
        *,
        metrics: Optional[MetricsSink] = None,
        mode: str = "thin",
        deadline_policy: Optional[DeadlinePolicy] = None,
        breaker: Optional[CircuitBreaker] = None,
        cache: Optional[Cache] = None,
        limiter: Optional[RateLimiter] = None,
        cache_query_ttl_s: int = 30,
        cache_caps_ttl_s: int = 30,
        cache_bulk_vertices_ttl_s: int = 30,
        cache_schema_ttl_s: int = 60,
        stream_deadline_check_every_n_chunks: int = 10,
    ) -> None:
        self._metrics: MetricsSink = metrics or NoopMetrics()

        m = (mode or "thin").strip().lower()
        if m not in {"thin", "standalone"}:
            m = "thin"
        self._mode = m

        if self._mode == "standalone":
            if metrics is None:
                LOG.warning("Using standalone graph adapter without metrics sink")
            self._deadline = deadline_policy or SimpleDeadline()
            self._breaker = breaker or SimpleCircuitBreaker()
            self._cache = cache or InMemoryTTLCache()
            self._limiter = limiter or SimpleTokenBucketLimiter()
        else:
            self._deadline = deadline_policy or NoopDeadline()
            self._breaker = breaker or NoopBreaker()
            self._cache = cache or NoopCache()
            self._limiter = limiter or NoopLimiter()

        self._cache_query_ttl_s = max(1, int(cache_query_ttl_s))
        self._cache_caps_ttl_s = max(1, int(cache_caps_ttl_s))
        self._cache_bulk_vertices_ttl_s = max(1, int(cache_bulk_vertices_ttl_s))
        self._cache_schema_ttl_s = max(1, int(cache_schema_ttl_s))
        self._stream_deadline_check_every_n_chunks = max(1, int(stream_deadline_check_every_n_chunks))

    # ---- lifecycle helpers --------------------------------------------------

    async def __aenter__(self) -> "BaseGraphAdapter":
        """Allow use as an async context manager in apps/services."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def close(self) -> None:
        """
        Clean up underlying resources (connections, pools, clients, etc.)

        Adapters should override when they own external resources.
        """
        return None

    # ---- internal helpers ---------------------------------------------------

    @staticmethod
    def _tenant_hash(tenant: Optional[str]) -> Optional[str]:
        if not tenant:
            return None
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
        try:
            ms = (time.monotonic() - t0) * 1000.0
            x = dict(extra or {})
            if ctx:
                th = self._tenant_hash(ctx.tenant)
                if th:
                    x.setdefault("tenant", th)
            self._metrics.observe(
                component=self._component,
                op=op,
                ms=ms,
                ok=ok,
                code=code,
                extra=x or None,
            )
        except Exception:
            # never let metrics break caller
            pass

    def _fail_if_deadline_expired(self, ctx: Optional[OperationContext]) -> None:
        if ctx is None:
            return
        rem = ctx.remaining_ms()
        if rem is not None and rem <= 0:
            raise DeadlineExceeded("deadline already exceeded", details={"remaining_ms": 0})

    async def _apply_deadline(
        self,
        awaitable: Awaitable[Any],
        ctx: Optional[OperationContext],
    ) -> Any:
        try:
            return await self._deadline.wrap(awaitable, ctx)
        except asyncio.TimeoutError as e:
            raise DeadlineExceeded("operation timed out", details={"remaining_ms": 0}) from e

    def _make_cache_key(
        self,
        *,
        op: str,
        spec: Any,
        ctx: Optional[OperationContext] = None,
    ) -> str:
        """
        Compose a cache key for read operations.

        Includes:
            - operation
            - stable repr of spec/asdict(spec)
            - tenant hash (if present) to avoid cross-tenant bleed.
        """
        if hasattr(spec, "__dataclass_fields__"):
            raw = asdict(spec)
        else:
            raw = spec
        base = f"graph:{op}:{repr(raw)}"
        if ctx and ctx.tenant:
            th = self._tenant_hash(ctx.tenant)
            if th:
                base += f":tenant:{th}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_properties_map(properties: Mapping[str, Any]) -> None:
        """
        Validate that a properties mapping is JSON-serializable.

        This is intentionally strict to avoid subtle wire failures later.
        """
        try:
            json.dumps(properties)
        except (TypeError, ValueError) as e:
            raise BadRequest(f"properties must be JSON-serializable: {e}")

    def _validate_node(self, node: Node) -> None:
        if node.properties is not None:
            self._validate_properties_map(node.properties)

    def _validate_edge(self, edge: Edge) -> None:
        if not isinstance(edge.label, str) or not edge.label:
            raise BadRequest("edge.label must be a non-empty string")
        if edge.properties is not None:
            self._validate_properties_map(edge.properties)

    # ---- DRY gate wrappers --------------------------------------------------

    async def _with_gates_unary(
        self,
        *,
        op: str,
        ctx: Optional[OperationContext],
        call: Callable[[], Awaitable[Any]],
        metric_extra: Mapping[str, Any] = None,
    ) -> Any:
        """
        DRY wrapper for unary operations:
        - checks deadline
        - circuit breaker
        - rate limiter
        - maps errors to metrics
        """
        metric_extra = dict(metric_extra or {})
        self._fail_if_deadline_expired(ctx)

        if not self._breaker.allow():
            e = Unavailable("circuit open")
            t0 = time.monotonic()
            self._record(op, t0, False, code=e.code, ctx=ctx, **metric_extra)
            raise e

        await self._limiter.acquire()
        t0 = time.monotonic()
        try:
            result = await self._apply_deadline(call(), ctx)
            self._record(op, t0, True, ctx=ctx, **metric_extra)
            self._breaker.on_success()
            return result
        except GraphAdapterError as e:
            self._record(op, t0, False, code=e.code or type(e).__name__, ctx=ctx, **metric_extra)
            self._breaker.on_error(e)
            raise
        except Exception:
            self._record(op, t0, False, code="UNAVAILABLE", ctx=ctx, **metric_extra)
            self._breaker.on_error(e)
            raise
        finally:
            self._limiter.release()

    async def _with_gates_stream(
        self,
        *,
        op: str,
        ctx: Optional[OperationContext],
        agen_factory: Callable[[], AsyncIterator[QueryChunk]],
        metric_extra: Mapping[str, Any] = None,
    ) -> AsyncIterator[QueryChunk]:
        """
        DRY wrapper for streaming operations:
        - preflight deadline, breaker, limiter
        - periodic deadline checks
        - metrics on completion or error
        """
        metric_extra = dict(metric_extra or {})
        self._fail_if_deadline_expired(ctx)

        if not self._breaker.allow():
            raise Unavailable("circuit open")

        await self._limiter.acquire()
        t0 = time.monotonic()
        check_n = self._stream_deadline_check_every_n_chunks

        async def _gen() -> AsyncIterator[QueryChunk]:
            chunk_count = 0
            try:
                agen = agen_factory()
                async for chunk in agen:
                    chunk_count += 1
                    if chunk_count % check_n == 0:
                        self._fail_if_deadline_expired(ctx)
                    yield chunk
                self._record(op, t0, True, ctx=ctx, **metric_extra)
                self._breaker.on_success()
            except GraphAdapterError as e:
                self._record(op, t0, False, code=e.code or type(e).__name__, ctx=ctx, **metric_extra)
                self._breaker.on_error(e)
                raise
            except Exception:
                self._record(op, t0, False, code="UNAVAILABLE", ctx=ctx, **metric_extra)
                self._breaker.on_error(e)
                raise
            finally:
                self._limiter.release()

        return _gen()

    # --- public API ----------------------------------------------------------

    async def capabilities(self) -> GraphCapabilities:
        """
        Return adapter capabilities.

        Standalone mode: may be cached briefly to reduce overhead.
        """
        t0 = time.monotonic()
        try:
            if isinstance(self._cache, InMemoryTTLCache):
                key = "graph:capabilities"
                cached = await self._cache.get(key)
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

            if isinstance(self._cache, InMemoryTTLCache):
                await self._cache.set("graph:capabilities", caps, ttl_s=self._cache_caps_ttl_s)

            self._record("capabilities", t0, True)
            return caps
        except GraphAdapterError as e:
            self._record("capabilities", t0, False, code=e.code or type(e).__name__)
            raise
        except Exception as e:
            self._record("capabilities", t0, False, code="UNAVAILABLE")
            raise Unavailable("capabilities fetch failed") from e

    async def query(
        self,
        spec: GraphQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        """
        Execute a non-streaming graph query.

        Dialect rules:
            - If capabilities.supported_query_dialects is non-empty, and spec.dialect
              is set, it MUST be a member.
            - If dialect is None, adapter may treat text as backend-native.
        """
        if not isinstance(spec.text, str) or not spec.text.strip():
            raise BadRequest("query.text must be a non-empty string")

        caps = await self.capabilities()
        if spec.dialect and caps.supported_query_dialects:
            if spec.dialect not in caps.supported_query_dialects:
                raise NotSupported(
                    f"dialect '{spec.dialect}' not supported",
                    details={"supported_query_dialects": caps.supported_query_dialects},
                )

        async def _call() -> QueryResult:
            # standalone: cache on read-only, non-streaming queries
            if isinstance(self._cache, InMemoryTTLCache) and not spec.stream:
                key = self._make_cache_key(op="query", spec=spec, ctx=ctx)
                cached = await self._cache.get(key)
                if cached:
                    self._metrics.counter(
                        component=self._component,
                        name="cache_hits",
                        value=1,
                        extra={"op": "query"},
                    )
                    return cached
                res = await self._do_query(spec, ctx=ctx)
                await self._cache.set(key, res, ttl_s=self._cache_query_ttl_s)
                return res
            return await self._do_query(spec, ctx=ctx)

        return await self._with_gates_unary(
            op="query",
            ctx=ctx,
            call=_call,
            metric_extra={"dialect": spec.dialect or "none"},
        )

    async def stream_query(
        self,
        spec: GraphQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[QueryChunk]:
        """
        Execute a streaming graph query.

        For large result sets; preferred over `query` when supported.
        """
        if not isinstance(spec.text, str) or not spec.text.strip():
            raise BadRequest("query.text must be a non-empty string")

        caps = await self.capabilities()
        if not caps.supports_stream_query:
            raise NotSupported("stream_query is not supported by this adapter")
        if spec.dialect and caps.supported_query_dialects:
            if spec.dialect not in caps.supported_query_dialects:
                raise NotSupported(
                    f"dialect '{spec.dialect}' not supported",
                    details={"supported_query_dialects": caps.supported_query_dialects},
                )

        async def agen_factory() -> AsyncIterator[QueryChunk]:
            async for chunk in self._do_stream_query(spec, ctx=ctx):
                yield chunk

        return await self._with_gates_stream(
            op="stream_query",
            ctx=ctx,
            agen_factory=lambda: agen_factory(),
            metric_extra={"dialect": spec.dialect or "none"},
        )

    async def upsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        """Batch upsert nodes with validation and gates."""
        if not spec.nodes:
            raise BadRequest("nodes must not be empty")
        for n in spec.nodes:
            self._validate_node(n)

        async def _call() -> UpsertResult:
            return await self._do_upsert_nodes(spec, ctx=ctx)

        return await self._with_gates_unary(
            op="upsert_nodes",
            ctx=ctx,
            call=_call,
            metric_extra={"count": len(spec.nodes)},
        )

    async def upsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        """Batch upsert edges with validation and gates."""
        if not spec.edges:
            raise BadRequest("edges must not be empty")
        for e in spec.edges:
            self._validate_edge(e)

        async def _call() -> UpsertResult:
            return await self._do_upsert_edges(spec, ctx=ctx)

        return await self._with_gates_unary(
            op="upsert_edges",
            ctx=ctx,
            call=_call,
            metric_extra={"count": len(spec.edges)},
        )

    async def delete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        """Batch delete nodes by IDs and/or filter."""
        if not spec.ids and not spec.filter:
            raise BadRequest("must provide ids or filter for delete_nodes")
        if spec.filter is not None:
            self._validate_properties_map(spec.filter)

        async def _call() -> DeleteResult:
            return await self._do_delete_nodes(spec, ctx=ctx)

        return await self._with_gates_unary(
            op="delete_nodes",
            ctx=ctx,
            call=_call,
            metric_extra={"ids": len(spec.ids)},
        )

    async def delete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        """Batch delete edges by IDs and/or filter."""
        if not spec.ids and not spec.filter:
            raise BadRequest("must provide ids or filter for delete_edges")
        if spec.filter is not None:
            self._validate_properties_map(spec.filter)

        async def _call() -> DeleteResult:
            return await self._do_delete_edges(spec, ctx=ctx)

        return await self._with_gates_unary(
            op="delete_edges",
            ctx=ctx,
            call=_call,
            metric_extra={"ids": len(spec.ids)},
        )

    async def bulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BulkVerticesResult:
        """
        Scan/list vertices in bulk with optional pagination.

        Only available if capabilities.supports_bulk_vertices is True.
        """
        caps = await self.capabilities()
        if not caps.supports_bulk_vertices:
            raise NotSupported("bulk_vertices is not supported by this adapter")
        if spec.limit <= 0:
            raise BadRequest("limit must be positive")
        if spec.filter is not None:
            self._validate_properties_map(spec.filter)

        async def _call() -> BulkVerticesResult:
            # optional cache for purely read-only scans (first page, no filter)
            if (
                isinstance(self._cache, InMemoryTTLCache)
                and spec.cursor is None
                and spec.filter is None
            ):
                key = self._make_cache_key(op="bulk_vertices", spec=spec, ctx=ctx)
                cached = await self._cache.get(key)
                if cached:
                    self._metrics.counter(
                        component=self._component,
                        name="cache_hits",
                        value=1,
                        extra={"op": "bulk_vertices"},
                    )
                    return cached
                res = await self._do_bulk_vertices(spec, ctx=ctx)
                await self._cache.set(key, res, ttl_s=self._cache_bulk_vertices_ttl_s)
                return res
            return await self._do_bulk_vertices(spec, ctx=ctx)

        return await self._with_gates_unary(
            op="bulk_vertices",
            ctx=ctx,
            call=_call,
            metric_extra={
                "limit": spec.limit,
                "cursor": "yes" if spec.cursor else "no",
            },
        )

    async def batch(
        self,
        ops: List[BatchOperation],
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchResult:
        """
        Execute a batch of graph operations.

        Semantics are adapter-defined; intended for reducing round-trips.
        """
        caps = await self.capabilities()
        if not caps.supports_batch:
            raise NotSupported("batch is not supported by this adapter")
        if not ops:
            raise BadRequest("ops must not be empty")

        async def _call() -> BatchResult:
            return await self._do_batch(ops, ctx=ctx)

        return await self._with_gates_unary(
            op="batch",
            ctx=ctx,
            call=_call,
            metric_extra={"ops": len(ops)},
        )

    async def get_schema(
        self,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> GraphSchema:
        """
        Return logical graph schema for tooling and routing.

        Behavior:
            - If capabilities.supports_schema is False -> NotSupported.
            - In standalone mode, result may be cached briefly.
        """
        caps = await self.capabilities()
        if not caps.supports_schema:
            raise NotSupported("get_schema is not supported by this adapter")

        async def _call() -> GraphSchema:
            if isinstance(self._cache, InMemoryTTLCache):
                key = self._make_cache_key(op="schema", spec="all", ctx=ctx)
                cached = await self._cache.get(key)
                if cached:
                    self._metrics.counter(
                        component=self._component,
                        name="cache_hits",
                        value=1,
                        extra={"op": "get_schema"},
                    )
                    return cached
            res = await self._do_get_schema(ctx=ctx)
            if isinstance(self._cache, InMemoryTTLCache):
                key = self._make_cache_key(op="schema", spec="all", ctx=ctx)
                await self._cache.set(key, res, ttl_s=self._cache_schema_ttl_s)
            return res

        return await self._with_gates_unary(
            op="get_schema",
            ctx=ctx,
            call=_call,
        )

    async def health(
        self,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> Mapping[str, Any]:
        """Health check with deadline + metrics; no cache."""
        self._fail_if_deadline_expired(ctx)
        t0 = time.monotonic()
        try:
            res = await self._apply_deadline(self._do_health(ctx=ctx), ctx)
            self._record("health", t0, True, ctx=ctx)
            return {
                "ok": bool(res.get("ok", True)),
                "server": str(res.get("server", "")),
                "version": str(res.get("version", "")),
                "namespaces": res.get("namespaces", {}),
            }
        except GraphAdapterError as e:
            self._record("health", t0, False, code=e.code or type(e).__name__, ctx=ctx)
            raise
        except Exception as e:
            self._record("health", t0, False, code="UNAVAILABLE", ctx=ctx)
            raise Unavailable("health check failed") from e

    # --- backend hooks -------------------------------------------------------

    async def _do_capabilities(self) -> GraphCapabilities:
        raise NotImplementedError

    async def _do_query(
        self,
        spec: GraphQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        raise NotImplementedError

    async def _do_stream_query(
        self,
        spec: GraphQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[QueryChunk]:
        raise NotImplementedError

    async def _do_upsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        raise NotImplementedError

    async def _do_upsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        raise NotImplementedError

    async def _do_delete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        raise NotImplementedError

    async def _do_delete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        raise NotImplementedError

    async def _do_bulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BulkVerticesResult:
        raise NotImplementedError

    async def _do_batch(
        self,
        ops: List[BatchOperation],
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchResult:
        raise NotImplementedError

    async def _do_get_schema(
        self,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> GraphSchema:
        raise NotImplementedError

    async def _do_health(
        self,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> Mapping[str, Any]:
        raise NotImplementedError


# =============================================================================
# Wire-Level Helpers (canonical envelopes)
# =============================================================================

def _ctx_from_wire(ctx_dict: Mapping[str, Any]) -> OperationContext:
    """
    Convert wire-level ctx dict to OperationContext. Unknown keys are ignored.
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
    Map GraphAdapterError (or unexpected Exception) to canonical error envelope.
    """
    if isinstance(e, GraphAdapterError):
        return {
            "ok": False,
            "code": e.code or type(e).__name__.upper(),
            "error": type(e).__name__,
            "message": e.message,
            "retry_after_ms": e.retry_after_ms,
            "details": e.details or None,
            "ms": ms,
        }
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
        payload = asdict(result)
    else:
        payload = result
    return {
        "ok": True,
        "code": "OK",
        "ms": ms,
        "result": payload,
    }


def _chunk_to_wire(chunk: QueryChunk, ms: float) -> Dict[str, Any]:
    """
    Map QueryChunk to streaming envelope.
    """
    return {
        "ok": True,
        "code": "OK",
        "ms": ms,
        "chunk": asdict(chunk),
    }


class WireGraphHandler:
    """
    Reference wire adapter for GraphProtocolV1.

    Transport-agnostic: plug into HTTP, gRPC, WebSocket, etc.
    """

    def __init__(self, adapter: GraphProtocolV1):
        self._adapter = adapter

    async def handle(self, envelope: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Handle unary graph operations via JSON envelope.

        Supported ops:
            - graph.capabilities
            - graph.query
            - graph.upsert_nodes
            - graph.upsert_edges
            - graph.delete_nodes
            - graph.delete_edges
            - graph.bulk_vertices
            - graph.batch
            - graph.get_schema
            - graph.health

        Streaming:
            - graph.stream_query via handle_stream(...)
        """
        t0 = time.monotonic()
        try:
            op = envelope.get("op")
            if not isinstance(op, str):
                raise BadRequest("missing or invalid 'op'")

            ctx = _ctx_from_wire(envelope.get("ctx") or {})
            args = envelope.get("args") or {}

            if op == "graph.capabilities":
                res = await self._adapter.capabilities()
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "graph.query":
                spec = GraphQuerySpec(**args)
                res = await self._adapter.query(spec, ctx=ctx)
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "graph.upsert_nodes":
                nodes = [Node(**n) for n in args.get("nodes", [])]
                spec = UpsertNodesSpec(nodes=nodes, namespace=args.get("namespace"))
                res = await self._adapter.upsert_nodes(spec, ctx=ctx)
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "graph.upsert_edges":
                edges = [Edge(**e) for e in args.get("edges", [])]
                spec = UpsertEdgesSpec(edges=edges, namespace=args.get("namespace"))
                res = await self._adapter.upsert_edges(spec, ctx=ctx)
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "graph.delete_nodes":
                ids = [GraphID(i) for i in args.get("ids", [])]
                spec = DeleteNodesSpec(
                    ids=ids,
                    namespace=args.get("namespace"),
                    filter=args.get("filter"),
                )
                res = await self._adapter.delete_nodes(spec, ctx=ctx)
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "graph.delete_edges":
                ids = [GraphID(i) for i in args.get("ids", [])]
                spec = DeleteEdgesSpec(
                    ids=ids,
                    namespace=args.get("namespace"),
                    filter=args.get("filter"),
                )
                res = await self._adapter.delete_edges(spec, ctx=ctx)
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "graph.bulk_vertices":
                spec = BulkVerticesSpec(**args)
                res = await self._adapter.bulk_vertices(spec, ctx=ctx)
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "graph.batch":
                ops = [BatchOperation(**o) for o in args.get("ops", [])]
                res = await self._adapter.batch(ops, ctx=ctx)
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "graph.get_schema":
                res = await self._adapter.get_schema(ctx=ctx)
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "graph.health":
                res = await self._adapter.health(ctx=ctx)
                return _success_to_wire(res, (time.monotonic() - t0) * 1000.0)

            raise NotSupported(f"unknown or non-unary operation '{op}'")
        except Exception as e:
            ms = (time.monotonic() - t0) * 1000.0
            return _error_to_wire(e, ms)

    async def handle_stream(self, envelope: Mapping[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """
        Handle streaming graph queries:

            op: "graph.stream_query"
        """
        t0 = time.monotonic()
        op = envelope.get("op")
        if op != "graph.stream_query":
            yield _error_to_wire(BadRequest("op must be 'graph.stream_query'"), 0.0)
            return

        ctx = _ctx_from_wire(envelope.get("ctx") or {})
        args = envelope.get("args") or {}
        spec = GraphQuerySpec(**args)

        try:
            agen = self._adapter.stream_query(spec, ctx=ctx)
            async for chunk in agen:
                ms = (time.monotonic() - t0) * 1000.0
                yield _chunk_to_wire(chunk, ms)
        except Exception as e:
            ms = (time.monotonic() - t0) * 1000.0
            yield _error_to_wire(e, ms)


__all__ = [
    "GRAPH_PROTOCOL_VERSION",
    "GRAPH_PROTOCOL_ID",
    "GraphID",
    "Node",
    "Edge",
    "GraphQuerySpec",
    "UpsertNodesSpec",
    "UpsertEdgesSpec",
    "DeleteNodesSpec",
    "DeleteEdgesSpec",
    "BulkVerticesSpec",
    "BulkVerticesResult",
    "BatchOperation",
    "BatchResult",
    "GraphSchema",
    "QueryResult",
    "QueryChunk",
    "UpsertResult",
    "DeleteResult",
    "GraphAdapterError",
    "BadRequest",
    "AuthError",
    "ResourceExhausted",
    "TransientNetwork",
    "Unavailable",
    "NotSupported",
    "DeadlineExceeded",
    "OperationContext",
    "MetricsSink",
    "NoopMetrics",
    "GraphCapabilities",
    "GraphProtocolV1",
    "BaseGraphAdapter",
    # infra hooks
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
    "WireGraphHandler",
    "_ctx_from_wire",
    "_error_to_wire",
    "_success_to_wire",
    "_chunk_to_wire",
]
