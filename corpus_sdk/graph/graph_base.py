# corpus_sdk/graph/graph_base.py
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
        created_at: Optional creation timestamp (epoch milliseconds)
        updated_at: Optional last update timestamp (epoch milliseconds)
    """
    id: GraphID
    labels: Tuple[str, ...] = ()
    properties: Mapping[str, Any] = None
    namespace: Optional[str] = None
    created_at: Optional[int] = None
    updated_at: Optional[int] = None

    def __post_init__(self) -> None:
        if self.properties is None:
            object.__setattr__(self, "properties", {})
        # Validate labels are all strings
        if self.labels:
            for i, label in enumerate(self.labels):
                if not isinstance(label, str):
                    # Map to normalized client error rather than generic ValueError
                    raise BadRequest(
                        f"labels[{i}] must be a string, got {type(label).__name__}"
                    )


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
        created_at: Optional creation timestamp (epoch milliseconds)
        updated_at: Optional last update timestamp (epoch milliseconds)
    """
    id: GraphID
    src: GraphID
    dst: GraphID
    label: str
    properties: Mapping[str, Any] = None
    namespace: Optional[str] = None
    created_at: Optional[int] = None
    updated_at: Optional[int] = None

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


# ---- Transaction and Traversal specs ----------------------------------------

@dataclass(frozen=True)
class GraphTransaction:
    """
    Transaction context for atomic batch operations.

    Attributes:
        operations: List of batch operations to execute atomically
        namespace: Optional namespace / graph for the transaction
        timeout_ms: Optional transaction-level timeout
    """
    operations: List[BatchOperation]
    namespace: Optional[str] = None
    timeout_ms: Optional[int] = None


@dataclass(frozen=True)
class GraphTraversalSpec:
    """
    Specification for graph traversal operations.

    Attributes:
        start_nodes: List of node IDs to start traversal from
        max_depth: Maximum traversal depth (default: 1)
        direction: Traversal direction - "OUTGOING", "INCOMING", or "BOTH"
        relationship_types: Optional filter for relationship types/labels
        node_filters: Optional property filters for nodes to include
        relationship_filters: Optional property filters for relationships to traverse
        return_properties: Optional list of node/relationship properties to return
        namespace: Optional namespace / graph for traversal
    """
    start_nodes: List[str]
    max_depth: int = 1
    direction: str = "OUTGOING"
    relationship_types: Optional[Tuple[str, ...]] = None
    node_filters: Optional[Mapping[str, Any]] = None
    relationship_filters: Optional[Mapping[str, Any]] = None
    return_properties: Optional[Tuple[str, ...]] = None
    namespace: Optional[str] = None

    def __post_init__(self) -> None:
        # Validate direction
        if self.direction not in {"OUTGOING", "INCOMING", "BOTH"}:
            raise BadRequest(
                f"direction must be OUTGOING, INCOMING, or BOTH, got {self.direction}"
            )
        # Validate max_depth
        if self.max_depth < 1:
            raise BadRequest("max_depth must be at least 1")


@dataclass
class TraversalResult:
    """
    Result for graph traversal operations.

    Attributes:
        nodes: List of nodes discovered during traversal
        relationships: List of relationships traversed
        paths: List of complete paths (node-relationship-node sequences)
        summary: Optional traversal metadata and statistics
        namespace: Target namespace / graph
    """
    nodes: List[Node]
    relationships: List[Edge]
    paths: List[List[Union[Node, Edge]]]  # Alternating node-edge-node sequence
    summary: Mapping[str, Any]
    namespace: Optional[str] = None

    def __post_init__(self) -> None:
        if self.nodes is None:
            object.__setattr__(self, "nodes", [])
        if self.relationships is None:
            object.__setattr__(self, "relationships", [])
        if self.paths is None:
            object.__setattr__(self, "paths", [])
        if self.summary is None:
            object.__setattr__(self, "summary", {})


# ---- Bulk / Batch specs ------------------------------------------

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
        success: Whether the entire batch succeeded
        error: Error message if batch failed
        transaction_id: Optional transaction identifier for atomic batches
    """
    results: List[Any]
    success: bool = True
    error: Optional[str] = None
    transaction_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.results is None:
            object.__setattr__(self, "results", [])


# ---- Schema Introspection ----------------------------------------

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
    attrs: Optional[Mapping[str, Any]] = None

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
    async def invalidate_pattern(self, pattern: str) -> None:
        """
        Invalidate cached entries whose cache keys match the given pattern.

        The pattern semantics (substring match, glob, prefix, etc.) are
        implementation-defined, but should be documented per Cache impl.
        """
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

    async def invalidate_pattern(self, pattern: str) -> None:
        # No-op by design
        return None


class InMemoryTTLCache:
    """
    Small in-memory TTL cache for read paths (standalone mode only).

    Not thread-safe, not process-safe, and not distributed. Intended for
    single-process, best-effort development and test use only. Do NOT
    share instances across multiple processes or threads in production.
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

    async def invalidate_pattern(self, pattern: str) -> None:
        """
        Simple substring-based invalidation for in-memory cache.
        """
        try:
            keys_to_remove = [k for k in list(self._store.keys()) if pattern in k]
            for key in keys_to_remove:
                self._store.pop(key, None)
        except Exception:
            # Invalidating cache must never break callers
            LOG.debug("InMemoryTTLCache.invalidate_pattern failed for pattern %s", pattern)


class NoopLimiter:
    async def acquire(self) -> None:
        ...
    def release(self) -> None:
        ...


class SimpleTokenBucketLimiter:
    """
    Simple token-bucket limiter; per-process; fail-open on internal error.

    Not concurrency-safe across threads or processes; intended for
    best-effort throttling in development and light production only.

    Note: release() is a no-op to maintain consistency with LLM-side
    TokenBucketLimiter semantics.
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
        # No-op to maintain consistency with LLM-side TokenBucketLimiter
        return


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
        max_batch_ops: Optional maximum number of ops per batch (adapter-defined).
        supports_transaction: Whether atomic transactions are supported.
        supports_traversal: Whether graph traversal operations are supported.
        max_traversal_depth: Optional maximum traversal depth supported.
        supports_path_queries: Whether path-based queries are supported.
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
    max_batch_ops: Optional[int] = None
    supports_transaction: bool = False
    supports_traversal: bool = False
    max_traversal_depth: Optional[int] = None
    supports_path_queries: bool = False


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

    # ---- Transaction and Traversal operations -------------------------------

    async def transaction(
        self,
        operations: List[BatchOperation],
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchResult:
        """
        Execute a batch of operations atomically as a transaction.

        This provides ACID guarantees for the entire batch - either all
        operations succeed or none are applied.

        Args:
            operations: List of batch operations to execute atomically
            ctx: Operation context including deadline and tracing

        Returns:
            BatchResult with transaction-level success/failure status

        Raises:
            NotSupported: If transactions are not supported by the backend
            BadRequest: If transaction operations are invalid
            DeadlineExceeded: If transaction exceeds context deadline
        """
        ...

    async def traversal(
        self,
        spec: GraphTraversalSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> TraversalResult:
        """
        Perform graph traversal starting from specified nodes.

        Traversal explores the graph structure by following relationships
        from starting nodes up to a specified depth.

        Args:
            spec: Traversal specification including start nodes, depth, filters
            ctx: Operation context including deadline and tracing

        Returns:
            TraversalResult containing discovered nodes, relationships, and paths

        Raises:
            NotSupported: If traversal operations are not supported
            BadRequest: If traversal specification is invalid
            DeadlineExceeded: If traversal exceeds context deadline
        """
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

    Standalone mode wires in lightweight, in-process helpers
    (SimpleCircuitBreaker, InMemoryTTLCache, SimpleTokenBucketLimiter)
    which are *not* thread-safe, process-safe, or distributed. For
    serious production deployments, supply hardened implementations
    via the constructor instead of relying on these defaults.
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
        auto_timestamp_writes: bool = False,
        strict_batch_validation: bool = False,
        batch_supported_ops: Optional[Iterable[str]] = None,
        allow_self_loops: bool = True,
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
        self._stream_deadline_check_every_n_chunks = max(
            1, int(stream_deadline_check_every_n_chunks)
        )

        # Behavior flags
        self._auto_timestamp_writes = bool(auto_timestamp_writes)
        self._strict_batch_validation = bool(strict_batch_validation)
        self._allow_self_loops = bool(allow_self_loops)

        default_supported_ops = {
            "query",
            "upsert_nodes",
            "upsert_edges",
            "delete_nodes",
            "delete_edges",
            "bulk_vertices",
            "get_schema",
            "health",
            "batch",
            "transaction",
            "traversal",
        }
        if batch_supported_ops is not None:
            self._batch_supported_ops = set(batch_supported_ops)
        else:
            self._batch_supported_ops = default_supported_ops

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
                    x.setdefault("tenant_hash", th)
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

    def _extract_namespace_from_spec(self, spec: Any) -> Optional[str]:
        """
        Extract namespace from various spec types.

        Returns None for operations that don't target a specific namespace
        to avoid unnecessary cache invalidation.
        """
        if hasattr(spec, "namespace"):
            return getattr(spec, "namespace")
        return None

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
            - stable JSON-serialized representation of spec/asdict(spec)
            - tenant hash (if present) to avoid cross-tenant bleed
            - namespace tag (if present) to support targeted invalidation

        Uses type-prefixed hashing (j:/p:) to prevent collisions between
        different types that serialize to identical JSON.
        """
        if hasattr(spec, "__dataclass_fields__"):
            raw = asdict(spec)
        else:
            raw = spec

        payload: Dict[str, Any] = {"op": op, "spec": raw}

        tenant_hash = None
        if ctx and ctx.tenant:
            tenant_hash = self._tenant_hash(ctx.tenant)
            if tenant_hash:
                payload["tenant"] = tenant_hash

        namespace = self._extract_namespace_from_spec(spec)
        ns_tag = namespace or "none"

        try:
            serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            base_hash = hashlib.sha256(f"j:{serialized}".encode()).hexdigest()
        except TypeError:
            base_hash = hashlib.sha256(f"p:{repr(payload)}".encode()).hexdigest()

        t_tag = tenant_hash or "none"
        # Human-readable prefix to allow pattern-based invalidation
        return f"ns={ns_tag}|t={t_tag}|{base_hash}"

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
        if not edge.src or not isinstance(edge.src, str):
            raise BadRequest("edge.src must be a non-empty string")
        if not edge.dst or not isinstance(edge.dst, str):
            raise BadRequest("edge.dst must be a non-empty string")
        
        if not self._allow_self_loops and edge.src == edge.dst:
            raise BadRequest("Self-loops are not allowed by this adapter")

        if edge.properties is not None:
            self._validate_properties_map(edge.properties)

    async def _invalidate_namespace_cache(self, namespace: Optional[str]) -> None:
        """
        Invalidate cache entries for a namespace (best-effort).

        If namespace is None, no invalidation is performed to avoid
        unnecessarily clearing caches for non-namespaced operations.
        """
        if namespace is None or isinstance(self._cache, NoopCache):
            return

        try:
            pattern = f"ns={namespace}|"
            await self._cache.invalidate_pattern(pattern)
        except Exception:
            # Never let cache invalidation break the main operation
            LOG.debug("Cache invalidation failed for namespace %s", namespace)

    def _batch_op_succeeded(self, op: BatchOperation, batch_result: BatchResult, idx: int) -> bool:
        """
        Determine if a specific batch operation succeeded.

        This inspects the batch result to see if the operation at the given
        index actually modified data, allowing for smarter cache invalidation.
        """
        if idx >= len(batch_result.results):
            return False

        result = batch_result.results[idx]

        # For write operations, check if they actually succeeded
        if op.op in {"upsert_nodes", "upsert_edges"}:
            if isinstance(result, UpsertResult):
                return result.upserted_count > 0
        elif op.op in {"delete_nodes", "delete_edges"}:
            if isinstance(result, DeleteResult):
                return result.deleted_count > 0

        # For query operations or unknown result types, assume no cache invalidation needed
        return False

    def _extract_namespace_from_batch_op(self, op: BatchOperation) -> Optional[str]:
        """
        Extract namespace from a batch operation.

        Returns None if no namespace is specified to avoid unnecessary
        cache invalidation for non-namespaced operations.
        """
        args = op.args or {}
        return args.get("namespace")

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
        except Exception as e:
            # Standardize on "UnhandledException" for cross-SDK metrics alignment
            self._record(op, t0, False, code="UnhandledException", ctx=ctx, **metric_extra)
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
        - chunk-level throughput metrics (chunks and records)
        - proper resource cleanup for abandoned streams
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
            agen: Optional[AsyncIterator[QueryChunk]] = None
            try:
                agen = agen_factory()
                async for chunk in agen:
                    chunk_count += 1
                    # chunk-level throughput metrics (best-effort, non-fatal)
                    try:
                        self._metrics.counter(
                            component=self._component,
                            name="stream_chunks_total",
                            value=1,
                            extra={"op": op},
                        )
                        self._metrics.counter(
                            component=self._component,
                            name="stream_records_total",
                            value=len(chunk.records),
                            extra={"op": op},
                        )
                    except Exception:
                        pass

                    if chunk_count % check_n == 0:
                        self._fail_if_deadline_expired(ctx)
                    yield chunk
                self._record(op, t0, True, ctx=ctx, **metric_extra)
                self._breaker.on_success()
            except GraphAdapterError as e:
                self._record(op, t0, False, code=e.code or type(e).__name__, ctx=ctx, **metric_extra)
                self._breaker.on_error(e)
                raise
            except Exception as e:
                # Standardize on "UnhandledException" for cross-SDK metrics alignment
                self._record(op, t0, False, code="UnhandledException", ctx=ctx, **metric_extra)
                self._breaker.on_error(e)
                raise
            finally:
                self._limiter.release()
                # Ensure underlying stream resources are cleaned up
                if agen is not None:
                    try:
                        await agen.aclose()  # If the async generator supports it
                    except (AttributeError, RuntimeError):
                        pass  # Not all async generators have aclose()

        return _gen()

    # --- public API ----------------------------------------------------------

    async def capabilities(self) -> GraphCapabilities:
        """
        Return adapter capabilities.

        Standalone mode: may be cached briefly to reduce overhead.
        Uses pluggable cache interface (not tied to InMemoryTTLCache).
        """
        t0 = time.monotonic()
        try:
            # Use pluggable cache interface
            if not isinstance(self._cache, NoopCache):
                key = self._make_cache_key(op="capabilities", spec="all", ctx=None)
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

            if not isinstance(self._cache, NoopCache):
                key = self._make_cache_key(op="capabilities", spec="all", ctx=None)
                await self._cache.set(key, caps, ttl_s=self._cache_caps_ttl_s)

            self._record("capabilities", t0, True)
            return caps
        except GraphAdapterError as e:
            self._record("capabilities", t0, False, code=e.code or type(e).__name__)
            raise
        except Exception as e:
            self._record("capabilities", t0, False, code="UnhandledException")
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
        async def _call() -> QueryResult:
            # Validate query spec (inside gate wrapper for metrics)
            if not isinstance(spec.text, str) or not spec.text.strip():
                raise BadRequest("query.text must be a non-empty string")

            caps = await self.capabilities()
            if spec.dialect and caps.supported_query_dialects:
                if spec.dialect not in caps.supported_query_dialects:
                    raise NotSupported(
                        f"dialect '{spec.dialect}' not supported",
                        details={"supported_query_dialects": caps.supported_query_dialects},
                    )

            # Use pluggable cache interface for read-only, non-streaming queries
            if not isinstance(self._cache, NoopCache) and not spec.stream:
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

    def stream_query(
        self,
        spec: GraphQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[QueryChunk]:
        """
        Execute a streaming graph query.

        For large result sets; preferred over `query` when supported.
        """
        async def _stream_impl() -> AsyncIterator[QueryChunk]:
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

            async for chunk in await self._with_gates_stream(
                op="stream_query",
                ctx=ctx,
                agen_factory=lambda: agen_factory(),
                metric_extra={"dialect": spec.dialect or "none"},
            ):
                yield chunk
        
        return _stream_impl()

    async def upsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        """Batch upsert nodes with validation, gates, and optional timestamp management."""
        if not spec.nodes:
            raise BadRequest("nodes must not be empty")
        # Validate that properties are JSON-serializable (hard requirement)
        for n in spec.nodes:
            if n.properties is not None:
                self._validate_properties_map(n.properties)
        # Note: Other node validation (labels, etc.) is delegated to the adapter
        # to allow for soft failures and per-node error reporting

        async def _call() -> UpsertResult:
            if self._auto_timestamp_writes:
                now_ms = int(time.time() * 1000)
                nodes_with_timestamps: List[Node] = []

                for node in spec.nodes:
                    if node.created_at is None or node.updated_at is None:
                        nodes_with_timestamps.append(
                            Node(
                                id=node.id,
                                labels=node.labels,
                                properties=node.properties,
                                namespace=node.namespace,
                                created_at=node.created_at or now_ms,
                                updated_at=now_ms,  # Always update
                            )
                        )
                    else:
                        nodes_with_timestamps.append(node)

                result = await self._do_upsert_nodes(
                    UpsertNodesSpec(
                        nodes=nodes_with_timestamps,
                        namespace=spec.namespace,
                    ),
                    ctx=ctx,
                )
            else:
                result = await self._do_upsert_nodes(spec, ctx=ctx)

            # Invalidate relevant caches only if writes succeeded and we have a namespace
            if result.upserted_count > 0:
                namespace = self._extract_namespace_from_spec(spec)
                await self._invalidate_namespace_cache(namespace)

            return result

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
        """Batch upsert edges with validation, gates, and optional timestamp management."""
        if not spec.edges:
            raise BadRequest("edges must not be empty")
        # Validate critical fields that would break graph integrity
        for e in spec.edges:
            if not isinstance(e.label, str) or not e.label:
                raise BadRequest("edge.label must be a non-empty string")
            if not e.src or not isinstance(e.src, str):
                raise BadRequest("edge.src must be a non-empty string")
            if not e.dst or not isinstance(e.dst, str):
                raise BadRequest("edge.dst must be a non-empty string")
            if e.properties is not None:
                self._validate_properties_map(e.properties)

        async def _call() -> UpsertResult:
            if self._auto_timestamp_writes:
                now_ms = int(time.time() * 1000)
                edges_with_timestamps: List[Edge] = []

                for edge in spec.edges:
                    if edge.created_at is None or edge.updated_at is None:
                        edges_with_timestamps.append(
                            Edge(
                                id=edge.id,
                                src=edge.src,
                                dst=edge.dst,
                                label=edge.label,
                                properties=edge.properties,
                                namespace=edge.namespace,
                                created_at=edge.created_at or now_ms,
                                updated_at=now_ms,  # Always update
                            )
                        )
                    else:
                        edges_with_timestamps.append(edge)

                result = await self._do_upsert_edges(
                    UpsertEdgesSpec(
                        edges=edges_with_timestamps,
                        namespace=spec.namespace,
                    ),
                    ctx=ctx,
                )
            else:
                result = await self._do_upsert_edges(spec, ctx=ctx)

            # Invalidate relevant caches only if writes succeeded and we have a namespace
            if result.upserted_count > 0:
                namespace = self._extract_namespace_from_spec(spec)
                await self._invalidate_namespace_cache(namespace)

            return result

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
            result = await self._do_delete_nodes(spec, ctx=ctx)

            # Invalidate relevant caches only if deletes succeeded and we have a namespace
            if result.deleted_count > 0:
                namespace = self._extract_namespace_from_spec(spec)
                await self._invalidate_namespace_cache(namespace)

            return result

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
            result = await self._do_delete_edges(spec, ctx=ctx)

            # Invalidate relevant caches only if deletes succeeded and we have a namespace
            if result.deleted_count > 0:
                namespace = self._extract_namespace_from_spec(spec)
                await self._invalidate_namespace_cache(namespace)

            return result

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
        Uses pluggable cache interface.
        """
        caps = await self.capabilities()
        if not caps.supports_bulk_vertices:
            raise NotSupported("bulk_vertices is not supported by this adapter")
        if spec.limit <= 0:
            raise BadRequest("limit must be positive")
        if spec.filter is not None:
            self._validate_properties_map(spec.filter)

        async def _call() -> BulkVerticesResult:
            # Use pluggable cache interface for read-only scans
            if (
                not isinstance(self._cache, NoopCache)
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
        Includes optional strict operation-specific validation and smart cache invalidation.
        """
        caps = await self.capabilities()
        if not caps.supports_batch:
            raise NotSupported("batch is not supported by this adapter")
        if not ops:
            raise BadRequest("ops must not be empty")

        if caps.max_batch_ops is not None and len(ops) > caps.max_batch_ops:
            raise BadRequest(
                f"batch ops count {len(ops)} exceeds maximum of {caps.max_batch_ops}",
                details={"max_batch_ops": caps.max_batch_ops},
            )

        supported_ops = self._batch_supported_ops

        # Track batch composition for granular metrics
        nodes_upserted = 0
        edges_upserted = 0
        nodes_deleted = 0
        edges_deleted = 0

        # Enhanced operation-specific validation (opt-in strictness)
        for idx, op_spec in enumerate(ops):
            if not isinstance(op_spec.op, str) or not op_spec.op:
                raise BadRequest(
                    "each BatchOperation.op must be a non-empty string",
                    details={"index": idx},
                )
            if not isinstance(op_spec.args, Mapping):
                raise BadRequest(
                    "each BatchOperation.args must be a mapping",
                    details={"index": idx, "type": type(op_spec.args).__name__},
                )

            # Calculate granular metrics
            if op_spec.op == "upsert_nodes":
                nodes_upserted += len(op_spec.args.get("nodes", []))
            elif op_spec.op == "upsert_edges":
                edges_upserted += len(op_spec.args.get("edges", []))
            elif op_spec.op == "delete_nodes":
                nodes_deleted += len(op_spec.args.get("ids", []))
            elif op_spec.op == "delete_edges":
                edges_deleted += len(op_spec.args.get("ids", []))

            if self._strict_batch_validation:
                if op_spec.op not in supported_ops:
                    raise BadRequest(
                        f"unsupported batch operation '{op_spec.op}'",
                        details={"index": idx, "supported_ops": list(supported_ops)},
                    )

                # Operation-specific schema validation (only for known ops)
                if op_spec.op == "upsert_nodes" and "nodes" not in op_spec.args:
                    raise BadRequest(
                        "upsert_nodes requires 'nodes' array",
                        details={"index": idx},
                    )
                if op_spec.op == "upsert_edges" and "edges" not in op_spec.args:
                    raise BadRequest(
                        "upsert_edges requires 'edges' array",
                        details={"index": idx},
                    )
                if op_spec.op in {"delete_nodes", "delete_edges"} and "ids" not in op_spec.args:
                    raise BadRequest(
                        f"{op_spec.op} requires 'ids' array",
                        details={"index": idx},
                    )

        async def _call() -> BatchResult:
            result = await self._do_batch(ops, ctx=ctx)

            # Smart cache invalidation: only invalidate namespaces where writes actually succeeded
            namespaces_to_invalidate = set()
            for idx, op in enumerate(ops):
                if op.op in {"upsert_nodes", "upsert_edges", "delete_nodes", "delete_edges"}:
                    if self._batch_op_succeeded(op, result, idx):
                        namespace = self._extract_namespace_from_batch_op(op)
                        if namespace is not None:
                            namespaces_to_invalidate.add(namespace)

            # Invalidate each affected namespace
            for namespace in namespaces_to_invalidate:
                await self._invalidate_namespace_cache(namespace)

            return result

        # Emit granular batch metrics
        if nodes_upserted > 0:
            self._metrics.counter(component=self._component, name="batch_nodes_upserted", value=nodes_upserted)
        if edges_upserted > 0:
            self._metrics.counter(component=self._component, name="batch_edges_upserted", value=edges_upserted)
        if nodes_deleted > 0:
            self._metrics.counter(component=self._component, name="batch_nodes_deleted", value=nodes_deleted)
        if edges_deleted > 0:
            self._metrics.counter(component=self._component, name="batch_edges_deleted", value=edges_deleted)

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
            - Uses pluggable cache interface.
        """
        caps = await self.capabilities()
        if not caps.supports_schema:
            raise NotSupported("get_schema is not supported by this adapter")

        async def _call() -> GraphSchema:
            # Use pluggable cache interface
            if not isinstance(self._cache, NoopCache):
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
            if not isinstance(self._cache, NoopCache):
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
        """
        Health check with deadline + metrics; no cache.

        Returns a normalized, SIEM-safe summary view derived from the
        backend's raw health mapping; backends may expose additional
        fields but they are not surfaced here.
        """
        self._fail_if_deadline_expired(ctx)
        t0 = time.monotonic()
        try:
            res = await self._apply_deadline(self._do_health(ctx=ctx), ctx)
            self._record("health", t0, True, ctx=ctx)
            ok = bool(res.get("ok", True))
            status = res.get("status")
            if not status:
                status = "ok" if ok else "degraded"
            return {
                "ok": ok,
                "status": status,
                "server": str(res.get("server", "")),
                "version": str(res.get("version", "")),
                "namespaces": res.get("namespaces", {}),
                # pass-through common flags if provided upstream; derive sensible defaults
                "read_only": bool(res.get("read_only", False)),
                "degraded": bool(res.get("degraded", status != "ok")),
            }
        except GraphAdapterError as e:
            self._record("health", t0, False, code=e.code or type(e).__name__, ctx=ctx)
            raise
        except Exception as e:
            self._record("health", t0, False, code="UnhandledException", ctx=ctx)
            raise Unavailable("health check failed") from e

    # ---- Transaction and Traversal operations -------------------------------

    async def transaction(
        self,
        operations: List[BatchOperation],
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchResult:
        """
        Execute a batch of operations atomically as a transaction.

        This provides ACID guarantees for the entire batch - either all
        operations succeed or none are applied.

        Args:
            operations: List of batch operations to execute atomically
            ctx: Operation context including deadline and tracing

        Returns:
            BatchResult with transaction-level success/failure status

        Raises:
            NotSupported: If transactions are not supported by the backend
            BadRequest: If transaction operations are invalid
            DeadlineExceeded: If transaction exceeds context deadline
        """
        caps = await self.capabilities()
        if not caps.supports_transaction:
            raise NotSupported("transactions are not supported by this adapter")

        if not operations:
            raise BadRequest("transaction operations must not be empty")

        if caps.max_batch_ops is not None and len(operations) > caps.max_batch_ops:
            raise BadRequest(
                f"transaction ops count {len(operations)} exceeds maximum of {caps.max_batch_ops}",
                details={"max_batch_ops": caps.max_batch_ops},
            )

        # Validate transaction operations
        for idx, op in enumerate(operations):
            if not isinstance(op.op, str) or not op.op:
                raise BadRequest(
                    "each BatchOperation.op must be a non-empty string",
                    details={"index": idx},
                )
            if not isinstance(op.args, Mapping):
                raise BadRequest(
                    "each BatchOperation.args must be a mapping",
                    details={"index": idx, "type": type(op.args).__name__},
                )

        async def _call() -> BatchResult:
            result = await self._do_transaction(operations, ctx=ctx)

            # Invalidate caches for successful write operations in transaction
            if result.success:
                namespaces_to_invalidate = set()
                for idx, op in enumerate(operations):
                    if op.op in {"upsert_nodes", "upsert_edges", "delete_nodes", "delete_edges"}:
                        if self._batch_op_succeeded(op, result, idx):
                            namespace = self._extract_namespace_from_batch_op(op)
                            if namespace is not None:
                                namespaces_to_invalidate.add(namespace)

                # Invalidate each affected namespace
                for namespace in namespaces_to_invalidate:
                    await self._invalidate_namespace_cache(namespace)

            return result

        return await self._with_gates_unary(
            op="transaction",
            ctx=ctx,
            call=_call,
            metric_extra={"ops": len(operations)},
        )

    async def traversal(
        self,
        spec: GraphTraversalSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> TraversalResult:
        """
        Perform graph traversal starting from specified nodes.

        Traversal explores the graph structure by following relationships
        from starting nodes up to a specified depth.

        Args:
            spec: Traversal specification including start nodes, depth, filters
            ctx: Operation context including deadline and tracing

        Returns:
            TraversalResult containing discovered nodes, relationships, and paths

        Raises:
            NotSupported: If traversal operations are not supported
            BadRequest: If traversal specification is invalid
            DeadlineExceeded: If traversal exceeds context deadline
        """
        caps = await self.capabilities()
        if not caps.supports_traversal:
            raise NotSupported("traversal operations are not supported by this adapter")

        if not spec.start_nodes:
            raise BadRequest("traversal requires at least one start node")

        if caps.max_traversal_depth is not None and spec.max_depth > caps.max_traversal_depth:
            raise BadRequest(
                f"traversal depth {spec.max_depth} exceeds maximum of {caps.max_traversal_depth}",
                details={"max_traversal_depth": caps.max_traversal_depth},
            )

        # Validate filters
        if spec.node_filters is not None:
            self._validate_properties_map(spec.node_filters)
        if spec.relationship_filters is not None:
            self._validate_properties_map(spec.relationship_filters)

        async def _call() -> TraversalResult:
            # Use pluggable cache interface for read-only traversals
            if not isinstance(self._cache, NoopCache):
                key = self._make_cache_key(op="traversal", spec=spec, ctx=ctx)
                cached = await self._cache.get(key)
                if cached:
                    self._metrics.counter(
                        component=self._component,
                        name="cache_hits",
                        value=1,
                        extra={"op": "traversal"},
                    )
                    return cached
            res = await self._do_traversal(spec, ctx=ctx)
            if not isinstance(self._cache, NoopCache):
                await self._cache.set(key, res, ttl_s=self._cache_query_ttl_s)
            return res

        return await self._with_gates_unary(
            op="traversal",
            ctx=ctx,
            call=_call,
            metric_extra={
                "start_nodes": len(spec.start_nodes),
                "max_depth": spec.max_depth,
                "direction": spec.direction,
            },
        )

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

    async def _do_transaction(
        self,
        operations: List[BatchOperation],
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchResult:
        """
        Backend hook for atomic transaction execution.

        Adapters should override this to provide transaction support.
        Default implementation raises NotSupported.
        """
        raise NotSupported("transactions are not implemented by this adapter")

    async def _do_traversal(
        self,
        spec: GraphTraversalSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> TraversalResult:
        """
        Backend hook for graph traversal operations.

        Adapters should override this to provide traversal support.
        Default implementation raises NotSupported.
        """
        raise NotSupported("traversal operations are not implemented by this adapter")


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
        "code": "UNAVAILABLE",  # Wire-level code remains UNAVAILABLE
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
            - graph.transaction
            - graph.traversal

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
                try:
                    spec = GraphQuerySpec(**args)
                except (TypeError, ValueError) as spec_err:
                    raise BadRequest(f"Invalid query spec: {spec_err}") from spec_err
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

            if op == "graph.transaction":
                ops = [BatchOperation(**o) for o in args.get("operations", [])]
                res = await self._adapter.transaction(ops, ctx=ctx)
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

            if op == "graph.traversal":
                spec = GraphTraversalSpec(**args)
                res = await self._adapter.traversal(spec, ctx=ctx)
                return _success_to_wire(asdict(res), (time.monotonic() - t0) * 1000.0)

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
    "GraphTransaction",
    "GraphTraversalSpec",
    "TraversalResult",
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