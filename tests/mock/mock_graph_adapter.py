# examples/graph/mock_graph_adapter.py
# SPDX-License-Identifier: Apache-2.0
"""
Mock Graph adapter used in Corpus SDK example scripts and conformance tests.

Implements BaseGraphAdapter hooks with deterministic behavior:
- Capabilities per Graph Protocol V1.0
- Query + Stream Query (with chunks and final summary)
- Upsert/Delete for nodes and edges (idempotent deletes)
- Bulk vertex scanning with pagination
- Batch operations with per-op results and overflow hinting
- Health reporting (ctx-driven degraded mode)
- Optional deterministic failure injection via ctx.attrs["simulate_error"]

No legacy overloads. Pure V1 surface only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Tuple

from corpus_sdk.graph.graph_base import (
    BaseGraphAdapter,
    GraphCapabilities,
    OperationContext,
    GraphID,
    Node,
    Edge,
    GraphQuerySpec,
    UpsertNodesSpec,
    UpsertEdgesSpec,
    DeleteNodesSpec,
    DeleteEdgesSpec,
    BulkVerticesSpec,
    BulkVerticesResult,
    BatchOperation,
    BatchResult,
    GraphSchema,
    QueryResult,
    QueryChunk,
    UpsertResult,
    DeleteResult,
    # Errors
    BadRequest,
    NotSupported,
    Unavailable,
    ResourceExhausted,
    TransientNetwork,
)


@dataclass
class MockGraphAdapter(BaseGraphAdapter):
    """A mock Graph adapter for Graph Protocol V1.0 demonstrations & tests."""

    # Tunables (deterministic defaults)
    name: str = "mock-graph"
    supported_dialects: Tuple[str, ...] = ("cypher", "opencypher")
    supports_stream: bool = True
    supports_bulk: bool = True
    supports_batch_ops: bool = True
    supports_schema_ops: bool = True
    max_ops_per_batch: int = 1000
    latency_ms: Tuple[int, int] = (2, 5)  # small, bounded sleeps
    failure_rate: float = 0.0  # keep 0.0 for conformance (can be raised for demos)

    def __post_init__(self) -> None:
        # Initialize the base class
        super().__init__()
        
        # Configuration validation
        if not isinstance(self.supported_dialects, tuple) or not self.supported_dialects:
            raise ValueError("supported_dialects must be a non-empty tuple of strings")
        if not (0.0 <= float(self.failure_rate) <= 1.0):
            raise ValueError("failure_rate must be between 0 and 1")
        lo, hi = self.latency_ms
        if lo < 0 or hi < lo:
            raise ValueError("Invalid latency range (min >= 0 and max >= min)")

    # -------------------------------------------------------------------------
    # Capabilities & Health
    # -------------------------------------------------------------------------
    async def _do_capabilities(self) -> GraphCapabilities:
        return GraphCapabilities(
            server=self.name,
            version="1.0.0",
            supported_query_dialects=self.supported_dialects,
            supports_stream_query=self.supports_stream,
            supports_namespaces=True,
            supports_property_filters=True,
            supports_bulk_vertices=self.supports_bulk,
            supports_batch=self.supports_batch_ops,
            supports_schema=self.supports_schema_ops,
            idempotent_writes=False,
            supports_multi_tenant=True,
            supports_deadline=True,
        )

    async def _do_health(self, *, ctx: Optional[OperationContext] = None) -> Mapping[str, Any]:
        # Deterministic; allow ctx to force degraded/ok for tests
        status = (ctx and ctx.attrs.get("health")) or "ok"
        ok = (status == "ok")
        return {
            "ok": ok,
            "server": self.name,
            "version": "1.0.0",
            "namespaces": {"default": ("degraded" if not ok else "ok")},
        }

    # -------------------------------------------------------------------------
    # Query (unary)
    # -------------------------------------------------------------------------
    async def _do_query(
        self,
        spec: GraphQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        self._maybe_fail(op="query", ctx=ctx)

        if not isinstance(spec.text, str) or not spec.text.strip():
            raise BadRequest("query.text must be a non-empty string")

        caps = await self._do_capabilities()
        if spec.dialect and caps.supported_query_dialects:
            if spec.dialect not in caps.supported_query_dialects:
                raise NotSupported(
                    f"dialect '{spec.dialect}' not supported",
                    details={"supported_query_dialects": caps.supported_query_dialects},
                )

        await self._sleep()

        # Deterministic records based on (text, params)
        seed = self._stable_int((spec.text, self._stable_params(spec.params))) % 3 + 1
        records = [{"row": i + 1, "ok": True, "dialect": spec.dialect or "native"} for i in range(seed)]

        summary: Dict[str, Any] = {
            "rows": len(records),
            "consumed_ms": self._avg_latency_ms(),
        }
        return QueryResult(
            records=records,
            summary=summary,
            dialect=spec.dialect,
            namespace=spec.namespace,
        )

    # -------------------------------------------------------------------------
    # Stream Query
    # -------------------------------------------------------------------------
    async def _do_stream_query(
        self,
        spec: GraphQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[QueryChunk]:
        self._maybe_fail(op="stream_query", ctx=ctx)

        caps = await self._do_capabilities()
        if not caps.supports_stream_query:
            raise NotSupported("stream_query is not supported by this adapter")

        if not isinstance(spec.text, str) or not spec.text.strip():
            raise BadRequest("query.text must be a non-empty string")

        if spec.dialect and caps.supported_query_dialects:
            if spec.dialect not in caps.supported_query_dialects:
                raise NotSupported(
                    f"dialect '{spec.dialect}' not supported",
                    details={"supported_query_dialects": caps.supported_query_dialects},
                )

        total = self._stable_int(spec.text) % 4 + 2  # 2..5 chunks deterministically
        for i in range(total - 1):
            await self._sleep()
            yield QueryChunk(records=[{"row": i + 1, "ok": True}], is_final=False)

        # final chunk with summary
        await self._sleep()
        yield QueryChunk(
            records=[{"row": total, "ok": True}],
            is_final=True,
            summary={"total_chunks": total, "dialect": spec.dialect or "native"},
        )

    # -------------------------------------------------------------------------
    # Upserts
    # -------------------------------------------------------------------------
    async def _do_upsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        self._maybe_fail(op="upsert_nodes", ctx=ctx)

        upserted = 0
        failures: List[Mapping[str, Any]] = []

        for idx, n in enumerate(spec.nodes):
            try:
                # Validation: at least one non-empty label; properties JSON-serializable
                if not n.labels or any((not isinstance(l, str) or not l) for l in n.labels):
                    raise BadRequest("node.labels must be a non-empty tuple of non-empty strings")
                json.dumps(n.properties or {})
                upserted += 1
            except Exception as e:
                failures.append(
                    {
                        "index": idx,
                        "id": str(n.id) if getattr(n, "id", None) else None,
                        "error": type(e).__name__,
                        "code": getattr(e, "code", None) or type(e).__name__.upper(),
                        "message": str(e),
                    }
                )

        await self._sleep()
        return UpsertResult(
            upserted_count=upserted,
            failed_count=len(failures),
            failures=failures,
        )

    async def _do_upsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        self._maybe_fail(op="upsert_edges", ctx=ctx)

        upserted = 0
        failures: List[Mapping[str, Any]] = []

        for idx, e in enumerate(spec.edges):
            try:
                if not isinstance(e.label, str) or not e.label:
                    raise BadRequest("edge.label must be a non-empty string")
                if not str(e.src) or not str(e.dst):
                    raise BadRequest("edge src/dst must be provided")
                json.dumps(e.properties or {})
                upserted += 1
            except Exception as ex:
                failures.append(
                    {
                        "index": idx,
                        "id": str(e.id) if getattr(e, "id", None) else None,
                        "error": type(ex).__name__,
                        "code": getattr(ex, "code", None) or type(ex).__name__.upper(),
                        "message": str(ex),
                    }
                )

        await self._sleep()
        return UpsertResult(
            upserted_count=upserted,
            failed_count=len(failures),
            failures=failures,
        )

    # -------------------------------------------------------------------------
    # Deletes (idempotent)
    # -------------------------------------------------------------------------
    async def _do_delete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        self._maybe_fail(op="delete_nodes", ctx=ctx)

        # Idempotent: treat unknown IDs as deleted=0 without error
        deleted = len(spec.ids or [])
        failures: List[Mapping[str, Any]] = []

        await self._sleep()
        return DeleteResult(
            deleted_count=deleted,
            failed_count=len(failures),
            failures=failures,
        )

    async def _do_delete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        self._maybe_fail(op="delete_edges", ctx=ctx)

        deleted = len(spec.ids or [])
        failures: List[Mapping[str, Any]] = []

        await self._sleep()
        return DeleteResult(
            deleted_count=deleted,
            failed_count=len(failures),
            failures=failures,
        )

    # -------------------------------------------------------------------------
    # Bulk Vertices (scan/paginate)
    # -------------------------------------------------------------------------
    async def _do_bulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BulkVerticesResult:
        self._maybe_fail(op="bulk_vertices", ctx=ctx)

        caps = await self._do_capabilities()
        if not caps.supports_bulk_vertices:
            raise NotSupported("bulk_vertices is not supported by this adapter")

        if spec.limit <= 0:
            raise BadRequest("limit must be positive")

        # Deterministic dataset per namespace
        total = 250  # pretend dataset size
        start = int(spec.cursor or 0)
        end = min(start + spec.limit, total)

        nodes: List[Node] = []
        for i in range(start, end):
            # Stable node ID
            nid = GraphID(f"v:{spec.namespace or 'default'}:{i}")
            nodes.append(
                Node(
                    id=nid,
                    labels=("Vertex",),
                    properties={"i": i},
                    namespace=spec.namespace,
                )
            )
            if (i - start + 1) % 50 == 0:
                await asyncio.sleep(0)

        next_cursor = str(end) if end < total else None
        has_more = end < total

        await self._sleep()
        return BulkVerticesResult(
            nodes=nodes,
            next_cursor=next_cursor,
            has_more=has_more,
        )

    # -------------------------------------------------------------------------
    # Batch
    # -------------------------------------------------------------------------
    async def _do_batch(
        self,
        ops: List[BatchOperation],
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchResult:
        self._maybe_fail(op="batch", ctx=ctx)

        caps = await self._do_capabilities()
        if not caps.supports_batch:
            raise NotSupported("batch is not supported by this adapter")

        if self.max_ops_per_batch and len(ops) > self.max_ops_per_batch:
            raise BadRequest(
                f"batch ops size {len(ops)} exceeds maximum {self.max_ops_per_batch}",
                details={"suggested_batch_reduction": int(math.ceil(len(ops) * 0.5))},
            )

        results: List[Any] = []
        for idx, op in enumerate(ops):
            try:
                kind = op.op
                args = dict(op.args or {})

                if kind == "upsert_nodes":
                    nodes = [Node(**n) for n in args.get("nodes", [])]
                    spec = UpsertNodesSpec(nodes=nodes, namespace=args.get("namespace"))
                    res = await self._do_upsert_nodes(spec, ctx=ctx)
                    results.append({"ok": True, "result": {"upserted": res.upserted_count, "failed": res.failed_count}})

                elif kind == "upsert_edges":
                    edges = [Edge(**e) for e in args.get("edges", [])]
                    spec = UpsertEdgesSpec(edges=edges, namespace=args.get("namespace"))
                    res = await self._do_upsert_edges(spec, ctx=ctx)
                    results.append({"ok": True, "result": {"upserted": res.upserted_count, "failed": res.failed_count}})

                elif kind == "delete_nodes":
                    ids = [GraphID(i) for i in args.get("ids", [])]
                    spec = DeleteNodesSpec(ids=ids, namespace=args.get("namespace"), filter=args.get("filter"))
                    res = await self._do_delete_nodes(spec, ctx=ctx)
                    results.append({"ok": True, "result": {"deleted": res.deleted_count}})

                elif kind == "delete_edges":
                    ids = [GraphID(i) for i in args.get("ids", [])]
                    spec = DeleteEdgesSpec(ids=ids, namespace=args.get("namespace"), filter=args.get("filter"))
                    res = await self._do_delete_edges(spec, ctx=ctx)
                    results.append({"ok": True, "result": {"deleted": res.deleted_count}})

                elif kind == "query":
                    qspec = GraphQuerySpec(**args)
                    qres = await self._do_query(qspec, ctx=ctx)
                    results.append({"ok": True, "result": {"rows": len(qres.records)}})

                else:
                    results.append(
                        {
                            "ok": False,
                            "error": "NotSupported",
                            "code": "NOT_SUPPORTED",
                            "message": f"unknown batch op '{kind}'",
                            "index": idx,
                        }
                    )
            except (BadRequest, NotSupported, Unavailable, ResourceExhausted, TransientNetwork) as e:
                results.append(
                    {
                        "ok": False,
                        "error": type(e).__name__,
                        "code": getattr(e, "code", None) or type(e).__name__.upper(),
                        "message": str(e),
                        "index": idx,
                    }
                )
            # let unexpected exceptions bubble via BaseGraphAdapter error mapping

        await self._sleep()
        return BatchResult(results=results)

    # -------------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------------
    async def _do_get_schema(self, *, ctx: Optional[OperationContext] = None) -> GraphSchema:
        caps = await self._do_capabilities()
        if not caps.supports_schema:
            raise NotSupported("get_schema is not supported by this adapter")

        await self._sleep()
        return GraphSchema(
            nodes={
                "User": {"properties": {"id": "string", "name": "string"}},
                "Doc": {"properties": {"id": "string", "title": "string"}},
            },
            edges={
                "READ": {"from": "User", "to": "Doc"},
                "LINKS": {"from": "Doc", "to": "Doc"},
            },
            metadata={"version": "1.0", "generated_by": self.name},
        )

    # -------------------------------------------------------------------------
    # Internals (helpers)
    # -------------------------------------------------------------------------
    def _avg_latency_ms(self) -> int:
        lo, hi = self.latency_ms
        return int((lo + hi) / 2)

    async def _sleep(self) -> None:
        # Use fixed small sleep to keep deterministic and deadline-friendly
        await asyncio.sleep(self._avg_latency_ms() / 1000.0)

    def _stable_int(self, obj: Any) -> int:
        h = hashlib.sha256(repr(obj).encode("utf-8")).hexdigest()
        return int(h[:12], 16)

    def _stable_params(self, params: Optional[Mapping[str, Any]]) -> Tuple[Tuple[str, Any], ...]:
        if not params:
            return ()
        # Make JSON-safe, sorted tuple for stable hashing
        def _jsonable(v: Any) -> Any:
            try:
                json.dumps(v)
                return v
            except Exception:
                return repr(v)
        return tuple(sorted((str(k), _jsonable(v)) for k, v in params.items()))

    def _maybe_fail(self, *, op: str, ctx: Optional[OperationContext]) -> None:
        """
        Deterministic, opt-in failure injection:
          ctx.attrs["simulate_error"] ∈ {"unavailable","rate_limited","transient"}
        """
        key = ctx and ctx.attrs.get("simulate_error")
        if key == "unavailable":
            raise Unavailable(f"Mocked {op} unavailable", retry_after_ms=500)
        if key == "rate_limited":
            raise ResourceExhausted(f"Mocked {op} rate-limited", retry_after_ms=800)
        if key == "transient":
            raise TransientNetwork(f"Mocked {op} transient network", retry_after_ms=600)
        # No RNG-based failures by default (keeps conformance runs stable)


# ---------------------------------------------------------------------------
# Minimal demo (optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def _demo() -> None:
        adapter = MockGraphAdapter()
        caps = await adapter.capabilities()
        print("[CAPABILITIES]", caps)

        health = await adapter.health()
        print("[HEALTH]", health)

        qres = await adapter.query(GraphQuerySpec(text="MATCH (n) RETURN n LIMIT 3", dialect="cypher"))
        print("[QUERY rows]", len(qres.records))

        print("[STREAM]")
        async for chunk in adapter.stream_query(GraphQuerySpec(text="MATCH () RETURN 1", dialect="cypher")):
            print("  chunk", chunk.records, "final:", chunk.is_final)

        ures = await adapter.upsert_nodes(UpsertNodesSpec(nodes=[Node(id=GraphID("n1"), labels=("User",), properties={"id": "u1"})]))
        print("[UPSERT NODES]", ures.upserted_count)

        bres = await adapter.bulk_vertices(BulkVerticesSpec(namespace="default", limit=5, cursor=None))
        print("[BULK] nodes", len(bres.nodes), "has_more", bres.has_more)

        batch = await adapter.batch(
            [
                BatchOperation(op="query", args={"text": "RETURN 1", "dialect": "cypher"}),
                BatchOperation(op="delete_nodes", args={"ids": ["v:default:1", "v:default:2"]}),
                BatchOperation(op="unknown_op", args={}),
            ]
        )
        print("[BATCH] results", len(batch.results))

        schema = await adapter.get_schema()
        print("[SCHEMA] keys", list(schema.nodes.keys()), list(schema.edges.keys()))

    asyncio.run(_demo())
