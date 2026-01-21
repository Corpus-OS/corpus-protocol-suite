# examples/graph/mock_graph_adapter.py
# SPDX-License-Identifier: Apache-2.0
"""
Mock Graph adapter used in Corpus SDK example scripts and conformance tests.

Implements BaseGraphAdapter hooks with deterministic behavior:
- Capabilities per Graph Protocol V1.0
- Query + Stream Query (with chunks and final summary)
- Upsert/Delete for nodes and edges (idempotent deletes)
- Bulk vertex scanning with pagination
- Batch operations with per-op mapping envelopes
- Transaction operations (configurable; deterministic)
- Traversal operations (configurable; deterministic)
- Health reporting (ctx-driven degraded mode)
- Optional deterministic failure injection via ctx.attrs["simulate_error"]

No legacy overloads. Pure V1 surface only.

ALIGNMENT NOTES (Non-breaking):
- This mock aligns its capabilities with the features it actually implements.
- Batch/transaction per-op results are mapping envelopes {"ok": True, "result": {...}} to
  interoperate with BaseGraphAdapter batch cache invalidation unwrapping logic.
- Batch/transaction processing avoids nested gates by calling hook methods directly while still
  applying minimal contract-level validation for dialects and JSON-serializable fields.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from dataclasses import asdict
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
    GraphTraversalSpec,
    TraversalResult,
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


class MockGraphAdapter(BaseGraphAdapter):
    """
    A mock Graph adapter for Graph Protocol V1.0 demonstrations & tests.

    This mock is deterministic by default (failure_rate=0.0, bounded sleeps).
    It supports multi-tenant capability advertisement and optional failure injection via ctx.attrs.
    """

    # Tunables (deterministic defaults)
    name: str
    supported_dialects: Tuple[str, ...]
    supports_stream: bool
    supports_bulk: bool
    supports_batch_ops: bool
    supports_schema_ops: bool
    supports_transaction_ops: bool
    supports_traversal_ops: bool
    max_traversal_depth: Optional[int]
    max_ops_per_batch: int
    latency_ms: Tuple[int, int]  # small, bounded sleeps
    failure_rate: float  # keep 0.0 for conformance (can be raised for demos)

    def __init__(
        self,
        *,
        name: str = "mock-graph",
        supported_dialects: Tuple[str, ...] = ("cypher", "opencypher"),
        supports_stream: bool = True,
        supports_bulk: bool = True,
        supports_batch_ops: bool = True,
        supports_schema_ops: bool = True,
        # NEW: configurable advanced ops
        supports_transaction_ops: bool = False,
        supports_traversal_ops: bool = False,
        max_traversal_depth: Optional[int] = None,
        max_ops_per_batch: int = 1000,
        latency_ms: Tuple[int, int] = (2, 5),  # small, bounded sleeps
        failure_rate: float = 0.0,  # deterministic default for conformance
        # BaseGraphAdapter configuration passthrough (keeps router/test control)
        metrics: Any = None,
        mode: str = "thin",
        deadline_policy: Any = None,
        breaker: Any = None,
        cache: Any = None,
        limiter: Any = None,
        cache_query_ttl_s: int = 30,
        cache_caps_ttl_s: int = 30,
        cache_bulk_vertices_ttl_s: int = 30,
        cache_schema_ttl_s: int = 60,
        stream_deadline_check_every_n_chunks: int = 10,
        auto_timestamp_writes: bool = False,
        strict_batch_validation: bool = False,
        batch_supported_ops: Any = None,
        allow_self_loops: bool = True,
    ) -> None:
        # Initialize BaseGraphAdapter with explicit configuration to avoid drift.
        super().__init__(
            metrics=metrics,
            mode=mode,
            deadline_policy=deadline_policy,
            breaker=breaker,
            cache=cache,
            limiter=limiter,
            cache_query_ttl_s=cache_query_ttl_s,
            cache_caps_ttl_s=cache_caps_ttl_s,
            cache_bulk_vertices_ttl_s=cache_bulk_vertices_ttl_s,
            cache_schema_ttl_s=cache_schema_ttl_s,
            stream_deadline_check_every_n_chunks=stream_deadline_check_every_n_chunks,
            auto_timestamp_writes=auto_timestamp_writes,
            strict_batch_validation=strict_batch_validation,
            batch_supported_ops=batch_supported_ops,
            allow_self_loops=allow_self_loops,
        )

        # Store tunables
        self.name = name
        self.supported_dialects = supported_dialects
        self.supports_stream = bool(supports_stream)
        self.supports_bulk = bool(supports_bulk)
        self.supports_batch_ops = bool(supports_batch_ops)
        self.supports_schema_ops = bool(supports_schema_ops)
        self.supports_transaction_ops = bool(supports_transaction_ops)
        self.supports_traversal_ops = bool(supports_traversal_ops)
        self.max_traversal_depth = max_traversal_depth if max_traversal_depth is None else int(max_traversal_depth)
        self.max_ops_per_batch = int(max_ops_per_batch)
        self.latency_ms = latency_ms
        self.failure_rate = float(failure_rate)

        # Configuration validation (fast, deterministic)
        if not isinstance(self.supported_dialects, tuple) or not self.supported_dialects:
            raise ValueError("supported_dialects must be a non-empty tuple of strings")
        if any((not isinstance(d, str) or not d) for d in self.supported_dialects):
            raise ValueError("supported_dialects must contain only non-empty strings")
        if not (0.0 <= self.failure_rate <= 1.0):
            raise ValueError("failure_rate must be between 0 and 1")
        lo, hi = self.latency_ms
        if lo < 0 or hi < lo:
            raise ValueError("Invalid latency range (min >= 0 and max >= min)")
        if self.max_ops_per_batch <= 0:
            raise ValueError("max_ops_per_batch must be positive")
        if self.max_traversal_depth is not None and self.max_traversal_depth < 1:
            raise ValueError("max_traversal_depth must be >= 1 when provided")

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
            max_batch_ops=self.max_ops_per_batch,
            supports_transaction=self.supports_transaction_ops,
            supports_traversal=self.supports_traversal_ops,
            max_traversal_depth=self.max_traversal_depth,
            supports_path_queries=False,
        )

    async def _do_health(self, *, ctx: Optional[OperationContext] = None) -> Mapping[str, Any]:
        status = (ctx and ctx.attrs.get("health")) or "ok"
        ok = (status == "ok")
        return {
            "ok": ok,
            "status": status,
            "server": self.name,
            "version": "1.0.0",
            "namespaces": {"default": ("ok" if ok else "degraded")},
            "read_only": False,
            "degraded": not ok,
        }

    # -------------------------------------------------------------------------
    # Query (unary hook)
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
    # Stream Query (hook)
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

        await self._sleep()
        yield QueryChunk(
            records=[{"row": total, "ok": True}],
            is_final=True,
            summary={"total_chunks": total, "dialect": spec.dialect or "native"},
        )

    # -------------------------------------------------------------------------
    # Upserts (hooks)
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
                if n.labels:
                    if any((not isinstance(l, str) or not l) for l in n.labels):
                        raise BadRequest("node.labels must be a tuple of non-empty strings when provided")
                json.dumps(n.properties or {})
                upserted += 1
            except Exception as e:
                failures.append(
                    {
                        "index": idx,
                        "id": str(n.id) if getattr(n, "id", None) else None,
                        "error": type(e).__name__,
                        "code": getattr(e, "code", None) or type(e).__name__.upper(),
                        "message": str(e) or type(e).__name__,
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
                        "message": str(ex) or type(ex).__name__,
                    }
                )

        await self._sleep()
        return UpsertResult(
            upserted_count=upserted,
            failed_count=len(failures),
            failures=failures,
        )

    # -------------------------------------------------------------------------
    # Deletes (idempotent hooks)
    # -------------------------------------------------------------------------
    async def _do_delete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        self._maybe_fail(op="delete_nodes", ctx=ctx)

        deleted = 0
        failures: List[Mapping[str, Any]] = []

        if spec.ids:
            deleted += len(spec.ids)

        if spec.filter:
            deleted += (self._stable_int(("delete_nodes", spec.namespace, self._stable_params(spec.filter))) % 5)

        await self._sleep()
        return DeleteResult(deleted_count=deleted, failed_count=len(failures), failures=failures)

    async def _do_delete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        self._maybe_fail(op="delete_edges", ctx=ctx)

        deleted = 0
        failures: List[Mapping[str, Any]] = []

        if spec.ids:
            deleted += len(spec.ids)

        if spec.filter:
            deleted += (self._stable_int(("delete_edges", spec.namespace, self._stable_params(spec.filter))) % 5)

        await self._sleep()
        return DeleteResult(deleted_count=deleted, failed_count=len(failures), failures=failures)

    # -------------------------------------------------------------------------
    # Bulk Vertices (scan/paginate hook)
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

        total = 250
        start = int(spec.cursor or 0)
        end = min(start + spec.limit, total)

        nodes: List[Node] = []
        for i in range(start, end):
            nid = GraphID(f"v:{spec.namespace or 'default'}:{i}")
            nodes.append(Node(id=nid, labels=("Vertex",), properties={"i": i}, namespace=spec.namespace))
            if (i - start + 1) % 50 == 0:
                await asyncio.sleep(0)

        next_cursor = str(end) if end < total else None
        has_more = end < total

        await self._sleep()
        return BulkVerticesResult(nodes=nodes, next_cursor=next_cursor, has_more=has_more)

    # -------------------------------------------------------------------------
    # Shared op executor for batch/transaction (no nested gates)
    # -------------------------------------------------------------------------
    async def _execute_ops_as_envelopes(
        self,
        ops: List[BatchOperation],
        *,
        ctx: Optional[OperationContext],
        caps: GraphCapabilities,
    ) -> List[Mapping[str, Any]]:
        results: List[Mapping[str, Any]] = []

        for idx, op in enumerate(ops):
            try:
                kind = op.op
                args = dict(op.args or {})

                if kind == "query":
                    qspec = GraphQuerySpec(**args)
                    if qspec.params is not None:
                        self._validate_properties_map(qspec.params)
                    if qspec.dialect and caps.supported_query_dialects and qspec.dialect not in caps.supported_query_dialects:
                        raise NotSupported(
                            f"dialect '{qspec.dialect}' not supported",
                            details={"supported_query_dialects": caps.supported_query_dialects},
                        )
                    qres = await self._do_query(qspec, ctx=ctx)
                    results.append(
                        {
                            "ok": True,
                            "result": {
                                "rows": len(qres.records),
                                "dialect": qres.dialect or (qspec.dialect or "native"),
                                "namespace": qres.namespace,
                            },
                        }
                    )

                elif kind == "upsert_nodes":
                    nodes_raw = args.get("nodes", []) or []
                    if not isinstance(nodes_raw, list):
                        raise BadRequest("upsert_nodes requires 'nodes' array", details={"index": idx})
                    if not nodes_raw:
                        raise BadRequest("upsert_nodes requires non-empty 'nodes' array", details={"index": idx})

                    nodes: List[Node] = []
                    for j, raw in enumerate(nodes_raw):
                        if not isinstance(raw, Mapping):
                            raise BadRequest("upsert_nodes nodes[*] must be an object", details={"index": idx, "node_index": j})
                        d = dict(raw)
                        nid = d.get("id")
                        if not isinstance(nid, str) or not nid:
                            raise BadRequest("nodes[*].id must be a non-empty string", details={"index": idx, "node_index": j})
                        d["id"] = GraphID(nid)
                        if "labels" in d and d["labels"] is not None:
                            if isinstance(d["labels"], list):
                                d["labels"] = tuple(d["labels"])
                            elif isinstance(d["labels"], tuple):
                                pass
                            else:
                                raise BadRequest(
                                    "nodes[*].labels must be a list or tuple of strings",
                                    details={"index": idx, "node_index": j, "type": type(d["labels"]).__name__},
                                )
                        nodes.append(Node(**d))

                    spec = UpsertNodesSpec(nodes=nodes, namespace=args.get("namespace"))
                    res = await self._do_upsert_nodes(spec, ctx=ctx)
                    results.append({"ok": True, "result": asdict(res)})

                elif kind == "upsert_edges":
                    edges_raw = args.get("edges", []) or []
                    if not isinstance(edges_raw, list):
                        raise BadRequest("upsert_edges requires 'edges' array", details={"index": idx})
                    if not edges_raw:
                        raise BadRequest("upsert_edges requires non-empty 'edges' array", details={"index": idx})

                    edges: List[Edge] = []
                    for j, raw in enumerate(edges_raw):
                        if not isinstance(raw, Mapping):
                            raise BadRequest("upsert_edges edges[*] must be an object", details={"index": idx, "edge_index": j})
                        d = dict(raw)
                        for field in ("id", "src", "dst", "label"):
                            if field not in d:
                                raise BadRequest(
                                    f"edges[*].{field} is required",
                                    details={"index": idx, "edge_index": j, "field": field},
                                )
                        for id_field in ("id", "src", "dst"):
                            v = d.get(id_field)
                            if not isinstance(v, str) or not v:
                                raise BadRequest(
                                    f"edges[*].{id_field} must be a non-empty string",
                                    details={"index": idx, "edge_index": j, "field": id_field},
                                )
                            d[id_field] = GraphID(v)
                        if not isinstance(d.get("label"), str) or not d.get("label"):
                            raise BadRequest("edges[*].label must be a non-empty string", details={"index": idx, "edge_index": j})

                        edge = Edge(**d)
                        self._validate_edge(edge)
                        edges.append(edge)

                    spec = UpsertEdgesSpec(edges=edges, namespace=args.get("namespace"))
                    res = await self._do_upsert_edges(spec, ctx=ctx)
                    results.append({"ok": True, "result": asdict(res)})

                elif kind == "delete_nodes":
                    ids_raw = args.get("ids", []) or []
                    if not isinstance(ids_raw, list):
                        raise BadRequest("delete_nodes requires 'ids' array", details={"index": idx})
                    ids = []
                    for j, v in enumerate(ids_raw):
                        if not isinstance(v, str):
                            raise BadRequest("delete_nodes ids[*] must be strings", details={"index": idx, "id_index": j})
                        ids.append(GraphID(v))

                    filt = args.get("filter")
                    if filt is not None:
                        self._validate_properties_map(filt)
                    if not ids and not filt:
                        raise BadRequest("delete_nodes requires ids or filter", details={"index": idx})

                    spec = DeleteNodesSpec(ids=ids, namespace=args.get("namespace"), filter=filt)
                    res = await self._do_delete_nodes(spec, ctx=ctx)
                    results.append({"ok": True, "result": asdict(res)})

                elif kind == "delete_edges":
                    ids_raw = args.get("ids", []) or []
                    if not isinstance(ids_raw, list):
                        raise BadRequest("delete_edges requires 'ids' array", details={"index": idx})
                    ids = []
                    for j, v in enumerate(ids_raw):
                        if not isinstance(v, str):
                            raise BadRequest("delete_edges ids[*] must be strings", details={"index": idx, "id_index": j})
                        ids.append(GraphID(v))

                    filt = args.get("filter")
                    if filt is not None:
                        self._validate_properties_map(filt)
                    if not ids and not filt:
                        raise BadRequest("delete_edges requires ids or filter", details={"index": idx})

                    spec = DeleteEdgesSpec(ids=ids, namespace=args.get("namespace"), filter=filt)
                    res = await self._do_delete_edges(spec, ctx=ctx)
                    results.append({"ok": True, "result": asdict(res)})

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
                        "message": str(e) or type(e).__name__,
                        "index": idx,
                    }
                )

        return results

    # -------------------------------------------------------------------------
    # Batch (hook)
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

        results = await self._execute_ops_as_envelopes(ops, ctx=ctx, caps=caps)
        await self._sleep()
        return BatchResult(results=list(results))

    # -------------------------------------------------------------------------
    # Transaction (hook)
    # -------------------------------------------------------------------------
    async def _do_transaction(
        self,
        operations: List[BatchOperation],
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchResult:
        self._maybe_fail(op="transaction", ctx=ctx)

        caps = await self._do_capabilities()
        if not caps.supports_transaction:
            raise NotSupported("transactions are not supported by this adapter")

        results = await self._execute_ops_as_envelopes(operations, ctx=ctx, caps=caps)

        # Deterministic "atomic" success marker:
        # if any op produced ok:false, mark entire transaction failed.
        all_ok = all(bool(r.get("ok")) for r in results)
        tx_id = f"tx:{self._stable_int((self.name, getattr(ctx, 'request_id', None), len(operations)))}"

        await self._sleep()
        return BatchResult(
            results=list(results),
            success=all_ok,
            error=None if all_ok else "transaction failed",
            transaction_id=tx_id,
        )

    # -------------------------------------------------------------------------
    # Traversal (hook)
    # -------------------------------------------------------------------------
    async def _do_traversal(
        self,
        spec: GraphTraversalSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> TraversalResult:
        self._maybe_fail(op="traversal", ctx=ctx)

        caps = await self._do_capabilities()
        if not caps.supports_traversal:
            raise NotSupported("traversal operations are not supported by this adapter")

        # Deterministic traversal: for each start node, create a single path of length max_depth.
        nodes: List[Node] = []
        edges: List[Edge] = []
        paths: List[List[Any]] = []

        for s_idx, start in enumerate(spec.start_nodes):
            # Start node
            start_id = GraphID(start)
            n0 = Node(
                id=start_id,
                labels=("Vertex",),
                properties={"id": start, "depth": 0},
                namespace=spec.namespace,
            )
            nodes.append(n0)

            prev = n0
            path: List[Any] = [n0]

            for depth in range(1, int(spec.max_depth) + 1):
                # Create a deterministic neighbor id
                nxt_raw = f"{start}|d={depth}|i={s_idx}"
                nxt_id = GraphID(f"v:{spec.namespace or 'default'}:{self._stable_int(nxt_raw)}")
                nxt = Node(
                    id=nxt_id,
                    labels=("Vertex",),
                    properties={"id": str(nxt_id), "depth": depth},
                    namespace=spec.namespace,
                )
                nodes.append(nxt)

                e_id = GraphID(f"e:{prev.id}->{nxt.id}:{depth}")
                rel = Edge(
                    id=e_id,
                    src=prev.id,
                    dst=nxt.id,
                    label="TRAVERSE",
                    properties={"depth": depth, "direction": spec.direction},
                    namespace=spec.namespace,
                )
                edges.append(rel)

                path.extend([rel, nxt])
                prev = nxt

            paths.append(path)

        # Deduplicate nodes by id while preserving deterministic order
        seen: set[str] = set()
        uniq_nodes: List[Node] = []
        for n in nodes:
            if str(n.id) in seen:
                continue
            seen.add(str(n.id))
            uniq_nodes.append(n)

        summary: Dict[str, Any] = {
            "start_nodes": list(spec.start_nodes),
            "max_depth": int(spec.max_depth),
            "direction": spec.direction,
            "nodes": len(uniq_nodes),
            "relationships": len(edges),
        }

        await self._sleep()
        return TraversalResult(
            nodes=uniq_nodes,
            relationships=edges,
            paths=paths,
            summary=summary,
            namespace=spec.namespace,
        )

    # -------------------------------------------------------------------------
    # Schema (hook)
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
        await asyncio.sleep(self._avg_latency_ms() / 1000.0)

    def _stable_int(self, obj: Any) -> int:
        h = hashlib.sha256(repr(obj).encode("utf-8")).hexdigest()
        return int(h[:12], 16)

    def _stable_params(self, params: Optional[Mapping[str, Any]]) -> Tuple[Tuple[str, Any], ...]:
        if not params:
            return ()

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
            raise Unavailable("Mocked unavailable", retry_after_ms=500, details={"op": op})
        if key == "rate_limited":
            raise ResourceExhausted("Mocked rate-limited", retry_after_ms=800, details={"op": op})
        if key == "transient":
            raise TransientNetwork("Mocked transient network", retry_after_ms=600, details={"op": op})
        # No RNG-based failures by default (keeps conformance runs stable)


# ---------------------------------------------------------------------------
# Minimal demo (optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def _demo() -> None:
        adapter = MockGraphAdapter(supports_transaction_ops=True, supports_traversal_ops=True, max_traversal_depth=3)

        caps = await adapter.capabilities()
        print("[CAPABILITIES]", caps)

        tx = await adapter.transaction(
            [
                BatchOperation(op="query", args={"text": "RETURN 1", "dialect": "cypher"}),
                BatchOperation(op="unknown_op", args={}),
            ]
        )
        print("[TX]", tx.success, "txid:", tx.transaction_id, "results:", tx.results)

        trav = await adapter.traversal(GraphTraversalSpec(start_nodes=["v:start:1"], max_depth=2))
        print("[TRAV] nodes", len(trav.nodes), "rels", len(trav.relationships), "paths", len(trav.paths))

    asyncio.run(_demo())
