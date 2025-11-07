# tests/graph/test_wire_handler.py
# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Wire-level envelopes & routing.

Asserts (SPECIFICATION.md refs):
  • §4.1         — Canonical op strings and envelope shapes
  • §4.1.6       — Streaming envelopes for graph.stream_query
  • §9           — Graph operations mapped via `graph.<op>` wire contract
  • §6.1         — OperationContext constructed from wire ctx (ignore unknowns)
  • §6.3, §12.4  — Normalized error envelopes for GraphAdapterError subclasses
  • §11.2, §13   — SIEM-safe behavior (no tenant leakage via Wire handler)

Covers:
  • graph.capabilities → success envelope
  • graph.query / upsert_* / delete_* / bulk_vertices / batch / get_schema / health → success envelopes
  • graph.stream_query via handle_stream() → chunk envelopes
  • Context translation: ctx → OperationContext passed into BaseGraphAdapter
  • Unknown op → NotSupported → normalized error envelope
  • Missing/invalid op → BadRequest → normalized error envelope
  • GraphAdapterError → mapped with correct code/error/message/details
  • Unexpected Exception → mapped to UNAVAILABLE per common error taxonomy
"""

import pytest
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional

from corpus_sdk.graph.graph_base import (
    GRAPH_PROTOCOL_ID,
    GraphID,
    Node,
    Edge,
    GraphQuerySpec,
    UpsertNodesSpec,
    UpsertEdgesSpec,
    DeleteNodesSpec,
    DeleteEdgesSpec,
    BulkVerticesSpec,
    BatchOperation,
    BatchResult,
    GraphSchema,
    QueryResult,
    QueryChunk,
    UpsertResult,
    DeleteResult,
    OperationContext,
    GraphAdapterError,
    BadRequest,
    NotSupported,
    Unavailable,
    BaseGraphAdapter,
    WireGraphHandler,
)

from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter

pytestmark = pytest.mark.asyncio


class TrackingMockGraphAdapter(MockGraphAdapter):
    """
    MockGraphAdapter wrapper for exercising WireGraphHandler.

    Uses the real mock implementation; only records last ctx/call/args
    and forces deterministic behavior where applicable.
    """

    def __init__(self, *args, **kwargs) -> None:
        # Ensure deterministic behavior in tests if MockGraphAdapter supports failure_rate.
        kwargs.setdefault("failure_rate", 0.0)
        super().__init__(*args, **kwargs)
        self.last_ctx: Optional[OperationContext] = None
        self.last_call: Optional[str] = None
        self.last_args: Dict[str, Any] = {}

    # --- helpers -------------------------------------------------------------

    def _track(
        self,
        op: str,
        ctx: Optional[OperationContext],
        **kwargs: Any,
    ) -> None:
        self.last_call = op
        self.last_ctx = ctx
        self.last_args = dict(kwargs)

    # --- backend hooks w/ tracking ------------------------------------------
    # These assume MockGraphAdapter implements the BaseGraphAdapter hook surface.

    async def _do_capabilities(self):
        self._track("capabilities", None)
        return await super()._do_capabilities()

    async def _do_query(
        self,
        spec: GraphQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        self._track("query", ctx, spec=spec)
        return await super()._do_query(spec=spec, ctx=ctx)

    async def _do_stream_query(
        self,
        spec: GraphQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[QueryChunk]:
        self._track("stream_query", ctx, spec=spec)
        async for chunk in super()._do_stream_query(spec=spec, ctx=ctx):
            yield chunk

    async def _do_upsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        self._track("upsert_nodes", ctx, spec=spec)
        return await super()._do_upsert_nodes(spec=spec, ctx=ctx)

    async def _do_upsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        self._track("upsert_edges", ctx, spec=spec)
        return await super()._do_upsert_edges(spec=spec, ctx=ctx)

    async def _do_delete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        self._track("delete_nodes", ctx, spec=spec)
        return await super()._do_delete_nodes(spec=spec, ctx=ctx)

    async def _do_delete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        self._track("delete_edges", ctx, spec=spec)
        return await super()._do_delete_edges(spec=spec, ctx=ctx)

    async def _do_bulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ):
        self._track("bulk_vertices", ctx, spec=spec)
        return await super()._do_bulk_vertices(spec=spec, ctx=ctx)

    async def _do_batch(
        self,
        ops: List[BatchOperation],
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchResult:
        self._track("batch", ctx, ops=ops)
        return await super()._do_batch(ops=ops, ctx=ctx)

    async def _do_get_schema(
        self,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> GraphSchema:
        self._track("get_schema", ctx)
        return await super()._do_get_schema(ctx=ctx)

    async def _do_health(
        self,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> Mapping[str, Any]:
        self._track("health", ctx)
        return await super()._do_health(ctx=ctx)


class ErrorAdapter(TrackingMockGraphAdapter):
    """
    Adapter that always raises a specific GraphAdapterError for testing mapping.
    """

    def __init__(self, exc: GraphAdapterError):
        super().__init__()
        self._exc = exc

    async def _do_query(
        self,
        spec: GraphQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        raise self._exc


# ---------------------------------------------------------------------------
# Success-path envelopes
# ---------------------------------------------------------------------------

async def test_wire_capabilities_success_envelope():
    a = TrackingMockGraphAdapter()
    h = WireGraphHandler(a)

    env = {
        "op": "graph.capabilities",
        "ctx": {},
        "args": {},
    }
    res = await h.handle(env)

    assert res["ok"] is True
    assert res["code"] == "OK"
    assert isinstance(res["result"], dict)

    caps = res["result"]
    assert caps["protocol"] == GRAPH_PROTOCOL_ID
    assert caps["server"]
    assert caps["version"]


async def test_wire_query_roundtrip_and_context_plumbing():
    a = TrackingMockGraphAdapter()
    h = WireGraphHandler(a)

    ctx_wire = {
        "request_id": "req_wire_graph",
        "idempotency_key": "idem-graph",
        "deadline_ms": 9999999999999,
        "traceparent": "00-abc-xyz-01",
        "tenant": "acme-tenant",
        "attrs": {"k": "v"},
        "ignore_me": "extra",  # MUST be ignored
    }
    args = {
        "text": "MATCH (n) RETURN n LIMIT 1",
        "dialect": "cypher",
        "namespace": "demo",
        "params": {"foo": "bar"},
        "timeout_ms": 1000,
        "stream": False,
    }

    res = await h.handle(
        {
            "op": "graph.query",
            "ctx": ctx_wire,
            "args": args,
        }
    )

    assert res["ok"] is True
    assert res["code"] == "OK"
    out = res["result"]
    assert isinstance(out.get("records"), list)
    assert out.get("namespace") in (None, "demo")  # adapter-defined
    assert out.get("dialect") in (None, "cypher")

    # Context propagation via BaseGraphAdapter -> TrackingMockGraphAdapter
    assert a.last_call == "query"
    assert isinstance(a.last_ctx, OperationContext)
    assert a.last_ctx.request_id == "req_wire_graph"
    assert a.last_ctx.idempotency_key == "idem-graph"
    assert a.last_ctx.traceparent == "00-abc-xyz-01"
    assert a.last_ctx.tenant == "acme-tenant"
    assert "ignore_me" not in (a.last_ctx.attrs or {})


async def test_wire_upsert_delete_bulk_batch_schema_health_envelopes():
    a = TrackingMockGraphAdapter()
    h = WireGraphHandler(a)

    # upsert_nodes
    up_nodes_env = {
        "op": "graph.upsert_nodes",
        "ctx": {"request_id": "up-n"},
        "args": {
            "namespace": "ns1",
            "nodes": [
                {
                    "id": "n1",
                    "labels": ["User"],
                    "properties": {"name": "alice"},
                    "namespace": "ns1",
                }
            ],
        },
    }
    up_nodes_res = await h.handle(up_nodes_env)
    assert up_nodes_res["ok"] is True
    assert up_nodes_res["result"]["upserted_count"] >= 1

    # upsert_edges
    up_edges_env = {
        "op": "graph.upsert_edges",
        "ctx": {"request_id": "up-e"},
        "args": {
            "namespace": "ns1",
            "edges": [
                {
                    "id": "e1",
                    "src": "n1",
                    "dst": "n1",
                    "label": "SELF",
                    "properties": {},
                    "namespace": "ns1",
                }
            ],
        },
    }
    up_edges_res = await h.handle(up_edges_env)
    assert up_edges_res["ok"] is True
    assert up_edges_res["result"]["upserted_count"] >= 1

    # delete_nodes
    del_nodes_env = {
        "op": "graph.delete_nodes",
        "ctx": {"request_id": "del-n"},
        "args": {
            "namespace": "ns1",
            "ids": ["n1"],
        },
    }
    del_nodes_res = await h.handle(del_nodes_env)
    assert del_nodes_res["ok"] is True
    assert "deleted_count" in del_nodes_res["result"]

    # delete_edges
    del_edges_env = {
        "op": "graph.delete_edges",
        "ctx": {"request_id": "del-e"},
        "args": {
            "namespace": "ns1",
            "ids": ["e1"],
        },
    }
    del_edges_res = await h.handle(del_edges_env)
    assert del_edges_res["ok"] is True
    assert "deleted_count" in del_edges_res["result"]

    # bulk_vertices
    bulk_env = {
        "op": "graph.bulk_vertices",
        "ctx": {"request_id": "bulk-1"},
        "args": {
            "namespace": "ns1",
            "limit": 10,
        },
    }
    bulk_res = await h.handle(bulk_env)
    assert bulk_res["ok"] is True
    assert "nodes" in bulk_res["result"]

    # batch
    batch_env = {
        "op": "graph.batch",
        "ctx": {"request_id": "batch-1"},
        "args": {
            "ops": [
                {
                    "op": "upsert_nodes",
                    "args": {
                        "nodes": [
                            {
                                "id": "n2",
                                "labels": ["User"],
                                "properties": {"name": "bob"},
                                "namespace": "ns1",
                            }
                        ],
                        "namespace": "ns1",
                    },
                }
            ]
        },
    }
    batch_res = await h.handle(batch_env)
    assert batch_res["ok"] is True
    assert "results" in batch_res["result"]

    # get_schema
    schema_env = {
        "op": "graph.get_schema",
        "ctx": {"request_id": "schema-1"},
        "args": {},
    }
    schema_res = await h.handle(schema_env)
    assert schema_res["ok"] is True
    assert "nodes" in schema_res["result"]
    assert "edges" in schema_res["result"]

    # health
    health_env = {
        "op": "graph.health",
        "ctx": {"request_id": "health-1"},
        "args": {},
    }
    health_res = await h.handle(health_env)
    assert health_res["ok"] is True
    hr = health_res["result"]
    assert "server" in hr
    assert "version" in hr


# ---------------------------------------------------------------------------
# Streaming via handle_stream
# ---------------------------------------------------------------------------

async def test_wire_stream_query_success_chunks_and_context():
    a = TrackingMockGraphAdapter()
    h = WireGraphHandler(a)

    env = {
        "op": "graph.stream_query",
        "ctx": {
            "request_id": "stream-1",
            "tenant": "stream-tenant",
        },
        "args": {
            "text": "MATCH (n) RETURN n",
            "dialect": "cypher",
            "namespace": "demo",
            "params": {},
            "timeout_ms": 1000,
            "stream": True,
        },
    }

    chunks: List[Dict[str, Any]] = []
    async for envelope in h.handle_stream(env):
        chunks.append(envelope)

    # Expect at least one chunk, final chunk indicated by is_final when adapter supports it.
    assert len(chunks) >= 1

    for env_out in chunks:
        assert env_out["ok"] is True
        assert env_out["code"] == "OK"
        assert "chunk" in env_out
        ch = env_out["chunk"]
        assert isinstance(ch.get("records"), list)

    # Ensure context was passed to streaming path
    assert a.last_call == "stream_query"
    assert isinstance(a.last_ctx, OperationContext)
    assert a.last_ctx.request_id == "stream-1"
    assert a.last_ctx.tenant == "stream-tenant"


# ---------------------------------------------------------------------------
# Error mapping semantics
# ---------------------------------------------------------------------------

async def test_wire_unknown_op_maps_to_not_supported():
    a = TrackingMockGraphAdapter()
    h = WireGraphHandler(a)

    res = await h.handle(
        {
            "op": "graph.nope",
            "ctx": {},
            "args": {},
        }
    )

    assert res["ok"] is False
    assert res["code"] in ("NOT_SUPPORTED", "NOTSUPPORTED")
    assert res["error"] == "NotSupported"
    assert "unknown" in res["message"]


async def test_wire_missing_or_invalid_op_maps_to_bad_request():
    a = TrackingMockGraphAdapter()
    h = WireGraphHandler(a)

    res = await h.handle(
        {
            "ctx": {},
            "args": {},
        }
    )

    assert res["ok"] is False
    assert res["code"] == "BAD_REQUEST"
    assert res["error"] == "BadRequest"
    assert "missing or invalid 'op'" in res["message"]


async def test_wire_maps_graph_adapter_error_to_normalized_envelope():
    exc = BadRequest("bad graph op")
    a = ErrorAdapter(exc)
    h = WireGraphHandler(a)

    res = await h.handle(
        {
            "op": "graph.query",
            "ctx": {"request_id": "err-g"},
            "args": {
                "text": "MATCH (n) RETURN n",
                "dialect": "cypher",
                "namespace": "demo",
            },
        }
    )

    assert res["ok"] is False
    assert res["code"] == "BAD_REQUEST"
    assert res["error"] == "BadRequest"
    assert res["message"] == "bad graph op"
    assert "details" in res  # JSON-safe details present (may be null)


async def test_wire_maps_unexpected_exception_to_unavailable():
    class BoomAdapter(TrackingMockGraphAdapter):
        async def _do_query(
            self,
            spec: GraphQuerySpec,
            *,
            ctx: Optional[OperationContext] = None,
        ) -> QueryResult:
            raise RuntimeError("boom")

    a = BoomAdapter()
    h = WireGraphHandler(a)

    res = await h.handle(
        {
            "op": "graph.query",
            "ctx": {"request_id": "boom"},
            "args": {
                "text": "MATCH (n) RETURN n",
                "dialect": "cypher",
                "namespace": "demo",
            },
        }
    )

    assert res["ok"] is False
    assert res["code"] == "UNAVAILABLE"
    assert res["error"] == "RuntimeError"
    assert "boom" in res["message"]
