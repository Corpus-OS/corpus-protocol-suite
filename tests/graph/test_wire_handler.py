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
  • §11.2, §13   — SIEM-safe behavior (no leakage via Wire handler hardening)
"""

from typing import Any, AsyncIterator, Dict, List, Mapping, Optional

import pytest

from corpus_sdk.graph.graph_base import (
    GRAPH_PROTOCOL_ID,
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
    GraphCapabilities,
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

pytestmark = pytest.mark.asyncio


class TrackingMockGraphAdapter(BaseGraphAdapter):
    """
    Mock adapter for exercising WireGraphHandler with tracking capabilities.

    Implements the internal *_do_* hooks and returns dataclasses so that
    WireGraphHandler's asdict()-based wiring works as intended.
    """

    def __init__(self) -> None:
        super().__init__()
        self.last_ctx: Optional[OperationContext] = None
        self.last_call: Optional[str] = None
        self.last_args: Dict[str, Any] = {}

    def _track(self, op: str, ctx: Optional[OperationContext], **kwargs: Any) -> None:
        self.last_call = op
        self.last_ctx = ctx
        self.last_args = dict(kwargs)

    async def _do_capabilities(self) -> GraphCapabilities:
        caps = GraphCapabilities(
            server="tracking-mock",
            version="1.0.0",
            supported_query_dialects=("cypher", "gremlin"),
            supports_stream_query=True,
            supports_namespaces=True,
            supports_property_filters=True,
            supports_bulk_vertices=True,
            supports_batch=True,
            supports_schema=True,
            idempotent_writes=True,
            supports_multi_tenant=True,
            supports_deadline=True,
            max_batch_ops=100,
        )
        self._track("capabilities", None)
        return caps

    async def _do_query(
        self, spec: GraphQuerySpec, *, ctx: Optional[OperationContext] = None
    ) -> QueryResult:
        self._track("query", ctx, spec=spec)
        return QueryResult(
            records=[{"n": 1}],
            summary={},
            dialect=spec.dialect,
            namespace=spec.namespace,
        )

    async def _do_stream_query(
        self, spec: GraphQuerySpec, *, ctx: Optional[OperationContext] = None
    ) -> AsyncIterator[QueryChunk]:
        self._track("stream_query", ctx, spec=spec)
        yield QueryChunk(records=[{"n": 1}], is_final=False)
        yield QueryChunk(records=[{"n": 2}], is_final=True, summary={"total": 2})

    async def _do_upsert_nodes(
        self, spec: UpsertNodesSpec, *, ctx: Optional[OperationContext] = None
    ) -> UpsertResult:
        self._track("upsert_nodes", ctx, spec=spec)
        return UpsertResult(upserted_count=len(spec.nodes), failed_count=0, failures=[])

    async def _do_upsert_edges(
        self, spec: UpsertEdgesSpec, *, ctx: Optional[OperationContext] = None
    ) -> UpsertResult:
        self._track("upsert_edges", ctx, spec=spec)
        return UpsertResult(upserted_count=len(spec.edges), failed_count=0, failures=[])

    async def _do_delete_nodes(
        self, spec: DeleteNodesSpec, *, ctx: Optional[OperationContext] = None
    ) -> DeleteResult:
        self._track("delete_nodes", ctx, spec=spec)
        return DeleteResult(deleted_count=len(spec.ids), failed_count=0, failures=[])

    async def _do_delete_edges(
        self, spec: DeleteEdgesSpec, *, ctx: Optional[OperationContext] = None
    ) -> DeleteResult:
        self._track("delete_edges", ctx, spec=spec)
        return DeleteResult(deleted_count=len(spec.ids), failed_count=0, failures=[])

    async def _do_bulk_vertices(
        self, spec: BulkVerticesSpec, *, ctx: Optional[OperationContext] = None
    ) -> BulkVerticesResult:
        self._track("bulk_vertices", ctx, spec=spec)
        return BulkVerticesResult(nodes=[], next_cursor=None, has_more=False)

    async def _do_batch(
        self, ops: List[BatchOperation], *, ctx: Optional[OperationContext] = None
    ) -> BatchResult:
        self._track("batch", ctx, ops=ops)
        return BatchResult(results=[{"ok": True} for _ in ops])

    async def _do_get_schema(self, *, ctx: Optional[OperationContext] = None) -> GraphSchema:
        self._track("get_schema", ctx)
        return GraphSchema(nodes={}, edges={}, metadata={})

    async def _do_health(self, *, ctx: Optional[OperationContext] = None) -> Mapping[str, Any]:
        self._track("health", ctx)
        return {
            "ok": True,
            "status": "ok",
            "server": "tracking-mock",
            "version": "1.0.0",
            "namespaces": {},
            "read_only": False,
            "degraded": False,
        }


class ErrorAdapter(TrackingMockGraphAdapter):
    """Adapter that always raises a specific GraphAdapterError for testing mapping."""

    def __init__(self, exc: GraphAdapterError):
        super().__init__()
        self._exc = exc

    async def _do_query(
        self, spec: GraphQuerySpec, *, ctx: Optional[OperationContext] = None
    ) -> QueryResult:
        raise self._exc


# ---------------------------------------------------------------------------
# Success-path envelopes
# ---------------------------------------------------------------------------

async def test_wire_contract_capabilities_success_envelope():
    """§4.1: Capabilities operation must return valid success envelope."""
    a = TrackingMockGraphAdapter()
    h = WireGraphHandler(a)

    env = {"op": "graph.capabilities", "ctx": {}, "args": {}}
    res = await h.handle(env)

    assert res["ok"] is True
    assert res["code"] == "OK"
    assert isinstance(res.get("ms"), (int, float))
    assert isinstance(res["result"], dict)

    caps = res["result"]
    assert caps["protocol"] == GRAPH_PROTOCOL_ID
    assert caps["server"]
    assert caps["version"]


async def test_wire_contract_query_roundtrip_and_context_plumbing():
    """§6.1: OperationContext must be properly propagated through wire handler."""
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

    res = await h.handle({"op": "graph.query", "ctx": ctx_wire, "args": args})

    assert res["ok"] is True
    assert res["code"] == "OK"
    assert isinstance(res.get("ms"), (int, float))

    out = res["result"]
    assert isinstance(out.get("records"), list)
    assert out.get("namespace") in (None, "demo")
    assert out.get("dialect") in (None, "cypher")

    # Context propagation
    assert a.last_call == "query"
    assert isinstance(a.last_ctx, OperationContext)
    assert a.last_ctx.request_id == "req_wire_graph"
    assert a.last_ctx.idempotency_key == "idem-graph"
    assert a.last_ctx.traceparent == "00-abc-xyz-01"
    assert a.last_ctx.tenant == "acme-tenant"
    assert "ignore_me" not in (a.last_ctx.attrs or {})


async def test_wire_contract_upsert_delete_bulk_batch_schema_health_envelopes():
    """§4.1: All graph operations must return valid success envelopes."""
    a = TrackingMockGraphAdapter()
    h = WireGraphHandler(a)

    up_nodes_env = {
        "op": "graph.upsert_nodes",
        "ctx": {"request_id": "up-n"},
        "args": {
            "namespace": "ns1",
            "nodes": [{"id": "n1", "labels": ["User"], "properties": {"name": "alice"}, "namespace": "ns1"}],
        },
    }
    up_nodes_res = await h.handle(up_nodes_env)
    assert up_nodes_res["ok"] is True
    assert up_nodes_res["code"] == "OK"
    assert isinstance(up_nodes_res.get("ms"), (int, float))
    assert up_nodes_res["result"]["upserted_count"] >= 1

    up_edges_env = {
        "op": "graph.upsert_edges",
        "ctx": {"request_id": "up-e"},
        "args": {
            "namespace": "ns1",
            "edges": [{"id": "e1", "src": "n1", "dst": "n1", "label": "SELF", "properties": {}, "namespace": "ns1"}],
        },
    }
    up_edges_res = await h.handle(up_edges_env)
    assert up_edges_res["ok"] is True
    assert up_edges_res["result"]["upserted_count"] >= 1

    del_nodes_env = {"op": "graph.delete_nodes", "ctx": {"request_id": "del-n"}, "args": {"namespace": "ns1", "ids": ["n1"]}}
    del_nodes_res = await h.handle(del_nodes_env)
    assert del_nodes_res["ok"] is True
    assert "deleted_count" in del_nodes_res["result"]

    del_edges_env = {"op": "graph.delete_edges", "ctx": {"request_id": "del-e"}, "args": {"namespace": "ns1", "ids": ["e1"]}}
    del_edges_res = await h.handle(del_edges_env)
    assert del_edges_res["ok"] is True
    assert "deleted_count" in del_edges_res["result"]

    bulk_env = {"op": "graph.bulk_vertices", "ctx": {"request_id": "bulk-1"}, "args": {"namespace": "ns1", "limit": 10}}
    bulk_res = await h.handle(bulk_env)
    assert bulk_res["ok"] is True
    assert "nodes" in bulk_res["result"]

    batch_env = {
        "op": "graph.batch",
        "ctx": {"request_id": "batch-1"},
        "args": {"ops": [{"op": "upsert_nodes", "args": {"nodes": [{"id": "n2", "labels": ["User"], "properties": {"name": "bob"}, "namespace": "ns1"}], "namespace": "ns1"}}]},
    }
    batch_res = await h.handle(batch_env)
    assert batch_res["ok"] is True
    assert "results" in batch_res["result"]

    schema_env = {"op": "graph.get_schema", "ctx": {"request_id": "schema-1"}, "args": {}}
    schema_res = await h.handle(schema_env)
    assert schema_res["ok"] is True
    assert "nodes" in schema_res["result"]
    assert "edges" in schema_res["result"]

    health_env = {"op": "graph.health", "ctx": {"request_id": "health-1"}, "args": {}}
    health_res = await h.handle(health_env)
    assert health_res["ok"] is True
    hr = health_res["result"]
    assert "server" in hr
    assert "version" in hr


async def test_wire_contract_get_schema_envelope_success():
    """§4.1: Schema operation must return valid success envelope."""
    a = TrackingMockGraphAdapter()
    h = WireGraphHandler(a)

    res = await h.handle({"op": "graph.get_schema", "ctx": {"request_id": "schema-only"}, "args": {}})
    assert res["ok"] is True
    assert res["code"] == "OK"
    assert isinstance(res.get("ms"), (int, float))
    assert isinstance(res["result"], dict)
    assert "nodes" in res["result"]
    assert "edges" in res["result"]


# ---------------------------------------------------------------------------
# Streaming via handle_stream
# ---------------------------------------------------------------------------

async def test_wire_contract_stream_query_success_chunks_and_context():
    """§4.1.6: Stream query must return valid chunk envelopes."""
    a = TrackingMockGraphAdapter()
    h = WireGraphHandler(a)

    env = {
        "op": "graph.stream_query",
        "ctx": {"request_id": "stream-1", "tenant": "stream-tenant"},
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

    assert len(chunks) >= 1

    for env_out in chunks:
        assert env_out["ok"] is True
        assert env_out["code"] == "STREAMING"
        assert isinstance(env_out.get("ms"), (int, float))
        assert "chunk" in env_out
        ch = env_out["chunk"]
        assert isinstance(ch.get("records"), list)

    assert a.last_call == "stream_query"
    assert isinstance(a.last_ctx, OperationContext)
    assert a.last_ctx.request_id == "stream-1"
    assert a.last_ctx.tenant == "stream-tenant"


async def test_wire_contract_stream_query_wrong_op_errors():
    """
    NEW: handle_stream called with wrong op yields one error envelope and stops.
    """
    a = TrackingMockGraphAdapter()
    h = WireGraphHandler(a)

    env = {"op": "graph.query", "ctx": {}, "args": {"text": "RETURN 1", "dialect": "cypher"}}

    out: List[Dict[str, Any]] = []
    async for e in h.handle_stream(env):
        out.append(e)

    assert len(out) == 1
    assert out[0]["ok"] is False
    assert out[0]["code"] == "BAD_REQUEST"
    assert out[0]["error"] == "BadRequest"
    assert isinstance(out[0].get("ms"), (int, float))


# ---------------------------------------------------------------------------
# Error mapping semantics
# ---------------------------------------------------------------------------

async def test_wire_contract_unknown_op_maps_to_not_supported():
    """§6.3: Unknown operations must return NOT_SUPPORTED error."""
    a = TrackingMockGraphAdapter()
    h = WireGraphHandler(a)

    res = await h.handle({"op": "graph.nope", "ctx": {}, "args": {}})

    assert res["ok"] is False
    assert res["code"] in ("NOT_SUPPORTED", "NOTSUPPORTED")
    assert res["error"] == "NotSupported"
    error_msg = res["message"].lower()
    assert any(term in error_msg for term in ["unknown", "support", "operation"])


async def test_wire_contract_missing_or_invalid_op_maps_to_bad_request():
    """§6.3: Missing operation must return BAD_REQUEST error."""
    a = TrackingMockGraphAdapter()
    h = WireGraphHandler(a)

    res = await h.handle({"ctx": {}, "args": {}})

    assert res["ok"] is False
    assert res["code"] == "BAD_REQUEST"
    assert res["error"] == "BadRequest"
    error_msg = res["message"].lower()
    assert any(term in error_msg for term in ["missing", "invalid", "op", "operation"])


async def test_wire_contract_requires_ctx_and_args_and_they_must_be_mappings():
    """
    NEW: Wire boundary is strict: envelopes MUST include ctx and args, and both MUST be objects.
    """
    a = TrackingMockGraphAdapter()
    h = WireGraphHandler(a)

    # Missing ctx
    res = await h.handle({"op": "graph.capabilities", "args": {}})
    assert res["ok"] is False
    assert res["code"] == "BAD_REQUEST"

    # Missing args
    res = await h.handle({"op": "graph.capabilities", "ctx": {}})
    assert res["ok"] is False
    assert res["code"] == "BAD_REQUEST"

    # ctx not mapping
    res = await h.handle({"op": "graph.capabilities", "ctx": "nope", "args": {}})
    assert res["ok"] is False
    assert res["code"] == "BAD_REQUEST"

    # args not mapping
    res = await h.handle({"op": "graph.capabilities", "ctx": {}, "args": "nope"})
    assert res["ok"] is False
    assert res["code"] == "BAD_REQUEST"


async def test_wire_contract_query_missing_required_fields_maps_to_bad_request():
    """§6.3: Missing required fields must return BAD_REQUEST."""
    a = TrackingMockGraphAdapter()
    h = WireGraphHandler(a)

    res = await h.handle({"op": "graph.query", "ctx": {}, "args": {}})

    assert res["ok"] is False
    assert res["code"] == "BAD_REQUEST"
    assert res["error"] == "BadRequest"
    error_msg = res["message"].lower()
    assert any(term in error_msg for term in ["text", "missing", "required"])


async def test_wire_contract_maps_graph_adapter_error_to_normalized_envelope():
    """§12.4: GraphAdapterError must be mapped to normalized error envelope."""
    exc = BadRequest("bad graph op")
    a = ErrorAdapter(exc)
    h = WireGraphHandler(a)

    res = await h.handle({"op": "graph.query", "ctx": {"request_id": "err-g"}, "args": {"text": "MATCH (n) RETURN n", "dialect": "cypher", "namespace": "demo"}})

    assert res["ok"] is False
    assert res["code"] == "BAD_REQUEST"
    assert res["error"] == "BadRequest"
    assert res["message"] == "bad graph op"
    assert "details" in res
    assert "retry_after_ms" in res
    assert isinstance(res.get("ms"), (int, float))


async def test_wire_contract_maps_notsupported_adapter_error_to_not_supported_code():
    """§12.4: NotSupported must be mapped to NOT_SUPPORTED code."""
    exc = NotSupported("nope")
    a = ErrorAdapter(exc)
    h = WireGraphHandler(a)

    res = await h.handle({"op": "graph.query", "ctx": {"request_id": "err-ns"}, "args": {"text": "MATCH (n) RETURN n", "dialect": "cypher", "namespace": "demo"}})

    assert res["ok"] is False
    assert res["code"] == "NOT_SUPPORTED"
    assert res["error"] == "NotSupported"
    assert "retry_after_ms" in res
    assert "details" in res


async def test_wire_contract_error_envelope_includes_message_and_type():
    """§12.4: Error envelopes must include message and type fields."""
    exc = BadRequest("bad things")
    a = ErrorAdapter(exc)
    h = WireGraphHandler(a)

    res = await h.handle({"op": "graph.query", "ctx": {"request_id": "err-msg"}, "args": {"text": "MATCH (n) RETURN n", "dialect": "cypher", "namespace": "demo"}})

    assert res["ok"] is False
    assert isinstance(res.get("code"), str) and res["code"]
    assert isinstance(res.get("error"), str) and res["error"]
    assert isinstance(res.get("message"), str) and res["message"]
    assert isinstance(res.get("ms"), (int, float))


async def test_wire_contract_graph_adapter_error_includes_retry_after_and_details_fields():
    """
    NEW: GraphAdapterError subclasses should map retry_after_ms and details through to wire.
    """
    exc = Unavailable("try later", retry_after_ms=123, details={"hint": "backoff"})
    a = ErrorAdapter(exc)
    h = WireGraphHandler(a)

    res = await h.handle({"op": "graph.query", "ctx": {"request_id": "err-ra"}, "args": {"text": "RETURN 1", "dialect": "cypher"}})

    assert res["ok"] is False
    assert res["code"] == "UNAVAILABLE"
    assert res["error"] == "Unavailable"
    assert res["retry_after_ms"] == 123
    assert isinstance(res.get("details"), dict)
    assert res["details"].get("hint") == "backoff"


async def test_wire_contract_maps_unexpected_exception_to_unavailable_and_hardens_message():
    """§12.4: Unexpected exceptions must be mapped to UNAVAILABLE and hardened."""

    class BoomAdapter(TrackingMockGraphAdapter):
        async def _do_query(self, spec: GraphQuerySpec, *, ctx: Optional[OperationContext] = None) -> QueryResult:
            raise RuntimeError("boom")

    a = BoomAdapter()
    h = WireGraphHandler(a)

    res = await h.handle({"op": "graph.query", "ctx": {"request_id": "boom"}, "args": {"text": "MATCH (n) RETURN n", "dialect": "cypher", "namespace": "demo"}})

    assert res["ok"] is False
    assert res["code"] == "UNAVAILABLE"
    assert res["error"] == "RuntimeError"

    # Hardened message: do not leak exception message ("boom")
    assert res["message"] == "internal error"
    assert "boom" not in res["message"].lower()

    details = res.get("details") or {}
    assert isinstance(details, dict)
    assert details.get("error_type") == "RuntimeError"
    assert isinstance(res.get("ms"), (int, float))
