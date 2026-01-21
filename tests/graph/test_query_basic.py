# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Basic query behavior.

Asserts (Spec refs):
  • list-of-mapping results                                   (§7.3.2)
  • dialect + text validation                                 (§7.4, §17.2)
  • params binding accepts odd strings safely                 (§7.3.2)
  • empty/None params accepted                                (§7.3.2)

"""
from __future__ import annotations

import json
import pytest

from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    BadRequest,
    NotSupported,
    BaseGraphAdapter,
    GraphQuerySpec,
    QueryResult,
    WireGraphHandler,
)

pytestmark = pytest.mark.asyncio


async def test_query_returns_json_serializable_records_list(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_query_rows", tenant="test")
    res = await adapter.query(GraphQuerySpec(text="RETURN 1 as value", dialect="cypher"), ctx=ctx)
    assert isinstance(res, QueryResult)
    assert isinstance(res.records, list)
    json.dumps(res.records)


async def test_query_requires_non_empty_text(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_query_req", tenant="test")
    with pytest.raises(BadRequest):
        await adapter.query(GraphQuerySpec(text="", dialect="cypher"), ctx=ctx)


async def test_query_params_are_bound_safely(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_query_bind", tenant="test")
    res = await adapter.query(
        GraphQuerySpec(text="RETURN $param as value", dialect="cypher", params={"param": "'; DROP ALL; --"}),
        ctx=ctx,
    )
    assert isinstance(res, QueryResult)


async def test_query_none_and_empty_params_allowed(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_query_empty_params", tenant="test")
    res1 = await adapter.query(GraphQuerySpec(text="RETURN 1", dialect="cypher", params=None), ctx=ctx)
    res2 = await adapter.query(GraphQuerySpec(text="RETURN 1", dialect="cypher", params={}), ctx=ctx)
    assert isinstance(res1.records, list)
    assert isinstance(res2.records, list)


async def test_query_params_must_be_json_serializable(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_query_params_json", tenant="test")
    with pytest.raises(BadRequest):
        await adapter.query(GraphQuerySpec(text="RETURN 1", dialect="cypher", params={"x": object()}), ctx=ctx)


async def test_query_accepts_params_with_non_string_keys_if_json_allows(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_query_params_keys", tenant="test")
    res = await adapter.query(GraphQuerySpec(text="RETURN 1", dialect="cypher", params={1: "one", "two": 2}), ctx=ctx)
    assert isinstance(res.records, list)


async def test_query_dialect_validation_is_capability_driven(adapter: BaseGraphAdapter):
    caps = await adapter.capabilities()
    declared = tuple(getattr(caps, "supported_query_dialects", ()) or ())
    ctx = GraphContext(request_id="t_query_dialect_cap", tenant="test")

    if declared:
        valid = declared[0]
        res = await adapter.query(GraphQuerySpec(text="RETURN 1", dialect=valid), ctx=ctx)
        assert isinstance(res.records, list)

        invalid = "__not_in_declared__"
        if invalid in declared:
            invalid = "__not_in_declared_2__"
        with pytest.raises(NotSupported):
            await adapter.query(GraphQuerySpec(text="RETURN 1", dialect=invalid), ctx=ctx)
        return

    res = await adapter.query(GraphQuerySpec(text="RETURN 1", dialect=None), ctx=ctx)
    assert isinstance(res.records, list)


# ---------------------------- NEW: wire envelope shape ----------------------------

async def test_wire_handle_capabilities_success_envelope_shape(adapter: BaseGraphAdapter):
    """
    NEW: Wire capabilities response must be a canonical success envelope.
    """
    h = WireGraphHandler(adapter)
    resp = await h.handle({"op": "graph.capabilities", "ctx": {}, "args": {}})
    assert resp.get("ok") is True
    assert resp.get("code") == "OK"
    assert isinstance(resp.get("ms"), (int, float))
    assert "result" in resp
    assert isinstance(resp["result"], dict)


async def test_wire_handle_query_success_envelope_shape(adapter: BaseGraphAdapter):
    """
    NEW: Wire query response must be a canonical success envelope with result payload.
    """
    caps = await adapter.capabilities()
    dialect = caps.supported_query_dialects[0] if caps.supported_query_dialects else None

    h = WireGraphHandler(adapter)
    args = {"text": "RETURN 1"}
    if dialect is not None:
        args["dialect"] = dialect

    resp = await h.handle({"op": "graph.query", "ctx": {"request_id": "wq1"}, "args": args})
    assert resp.get("ok") is True
    assert resp.get("code") == "OK"
    assert isinstance(resp.get("ms"), (int, float))
    assert isinstance(resp.get("result"), dict)
    assert isinstance(resp["result"].get("records"), list)


async def test_wire_handle_unknown_op_error_envelope_shape(adapter: BaseGraphAdapter):
    """
    NEW: Unknown wire op must return canonical error envelope.
    """
    h = WireGraphHandler(adapter)
    resp = await h.handle({"op": "graph.nope", "ctx": {}, "args": {}})
    assert resp.get("ok") is False
    assert isinstance(resp.get("code"), str) and resp["code"]
    assert isinstance(resp.get("error"), str) and resp["error"]
    assert isinstance(resp.get("message"), str)
    assert isinstance(resp.get("ms"), (int, float))


async def test_wire_error_envelope_hardens_unexpected_exceptions(adapter: BaseGraphAdapter):
    """
    NEW: Error envelopes must be stable and SIEM-safe (details mapping or null).
    """
    # Use an invalid dialect that should trigger NotSupported when dialects are declared;
    # otherwise use invalid args type to force BadRequest.
    caps = await adapter.capabilities()
    h = WireGraphHandler(adapter)

    if caps.supported_query_dialects:
        resp = await h.handle(
            {"op": "graph.query", "ctx": {"attrs": {"simulate_error": "transient"}}, "args": {"text": "RETURN 1", "dialect": "__nope__"}}
        )
    else:
        resp = await h.handle({"op": "graph.query", "ctx": {}, "args": {"text": ""}})

    assert resp.get("ok") is False
    assert isinstance(resp.get("code"), str) and resp["code"]
    assert isinstance(resp.get("error"), str) and resp["error"]
    assert isinstance(resp.get("message"), str)
    assert isinstance(resp.get("ms"), (int, float))
    d = resp.get("details")
    assert d is None or isinstance(d, dict)
