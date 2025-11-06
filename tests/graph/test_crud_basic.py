# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — CRUD basics.

Asserts (Spec refs):
  • create_vertex/create_edge return GraphID                 (§7.3.1)
  • labels/ids validated                                     (§7.3.1, §17.2)
  • delete ops are idempotent                                (§7.3.1)
  • properties normalized to JSON-safe keys                  (§7.3.1, §17.2)
"""
import pytest

from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    GraphID,
    BadRequest,
)
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_create_vertex_returns_graph_id():
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_crud_v", tenant="t1")
    vid = await a.create_vertex("User", {"name": "Ada"}, ctx=ctx)
    assert isinstance(vid, GraphID)
    assert str(vid).startswith("v:User:")


async def test_create_edge_returns_graph_id():
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_crud_e", tenant="t1")
    e = await a.create_edge("FOLLOWS", "v:U:1", "v:U:2", {"since": 2020}, ctx=ctx)
    assert isinstance(e, GraphID)
    assert str(e).startswith("e:FOLLOWS:")


async def test_vertex_requires_label_and_props():
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_crud_req1", tenant="t1")
    with pytest.raises(BadRequest):
        await a.create_vertex("", {"x": 1}, ctx=ctx)
    with pytest.raises(BadRequest):
        await a.create_vertex("User", None, ctx=ctx)  # type: ignore[arg-type]


async def test_edge_requires_from_to_label():
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_crud_req2", tenant="t1")
    with pytest.raises(BadRequest):
        await a.create_edge("", "v:1", "v:2", {}, ctx=ctx)
    with pytest.raises(BadRequest):
        await a.create_edge("READ", "", "v:2", {}, ctx=ctx)
    with pytest.raises(BadRequest):
        await a.create_edge("READ", "v:1", "", {}, ctx=ctx)


async def test_delete_vertex_idempotent():
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_crud_del_v", tenant="t1")
    await a.delete_vertex("v:missing", ctx=ctx)


async def test_delete_edge_idempotent():
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_crud_del_e", tenant="t1")
    await a.delete_edge("e:missing", ctx=ctx)


async def test_properties_are_json_serializable():
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_crud_props", tenant="t1")
    vid = await a.create_vertex("Obj", {1: "one", "two": 2}, ctx=ctx)
    assert str(vid).startswith("v:Obj:")
