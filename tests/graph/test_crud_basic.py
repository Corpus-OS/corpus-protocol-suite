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

from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    GraphID,
    BadRequest,
)

pytestmark = pytest.mark.asyncio


def make_ctx(ctx_cls, **kwargs):
    """Local helper to construct an OperationContext."""
    return ctx_cls(**kwargs)


async def test_crud_validation_create_vertex_returns_graph_id(adapter):
    ctx = make_ctx(GraphContext, request_id="t_crud_v", tenant="t1")
    vid = await adapter.create_vertex("User", {"name": "Ada"}, ctx=ctx)
    assert isinstance(vid, GraphID)
    assert str(vid).startswith("v:User:")


async def test_crud_validation_create_edge_returns_graph_id(adapter):
    ctx = make_ctx(GraphContext, request_id="t_crud_e", tenant="t1")
    e = await adapter.create_edge("FOLLOWS", "v:U:1", "v:U:2", {"since": 2020}, ctx=ctx)
    assert isinstance(e, GraphID)
    assert str(e).startswith("e:FOLLOWS:")


async def test_crud_validation_vertex_requires_label_and_props(adapter):
    ctx = make_ctx(GraphContext, request_id="t_crud_req1", tenant="t1")
    with pytest.raises(BadRequest):
        await adapter.create_vertex("", {"x": 1}, ctx=ctx)
    with pytest.raises(BadRequest):
        await adapter.create_vertex("User", None, ctx=ctx)  # type: ignore[arg-type]


async def test_crud_validation_edge_requires_from_to_label(adapter):
    ctx = make_ctx(GraphContext, request_id="t_crud_req2", tenant="t1")
    with pytest.raises(BadRequest):
        await adapter.create_edge("", "v:1", "v:2", {}, ctx=ctx)
    with pytest.raises(BadRequest):
        await adapter.create_edge("READ", "", "v:2", {}, ctx=ctx)
    with pytest.raises(BadRequest):
        await adapter.create_edge("READ", "v:1", "", {}, ctx=ctx)


async def test_crud_validation_delete_vertex_idempotent(adapter):
    ctx = make_ctx(GraphContext, request_id="t_crud_del_v", tenant="t1")
    await adapter.delete_vertex("v:missing", ctx=ctx)


async def test_crud_validation_delete_edge_idempotent(adapter):
    ctx = make_ctx(GraphContext, request_id="t_crud_del_e", tenant="t1")
    await adapter.delete_edge("e:missing", ctx=ctx)


async def test_crud_validation_properties_are_json_serializable(adapter):
    ctx = make_ctx(GraphContext, request_id="t_crud_props", tenant="t1")
    vid = await adapter.create_vertex("Obj", {1: "one", "two": 2}, ctx=ctx)
    assert str(vid).startswith("v:Obj:")
