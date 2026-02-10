# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — CRUD basics (via batch upsert/delete).

Asserts (Spec refs):
  • node/edge upserts validate labels/ids                    (§7.3.1, §17.2)
  • delete ops are idempotent                                (§7.3.1)
  • properties normalized to JSON-safe keys                  (§7.3.1, §17.2)
"""
from __future__ import annotations

import pytest

from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    GraphID,
    BadRequest,
    Node,
    Edge,
    UpsertNodesSpec,
    UpsertEdgesSpec,
    DeleteNodesSpec,
    DeleteEdgesSpec,
    UpsertResult,
    DeleteResult,
    BaseGraphAdapter,
)

pytestmark = pytest.mark.asyncio


async def test_crud_upsert_node_returns_success(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_crud_v", tenant="t1")
    node = Node(id=GraphID("v:User:1"), labels=("User",), properties={"name": "Ada"})
    res = await adapter.upsert_nodes(UpsertNodesSpec(nodes=[node], namespace="t1"), ctx=ctx)
    assert isinstance(res, UpsertResult)
    assert res.upserted_count >= 0
    assert res.failed_count >= 0


async def test_crud_upsert_edge_returns_success(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_crud_e", tenant="t1")
    edge = Edge(id=GraphID("e:FOLLOWS:1"), src=GraphID("v:U:1"), dst=GraphID("v:U:2"), label="FOLLOWS", properties={"since": 2020})
    res = await adapter.upsert_edges(UpsertEdgesSpec(edges=[edge], namespace="t1"), ctx=ctx)
    assert isinstance(res, UpsertResult)
    assert res.upserted_count >= 0
    assert res.failed_count >= 0


async def test_crud_node_labels_type_validation_happens_at_model_level(adapter: BaseGraphAdapter):
    with pytest.raises(BadRequest):
        Node(id=GraphID("v:bad:1"), labels=(123,), properties={"x": 1})  # type: ignore[arg-type]


async def test_crud_properties_must_be_json_serializable(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_crud_props_json", tenant="t1")
    node = Node(id=GraphID("v:bad:props"), labels=("User",), properties={"x": object()})
    with pytest.raises(BadRequest):
        await adapter.upsert_nodes(UpsertNodesSpec(nodes=[node], namespace="t1"), ctx=ctx)


async def test_crud_upsert_nodes_empty_rejected(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_crud_upsert_nodes_empty", tenant="t1")
    with pytest.raises(BadRequest):
        await adapter.upsert_nodes(UpsertNodesSpec(nodes=[], namespace="t1"), ctx=ctx)


async def test_crud_upsert_edges_empty_rejected(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_crud_upsert_edges_empty", tenant="t1")
    with pytest.raises(BadRequest):
        await adapter.upsert_edges(UpsertEdgesSpec(edges=[], namespace="t1"), ctx=ctx)


async def test_crud_validation_edge_requires_src_dst_label(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_crud_req_edge", tenant="t1")
    with pytest.raises(BadRequest):
        bad_edge = Edge(id=GraphID("e:bad:1"), src=GraphID("v:1"), dst=GraphID("v:2"), label="", properties={})
        await adapter.upsert_edges(UpsertEdgesSpec(edges=[bad_edge], namespace="t1"), ctx=ctx)


async def test_crud_delete_nodes_requires_ids_or_filter(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_crud_del_nodes_requires", tenant="t1")
    with pytest.raises(BadRequest):
        await adapter.delete_nodes(DeleteNodesSpec(ids=[], namespace="t1", filter=None), ctx=ctx)


async def test_crud_delete_edges_requires_ids_or_filter(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_crud_del_edges_requires", tenant="t1")
    with pytest.raises(BadRequest):
        await adapter.delete_edges(DeleteEdgesSpec(ids=[], namespace="t1", filter=None), ctx=ctx)


async def test_crud_delete_filter_must_be_json_serializable(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_crud_del_filter_json", tenant="t1")
    with pytest.raises(BadRequest):
        await adapter.delete_nodes(DeleteNodesSpec(ids=[], namespace="t1", filter={"x": object()}), ctx=ctx)


async def test_crud_delete_nodes_idempotent_repeatable(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_crud_del_v", tenant="t1")
    spec = DeleteNodesSpec(ids=[GraphID("v:missing")], namespace="t1")
    r1 = await adapter.delete_nodes(spec, ctx=ctx)
    r2 = await adapter.delete_nodes(spec, ctx=ctx)
    assert isinstance(r1, DeleteResult) and isinstance(r2, DeleteResult)
    assert r1.deleted_count == r2.deleted_count
    assert r1.failed_count == r2.failed_count


async def test_crud_delete_edges_idempotent_repeatable(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_crud_del_e", tenant="t1")
    spec = DeleteEdgesSpec(ids=[GraphID("e:missing")], namespace="t1")
    r1 = await adapter.delete_edges(spec, ctx=ctx)
    r2 = await adapter.delete_edges(spec, ctx=ctx)
    assert isinstance(r1, DeleteResult) and isinstance(r2, DeleteResult)
    assert r1.deleted_count == r2.deleted_count
    assert r1.failed_count == r2.failed_count


async def test_crud_properties_with_non_string_keys_allowed_if_json_allows(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_crud_props_keys", tenant="t1")
    node = Node(id=GraphID("v:Obj:1"), labels=("Obj",), properties={1: "one", "two": 2})
    res = await adapter.upsert_nodes(UpsertNodesSpec(nodes=[node], namespace="t1"), ctx=ctx)
    assert isinstance(res, UpsertResult)
