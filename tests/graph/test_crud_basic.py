# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — CRUD basics (via batch upsert/delete).

Asserts (Spec refs):
  • node/edge upserts validate labels/ids                    (§7.3.1, §17.2)
  • delete ops are idempotent                                (§7.3.1)
  • properties normalized to JSON-safe keys                  (§7.3.1, §17.2)
"""
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
    spec = UpsertNodesSpec(nodes=[node], namespace="t1")

    res = await adapter.upsert_nodes(spec, ctx=ctx)
    assert isinstance(res, UpsertResult)
    assert res.upserted_count == 1
    assert res.failed_count == 0
    assert isinstance(node.id, GraphID)


async def test_crud_upsert_edge_returns_success(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_crud_e", tenant="t1")
    edge = Edge(
        id=GraphID("e:FOLLOWS:1"),
        src=GraphID("v:U:1"),
        dst=GraphID("v:U:2"),
        label="FOLLOWS",
        properties={"since": 2020},
    )
    spec = UpsertEdgesSpec(edges=[edge], namespace="t1")

    res = await adapter.upsert_edges(spec, ctx=ctx)
    assert isinstance(res, UpsertResult)
    assert res.upserted_count == 1
    assert res.failed_count == 0
    assert isinstance(edge.id, GraphID)


async def test_crud_validation_node_labels_and_props(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_crud_req1", tenant="t1")

    # Invalid labels should result in failures
    bad_node = Node(id=GraphID("v:bad:1"), labels=("",), properties={"x": 1})
    spec = UpsertNodesSpec(nodes=[bad_node], namespace="t1")
    res = await adapter.upsert_nodes(spec, ctx=ctx)
    assert res.failed_count == 1

    # Non-JSON-serializable properties should raise BadRequest at validation time
    with pytest.raises(BadRequest):
        bad_node2 = Node(
            id=GraphID("v:bad:2"),
            labels=("User",),
            properties={"x": object()},  # not JSON serializable
        )
        spec2 = UpsertNodesSpec(nodes=[bad_node2], namespace="t1")
        await adapter.upsert_nodes(spec2, ctx=ctx)


async def test_crud_validation_edge_requires_src_dst_label(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_crud_req2", tenant="t1")

    # Invalid label
    bad_edge1 = Edge(
        id=GraphID("e:bad:1"),
        src=GraphID("v:1"),
        dst=GraphID("v:2"),
        label="",
        properties={},
    )
    spec1 = UpsertEdgesSpec(edges=[bad_edge1], namespace="t1")
    res1 = await adapter.upsert_edges(spec1, ctx=ctx)
    assert res1.failed_count == 1

    # Invalid src/dst must raise at validation level
    with pytest.raises(BadRequest):
        bad_edge2 = Edge(
            id=GraphID("e:bad:2"),
            src=GraphID(""),
            dst=GraphID("v:2"),
            label="READ",
            properties={},
        )
        spec2 = UpsertEdgesSpec(edges=[bad_edge2], namespace="t1")
        await adapter.upsert_edges(spec2, ctx=ctx)


async def test_crud_validation_delete_vertex_idempotent(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_crud_del_v", tenant="t1")
    spec = DeleteNodesSpec(ids=[GraphID("v:missing")], namespace="t1")

    res1 = await adapter.delete_nodes(spec, ctx=ctx)
    res2 = await adapter.delete_nodes(spec, ctx=ctx)

    assert isinstance(res1, DeleteResult)
    assert isinstance(res2, DeleteResult)
    assert res1.deleted_count == 1
    assert res2.deleted_count == 1  # idempotent semantics


async def test_crud_validation_delete_edge_idempotent(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_crud_del_e", tenant="t1")
    spec = DeleteEdgesSpec(ids=[GraphID("e:missing")], namespace="t1")

    res1 = await adapter.delete_edges(spec, ctx=ctx)
    res2 = await adapter.delete_edges(spec, ctx=ctx)

    assert isinstance(res1, DeleteResult)
    assert isinstance(res2, DeleteResult)
    assert res1.deleted_count == 1
    assert res2.deleted_count == 1


async def test_crud_validation_properties_are_json_serializable(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_crud_props", tenant="t1")
    # Keys will be normalized by JSON; this should succeed
    node = Node(
        id=GraphID("v:Obj:1"),
        labels=("Obj",),
        properties={1: "one", "two": 2},
    )
    spec = UpsertNodesSpec(nodes=[node], namespace="t1")

    res = await adapter.upsert_nodes(spec, ctx=ctx)
    assert res.upserted_count == 1
