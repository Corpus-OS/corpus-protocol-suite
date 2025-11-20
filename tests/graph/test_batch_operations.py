# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Batch & bulk operations.

Asserts (Spec refs):
  • bulk_vertices returns GraphID list                        (§7.3.3)
  • max_batch_ops enforced with guidance                     (§7.2, §7.3.3, §12.5)
  • batch() returns per-op results                           (§7.3.3, §12.5)
  • helpers construct well-formed ops                        (§7.3.3)
"""
import pytest

from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    BatchOperations,
    GraphID,
    BadRequest,
)

pytestmark = pytest.mark.asyncio


async def test_batch_ops_bulk_vertices_returns_graph_ids(adapter):
    ctx = GraphContext(request_id="t_bulk_ids", tenant="t")
    ids = await adapter.bulk_vertices(
        [("Doc", {"id": "d1"}), ("Doc", {"id": "d2"})],
        ctx=ctx,
    )
    assert all(isinstance(i, GraphID) for i in ids)


async def test_batch_ops_bulk_vertices_respects_max_batch_ops(adapter):
    caps = await adapter.capabilities()
    if getattr(caps, "max_batch_ops", None) is None:
        pytest.skip("Adapter does not declare max_batch_ops; cannot enforce max batch ops test")

    ctx = GraphContext(request_id="t_bulk_limit", tenant="t")
    too_many = caps.max_batch_ops + 1
    with pytest.raises(BadRequest) as ei:
        await adapter.bulk_vertices(
            [("User", {"i": i}) for i in range(too_many)],
            ctx=ctx,
        )
    assert "max_batch_ops" in str(ei.value)


async def test_batch_ops_batch_operations_returns_results_per_op(adapter):
    ctx = GraphContext(request_id="t_batch_results", tenant="t")
    ops = [
        BatchOperations.create_vertex_op("User", {"name": "Ada"}),
        BatchOperations.create_edge_op("READ", "v:User:1", "v:Doc:1", {}),
        {"type": "unknown_op"},
    ]
    res = await adapter.batch(ops, ctx=ctx)
    assert len(res) == 3
    assert res[0]["ok"] is True
    assert res[1]["ok"] is True
    assert res[2]["ok"] is False


def test_batch_ops_helper_create_vertex_op():
    op = BatchOperations.create_vertex_op("A", {"x": 1})
    assert op["type"] == "create_vertex"
    assert op["label"] == "A"
    assert op["props"]["x"] == 1


def test_batch_ops_helper_create_edge_op():
    op = BatchOperations.create_edge_op("E", "v:1", "v:2", {"w": 1})
    assert op["type"] == "create_edge"
    assert op["from_id"] == "v:1"
    assert op["to_id"] == "v:2"
    assert op["props"]["w"] == 1


def test_batch_ops_helper_query_op():
    op = BatchOperations.query_op("cypher", "RETURN 1", {"x": 1})
    assert op["type"] == "query"
    assert op["dialect"] == "cypher"
    assert op["text"] == "RETURN 1"
    assert op["params"]["x"] == 1


async def test_batch_ops_batch_size_exceeded_includes_suggestion(adapter):
    caps = await adapter.capabilities()
    if getattr(caps, "max_batch_ops", None) is None:
        pytest.skip("Adapter does not declare max_batch_ops; cannot enforce suggestion hint test")

    ctx = GraphContext(request_id="t_bulk_suggest", tenant="t")
    too_many = caps.max_batch_ops * 2
    with pytest.raises(BadRequest) as ei:
        await adapter.bulk_vertices(
            [("U", {"i": i}) for i in range(too_many)],
            ctx=ctx,
        )
    err = ei.value
    assert getattr(err, "suggested_batch_reduction", None) is not None
