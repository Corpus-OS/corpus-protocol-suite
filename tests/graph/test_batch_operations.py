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

from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    BatchOperations,
    GraphID,
    BadRequest,
)
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_bulk_vertices_returns_graph_ids():
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_bulk_ids", tenant="t")
    ids = await a.bulk_vertices([("Doc", {"id": "d1"}), ("Doc", {"id": "d2"})], ctx=ctx)
    assert all(isinstance(i, GraphID) for i in ids)


async def test_bulk_vertices_respects_max_batch_ops():
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_bulk_limit", tenant="t")
    with pytest.raises(BadRequest) as ei:
        await a.bulk_vertices([("User", {"i": i}) for i in range(2001)], ctx=ctx)
    assert "max_batch_ops" in str(ei.value)


async def test_batch_operations_returns_results_per_op():
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_batch_results", tenant="t")
    ops = [
        BatchOperations.create_vertex_op("User", {"name": "Ada"}),
        BatchOperations.create_edge_op("READ", "v:User:1", "v:Doc:1", {}),
        {"type": "unknown_op"},
    ]
    res = await a.batch(ops, ctx=ctx)
    assert len(res) == 3
    assert res[0]["ok"] is True and res[1]["ok"] is True and res[2]["ok"] is False


def test_batch_helper_create_vertex_op():
    op = BatchOperations.create_vertex_op("A", {"x": 1})
    assert op["type"] == "create_vertex" and op["label"] == "A" and op["props"]["x"] == 1


def test_batch_helper_create_edge_op():
    op = BatchOperations.create_edge_op("E", "v:1", "v:2", {"w": 1})
    assert op["type"] == "create_edge" and op["from_id"] == "v:1" and op["to_id"] == "v:2"


def test_batch_helper_query_op():
    op = BatchOperations.query_op("cypher", "RETURN 1", {"x": 1})
    assert op["type"] == "query" and op["dialect"] == "cypher" and op["text"] == "RETURN 1"


async def test_batch_size_exceeded_includes_suggestion():
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_bulk_suggest", tenant="t")
    with pytest.raises(BadRequest) as ei:
        await a.bulk_vertices([("U", {"i": i}) for i in range(5000)], ctx=ctx)
    err = ei.value
    assert getattr(err, "suggested_batch_reduction", None) is not None
