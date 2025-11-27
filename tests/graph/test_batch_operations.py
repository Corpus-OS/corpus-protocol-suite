# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Batch & bulk operations.

Asserts (Spec refs):
  • bulk_vertices returns nodes with GraphID IDs              (§7.3.3)
  • max_batch_ops enforced with guidance                     (§7.2, §7.3.3, §12.5)
  • batch() returns per-op results                           (§7.3.3, §12.5)
"""
import pytest

from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    GraphID,
    BadRequest,
    NotSupported,
    BulkVerticesSpec,
    BulkVerticesResult,
    BatchOperation,
    BatchResult,
    Node,
)

pytestmark = pytest.mark.asyncio


async def test_batch_ops_bulk_vertices_returns_graph_ids(adapter):
    """bulk_vertices must return nodes whose IDs are GraphID-compatible."""
    ctx = GraphContext(request_id="t_bulk_ids", tenant="t")

    spec = BulkVerticesSpec(namespace="docs", limit=5)
    try:
        result = await adapter.bulk_vertices(spec, ctx=ctx)
    except NotSupported:
        pytest.skip("Adapter does not implement bulk_vertices")

    assert isinstance(result, BulkVerticesResult)
    assert isinstance(result.nodes, list)

    for n in result.nodes:
        assert isinstance(n, Node)
        # GraphID is a NewType over str, so runtime type is str
        assert isinstance(n.id, str)
        # round-trip through GraphID to ensure type compatibility
        _gid = GraphID(n.id)
        assert isinstance(_gid, str)


async def test_batch_ops_batch_respects_max_batch_ops(adapter):
    """
    If adapter declares max_batch_ops, batch() must enforce it.
    """
    caps = await adapter.capabilities()
    if getattr(caps, "max_batch_ops", None) is None:
        pytest.skip("Adapter does not declare max_batch_ops; cannot enforce max batch ops test")

    ctx = GraphContext(request_id="t_batch_limit", tenant="t")
    too_many = caps.max_batch_ops + 1

    # Use simple query ops as the batch payload
    dialects = getattr(caps, "supported_query_dialects", ()) or ("cypher",)
    ops = [
        BatchOperation(
            op="query",
            args={"text": "RETURN 1", "dialect": dialects[0]},
        )
        for _ in range(too_many)
    ]

    with pytest.raises(BadRequest) as ei:
        await adapter.batch(ops, ctx=ctx)

    msg = str(ei.value).lower()
    # BaseGraphAdapter includes "max_batch_ops" in the error details; message
    # will mention exceeding the maximum.
    assert any(term in msg for term in ["max_batch_ops", "exceeds maximum", "exceeds maximum of"])


async def test_batch_ops_batch_operations_returns_results_per_op(adapter):
    """
    batch() must return per-operation results.
    """
    ctx = GraphContext(request_id="t_batch_results", tenant="t")

    ops = [
        # 1) upsert_nodes
        BatchOperation(
            op="upsert_nodes",
            args={
                "nodes": [
                    {
                        "id": "n1",
                        "labels": ["User"],
                        "properties": {"name": "Ada"},
                        "namespace": "ns1",
                    }
                ],
                "namespace": "ns1",
            },
        ),
        # 2) upsert_edges
        BatchOperation(
            op="upsert_edges",
            args={
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
                "namespace": "ns1",
            },
        ),
        # 3) unknown op to exercise error path
        BatchOperation(
            op="unknown_op",
            args={},
        ),
    ]

    result = await adapter.batch(ops, ctx=ctx)
    assert isinstance(result, BatchResult)
    assert isinstance(result.results, list)
    assert len(result.results) == 3

    r1, r2, r3 = result.results

    assert r1.get("ok") is True
    assert r2.get("ok") is True
    assert r3.get("ok") is False
    assert r3.get("code") in ("NOT_SUPPORTED", "NOTSUPPORTED")


async def test_batch_ops_batch_size_exceeded_includes_hint(adapter):
    """
    When batch size exceeds adapter limits, BadRequest should include
    a details dict with some machine-actionable hint (max_batch_ops and/or
    suggested_batch_reduction).
    """
    caps = await adapter.capabilities()
    if getattr(caps, "max_batch_ops", None) is None:
        pytest.skip("Adapter does not declare max_batch_ops; cannot enforce suggestion hint test")

    ctx = GraphContext(request_id="t_batch_hint", tenant="t")
    too_many = caps.max_batch_ops * 2

    dialects = getattr(caps, "supported_query_dialects", ()) or ("cypher",)
    ops = [
        BatchOperation(
            op="query",
            args={"text": "RETURN 1", "dialect": dialects[0]},
        )
        for _ in range(too_many)
    ]

    with pytest.raises(BadRequest) as ei:
        await adapter.batch(ops, ctx=ctx)

    err = ei.value
    details = getattr(err, "details", None) or {}
    assert isinstance(details, dict)
    # Different adapters may expose different hints; look for at least one.
    assert any(
        k in details for k in ("max_batch_ops", "suggested_batch_reduction")
    ), f"Expected hint field in error details, got: {details!r}"
