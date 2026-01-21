# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Batch & bulk operations.

Asserts (Spec refs):
  • bulk_vertices returns nodes with GraphID IDs              (§7.3.3)
  • max_batch_ops enforced with guidance                     (§7.2, §7.3.3, §12.5)
  • batch() returns per-op results                           (§7.3.3, §12.5)
  • transaction/traversal capability alignment and basic invariants
"""
from __future__ import annotations

from typing import Any, Mapping

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
    GraphTraversalSpec,
    TraversalResult,
)

pytestmark = pytest.mark.asyncio


async def test_batch_ops_bulk_vertices_returns_graph_ids(adapter):
    """bulk_vertices must return nodes whose IDs are GraphID-compatible."""
    caps = await adapter.capabilities()
    ctx = GraphContext(request_id="t_bulk_ids", tenant="t")

    spec = BulkVerticesSpec(namespace="docs", limit=5)

    if not getattr(caps, "supports_bulk_vertices", False):
        with pytest.raises(NotSupported):
            await adapter.bulk_vertices(spec, ctx=ctx)
        return

    result = await adapter.bulk_vertices(spec, ctx=ctx)
    assert isinstance(result, BulkVerticesResult)
    assert isinstance(result.nodes, list)

    for n in result.nodes:
        assert isinstance(n, Node)
        assert isinstance(n.id, str)  # GraphID is NewType(str)
        _gid = GraphID(n.id)
        assert isinstance(_gid, str)


async def test_batch_ops_batch_respects_max_batch_ops(adapter):
    """
    If adapter declares max_batch_ops, batch() must enforce it.
    If adapter enforces a size limit, it must declare max_batch_ops.
    """
    caps = await adapter.capabilities()
    ctx = GraphContext(request_id="t_batch_limit", tenant="t")

    if not getattr(caps, "supports_batch", False):
        with pytest.raises(NotSupported):
            await adapter.batch([BatchOperation(op="query", args={"text": "RETURN 1"})], ctx=ctx)
        return

    max_ops = getattr(caps, "max_batch_ops", None)
    if max_ops is None:
        # If a size limit is enforced anyway, that's a capabilities mismatch.
        ops = [BatchOperation(op="query", args={"text": "RETURN 1"}) for _ in range(100)]
        try:
            await adapter.batch(ops, ctx=ctx)
        except BadRequest as e:
            details = getattr(e, "details", {}) or {}
            msg = str(e).lower()
            looks_like_size_limit = ("max_batch_ops" in details) or ("exceed" in msg and "batch" in msg)
            assert not looks_like_size_limit, (
                "Adapter enforced a batch size limit but did not declare max_batch_ops in capabilities."
            )
        return

    too_many = int(max_ops) + 1
    ops = [BatchOperation(op="query", args={"text": "RETURN 1"}) for _ in range(too_many)]
    with pytest.raises(BadRequest) as ei:
        await adapter.batch(ops, ctx=ctx)

    msg = str(ei.value).lower()
    assert any(term in msg for term in ["max_batch_ops", "exceed", "maximum"])


async def test_batch_ops_batch_operations_returns_results_per_op(adapter):
    """
    batch() must return per-operation results when it succeeds.
    For unknown ops, adapters may either:
      - return a per-op error entry, OR
      - reject the batch with BadRequest (strict validation).
    """
    caps = await adapter.capabilities()
    ctx = GraphContext(request_id="t_batch_results", tenant="t")

    ops = [
        BatchOperation(
            op="upsert_nodes",
            args={
                "nodes": [{"id": "n1", "labels": ["User"], "properties": {"name": "Ada"}, "namespace": "ns1"}],
                "namespace": "ns1",
            },
        ),
        BatchOperation(
            op="upsert_edges",
            args={
                "edges": [{"id": "e1", "src": "n1", "dst": "n1", "label": "SELF", "properties": {}, "namespace": "ns1"}],
                "namespace": "ns1",
            },
        ),
        BatchOperation(op="unknown_op", args={}),
    ]

    if not getattr(caps, "supports_batch", False):
        with pytest.raises(NotSupported):
            await adapter.batch(ops, ctx=ctx)
        return

    try:
        result = await adapter.batch(ops, ctx=ctx)
    except BadRequest:
        return

    assert isinstance(result, BatchResult)
    assert isinstance(result.results, list)
    assert len(result.results) == 3

    # Tolerant: entries may be mappings or dataclasses. If a mapping has ok, it must be bool.
    for r in result.results:
        if isinstance(r, Mapping) and "ok" in r:
            assert isinstance(r["ok"], bool)

    # If unknown op is represented as mapping result, it must be an error.
    r3 = result.results[2]
    if isinstance(r3, Mapping) and "ok" in r3:
        assert r3.get("ok") is False
        code = r3.get("code")
        if code is not None:
            assert isinstance(code, str)


async def test_batch_ops_batch_size_exceeded_includes_hint(adapter):
    """
    When batch size exceeds adapter limits, BadRequest should include a machine-actionable hint
    in details (e.g., max_batch_ops and/or suggested_batch_reduction), if a limit is declared.
    """
    caps = await adapter.capabilities()
    ctx = GraphContext(request_id="t_batch_hint", tenant="t")

    if not getattr(caps, "supports_batch", False):
        with pytest.raises(NotSupported):
            await adapter.batch([BatchOperation(op="query", args={"text": "RETURN 1"})], ctx=ctx)
        return

    if getattr(caps, "max_batch_ops", None) is None:
        # No declared limit => do not require hint fields.
        return

    too_many = int(caps.max_batch_ops) * 2
    ops = [BatchOperation(op="query", args={"text": "RETURN 1"}) for _ in range(too_many)]

    with pytest.raises(BadRequest) as ei:
        await adapter.batch(ops, ctx=ctx)

    details = getattr(ei.value, "details", None) or {}
    assert isinstance(details, dict)
    assert any(k in details for k in ("max_batch_ops", "suggested_batch_reduction")) or (
        "max_batch_ops" in str(ei.value).lower()
    )


# ---------------------------- NEW: bulk pagination ----------------------------

async def test_bulk_vertices_pagination_invariants_when_supported(adapter):
    """
    NEW: If bulk_vertices is supported, pagination flags must be consistent:
      - has_more == True  => next_cursor is a non-empty string
      - has_more == False => next_cursor is None
    """
    caps = await adapter.capabilities()
    ctx = GraphContext(request_id="t_bulk_page_invariants", tenant="t")

    spec = BulkVerticesSpec(namespace="docs", limit=5, cursor=None)

    if not getattr(caps, "supports_bulk_vertices", False):
        with pytest.raises(NotSupported):
            await adapter.bulk_vertices(spec, ctx=ctx)
        return

    res = await adapter.bulk_vertices(spec, ctx=ctx)
    assert isinstance(res, BulkVerticesResult)
    assert isinstance(res.has_more, bool)
    if res.has_more:
        assert isinstance(res.next_cursor, str) and res.next_cursor
    else:
        assert res.next_cursor is None


async def test_bulk_vertices_cursor_progresses_when_supported(adapter):
    """
    NEW: If bulk_vertices is supported and returns has_more, the next page should not be identical.
    """
    caps = await adapter.capabilities()
    ctx = GraphContext(request_id="t_bulk_cursor_progress", tenant="t")

    spec1 = BulkVerticesSpec(namespace="docs", limit=5, cursor=None)

    if not getattr(caps, "supports_bulk_vertices", False):
        with pytest.raises(NotSupported):
            await adapter.bulk_vertices(spec1, ctx=ctx)
        return

    page1 = await adapter.bulk_vertices(spec1, ctx=ctx)
    ids1 = [str(n.id) for n in page1.nodes]

    if not page1.has_more or not page1.next_cursor:
        # Single-page adapters are allowed.
        return

    page2 = await adapter.bulk_vertices(BulkVerticesSpec(namespace="docs", limit=5, cursor=page1.next_cursor), ctx=ctx)
    ids2 = [str(n.id) for n in page2.nodes]
    assert ids1 != ids2


# ---------------------------- NEW: transaction ----------------------------

async def test_transaction_success_path_when_supported(adapter):
    """
    NEW: If supports_transaction is True, transaction() must succeed and return BatchResult.
    If False, it must raise NotSupported.
    """
    caps = await adapter.capabilities()
    ctx = GraphContext(request_id="t_tx_success", tenant="t")

    ops = [BatchOperation(op="query", args={"text": "RETURN 1"})]

    if not getattr(caps, "supports_transaction", False):
        with pytest.raises(NotSupported):
            await adapter.transaction(ops, ctx=ctx)
        return

    res = await adapter.transaction(ops, ctx=ctx)
    assert isinstance(res, BatchResult)
    assert isinstance(res.results, list)
    assert isinstance(res.success, bool)
    # transaction_id is optional but if present must be a string
    if res.transaction_id is not None:
        assert isinstance(res.transaction_id, str) and res.transaction_id


async def test_transaction_enforces_max_batch_ops_when_declared(adapter):
    """
    NEW: If supports_transaction and max_batch_ops is declared, exceeding it must raise BadRequest.
    """
    caps = await adapter.capabilities()
    ctx = GraphContext(request_id="t_tx_max_ops", tenant="t")

    if not getattr(caps, "supports_transaction", False):
        with pytest.raises(NotSupported):
            await adapter.transaction([BatchOperation(op="query", args={"text": "RETURN 1"})], ctx=ctx)
        return

    max_ops = getattr(caps, "max_batch_ops", None)
    if max_ops is None:
        return

    ops = [BatchOperation(op="query", args={"text": "RETURN 1"}) for _ in range(int(max_ops) + 1)]
    with pytest.raises(BadRequest):
        await adapter.transaction(ops, ctx=ctx)


# ---------------------------- NEW: traversal ----------------------------

async def test_traversal_success_path_when_supported(adapter):
    """
    NEW: If supports_traversal is True, traversal() must succeed and return TraversalResult.
    If False, it must raise NotSupported.
    """
    caps = await adapter.capabilities()
    ctx = GraphContext(request_id="t_trav_success", tenant="t")

    spec = GraphTraversalSpec(start_nodes=["v:start:1"], max_depth=1, direction="OUTGOING")

    if not getattr(caps, "supports_traversal", False):
        with pytest.raises(NotSupported):
            await adapter.traversal(spec, ctx=ctx)
        return

    res = await adapter.traversal(spec, ctx=ctx)
    assert isinstance(res, TraversalResult)
    assert isinstance(res.nodes, list)
    assert isinstance(res.relationships, list)
    assert isinstance(res.paths, list)
    assert isinstance(res.summary, Mapping)


async def test_traversal_enforces_max_depth_when_declared(adapter):
    """
    NEW: If max_traversal_depth is declared, exceeding it must raise BadRequest.
    """
    caps = await adapter.capabilities()
    ctx = GraphContext(request_id="t_trav_max_depth", tenant="t")

    if not getattr(caps, "supports_traversal", False):
        with pytest.raises(NotSupported):
            await adapter.traversal(GraphTraversalSpec(start_nodes=["v:start:1"], max_depth=1), ctx=ctx)
        return

    max_depth = getattr(caps, "max_traversal_depth", None)
    if max_depth is None:
        return

    with pytest.raises(BadRequest):
        await adapter.traversal(GraphTraversalSpec(start_nodes=["v:start:1"], max_depth=int(max_depth) + 1), ctx=ctx)
