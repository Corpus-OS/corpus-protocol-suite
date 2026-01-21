# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Error mapping & retry hints.

Asserts (Spec refs):
  • Retryable errors include retry_after_ms hints             (§6.3, §12.1, §12.4)
  • Unknown dialects rejected with NotSupported               (§7.4)
  • Input validation surfaces BadRequest                      (§17.2, §7.4)
"""
from __future__ import annotations

import pytest

from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    GraphAdapterError,
    NotSupported,
    BadRequest,
    GraphQuerySpec,
    UpsertEdgesSpec,
    Edge,
    GraphID,
    BaseGraphAdapter,
)

pytestmark = pytest.mark.asyncio


def test_error_handling_retryable_errors_with_hints():
    err = GraphAdapterError("retryable", retry_after_ms=123)
    assert isinstance(err.retry_after_ms, int) and err.retry_after_ms >= 0


async def test_error_handling_bad_request_on_empty_edge_label(adapter: BaseGraphAdapter):
    ctx = GraphContext(request_id="t_err_badreq", tenant="t")
    spec = UpsertEdgesSpec(
        edges=[Edge(id=GraphID("e1"), src=GraphID("v:1"), dst=GraphID("v:2"), label="", properties={}, namespace="ns")],
        namespace="ns",
    )
    with pytest.raises(BadRequest):
        await adapter.upsert_edges(spec, ctx=ctx)


async def test_not_supported_on_unknown_dialect_when_declared(adapter: BaseGraphAdapter):
    caps = await adapter.capabilities()
    declared = tuple(getattr(caps, "supported_query_dialects", ()) or ())
    ctx = GraphContext(request_id="t_err_notsup_dialect", tenant="t")

    unknown = "__sparql_like__"
    spec = GraphQuerySpec(text="RETURN 1", dialect=unknown)

    if declared and unknown not in declared:
        with pytest.raises(NotSupported):
            await adapter.query(spec, ctx=ctx)
        return

    try:
        await adapter.query(spec, ctx=ctx)
    except NotSupported:
        pass


async def test_error_message_includes_dialect_name_when_rejected_due_to_declared_list(adapter: BaseGraphAdapter):
    caps = await adapter.capabilities()
    declared = tuple(getattr(caps, "supported_query_dialects", ()) or ())
    if not declared:
        return

    dialect = "__nope__"
    if dialect in declared:
        return

    ctx = GraphContext(request_id="t_err_dialect_msg", tenant="t")
    with pytest.raises(NotSupported) as ei:
        await adapter.query(GraphQuerySpec(text="RETURN 1", dialect=dialect), ctx=ctx)
    assert dialect in str(ei.value)


def test_graph_adapter_error_details_is_mapping():
    err = GraphAdapterError("x", details={"k": "v"})
    assert isinstance(err.details, dict)
