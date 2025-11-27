# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Error mapping & retry hints.

Asserts (Spec refs):
  • Retryable errors include retry_after_ms hints             (§6.3, §12.1, §12.4)
  • Unknown dialects rejected with NotSupported               (§7.4)
  • Input validation surfaces BadRequest                      (§17.2, §7.4)
"""
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
)


def test_error_handling_retryable_errors_with_hints():
    """
    Validate that GraphAdapterError supports retry_after_ms hints
    for retryable errors.
    """
    err = GraphAdapterError("retryable", retry_after_ms=123)
    assert getattr(err, "retry_after_ms", None) is not None
    assert isinstance(err.retry_after_ms, int) and err.retry_after_ms >= 0


async def test_error_handling_not_supported_on_unknown_dialect(adapter):
    """
    Unknown dialects must surface NotSupported errors.
    """
    caps = await adapter.capabilities()
    supported = getattr(caps, "supported_query_dialects", ()) or ()

    unknown = "__sparql_like__"
    if unknown in supported:
        pytest.skip(f"Adapter unexpectedly supports '{unknown}' dialect")

    ctx = GraphContext(
        request_id="t_err_notsup",
        tenant="t",
    )
    spec = GraphQuerySpec(text="SELECT * WHERE {}", dialect=unknown)

    with pytest.raises(NotSupported):
        await adapter.query(spec, ctx=ctx)


async def test_error_handling_bad_request_on_empty_edge_label(adapter):
    """
    Invalid edge label must surface BadRequest from BaseGraphAdapter validation.
    """
    ctx = GraphContext(
        request_id="t_err_badreq",
        tenant="t",
    )

    spec = UpsertEdgesSpec(
        edges=[
            Edge(
                id=GraphID("e1"),
                src=GraphID("v:1"),
                dst=GraphID("v:2"),
                label="",  # invalid (empty)
                properties={},
                namespace="ns",
            )
        ],
        namespace="ns",
    )

    with pytest.raises(BadRequest):
        await adapter.upsert_edges(spec, ctx=ctx)


async def test_error_message_includes_dialect_name_in_not_supported(adapter):
    """
    NotSupported error messages for dialect validation should include the
    offending dialect name for debuggability.
    """
    caps = await adapter.capabilities()
    supported = getattr(caps, "supported_query_dialects", ()) or ()

    dialect = "gql"
    if dialect in supported:
        pytest.skip(f"Adapter unexpectedly supports '{dialect}' dialect")

    ctx = GraphContext(request_id="t_err_dialect_msg", tenant="t")
    spec = GraphQuerySpec(text="{}", dialect=dialect)

    with pytest.raises(NotSupported) as ei:
        await adapter.query(spec, ctx=ctx)

    assert dialect in str(ei.value)
