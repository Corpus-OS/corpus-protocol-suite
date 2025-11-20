# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Error mapping & retry hints.

Asserts (Spec refs):
  • Retryable errors include retry_after_ms hints             (§6.3, §12.1, §12.4)
  • Operation and dialect fields propagate in errors          (§6.3)
  • Input validation surfaces BadRequest / NotSupported       (§17.2, §7.4)
"""
import pytest

from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    AdapterError,
    NotSupported,
    BadRequest,
)

pytestmark = pytest.mark.asyncio


def make_ctx(ctx_cls, **kwargs):
    """Local helper to construct an OperationContext."""
    return ctx_cls(**kwargs)


def test_error_handling_retryable_errors_with_hints():
    """
    Validate that AdapterError supports retry_after_ms hints for retryable errors.
    """
    err = AdapterError("retryable", retry_after_ms=123)
    assert getattr(err, "retry_after_ms", None) is not None
    assert isinstance(err.retry_after_ms, int) and err.retry_after_ms >= 0


async def test_error_handling_error_includes_operation_field(adapter):
    caps = await adapter.capabilities()
    if getattr(caps, "max_batch_ops", None) is None:
        pytest.skip("Adapter does not declare max_batch_ops; cannot trigger bulk_vertices error")

    ctx = make_ctx(
        GraphContext,
        request_id="t_err_opfield",
        tenant="t",
    )
    too_many = caps.max_batch_ops * 2
    with pytest.raises(BadRequest) as ei:
        await adapter.bulk_vertices(
            [("U", {"i": i}) for i in range(too_many)],
            ctx=ctx,
        )
    assert getattr(ei.value, "operation", None) == "bulk_vertices"


async def test_error_handling_error_includes_dialect_field(adapter):
    caps = await adapter.capabilities()
    unknown = "__err_dialect__"
    assert unknown not in caps.dialects

    ctx = make_ctx(
        GraphContext,
        request_id="t_err_dialect",
        tenant="t",
    )
    with pytest.raises(NotSupported) as ei:
        await adapter.query(dialect=unknown, text="g.V()", ctx=ctx)
    assert getattr(ei.value, "dialect", None) == unknown


async def test_error_handling_bad_request_on_empty_label(adapter):
    ctx = make_ctx(
        GraphContext,
        request_id="t_err_badreq",
        tenant="t",
    )
    with pytest.raises(BadRequest):
        await adapter.create_vertex("", {"x": 1}, ctx=ctx)


async def test_error_handling_not_supported_on_unknown_dialect(adapter):
    caps = await adapter.capabilities()
    unknown = "__sparql_like__"
    assert unknown not in caps.dialects

    ctx = make_ctx(
        GraphContext,
        request_id="t_err_notsup",
        tenant="t",
    )
    with pytest.raises(NotSupported):
        await adapter.query(dialect=unknown, text="SELECT * WHERE {}", ctx=ctx)
