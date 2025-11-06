# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Error mapping & retry hints.

Asserts (Spec refs):
  • Retryable errors include retry_after_ms hints             (§6.3, §12.1, §12.4)
  • Operation and dialect fields propagate in errors          (§6.3)
  • Input validation surfaces BadRequest / NotSupported       (§17.2, §7.4)
"""
import random
import pytest

from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    AdapterError,
    NotSupported,
    BadRequest,
)
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_retryable_errors_with_hints():
    random.seed(202)  # deterministic sequence
    a = MockGraphAdapter(failure_rate=0.9)
    ctx = make_ctx(GraphContext, request_id="t_err_retry", tenant="t")
    with pytest.raises(AdapterError) as ei:
        await a.query(dialect="cypher", text="RETURN 1", ctx=ctx)
    err = ei.value
    assert getattr(err, "retry_after_ms", None) is not None


async def test_error_includes_operation_field():
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_err_opfield", tenant="t")
    with pytest.raises(BadRequest) as ei:
        await a.bulk_vertices([("U", {"i": i}) for i in range(5000)], ctx=ctx)
    assert getattr(ei.value, "operation", None) == "bulk_vertices"


async def test_error_includes_dialect_field():
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_err_dialect", tenant="t")
    with pytest.raises(NotSupported) as ei:
        await a.query(dialect="gremlin", text="g.V()", ctx=ctx)
    assert getattr(ei.value, "dialect", None) == "gremlin"


async def test_bad_request_on_empty_label():
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_err_badreq", tenant="t")
    with pytest.raises(BadRequest):
        await a.create_vertex("", {"x": 1}, ctx=ctx)


async def test_not_supported_on_unknown_dialect():
    a = MockGraphAdapter()
    ctx = make_ctx(GraphContext, request_id="t_err_notsup", tenant="t")
    with pytest.raises(NotSupported):
        await a.query(dialect="sparql", text="SELECT * WHERE {}", ctx=ctx)
