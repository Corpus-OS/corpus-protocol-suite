# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — SIEM-safe metrics & context propagation.

Asserts (Spec refs):
  • metrics emitted with hashed tenant, never raw             (§13.1, §13.2, §6.1)
  • no raw query text in metrics dimensions                   (§13.2)
  • error path emits metrics                                  (§13.1, §12)
  • batch metrics include op count                            (§13.1)
"""
import pytest
from typing import Optional, Mapping, Any, List

from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    MetricsSink,
)
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


class CaptureMetrics(MetricsSink):
    def __init__(self) -> None:
        self.observations: List[dict] = []
        self.counters: List[dict] = []
    def observe(self, *, component: str, op: str, ms: float, ok: bool, code: str = "OK", extra: Optional[Mapping[str, Any]] = None) -> None:
        self.observations.append({"component": component, "op": op, "ok": ok, "code": code, "extra": dict(extra or {})})
    def counter(self, *, component: str, name: str, value: int = 1, extra: Optional[Mapping[str, Any]] = None) -> None:
        self.counters.append({"component": component, "name": name, "value": value, "extra": dict(extra or {})})


async def test_context_propagates_to_metrics_siem_safe():
    m = CaptureMetrics()
    a = MockGraphAdapter(metrics=m)
    ctx = make_ctx(GraphContext, request_id="t_siem_ctx", tenant="acme")
    await a.query(dialect="cypher", text="RETURN 1", ctx=ctx)
    assert any(o["op"] == "query" for o in m.observations)


async def test_tenant_hashed_never_raw():
    m = CaptureMetrics()
    tenant = "super-secret-tenant"
    a = MockGraphAdapter(metrics=m)
    ctx = make_ctx(GraphContext, request_id="t_siem_hash", tenant=tenant)
    await a.query(dialect="cypher", text="RETURN 1", ctx=ctx)
    extras = [o["extra"] for o in m.observations if o["op"] == "query"]
    assert extras and "tenant" in extras[-1]
    assert tenant not in str(extras[-1])


async def test_no_query_text_in_metrics():
    m = CaptureMetrics()
    a = MockGraphAdapter(metrics=m)
    ctx = make_ctx(GraphContext, request_id="t_siem_noq", tenant="t")
    await a.query(dialect="cypher", text="MATCH (n) RETURN n LIMIT 2", ctx=ctx)
    extra = [o["extra"] for o in m.observations if o["op"] == "query"][-1]
    for k in ("text", "query"):
        assert k not in extra


async def test_metrics_emitted_on_error_path():
    m = CaptureMetrics()
    a = MockGraphAdapter(metrics=m)
    ctx = make_ctx(GraphContext, request_id="t_siem_err", tenant="t")
    with pytest.raises(Exception):
        await a.query(dialect="gremlin", text="g.V()", ctx=ctx)
    assert any((o["ok"] is False and o["op"] == "query") for o in m.observations)


async def test_query_metrics_include_dialect():
    m = CaptureMetrics()
    a = MockGraphAdapter(metrics=m)
    ctx = make_ctx(GraphContext, request_id="t_siem_dialect", tenant="t")
    await a.query(dialect="cypher", text="RETURN 1", ctx=ctx)
    extra = [o["extra"] for o in m.observations if o["op"] == "query"][-1]
    assert "dialect" in extra


async def test_batch_metrics_include_op_count():
    m = CaptureMetrics()
    a = MockGraphAdapter(metrics=m)
    ctx = make_ctx(GraphContext, request_id="t_siem_batch", tenant="t")
    ops = [{"type": "create_vertex", "label": "X", "props": {}}]
    await a.batch(ops, ctx=ctx)
    extra = [o["extra"] for o in m.observations if o["op"] == "batch"][-1]
    assert "ops" in extra and extra["ops"] == 1
