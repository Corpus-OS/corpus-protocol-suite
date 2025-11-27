# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — SIEM-safe metrics & context propagation.

Asserts (Spec refs):
  • metrics emitted with hashed tenant, never raw             (§13.1, §13.2, §6.1)
  • no raw query text in metrics dimensions                   (§13.2)
  • error path emits metrics                                  (§13.1, §12)
  • batch metrics include op count                            (§13.1)
"""
from typing import Optional, Mapping, Any, List

import pytest

from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    MetricsSink,
    BaseGraphAdapter,
    GraphQuerySpec,
    BatchOperation,
)

pytestmark = pytest.mark.asyncio


class CaptureMetrics(MetricsSink):
    def __init__(self) -> None:
        self.observations: List[dict] = []
        self.counters: List[dict] = []

    def observe(
        self,
        *,
        component: str,
        op: str,
        ms: float,
        ok: bool,
        code: str = "OK",
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.observations.append(
            {"component": component, "op": op, "ok": ok, "code": code, "extra": dict(extra or {})}
        )

    def counter(
        self,
        *,
        component: str,
        name: str,
        value: int = 1,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.counters.append(
            {"component": component, "name": name, "value": value, "extra": dict(extra or {})}
        )


async def test_observability_context_propagates_to_metrics_siem_safe(
    adapter: BaseGraphAdapter,
):
    m = CaptureMetrics()
    # Attach metrics sink directly to adapter
    adapter._metrics = m  # type: ignore[attr-defined]

    ctx = GraphContext(request_id="t_siem_ctx", tenant="acme")
    spec = GraphQuerySpec(text="RETURN 1", dialect="cypher")
    await adapter.query(spec, ctx=ctx)

    assert any(o["op"] == "query" for o in m.observations)


async def test_observability_tenant_hashed_never_raw(adapter: BaseGraphAdapter):
    m = CaptureMetrics()
    adapter._metrics = m  # type: ignore[attr-defined]

    tenant = "super-secret-tenant"
    ctx = GraphContext(request_id="t_siem_hash", tenant=tenant)
    spec = GraphQuerySpec(text="RETURN 1", dialect="cypher")
    await adapter.query(spec, ctx=ctx)

    extras = [o["extra"] for o in m.observations if o["op"] == "query"]
    assert extras
    last_extra = extras[-1]
    # BaseGraphAdapter stores tenant as a hash under 'tenant_hash'
    assert "tenant_hash" in last_extra
    assert tenant not in str(last_extra)


async def test_observability_no_query_text_in_metrics(adapter: BaseGraphAdapter):
    m = CaptureMetrics()
    adapter._metrics = m  # type: ignore[attr-defined]

    ctx = GraphContext(request_id="t_siem_noq", tenant="t")
    spec = GraphQuerySpec(text="MATCH (n) RETURN n LIMIT 2", dialect="cypher")
    await adapter.query(spec, ctx=ctx)

    extra = [o["extra"] for o in m.observations if o["op"] == "query"][-1]
    for k in ("text", "query"):
        assert k not in extra


async def test_observability_metrics_emitted_on_error_path(adapter: BaseGraphAdapter):
    m = CaptureMetrics()
    adapter._metrics = m  # type: ignore[attr-defined]

    ctx = GraphContext(request_id="t_siem_err", tenant="t")
    bad_spec = GraphQuerySpec(text="g.V()", dialect="__no_such_dialect__")

    with pytest.raises(Exception):
        await adapter.query(bad_spec, ctx=ctx)

    assert any((o["ok"] is False and o["op"] == "query") for o in m.observations)


async def test_observability_query_metrics_include_dialect(adapter: BaseGraphAdapter):
    m = CaptureMetrics()
    adapter._metrics = m  # type: ignore[attr-defined]

    ctx = GraphContext(request_id="t_siem_dialect", tenant="t")
    spec = GraphQuerySpec(text="RETURN 1", dialect="cypher")
    await adapter.query(spec, ctx=ctx)

    extra = [o["extra"] for o in m.observations if o["op"] == "query"][-1]
    assert "dialect" in extra
    assert extra["dialect"] in ("cypher", "none")


async def test_observability_batch_metrics_include_op_count(adapter: BaseGraphAdapter):
    m = CaptureMetrics()
    adapter._metrics = m  # type: ignore[attr-defined]

    ctx = GraphContext(request_id="t_siem_batch", tenant="t")
    ops = [
        BatchOperation(
            op="query",
            args={"text": "RETURN 1", "dialect": "cypher"},
        )
    ]

    await adapter.batch(ops, ctx=ctx)

    # batch() uses metric_extra={"ops": len(ops)} in the observation
    extra = [o["extra"] for o in m.observations if o["op"] == "batch"][-1]
    assert "ops" in extra and extra["ops"] == len(ops)
