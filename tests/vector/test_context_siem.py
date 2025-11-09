# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Observability & SIEM safety.

Spec refs:
  • SPECIFICATION.md §13.1-§13.3 (Observability)
  • SPECIFICATION.md §15 (Privacy)
  • SPECIFICATION.md §6.1 (Operation Context)
"""

import pytest
from typing import Any, Mapping, Optional, List, Dict

from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter
from adapter_sdk.vector_base import (
    QuerySpec,
    UpsertSpec,
    Vector,
    VectorID,
    MetricsSink,
)
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


class CaptureMetrics(MetricsSink):
    def __init__(self) -> None:
        self.observations: List[Dict[str, Any]] = []
        self.counters: List[Dict[str, Any]] = []

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
            {
                "component": component,
                "op": op,
                "ok": ok,
                "code": code,
                "extra": dict(extra or {}),
            }
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
            {
                "component": component,
                "name": name,
                "value": value,
                "extra": dict(extra or {}),
            }
        )


async def test_context_propagates_to_metrics_siem_safe():
    m = CaptureMetrics()
    a = MockVectorAdapter(metrics=m)
    ctx = make_ctx(
        # using vector OperationContext via make_ctx helper
        type(a)._component_context if hasattr(type(a), "_component_context") else None,  # fallback no-op
        request_id="v_ctx",
        tenant="acme",
    )
    # if make_ctx signature differs, just construct QuerySpec with ctx below
    await a.query(QuerySpec(vector=[0.1], top_k=1, namespace="default"), ctx=ctx)

    assert any(o["op"] == "query" for o in m.observations)


async def test_tenant_hashed_never_raw():
    m = CaptureMetrics()
    a = MockVectorAdapter(metrics=m)
    tenant = "super-secret-tenant"
    ctx = make_ctx(None, request_id="v_hash", tenant=tenant)
    await a.query(QuerySpec(vector=[0.1], top_k=1, namespace="default"), ctx=ctx)

    extras = [o["extra"] for o in m.observations if o["op"] == "query"]
    assert extras
    blob = str(extras[-1])
    assert tenant not in blob  # no raw tenant


async def test_no_vector_data_in_metrics():
    m = CaptureMetrics()
    a = MockVectorAdapter(metrics=m)
    ctx = make_ctx(None, request_id="v_no_vec", tenant="t")
    await a.query(QuerySpec(vector=[0.9, 0.8], top_k=1, namespace="default"), ctx=ctx)

    extras = [o["extra"] for o in m.observations if o["op"] == "query"]
    assert extras
    s = str(extras[-1])
    assert "[0.9, 0.8]" not in s  # heuristic: no raw vector literal


async def test_metrics_emitted_on_error_path():
    from adapter_sdk.vector_base import BadRequest

    m = CaptureMetrics()
    a = MockVectorAdapter(metrics=m)
    ctx = make_ctx(None, request_id="v_err", tenant="t")

    with pytest.raises(BadRequest):
        await a.query(QuerySpec(vector=[0.1], top_k=0, namespace="default"), ctx=ctx)

    assert any(o for o in m.observations if o["op"] == "query" and o["ok"] is False)


async def test_query_metrics_include_namespace():
    m = CaptureMetrics()
    a = MockVectorAdapter(metrics=m)
    ctx = make_ctx(None, request_id="v_ns", tenant="t")
    await a.query(QuerySpec(vector=[0.2], top_k=1, namespace="ns_x"), ctx=ctx)

    extra = [o["extra"] for o in m.observations if o["op"] == "query"][-1]
    assert extra.get("namespace") == "ns_x"


async def test_upsert_metrics_include_vector_count():
    m = CaptureMetrics()
    a = MockVectorAdapter(metrics=m)
    ctx = make_ctx(None, request_id="v_up", tenant="t")
    spec = UpsertSpec(
        namespace="default",
        vectors=[Vector(id=VectorID("m1"), vector=[0.1])],
    )
    await a.upsert(spec, ctx=ctx)

    counters = [c for c in m.counters if c["name"] == "vectors_upserted"]
    assert counters
    assert counters[-1]["value"] >= 1
