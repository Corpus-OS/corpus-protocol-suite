# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — SIEM-safe metrics & context propagation.

Spec refs:
  • §13.1-13.3 Observability & Privacy
  • §6.1 Context propagation
"""

import pytest
from typing import Optional, Mapping, Any, List

from corpus_sdk.embedding.embedding_base import (
    EmbedSpec,
    BatchEmbedSpec,
    OperationContext,
    MetricsSink,
)
from corpus_sdk.examples.embedding.mock_embedding_adapter import MockEmbeddingAdapter
from corpus_sdk.examples.common.ctx import make_ctx

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
    a = MockEmbeddingAdapter(metrics=m, failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_siem_ctx", tenant="acme")
    await a.embed(EmbedSpec(text="hello", model=a.supported_models[0]), ctx=ctx)
    assert any(o["op"] == "embed" for o in m.observations)


async def test_tenant_hashed_never_raw():
    m = CaptureMetrics()
    tenant = "super-secret-tenant"
    a = MockEmbeddingAdapter(metrics=m, failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_siem_hash", tenant=tenant)
    await a.embed(EmbedSpec(text="hi", model=a.supported_models[0]), ctx=ctx)
    extras = [o["extra"] for o in m.observations if o["op"] == "embed"]
    assert extras
    last = extras[-1]
    # Embedding base uses 'tenant_hash' key
    assert "tenant_hash" in last
    assert tenant not in str(last)


async def test_no_text_or_vector_data_in_metrics():
    m = CaptureMetrics()
    a = MockEmbeddingAdapter(metrics=m, failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_siem_notext", tenant="t")
    await a.embed(EmbedSpec(text="do not leak this", model=a.supported_models[0]), ctx=ctx)
    extra = [o["extra"] for o in m.observations if o["op"] == "embed"][-1]
    # Ensure we don't leak raw inputs in metric dimensions
    for banned in ("text", "vector", "embedding"):
        assert banned not in extra


async def test_metrics_emitted_on_error_path():
    m = CaptureMetrics()

    class ErrorAdapter(MockEmbeddingAdapter):
        async def _do_embed(self, spec, *, ctx=None):
            raise ValueError("boom")

    a = ErrorAdapter(metrics=m, failure_rate=0.0)
    with pytest.raises(Exception):
        await a.embed(EmbedSpec(text="x", model=a.supported_models[0]))
    assert any(not o["ok"] for o in m.observations)

