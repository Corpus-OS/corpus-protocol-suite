# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — SIEM-safe metrics & context propagation.

Spec refs:
  • §13.1-13.3 Observability & Privacy
  • §6.1 Context propagation
  • §12.5 Partial Success & Caching

"""

import time
import pytest
from typing import Optional, Mapping, Any, List

from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter,
    EmbedSpec,
    BatchEmbedSpec,
    OperationContext,
    MetricsSink,
    NotSupported,
    BadRequest,
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
            {
                "component": component,
                "op": op,
                "ok": ok,
                "code": code,
                "extra": dict(extra or {}),
                "ms": ms,
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


class _SwapAdapterMetrics:
    def __init__(self, adapter: BaseEmbeddingAdapter, sink: MetricsSink):
        self.adapter = adapter
        self.sink = sink
        self._old = None

    def __enter__(self):
        self._old = getattr(self.adapter, "_metrics", None)
        setattr(self.adapter, "_metrics", self.sink)
        return self.sink

    def __exit__(self, exc_type, exc, tb):
        if self._old is not None:
            setattr(self.adapter, "_metrics", self._old)


async def test_observability_context_propagates_to_metrics(adapter: BaseEmbeddingAdapter):
    """§13.1: Metrics must include correct component and operation."""
    m = CaptureMetrics()
    caps = await adapter.capabilities()
    model = caps.supported_models[0]
    ctx = OperationContext(request_id="t_siem_ctx", tenant="acme")

    with _SwapAdapterMetrics(adapter, m):
        await adapter.embed(EmbedSpec(text="hello", model=model), ctx=ctx)

    embed_obs = [o for o in m.observations if o["op"] == "embed"]
    assert embed_obs, "Expected observe() call for embed()"
    last = embed_obs[-1]
    assert last["component"] == "embedding"
    assert last["ok"] is True
    assert last["code"] == "OK"


async def test_observability_tenant_hashed_never_raw(adapter: BaseEmbeddingAdapter):
    """§13.2: Tenant identifiers must be hashed in all metrics."""
    m = CaptureMetrics()
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    tenant = "super-secret-tenant"
    ctx = OperationContext(request_id="t_siem_hash", tenant=tenant)

    with _SwapAdapterMetrics(adapter, m):
        await adapter.embed(EmbedSpec(text="hi", model=model), ctx=ctx)

    for observation in m.observations:
        extra = observation["extra"]
        serialized = str(extra).lower()
        assert tenant.lower() not in serialized, f"Raw tenant leaked in {observation}"

        if "tenant_hash" in extra:
            value = str(extra["tenant_hash"])
            assert tenant not in value
            assert len(value) >= 8


async def test_observability_no_sensitive_data_in_metrics(adapter: BaseEmbeddingAdapter):
    """§13.3: No raw request payloads (raw texts) in metric tags."""
    m = CaptureMetrics()
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    sensitive_texts = ["secret password", "private key 123", "confidential"]
    ctx = OperationContext(request_id="t_siem_sensitive", tenant="t")

    with _SwapAdapterMetrics(adapter, m):
        await adapter.embed(EmbedSpec(text=sensitive_texts[0], model=model), ctx=ctx)

        if caps.supports_batch_embedding:
            try:
                await adapter.embed_batch(
                    BatchEmbedSpec(texts=sensitive_texts, model=model),
                    ctx=ctx,
                )
            except Exception:
                pass
        else:
            with pytest.raises(NotSupported):
                await adapter.embed_batch(
                    BatchEmbedSpec(texts=sensitive_texts, model=model),
                    ctx=ctx,
                )

    for observation in m.observations:
        extra_str = str(observation["extra"]).lower()
        for s in sensitive_texts:
            assert s.lower() not in extra_str, f"Sensitive data '{s}' leaked in metrics extra"


async def test_observability_metrics_emitted_on_error_path(adapter: BaseEmbeddingAdapter):
    """§13.1: Error paths that occur inside the gated path must emit metrics + errors_total."""
    m = CaptureMetrics()
    ctx = OperationContext(request_id="t_siem_error", tenant="t")

    with _SwapAdapterMetrics(adapter, m):
        with pytest.raises(Exception):
            await adapter.embed(EmbedSpec(text="test", model="invalid-model-123"), ctx=ctx)

    failed_obs = [o for o in m.observations if o["op"] == "embed" and not o["ok"]]
    assert failed_obs, "Expected failed observation for gated error case"

    error_counters = [c for c in m.counters if c["name"] == "errors_total"]
    assert error_counters, "Expected errors_total counter for failed operation"


async def test_observability_batch_metrics_include_accurate_counts(adapter: BaseEmbeddingAdapter):
    """§12.5: Batch metrics must include batch_size; invalid items may fail fast or yield failures."""
    caps = await adapter.capabilities()
    m = CaptureMetrics()

    ctx = OperationContext(request_id="t_batch_metrics", tenant="t")
    texts = ["valid1", "", "valid2", "another valid"]
    model = caps.supported_models[0]

    with _SwapAdapterMetrics(adapter, m):
        if not caps.supports_batch_embedding:
            with pytest.raises(NotSupported):
                await adapter.embed_batch(BatchEmbedSpec(texts=texts, model=model), ctx=ctx)
        else:
            try:
                result = await adapter.embed_batch(BatchEmbedSpec(texts=texts, model=model), ctx=ctx)
            except BadRequest:
                pass
            else:
                assert hasattr(result, "failed_texts")
                assert len(result.failed_texts) >= 1, (
                    "If batch returns normally with invalid items present, failed_texts must be populated."
                )
                for f in result.failed_texts:
                    assert "index" in f and "error" in f and "message" in f, f"Failure record missing fields: {f}"

    batch_obs = [o for o in m.observations if o["op"] == "embed_batch"]
    if batch_obs:
        extra = batch_obs[-1]["extra"]
        assert extra.get("batch_size") == len(texts), f"Expected batch_size={len(texts)} in metrics extra, got {extra}"


async def test_observability_deadline_metrics_include_bucket_tags(adapter: BaseEmbeddingAdapter):
    """§6.1: If metrics are emitted for a ctx with deadline_ms, deadline_bucket must be present."""
    m = CaptureMetrics()
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    now = int(time.time() * 1000)
    deadlines = [now + 5000, now + 500, now + 50]

    with _SwapAdapterMetrics(adapter, m):
        for deadline in deadlines:
            ctx = OperationContext(
                request_id=f"t_deadline_{deadline}",
                tenant="t",
                deadline_ms=deadline,
            )
            try:
                await adapter.embed(EmbedSpec(text="test", model=model), ctx=ctx)
            except Exception:
                pass

    embed_obs = [o for o in m.observations if o["op"] == "embed"]
    for obs in embed_obs:
        extra = obs.get("extra", {})
        if extra:
            assert "deadline_bucket" in extra, f"Expected deadline_bucket tag in metrics extra, got {extra}"


async def test_observability_metrics_include_operation_specific_tags(adapter: BaseEmbeddingAdapter):
    """§13.1: If model tags are present, they must be well-formed."""
    m = CaptureMetrics()
    caps = await adapter.capabilities()
    model = caps.supported_models[0]
    ctx = OperationContext(request_id="t_operation_tags", tenant="t")

    with _SwapAdapterMetrics(adapter, m):
        await adapter.embed(EmbedSpec(text="test", model=model, normalize=True), ctx=ctx)

    embed_obs = [o for o in m.observations if o["op"] == "embed"]
    assert embed_obs
    extra = embed_obs[-1]["extra"]

    for key in [k for k in extra.keys() if "model" in k.lower()]:
        value = extra[key]
        assert value not in (None, "", "unknown"), f"Unexpected model tag value {value!r} in {extra}"


async def test_observability_errors_total_counter_incremented_on_failure(adapter: BaseEmbeddingAdapter):
    """
    errors_total should increment for failures that occur inside the gated path.
    Use an unknown (but non-empty) model to ensure the error happens after gates.
    """
    m = CaptureMetrics()
    ctx = OperationContext(request_id="t_err_counter", tenant="t")

    with _SwapAdapterMetrics(adapter, m):
        with pytest.raises(Exception):
            await adapter.embed(EmbedSpec(text="x", model="invalid-model-123"), ctx=ctx)

    errors_total = [c for c in m.counters if c["name"] == "errors_total"]
    assert errors_total, "Expected errors_total counter increment(s) on gated failure"
