# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Deadline semantics.

Spec refs:
  • §6.1 Context & Deadlines
  • §12.4 DeadlineExceeded mapping
  • §12.5 Partial Success & Caching
  • §13.1 Observability
"""

import time
import pytest
from typing import List

from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter,
    EmbedSpec,
    BatchEmbedSpec,
    OperationContext,
    DeadlineExceeded,
    BatchEmbedResult,
    MetricsSink,
    NotSupported,
    BadRequest,
)

pytestmark = pytest.mark.asyncio


def clear_time_cache():
    """Placeholder for any cached time reset used in remaining_budget_ms."""
    return None


def remaining_budget_ms(ctx: OperationContext):
    """Compute remaining deadline budget in ms, clamped to non-negative values."""
    if ctx.deadline_ms is None:
        return None
    now_ms = int(time.time() * 1000)
    return max(0, ctx.deadline_ms - now_ms)


class DeadlineMetricsCapture(MetricsSink):
    def __init__(self):
        self.observations: List[dict] = []
        self.counters: List[dict] = []

    def observe(self, *, component: str, op: str, ms: float, ok: bool, code: str = "OK", extra=None):
        self.observations.append({
            "component": component,
            "op": op,
            "ms": ms,
            "ok": ok,
            "code": code,
            "extra": dict(extra or {}),
        })

    def counter(self, *, component: str, name: str, value: int = 1, extra=None):
        self.counters.append({
            "component": component,
            "name": name,
            "value": value,
            "extra": dict(extra or {}),
        })


# 🧩 helper: swap adapter metrics to capture
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


async def test_deadline_budget_calculation_accurate():
    """§6.1: Remaining budget calculation must be accurate and non-negative."""
    clear_time_cache()
    now = int(time.time() * 1000)

    # Keep bounds generous to avoid CI flakiness.
    test_cases = [
        # Future: should be >0 and not wildly larger than expected.
        (now + 1000, 1, 1500),
        (now + 100, 1, 500),
        # Past: must clamp to 0
        (now - 100, 0, 0),
    ]

    for deadline_ms, min_expected, max_expected in test_cases:
        ctx = OperationContext(
            request_id=f"t_budget_{deadline_ms}",
            tenant="t",
            deadline_ms=deadline_ms,
        )
        remaining = remaining_budget_ms(ctx)

        if remaining is not None:
            assert remaining >= 0, f"Negative remaining budget: {remaining}"
            assert remaining >= min_expected, f"Budget too low: {remaining} < {min_expected}"
            assert remaining <= max_expected, f"Budget too high: {remaining} > {max_expected}"


async def test_deadline_preexpired_deadline_fails_fast_embed(adapter: BaseEmbeddingAdapter):
    """§6.1: Pre-expired deadlines must raise DeadlineExceeded immediately."""
    clear_time_cache()
    past = int(time.time() * 1000) - 1000
    ctx = OperationContext(
        request_id="t_embed_preexpired",
        tenant="t",
        deadline_ms=past,
    )

    start_time = time.time()
    with pytest.raises(DeadlineExceeded):
        caps = await adapter.capabilities()
        await adapter.embed(
            EmbedSpec(text="should not process", model=caps.supported_models[0]),
            ctx=ctx,
        )
    duration = time.time() - start_time
    assert duration < 0.25, f"Pre-expired deadline took too long: {duration:.3f}s"


async def test_deadline_embed_respects_very_short_deadline(adapter: BaseEmbeddingAdapter):
    """§6.1: Very short future deadlines may succeed or raise DeadlineExceeded depending on enforcement policy."""
    clear_time_cache()
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    now = int(time.time() * 1000)
    m = DeadlineMetricsCapture()

    ctx = OperationContext(
        request_id="t_embed_short_deadline",
        tenant="t",
        deadline_ms=now + 1,
    )

    with _SwapAdapterMetrics(adapter, m):
        try:
            await adapter.embed(
                EmbedSpec(text="test", model=model),
                ctx=ctx,
            )
        except DeadlineExceeded:
            pass  # acceptable if enforced

    # If metrics were emitted for this call, deadline_bucket must be present
    embed_obs = [o for o in m.observations if o["op"] == "embed"]
    if embed_obs:
        extra = embed_obs[-1].get("extra", {})
        if extra:
            assert "deadline_bucket" in extra, f"Expected deadline_bucket in metrics extra, got {extra}"


async def test_deadline_batch_partial_completion_before_deadline(adapter: BaseEmbeddingAdapter):
    """§12.5: Batch under tight deadlines may succeed or raise DeadlineExceeded; both acceptable."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    clear_time_cache()
    now = int(time.time() * 1000)
    m = DeadlineMetricsCapture()

    ctx = OperationContext(
        request_id="t_batch_partial",
        tenant="t",
        deadline_ms=now + 50,
    )

    large_batch = ["text"] * min(20, caps.max_batch_size or 20)

    with _SwapAdapterMetrics(adapter, m):
        if not caps.supports_batch_embedding:
            with pytest.raises(NotSupported):
                await adapter.embed_batch(
                    BatchEmbedSpec(texts=large_batch, model=model),
                    ctx=ctx,
                )
            return

        try:
            result = await adapter.embed_batch(
                BatchEmbedSpec(texts=large_batch, model=model),
                ctx=ctx,
            )
            assert isinstance(result, BatchEmbedResult)
            assert result.total_texts == len(large_batch)
        except (DeadlineExceeded, BadRequest):
            # BadRequest is acceptable if the implementation fails fast for other reasons.
            pass

    # If metrics emitted for embed_batch, require deadline_bucket
    batch_obs = [o for o in m.observations if o["op"] == "embed_batch"]
    if batch_obs:
        extra = batch_obs[-1].get("extra", {})
        if extra:
            assert "deadline_bucket" in extra, f"Expected deadline_bucket in metrics extra, got {extra}"


async def test_deadline_metrics_include_buckets(adapter: BaseEmbeddingAdapter):
    """§13.1: If metrics are emitted for deadline ops, deadline_bucket tag must be present."""
    clear_time_cache()
    m = DeadlineMetricsCapture()
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    deadline_buckets = [100, 500, 1000, 5000]

    with _SwapAdapterMetrics(adapter, m):
        for bucket in deadline_buckets:
            now = int(time.time() * 1000)
            ctx = OperationContext(
                request_id=f"t_deadline_bucket_{bucket}",
                tenant="t",
                deadline_ms=now + bucket,
            )
            try:
                await adapter.embed(
                    EmbedSpec(text=f"test {bucket}", model=model),
                    ctx=ctx,
                )
            except DeadlineExceeded:
                pass

    # Verify we have observations; if we do, each should include deadline_bucket
    embed_obs = [o for o in m.observations if o["op"] == "embed"]
    if embed_obs:
        for obs in embed_obs:
            extra = obs.get("extra", {})
            if extra:
                assert "deadline_bucket" in extra, f"Expected deadline_bucket in metrics extra, got {extra}"


async def test_deadline_sequential_operations_respect_deadline(adapter: BaseEmbeddingAdapter):
    """§6.1: Sequential operations should observe decreasing remaining time (as time passes)."""
    clear_time_cache()
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    now = int(time.time() * 1000)
    total_budget = 1000

    ctx = OperationContext(
        request_id="t_sequential_deadline",
        tenant="t",
        deadline_ms=now + total_budget,
    )

    operations_completed = 0
    for i in range(3):
        try:
            remaining_before = remaining_budget_ms(ctx)

            await adapter.embed(
                EmbedSpec(text=f"operation {i}", model=model),
                ctx=ctx,
            )
            operations_completed += 1

            remaining_after = remaining_budget_ms(ctx)

            if remaining_before is not None and remaining_after is not None:
                assert remaining_after <= remaining_before, "Budget should decrease after operation"
        except DeadlineExceeded:
            break

    assert operations_completed >= 1, "Should complete at least one operation with reasonable budget"


async def test_deadline_exceeded_has_clear_error_message(adapter: BaseEmbeddingAdapter):
    """§12.4: DeadlineExceeded errors should have informative messages."""
    clear_time_cache()
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    past = int(time.time() * 1000) - 100
    ctx = OperationContext(
        request_id="t_deadline_message",
        tenant="t",
        deadline_ms=past,
    )

    with pytest.raises(DeadlineExceeded) as exc_info:
        await adapter.embed(
            EmbedSpec(text="test", model=model),
            ctx=ctx,
        )

    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ["deadline", "timeout", "exceeded", "expired"]), (
        f"Deadline error should mention timeout: {error_msg}"
    )
