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
)
from examples.common.ctx import make_ctx, remaining_budget_ms, clear_time_cache

pytestmark = pytest.mark.asyncio


class DeadlineMetricsCapture(MetricsSink):
    def __init__(self):
        self.observations: List[dict] = []
        self.counters: List[dict] = []
    
    def observe(self, *, component: str, op: str, ms: float, ok: bool, code: str = "OK", extra=None):
        self.observations.append({
            "component": component, "op": op, "ms": ms, "ok": ok, "code": code, "extra": extra or {}
        })
    
    def counter(self, *, component: str, name: str, value: int = 1, extra=None):
        self.counters.append({
            "component": component, "name": name, "value": value, "extra": extra or {}
        })


async def test_deadline_budget_calculation_accurate():
    """§6.1: Remaining budget calculation must be accurate and non-negative."""
    clear_time_cache()
    now = int(time.time() * 1000)
    
    test_cases = [
        (now + 1000, 500, 500),   # 1s deadline, 0ms elapsed → 500ms remaining
        (now + 100, None, None),  # Very short deadline might be None
        (now - 100, 0, 0),        # Expired deadline → 0 remaining
    ]
    
    for deadline_ms, min_expected, max_expected in test_cases:
        ctx = make_ctx(
            OperationContext,
            request_id=f"t_budget_{deadline_ms}",
            tenant="t",
            deadline_ms=deadline_ms,
        )
        remaining = remaining_budget_ms(ctx)
        
        if remaining is not None:
            assert remaining >= 0, f"Negative remaining budget: {remaining}"
            if min_expected is not None:
                assert remaining >= min_expected - 50, f"Budget too low: {remaining} < {min_expected}"
            if max_expected is not None:
                assert remaining <= max_expected + 50, f"Budget too high: {remaining} > {max_expected}"


async def test_deadline_preexpired_deadline_fails_fast_embed(adapter: BaseEmbeddingAdapter):
    """§6.1: Pre-expired deadlines must raise DeadlineExceeded immediately."""
    clear_time_cache()
    past = int(time.time() * 1000) - 1000  # 1 second in past
    m = DeadlineMetricsCapture()
    
    ctx = make_ctx(
        OperationContext,
        request_id="t_embed_preexpired",
        tenant="t",
        deadline_ms=past,
        metrics=m,
    )
    
    start_time = time.time()
    with pytest.raises(DeadlineExceeded):
        await adapter.embed(
            EmbedSpec(text="should not process", model=adapter.supported_models[0]),
            ctx=ctx,
        )
    duration = time.time() - start_time
    
    # Should fail very quickly (under 100ms for pre-expired)
    assert duration < 0.1, f"Pre-expired deadline took too long: {duration:.3f}s"
    
    # Should have metrics for the failed operation
    deadline_obs = [o for o in m.observations if o["op"] == "embed" and not o["ok"]]
    assert deadline_obs, "Expected failed observation for expired deadline"


async def test_deadline_embed_respects_very_short_deadline(adapter: BaseEmbeddingAdapter):
    """§6.1: Very short deadlines should cause DeadlineExceeded."""
    clear_time_cache()
    now = int(time.time() * 1000)
    m = DeadlineMetricsCapture()
    
    ctx = make_ctx(
        OperationContext,
        request_id="t_embed_short_deadline",
        tenant="t",
        deadline_ms=now + 1,  # 1ms deadline - should be impossible
        metrics=m,
    )
    
    try:
        await adapter.embed(
            EmbedSpec(text="test", model=adapter.supported_models[0]),
            ctx=ctx,
        )
        # Some adapters might complete very quickly, which is acceptable
    except DeadlineExceeded:
        pass  # Expected for most implementations
    
    # Verify deadline-related metrics
    deadline_metrics = [o for o in m.observations if any("deadline" in str(k) for k in o.get("extra", {}))]
    assert deadline_metrics, "Expected deadline-related metrics for short deadline operation"


async def test_deadline_batch_partial_completion_before_deadline(adapter: BaseEmbeddingAdapter):
    """§12.5: Batch operations may complete partially before deadline."""
    caps = await adapter.capabilities()
    if not getattr(caps, "supports_batch_embedding", False):
        pytest.skip("Batch embedding not supported")
    
    clear_time_cache()
    now = int(time.time() * 1000)
    m = DeadlineMetricsCapture()
    
    # Create a deadline that might allow partial processing
    ctx = make_ctx(
        OperationContext,
        request_id="t_batch_partial",
        tenant="t",
        deadline_ms=now + 50,  # Very short but might allow some work
        metrics=m,
    )
    
    # Larger batch that might not complete fully
    large_batch = ["text"] * min(20, caps.max_batch_size or 20)
    
    try:
        result = await adapter.embed_batch(
            BatchEmbedSpec(texts=large_batch, model=adapter.supported_models[0]),
            ctx=ctx,
        )
        
        # If we get here, verify it's a valid partial result
        assert isinstance(result, BatchEmbedResult)
        total_processed = len(result.embeddings) + len(result.failed_texts)
        assert total_processed <= len(large_batch)
        
        # Should have metrics indicating partial processing
        batch_obs = [o for o in m.observations if o["op"] == "embed_batch"]
        if batch_obs:
            extra = batch_obs[-1].get("extra", {})
            if "success_count" in extra and "failure_count" in extra:
                assert extra["success_count"] + extra["failure_count"] == total_processed
        
    except DeadlineExceeded:
        # Also acceptable - adapter chose to fail fast rather than partial complete
        deadline_obs = [o for o in m.observations if not o["ok"] and "deadline" in str(o.get("extra", {}))]
        assert deadline_obs, "Expected deadline-related observation"


async def test_deadline_metrics_include_buckets(adapter: BaseEmbeddingAdapter):
    """§13.1: Deadline operations must include appropriate bucket tags in metrics."""
    clear_time_cache()
    m = DeadlineMetricsCapture()
    
    deadline_buckets = [100, 500, 1000, 5000]  # Various deadline ranges
    
    for bucket in deadline_buckets:
        now = int(time.time() * 1000)
        ctx = make_ctx(
            OperationContext,
            request_id=f"t_deadline_bucket_{bucket}",
            tenant="t",
            deadline_ms=now + bucket,
            metrics=m,
        )
        
        try:
            await adapter.embed(
                EmbedSpec(text=f"test {bucket}", model=adapter.supported_models[0]),
                ctx=ctx,
            )
        except DeadlineExceeded:
            pass  # Expected for very short deadlines
    
    # Verify we have deadline bucket information in metrics
    deadline_observations = []
    for obs in m.observations:
        extra = obs.get("extra", {})
        if any("deadline" in str(k).lower() for k in extra.keys()):
            deadline_observations.append(obs)
    
    assert deadline_observations, "Expected deadline bucket tags in metrics"
    
    # Verify bucket values are reasonable
    for obs in deadline_observations:
        extra = obs["extra"]
        for key, value in extra.items():
            if "deadline" in str(key).lower() and "bucket" in str(key).lower():
                assert isinstance(value, (int, str)), f"Deadline bucket should be int or string: {value}"
                if isinstance(value, int):
                    assert value >= 0, f"Negative deadline bucket: {value}"


async def test_deadline_sequential_operations_respect_deadline(adapter: BaseEmbeddingAdapter):
    """§6.1: Sequential operations should consume from shared deadline."""
    clear_time_cache()
    now = int(time.time() * 1000)
    total_budget = 1000  # 1 second total
    m = DeadlineMetricsCapture()
    
    ctx = make_ctx(
        OperationContext,
        request_id="t_sequential_deadline",
        tenant="t",
        deadline_ms=now + total_budget,
        metrics=m,
    )
    
    # Perform multiple operations
    operations_completed = 0
    for i in range(3):
        try:
            remaining_before = remaining_budget_ms(ctx)
            
            await adapter.embed(
                EmbedSpec(text=f"operation {i}", model=adapter.supported_models[0]),
                ctx=ctx,
            )
            operations_completed += 1
            
            remaining_after = remaining_budget_ms(ctx)
            
            # Budget should decrease (unless it's None/not tracked)
            if remaining_before is not None and remaining_after is not None:
                assert remaining_after <= remaining_before, "Budget should decrease after operation"
                
        except DeadlineExceeded:
            break  # Expected if we run out of budget
    
    # Should have completed at least one operation
    assert operations_completed >= 1, "Should complete at least one operation with reasonable budget"
    
    # Should have metrics for completed operations
    completed_obs = [o for o in m.observations if o["ok"]]
    assert len(completed_obs) >= operations_completed


async def test_deadline_exceeded_has_clear_error_message(adapter: BaseEmbeddingAdapter):
    """§12.4: DeadlineExceeded errors should have informative messages."""
    clear_time_cache()
    past = int(time.time() * 1000) - 100
    
    ctx = make_ctx(
        OperationContext,
        request_id="t_deadline_message",
        tenant="t", 
        deadline_ms=past,
    )
    
    with pytest.raises(DeadlineExceeded) as exc_info:
        await adapter.embed(
            EmbedSpec(text="test", model=adapter.supported_models[0]),
            ctx=ctx,
        )
    
    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ['deadline', 'timeout', 'exceeded', 'expired']), \
        f"Deadline error should mention timeout: {error_msg}"