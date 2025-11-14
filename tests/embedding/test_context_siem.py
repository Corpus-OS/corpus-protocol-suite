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
import hashlib
from typing import Optional, Mapping, Any, List

from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter,
    EmbedSpec,
    BatchEmbedSpec,
    OperationContext,
    MetricsSink,
    NotSupported,
)
from examples.common.ctx import make_ctx

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


async def test_observability_context_propagates_to_metrics(adapter: BaseEmbeddingAdapter):
    """§13.1: Metrics must include correct component and operation."""
    m = CaptureMetrics()
    ctx = make_ctx(OperationContext, request_id="t_siem_ctx", tenant="acme", metrics=m)

    await adapter.embed(
        EmbedSpec(text="hello", model=adapter.supported_models[0]),
        ctx=ctx,
    )

    embed_obs = [o for o in m.observations if o["op"] == "embed"]
    assert embed_obs, "Expected observe() call for embed()"
    last = embed_obs[-1]
    assert last["component"] == "embedding"
    assert last["ok"] is True
    assert last["code"] == "OK"


async def test_observability_tenant_hashed_never_raw(adapter: BaseEmbeddingAdapter):
    """§13.2: Tenant identifiers must be hashed in all metrics."""
    m = CaptureMetrics()
    tenant = "super-secret-tenant"
    ctx = make_ctx(
        OperationContext,
        request_id="t_siem_hash",
        tenant=tenant,
        metrics=m,
    )

    await adapter.embed(
        EmbedSpec(text="hi", model=adapter.supported_models[0]),
        ctx=ctx,
    )

    # Check all observations for tenant leakage
    for observation in m.observations:
        extra = observation["extra"]
        serialized = str(extra).lower()
        
        # Raw tenant must never appear
        assert tenant.lower() not in serialized, f"Raw tenant leaked in {observation}"
        
        # Tenant-related keys should contain hashes, not raw values
        for key in extra:
            if "tenant" in key.lower():
                value = str(extra[key])
                assert tenant not in value, f"Raw tenant in {key}: {value}"
                assert len(value) >= 8, f"Tenant identifier too short: {value}"


async def test_observability_no_sensitive_data_in_metrics(adapter: BaseEmbeddingAdapter):
    """§13.3: No raw text, vectors, or embeddings in metrics."""
    m = CaptureMetrics()
    sensitive_texts = ["secret password", "private key 123", "confidential"]
    ctx = make_ctx(
        OperationContext,
        request_id="t_siem_sensitive",
        tenant="t",
        metrics=m,
    )

    # Test both single and batch operations
    await adapter.embed(
        EmbedSpec(text=sensitive_texts[0], model=adapter.supported_models[0]),
        ctx=ctx,
    )

    try:
        await adapter.embed_batch(
            BatchEmbedSpec(texts=sensitive_texts, model=adapter.supported_models[0]),
            ctx=ctx,
        )
    except NotSupported:
        pass  # Batch not supported is OK for this test

    # Verify no sensitive data in any metrics
    banned_patterns = sensitive_texts + ["text", "texts", "vector", "embedding", "embeddings"]
    
    for observation in m.observations:
        extra_str = str(observation["extra"]).lower()
        for pattern in banned_patterns:
            assert pattern.lower() not in extra_str, f"Sensitive data '{pattern}' leaked in metrics"


async def test_observability_metrics_emitted_on_error_path(adapter: BaseEmbeddingAdapter):
    """§13.1: Error paths must emit metrics with appropriate error codes."""
    m = CaptureMetrics()
    ctx = make_ctx(
        OperationContext,
        request_id="t_siem_error",
        tenant="t",
        metrics=m,
    )

    # Test multiple error scenarios
    error_cases = [
        ("invalid-model-123", "ModelNotAvailable"),
        ("", "BadRequest"),  # Empty model
    ]

    for model, expected_error_type in error_cases:
        try:
            await adapter.embed(
                EmbedSpec(text="test", model=model),
                ctx=ctx,
            )
        except Exception:
            pass  # Expected to fail

    # Verify error observations
    failed_obs = [o for o in m.observations if not o["ok"]]
    assert failed_obs, "Expected failed observations for error cases"
    
    # Verify error counters
    error_counters = [c for c in m.counters if "error" in c["name"].lower()]
    assert error_counters, "Expected error counters for failed operations"


async def test_observability_batch_metrics_include_accurate_counts(adapter: BaseEmbeddingAdapter):
    """§12.5: Batch metrics must accurately reflect success/failure counts."""
    caps = await adapter.capabilities()
    if not getattr(caps, "supports_batch_embedding", False):
        pytest.skip("Batch embedding not supported")

    m = CaptureMetrics()
    ctx = make_ctx(
        OperationContext,
        request_id="t_batch_metrics",
        tenant="t",
        metrics=m,
    )

    # Mix of valid and potentially problematic texts
    texts = ["valid1", "", "valid2", "another valid"]
    result = await adapter.embed_batch(
        BatchEmbedSpec(texts=texts, model=adapter.supported_models[0]),
        ctx=ctx,
    )

    batch_obs = [o for o in m.observations if o["op"] == "embed_batch"]
    assert batch_obs, "Expected observe() for embed_batch"
    
    last_obs = batch_obs[-1]
    extra = last_obs["extra"]

    # Verify batch_size matches input
    assert "batch_size" in extra or "size" in extra or "n_items" in extra
    batch_size_key = next((k for k in extra.keys() if "batch" in k or "size" in k or "n_items" in k), None)
    assert extra[batch_size_key] == len(texts)

    # Verify success/failure counts match actual results
    success_count = len(result.embeddings)
    failure_count = len(result.failed_texts)
    
    # Check if metrics include success/failure breakdown
    if "success_count" in extra:
        assert extra["success_count"] == success_count
    if "failure_count" in extra:
        assert extra["failure_count"] == failure_count

    # Operation should be marked OK if any successes, or False if all failed
    expected_ok = success_count > 0
    assert last_obs["ok"] == expected_ok, f"Expected ok={expected_ok} for {success_count} successes, {failure_count} failures"


async def test_observability_deadline_metrics_include_bucket_tags(adapter: BaseEmbeddingAdapter):
    """§6.1: Deadline contexts must include deadline_bucket in metrics."""
    m = CaptureMetrics()

    now = int(time.time() * 1000)
    deadlines = [
        now + 5000,   # 5s bucket
        now + 500,    # 500ms bucket  
        now + 50,     # 50ms bucket
    ]

    for deadline in deadlines:
        ctx = make_ctx(
            OperationContext,
            request_id=f"t_deadline_{deadline}",
            tenant="t",
            deadline_ms=deadline,
            metrics=m,
        )

        try:
            await adapter.embed(
                EmbedSpec(text="test", model=adapter.supported_models[0]),
                ctx=ctx,
            )
        except Exception:
            pass  # Deadline might be exceeded

    # Verify deadline-related tags in observations
    deadline_observations = []
    for obs in m.observations:
        if "deadline" in str(obs.get("extra", {})).lower():
            deadline_observations.append(obs)

    assert deadline_observations, "Expected deadline-related tags in metrics"
    
    for obs in deadline_observations:
        extra = obs["extra"]
        deadline_keys = [k for k in extra.keys() if "deadline" in k.lower()]
        assert deadline_keys, f"Expected deadline bucket key, got {extra}"


async def test_observability_metrics_include_operation_specific_tags(adapter: BaseEmbeddingAdapter):
    """§13.1: Metrics should include operation-specific context tags."""
    m = CaptureMetrics()
    ctx = make_ctx(
        OperationContext,
        request_id="t_operation_tags",
        tenant="t",
        metrics=m,
    )

    model = adapter.supported_models[0]
    
    # Test different operations
    await adapter.embed(
        EmbedSpec(text="test", model=model, normalize=True),
        ctx=ctx,
    )

    # Check for operation-specific tags
    embed_obs = [o for o in m.observations if o["op"] == "embed"]
    assert embed_obs
    
    last_embed = embed_obs[-1]
    extra = last_embed["extra"]
    
    # Should include model information
    assert any("model" in k.lower() for k in extra.keys()), f"Expected model tag in {extra}"
    
    # Should include normalization context if applied
    if any("normalize" in k.lower() for k in extra.keys()):
        norm_key = next(k for k in extra.keys() if "normalize" in k.lower())
        assert extra[norm_key] in [True, False, "true", "false"]