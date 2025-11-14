# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Context propagation and SIEM-safe metrics.
Covers:
  • Tenant identifiers are hashed, never logged raw (§13.2, §15)
  • Metrics include operation metadata (component, op, code)
  • Request context flows through to observability layer
  • No PII or sensitive data appears in telemetry
"""
from typing import Optional, Mapping, Any, List

import pytest
from corpus_sdk.llm.llm_base import (
    OperationContext,
    MetricsSink,
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
        extra: Optional[Mapping[str, Any,]] = None,
    ) -> None:
        self.counters.append(
            {
                "component": component,
                "name": name,
                "value": value,
                "extra": dict(extra or {}),
            }
        )


async def test_observability_context_propagates_to_metrics_siem_safe(adapter):
    """
    SPECIFICATION.md §13.2, §15 — SIEM-Safe Observability

    Verify that:
    1. Raw tenant IDs NEVER appear in metrics output
    2. Tenant is hashed when present in telemetry
    3. Operation metadata (component, op, code) is emitted
    4. No prompt content or PII leaks into metrics
    """
    metrics = CaptureMetrics()
    original_metrics = getattr(adapter, "_metrics", None)
    adapter._metrics = metrics  # type: ignore[attr-defined]

    caps = await adapter.capabilities()
    secret_tenant = "acme-corp-secret-12345"
    ctx = make_ctx(
        OperationContext,
        tenant=secret_tenant,
        request_id="test-req-ctx-001",
        timeout_ms=30_000,
    )

    await adapter.complete(
        messages=[{"role": "user", "content": "sensitive prompt data"}],
        model=caps.supported_models[0],
        ctx=ctx,
    )

    # Restore original metrics
    if original_metrics is not None:
        adapter._metrics = original_metrics  # type: ignore[attr-defined]

    # At least one observation
    assert metrics.observations, "Expected at least one observation metric"
    obs = metrics.observations[0]

    # 1. Raw tenant ID MUST NOT appear anywhere
    serialized = str(metrics.observations) + str(metrics.counters)
    assert secret_tenant not in serialized, \
        f"PRIVACY VIOLATION: Raw tenant ID '{secret_tenant}' found in metrics"

    # 2. Prompt content MUST NOT appear in metrics
    assert "sensitive prompt data" not in serialized, \
        "PRIVACY VIOLATION: Prompt content leaked into metrics"

    # 3. Component should be 'llm'
    assert obs["component"] == "llm"

    # 4. Operation should be 'complete'
    assert obs["op"] == "complete"

    # 5. Status code should be 'OK' for successful operation
    assert obs["code"] == "OK"

    # 6. Operation should be marked successful
    assert obs["ok"] is True

    # 7. Latency should be recorded
    assert isinstance(obs["ms"], (int, float)) and obs["ms"] >= 0

    # 8. If tenant is included, it should be hashed
    extra = obs.get("extra") or {}
    if "tenant" in extra:
        tenant_value = extra["tenant"]
        assert isinstance(tenant_value, str)
        assert len(tenant_value) >= 8
        assert tenant_value != secret_tenant
        assert secret_tenant not in tenant_value


async def test_observability_metrics_emitted_on_error_path(adapter):
    """
    Verify that metrics are emitted even when operations fail.
    Error paths MUST NOT leak sensitive information either.
    """
    metrics = CaptureMetrics()
    original_metrics = getattr(adapter, "_metrics", None)
    adapter._metrics = metrics  # type: ignore[attr-defined]

    secret_tenant = "error-tenant-secret"
    ctx = make_ctx(
        OperationContext,
        tenant=secret_tenant,
        request_id="test-err-001",
    )

    # Force an error via obviously invalid model name
    with pytest.raises(Exception):
        await adapter.complete(
            messages=[{"role": "user", "content": "trigger error"}],
            model="__no_such_model__",
            ctx=ctx,
        )

    if original_metrics is not None:
        adapter._metrics = original_metrics  # type: ignore[attr-defined]

    serialized = str(metrics.observations) + str(metrics.counters)

    # No raw tenant in error metrics
    assert secret_tenant not in serialized, \
        "Raw tenant ID leaked in error metrics"

    # Expect at least one observation on error path
    assert metrics.observations, "Expected observation even on error"
    obs = metrics.observations[-1]

    # Error should be marked
    assert obs["ok"] is False, "Error operations should have ok=false"
    assert obs["code"] != "OK", "Error should have non-OK code"


async def test_observability_streaming_metrics_siem_safe(adapter):
    """
    Verify streaming operations also maintain SIEM-safe metrics.
    """
    caps = await adapter.capabilities()
    if not caps.supports_streaming:
        pytest.skip("Adapter does not support streaming")

    metrics = CaptureMetrics()
    original_metrics = getattr(adapter, "_metrics", None)
    adapter._metrics = metrics  # type: ignore[attr-defined]

    secret_tenant = "stream-tenant-secret"
    ctx = make_ctx(
        OperationContext,
        tenant=secret_tenant,
        request_id="test-stream-001",
    )

    chunks = []
    async for chunk in adapter.stream(
        messages=[{"role": "user", "content": "stream sensitive data"}],
        model=caps.supported_models[0],
        ctx=ctx,
    ):
        chunks.append(chunk)

    if original_metrics is not None:
        adapter._metrics = original_metrics  # type: ignore[attr-defined]

    serialized = str(metrics.observations) + str(metrics.counters)

    # Privacy checks for streaming
    assert secret_tenant not in serialized, \
        "Raw tenant ID leaked in streaming metrics"
    assert "stream sensitive data" not in serialized, \
        "Prompt content leaked in streaming metrics"

    # Expect a stream observation
    stream_obs = [o for o in metrics.observations if o["op"] == "stream"]
    assert stream_obs, "Expected stream observation"
    last = stream_obs[-1]
    assert last["component"] == "llm"
    assert last["ok"] is True


async def test_observability_token_counter_metrics_present(adapter):
    """
    Verify counter metrics are emitted for token usage.
    """
    caps = await adapter.capabilities()
    if not caps.supports_count_tokens:
        pytest.skip("Adapter does not support token counting")

    metrics = CaptureMetrics()
    original_metrics = getattr(adapter, "_metrics", None)
    adapter._metrics = metrics  # type: ignore[attr-defined]

    ctx = make_ctx(
        OperationContext,
        tenant="counter-test",
        request_id="test-ctr-001",
    )

    await adapter.complete(
        messages=[{"role": "user", "content": "test"}],
        model=caps.supported_models[0],
        ctx=ctx,
    )

    if original_metrics is not None:
        adapter._metrics = original_metrics  # type: ignore[attr-defined]

    # Expect counter metrics
    assert metrics.counters, "Expected counter metrics"

    # Expect tokens_processed counter
    token_counters = [
        c for c in metrics.counters
        if "tokens_processed" in c["name"]
    ]
    assert token_counters, "Expected tokens_processed counter"