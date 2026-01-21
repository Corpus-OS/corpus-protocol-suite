# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Context propagation and SIEM-safe metrics.
Covers:
  • Tenant identifiers are hashed, never logged raw (§13.2, §15)
  • Metrics include operation metadata (component, op, code)
  • Request context flows through to observability layer
  • No PII or sensitive data appears in telemetry
  • Error paths maintain privacy guarantees
"""
from typing import Optional, Mapping, Any, List

import pytest
from corpus_sdk.llm.llm_base import (
    OperationContext,
    MetricsSink,
)

pytestmark = pytest.mark.asyncio


class CaptureMetrics(MetricsSink):
    """Test metrics sink that captures all observations and counters."""

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
            {"component": component, "op": op, "ok": ok, "code": code, "extra": dict(extra or {}), "ms": ms}
        )

    def counter(
        self,
        *,
        component: str,
        name: str,
        value: int = 1,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.counters.append({"component": component, "name": name, "value": value, "extra": dict(extra or {})})


@pytest.fixture
def metrics_capture(adapter):
    """Fixture to safely capture metrics for testing."""
    if not hasattr(adapter, "_metrics"):
        pytest.fail("Adapter does not expose metrics sink required for SIEM-safe observability conformance")

    original_metrics = getattr(adapter, "_metrics", None)
    capture = CaptureMetrics()
    adapter._metrics = capture  # type: ignore[attr-defined]
    yield capture
    if original_metrics is not None:
        adapter._metrics = original_metrics  # type: ignore[attr-defined]


def assert_no_sensitive_data_leakage(metrics: CaptureMetrics, sensitive_strings: List[str]):
    serialized = str(metrics.observations) + str(metrics.counters)
    for sensitive in sensitive_strings:
        assert sensitive not in serialized, f"PRIVACY VIOLATION: Sensitive string '{sensitive}' found in metrics"


async def test_observability_context_propagates_to_metrics_siem_safe(adapter, metrics_capture):
    caps = await adapter.capabilities()
    secret_tenant = "acme-corp-secret-12345"
    sensitive_prompt = "sensitive prompt data with PII: user@example.com"

    ctx = OperationContext(tenant=secret_tenant, request_id="test-req-ctx-001")

    await adapter.complete(
        messages=[{"role": "user", "content": sensitive_prompt}],
        model=caps.supported_models[0],
        ctx=ctx,
    )

    assert metrics_capture.observations, "Expected at least one observation metric"
    assert_no_sensitive_data_leakage(metrics_capture, [secret_tenant, sensitive_prompt, "user@example.com"])

    complete_obs = next((o for o in metrics_capture.observations if o.get("op") == "complete"), metrics_capture.observations[-1])

    assert complete_obs["component"] == "llm"
    assert isinstance(complete_obs["op"], str) and complete_obs["op"]
    assert complete_obs["code"] == "OK"
    assert complete_obs["ok"] is True
    assert isinstance(complete_obs["ms"], (int, float)) and complete_obs["ms"] >= 0

    extra = complete_obs.get("extra") or {}
    assert "tenant_hash" in extra, "Tenant hash should be present in metrics extra when ctx.tenant is set"
    tenant_hash = extra["tenant_hash"]
    assert isinstance(tenant_hash, str) and len(tenant_hash) >= 8
    assert tenant_hash != secret_tenant


async def test_observability_metrics_emitted_on_error_path(adapter, metrics_capture):
    secret_tenant = "error-tenant-secret-789"
    sensitive_content = "trigger error with sensitive data: 555-1234"
    ctx = OperationContext(tenant=secret_tenant, request_id="test-err-001")

    with pytest.raises(Exception):
        await adapter.complete(
            messages=[{"role": "user", "content": sensitive_content}],
            model="__no_such_model_should_always_fail_123__",
            ctx=ctx,
        )

    assert metrics_capture.observations, "Expected observation even on error path"
    assert_no_sensitive_data_leakage(metrics_capture, [secret_tenant, sensitive_content, "555-1234"])

    error_obs = next((o for o in metrics_capture.observations if o.get("ok") is False), None)
    assert error_obs is not None, "Expected at least one ok=False observation"
    assert error_obs["code"] != "OK"


async def test_observability_streaming_metrics_siem_safe(adapter, metrics_capture):
    caps = await adapter.capabilities()
    secret_tenant = "stream-tenant-secret-xyz"
    sensitive_data = "stream sensitive data: credit card 4111-1111-1111-1111"
    ctx = OperationContext(tenant=secret_tenant, request_id="test-stream-001")

    if caps.supports_streaming:
        async for _ in adapter.stream(
            messages=[{"role": "user", "content": sensitive_data}],
            model=caps.supported_models[0],
            ctx=ctx,
        ):
            pass
    else:
        with pytest.raises(Exception):
            agen = adapter.stream(
                messages=[{"role": "user", "content": sensitive_data}],
                model=caps.supported_models[0],
                ctx=ctx,
            )
            async for _ in agen:
                pass

    assert_no_sensitive_data_leakage(metrics_capture, [secret_tenant, sensitive_data, "4111-1111-1111-1111"])
    assert metrics_capture.observations, "Expected observations to be recorded"


async def test_observability_token_counter_metrics_present(adapter, metrics_capture):
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="counter-test-tenant", request_id="test-ctr-001")

    if caps.supports_count_tokens:
        await adapter.count_tokens("test token counting metrics", model=caps.supported_models[0], ctx=ctx)
    else:
        with pytest.raises(Exception):
            await adapter.count_tokens("test token counting metrics", model=caps.supported_models[0], ctx=ctx)

    assert_no_sensitive_data_leakage(metrics_capture, ["counter-test-tenant", "test token counting metrics"])

    if metrics_capture.counters:
        for c in metrics_capture.counters:
            assert "component" in c and "name" in c and "value" in c


async def test_observability_metrics_structure_consistency(adapter, metrics_capture):
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="structure-test", request_id="test-struct-001")

    await adapter.complete(
        messages=[{"role": "user", "content": "test structure"}],
        model=caps.supported_models[0],
        ctx=ctx,
    )

    if caps.supports_count_tokens:
        await adapter.count_tokens("test tokens", model=caps.supported_models[0], ctx=ctx)

    assert metrics_capture.observations, "Expected observations to be recorded"
    for obs in metrics_capture.observations:
        assert obs["component"] == "llm"
        assert isinstance(obs["op"], str)
        assert isinstance(obs["ok"], bool)
        assert isinstance(obs["code"], str)
        assert isinstance(obs["ms"], (int, float)) and obs["ms"] >= 0


async def test_observability_no_metric_leakage_between_tenants(adapter, metrics_capture):
    caps = await adapter.capabilities()

    tenant_a = "tenant-alpha-secret"
    ctx_a = OperationContext(tenant=tenant_a, request_id="req-tenant-a")
    await adapter.complete(messages=[{"role": "user", "content": "request from tenant A"}], model=caps.supported_models[0], ctx=ctx_a)

    tenant_b = "tenant-beta-secret"
    ctx_b = OperationContext(tenant=tenant_b, request_id="req-tenant-b")
    await adapter.complete(messages=[{"role": "user", "content": "request from tenant B"}], model=caps.supported_models[0], ctx=ctx_b)

    serialized = str(metrics_capture.observations) + str(metrics_capture.counters)
    assert tenant_a not in serialized and tenant_b not in serialized, "Raw tenant IDs should not appear in metrics"


async def test_observability_tenant_hash_is_emitted_not_raw_tenant(adapter, metrics_capture):
    caps = await adapter.capabilities()
    tenant = "tenant-hash-check-123"
    ctx = OperationContext(tenant=tenant, request_id="tenant-hash-001")

    await adapter.complete(
        messages=[{"role": "user", "content": "hello"}],
        model=caps.supported_models[0],
        ctx=ctx,
    )

    obs = next((o for o in metrics_capture.observations if o.get("op") == "complete"), metrics_capture.observations[-1])
    extra = obs.get("extra") or {}
    assert "tenant_hash" in extra
    assert isinstance(extra["tenant_hash"], str) and len(extra["tenant_hash"]) >= 8
    assert extra["tenant_hash"] != tenant


async def test_observability_error_metrics_include_code_and_no_prompt_leak(adapter, metrics_capture):
    caps = await adapter.capabilities()
    tenant = "tenant-err-check"
    prompt = "pii email: user@example.com"
    ctx = OperationContext(tenant=tenant, request_id="err-metrics-001")

    with pytest.raises(Exception):
        await adapter.complete(
            messages=[{"role": "user", "content": prompt}],
            model="__no_such_model__",
            ctx=ctx,
        )

    assert_no_sensitive_data_leakage(metrics_capture, [tenant, prompt, "user@example.com"])
    err_obs = next((o for o in metrics_capture.observations if o.get("ok") is False), None)
    assert err_obs is not None
    assert err_obs["code"] != "OK"
