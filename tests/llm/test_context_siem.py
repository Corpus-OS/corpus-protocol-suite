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
    NotSupported,
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


@pytest.fixture
def metrics_capture(adapter):
    """Fixture to safely capture metrics for testing."""
    # Conformance assumes Base-style adapters expose a metrics sink.
    if not hasattr(adapter, "_metrics"):
        pytest.fail("Adapter does not expose metrics sink required for SIEM-safe observability conformance")
    original_metrics = getattr(adapter, "_metrics", None)
    capture = CaptureMetrics()
    adapter._metrics = capture  # type: ignore[attr-defined]
    yield capture
    if original_metrics is not None:
        adapter._metrics = original_metrics  # type: ignore[attr-defined]


def assert_no_sensitive_data_leakage(metrics: CaptureMetrics, sensitive_strings: List[str]):
    """Assert that no sensitive data appears in metrics output."""
    serialized = str(metrics.observations) + str(metrics.counters)
    for sensitive in sensitive_strings:
        assert sensitive not in serialized, f"PRIVACY VIOLATION: Sensitive string '{sensitive}' found in metrics"


async def test_observability_context_propagates_to_metrics_siem_safe(adapter, metrics_capture):
    """
    SPECIFICATION.md §13.2, §15 — SIEM-Safe Observability

    Verify that:
    1. Raw tenant IDs NEVER appear in metrics output
    2. Tenant is hashed when present in telemetry (tenant_hash)
    3. Operation metadata (component, op, code) is emitted
    4. No prompt content or PII leaks into metrics
    """
    caps = await adapter.capabilities()
    secret_tenant = "acme-corp-secret-12345"
    sensitive_prompt = "sensitive prompt data with PII: user@example.com"

    ctx = OperationContext(
        tenant=secret_tenant,
        request_id="test-req-ctx-001",
    )

    await adapter.complete(
        messages=[{"role": "user", "content": sensitive_prompt}],
        model=caps.supported_models[0],
        ctx=ctx,
    )

    assert metrics_capture.observations, "Expected at least one observation metric"

    complete_obs = None
    for obs in metrics_capture.observations:
        if obs.get("op") == "complete":
            complete_obs = obs
            break
    if complete_obs is None:
        complete_obs = metrics_capture.observations[-1]

    sensitive_strings = [secret_tenant, sensitive_prompt, "user@example.com"]
    assert_no_sensitive_data_leakage(metrics_capture, sensitive_strings)

    assert complete_obs["component"] == "llm", "Component should be 'llm'"
    assert isinstance(complete_obs["op"], str) and complete_obs["op"], "Operation should be present"
    assert complete_obs["code"] == "OK", "Status code should be 'OK' for successful operation"
    assert complete_obs["ok"] is True, "Operation should be marked successful"
    assert isinstance(complete_obs["ms"], (int, float)) and complete_obs["ms"] >= 0, "Latency should be recorded"

    extra = complete_obs.get("extra") or {}
    # Base contract emits tenant_hash (not raw tenant)
    if secret_tenant:
        assert "tenant_hash" in extra, "Tenant hash should be present in metrics extra when ctx.tenant is set"
        tenant_hash = extra["tenant_hash"]
        assert isinstance(tenant_hash, str) and len(tenant_hash) >= 8
        assert tenant_hash != secret_tenant
        assert secret_tenant not in tenant_hash


async def test_observability_metrics_emitted_on_error_path(adapter, metrics_capture):
    """
    Verify that metrics are emitted even when operations fail.
    Error paths MUST NOT leak sensitive information either.
    """
    secret_tenant = "error-tenant-secret-789"
    sensitive_content = "trigger error with sensitive data: 555-1234"

    ctx = OperationContext(
        tenant=secret_tenant,
        request_id="test-err-001",
    )

    with pytest.raises(Exception):
        await adapter.complete(
            messages=[{"role": "user", "content": sensitive_content}],
            model="__no_such_model_should_always_fail_123__",
            ctx=ctx,
        )

    assert metrics_capture.observations, "Expected observation even on error path"

    sensitive_strings = [secret_tenant, sensitive_content, "555-1234"]
    assert_no_sensitive_data_leakage(metrics_capture, sensitive_strings)

    error_obs = None
    for obs in metrics_capture.observations:
        if obs.get("ok") is False:
            error_obs = obs
            break
    if error_obs is None:
        pytest.fail("Expected at least one observation with ok=False on error path")

    assert error_obs["ok"] is False
    assert error_obs["code"] != "OK"
    assert isinstance(error_obs.get("op"), str) and error_obs["op"]


async def test_observability_streaming_metrics_siem_safe(adapter, metrics_capture):
    """
    Verify streaming operations also maintain SIEM-safe metrics.

    Capability↔behavior:
      - If supports_streaming is False: stream() MUST raise NotSupported.
      - If True: consuming stream MUST NOT leak prompt/tenant into metrics.
    """
    caps = await adapter.capabilities()

    secret_tenant = "stream-tenant-secret-xyz"
    sensitive_data = "stream sensitive data: credit card 4111-1111-1111-1111"

    ctx = OperationContext(
        tenant=secret_tenant,
        request_id="test-stream-001",
    )

    if not caps.supports_streaming:
        with pytest.raises(NotSupported):
            agen = adapter.stream(
                messages=[{"role": "user", "content": sensitive_data}],
                model=caps.supported_models[0],
                ctx=ctx,
            )
            async for _ in agen:
                pass
        # Even on NotSupported, must not leak sensitive data into metrics.
        assert_no_sensitive_data_leakage(metrics_capture, [secret_tenant, sensitive_data, "4111-1111-1111-1111"])
        return

    chunks = []
    async for chunk in adapter.stream(
        messages=[{"role": "user", "content": sensitive_data}],
        model=caps.supported_models[0],
        ctx=ctx,
    ):
        chunks.append(chunk)

    assert chunks, "Should receive stream chunks"

    sensitive_strings = [secret_tenant, sensitive_data, "4111-1111-1111-1111"]
    assert_no_sensitive_data_leakage(metrics_capture, sensitive_strings)

    # Validate basic shape for any emitted observations
    assert metrics_capture.observations, "Expected at least one observation for streaming operation"
    for obs in metrics_capture.observations:
        assert obs["component"] == "llm"
        assert isinstance(obs["op"], str) and obs["op"]


async def test_observability_token_counter_metrics_present(adapter, metrics_capture):
    """
    Verify counter metrics are SIEM-safe and structured when present.

    Capability↔behavior:
      - If supports_count_tokens is False: count_tokens() MUST raise NotSupported.
      - If True: token counting operations must not leak prompt/tenant.
    """
    caps = await adapter.capabilities()

    ctx = OperationContext(
        tenant="counter-test-tenant",
        request_id="test-ctr-001",
    )

    if not caps.supports_count_tokens:
        with pytest.raises(NotSupported):
            await adapter.count_tokens("test token counting metrics", ctx=ctx)
        assert_no_sensitive_data_leakage(metrics_capture, ["counter-test-tenant", "test token counting metrics"])
        return

    await adapter.count_tokens("test token counting metrics", ctx=ctx)

    sensitive_strings = ["counter-test-tenant", "test token counting metrics"]
    assert_no_sensitive_data_leakage(metrics_capture, sensitive_strings)

    if metrics_capture.counters:
        for counter in metrics_capture.counters:
            assert "component" in counter and isinstance(counter["component"], str)
            assert "name" in counter and isinstance(counter["name"], str)
            assert "value" in counter and isinstance(counter["value"], int)


async def test_observability_metrics_structure_consistency(adapter, metrics_capture):
    """
    Verify metrics structure is consistent across different operation types.
    """
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="structure-test", request_id="test-struct-001")

    await adapter.complete(
        messages=[{"role": "user", "content": "test structure"}],
        model=caps.supported_models[0],
        ctx=ctx,
    )

    if caps.supports_count_tokens:
        await adapter.count_tokens("test tokens", ctx=ctx)

    assert metrics_capture.observations, "Expected observations to be recorded"

    for obs in metrics_capture.observations:
        assert "component" in obs and obs["component"] == "llm"
        assert "op" in obs and isinstance(obs["op"], str)
        assert "ok" in obs and isinstance(obs["ok"], bool)
        assert "code" in obs and isinstance(obs["code"], str)
        assert "ms" in obs and isinstance(obs["ms"], (int, float)) and obs["ms"] >= 0


async def test_observability_no_metric_leakage_between_tenants(adapter, metrics_capture):
    """
    Verify metrics don't leak information between different tenant contexts.
    """
    caps = await adapter.capabilities()

    tenant_a = "tenant-alpha-secret"
    ctx_a = OperationContext(tenant=tenant_a, request_id="req-tenant-a")

    await adapter.complete(
        messages=[{"role": "user", "content": "request from tenant A"}],
        model=caps.supported_models[0],
        ctx=ctx_a,
    )

    observations_after_a = len(metrics_capture.observations)

    tenant_b = "tenant-beta-secret"
    ctx_b = OperationContext(tenant=tenant_b, request_id="req-tenant-b")

    await adapter.complete(
        messages=[{"role": "user", "content": "request from tenant B"}],
        model=caps.supported_models[0],
        ctx=ctx_b,
    )

    assert len(metrics_capture.observations) > observations_after_a, "Second tenant request should generate additional metrics"

    serialized = str(metrics_capture.observations) + str(metrics_capture.counters)
    assert tenant_a not in serialized and tenant_b not in serialized, "Raw tenant IDs should not appear in metrics"
