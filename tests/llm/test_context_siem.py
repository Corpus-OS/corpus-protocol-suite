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
import re

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


@pytest.fixture
def metrics_capture(adapter):
    """Fixture to safely capture metrics for testing."""
    original_metrics = getattr(adapter, "_metrics", None)
    capture = CaptureMetrics()
    adapter._metrics = capture  # type: ignore[attr-defined]
    yield capture
    # Restore original metrics
    if original_metrics is not None:
        adapter._metrics = original_metrics  # type: ignore[attr-defined]


def assert_no_sensitive_data_leakage(metrics: CaptureMetrics, sensitive_strings: List[str]):
    """Assert that no sensitive data appears in metrics output."""
    serialized = str(metrics.observations) + str(metrics.counters)
    
    for sensitive in sensitive_strings:
        assert sensitive not in serialized, \
            f"PRIVACY VIOLATION: Sensitive string '{sensitive}' found in metrics"


async def test_observability_context_propagates_to_metrics_siem_safe(adapter, metrics_capture):
    """
    SPECIFICATION.md §13.2, §15 — SIEM-Safe Observability

    Verify that:
    1. Raw tenant IDs NEVER appear in metrics output
    2. Tenant is hashed when present in telemetry
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

    # At least one observation should be emitted
    assert metrics_capture.observations, "Expected at least one observation metric"
    obs = metrics_capture.observations[0]

    # 1. No sensitive data leakage
    sensitive_strings = [secret_tenant, sensitive_prompt, "user@example.com"]
    assert_no_sensitive_data_leakage(metrics_capture, sensitive_strings)

    # 2. Required observation fields
    assert obs["component"] == "llm", "Component should be 'llm'"
    assert obs["op"] == "complete", "Operation should be 'complete'"
    assert obs["code"] == "OK", "Status code should be 'OK' for successful operation"
    assert obs["ok"] is True, "Operation should be marked successful"
    assert isinstance(obs["ms"], (int, float)) and obs["ms"] >= 0, "Latency should be recorded"

    # 3. If tenant is included in extra, it must be hashed/obfuscated
    extra = obs.get("extra") or {}
    if "tenant" in extra:
        tenant_value = extra["tenant"]
        assert isinstance(tenant_value, str), "Tenant in extra must be string"
        assert len(tenant_value) >= 8, "Tenant hash should be reasonable length"
        assert tenant_value != secret_tenant, "Tenant must not be raw value"
        # Should not contain original tenant as substring
        assert secret_tenant not in tenant_value, "Tenant hash must not contain original"


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

    # Force an error via obviously invalid model name
    with pytest.raises(Exception):
        await adapter.complete(
            messages=[{"role": "user", "content": sensitive_content}],
            model="__no_such_model_should_always_fail_123__",
            ctx=ctx,
        )

    # Verify metrics were emitted even for error path
    assert metrics_capture.observations, "Expected observation even on error path"
    
    # No sensitive data in error metrics
    sensitive_strings = [secret_tenant, sensitive_content, "555-1234"]
    assert_no_sensitive_data_leakage(metrics_capture, sensitive_strings)

    # Find the error observation (last one should be the error)
    error_obs = metrics_capture.observations[-1]
    assert error_obs["ok"] is False, "Error operations should have ok=false"
    assert error_obs["code"] != "OK", "Error should have non-OK code"
    assert error_obs["op"] == "complete", "Error observation should indicate operation"


async def test_observability_streaming_metrics_siem_safe(adapter, metrics_capture):
    """
    Verify streaming operations also maintain SIEM-safe metrics.
    """
    caps = await adapter.capabilities()
    if not caps.supports_streaming:
        pytest.skip("Adapter does not support streaming")

    secret_tenant = "stream-tenant-secret-xyz"
    sensitive_data = "stream sensitive data: credit card 4111-1111-1111-1111"
    
    ctx = OperationContext(
        tenant=secret_tenant,
        request_id="test-stream-001",
    )

    # Consume entire stream to ensure complete observation
    chunks = []
    async for chunk in adapter.stream(
        messages=[{"role": "user", "content": sensitive_data}],
        model=caps.supported_models[0],
        ctx=ctx,
    ):
        chunks.append(chunk)

    # Verify we got a complete stream
    assert chunks, "Should receive stream chunks"
    assert hasattr(chunks[-1], 'is_final') and chunks[-1].is_final, "Stream should end with final chunk"

    # No sensitive data in streaming metrics
    sensitive_strings = [secret_tenant, sensitive_data, "4111-1111-1111-1111"]
    assert_no_sensitive_data_leakage(metrics_capture, sensitive_strings)

    # Expect streaming observations
    stream_obs = [o for o in metrics_capture.observations if o["op"] == "stream"]
    assert stream_obs, "Expected stream observation"
    
    last_stream = stream_obs[-1]
    assert last_stream["component"] == "llm"
    assert last_stream["ok"] is True, "Successful stream should have ok=true"


async def test_observability_token_counter_metrics_present(adapter, metrics_capture):
    """
    Verify counter metrics are emitted for token usage.
    """
    caps = await adapter.capabilities()
    if not caps.supports_count_tokens:
        pytest.skip("Adapter does not support token counting")

    ctx = OperationContext(
        tenant="counter-test-tenant",
        request_id="test-ctr-001",
    )

    await adapter.complete(
        messages=[{"role": "user", "content": "test token counting metrics"}],
        model=caps.supported_models[0],
        ctx=ctx,
    )

    # Expect counter metrics
    assert metrics_capture.counters, "Expected counter metrics for token usage"

    # Look for token-related counters
    token_counters = [
        c for c in metrics_capture.counters
        if any(token_keyword in c["name"] for token_keyword in 
               ["tokens", "token", "prompt_tokens", "completion_tokens"])
    ]
    assert token_counters, "Expected token-related counter metrics"


async def test_observability_metrics_structure_consistency(adapter, metrics_capture):
    """
    Verify metrics structure is consistent across different operation types.
    """
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="structure-test", request_id="test-struct-001")

    # Test multiple operation types
    operations = []
    
    # Complete operation
    await adapter.complete(
        messages=[{"role": "user", "content": "test structure"}],
        model=caps.supported_models[0],
        ctx=ctx,
    )
    operations.append("complete")

    # Count tokens if supported
    if caps.supports_count_tokens:
        await adapter.count_tokens("test tokens", ctx=ctx)
        operations.append("count_tokens")

    # Verify consistent structure across all observations
    for obs in metrics_capture.observations:
        assert "component" in obs and obs["component"] == "llm"
        assert "op" in obs and obs["op"] in operations
        assert "ok" in obs and isinstance(obs["ok"], bool)
        assert "code" in obs and isinstance(obs["code"], str)
        assert "ms" in obs and isinstance(obs["ms"], (int, float)) and obs["ms"] >= 0


async def test_observability_no_metric_leakage_between_tenants(adapter, metrics_capture):
    """
    Verify metrics don't leak information between different tenant contexts.
    """
    caps = await adapter.capabilities()
    
    # First tenant
    tenant_a = "tenant-alpha-secret"
    ctx_a = OperationContext(tenant=tenant_a, request_id="req-tenant-a")
    
    await adapter.complete(
        messages=[{"role": "user", "content": "request from tenant A"}],
        model=caps.supported_models[0],
        ctx=ctx_a,
    )

    # Clear metrics to test isolation
    observations_after_a = len(metrics_capture.observations)
    
    # Second tenant  
    tenant_b = "tenant-beta-secret"
    ctx_b = OperationContext(tenant=tenant_b, request_id="req-tenant-b")
    
    await adapter.complete(
        messages=[{"role": "user", "content": "request from tenant B"}],
        model=caps.supported_models[0],
        ctx=ctx_b,
    )

    # Verify both requests generated metrics
    assert len(metrics_capture.observations) > observations_after_a, \
        "Second tenant request should generate additional metrics"

    # Verify no cross-tenant leakage
    serialized = str(metrics_capture.observations) + str(metrics_capture.counters)
    assert tenant_a not in serialized and tenant_b not in serialized, \
        "Raw tenant IDs should not appear in metrics"