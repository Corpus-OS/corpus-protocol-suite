# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Context propagation and SIEM-safe metrics.
Covers:
  • Tenant identifiers are hashed, never logged raw (§13.2, §15)
  • Metrics include operation metadata (component, op, code)
  • Request context flows through to observability layer
  • No PII or sensitive data appears in telemetry
"""
import io
import json
import pytest

from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.llm.llm_base import (
    OperationContext,
    Unavailable,
    ResourceExhausted,
)
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.metrics_console import ConsoleMetrics

pytestmark = pytest.mark.asyncio


async def test_context_propagates_to_metrics_siem_safe():
    """
    SPECIFICATION.md §13.2, §15 — SIEM-Safe Observability

    Verify that:
    1. Raw tenant IDs NEVER appear in metrics output
    2. Tenant is hashed when present in telemetry
    3. Operation metadata (component, op, code) is emitted
    4. No prompt content or PII leaks into metrics
    """
    output = io.StringIO()
    metrics = ConsoleMetrics(output_file=output, colored=False, flush=True)

    adapter = MockLLMAdapter(failure_rate=0.0)
    adapter._metrics = metrics  # use console metrics sink

    secret_tenant = "acme-corp-secret-12345"
    ctx = make_ctx(
        OperationContext,
        tenant=secret_tenant,
        request_id="test-req-ctx-001",
        timeout_ms=30_000,
    )

    await adapter.complete(
        messages=[{"role": "user", "content": "sensitive prompt data"}],
        model="mock-model",
        ctx=ctx,
    )

    metrics_output = output.getvalue()

    # --- CRITICAL PRIVACY CHECKS (MUST) ---

    # 1. Raw tenant ID MUST NOT appear anywhere
    assert secret_tenant not in metrics_output, \
        f"PRIVACY VIOLATION: Raw tenant ID '{secret_tenant}' found in metrics output"

    # 2. Prompt content MUST NOT appear in metrics
    assert "sensitive prompt data" not in metrics_output, \
        "PRIVACY VIOLATION: Prompt content leaked into metrics"

    # --- OBSERVABILITY CHECKS (SHOULD) ---

    # ConsoleMetrics emits one JSON object per line
    metrics_lines = [line for line in metrics_output.strip().split("\n") if line.strip()]
    assert metrics_lines, "Expected at least one metrics line"

    # Find observation line(s)
    obs_lines = [line for line in metrics_lines if "[OBS]" in line]
    assert obs_lines, "Expected at least one observation metric"

    # Parse first observation
    obs_line = obs_lines[0]
    json_start = obs_line.index("{")
    obs_data = json.loads(obs_line[json_start:])

    # 3. Component should be 'llm'
    assert obs_data.get("component") == "llm", \
        f"Expected component='llm', got '{obs_data.get('component')}'"

    # 4. Operation should be 'complete'
    assert obs_data.get("op") == "complete", \
        f"Expected op='complete', got '{obs_data.get('op')}'"

    # 5. Status code should be 'OK' for successful operation
    assert obs_data.get("code") == "OK", \
        f"Expected code='OK', got '{obs_data.get('code')}'"

    # 6. Operation should be marked successful
    assert obs_data.get("ok") is True, \
        f"Expected ok=true, got {obs_data.get('ok')}"

    # 7. Latency should be recorded
    assert "ms" in obs_data and isinstance(obs_data["ms"], (int, float)), \
        "Expected 'ms' latency field in metrics"
    assert obs_data["ms"] >= 0, "Latency should be non-negative"

    # 8. If tenant is included, it should be hashed (adapter base hashes tenant)
    extra = obs_data.get("extra") or {}
    if "tenant" in extra:
        tenant_value = extra["tenant"]
        assert isinstance(tenant_value, str), "Tenant should be a string"
        assert len(tenant_value) >= 12, "Tenant hash should be at least 12 chars"
        assert tenant_value != secret_tenant, \
            "Tenant in metrics should be hashed, not raw"
        assert tenant_value.isalnum(), "Tenant hash should be alphanumeric"


async def test_metrics_emitted_on_error_path():
    """
    Verify that metrics are emitted even when operations fail.
    Error paths MUST NOT leak sensitive information either.
    """
    output = io.StringIO()
    metrics = ConsoleMetrics(output_file=output, colored=False, flush=True)

    # Force failure path
    adapter = MockLLMAdapter(failure_rate=1.0)
    adapter._metrics = metrics

    secret_tenant = "error-tenant-secret"
    ctx = make_ctx(
        OperationContext,
        tenant=secret_tenant,
        request_id="test-err-001",
    )

    with pytest.raises((Unavailable, ResourceExhausted)):
        await adapter.complete(
            messages=[{"role": "user", "content": "trigger error"}],
            ctx=ctx,
        )

    metrics_output = output.getvalue()

    # No raw tenant in error metrics
    assert secret_tenant not in metrics_output, \
        "Raw tenant ID leaked in error metrics"

    # Expect at least one observation on error path
    obs_lines = [line for line in metrics_output.strip().split("\n") if "[OBS]" in line]
    assert obs_lines, "Expected observation even on error"

    obs_line = obs_lines[0]
    json_start = obs_line.index("{")
    obs_data = json.loads(obs_line[json_start:])

    # Error should be marked
    assert obs_data.get("ok") is False, "Error operations should have ok=false"

    # Error code present and non-OK. For these errors, codes are UPPER_SNAKE_CASE.
    assert obs_data.get("code") != "OK", "Error should have non-OK code"
    assert obs_data.get("code") in ("UNAVAILABLE", "RESOURCE_EXHAUSTED"), \
        f"Unexpected error code '{obs_data.get('code')}'"


async def test_streaming_metrics_siem_safe():
    """
    Verify streaming operations also maintain SIEM-safe metrics.
    """
    output = io.StringIO()
    metrics = ConsoleMetrics(output_file=output, colored=False, flush=True)

    adapter = MockLLMAdapter(failure_rate=0.0)
    adapter._metrics = metrics

    secret_tenant = "stream-tenant-secret"
    ctx = make_ctx(
        OperationContext,
        tenant=secret_tenant,
        request_id="test-stream-001",
    )

    chunks = []
    async for chunk in adapter.stream(
        messages=[{"role": "user", "content": "stream sensitive data"}],
        model="mock-model",
        ctx=ctx,
    ):
        chunks.append(chunk)

    metrics_output = output.getvalue()

    # Privacy checks for streaming
    assert secret_tenant not in metrics_output, \
        "Raw tenant ID leaked in streaming metrics"
    assert "stream sensitive data" not in metrics_output, \
        "Prompt content leaked in streaming metrics"

    # Expect a stream observation
    obs_lines = [line for line in metrics_output.strip().split("\n") if "[OBS]" in line]
    assert obs_lines, "Expected stream observation"

    obs_line = obs_lines[-1]
    json_start = obs_line.index("{")
    obs_data = json.loads(obs_line[json_start:])

    assert obs_data.get("component") == "llm"
    assert obs_data.get("op") == "stream"
    assert obs_data.get("ok") is True


async def test_token_counter_metrics_present():
    """
    Verify counter metrics are emitted for token usage.
    """
    output = io.StringIO()
    metrics = ConsoleMetrics(output_file=output, colored=False, flush=True)

    adapter = MockLLMAdapter(failure_rate=0.0)
    adapter._metrics = metrics

    ctx = make_ctx(
        OperationContext,
        tenant="counter-test",
        request_id="test-ctr-001",
    )

    await adapter.complete(
        messages=[{"role": "user", "content": "test"}],
        ctx=ctx,
    )

    metrics_output = output.getvalue()

    # Expect counter metrics
    ctr_lines = [line for line in metrics_output.strip().split("\n") if "[CTR]" in line]
    assert ctr_lines, "Expected counter metrics"

    # Expect tokens_processed counter
    token_counters = [line for line in ctr_lines if "tokens_processed" in line]
    assert token_counters, "Expected tokens_processed counter"
