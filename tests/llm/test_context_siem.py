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
from corpus_sdk.llm.llm_base import OperationContext
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
    # Capture metrics output to an in-memory buffer
    output = io.StringIO()
    metrics = ConsoleMetrics(output_file=output, colored=False, flush=True)
    
    # Create adapter with our metrics sink
    adapter = MockLLMAdapter(failure_rate=0.0)
    adapter._metrics = metrics
    
    # Use a clearly identifiable tenant ID that should NEVER appear raw
    secret_tenant = "acme-corp-secret-12345"
    ctx = make_ctx(
        OperationContext,
        tenant=secret_tenant,
        request_id="test-req-ctx-001",
        timeout_ms=30_000
    )
    
    # Execute operation
    result = await adapter.complete(
        messages=[{"role": "user", "content": "sensitive prompt data"}],
        model="mock-model",
        ctx=ctx
    )
    
    # Get all metrics output
    metrics_output = output.getvalue()
    
    # === CRITICAL PRIVACY CHECKS (MUST) ===
    
    # 1. Raw tenant ID MUST NOT appear anywhere
    assert secret_tenant not in metrics_output, \
        f"PRIVACY VIOLATION: Raw tenant ID '{secret_tenant}' found in metrics output"
    
    # 2. Prompt content MUST NOT appear in metrics
    assert "sensitive prompt data" not in metrics_output, \
        "PRIVACY VIOLATION: Prompt content leaked into metrics"
    
    # === OBSERVABILITY CHECKS (SHOULD) ===
    
    # Parse metrics lines (ConsoleMetrics emits JSON per line)
    metrics_lines = [line for line in metrics_output.strip().split('\n') if line.strip()]
    assert len(metrics_lines) > 0, "Expected at least one metrics line"
    
    # Find the observation line (marked with [OBS])
    obs_lines = [line for line in metrics_lines if '[OBS]' in line]
    assert len(obs_lines) > 0, "Expected at least one observation metric"
    
    # Parse the first observation
    obs_line = obs_lines[0]
    # Extract JSON after the [OBS] prefix
    json_start = obs_line.index('{')
    obs_data = json.loads(obs_line[json_start:])
    
    # 3. Component should be 'llm'
    assert obs_data.get('component') == 'llm', \
        f"Expected component='llm', got '{obs_data.get('component')}'"
    
    # 4. Operation should be 'complete'
    assert obs_data.get('op') == 'complete', \
        f"Expected op='complete', got '{obs_data.get('op')}'"
    
    # 5. Status code should be 'OK' for successful operation
    assert obs_data.get('code') == 'OK', \
        f"Expected code='OK', got '{obs_data.get('code')}'"
    
    # 6. Operation should be marked successful
    assert obs_data.get('ok') is True, \
        f"Expected ok=true, got {obs_data.get('ok')}"
    
    # 7. Latency should be recorded
    assert 'ms' in obs_data and isinstance(obs_data['ms'], (int, float)), \
        "Expected 'ms' latency field in metrics"
    assert obs_data['ms'] >= 0, "Latency should be non-negative"
    
    # 8. If tenant is included, it should be in 'extra' and hashed
    if 'extra' in obs_data and obs_data['extra']:
        extra = obs_data['extra']
        if 'tenant' in extra:
            tenant_value = extra['tenant']
            # Should be a hash (typically hex string, 12+ chars)
            assert isinstance(tenant_value, str), "Tenant should be a string"
            assert len(tenant_value) >= 12, "Tenant hash should be at least 12 chars"
            assert tenant_value != secret_tenant, \
                "Tenant in metrics should be hashed, not raw"
            # Hash should be deterministic (same tenant = same hash)
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
    ctx = make_ctx(OperationContext, tenant=secret_tenant, request_id="test-err-001")
    
    # Attempt operation that will fail
    from corpus_sdk.examples.common.errors import Unavailable, ResourceExhausted
    with pytest.raises((Unavailable, ResourceExhausted)):
        await adapter.complete(
            messages=[{"role": "user", "content": "trigger error"}],
            ctx=ctx
        )
    
    metrics_output = output.getvalue()
    
    # Privacy check on error path
    assert secret_tenant not in metrics_output, \
        "Raw tenant ID leaked in error metrics"
    
    # Should have error observation
    obs_lines = [line for line in metrics_output.strip().split('\n') if '[OBS]' in line]
    assert len(obs_lines) > 0, "Expected observation even on error"
    
    # Parse error observation
    obs_line = obs_lines[0]
    json_start = obs_line.index('{')
    obs_data = json.loads(obs_line[json_start:])
    
    # Error should be marked
    assert obs_data.get('ok') is False, "Error operations should have ok=false"
    
    # Error code should be present and not 'OK'
    assert obs_data.get('code') != 'OK', "Error should have non-OK code"
    assert obs_data.get('code') in ['Unavailable', 'ResourceExhausted'], \
        f"Expected error code, got '{obs_data.get('code')}'"


async def test_streaming_metrics_siem_safe():
    """
    Verify streaming operations also maintain SIEM-safe metrics.
    """
    output = io.StringIO()
    metrics = ConsoleMetrics(output_file=output, colored=False, flush=True)
    
    adapter = MockLLMAdapter(failure_rate=0.0)
    adapter._metrics = metrics
    
    secret_tenant = "stream-tenant-secret"
    ctx = make_ctx(OperationContext, tenant=secret_tenant, request_id="test-stream-001")
    
    # Consume stream
    chunks = []
    async for chunk in adapter.stream(
        messages=[{"role": "user", "content": "stream sensitive data"}],
        model="mock-model",
        ctx=ctx
    ):
        chunks.append(chunk)
    
    metrics_output = output.getvalue()
    
    # Privacy checks for streaming
    assert secret_tenant not in metrics_output, \
        "Raw tenant ID leaked in streaming metrics"
    assert "stream sensitive data" not in metrics_output, \
        "Prompt content leaked in streaming metrics"
    
    # Should have stream observation
    obs_lines = [line for line in metrics_output.strip().split('\n') if '[OBS]' in line]
    assert len(obs_lines) > 0, "Expected stream observation"
    
    # Parse stream observation
    obs_line = obs_lines[-1]  # Last observation (final outcome)
    json_start = obs_line.index('{')
    obs_data = json.loads(obs_line[json_start:])
    
    assert obs_data.get('component') == 'llm'
    assert obs_data.get('op') == 'stream'
    assert obs_data.get('ok') is True


async def test_token_counter_metrics_present():
    """
    Verify counter metrics are emitted for token usage.
    """
    output = io.StringIO()
    metrics = ConsoleMetrics(output_file=output, colored=False, flush=True)
    
    adapter = MockLLMAdapter(failure_rate=0.0)
    adapter._metrics = metrics
    
    ctx = make_ctx(OperationContext, tenant="counter-test", request_id="test-ctr-001")
    
    await adapter.complete(
        messages=[{"role": "user", "content": "test"}],
        ctx=ctx
    )
    
    metrics_output = output.getvalue()
    
    # Should have counter metrics
    ctr_lines = [line for line in metrics_output.strip().split('\n') if '[CTR]' in line]
    assert len(ctr_lines) > 0, "Expected counter metrics"
    
    # Check for token counter
    token_counters = [line for line in ctr_lines if 'tokens_processed' in line]
    assert len(token_counters) > 0, "Expected tokens_processed counter"
