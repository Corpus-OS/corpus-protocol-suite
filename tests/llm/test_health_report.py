# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Health report shape (enhanced).

Specification references:
  • §7.6 (Health): shape normalization — adapters SHOULD still report server/version and not crash on health checks
  • §6.2 (Capability Discovery): identity fields (server, version) are stable identifiers (reused by health responses)
  • §6.1 (Operation Context): deadline semantics — pre-expired budgets MUST fail fast with DeadlineExceeded

Covers:
  • Health returns dict with required keys (ok, server, version) and correct types
  • Shape is consistent regardless of status (healthy/degraded)
  • Identity fields (server/version) are stable across calls
  • Pre-expired deadline triggers DeadlineExceeded when supports_deadline=True
"""

import pytest
from corpus_sdk.llm.llm_base import OperationContext, DeadlineExceeded

pytestmark = pytest.mark.asyncio

HEALTH_PROBE_ITERATIONS = 20
GUARANTEED_EXPIRED_MS = 0


async def test_health_health_has_required_fields(adapter):
    ctx = OperationContext(request_id="t_health", tenant="test")
    h = await adapter.health(ctx=ctx)

    assert isinstance(h, dict), "health() must return a dict"
    assert "ok" in h and isinstance(h["ok"], bool), "health() must include boolean 'ok'"
    assert "server" in h and isinstance(h["server"], str) and h["server"].strip(), "'server' must be a non-empty string"
    assert "version" in h and isinstance(h["version"], str) and h["version"].strip(), "'version' must be a non-empty string"


async def test_health_health_shape_consistent_when_degraded(adapter):
    ctx = OperationContext(tenant="test")

    for i in range(HEALTH_PROBE_ITERATIONS):
        h = await adapter.health(ctx=ctx)
        assert isinstance(h, dict), f"Health response must be dict at iteration {i}"
        assert "ok" in h and isinstance(h["ok"], bool), f"ok must be boolean at iteration {i}"
        assert "server" in h and isinstance(h["server"], str), f"server must be string at iteration {i}"
        assert "version" in h and isinstance(h["version"], str), f"version must be string at iteration {i}"

        if "status" in h:
            assert isinstance(h["status"], str), f"status must be string at iteration {i}"


async def test_health_health_identity_fields_are_stable_across_calls(adapter):
    ctx = OperationContext(request_id="t_health_stability", tenant="test")
    h1 = await adapter.health(ctx=ctx)
    h2 = await adapter.health(ctx=ctx)

    assert h1["server"] == h2["server"]
    assert h1["version"] == h2["version"]


async def test_health_health_deadline_preexpired_raises_deadline_exceeded(adapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(deadline_ms=GUARANTEED_EXPIRED_MS, tenant="test")

    if caps.supports_deadline:
        with pytest.raises(DeadlineExceeded) as exc_info:
            await adapter.health(ctx=ctx)
        err = exc_info.value
        assert err.code in ("DEADLINE", "DEADLINE_EXCEEDED"), f"Unexpected deadline code: {err.code}"
    else:
        h = await adapter.health(ctx=ctx)
        assert isinstance(h, dict) and "ok" in h


async def test_health_health_includes_optional_uptime_if_provided(adapter):
    ctx = OperationContext(request_id="t_health_uptime", tenant="test")
    h = await adapter.health(ctx=ctx)

    if "uptime_seconds" in h:
        assert isinstance(h["uptime_seconds"], int)
        assert h["uptime_seconds"] >= 0


async def test_health_health_includes_optional_details_if_provided(adapter):
    ctx = OperationContext(request_id="t_health_details", tenant="test")
    h = await adapter.health(ctx=ctx)

    if "details" in h:
        assert isinstance(h["details"], dict)


async def test_health_deadline_capability_alignment(adapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(deadline_ms=0, tenant="test")

    if caps.supports_deadline:
        with pytest.raises(DeadlineExceeded) as exc_info:
            await adapter.health(ctx=ctx)
        assert exc_info.value.code in ("DEADLINE", "DEADLINE_EXCEEDED")
    else:
        h = await adapter.health(ctx=ctx)
        assert isinstance(h, dict) and "ok" in h
