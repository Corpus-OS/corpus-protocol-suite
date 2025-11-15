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
  • Identity fields (server/version) are stable across calls (idempotent probe)
  • Pre-expired deadline triggers DeadlineExceeded for health
"""

import pytest
from corpus_sdk.llm.llm_base import OperationContext, DeadlineExceeded

pytestmark = pytest.mark.asyncio

# Constants for test stability
HEALTH_PROBE_ITERATIONS = 20  # Statistical confidence for degraded state testing
GUARANTEED_EXPIRED_MS = 0     # Epoch 0 ensures pre-expired deadline


async def test_health_health_has_required_fields(adapter):
    """
    §7.6 — Health MUST provide stable shape; §6.2 — identity fields present.
    """
    ctx = OperationContext(request_id="t_health", tenant="test")

    h = await adapter.health(ctx=ctx)

    # Must be a dict
    assert isinstance(h, dict), "health() must return a dict"

    # Required field: ok (boolean)
    assert "ok" in h, "health() must include 'ok' field"
    assert isinstance(h["ok"], bool), "'ok' must be a boolean"

    # Required field: server (non-empty string)
    assert "server" in h, "health() must include 'server' field"
    assert isinstance(h["server"], str) and h["server"].strip(), "'server' must be a non-empty string"

    # Required field: version (non-empty string)
    assert "version" in h, "health() must include 'version' field"
    assert isinstance(h["version"], str) and h["version"].strip(), "'version' must be a non-empty string"


async def test_health_health_shape_consistent_when_degraded(adapter):
    """
    §7.6 — Shape MUST be consistent even when degraded/unavailable.
    """
    ctx = OperationContext(tenant="test")

    # Probe multiple times to catch potential flakiness in degraded states
    for iteration in range(HEALTH_PROBE_ITERATIONS):
        h = await adapter.health(ctx=ctx)

        # Shape must always be valid regardless of health status
        assert isinstance(h, dict), f"Health response must be dict at iteration {iteration}"
        assert "ok" in h and isinstance(h["ok"], bool), f"ok must be boolean at iteration {iteration}"
        assert "server" in h and isinstance(h["server"], str), f"server must be string at iteration {iteration}"
        assert "version" in h and isinstance(h["version"], str), f"version must be string at iteration {iteration}"

        # Optional status text (if present) is string
        if "status" in h:
            assert isinstance(h["status"], str), f"status must be string at iteration {iteration}"


async def test_health_health_identity_fields_are_stable_across_calls(adapter):
    """
    §7.6 / §6.2 — Identity information (server/version) SHOULD be stable across health probes.
    """
    ctx = OperationContext(request_id="t_health_stability", tenant="test")

    h1 = await adapter.health(ctx=ctx)
    h2 = await adapter.health(ctx=ctx)

    assert h1["server"] == h2["server"], "server identifier should be stable across calls"
    assert h1["version"] == h2["version"], "version should be stable across calls"


async def test_health_health_deadline_preexpired_raises_deadline_exceeded(adapter):
    """
    §6.1 — Pre-expired budgets MUST fail fast with DeadlineExceeded.
    """
    # Use epoch 0 to guarantee expired deadline (avoids timing flakiness)
    ctx = OperationContext(deadline_ms=GUARANTEED_EXPIRED_MS, tenant="test")

    with pytest.raises(DeadlineExceeded) as exc_info:
        await adapter.health(ctx=ctx)
    
    # Verify error contains expected code
    err = exc_info.value
    assert err.code in ("DEADLINE", "DEADLINE_EXCEEDED"), \
        f"Expected DEADLINE code, got: {err.code}"


async def test_health_health_includes_optional_uptime_if_provided(adapter):
    """
    §7.6 — Optional uptime field, if present, must be positive integer.
    """
    ctx = OperationContext(request_id="t_health_uptime", tenant="test")

    h = await adapter.health(ctx=ctx)

    # uptime is optional, but if present must be valid
    if "uptime_seconds" in h:
        assert isinstance(h["uptime_seconds"], int), "uptime_seconds must be integer if provided"
        assert h["uptime_seconds"] >= 0, "uptime_seconds must be non-negative"


async def test_health_health_includes_optional_details_if_provided(adapter):
    """
    §7.6 — Optional details field, if present, must be dict.
    """
    ctx = OperationContext(request_id="t_health_details", tenant="test")

    h = await adapter.health(ctx=ctx)

    # details is optional, but if present must be valid
    if "details" in h:
        assert isinstance(h["details"], dict), "details must be dict if provided"