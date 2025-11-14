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
from examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_health_health_has_required_fields(adapter):
    """
    §7.6 — Health MUST provide stable shape; §6.2 — identity fields present.
    """
    ctx = make_ctx(OperationContext, request_id="t_health", tenant="test")

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
    ctx = make_ctx(OperationContext, tenant="test")

    # Probe several times; implementation may occasionally report degraded
    for _ in range(20):
        h = await adapter.health(ctx=ctx)

        # Shape must always be valid
        assert isinstance(h, dict)
        assert "ok" in h and isinstance(h["ok"], bool)
        assert "server" in h and isinstance(h["server"], str)
        assert "version" in h and isinstance(h["version"], str)

        # Optional status text (if present) is string
        if "status" in h:
            assert isinstance(h["status"], str)


async def test_health_health_identity_fields_are_stable_across_calls(adapter):
    """
    §7.6 / §6.2 — Identity information (server/version) SHOULD be stable across health probes.
    """
    ctx = make_ctx(OperationContext, request_id="t_health_stability", tenant="test")

    h1 = await adapter.health(ctx=ctx)
    h2 = await adapter.health(ctx=ctx)

    assert h1["server"] == h2["server"], "server identifier should be stable"
    assert h1["version"] == h2["version"], "version should be stable"


async def test_health_health_deadline_preexpired_raises_deadline_exceeded(adapter):
    """
    §6.1 — Pre-expired budgets MUST fail fast with DeadlineExceeded.
    """
    # Absolute epoch 0 guarantees elapsed deadline
    ctx = OperationContext(deadline_ms=0, tenant="test")

    with pytest.raises(DeadlineExceeded):
        await adapter.health(ctx=ctx)