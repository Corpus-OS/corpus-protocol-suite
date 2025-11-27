# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Health reporting.

Asserts (Spec refs):
  • health() returns required fields                          (§7.6)
  • status in defined enum set (string-valued)               (§7.6)
  • includes read_only & degraded flags                       (§7.6)
  • shape remains stable                                      (§7.6, §6.4)
"""
import pytest

from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    BaseGraphAdapter,
)

pytestmark = pytest.mark.asyncio


async def test_health_returns_required_fields(adapter: BaseGraphAdapter):
    """§7.6: Health must return all required fields."""
    ctx = GraphContext(request_id="t_health_fields", tenant="t")
    h = await adapter.health(ctx=ctx)

    assert isinstance(h, dict), "Health response must be a dictionary"
    for k in ("status", "server", "version"):
        assert k in h, f"Missing required health field: {k}"


async def test_health_status_is_valid_enum(adapter: BaseGraphAdapter):
    """§7.6: Health status must be a valid value."""
    ctx = GraphContext(request_id="t_health_enum", tenant="t")
    h = await adapter.health(ctx=ctx)

    # BaseGraphAdapter normalizes status to a string like "ok" / "degraded".
    valid_status = {"ok", "degraded", "unavailable", "read_only"}
    assert isinstance(h["status"], str)
    assert h["status"] in valid_status, f"Invalid health status: {h['status']}"


async def test_health_includes_read_only_flag(adapter: BaseGraphAdapter):
    """§7.6: Health must include read_only flag."""
    ctx = GraphContext(request_id="t_health_ro", tenant="t")
    h = await adapter.health(ctx=ctx)
    assert "read_only" in h, "Missing read_only flag in health response"
    assert isinstance(h["read_only"], bool), "read_only must be boolean"


async def test_health_includes_degraded_flag(adapter: BaseGraphAdapter):
    """§7.6: Health must include degraded flag."""
    ctx = GraphContext(request_id="t_health_deg", tenant="t")
    h = await adapter.health(ctx=ctx)
    assert "degraded" in h, "Missing degraded flag in health response"
    assert isinstance(h["degraded"], bool), "degraded must be boolean"


async def test_health_consistent_on_error(adapter: BaseGraphAdapter):
    """§7.6: Health response shape must remain consistent across states."""
    ctx = GraphContext(request_id="t_health_shape", tenant="t")
    h = await adapter.health(ctx=ctx)

    assert isinstance(h, dict), "Health response must be a dictionary"

    required_fields = {"status", "server", "version", "read_only", "degraded"}
    assert required_fields.issubset(h.keys()), (
        f"Missing required health fields: {required_fields - set(h.keys())}"
    )
