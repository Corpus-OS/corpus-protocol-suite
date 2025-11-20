# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Health reporting.

Asserts (Spec refs):
  • health() returns required fields                          (§7.6)
  • status in defined enum set                                (§7.6)
  • includes read_only & degraded flags                       (§7.6)
  • shape remains stable                                      (§7.6, §6.4)
"""
import pytest

from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    HealthStatus,
    BaseGraphAdapter,
)

pytestmark = pytest.mark.asyncio


def make_ctx(ctx_cls, **kwargs):
    """Local helper to construct an OperationContext."""
    return ctx_cls(**kwargs)


async def test_health_returns_required_fields(adapter: BaseGraphAdapter):
    """§7.6: Health must return all required fields."""
    ctx = make_ctx(GraphContext, request_id="t_health_fields", tenant="t")
    h = await adapter.health(ctx=ctx)
    for k in ("status", "server", "version"):
        assert k in h, f"Missing required health field: {k}"


async def test_health_status_is_valid_enum(adapter: BaseGraphAdapter):
    """§7.6: Health status must be a valid enum value."""
    ctx = make_ctx(GraphContext, request_id="t_health_enum", tenant="t")
    h = await adapter.health(ctx=ctx)
    assert h["status"] in (
        HealthStatus.OK,
        HealthStatus.DEGRADED,
        HealthStatus.UNAVAILABLE,
        HealthStatus.READ_ONLY,
    ), f"Invalid health status: {h['status']}"


async def test_health_includes_read_only_flag(adapter: BaseGraphAdapter):
    """§7.6: Health must include read_only flag."""
    ctx = make_ctx(GraphContext, request_id="t_health_ro", tenant="t")
    h = await adapter.health(ctx=ctx)
    assert "read_only" in h, "Missing read_only flag in health response"
    assert isinstance(h["read_only"], bool), "read_only must be boolean"


async def test_health_includes_degraded_flag(adapter: BaseGraphAdapter):
    """§7.6: Health must include degraded flag."""
    ctx = make_ctx(GraphContext, request_id="t_health_deg", tenant="t")
    h = await adapter.health(ctx=ctx)
    assert "degraded" in h, "Missing degraded flag in health response"
    assert isinstance(h["degraded"], bool), "degraded must be boolean"


async def test_health_consistent_on_error(adapter: BaseGraphAdapter):
    """§7.6: Health response shape must remain consistent even in error states."""
    ctx = make_ctx(GraphContext, request_id="t_health_shape", tenant="t")
    h = await adapter.health(ctx=ctx)
    assert isinstance(h, dict), "Health response must be a dictionary"
    # Should maintain consistent structure regardless of health state
    required_fields = {"status", "server", "version", "read_only", "degraded"}
    assert required_fields.issubset(h.keys()), (
        f"Missing required health fields: {required_fields - set(h.keys())}"
    )
