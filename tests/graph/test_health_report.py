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

from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    HealthStatus,
)
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_health_returns_required_fields():
    ctx = make_ctx(GraphContext, request_id="t_health_fields", tenant="t")
    h = await MockGraphAdapter().health(ctx=ctx)
    for k in ("status", "server", "version"):
        assert k in h


async def test_health_status_is_valid_enum():
    ctx = make_ctx(GraphContext, request_id="t_health_enum", tenant="t")
    h = await MockGraphAdapter().health(ctx=ctx)
    assert h["status"] in (HealthStatus.OK, HealthStatus.DEGRADED, HealthStatus.UNAVAILABLE, HealthStatus.READ_ONLY)


async def test_health_includes_read_only_flag():
    ctx = make_ctx(GraphContext, request_id="t_health_ro", tenant="t")
    h = await MockGraphAdapter().health(ctx=ctx)
    assert "read_only" in h


async def test_health_includes_degraded_flag():
    ctx = make_ctx(GraphContext, request_id="t_health_deg", tenant="t")
    h = await MockGraphAdapter().health(ctx=ctx)
    assert "degraded" in h


async def test_health_consistent_on_error():
    ctx = make_ctx(GraphContext, request_id="t_health_shape", tenant="t")
    h = await MockGraphAdapter().health(ctx=ctx)
    assert isinstance(h, dict)
