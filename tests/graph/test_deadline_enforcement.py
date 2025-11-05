# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Deadline enforcement.

Asserts (Spec refs):
  • Non-negative remaining budget                            (§6.1)
  • Expired deadlines fail fast (preflight)                  (§6.1, §12.1)
  • Stream enforces deadline mid-operation                   (§6.1, §12.1)
Note: The spec suggests Graph MAY normalize to Unavailable when budget elapses (§6.1),
but the reference base raises DeadlineExceeded; tests follow the reference behavior.
"""
import asyncio
import time
import pytest

from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    DeadlineExceeded,
)
from corpus_sdk.examples.common.ctx import make_ctx, remaining_budget_ms, clear_time_cache

pytestmark = pytest.mark.asyncio


async def test_deadline_budget_nonnegative():
    clear_time_cache()
    now = int(time.time() * 1000)
    ctx = make_ctx(GraphContext, request_id="t_deadline_budget", tenant="t", deadline_ms=now + 50)
    rem = remaining_budget_ms(ctx)
    assert rem is None or rem >= 0


async def test_deadline_exceeded_on_expired_budget():
    a = MockGraphAdapter()
    clear_time_cache()
    ctx = make_ctx(GraphContext, request_id="t_deadline_expired", tenant="t", deadline_ms=int(time.time() * 1000) - 1)
    with pytest.raises(DeadlineExceeded):
        await a.query(dialect="cypher", text="RETURN 1", ctx=ctx)


async def test_preflight_deadline_check():
    a = MockGraphAdapter()
    clear_time_cache()
    ctx = make_ctx(GraphContext, request_id="t_deadline_preflight", tenant="t", deadline_ms=int(time.time() * 1000) - 100)
    with pytest.raises(DeadlineExceeded):
        await a.create_vertex("User", {"x": 1}, ctx=ctx)


async def test_stream_respects_deadline_mid_operation():
    a = MockGraphAdapter()
    clear_time_cache()
    now = int(time.time() * 1000)
    ctx = make_ctx(GraphContext, request_id="t_deadline_stream", tenant="t", deadline_ms=now + 1)
    await asyncio.sleep(0.01)  # burn budget
    with pytest.raises(DeadlineExceeded):
        async for _ in a.stream_query(dialect="cypher", text="MATCH (n) RETURN n LIMIT 50", ctx=ctx):
            pass
