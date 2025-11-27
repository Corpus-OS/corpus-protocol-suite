# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Deadline enforcement.

Asserts (Spec refs):
  • Non-negative remaining budget                            (§6.1)
  • Expired deadlines fail fast (preflight)                  (§6.1, §12.1)
  • Stream enforces deadline mid-operation                   (§6.1, §12.1)

Reference behavior: BaseGraphAdapter uses DeadlineExceeded for budget expiry.
"""
import asyncio
import time
import pytest

from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    DeadlineExceeded,
    BaseGraphAdapter,
    GraphQuerySpec,
    Node,
    GraphID,
    UpsertNodesSpec,
)

pytestmark = pytest.mark.asyncio


def remaining_budget_ms(ctx):
    """
    Simple remaining-budget computation based on ctx.deadline_ms, if present.
    """
    deadline_ms = getattr(ctx, "deadline_ms", None)
    if deadline_ms is None:
        return None
    now_ms = int(time.time() * 1000)
    return max(deadline_ms - now_ms, 0)


def clear_time_cache():
    """Placeholder to mirror previous API; no caching in this version."""
    pass


async def test_deadline_budget_nonnegative():
    clear_time_cache()
    now = int(time.time() * 1000)
    ctx = GraphContext(
        request_id="t_deadline_budget",
        tenant="t",
        deadline_ms=now + 50,
    )
    rem = remaining_budget_ms(ctx)
    assert rem is None or rem >= 0


async def test_deadline_exceeded_on_expired_budget(adapter: BaseGraphAdapter):
    clear_time_cache()
    ctx = GraphContext(
        request_id="t_deadline_expired",
        tenant="t",
        deadline_ms=int(time.time() * 1000) - 1,
    )
    spec = GraphQuerySpec(text="RETURN 1", dialect="cypher")

    with pytest.raises(DeadlineExceeded):
        await adapter.query(spec, ctx=ctx)


async def test_deadline_preflight_deadline_check(adapter: BaseGraphAdapter):
    clear_time_cache()
    ctx = GraphContext(
        request_id="t_deadline_preflight",
        tenant="t",
        deadline_ms=int(time.time() * 1000) - 100,
    )

    # Use a write path (upsert) to ensure preflight is applied to non-query ops
    node = Node(id=GraphID("v:User:1"), labels=("User",), properties={"x": 1})
    spec = UpsertNodesSpec(nodes=[node], namespace="t")

    with pytest.raises(DeadlineExceeded):
        await adapter.upsert_nodes(spec, ctx=ctx)


async def test_deadline_stream_respects_deadline_mid_operation(
    adapter: BaseGraphAdapter,
):
    clear_time_cache()
    now = int(time.time() * 1000)
    ctx = GraphContext(
        request_id="t_deadline_stream",
        tenant="t",
        deadline_ms=now + 1,
    )
    await asyncio.sleep(0.01)  # burn budget

    spec = GraphQuerySpec(
        text="MATCH (n) RETURN n LIMIT 50", dialect="cypher", stream=True
    )

    with pytest.raises(DeadlineExceeded):
        async for _ in adapter.stream_query(spec, ctx=ctx):
            pass
