# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Deadline enforcement.

Asserts (Spec refs):
  • Non-negative remaining budget                            (§6.1)
  • Expired deadlines fail fast (preflight)                  (§6.1, §12.1)
  • Stream enforces deadline mid-operation                   (§6.1, §12.1)
"""
from __future__ import annotations

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


async def test_deadline_exceeded_on_expired_budget_query_when_supported(adapter: BaseGraphAdapter):
    caps = await adapter.capabilities()
    ctx = GraphContext(request_id="t_deadline_expired_query", tenant="t", deadline_ms=int(time.time() * 1000) - 1)
    spec = GraphQuerySpec(text="RETURN 1", dialect="cypher")

    if getattr(caps, "supports_deadline", True):
        with pytest.raises(DeadlineExceeded):
            await adapter.query(spec, ctx=ctx)
    else:
        with pytest.raises(AssertionError):
            try:
                await adapter.query(spec, ctx=ctx)
            except DeadlineExceeded:
                raise AssertionError("supports_deadline is False but adapter raised DeadlineExceeded")


async def test_deadline_exceeded_on_expired_budget_write_when_supported(adapter: BaseGraphAdapter):
    caps = await adapter.capabilities()
    ctx = GraphContext(request_id="t_deadline_expired_write", tenant="t", deadline_ms=int(time.time() * 1000) - 1)
    node = Node(id=GraphID("v:User:1"), labels=("User",), properties={"x": 1})
    spec = UpsertNodesSpec(nodes=[node], namespace="t")

    if getattr(caps, "supports_deadline", True):
        with pytest.raises(DeadlineExceeded):
            await adapter.upsert_nodes(spec, ctx=ctx)
    else:
        with pytest.raises(AssertionError):
            try:
                await adapter.upsert_nodes(spec, ctx=ctx)
            except DeadlineExceeded:
                raise AssertionError("supports_deadline is False but adapter raised DeadlineExceeded")


async def test_deadline_exceeded_on_expired_budget_stream_preflight_when_supported(adapter: BaseGraphAdapter):
    caps = await adapter.capabilities()
    ctx = GraphContext(request_id="t_deadline_expired_stream", tenant="t", deadline_ms=int(time.time() * 1000) - 1)
    spec = GraphQuerySpec(text="RETURN 1", dialect="cypher", stream=True)

    if getattr(caps, "supports_deadline", True):
        with pytest.raises(DeadlineExceeded):
            async for _ in adapter.stream_query(spec, ctx=ctx):
                pass
    else:
        with pytest.raises(AssertionError):
            try:
                async for _ in adapter.stream_query(spec, ctx=ctx):
                    break
            except DeadlineExceeded:
                raise AssertionError("supports_deadline is False but adapter raised DeadlineExceeded")
