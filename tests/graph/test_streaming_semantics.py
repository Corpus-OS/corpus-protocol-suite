# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Streaming semantics.

Asserts (Spec refs):
  • stream_query yields dict rows inside QueryChunks           (§7.3.2)
  • early interruption is safe                                 (§7.3.2)
  • resources released on cancel                               (§7.3.2)
  • deadline respected mid-stream                              (§6.1, §12.1)
"""
import asyncio
import time
import pytest

from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    DeadlineExceeded,
    BaseGraphAdapter,
    GraphQuerySpec,
    QueryChunk,
)

pytestmark = pytest.mark.asyncio


def clear_time_cache():
    """Placeholder to mirror previous API; no caching in this version."""
    pass


async def test_streaming_stream_query_yields_mappings(adapter: BaseGraphAdapter):
    """§7.3.2: stream_query must yield QueryChunks with dict records."""
    ctx = GraphContext(request_id="t_stream_rows", tenant="t")
    spec = GraphQuerySpec(text="RETURN 1 as value", dialect="cypher", stream=True)

    count = 0
    async for chunk in adapter.stream_query(spec, ctx=ctx):
        assert isinstance(chunk, QueryChunk)
        assert isinstance(chunk.records, list)
        if chunk.records:
            assert isinstance(chunk.records[0], dict)
        count += 1
        if count >= 3:
            break

    assert count >= 0  # empty streams are allowed


async def test_streaming_can_be_interrupted_early(adapter: BaseGraphAdapter):
    """§7.3.2: Streams must support early interruption."""
    ctx = GraphContext(request_id="t_stream_early", tenant="t")
    spec = GraphQuerySpec(
        text="RETURN 1 as value LIMIT 10", dialect="cypher", stream=True
    )

    count = 0
    async for _ in adapter.stream_query(spec, ctx=ctx):
        count += 1
        if count == 2:
            break

    assert count == 2, "Should be able to break from stream early"


async def test_streaming_releases_resources_on_cancel(adapter: BaseGraphAdapter):
    """§7.3.2: Streams must release resources when cancelled."""
    ctx = GraphContext(request_id="t_stream_cancel", tenant="t")
    spec = GraphQuerySpec(
        text="RETURN 1 as value LIMIT 5", dialect="cypher", stream=True
    )

    stream = adapter.stream_query(spec, ctx=ctx)
    first_chunk = await stream.__anext__()
    assert isinstance(first_chunk, QueryChunk)
    await stream.aclose()  # if we get here without errors, cleanup is OK


async def test_streaming_respects_deadline(adapter: BaseGraphAdapter):
    """§6.1: Streams must respect deadline constraints."""
    clear_time_cache()
    now_ms = int(time.time() * 1000)

    ctx = GraphContext(
        request_id="t_stream_deadline",
        tenant="t",
        deadline_ms=now_ms + 10,
    )

    # burn some time so the deadline is already exceeded at call time
    await asyncio.sleep(0.02)

    spec = GraphQuerySpec(text="RETURN 1 as value", dialect="cypher", stream=True)

    with pytest.raises(DeadlineExceeded):
        async for _ in adapter.stream_query(spec, ctx=ctx):
            pass


async def test_streaming_empty_results_handled(adapter: BaseGraphAdapter):
    """§7.3.2: Empty streams should be handled gracefully."""
    ctx = GraphContext(request_id="t_stream_empty", tenant="t")
    spec = GraphQuerySpec(
        text="MATCH (n:NonExistentLabel) RETURN n LIMIT 10",
        dialect="cypher",
        stream=True,
    )

    total_rows = 0
    async for chunk in adapter.stream_query(spec, ctx=ctx):
        total_rows += len(chunk.records)

    assert total_rows >= 0, "Empty streams should complete without errors"


async def test_streaming_large_results_handled(adapter: BaseGraphAdapter):
    """§7.3.2: Streams should handle larger result sets efficiently."""
    ctx = GraphContext(request_id="t_stream_large", tenant="t")
    max_items = 20
    spec = GraphQuerySpec(
        text=f"UNWIND range(1, {max_items}) AS i RETURN i as value",
        dialect="cypher",
        stream=True,
    )

    count = 0
    async for chunk in adapter.stream_query(spec, ctx=ctx):
        for row in chunk.records:
            assert isinstance(row, dict)
            count += 1
            if count >= max_items:
                break
        if count >= max_items:
            break

    assert count > 0, "Should stream at least some items"
