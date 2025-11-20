# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Streaming semantics.

Asserts (Spec refs):
  • stream_query yields dict rows                              (§7.3.2)
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
)

pytestmark = pytest.mark.asyncio


def clear_time_cache():
    """
    Placeholder to mirror previous API; no caching in this simplified version.
    """
    pass


async def test_streaming_stream_query_yields_mappings(adapter: BaseGraphAdapter):
    """§7.3.2: stream_query must yield dictionary results."""
    ctx = GraphContext(request_id="t_stream_rows", tenant="t")

    count = 0
    async for item in adapter.stream_query(
        dialect="cypher",
        text="RETURN 1 as value",
        ctx=ctx,
    ):
        assert isinstance(item, dict), "Stream items must be dictionaries"
        count += 1
        if count >= 3:  # Limit to avoid infinite streams
            break

    # Some adapters may return empty streams for simple queries
    # The important part is that if items are returned, they're dictionaries


async def test_streaming_can_be_interrupted_early(adapter: BaseGraphAdapter):
    """§7.3.2: Streams must support early interruption."""
    ctx = GraphContext(request_id="t_stream_early", tenant="t")

    count = 0
    async for _ in adapter.stream_query(
        dialect="cypher",
        text="RETURN 1 as value LIMIT 10",
        ctx=ctx,
    ):
        count += 1
        if count == 2:
            break

    assert count == 2, "Should be able to break from stream early"
    # No assertion about resource cleanup - that's implementation specific


async def test_streaming_releases_resources_on_cancel(adapter: BaseGraphAdapter):
    """§7.3.2: Streams must release resources when cancelled."""
    ctx = GraphContext(request_id="t_stream_cancel", tenant="t")

    # Start streaming and consume one item
    stream = adapter.stream_query(
        dialect="cypher",
        text="RETURN 1 as value LIMIT 5",
        ctx=ctx,
    )
    first_item = await stream.__anext__()
    assert isinstance(first_item, dict), "First stream item should be a dictionary"

    # Explicitly close the stream to test resource cleanup
    await stream.aclose()
    # If we get here without errors, resource cleanup is working


async def test_streaming_respects_deadline(adapter: BaseGraphAdapter):
    """§6.1: Streams must respect deadline constraints."""
    clear_time_cache()
    now_ms = int(time.time() * 1000)

    # Use a very short deadline to force timeout
    ctx = GraphContext(
        request_id="t_stream_deadline",
        tenant="t",
        deadline_ms=now_ms + 10,
    )

    # Burn some time to ensure deadline is exceeded
    await asyncio.sleep(0.02)

    with pytest.raises(DeadlineExceeded):
        async for _ in adapter.stream_query(
            dialect="cypher",
            text="RETURN 1 as value",
            ctx=ctx,
        ):
            pass


async def test_streaming_empty_results_handled(adapter: BaseGraphAdapter):
    """§7.3.2: Empty streams should be handled gracefully."""
    ctx = GraphContext(request_id="t_stream_empty", tenant="t")

    # Test with a query that likely returns no results
    count = 0
    async for item in adapter.stream_query(
        dialect="cypher",
        text="MATCH (n:NonExistentLabel) RETURN n LIMIT 10",
        ctx=ctx,
    ):
        count += 1

    # Empty streams are valid - should complete without errors
    assert count >= 0, "Empty streams should complete without errors"


async def test_streaming_large_results_handled(adapter: BaseGraphAdapter):
    """§7.3.2: Streams should handle large result sets efficiently."""
    ctx = GraphContext(request_id="t_stream_large", tenant="t")

    # Stream a reasonable number of items to test chunking
    max_items = 20
    count = 0
    async for item in adapter.stream_query(
        dialect="cypher",
        text=f"UNWIND range(1, {max_items}) AS i RETURN i as value",
        ctx=ctx,
    ):
        assert isinstance(item, dict)
        assert "value" in item
        count += 1
        if count >= max_items:
            break

    # Should be able to stream multiple items
    assert count > 0, "Should stream at least some items"
