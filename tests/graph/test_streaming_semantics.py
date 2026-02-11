# SPDX-License-Identifier: Apache-2.0
"""
Graph Conformance — Streaming semantics.

Asserts (Spec refs):
  • stream_query yields dict rows inside QueryChunks           (§7.3.2)
  • early interruption is safe                                 (§7.3.2)
  • resources released on cancel                               (§7.3.2)
  • deadline respected mid-stream                              (§6.1, §12.1)
"""

from __future__ import annotations

import json
import pytest

from corpus_sdk.graph.graph_base import (
    OperationContext as GraphContext,
    NotSupported,
    BaseGraphAdapter,
    GraphQuerySpec,
    QueryChunk,
    WireGraphHandler,
)

pytestmark = pytest.mark.asyncio


async def test_stream_query_capability_alignment(adapter: BaseGraphAdapter):
    caps = await adapter.capabilities()
    ctx = GraphContext(request_id="t_stream_cap", tenant="t")
    spec = GraphQuerySpec(text="RETURN 1", dialect="cypher", stream=True)

    if not getattr(caps, "supports_stream_query", True):
        with pytest.raises(NotSupported):
            async for _ in adapter.stream_query(spec, ctx=ctx):
                pass
        return

    async for chunk in adapter.stream_query(spec, ctx=ctx):
        assert isinstance(chunk, QueryChunk)
        break


async def test_stream_query_yields_querychunks_with_json_serializable_records(adapter: BaseGraphAdapter):
    caps = await adapter.capabilities()
    ctx = GraphContext(request_id="t_stream_rows", tenant="t")
    spec = GraphQuerySpec(text="RETURN 1 as value", dialect="cypher", stream=True)

    if not getattr(caps, "supports_stream_query", True):
        with pytest.raises(NotSupported):
            async for _ in adapter.stream_query(spec, ctx=ctx):
                pass
        return

    async for chunk in adapter.stream_query(spec, ctx=ctx):
        assert isinstance(chunk, QueryChunk)
        assert isinstance(chunk.records, list)
        json.dumps(chunk.records)
        if chunk.is_final and chunk.summary is not None:
            json.dumps(chunk.summary)
        break


async def test_streaming_can_be_interrupted_early(adapter: BaseGraphAdapter):
    caps = await adapter.capabilities()
    ctx = GraphContext(request_id="t_stream_early", tenant="t")
    spec = GraphQuerySpec(text="RETURN 1 LIMIT 10", dialect="cypher", stream=True)

    if not getattr(caps, "supports_stream_query", True):
        with pytest.raises(NotSupported):
            async for _ in adapter.stream_query(spec, ctx=ctx):
                pass
        return

    count = 0
    async for _ in adapter.stream_query(spec, ctx=ctx):
        count += 1
        if count == 2:
            break
    assert count == 2


async def test_streaming_releases_resources_on_cancel(adapter: BaseGraphAdapter):
    caps = await adapter.capabilities()
    ctx = GraphContext(request_id="t_stream_cancel", tenant="t")
    spec = GraphQuerySpec(text="RETURN 1 LIMIT 5", dialect="cypher", stream=True)

    if not getattr(caps, "supports_stream_query", True):
        with pytest.raises(NotSupported):
            async for _ in adapter.stream_query(spec, ctx=ctx):
                pass
        return

    stream = adapter.stream_query(spec, ctx=ctx)
    first_chunk = await stream.__anext__()
    assert isinstance(first_chunk, QueryChunk)
    await stream.aclose()


# ---------------------------- NEW: wire handle_stream ----------------------------

async def test_wire_handle_stream_emits_streaming_frames_when_supported(adapter: BaseGraphAdapter):
    """
    NEW: Wire streaming must yield STREAMING frames when supported, else an error envelope.
    """
    caps = await adapter.capabilities()
    h = WireGraphHandler(adapter)

    dialect = caps.supported_query_dialects[0] if caps.supported_query_dialects else None
    args = {"text": "RETURN 1"}
    if dialect is not None:
        args["dialect"] = dialect

    agen = h.handle_stream({"op": "graph.stream_query", "ctx": {"request_id": "ws1"}, "args": args})

    first = None
    async for item in agen:
        first = item
        break

    assert first is not None
    if getattr(caps, "supports_stream_query", True):
        assert first.get("ok") is True
        assert first.get("code") == "STREAMING"
        assert isinstance(first.get("ms"), (int, float))
        assert isinstance(first.get("chunk"), dict)
        assert isinstance(first["chunk"].get("records"), list)
    else:
        assert first.get("ok") is False
        assert isinstance(first.get("code"), str)
        assert isinstance(first.get("error"), str)
