# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Streaming semantics (enhanced).

Specification references:
  • §8.3 (LLM Protocol V1 — Operations / stream): progressive chunking, exactly one terminal chunk
  • §6.1 (Common Foundation — Operation Context): deadline budget handling & fail-fast on pre-expired budgets
  • §12.4 (Error Handling and Resilience — Error Mapping Table): DeadlineExceeded mapping/semantics

Covers:
  • Stream yields multiple chunks and exactly one terminal chunk with is_final=True (last)
  • usage_so_far.total_tokens is monotonically non-decreasing when present
  • Model, when present on chunks, is consistent across the stream
  • Early consumer cancellation releases resources (a subsequent stream still succeeds)
  • Pre-expired deadlines raise DeadlineExceeded with no partial emission when supports_deadline=True
  • Content progression: aggregate body is non-empty for normal text turns
  • (Parity check) Streamed body ≈ non-stream completion
"""

import pytest
from corpus_sdk.llm.llm_base import OperationContext, LLMChunk, DeadlineExceeded, NotSupported

pytestmark = pytest.mark.asyncio


async def test_streaming_stream_has_single_final_chunk_and_progress_usage(adapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(request_id="t_stream_semantics", tenant="test")

    if not caps.supports_streaming:
        with pytest.raises(NotSupported):
            agen = adapter.stream(messages=[{"role": "user", "content": "stream me"}], model=caps.supported_models[0], ctx=ctx)
            async for _ in agen:
                pass
        return

    chunks: list[LLMChunk] = []
    async for ch in adapter.stream(
        messages=[{"role": "user", "content": "stream me"}],
        model=caps.supported_models[0],
        ctx=ctx,
    ):
        assert isinstance(ch, LLMChunk)
        chunks.append(ch)

    assert len(chunks) >= 2, "expected multiple chunks including final"

    finals = [c for c in chunks if getattr(c, "is_final", False)]
    assert len(finals) == 1
    assert chunks[-1].is_final

    prev_total = 0
    for c in chunks:
        if c.usage_so_far:
            total = c.usage_so_far.total_tokens
            assert total >= prev_total
            prev_total = total

    body_text = "".join(c.text for c in chunks[:-1]).strip()
    assert body_text, "non-final streamed text should be non-empty"


async def test_streaming_stream_model_consistent_when_present(adapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(request_id="t_stream_model", tenant="test")

    if not caps.supports_streaming:
        with pytest.raises(NotSupported):
            agen = adapter.stream(messages=[{"role": "user", "content": "stream me"}], model=caps.supported_models[0], ctx=ctx)
            async for _ in agen:
                pass
        return

    models = []
    async for ch in adapter.stream(
        messages=[{"role": "user", "content": "stream me"}],
        model=caps.supported_models[0],
        ctx=ctx,
    ):
        if ch.model:
            models.append(ch.model)

    if models:
        assert all(m == models[0] for m in models)


async def test_streaming_stream_early_cancel_then_new_stream_ok(adapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(request_id="t_stream_cancel", tenant="test")

    if not caps.supports_streaming:
        with pytest.raises(NotSupported):
            agen = adapter.stream(messages=[{"role": "user", "content": "partial"}], model=caps.supported_models[0], ctx=ctx)
            async for _ in agen:
                pass
        return

    it = adapter.stream(messages=[{"role": "user", "content": "partial"}], model=caps.supported_models[0], ctx=ctx)
    seen = 0
    async for _ in it:
        seen += 1
        if seen >= 2:
            break
    assert seen >= 1

    chunks = []
    async for ch in adapter.stream(messages=[{"role": "user", "content": "fresh run"}], model=caps.supported_models[0], ctx=ctx):
        chunks.append(ch)
    assert chunks and chunks[-1].is_final


async def test_streaming_stream_deadline_preexpired_yields_no_chunks(adapter):
    caps = await adapter.capabilities()
    if not caps.supports_streaming:
        with pytest.raises(NotSupported):
            agen = adapter.stream(messages=[{"role": "user", "content": "late"}], model=caps.supported_models[0], ctx=OperationContext(deadline_ms=0, tenant="test"))
            async for _ in agen:
                pass
        return

    ctx = OperationContext(deadline_ms=0, tenant="test")

    if caps.supports_deadline:
        async def _collect():
            items = []
            async for ch in adapter.stream(messages=[{"role": "user", "content": "late"}], model=caps.supported_models[0], ctx=ctx):
                items.append(ch)
            return items

        with pytest.raises(DeadlineExceeded):
            await _collect()
    else:
        got_any = False
        async for _ in adapter.stream(messages=[{"role": "user", "content": "late"}], model=caps.supported_models[0], ctx=ctx):
            got_any = True
            break
        assert got_any is True


async def test_streaming_stream_content_progress_and_terminal_rules(adapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(request_id="t_stream_progress", tenant="test")

    if not caps.supports_streaming:
        with pytest.raises(NotSupported):
            agen = adapter.stream(messages=[{"role": "user", "content": "stream me"}], model=caps.supported_models[0], ctx=ctx)
            async for _ in agen:
                pass
        return

    chunks: list[LLMChunk] = []
    async for ch in adapter.stream(messages=[{"role": "user", "content": "stream me"}], model=caps.supported_models[0], ctx=ctx):
        chunks.append(ch)

    assert len(chunks) >= 2
    for c in chunks[:-1]:
        assert c.text.strip()

    final = chunks[-1]
    assert final.is_final

    cumulative = ""
    for c in chunks[:-1]:
        prev = cumulative
        cumulative += c.text
        assert cumulative != prev
    assert cumulative.strip()


async def test_streaming_stream_body_matches_complete_result(adapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(request_id="t_stream_vs_complete", tenant="test")

    if not caps.supports_streaming:
        with pytest.raises(NotSupported):
            agen = adapter.stream(messages=[{"role": "user", "content": "stream parity"}], model=caps.supported_models[0], ctx=ctx)
            async for _ in agen:
                pass
        return

    chunks: list[LLMChunk] = []
    async for ch in adapter.stream(messages=[{"role": "user", "content": "stream parity"}], model=caps.supported_models[0], ctx=ctx):
        chunks.append(ch)
    body_text = "".join(c.text for c in chunks if not c.is_final).strip()

    comp = await adapter.complete(messages=[{"role": "user", "content": "stream parity"}], model=caps.supported_models[0], ctx=ctx)

    assert body_text
    assert comp.text.strip()
    assert body_text in comp.text or comp.text in body_text
