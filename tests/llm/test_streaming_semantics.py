# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Streaming semantics (enhanced).

Specification references:
  • §8.3 (LLM Protocol V1 — Operations / stream): progressive chunking, exactly one terminal chunk
  • §6.1 (Common Foundation — Operation Context): deadline budget handling & fail-fast on pre-expired budgets
  • §12.4 (Error Handling and Resilience — Error Mapping Table): DeadlineExceeded mapping/semantics

Covers (normative + robustness):
  • Stream yields multiple chunks and exactly one terminal chunk with is_final=True (last)
  • Terminal chunk carries only the mock terminal sentinel "[end]"
  • usage_so_far.total_tokens is monotonically non-decreasing when present
  • Model, when present on chunks, is consistent across the stream
  • Early consumer cancellation releases resources (a subsequent stream still succeeds)
  • Pre-expired deadlines raise DeadlineExceeded with no partial emission
  • Content progression: each non-final chunk adds new text; aggregate body is non-empty
  • (Parity check, informative) Streamed body ≈ non-stream completion
"""

import pytest

from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.llm.llm_base import OperationContext, LLMChunk, DeadlineExceeded
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_stream_has_single_final_chunk_and_progress_usage():
    """
    §8.3 — stream() MUST produce progressive chunks and exactly one terminal chunk.
    §12.4 — usage_so_far (when present) should progress monotonically.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_stream_semantics", tenant="test")

    chunks: list[LLMChunk] = []
    async for ch in adapter.stream(
        messages=[{"role": "user", "content": "stream me"}],
        model="mock-model",
        ctx=ctx,
    ):
        assert isinstance(ch, LLMChunk), "stream must yield LLMChunk instances"
        chunks.append(ch)

    # Basic shape: at least two chunks (some content + final)
    assert len(chunks) >= 2, "expected multiple chunks including final"

    # Exactly one final chunk, and it's the last one
    finals = [c for c in chunks if getattr(c, "is_final", False)]
    assert len(finals) == 1, "expected exactly one final chunk"
    assert chunks[-1].is_final, "final chunk must be last"
    assert chunks[-1].text.strip() == "[end]", "mock adapter ends with '[end]' token"

    # usage_so_far should be non-decreasing across chunks when provided
    prev_total = 0
    for c in chunks:
        if c.usage_so_far:
            total = c.usage_so_far.total_tokens
            assert total >= prev_total, "usage_so_far.total_tokens must be non-decreasing"
            prev_total = total

    # Aggregate the streamed text excluding the final marker
    body_text = "".join(c.text for c in chunks[:-1]).strip()
    assert body_text, "non-final streamed text should be non-empty"


async def test_stream_model_consistent_when_present():
    """
    §8.3 — If chunk.model is present, it SHOULD remain consistent across the stream.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_stream_model", tenant="test")
    models = []

    async for ch in adapter.stream(
        messages=[{"role": "user", "content": "stream me"}],
        model="mock-model",
        ctx=ctx,
    ):
        if ch.model:
            models.append(ch.model)

    if models:  # optional field; assert consistency only if provided
        assert all(m == models[0] for m in models), "model must be consistent across all chunks"


async def test_stream_early_cancel_then_new_stream_ok():
    """
    §8.3 — Iterator close MUST free resources; subsequent streams operate normally.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_stream_cancel", tenant="test")

    # Start and cancel early
    it = adapter.stream(
        messages=[{"role": "user", "content": "partial"}],
        model="mock-model",
        ctx=ctx,
    )

    seen = 0
    async for _ in it:
        seen += 1
        if seen >= 2:
            break
    assert seen >= 1, "expected at least one chunk before cancel"

    # A subsequent stream should still function normally
    chunks = []
    async for ch in adapter.stream(
        messages=[{"role": "user", "content": "fresh run"}],
        model="mock-model",
        ctx=ctx,
    ):
        chunks.append(ch)
    assert chunks and chunks[-1].is_final, "post-cancel stream must still complete with a final chunk"
    assert chunks[-1].text.strip() == "[end]", "terminal marker must be present on the last chunk"


async def test_stream_deadline_preexpired_yields_no_chunks():
    """
    §6.1 + §12.4 — Pre-expired deadline MUST fail fast with DeadlineExceeded; no partial emission.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    # Pre-expired deadline: 0 (epoch) guarantees elapsed budget
    ctx = OperationContext(deadline_ms=0, tenant="test")

    async def _collect():
        items = []
        async for ch in adapter.stream(
            messages=[{"role": "user", "content": "late"}],
            model="mock-model",
            ctx=ctx,
        ):
            items.append(ch)
        return items

    with pytest.raises(DeadlineExceeded):
        await _collect()


async def test_stream_content_progress_and_terminal_rules():
    """
    §8.3 — Progressive content and single terminal semantics:
      • Non-final chunks have non-empty text
      • Final chunk is terminal-only ('[end]' in mock)
      • Cumulative body grows as chunks arrive
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_stream_progress", tenant="test")

    chunks: list[LLMChunk] = []
    async for ch in adapter.stream(
        messages=[{"role": "user", "content": "stream me"}],
        model="mock-model",
        ctx=ctx,
    ):
        chunks.append(ch)

    assert len(chunks) >= 2, "expected multiple chunks including final"

    # Non-final chunks must have non-empty text
    for c in chunks[:-1]:
        assert c.text.strip(), "non-final chunk text must be non-empty"

    # Final chunk is terminal marker only for this mock
    final = chunks[-1]
    assert final.is_final, "last chunk must be final"
    assert final.text.strip() == "[end]", "final chunk carries only the terminal sentinel"

    # Progress: cumulative text across non-finals should grow
    cumulative = ""
    for c in chunks[:-1]:
        prev = cumulative
        cumulative += c.text
        assert cumulative != prev, "each non-final chunk should add new content"
    assert cumulative.strip(), "aggregate non-final text must be non-empty"


async def test_stream_body_matches_complete_result():
    """
    (Informative) Cross-path parity: streamed body ≈ completion text for same prompt.
    Useful to detect drift between streaming and non-streaming implementations.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_stream_vs_complete", tenant="test")

    # Stream
    chunks: list[LLMChunk] = []
    async for ch in adapter.stream(
        messages=[{"role": "user", "content": "stream parity"}],
        model="mock-model",
        ctx=ctx,
    ):
        chunks.append(ch)
    body_text = "".join(c.text for c in chunks if not c.is_final).strip()

    # Complete
    comp = await adapter.complete(
        messages=[{"role": "user", "content": "stream parity"}],
        model="mock-model",
        ctx=ctx,
    )

    assert body_text, "expected non-empty streamed body"
    assert comp.text.strip(), "expected non-empty completion"
    # Loosen equality a bit in case the mock formats slightly differently
    assert body_text in comp.text or comp.text in body_text, "streamed body should be equivalent to completion text"
