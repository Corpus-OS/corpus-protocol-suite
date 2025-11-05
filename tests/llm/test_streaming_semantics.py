# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Streaming semantics.

Covers:
  • Stream yields multiple chunks
  • Exactly one terminal chunk with is_final=True
  • Terminal chunk appears last
  • usage_so_far is monotonically non-decreasing (when present)
  • Aggregate text is non-empty and includes all non-final pieces
"""

import pytest

from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.llm.llm_base import OperationContext, LLMChunk
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_stream_has_single_final_chunk_and_progress_usage():
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
