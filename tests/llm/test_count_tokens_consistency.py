# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Token counting consistency.
Asserts:
  • count_tokens returns non-negative ints
  • Longer texts yield counts >= shorter texts (monotonic heuristic)
  • Edge cases (empty, unicode) handled gracefully
"""
import pytest
from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.llm.llm_base import OperationContext
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_count_tokens_monotonic():
    """
    SPECIFICATION.md §8.3 — Token Counting
    
    Longer texts SHOULD yield higher or equal token counts (monotonic property).
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_count_tokens", tenant="test")
    
    s1 = "short text"
    s2 = "short text plus some more words here"
    
    c1 = await adapter.count_tokens(s1, ctx=ctx)
    c2 = await adapter.count_tokens(s2, ctx=ctx)
    
    assert isinstance(c1, int) and isinstance(c2, int), \
        "count_tokens must return integers"
    assert c1 >= 0 and c2 >= 0, \
        "Token counts must be non-negative"
    assert c2 >= c1, \
        "Longer text should not return fewer tokens"


async def test_count_tokens_empty_string():
    """Empty string should return 0 or small overhead for special tokens."""
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, tenant="test")
    
    count = await adapter.count_tokens("", ctx=ctx)
    
    assert isinstance(count, int), "Must return integer"
    assert count >= 0, "Token count must be non-negative"
    assert count <= 10, "Empty string should have minimal token overhead"


async def test_count_tokens_unicode_handling():
    """Token counting should handle unicode characters gracefully."""
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, tenant="test")
    
    # Mix of ASCII, CJK, and emoji
    text = "Hello 世界 🌍 مرحبا"
    count = await adapter.count_tokens(text, ctx=ctx)
    
    assert isinstance(count, int), "Must return integer"
    assert count > 0, "Non-empty text should have positive token count"
