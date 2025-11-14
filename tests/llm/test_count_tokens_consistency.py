# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Token counting consistency.
Asserts:
  • count_tokens returns non-negative ints
  • Longer texts yield counts >= shorter texts (monotonic heuristic)
  • Edge cases (empty, unicode) handled gracefully
"""
import pytest
from corpus_sdk.llm.llm_base import OperationContext
from examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_token_counting_count_tokens_monotonic(adapter):
    """
    SPECIFICATION.md §8.3 — Token Counting

    Longer texts SHOULD yield higher or equal token counts (monotonic property).
    """
    caps = await adapter.capabilities()
    if not caps.supports_count_tokens:
        pytest.skip("Adapter does not support count_tokens")

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


async def test_token_counting_empty_string(adapter):
    """Empty string should return 0 or small overhead for special tokens."""
    caps = await adapter.capabilities()
    if not caps.supports_count_tokens:
        pytest.skip("Adapter does not support count_tokens")

    ctx = make_ctx(OperationContext, tenant="test")

    count = await adapter.count_tokens("", ctx=ctx)

    assert isinstance(count, int), "Must return integer"
    assert count >= 0, "Token count must be non-negative"
    assert count <= 10, "Empty string should have minimal token overhead"


async def test_token_counting_unicode_handling(adapter):
    """Token counting should handle unicode characters gracefully."""
    caps = await adapter.capabilities()
    if not caps.supports_count_tokens:
        pytest.skip("Adapter does not support count_tokens")

    ctx = make_ctx(OperationContext, tenant="test")

    # Mix of ASCII, CJK, and emoji
    text = "Hello 世界 🌍 مرحبا"
    count = await adapter.count_tokens(text, ctx=ctx)

    assert isinstance(count, int), "Must return integer"
    assert count > 0, "Non-empty text should have positive token count"