# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Token counting consistency.
Asserts:
  • count_tokens returns non-negative ints
  • Longer texts yield counts >= shorter texts (monotonic heuristic)
  • Edge cases (empty, unicode) handled gracefully
"""
import pytest
from corpus_sdk.llm.llm_base import OperationContext, NotSupported, BadRequest

pytestmark = pytest.mark.asyncio

MAX_EMPTY_STRING_TOKENS = 100
MIN_NONEMPTY_TOKENS = 1


async def test_token_counting_count_tokens_monotonic(adapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(request_id="t_count_tokens", tenant="test")

    if not caps.supports_count_tokens:
        with pytest.raises(NotSupported):
            await adapter.count_tokens("short", model=caps.supported_models[0], ctx=ctx)
        return

    texts = [
        "short",
        "short text",
        "short text plus",
        "short text plus some",
        "short text plus some more words here",
    ]

    counts = []
    for text in texts:
        count = await adapter.count_tokens(text, model=caps.supported_models[0], ctx=ctx)
        assert isinstance(count, int)
        assert count >= 0
        counts.append(count)

    for i in range(1, len(counts)):
        assert counts[i] >= counts[i - 1]


async def test_token_counting_empty_string(adapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test")

    if not caps.supports_count_tokens:
        with pytest.raises(NotSupported):
            await adapter.count_tokens("", model=caps.supported_models[0], ctx=ctx)
        return

    count = await adapter.count_tokens("", model=caps.supported_models[0], ctx=ctx)
    assert isinstance(count, int)
    assert count >= 0
    assert count <= MAX_EMPTY_STRING_TOKENS


async def test_token_counting_unicode_handling(adapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test")

    if not caps.supports_count_tokens:
        with pytest.raises(NotSupported):
            await adapter.count_tokens("Hello 世界 🌍 مرحبا", model=caps.supported_models[0], ctx=ctx)
        return

    for text in ["Hello 世界 🌍 مرحبا", "🎉庆祝🎊", "café naïve façade", "🦄🐉🎯", "Hello 世界 🦄 café"]:
        count = await adapter.count_tokens(text, model=caps.supported_models[0], ctx=ctx)
        assert isinstance(count, int)
        assert count >= MIN_NONEMPTY_TOKENS


async def test_token_counting_whitespace_variations(adapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test")

    if not caps.supports_count_tokens:
        with pytest.raises(NotSupported):
            await adapter.count_tokens("hello world", model=caps.supported_models[0], ctx=ctx)
        return

    test_cases = ["hello world", "hello   world", "hello\tworld", "hello\nworld", "hello \t\n world"]
    for text in test_cases:
        count = await adapter.count_tokens(text, model=caps.supported_models[0], ctx=ctx)
        assert isinstance(count, int)
        assert count >= MIN_NONEMPTY_TOKENS


async def test_token_counting_consistent_for_identical_inputs(adapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test")
    text = "consistent token counting test"

    if not caps.supports_count_tokens:
        with pytest.raises(NotSupported):
            await adapter.count_tokens(text, model=caps.supported_models[0], ctx=ctx)
        return

    counts = [await adapter.count_tokens(text, model=caps.supported_models[0], ctx=ctx) for _ in range(5)]
    assert len(set(counts)) == 1


async def test_token_counting_not_supported_raises_notsupported(adapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test")

    if caps.supports_count_tokens:
        n = await adapter.count_tokens("x", model=caps.supported_models[0], ctx=ctx)
        assert isinstance(n, int) and n >= 0
    else:
        with pytest.raises(NotSupported):
            await adapter.count_tokens("x", model=caps.supported_models[0], ctx=ctx)


async def test_token_counting_model_gate_enforced_when_listed(adapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test", request_id="t_count_tokens_model_gate")

    if not caps.supports_count_tokens:
        with pytest.raises(NotSupported):
            await adapter.count_tokens("x", model="__no_such_model__", ctx=ctx)
        return

    if caps.supported_models:
        with pytest.raises(BadRequest):
            await adapter.count_tokens("x", model="__no_such_model__", ctx=ctx)


async def test_token_counting_respects_context_limits(adapter):
    """
    Token counting should return an int and never be negative; it is not required to be <= max_context_length.
    """
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test")

    if not caps.supports_count_tokens:
        with pytest.raises(NotSupported):
            await adapter.count_tokens("This is a reasonable length text.", model=caps.supported_models[0], ctx=ctx)
        return

    n = await adapter.count_tokens("This is a reasonable length text. " * 10, model=caps.supported_models[0], ctx=ctx)
    assert isinstance(n, int)
    assert n >= 0
