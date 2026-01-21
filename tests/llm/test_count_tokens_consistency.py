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

# Constants for token counting validation
MAX_EMPTY_STRING_TOKENS = 100  # Conservative upper bound for empty-string overhead
MIN_NONEMPTY_TOKENS = 1        # Minimum tokens for non-empty text


async def test_token_counting_count_tokens_monotonic(adapter):
    """
    SPECIFICATION.md §8.3 — Token Counting

    Longer texts SHOULD yield higher or equal token counts (monotonic property).
    """
    caps = await adapter.capabilities()
    ctx = OperationContext(request_id="t_count_tokens", tenant="test")

    if not caps.supports_count_tokens:
        with pytest.raises(NotSupported):
            await adapter.count_tokens("short", ctx=ctx)
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
        count = await adapter.count_tokens(text, ctx=ctx)
        assert isinstance(count, int), f"count_tokens must return integer for '{text}'"
        assert count >= 0, f"Token count must be non-negative for '{text}'"
        counts.append(count)

    for i in range(1, len(counts)):
        assert counts[i] >= counts[i - 1], (
            f"Longer text '{texts[i]}' should not have fewer tokens than '{texts[i-1]}'"
        )


async def test_token_counting_empty_string(adapter):
    """Empty string should return 0 or small overhead for special tokens."""
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test")

    if not caps.supports_count_tokens:
        with pytest.raises(NotSupported):
            await adapter.count_tokens("", ctx=ctx)
        return

    count = await adapter.count_tokens("", ctx=ctx)

    assert isinstance(count, int), "Must return integer for empty string"
    assert count >= 0, "Token count must be non-negative for empty string"
    assert count <= MAX_EMPTY_STRING_TOKENS, (
        f"Empty string should have at most {MAX_EMPTY_STRING_TOKENS} tokens overhead"
    )


async def test_token_counting_unicode_handling(adapter):
    """Token counting should handle unicode characters gracefully."""
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test")

    if not caps.supports_count_tokens:
        with pytest.raises(NotSupported):
            await adapter.count_tokens("Hello 世界 🌍 مرحبا", ctx=ctx)
        return

    test_cases = [
        "Hello 世界 🌍 مرحبا",
        "🎉庆祝🎊",
        "café naïve façade",
        "🦄🐉🎯",
        "Hello 世界 🦄 café",
    ]

    for text in test_cases:
        count = await adapter.count_tokens(text, ctx=ctx)
        assert isinstance(count, int), f"Must return integer for unicode text: {text}"
        assert count >= MIN_NONEMPTY_TOKENS, (
            f"Non-empty unicode text should have at least {MIN_NONEMPTY_TOKENS} token: {text}"
        )


async def test_token_counting_whitespace_variations(adapter):
    """Token counting should handle different whitespace patterns."""
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test")

    if not caps.supports_count_tokens:
        with pytest.raises(NotSupported):
            await adapter.count_tokens("hello world", ctx=ctx)
        return

    test_cases = [
        ("normal spacing", "hello world"),
        ("multiple spaces", "hello   world"),
        ("tabs", "hello\tworld"),
        ("newlines", "hello\nworld"),
        ("mixed whitespace", "hello \t\n world"),
    ]

    for description, text in test_cases:
        count = await adapter.count_tokens(text, ctx=ctx)
        assert isinstance(count, int), f"Must return integer for {description}"
        assert count >= MIN_NONEMPTY_TOKENS, f"Non-empty text with {description} should have tokens"


async def test_token_counting_consistent_for_identical_inputs(adapter):
    """Token counting should be consistent for identical inputs (no randomness)."""
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test")
    text = "consistent token counting test"

    if not caps.supports_count_tokens:
        with pytest.raises(NotSupported):
            await adapter.count_tokens(text, ctx=ctx)
        return

    counts = []
    for _ in range(5):
        count = await adapter.count_tokens(text, ctx=ctx)
        counts.append(count)

    assert len(set(counts)) == 1, f"Token counts for identical input should be consistent, got: {counts}"


async def test_token_counting_model_gate_enforced_when_listed(adapter):
    """
    If capabilities enumerate supported_models, passing an unknown model MUST raise BadRequest.
    (Only applicable if count_tokens is supported.)
    """
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test", request_id="t_count_tokens_model_gate")

    if not caps.supports_count_tokens:
        with pytest.raises(NotSupported):
            await adapter.count_tokens("x", model="__no_such_model__", ctx=ctx)
        return

    if caps.supported_models:
        with pytest.raises(BadRequest):
            await adapter.count_tokens("x", model="__no_such_model__", ctx=ctx)
