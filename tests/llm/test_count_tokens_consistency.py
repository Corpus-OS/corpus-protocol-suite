# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Token counting consistency.
Asserts:
  • count_tokens returns non-negative ints
  • Longer texts yield counts >= shorter texts (monotonic heuristic)
  • Edge cases (empty, unicode) handled gracefully
  • Token counting respects context limits
"""
import pytest
from corpus_sdk.llm.llm_base import OperationContext

pytestmark = pytest.mark.asyncio

# Constants for token counting validation
MAX_EMPTY_STRING_TOKENS = 10  # Reasonable upper bound for empty string overhead
MIN_NONEMPTY_TOKENS = 1       # Minimum tokens for non-empty text


async def test_token_counting_count_tokens_monotonic(adapter):
    """
    SPECIFICATION.md §8.3 — Token Counting

    Longer texts SHOULD yield higher or equal token counts (monotonic property).
    """
    caps = await adapter.capabilities()
    if not caps.supports_count_tokens:
        pytest.skip("Adapter does not support count_tokens")

    ctx = OperationContext(request_id="t_count_tokens", tenant="test")

    # Test text progression for monotonicity
    texts = [
        "short",
        "short text", 
        "short text plus",
        "short text plus some",
        "short text plus some more words here"
    ]

    counts = []
    for text in texts:
        count = await adapter.count_tokens(text, ctx=ctx)
        assert isinstance(count, int), f"count_tokens must return integer for '{text}'"
        assert count >= 0, f"Token count must be non-negative for '{text}'"
        counts.append(count)

    # Verify monotonic progression (each text should have >= tokens than previous)
    for i in range(1, len(counts)):
        assert counts[i] >= counts[i-1], \
            f"Longer text '{texts[i]}' should not have fewer tokens than '{texts[i-1]}'"


async def test_token_counting_empty_string(adapter):
    """Empty string should return 0 or small overhead for special tokens."""
    caps = await adapter.capabilities()
    if not caps.supports_count_tokens:
        pytest.skip("Adapter does not support count_tokens")

    ctx = OperationContext(tenant="test")

    count = await adapter.count_tokens("", ctx=ctx)

    assert isinstance(count, int), "Must return integer for empty string"
    assert count >= 0, "Token count must be non-negative for empty string"
    assert count <= MAX_EMPTY_STRING_TOKENS, \
        f"Empty string should have at most {MAX_EMPTY_STRING_TOKENS} tokens for special tokens overhead"


async def test_token_counting_unicode_handling(adapter):
    """Token counting should handle unicode characters gracefully."""
    caps = await adapter.capabilities()
    if not caps.supports_count_tokens:
        pytest.skip("Adapter does not support count_tokens")

    ctx = OperationContext(tenant="test")

    # Test various unicode scenarios
    test_cases = [
        "Hello 世界 🌍 مرحبا",           # Mixed scripts + emoji
        "🎉庆祝🎊",                       # Emoji only  
        "café naïve façade",             # Latin with accents
        "🦄🐉🎯",                         # Multiple emoji
        "Hello 世界 🦄 café",             # Combined
    ]

    for text in test_cases:
        count = await adapter.count_tokens(text, ctx=ctx)
        assert isinstance(count, int), f"Must return integer for unicode text: {text}"
        assert count >= MIN_NONEMPTY_TOKENS, \
            f"Non-empty unicode text should have at least {MIN_NONEMPTY_TOKENS} token: {text}"


async def test_token_counting_whitespace_variations(adapter):
    """Token counting should handle different whitespace patterns."""
    caps = await adapter.capabilities()
    if not caps.supports_count_tokens:
        pytest.skip("Adapter does not support count_tokens")

    ctx = OperationContext(tenant="test")

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
        assert count >= MIN_NONEMPTY_TOKENS, \
            f"Non-empty text with {description} should have tokens"


async def test_token_counting_consistent_for_identical_inputs(adapter):
    """Token counting should be consistent for identical inputs."""
    caps = await adapter.capabilities()
    if not caps.supports_count_tokens:
        pytest.skip("Adapter does not support count_tokens")

    ctx = OperationContext(tenant="test")
    text = "consistent token counting test"

    # Count same text multiple times
    counts = []
    for i in range(5):
        count = await adapter.count_tokens(text, ctx=ctx)
        counts.append(count)

    # All counts should be identical
    assert len(set(counts)) == 1, \
        f"Token counts for identical input should be consistent, got: {counts}"


async def test_token_counting_respects_context_limits(adapter):
    """Token counting should work within context limits."""
    caps = await adapter.capabilities()
    if not caps.supports_count_tokens:
        pytest.skip("Adapter does not support count_tokens")

    ctx = OperationContext(tenant="test")

    # Create text that's well within typical context limits
    reasonable_text = "This is a reasonable length text for token counting. " * 10

    count = await adapter.count_tokens(reasonable_text, ctx=ctx)
    assert isinstance(count, int), "Must return integer for reasonable length text"
    assert count >= MIN_NONEMPTY_TOKENS, "Reasonable text should have tokens"
    assert count <= caps.max_context_length, \
        f"Token count should not exceed max_context_length ({caps.max_context_length})"