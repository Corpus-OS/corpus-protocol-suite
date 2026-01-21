# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — count_tokens behavior.

Spec refs:
  • §10.3 count_tokens()
  • §10.4 Errors (Embedding-Specific)
  • §12 Error Handling — NotSupported, ModelNotAvailable
  • §6.1 Context Propagation
"""

import pytest

from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter,
    OperationContext,
    NotSupported,
    ModelNotAvailable,
    BadRequest,
)

pytestmark = pytest.mark.asyncio


async def test_token_counting_returns_non_negative_int(adapter: BaseEmbeddingAdapter):
    """§10.3: count_tokens must return non-negative integer OR raise NotSupported if capability says so."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    if not caps.supports_token_counting:
        with pytest.raises(NotSupported):
            await adapter.count_tokens("hello world", model)
        return

    ctx = OperationContext(request_id="t_tokens_basic", tenant="t")
    n = await adapter.count_tokens("hello world", model, ctx=ctx)
    assert isinstance(n, int), f"Expected int, got {type(n)}"
    assert n >= 0, f"Token count must be non-negative, got {n}"


async def test_token_counting_context_propagation(adapter: BaseEmbeddingAdapter):
    """§6.1: Context must be accepted and not break count_tokens behavior."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    ctx = OperationContext(
        request_id="t_tokens_ctx",
        tenant="test-tenant",
        deadline_ms=int(__import__("time").time() * 1000) + 5000,
        attrs={"any": "value"},
    )

    if not caps.supports_token_counting:
        with pytest.raises(NotSupported):
            await adapter.count_tokens("test context", model, ctx=ctx)
        return

    result = await adapter.count_tokens("test context", model, ctx=ctx)
    assert isinstance(result, int) and result >= 0


async def test_token_counting_monotonic_with_text_length(adapter: BaseEmbeddingAdapter):
    """§10.3: Token count should generally increase with text length (allow heuristic variance)."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    if not caps.supports_token_counting:
        with pytest.raises(NotSupported):
            await adapter.count_tokens("a", model)
        return

    texts = ["a", "a b", "a b c", "a b c d e f g h i j k l m n o p"]
    counts = []
    for text in texts:
        count = await adapter.count_tokens(text, model)
        assert isinstance(count, int) and count >= 0
        counts.append(count)

    for i in range(1, len(counts)):
        assert counts[i] >= max(0, counts[i - 1] - 2), (
            f"Non-monotonic: {counts[i-1]} -> {counts[i]} for texts: "
            f"'{texts[i-1]}' -> '{texts[i]}'"
        )


async def test_token_counting_empty_string_handling(adapter: BaseEmbeddingAdapter):
    """§10.3: Empty string should return minimal token count (or NotSupported if unsupported)."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    if not caps.supports_token_counting:
        with pytest.raises(NotSupported):
            await adapter.count_tokens("", model)
        return

    count = await adapter.count_tokens("", model)
    assert isinstance(count, int)
    assert 0 <= count <= 5, f"Empty string should have minimal tokens, got {count}"


async def test_token_counting_unicode_boundary_cases(adapter: BaseEmbeddingAdapter):
    """§10.3: Token counting should handle Unicode correctly (or NotSupported if unsupported)."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    if not caps.supports_token_counting:
        with pytest.raises(NotSupported):
            await adapter.count_tokens("こんにちは世界", model)
        return

    test_cases = [
        "hello world",        # ASCII
        "こんにちは世界",        # Japanese
        "👋🌍✨",                # Emoji
        "mixed 文字 and 🔥 emoji",  # Mixed
        "ñáéíóú",             # Accented characters
    ]

    for text in test_cases:
        count = await adapter.count_tokens(text, model)
        assert isinstance(count, int) and count >= 0, f"Failed for: {text}"
        assert count <= len(text) * 4, (
            f"Excessive tokens {count} for text length {len(text)}: '{text}'"
        )


async def test_token_counting_consistent_for_identical_inputs(adapter: BaseEmbeddingAdapter):
    """§10.3: Identical inputs should produce identical token counts (if supported)."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]
    text = "consistent token counting test"

    if not caps.supports_token_counting:
        with pytest.raises(NotSupported):
            await adapter.count_tokens(text, model)
        return

    count1 = await adapter.count_tokens(text, model)
    count2 = await adapter.count_tokens(text, model)
    assert count1 == count2, f"Inconsistent counts: {count1} vs {count2}"


async def test_token_counting_unknown_model_raises_model_not_available(adapter: BaseEmbeddingAdapter):
    """§10.4: Unknown models must raise ModelNotAvailable (if supported)."""
    caps = await adapter.capabilities()
    if not caps.supports_token_counting:
        with pytest.raises(NotSupported):
            await adapter.count_tokens("test text", "invalid-model-12345")
        return

    with pytest.raises(ModelNotAvailable) as exc_info:
        await adapter.count_tokens("test text", "invalid-model-12345")

    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ["model", "available", "support", "invalid"]), (
        f"Error message should mention model issue: {error_msg}"
    )


async def test_token_counting_invalid_input_raises_bad_request(adapter: BaseEmbeddingAdapter):
    """§10.4: Invalid inputs should raise BadRequest (for clearly invalid types) if supported."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    if not caps.supports_token_counting:
        with pytest.raises(NotSupported):
            await adapter.count_tokens("test", model)
        return

    with pytest.raises(BadRequest):
        await adapter.count_tokens(None, model)  # type: ignore[arg-type]


async def test_token_counting_various_whitespace_handling(adapter: BaseEmbeddingAdapter):
    """§10.3: Token counting should handle various whitespace patterns (or NotSupported)."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    if not caps.supports_token_counting:
        with pytest.raises(NotSupported):
            await adapter.count_tokens("normal spacing", model)
        return

    whitespace_cases = [
        "normal spacing",
        "  leading",
        "trailing  ",
        "multiple   spaces",
        "tabs\tin\ttext",
        "new\nlines",
        "carriage\rreturns",
    ]

    for text in whitespace_cases:
        count = await adapter.count_tokens(text, model)
        assert isinstance(count, int) and count >= 0, (
            f"Failed for whitespace case: {repr(text)}"
        )

async def test_token_counting_support_matches_capabilities(adapter: BaseEmbeddingAdapter):
    """Capability ↔ behavior consistency: supports_token_counting must match runtime behavior."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    if caps.supports_token_counting:
        n = await adapter.count_tokens("probe", model)
        assert isinstance(n, int) and n >= 0
    else:
        with pytest.raises(NotSupported):
            await adapter.count_tokens("probe", model)
