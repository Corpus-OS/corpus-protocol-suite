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
from examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


def supports_token_counting(adapter: BaseEmbeddingAdapter) -> bool:
    """Check token counting capability."""
    caps = adapter.capabilities
    return getattr(caps, "supports_token_counting", False)


async def test_token_counting_returns_non_negative_int(adapter: BaseEmbeddingAdapter):
    """§10.3: count_tokens must return non-negative integer."""
    if not supports_token_counting(adapter):
        pytest.skip("Adapter does not support token counting")

    ctx = make_ctx(OperationContext, request_id="t_tokens_basic", tenant="t")
    n = await adapter.count_tokens("hello world", adapter.supported_models[0], ctx=ctx)
    assert isinstance(n, int), f"Expected int, got {type(n)}"
    assert n >= 0, f"Token count must be non-negative, got {n}"


async def test_token_counting_context_propagation(adapter: BaseEmbeddingAdapter):
    """§6.1: Context must be properly propagated to token counting."""
    if not supports_token_counting(adapter):
        pytest.skip("Adapter does not support token counting")

    from unittest.mock import Mock
    mock_metrics = Mock()
    ctx = make_ctx(
        OperationContext, 
        request_id="t_tokens_ctx", 
        tenant="test-tenant",
        metrics=mock_metrics
    )

    result = await adapter.count_tokens("test context", adapter.supported_models[0], ctx=ctx)
    assert isinstance(result, int) and result >= 0
    
    # Verify metrics were called (if adapter implements observability)
    if mock_metrics.method_calls:
        assert any(call[0] in ['observe', 'counter'] for call in mock_metrics.method_calls)


async def test_token_counting_monotonic_with_text_length(adapter: BaseEmbeddingAdapter):
    """§10.3: Token count should generally increase with text length."""
    if not supports_token_counting(adapter):
        pytest.skip("Adapter does not support token counting")

    model = adapter.supported_models[0]
    texts = ["a", "a b", "a b c", "a b c d e f g h i j k l m n o p"]
    
    counts = []
    for text in texts:
        count = await adapter.count_tokens(text, model)
        assert isinstance(count, int) and count >= 0
        counts.append(count)
    
    # Verify monotonicity (longer texts should have equal or more tokens)
    for i in range(1, len(counts)):
        assert counts[i] >= counts[i-1] - 2, f"Non-monotonic: {counts[i-1]} -> {counts[i]} for texts: '{texts[i-1]}' -> '{texts[i]}'"


async def test_token_counting_empty_string_handling(adapter: BaseEmbeddingAdapter):
    """§10.3: Empty string should return minimal token count."""
    if not supports_token_counting(adapter):
        pytest.skip("Adapter does not support token counting")

    model = adapter.supported_models[0]
    count = await adapter.count_tokens("", model)
    assert isinstance(count, int)
    assert 0 <= count <= 5, f"Empty string should have minimal tokens, got {count}"


async def test_token_counting_unicode_boundary_cases(adapter: BaseEmbeddingAdapter):
    """§10.3: Token counting should handle Unicode correctly."""
    if not supports_token_counting(adapter):
        pytest.skip("Adapter does not support token counting")

    model = adapter.supported_models[0]
    test_cases = [
        "hello world",  # ASCII
        "こんにちは世界",  # Japanese
        "👋🌍✨",  # Emoji
        "mixed 文字 and 🔥 emoji",  # Mixed
        "ñáéíóú",  # Accented characters
    ]

    for text in test_cases:
        count = await adapter.count_tokens(text, model)
        assert isinstance(count, int) and count >= 0, f"Failed for: {text}"
        # Unicode text should generally have reasonable token count
        assert count <= len(text) * 4, f"Excessive tokens {count} for text length {len(text)}: '{text}'"


async def test_token_counting_consistent_for_identical_inputs(adapter: BaseEmbeddingAdapter):
    """§10.3: Identical inputs should produce identical token counts."""
    if not supports_token_counting(adapter):
        pytest.skip("Adapter does not support token counting")

    model = adapter.supported_models[0]
    text = "consistent token counting test"
    
    count1 = await adapter.count_tokens(text, model)
    count2 = await adapter.count_tokens(text, model)
    
    assert count1 == count2, f"Inconsistent counts: {count1} vs {count2}"


async def test_token_counting_unknown_model_raises_model_not_available(adapter: BaseEmbeddingAdapter):
    """§10.4: Unknown models must raise ModelNotAvailable."""
    if not supports_token_counting(adapter):
        pytest.skip("Adapter does not support token counting")

    with pytest.raises(ModelNotAvailable) as exc_info:
        await adapter.count_tokens("test text", "invalid-model-12345")
    
    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ['model', 'available', 'support', 'invalid']), \
        f"Error message should mention model issue: {error_msg}"


async def test_token_counting_invalid_input_raises_bad_request(adapter: BaseEmbeddingAdapter):
    """§10.4: Invalid inputs should raise BadRequest."""
    if not supports_token_counting(adapter):
        pytest.skip("Adapter does not support token counting")

    model = adapter.supported_models[0]
    
    # Test None text
    with pytest.raises(BadRequest):
        await adapter.count_tokens(None, model)  # type: ignore
    
    # Test extremely long text (if there's a limit)
    caps = adapter.capabilities
    if getattr(caps, 'max_text_length', None):
        long_text = "x" * (caps.max_text_length + 100)
        with pytest.raises((BadRequest, ValueError)):
            await adapter.count_tokens(long_text, model)


async def test_token_counting_not_supported_raises_clear_error(adapter: BaseEmbeddingAdapter):
    """§10.4: Unsupported token counting must raise NotSupported with clear message."""
    if supports_token_counting(adapter):
        pytest.skip("Adapter supports token counting")

    with pytest.raises(NotSupported) as exc_info:
        await adapter.count_tokens("test", adapter.supported_models[0])
    
    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ['support', 'implement', 'available', 'token']), \
        f"Error message should indicate lack of support: {error_msg}"


async def test_token_counting_various_whitespace_handling(adapter: BaseEmbeddingAdapter):
    """§10.3: Token counting should handle various whitespace patterns."""
    if not supports_token_counting(adapter):
        pytest.skip("Adapter does not support token counting")

    model = adapter.supported_models[0]
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
        assert isinstance(count, int) and count >= 0, f"Failed for whitespace case: {repr(text)}"