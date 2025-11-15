# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Single embed behavior.

Spec refs:
  • §10.3 embed() — basic contract
  • §10.4 Errors (Embedding-Specific) 
  • §10.5 Capabilities Discovery
  • §12 Error Handling — BadRequest, ModelNotAvailable, TextTooLong
  • §10.6 Normalization & Truncation Semantics
"""

import math
import pytest

from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter,
    EmbedSpec,
    OperationContext,
    BadRequest,
    ModelNotAvailable,
    TextTooLong,
    NotSupported,
)
from examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_core_ops_embed_returns_valid_embedding_structure(adapter: BaseEmbeddingAdapter):
    """§10.3: embed() must return valid EmbedResult with correct structure."""
    ctx = make_ctx(OperationContext, request_id="t_embed_structure", tenant="test")
    spec = EmbedSpec(text="hello world", model=adapter.supported_models[0])
    res = await adapter.embed(spec, ctx=ctx)

    # Validate embedding vector
    assert res.embedding.vector, "Embedding vector cannot be empty"
    assert isinstance(res.embedding.vector, list), "Vector must be a list"
    assert all(isinstance(v, (int, float)) for v in res.embedding.vector), "Vector must contain numbers"
    
    # Validate dimensions match vector length
    assert res.embedding.dimensions == len(res.embedding.vector), "Dimensions must match vector length"
    assert res.embedding.dimensions > 0, "Dimensions must be positive"
    
    # Validate text preservation
    assert res.embedding.text == "hello world", "Original text must be preserved"
    
    # Validate model information
    assert res.model == spec.model, "Result model must match spec model"
    
    # Validate optional fields
    if res.truncated is not None:
        assert isinstance(res.truncated, bool), "truncated must be boolean if present"


async def test_core_ops_embed_requires_valid_text(adapter: BaseEmbeddingAdapter):
    """§10.4: embed() must validate text input and raise BadRequest for invalid values."""
    model = adapter.supported_models[0]
    
    invalid_cases = [
        ("", "empty text"),
        (None, "null text"),  # type: ignore
        ("   ", "whitespace-only text"),
    ]
    
    for text, description in invalid_cases:
        spec = EmbedSpec(text=text, model=model)
        with pytest.raises(BadRequest) as exc_info:
            await adapter.embed(spec)
        
        error_msg = str(exc_info.value).lower()
        assert any(term in error_msg for term in ['text', 'input', 'invalid', 'empty']), \
            f"BadRequest should mention text issue for {description}: {error_msg}"


async def test_core_ops_embed_requires_valid_model(adapter: BaseEmbeddingAdapter):
    """§10.4: embed() must validate model parameter."""
    invalid_models = [
        "",
        "invalid-model-12345",
        "unknown/model/name",
    ]
    
    for model in invalid_models:
        spec = EmbedSpec(text="test", model=model)
        with pytest.raises((BadRequest, ModelNotAvailable)):
            await adapter.embed(spec)


async def test_core_ops_embed_unknown_model_clear_error(adapter: BaseEmbeddingAdapter):
    """§10.4: Unknown models must raise ModelNotAvailable with clear message."""
    spec = EmbedSpec(text="hello", model="nonexistent-model-123")
    
    with pytest.raises(ModelNotAvailable) as exc_info:
        await adapter.embed(spec)
    
    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ['model', 'available', 'support', 'unknown']), \
        f"Error should mention model issue: {error_msg}"


async def test_core_ops_embed_truncation_behavior_matches_capabilities(adapter: BaseEmbeddingAdapter):
    """§10.6: Truncation behavior must match declared capabilities."""
    caps = await adapter.capabilities()
    
    if caps.max_text_length:
        # Create text that exceeds max length
        long_text = "x" * (caps.max_text_length + 100)
        
        # Test with truncation enabled
        spec_truncate = EmbedSpec(
            text=long_text, 
            model=caps.supported_models[0], 
            truncate=True
        )
        
        if caps.truncation_mode != "none":
            # Should succeed with truncation
            result = await adapter.embed(spec_truncate)
            assert result is not None
            if result.truncated is not None:
                assert result.truncated == True
            # Result text might be truncated
            assert len(result.embedding.text) <= len(long_text)
        else:
            # Should fail if truncation not supported
            with pytest.raises((TextTooLong, NotSupported)):
                await adapter.embed(spec_truncate)
        
        # Test with truncation disabled
        spec_no_truncate = EmbedSpec(
            text=long_text,
            model=caps.supported_models[0],
            truncate=False
        )
        
        with pytest.raises(TextTooLong):
            await adapter.embed(spec_no_truncate)


async def test_core_ops_embed_normalization_produces_unit_vectors(adapter: BaseEmbeddingAdapter):
    """§10.6: Normalization should produce vectors with unit length."""
    caps = await adapter.capabilities()
    
    if not caps.supports_normalization:
        pytest.skip("Adapter does not support normalization")
    
    test_texts = [
        "normalize this text",
        "short",
        "a longer piece of text that should be normalized properly",
    ]
    
    for text in test_texts:
        spec = EmbedSpec(text=text, model=caps.supported_models[0], normalize=True)
        result = await adapter.embed(spec)
        
        # Calculate vector norm
        vector = result.embedding.vector
        norm = math.sqrt(sum(v * v for v in vector))
        
        # Should be approximately unit length (allow small floating point errors)
        assert 0.99 <= norm <= 1.01, f"Normalized vector should have unit length, got {norm} for text: '{text}'"


async def test_core_ops_embed_normalization_unsupported_raises_clear_error(adapter: BaseEmbeddingAdapter):
    """§10.4: Normalization requests must raise clear error when unsupported."""
    caps = await adapter.capabilities()
    
    if caps.supports_normalization:
        pytest.skip("Adapter supports normalization")
    
    spec = EmbedSpec(text="test", model=caps.supported_models[0], normalize=True)
    
    with pytest.raises(NotSupported) as exc_info:
        await adapter.embed(spec)
    
    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ['normaliz', 'support', 'implement']), \
        f"Error should mention normalization: {error_msg}"


async def test_core_ops_embed_vector_quality_and_consistency(adapter: BaseEmbeddingAdapter):
    """§10.3: Embeddings should be consistent and of reasonable quality."""
    # Test identical inputs produce identical outputs
    spec1 = EmbedSpec(text="consistent embedding", model=adapter.supported_models[0])
    spec2 = EmbedSpec(text="consistent embedding", model=adapter.supported_models[0])
    
    result1 = await adapter.embed(spec1)
    result2 = await adapter.embed(spec2)
    
    # Vectors should be identical for identical inputs
    assert result1.embedding.vector == result2.embedding.vector, \
        "Identical inputs should produce identical embeddings"
    
    # Test different inputs produce different outputs
    spec3 = EmbedSpec(text="different text", model=adapter.supported_models[0])
    result3 = await adapter.embed(spec3)
    
    # Vectors should be different for different inputs (with high probability)
    assert result1.embedding.vector != result3.embedding.vector, \
        "Different inputs should produce different embeddings"


async def test_core_ops_embed_special_character_handling(adapter: BaseEmbeddingAdapter):
    """§10.3: embed() should handle special characters and Unicode correctly."""
    test_cases = [
        "hello world!",
        "text with @#$% symbols",
        "Unicode: 中文, Español, Français",
        "Emoji: 🚀🌟😊",
        "Mixed: hello 世界 🌍!",
        "Numbers: 12345 67.89",
        "Whitespace:   multiple   spaces   ",
    ]
    
    for text in test_cases:
        spec = EmbedSpec(text=text, model=adapter.supported_models[0])
        result = await adapter.embed(spec)
        
        # Should succeed and produce valid embedding
        assert result.embedding.vector, f"Failed for text: {repr(text)}"
        assert len(result.embedding.vector) > 0
        assert all(isinstance(v, (int, float)) for v in result.embedding.vector)


async def test_core_ops_embed_context_propagation(adapter: BaseEmbeddingAdapter):
    """§6.1: Operation context should be properly propagated."""
    from unittest.mock import Mock
    mock_metrics = Mock()
    
    ctx = make_ctx(
        OperationContext,
        request_id="test_context_123",
        tenant="test-tenant",
        metrics=mock_metrics,
    )
    
    spec = EmbedSpec(text="context test", model=adapter.supported_models[0])
    result = await adapter.embed(spec, ctx=ctx)
    
    # Should produce valid result
    assert result.embedding.vector
    assert len(result.embedding.vector) > 0
    
    # If adapter implements metrics, they should be called with context
    if hasattr(adapter, '_emit_metrics') or mock_metrics.method_calls:
        # Verify metrics were called with embedding component
        assert mock_metrics.observe.called or not mock_metrics.method_calls
        if mock_metrics.observe.called:
            call_args = mock_metrics.observe.call_args
            assert call_args.kwargs.get('component') == 'embedding'


async def test_core_ops_embed_dimensions_consistent_with_capabilities(adapter: BaseEmbeddingAdapter):
    """§10.5: Embedding dimensions should be consistent with capabilities."""
    caps = await adapter.capabilities()
    
    test_texts = ["short", "medium length text", "a longer piece of text for testing dimensions"]
    
    for text in test_texts:
        spec = EmbedSpec(text=text, model=caps.supported_models[0])
        result = await adapter.embed(spec)
        
        dimensions = result.embedding.dimensions
        vector_length = len(result.embedding.vector)
        
        # Dimensions should match vector length
        assert dimensions == vector_length
        
        # Dimensions should be consistent across calls
        if caps.max_dimensions:
            assert dimensions <= caps.max_dimensions
        
        # Dimensions should be positive
        assert dimensions > 0