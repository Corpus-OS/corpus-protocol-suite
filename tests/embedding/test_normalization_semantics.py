# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Normalization semantics.

Spec refs:
  • §10.6 Normalization & Truncation Semantics
  • §10.5 Capabilities Discovery
  • §10.4 Errors (Embedding-Specific)
  • §13.3 Observability & Privacy
"""

import math
import pytest

from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter,
    OperationContext,
    EmbedSpec,
    BatchEmbedSpec,
    NotSupported,
)

pytestmark = pytest.mark.asyncio


def _norm(vec):
    """Calculate L2 norm of a vector."""
    return math.sqrt(sum(v * v for v in vec))


def supports_normalization(adapter: BaseEmbeddingAdapter) -> bool:
    """Check normalization capability."""
    caps = adapter.capabilities
    return getattr(caps, "supports_normalization", False)


async def test_normalization_single_embed_normalize_true_produces_unit_vector(adapter: BaseEmbeddingAdapter):
    """§10.6: normalize=True must produce approximately unit vectors."""
    if not supports_normalization(adapter):
        pytest.skip("Adapter does not support normalization")

    ctx = OperationContext(request_id="t_norm_single_true", tenant="t")

    spec = EmbedSpec(
        text="normalize this text to unit length",
        model=adapter.supported_models[0],
        truncate=True,
        normalize=True,
    )
    res = await adapter.embed(spec, ctx=ctx)

    norm = _norm(res.embedding.vector)
    assert 0.99 <= norm <= 1.01, f"Expected unit norm (1.0), got {norm:.6f}"
    assert res.truncated is False or res.truncated is None


async def test_normalization_single_embed_normalize_false_not_forced_unit_norm(adapter: BaseEmbeddingAdapter):
    """§10.6: normalize=False should not force unit norm when normalizes_at_source=False."""
    if not supports_normalization(adapter):
        pytest.skip("Adapter does not support normalization")
    
    caps = adapter.capabilities
    if getattr(caps, "normalizes_at_source", False):
        pytest.skip("Adapter normalizes at source; cannot test non-unit norms")

    ctx = OperationContext(request_id="t_norm_single_false", tenant="t")

    spec = EmbedSpec(
        text="provide raw vector without normalization",
        model=adapter.supported_models[0],
        truncate=True,
        normalize=False,
    )
    res = await adapter.embed(spec, ctx=ctx)

    norm = _norm(res.embedding.vector)
    # For non-source-normalizing adapters, norm should be meaningfully different from 1
    assert abs(norm - 1.0) > 1e-3, f"Expected non-unit norm for normalize=False, got {norm:.6f}"


async def test_normalization_batch_embed_normalize_true_all_unit_vectors(adapter: BaseEmbeddingAdapter):
    """§10.6: Batch normalization must apply consistently to all vectors."""
    if not supports_normalization(adapter):
        pytest.skip("Adapter does not support normalization")

    ctx = OperationContext(request_id="t_norm_batch_true", tenant="t")

    spec = BatchEmbedSpec(
        texts=["first text to normalize", "second text", "third example"],
        model=adapter.supported_models[0],
        truncate=True,
        normalize=True,
    )
    res = await adapter.embed_batch(spec, ctx=ctx)

    assert len(res.embeddings) == 3, "Batch should process all items"
    for i, embedding in enumerate(res.embeddings):
        norm = _norm(embedding.vector)
        assert 0.99 <= norm <= 1.01, f"Vector {i} should be unit norm, got {norm:.6f}"


async def test_normalization_not_supported_raises_clear_error(adapter: BaseEmbeddingAdapter):
    """§10.4: Normalization requests must raise NotSupported when unsupported."""
    if supports_normalization(adapter):
        pytest.skip("Adapter supports normalization")

    ctx = OperationContext(request_id="t_norm_nosupport", tenant="t")

    spec = EmbedSpec(
        text="test normalization error",
        model=adapter.supported_models[0],
        truncate=True,
        normalize=True,
    )
    with pytest.raises(NotSupported) as exc_info:
        await adapter.embed(spec, ctx=ctx)
    
    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ['normaliz', 'support', 'implement']), \
        f"Error should mention normalization: {error_msg}"


async def test_normalization_normalizes_at_source_respected(adapter: BaseEmbeddingAdapter):
    """§10.6: normalizes_at_source=True adapters should work without double-normalization."""
    caps = adapter.capabilities
    if not getattr(caps, "normalizes_at_source", False):
        pytest.skip("Adapter does not normalize at source")

    ctx = OperationContext(request_id="t_norm_source", tenant="t")

    spec = EmbedSpec(
        text="text for source-normalizing adapter",
        model=adapter.supported_models[0],
        truncate=True,
        normalize=True,
    )
    res = await adapter.embed(spec, ctx=ctx)

    # Should still produce unit vectors
    norm = _norm(res.embedding.vector)
    assert 0.99 <= norm <= 1.01, f"Source-normalizing adapter should produce unit norm, got {norm:.6f}"


async def test_normalization_consistency_across_calls(adapter: BaseEmbeddingAdapter):
    """§6.1: Normalization should be deterministic across identical calls."""
    if not supports_normalization(adapter):
        pytest.skip("Adapter does not support normalization")

    ctx = OperationContext(request_id="t_norm_consistent", tenant="t")

    spec = EmbedSpec(
        text="identical text for consistency check",
        model=adapter.supported_models[0],
        normalize=True,
    )
    
    # Multiple calls should produce identical normalized vectors
    result1 = await adapter.embed(spec, ctx=ctx)
    result2 = await adapter.embed(spec, ctx=ctx)

    vec1 = result1.embedding.vector
    vec2 = result2.embedding.vector
    
    assert vec1 == vec2, "Normalized vectors should be identical for identical inputs"
    
    # Both should be unit length
    norm1 = _norm(vec1)
    norm2 = _norm(vec2)
    assert 0.99 <= norm1 <= 1.01 and 0.99 <= norm2 <= 1.01


async def test_normalization_different_texts_different_vectors(adapter: BaseEmbeddingAdapter):
    """§10.6: Different texts should produce different normalized vectors."""
    if not supports_normalization(adapter):
        pytest.skip("Adapter does not support normalization")

    ctx = OperationContext(request_id="t_norm_different", tenant="t")

    spec1 = EmbedSpec(text="first unique text", model=adapter.supported_models[0], normalize=True)
    spec2 = EmbedSpec(text="second different text", model=adapter.supported_models[0], normalize=True)
    
    result1 = await adapter.embed(spec1, ctx=ctx)
    result2 = await adapter.embed(spec2, ctx=ctx)

    vec1 = result1.embedding.vector
    vec2 = result2.embedding.vector
    
    # Different texts should produce different vectors (with high probability)
    assert vec1 != vec2, "Different texts should produce different normalized vectors"
    
    # Both should still be unit length
    assert 0.99 <= _norm(vec1) <= 1.01
    assert 0.99 <= _norm(vec2) <= 1.01


async def test_normalization_small_vectors_handled(adapter: BaseEmbeddingAdapter):
    """§10.6: Normalization should handle very small input vectors correctly."""
    if not supports_normalization(adapter):
        pytest.skip("Adapter does not support normalization")

    ctx = OperationContext(request_id="t_norm_small", tenant="t")

    # Test with very short text that might produce small vectors
    short_texts = ["a", "hi", "ok"]
    
    for text in short_texts:
        spec = EmbedSpec(text=text, model=adapter.supported_models[0], normalize=True)
        result = await adapter.embed(spec, ctx=ctx)
        
        norm = _norm(result.embedding.vector)
        # Even small vectors should be properly normalized
        assert 0.99 <= norm <= 1.01, f"Small text '{text}' should produce unit norm, got {norm:.6f}"


async def test_normalization_batch_mixed_normalization(adapter: BaseEmbeddingAdapter):
    """§10.6: Batch should handle mixed normalization settings per spec."""
    if not supports_normalization(adapter):
        pytest.skip("Adapter does not support normalization")

    ctx = OperationContext(request_id="t_norm_batch_mixed", tenant="t")

    # Note: BatchEmbedSpec applies normalization uniformly to all items
    # This tests that the batch-level normalization flag works correctly
    spec_normalized = BatchEmbedSpec(
        texts=["batch text one", "batch text two"],
        model=adapter.supported_models[0],
        normalize=True,
    )
    
    spec_raw = BatchEmbedSpec(
        texts=["batch text one", "batch text two"], 
        model=adapter.supported_models[0],
        normalize=False,
    )
    
    result_norm = await adapter.embed_batch(spec_normalized, ctx=ctx)
    result_raw = await adapter.embed_batch(spec_raw, ctx=ctx)
    
    # All normalized batch vectors should be unit length
    for embedding in result_norm.embeddings:
        norm = _norm(embedding.vector)
        assert 0.99 <= norm <= 1.01, f"Normalized batch vector should be unit norm, got {norm:.6f}"
    
    # Raw batch vectors might not be unit length
    caps = adapter.capabilities
    if not getattr(caps, "normalizes_at_source", False):
        for embedding in result_raw.embeddings:
            norm = _norm(embedding.vector)
            # Allow some tolerance but expect difference from 1.0
            assert abs(norm - 1.0) > 1e-3 or norm == 0, f"Raw batch vector unexpectedly near unit norm: {norm:.6f}"
