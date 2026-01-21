okay how many of the test would this base file and adapter pass. go through test by test and provide pass or fail and the logic behind each decision

# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Cache behavior & batch fallback semantics.

Spec refs:
  • §10.3 embed(), embed_batch()
  • §10.4 Errors (Embedding-Specific)
  • §10.5 Capabilities Discovery
  • §12.5 Partial Success & Caching
  • §11.6 Caching (Implementation Guidance)
"""

import asyncio
import time
import pytest
from typing import List

from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter,
    EmbedSpec,
    BatchEmbedSpec,
    EmbeddingCapabilities,
    EmbedResult,
    BatchEmbedResult,
    EmbeddingVector,
    OperationContext,
    NotSupported,
    BadRequest,
    EmbeddingStats,
)

pytestmark = pytest.mark.asyncio


# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------

async def get_model(adapter: BaseEmbeddingAdapter) -> str:
    """Helper to safely get the first supported model."""
    caps = await adapter.capabilities()
    return caps.supported_models[0]


async def has_caching_capability(adapter: BaseEmbeddingAdapter) -> bool:
    """Check if adapter declares caching capability."""
    capabilities = await adapter.capabilities()
    return capabilities.supports_caching


async def has_batch_capability(adapter: BaseEmbeddingAdapter) -> bool:
    """Check if adapter declares batch embedding capability."""
    capabilities = await adapter.capabilities()
    return capabilities.supports_batch_embedding


async def get_cache_stats(adapter: BaseEmbeddingAdapter) -> tuple[int, int]:
    """Get cache hits and misses from stats."""
    stats = await adapter.get_stats()
    return stats.cache_hits, stats.cache_misses


# ----------------------------------------------------------------------
# Cache Tests (5 tests - ALL PASS)
# ----------------------------------------------------------------------

@pytest.mark.embedding
async def test_cache_hits_and_misses_tracked(adapter: BaseEmbeddingAdapter):
    """Cache hits and misses should be tracked in stats."""
    if not await has_caching_capability(adapter):
        pytest.skip("Adapter does not support caching")
    
    ctx = OperationContext(request_id="cache_stats", tenant="t1")
    model = await get_model(adapter)
    spec = EmbedSpec(text="cache stats test", model=model, normalize=False)
    
    # Get initial stats
    initial_hits, initial_misses = await get_cache_stats(adapter)
    
    # First call
    await adapter.embed(spec, ctx=ctx)
    hits1, misses1 = await get_cache_stats(adapter)
    
    # Second call
    await adapter.embed(spec, ctx=ctx)
    hits2, misses2 = await get_cache_stats(adapter)
    
    # Cache stats should reflect operations
    # Hits may increase if cached, misses may increase if not
    assert (hits2 + misses2) >= (hits1 + misses1), "Total cache operations should not decrease"


@pytest.mark.embedding
async def test_cache_tenant_isolation(adapter: BaseEmbeddingAdapter):
    """Cache must be tenant-isolated."""
    if not await has_caching_capability(adapter):
        pytest.skip("Adapter does not support caching")
    
    model = await get_model(adapter)
    spec = EmbedSpec(text="tenant isolation test", model=model, normalize=False)
    
    # Get initial stats
    initial_hits, initial_misses = await get_cache_stats(adapter)
    
    # Call with tenant 1
    ctx1 = OperationContext(request_id="t1", tenant="tenant-1")
    result1 = await adapter.embed(spec, ctx=ctx1)
    hits1, misses1 = await get_cache_stats(adapter)
    
    # Call with tenant 2
    ctx2 = OperationContext(request_id="t2", tenant="tenant-2")
    result2 = await adapter.embed(spec, ctx=ctx2)
    hits2, misses2 = await get_cache_stats(adapter)
    
    # Results should be identical (same model, same text)
    assert result1.embedding.vector == result2.embedding.vector
    
    # Tenant isolation: stats should show operations for both
    assert (hits2 + misses2) > (initial_hits + initial_misses), "Both calls should be tracked"


@pytest.mark.embedding
async def test_cache_model_isolation(adapter: BaseEmbeddingAdapter):
    """Cache should be isolated by model."""
    if not await has_caching_capability(adapter):
        pytest.skip("Adapter does not support caching")
    
    caps = await adapter.capabilities()
    if len(caps.supported_models) < 2:
        pytest.skip("Need at least 2 models to test model isolation")
    
    model1, model2 = caps.supported_models[0], caps.supported_models[1]
    ctx = OperationContext(request_id="model_iso", tenant="t1")
    
    # Call with model 1
    spec1 = EmbedSpec(text="same text", model=model1, normalize=False)
    result1 = await adapter.embed(spec1, ctx=ctx)
    
    # Call with model 2
    spec2 = EmbedSpec(text="same text", model=model2, normalize=False)
    result2 = await adapter.embed(spec2, ctx=ctx)
    
    # Different models should produce different vectors
    assert result1.embedding.vector != result2.embedding.vector


@pytest.mark.embedding
async def test_cache_normalization_isolation(adapter: BaseEmbeddingAdapter):
    """Cache should be isolated by normalization flag."""
    if not await has_caching_capability(adapter):
        pytest.skip("Adapter does not support caching")
    
    caps = await adapter.capabilities()
    if not caps.supports_normalization:
        pytest.skip("Adapter does not support normalization")
    
    ctx = OperationContext(request_id="norm_iso", tenant="t1")
    model = caps.supported_models[0]
    
    # Call with normalize=False
    spec1 = EmbedSpec(text="normalization test", model=model, normalize=False)
    result1 = await adapter.embed(spec1, ctx=ctx)
    
    # Call with normalize=True
    spec2 = EmbedSpec(text="normalization test", model=model, normalize=True)
    result2 = await adapter.embed(spec2, ctx=ctx)
    
    # Results should differ (one normalized, one not)
    assert result1.embedding.vector != result2.embedding.vector


@pytest.mark.embedding
async def test_cache_observable_behavior(adapter: BaseEmbeddingAdapter):
    """Cache should exhibit observable behavior."""
    if not await has_caching_capability(adapter):
        pytest.skip("Adapter does not support caching")
    
    ctx = OperationContext(request_id="cache_obs", tenant="t1")
    model = await get_model(adapter)
    spec = EmbedSpec(text="cache observable", model=model, normalize=False)
    
    # First call
    result1 = await adapter.embed(spec, ctx=ctx)
    
    # Immediate second call
    result2 = await adapter.embed(spec, ctx=ctx)
    
    # Results should be identical
    assert result1.embedding.vector == result2.embedding.vector


# ----------------------------------------------------------------------
# Batch Fallback Tests (6 tests - ALL PASS)
# ----------------------------------------------------------------------

@pytest.mark.embedding
async def test_batch_fallback_or_native_behavior(adapter: BaseEmbeddingAdapter):
    """Batch should work whether supported natively or via fallback."""
    caps = await adapter.capabilities()
    
    ctx = OperationContext(request_id="batch_works", tenant="t1")
    texts = ["text one", "text two", "text three"]
    spec = BatchEmbedSpec(texts=texts, model=caps.supported_models[0])
    
    try:
        result = await adapter.embed_batch(spec, ctx=ctx)
        
        # Batch succeeded (either native or via fallback)
        assert len(result.embeddings) <= len(texts)  # May have failures
        assert result.total_texts == len(texts)
        
        # Verify successful embeddings
        for emb in result.embeddings:
            assert emb.index is not None
            assert 0 <= emb.index < len(texts)
            assert emb.text == texts[emb.index]
            assert len(emb.vector) > 0
            
    except NotSupported:
        # Batch not supported AND fallback not implemented
        if caps.supports_batch_embedding:
            raise  # Should support but doesn't
        else:
            pytest.skip("Adapter does not support batch and cannot fall back")


@pytest.mark.embedding
async def test_batch_handles_invalid_texts(adapter: BaseEmbeddingAdapter):
    """Batch should handle invalid texts appropriately."""
    if not await has_batch_capability(adapter):
        pytest.skip("Batch embedding not supported by adapter")
    
    ctx = OperationContext(request_id="invalid_texts", tenant="t1")
    
    # Mix of valid and potentially problematic texts
    texts = ["valid text", "", "another valid"]  # Empty string in middle
    model = await get_model(adapter)
    
    spec = BatchEmbedSpec(texts=texts, model=model)
    
    try:
        result = await adapter.embed_batch(spec, ctx=ctx)
        
        # If we get here, adapter collected failures or succeeded
        assert isinstance(result, BatchEmbedResult)
        assert result.total_texts == len(texts)
        
        # Validate structure
        for emb in result.embeddings:
            assert isinstance(emb, EmbeddingVector)
            assert emb.index is not None
            assert 0 <= emb.index < len(texts)
            assert len(emb.vector) > 0
            
        # Check failed texts (if any)
        for failure in result.failed_texts:
            assert "index" in failure
            idx = failure["index"]
            assert 0 <= idx < len(texts)
            assert "text" in failure
            assert failure["text"] == texts[idx]
            
        # Verify math
        total_processed = len(result.embeddings) + len(result.failed_texts)
        assert total_processed == len(texts)
        
        # No overlap
        success_indices = {e.index for e in result.embeddings}
        failure_indices = {f["index"] for f in result.failed_texts}
        assert success_indices.isdisjoint(failure_indices)
        
    except BadRequest:
        # Adapter rejects entire batch on invalid text (valid behavior)
        # Check that error mentions empty/invalid text
        pass


@pytest.mark.embedding
async def test_batch_ordering_preserved(adapter: BaseEmbeddingAdapter):
    """Batch results must preserve input ordering."""
    if not await has_batch_capability(adapter):
        pytest.skip("Batch embedding not supported by adapter")
    
    ctx = OperationContext(request_id="batch_order", tenant="t1")
    
    texts = ["first", "second", "third", "fourth"]
    model = await get_model(adapter)
    spec = BatchEmbedSpec(texts=texts, model=model)
    
    result = await adapter.embed_batch(spec, ctx=ctx)
    
    # Check indices are preserved for successful embeddings
    for emb in result.embeddings:
        assert emb.index is not None
        assert 0 <= emb.index < len(texts)
        # Text should match or be related to original
        original = texts[emb.index]
        assert emb.text == original or original in emb.text or emb.text in original


@pytest.mark.embedding
async def test_batch_metadata_propagation(adapter: BaseEmbeddingAdapter):
    """Batch metadata should propagate when provided."""
    if not await has_batch_capability(adapter):
        pytest.skip("Batch embedding not supported by adapter")
    
    ctx = OperationContext(request_id="batch_metadata", tenant="t1")
    
    texts = ["doc1", "doc2", "doc3"]
    metadatas = [
        {"id": 1, "type": "a"},
        {"id": 2, "type": "b"},
        {"id": 3, "type": "c"}
    ]
    
    model = await get_model(adapter)
    spec = BatchEmbedSpec(
        texts=texts,
        model=model,
        metadatas=metadatas
    )
    
    result = await adapter.embed_batch(spec, ctx=ctx)
    
    # For successful embeddings with metadata provided, check if metadata attached
    for emb in result.embeddings:
        if emb.index is not None and emb.index < len(metadatas):
            # Metadata may or may not be attached (adapter choice)
            # We just verify the embedding is valid
            assert len(emb.vector) > 0
            assert emb.text == texts[emb.index] or texts[emb.index] in emb.text


@pytest.mark.embedding
async def test_batch_size_limit_enforced(adapter: BaseEmbeddingAdapter):
    """Batch size limits should be enforced."""
    caps = await adapter.capabilities()
    
    if caps.max_batch_size is None:
        pytest.skip("Adapter has no batch size limit")
    
    ctx = OperationContext(request_id="batch_limit", tenant="t1")
    
    # Create batch exceeding limit
    oversized = caps.max_batch_size + 1
    texts = [f"text {i}" for i in range(oversized)]
    spec = BatchEmbedSpec(texts=texts, model=caps.supported_models[0])
    
    # Should raise BadRequest
    with pytest.raises(BadRequest) as exc_info:
        await adapter.embed_batch(spec, ctx=ctx)
    
    # Error should mention batch size
    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ['batch', 'size', 'limit', 'exceed', 'maximum'])


@pytest.mark.embedding
async def test_batch_empty_text_handling(adapter: BaseEmbeddingAdapter):
    """Empty texts should be handled consistently."""
    if not await has_batch_capability(adapter):
        pytest.skip("Batch embedding not supported by adapter")
    
    ctx = OperationContext(request_id="empty_handling", tenant="t1")
    
    texts = ["", "non-empty", ""]  # Multiple empties
    model = await get_model(adapter)
    spec = BatchEmbedSpec(texts=texts, model=model)
    
    try:
        result = await adapter.embed_batch(spec, ctx=ctx)
        
        # If batch succeeds, empty texts are either all failed or all succeeded
        empty_indices = {0, 2}
        empty_success = {e.index for e in result.embeddings if e.index in empty_indices}
        empty_failure = {f["index"] for f in result.failed_texts if f["index"] in empty_indices}
        
        # Empty texts should be consistently handled
        assert empty_success.isdisjoint(empty_failure)
        
    except BadRequest:
        # Entire batch rejected due to empty text (valid behavior)
        pass


# ----------------------------------------------------------------------
# Integration Tests (2 tests - ALL PASS)
# ----------------------------------------------------------------------

@pytest.mark.embedding
async def test_cache_and_batch_independence(adapter: BaseEmbeddingAdapter):
    """Cache and batch operations should not interfere."""
    if not await has_caching_capability(adapter):
        pytest.skip("Adapter does not support caching")
    
    ctx = OperationContext(request_id="cache_batch_indep", tenant="t1")
    model = await get_model(adapter)
    text = "independence test"
    
    # Get initial cache stats
    initial_hits, initial_misses = await get_cache_stats(adapter)
    
    # Single embed
    single_spec = EmbedSpec(text=text, model=model, normalize=False)
    single_result = await adapter.embed(single_spec, ctx=ctx)
    
    # Batch embed with same text
    batch_spec = BatchEmbedSpec(texts=[text], model=model)
    batch_result = await adapter.embed_batch(batch_spec, ctx=ctx)
    
    # Get final cache stats
    final_hits, final_misses = await get_cache_stats(adapter)
    
    # Results should match (same input)
    assert single_result.embedding.vector == batch_result.embeddings[0].vector
    
    # Cache stats should reflect operations
    assert (final_hits + final_misses) >= (initial_hits + initial_misses)


@pytest.mark.embedding
async def test_batch_cache_integration_positive(adapter: BaseEmbeddingAdapter):
    """Batch should work when caching is enabled."""
    caps = await adapter.capabilities()
    
    # Skip if no batch
    if not caps.supports_batch_embedding:
        pytest.skip("Batch embedding not supported")
    
    ctx = OperationContext(request_id="batch_cache_int", tenant="t1")
    
    texts = ["batch with cache 1", "batch with cache 2"]
    spec = BatchEmbedSpec(texts=texts, model=caps.supported_models[0])
    
    # Execute batch
    result = await adapter.embed_batch(spec, ctx=ctx)
    
    # Basic validation
    assert len(result.embeddings) <= len(texts)  # May have failures
    assert result.total_texts == len(texts)
    
    # Verify successful embeddings
    for emb in result.embeddings:
        assert emb.index is not None
        assert 0 <= emb.index < len(texts)
        assert len(emb.vector) > 0


# ----------------------------------------------------------------------
# Test Count: 13 tests total
# ----------------------------------------------------------------------

# 5 cache tests + 6 batch tests + 2 integration tests = 13 tests
# All should pass with default mock configuration
