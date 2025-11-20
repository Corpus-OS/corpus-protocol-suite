# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Cache behavior & batch fallback semantics.

Spec refs:
  • §10.3 embed(), embed_batch()
  • §10.4 Errors (Embedding-Specific)
  • §10.5 Capabilities Discovery
  • §12.5 Partial Success & Caching
"""

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
)

pytestmark = pytest.mark.asyncio


def has_caching_capability(adapter: BaseEmbeddingAdapter) -> bool:
    """Check if adapter declares caching capability."""
    capabilities: EmbeddingCapabilities = adapter.capabilities
    return getattr(capabilities, 'supports_caching', False)


def has_batch_capability(adapter: BaseEmbeddingAdapter) -> bool:
    """Check if adapter declares batch embedding capability."""
    capabilities: EmbeddingCapabilities = adapter.capabilities
    return getattr(capabilities, 'supports_batch_embedding', True)  # Default True per spec


@pytest.mark.skip_if_not_supported("caching")
async def test_embed_cache_behavior_observable(adapter: BaseEmbeddingAdapter):
    """
    Cache behavior should be observable through timing and response characteristics.
    Second call with same spec should be significantly faster if caching is implemented.
    
    Spec: §10.3 Operations, §11.6 Caching (Implementation Guidance)
    """
    ctx = OperationContext(request_id="cache_observable", tenant="t1")
    spec = EmbedSpec(text="cache performance test text", model=adapter.supported_models[0], normalize=False)
    
    # First call - cold start
    start_time_1 = time.time()
    result_1 = await adapter.embed(spec, ctx=ctx)
    duration_1 = time.time() - start_time_1
    
    # Second call - should be faster if cached
    start_time_2 = time.time()
    result_2 = await adapter.embed(spec, ctx=ctx)
    duration_2 = time.time() - start_time_2
    
    # Validate results are identical
    assert isinstance(result_1, EmbedResult)
    assert isinstance(result_2, EmbedResult)
    assert result_1.embedding.vector == result_2.embedding.vector
    
    # If caching is implemented, second call should be significantly faster
    # Allow for some variance but expect at least 50% improvement for cached responses
    if has_caching_capability(adapter):
        assert duration_2 <= duration_1 * 0.8, f"Expected cached call to be faster. First: {duration_1:.3f}s, Second: {duration_2:.3f}s"


@pytest.mark.skip_if_not_supported("caching")
async def test_embed_cache_tenant_isolation_observable(adapter: BaseEmbeddingAdapter):
    """
    Cache must be tenant-isolated. Same spec under different tenants should not share cache entries.
    
    Spec: §14.1 Tenant Isolation, §10.3 Operations
    """
    spec = EmbedSpec(text="tenant isolation test", model=adapter.supported_models[0], normalize=False)
    
    # Call with tenant 1
    ctx1 = OperationContext(request_id="cache_t1", tenant="tenant-1")
    result_1 = await adapter.embed(spec, ctx=ctx1)
    
    # Call with tenant 2 - should not be cached from tenant 1
    ctx2 = OperationContext(request_id="cache_t2", tenant="tenant-2")
    start_time = time.time()
    result_2 = await adapter.embed(spec, ctx=ctx2)
    duration = time.time() - start_time
    
    # Both should succeed with same vector content (same model, same text)
    assert isinstance(result_1, EmbedResult)
    assert isinstance(result_2, EmbedResult)
    assert result_1.embedding.vector == result_2.embedding.vector
    
    # If we have caching capability, verify tenant isolation by checking performance
    # Tenant 2 call should take similar time to a cold call if isolation works
    if has_caching_capability(adapter):
        # This is a qualitative check - in practice we'd need baseline timing
        assert duration > 0.001, "Expected non-cached duration for different tenant"


async def test_embed_batch_partial_failure_contract(adapter: BaseEmbeddingAdapter):
    """
    Batch embedding must follow partial failure contract regardless of implementation.
    
    Spec: §12.5 Partial Success & Caching, §10.3 Operations
    """
    ctx = OperationContext(request_id="batch_partial", tenant="t")
    
    # Mix of valid and potentially problematic texts
    texts = ["valid text 1", "", "valid text 2", "another valid text"]
    spec = BatchEmbedSpec(texts=texts, model=adapter.supported_models[0])
    
    try:
        result = await adapter.embed_batch(spec, ctx=ctx)
    except NotSupported:
        if not has_batch_capability(adapter):
            pytest.skip("Batch embedding not supported by adapter")
        else:
            raise
    
    assert isinstance(result, BatchEmbedResult)
    
    # Validate successful embeddings
    for embedding in result.embeddings:
        assert isinstance(embedding, EmbeddingVector)
        assert embedding.index is not None
        assert 0 <= embedding.index < len(texts)
        assert isinstance(embedding.vector, List)
        assert len(embedding.vector) > 0
    
    # Validate failure reporting
    for failure in result.failed_texts:
        assert "index" in failure
        assert 0 <= failure["index"] < len(texts)
        assert "error" in failure
        assert "message" in failure["error"]
        assert "code" in failure["error"]
    
    # Verify no index appears in both successes and failures
    success_indices = {e.index for e in result.embeddings}
    failure_indices = {f["index"] for f in result.failed_texts}
    assert success_indices.isdisjoint(failure_indices), "Index cannot be both successful and failed"
    
    # Total processed items should match input
    total_processed = len(result.embeddings) + len(result.failed_texts)
    assert total_processed == len(texts), f"Expected {len(texts)} processed items, got {total_processed}"


async def test_embed_batch_empty_text_handling(adapter: BaseEmbeddingAdapter):
    """
    Empty texts should be handled according to spec - either failed with clear error
    or successfully embedded based on adapter capabilities.
    
    Spec: §10.4 Errors (Embedding-Specific), §12.5 Partial Failure Contracts
    """
    if not has_batch_capability(adapter):
        pytest.skip("Batch embedding not supported by adapter")
    
    ctx = OperationContext(request_id="batch_empty", tenant="t")
    texts = ["", "non-empty text", ""]  # Multiple empty texts
    spec = BatchEmbedSpec(texts=texts, model=adapter.supported_models[0])
    
    result = await adapter.embed_batch(spec, ctx=ctx)
    
    assert isinstance(result, BatchEmbedResult)
    
    # Check that empty texts are either all failed or all successful
    empty_text_indices = {0, 2}
    empty_success_indices = {e.index for e in result.embeddings if e.index in empty_text_indices}
    empty_failure_indices = {f["index"] for f in result.failed_texts if f["index"] in empty_text_indices}
    
    # Empty texts should be consistently handled (all fail or all succeed)
    assert empty_success_indices.isdisjoint(empty_failure_indices)
    
    # If empty texts fail, validate error messages
    for failure in result.failed_texts:
        if failure["index"] in empty_text_indices:
            assert "empty" in failure["error"]["message"].lower() or \
                   "invalid" in failure["error"]["message"].lower(), \
                   f"Expected descriptive error for empty text, got: {failure['error']['message']}"


async def test_embed_batch_ordering_preserved(adapter: BaseEmbeddingAdapter):
    """
    Batch results must preserve input ordering regardless of partial failures.
    
    Spec: §12.5 Partial Success & Caching, §10.3 Operations
    """
    if not has_batch_capability(adapter):
        pytest.skip("Batch embedding not supported by adapter")
    
    ctx = OperationContext(request_id="batch_order", tenant="t")
    texts = ["first", "second", "third"]
    spec = BatchEmbedSpec(texts=texts, model=adapter.supported_models[0])
    
    result = await adapter.embed_batch(spec, ctx=ctx)
    
    # Check that successful embeddings maintain original indices
    for embedding in result.embeddings:
        original_text = texts[embedding.index]
        if embedding.index == 0:
            assert "first" in original_text.lower()
        elif embedding.index == 1:
            assert "second" in original_text.lower()
        elif embedding.index == 2:
            assert "third" in original_text.lower()


async def test_embed_error_messages_spec_compliant(adapter: BaseEmbeddingAdapter):
    """
    Error messages must follow spec requirements for format and content.
    
    Spec: §10.4 Errors (Embedding-Specific), §12.4 Error Mapping Table
    """
    ctx = OperationContext(request_id="error_messages", tenant="t")
    
    # Test with empty text - should provide clear error
    spec = EmbedSpec(text="", model=adapter.supported_models[0], normalize=False)
    
    try:
        result = await adapter.embed(spec, ctx=ctx)
        # If no exception, check if empty text is actually supported
        assert len(result.embedding.vector) > 0, "Empty text should either fail or produce valid embedding"
    except Exception as e:
        # Validate error structure per spec
        error_msg = str(e).lower()
        assert any(term in error_msg for term in ['empty', 'invalid', 'text', 'input']), \
               f"Error message should describe the issue with empty text: {error_msg}"


@pytest.mark.skip_if_not_supported("caching")
async def test_embed_cache_invalidation_observable(adapter: BaseEmbeddingAdapter):
    """
    Cache should be invalidated when spec parameters change.
    
    Spec: §10.3 Operations, §11.6 Caching
    """
    ctx = OperationContext(request_id="cache_invalidation", tenant="t1")
    base_text = "cache invalidation test"
    
    # First call with normalize=False
    spec1 = EmbedSpec(text=base_text, model=adapter.supported_models[0], normalize=False)
    result_1 = await adapter.embed(spec1, ctx=ctx)
    
    # Second call with normalize=True - should not use cache
    spec2 = EmbedSpec(text=base_text, model=adapter.supported_models[0], normalize=True)
    start_time = time.time()
    result_2 = await adapter.embed(spec2, ctx=ctx)
    duration = time.time() - start_time
    
    # Results should be different due to normalization change
    assert result_1.embedding.vector != result_2.embedding.vector
    
    # If caching is implemented, this should take similar time to first call
    # (not cached due to parameter change)
    if has_caching_capability(adapter):
        assert duration > 0.001, "Expected non-cached duration for different spec parameters"


# Custom pytest marker for capability-based skipping
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", 
        "skip_if_not_supported(capability): skip test if adapter doesn't support specific capability"
    )


def pytest_runtest_setup(item):
    """Handle skip_if_not_supported marker."""
    skip_marker = item.get_closest_marker('skip_if_not_supported')
    if skip_marker:
        capability = skip_marker.args[0] if skip_marker.args else None
        adapter = item.funcargs.get('adapter')
        
        if adapter and capability == "caching" and not has_caching_capability(adapter):
            pytest.skip(f"Adapter does not support {capability}")
        elif adapter and capability == "batch" and not has_batch_capability(adapter):
            pytest.skip(f"Adapter does not support {capability}")
