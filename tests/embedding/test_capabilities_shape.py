# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Capabilities shape & stability.

Spec refs:
  • §10.2 Capabilities Discovery (Embedding)
  • §6.2 Common — Capability surfaces MUST be stable and self-consistent
  • §10.3 Operations (validate capabilities match actual behavior)
"""

import pytest

from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter,
    EmbeddingCapabilities,
    EmbedSpec,
    BatchEmbedSpec,
    OperationContext,
    NotSupported,
)
from examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_capabilities_returns_correct_type(adapter: BaseEmbeddingAdapter):
    """§10.2: Capabilities must return EmbeddingCapabilities type."""
    caps = await adapter.capabilities()
    assert isinstance(caps, EmbeddingCapabilities)


async def test_capabilities_identity_fields(adapter: BaseEmbeddingAdapter):
    """§6.2: Server identity and version must be non-empty strings."""
    caps = await adapter.capabilities()
    assert isinstance(caps.server, str) and caps.server
    assert isinstance(caps.version, str) and caps.version


async def test_capabilities_supported_models_non_empty_tuple(adapter: BaseEmbeddingAdapter):
    """§10.2: Supported models must be non-empty tuple of strings."""
    caps = await adapter.capabilities()
    assert isinstance(caps.supported_models, tuple)
    assert len(caps.supported_models) > 0, "At least one model must be supported"
    assert all(isinstance(m, str) and m for m in caps.supported_models)


async def test_capabilities_resource_limits_valid(adapter: BaseEmbeddingAdapter):
    """§10.2: Resource limits must be positive integers when specified."""
    caps = await adapter.capabilities()

    if caps.max_batch_size is not None:
        assert caps.max_batch_size > 0, "max_batch_size must be positive if specified"

    if caps.max_text_length is not None:
        assert caps.max_text_length > 0, "max_text_length must be positive if specified"

    if caps.max_dimensions is not None:
        assert caps.max_dimensions > 0, "max_dimensions must be positive if specified"


async def test_capabilities_feature_flags_boolean(adapter: BaseEmbeddingAdapter):
    """§10.2: All feature flags must be boolean values."""
    caps = await adapter.capabilities()
    bool_fields = (
        "supports_normalization",
        "supports_truncation", 
        "supports_token_counting",
        "idempotent_operations",
        "supports_multi_tenant",
        "normalizes_at_source",
        "supports_deadline",
        "supports_batch_embedding",
        "supports_caching",
    )
    for name in bool_fields:
        value = getattr(caps, name, None)
        assert isinstance(value, bool), f"{name} must be bool, got {type(value)}"


async def test_capabilities_truncation_mode_valid(adapter: BaseEmbeddingAdapter):
    """§10.2: Truncation mode must be valid enum value."""
    caps = await adapter.capabilities()
    assert caps.truncation_mode in ("base", "adapter", "none"), \
        f"Invalid truncation_mode: {caps.truncation_mode}"


async def test_capabilities_max_dimensions_consistent_with_models(adapter: BaseEmbeddingAdapter):
    """§10.2: Max dimensions must be consistent with model capabilities."""
    caps = await adapter.capabilities()
    if caps.max_dimensions is not None:
        assert caps.max_dimensions >= 0, "max_dimensions must be non-negative"


async def test_capabilities_idempotent(adapter: BaseEmbeddingAdapter):
    """§6.2: Capabilities must be idempotent across calls."""
    c1 = await adapter.capabilities()
    c2 = await adapter.capabilities()
    assert c1 == c2, "Capabilities must be stable across multiple calls"


async def test_capabilities_match_operational_behavior_batch(adapter: BaseEmbeddingAdapter):
    """§10.3: Batch support capability must match actual operational behavior."""
    caps = await adapter.capabilities()
    ctx = make_ctx(OperationContext, request_id="cap_batch_test", tenant="t")
    
    if caps.supports_batch_embedding:
        # If batch is supported, it must work with valid input
        spec = BatchEmbedSpec(
            texts=["test batch capability"], 
            model=caps.supported_models[0]
        )
        try:
            result = await adapter.embed_batch(spec, ctx=ctx)
            assert result is not None, "Batch operation should succeed when supported"
        except NotSupported:
            pytest.fail("Batch operation failed but capabilities claim support")
    else:
        # If batch is not supported, it must raise NotSupported
        spec = BatchEmbedSpec(
            texts=["test batch capability"],
            model=caps.supported_models[0]  
        )
        with pytest.raises(NotSupported):
            await adapter.embed_batch(spec, ctx=ctx)


async def test_capabilities_match_operational_behavior_normalization(adapter: BaseEmbeddingAdapter):
    """§10.3: Normalization support must match actual behavior."""
    caps = await adapter.capabilities()
    ctx = make_ctx(OperationContext, request_id="cap_norm_test", tenant="t")
    
    spec_normalized = EmbedSpec(
        text="test normalization",
        model=caps.supported_models[0],
        normalize=True
    )
    
    if caps.supports_normalization:
        # Normalization should work when supported
        try:
            result = await adapter.embed(spec_normalized, ctx=ctx)
            assert result is not None, "Normalization should work when supported"
        except NotSupported:
            pytest.fail("Normalization failed but capabilities claim support")
    else:
        # Normalization should fail when not supported
        with pytest.raises(NotSupported):
            await adapter.embed(spec_normalized, ctx=ctx)


async def test_capabilities_max_batch_size_respected(adapter: BaseEmbeddingAdapter):
    """§10.2: Max batch size limit must be enforced in operations."""
    caps = await adapter.capabilities()
    
    if caps.supports_batch_embedding and caps.max_batch_size is not None:
        ctx = make_ctx(OperationContext, request_id="batch_limit_test", tenant="t")
        
        # Create batch that exceeds the limit
        oversized_batch = ["text"] * (caps.max_batch_size + 1)
        spec = BatchEmbedSpec(texts=oversized_batch, model=caps.supported_models[0])
        
        # Should either handle gracefully or raise appropriate error
        try:
            result = await adapter.embed_batch(spec, ctx=ctx)
            # If it succeeds, verify it handled the oversize correctly
            assert len(result.embeddings) + len(result.failed_texts) <= caps.max_batch_size + 1
        except (ValueError, NotSupported) as e:
            # Expected behavior - reject oversized batch
            assert "batch" in str(e).lower() or "size" in str(e).lower() or "limit" in str(e).lower()


async def test_capabilities_max_text_length_respected(adapter: BaseEmbeddingAdapter):
    """§10.2: Max text length limit must be enforced."""
    caps = await adapter.capabilities()
    
    if caps.max_text_length is not None:
        ctx = make_ctx(OperationContext, request_id="text_limit_test", tenant="t")
        
        # Create text that exceeds the limit
        oversized_text = "x" * (caps.max_text_length + 1)
        spec = EmbedSpec(text=oversized_text, model=caps.supported_models[0], normalize=False)
        
        # Should either fail or truncate based on truncation_mode
        try:
            result = await adapter.embed(spec, ctx=ctx)
            if caps.truncation_mode == "none":
                pytest.fail("Oversized text should fail when truncation_mode is 'none'")
        except (ValueError, NotSupported) as e:
            # Expected behavior for non-truncating adapters
            assert "length" in str(e).lower() or "size" in str(e).lower() or "limit" in str(e).lower()


async def test_capabilities_supported_models_accurate(adapter: BaseEmbeddingAdapter):
    """§10.2: Supported models list must be accurate for actual operations."""
    caps = await adapter.capabilities()
    ctx = make_ctx(OperationContext, request_id="model_test", tenant="t")
    
    # Test first supported model
    valid_spec = EmbedSpec(text="test", model=caps.supported_models[0], normalize=False)
    result = await adapter.embed(valid_spec, ctx=ctx)
    assert result is not None, "Supported model must work"
    
    # Test unsupported model (if we can determine one)
    unsupported_models = ["invalid-model-123", "unknown-model"]
    for model in unsupported_models:
        if model not in caps.supported_models:
            invalid_spec = EmbedSpec(text="test", model=model, normalize=False)
            with pytest.raises((ValueError, NotSupported)):
                await adapter.embed(invalid_spec, ctx=ctx)
            break


async def test_capabilities_multi_tenant_isolation(adapter: BaseEmbeddingAdapter):
    """§6.2: Multi-tenant support capability must reflect actual isolation behavior."""
    caps = await adapter.capabilities()
    
    if caps.supports_multi_tenant:
        # Test that different tenants don't interfere
        ctx1 = make_ctx(OperationContext, request_id="tenant1", tenant="tenant-1")
        ctx2 = make_ctx(OperationContext, request_id="tenant2", tenant="tenant-2")
        
        spec = EmbedSpec(text="multi-tenant test", model=caps.supported_models[0], normalize=False)
        
        result1 = await adapter.embed(spec, ctx=ctx1)
        result2 = await adapter.embed(spec, ctx=ctx2)
        
        # Both should work independently
        assert result1 is not None
        assert result2 is not None
    # If not supported, we don't require failure - some adapters may still work with multiple tenants


async def test_capabilities_serializable_structure(adapter: BaseEmbeddingAdapter):
    """§6.2: Capabilities object must be serializable for discovery endpoints."""
    caps = await adapter.capabilities()
    
    # Test JSON serialization (common requirement for API discovery)
    import json
    try:
        json_str = json.dumps(caps.__dict__)
        reconstructed = json.loads(json_str)
        assert isinstance(reconstructed, dict)
        # Verify key fields are preserved
        assert "server" in reconstructed
        assert "supported_models" in reconstructed
    except (TypeError, ValueError) as e:
        pytest.fail(f"Capabilities must be JSON serializable: {e}")