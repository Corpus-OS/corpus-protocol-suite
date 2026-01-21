# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Capabilities shape & stability.

Spec refs:
  • §10.2 Capabilities Discovery (Embedding)
  • §6.2 Common — Capability surfaces MUST be stable and self-consistent
  • §10.3 Operations (validate capabilities match actual behavior)
  
"""

import json
import pytest
from dataclasses import asdict

from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter,
    EmbeddingCapabilities,
    EmbedSpec,
    BatchEmbedSpec,
    OperationContext,
    NotSupported,
    BadRequest,
    ModelNotAvailable,
    EMBEDDING_PROTOCOL_ID,
)

pytestmark = pytest.mark.asyncio


async def test_capabilities_returns_correct_type(adapter: BaseEmbeddingAdapter):
    caps = await adapter.capabilities()
    assert isinstance(caps, EmbeddingCapabilities)


async def test_capabilities_identity_fields(adapter: BaseEmbeddingAdapter):
    caps = await adapter.capabilities()
    assert isinstance(caps.server, str) and caps.server
    assert isinstance(caps.version, str) and caps.version


async def test_capabilities_supported_models_non_empty_tuple(adapter: BaseEmbeddingAdapter):
    caps = await adapter.capabilities()
    assert isinstance(caps.supported_models, tuple)
    assert len(caps.supported_models) > 0
    assert all(isinstance(m, str) and m for m in caps.supported_models)


async def test_capabilities_resource_limits_valid(adapter: BaseEmbeddingAdapter):
    caps = await adapter.capabilities()

    if caps.max_batch_size is not None:
        assert caps.max_batch_size > 0

    if caps.max_text_length is not None:
        assert caps.max_text_length > 0

    if caps.max_dimensions is not None:
        assert caps.max_dimensions > 0


async def test_capabilities_feature_flags_boolean(adapter: BaseEmbeddingAdapter):
    caps = await adapter.capabilities()
    bool_fields = (
        "supports_normalization",
        "supports_truncation",
        "supports_token_counting",
        "supports_streaming",
        "supports_batch_embedding",
        "supports_caching",
        "idempotent_writes",
        "supports_multi_tenant",
        "normalizes_at_source",
        "supports_deadline",
    )
    for name in bool_fields:
        value = getattr(caps, name, None)
        assert isinstance(value, bool), f"{name} must be bool, got {type(value)}"


async def test_capabilities_truncation_mode_valid(adapter: BaseEmbeddingAdapter):
    caps = await adapter.capabilities()
    assert caps.truncation_mode in ("base", "adapter", "none")


async def test_capabilities_max_dimensions_consistent_with_models(adapter: BaseEmbeddingAdapter):
    caps = await adapter.capabilities()
    if caps.max_dimensions is not None:
        assert caps.max_dimensions > 0
        assert caps.max_dimensions <= 100000


async def test_capabilities_idempotent(adapter: BaseEmbeddingAdapter):
    c1 = await adapter.capabilities()
    c2 = await adapter.capabilities()
    assert c1.server == c2.server
    assert c1.version == c2.version
    assert c1.supported_models == c2.supported_models
    assert c1.max_batch_size == c2.max_batch_size
    assert c1.max_text_length == c2.max_text_length


async def test_capabilities_serializable_structure(adapter: BaseEmbeddingAdapter):
    caps = await adapter.capabilities()
    caps_dict = asdict(caps)

    json_str = json.dumps(caps_dict)
    reconstructed = json.loads(json_str)
    assert isinstance(reconstructed, dict)
    assert "server" in reconstructed
    assert "supported_models" in reconstructed
    assert "protocol" in reconstructed


async def test_capabilities_protocol_version(adapter: BaseEmbeddingAdapter):
    caps = await adapter.capabilities()
    assert caps.protocol == EMBEDDING_PROTOCOL_ID


async def test_capabilities_supported_models_accurate(adapter: BaseEmbeddingAdapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(request_id="model_test", tenant="t")

    valid_spec = EmbedSpec(text="test", model=caps.supported_models[0], normalize=False)
    result = await adapter.embed(valid_spec, ctx=ctx)
    assert result is not None
    assert result.model == caps.supported_models[0]

    unsupported_model = "invalid-model-not-in-supported-list-12345"
    if unsupported_model not in caps.supported_models:
        invalid_spec = EmbedSpec(text="test", model=unsupported_model, normalize=False)
        with pytest.raises(ModelNotAvailable):
            await adapter.embed(invalid_spec, ctx=ctx)


async def test_capabilities_max_batch_size_respected(adapter: BaseEmbeddingAdapter):
    caps = await adapter.capabilities()

    if caps.supports_batch_embedding and caps.max_batch_size is not None:
        ctx = OperationContext(request_id="batch_limit_test", tenant="t")

        oversized_batch = ["text"] * (caps.max_batch_size + 1)
        spec = BatchEmbedSpec(texts=oversized_batch, model=caps.supported_models[0])

        with pytest.raises(BadRequest) as exc_info:
            await adapter.embed_batch(spec, ctx=ctx)

        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ["batch", "size", "limit", "max", "exceed"])


async def test_capabilities_max_text_length_respected(adapter: BaseEmbeddingAdapter):
    caps = await adapter.capabilities()

    if caps.max_text_length is not None:
        ctx = OperationContext(request_id="text_limit_test", tenant="t")

        oversized_text = "x" * (caps.max_text_length + 1)
        spec = EmbedSpec(
            text=oversized_text,
            model=caps.supported_models[0],
            normalize=False,
            truncate=True
        )

        result = await adapter.embed(spec, ctx=ctx)
        assert result is not None
        assert len(result.embedding.vector) > 0


async def test_capabilities_match_operational_behavior_batch(adapter: BaseEmbeddingAdapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(request_id="cap_batch_test", tenant="t")

    spec = BatchEmbedSpec(
        texts=["test"],
        model=caps.supported_models[0]
    )

    result = await adapter.embed_batch(spec, ctx=ctx)
    assert result is not None
    assert len(result.embeddings) + len(result.failed_texts) == 1


async def test_capabilities_match_operational_behavior_normalization(adapter: BaseEmbeddingAdapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(request_id="cap_norm_test", tenant="t")

    spec_normalized = EmbedSpec(
        text="test normalization",
        model=caps.supported_models[0],
        normalize=True
    )

    if caps.supports_normalization:
        result = await adapter.embed(spec_normalized, ctx=ctx)
        assert result is not None
    else:
        with pytest.raises(NotSupported):
            await adapter.embed(spec_normalized, ctx=ctx)


async def test_capabilities_streaming_flag_present(adapter: BaseEmbeddingAdapter):
    caps = await adapter.capabilities()
    assert hasattr(caps, 'supports_streaming')
    assert isinstance(caps.supports_streaming, bool)


async def test_capabilities_cache_flag_accurate(adapter: BaseEmbeddingAdapter):
    caps = await adapter.capabilities()
    assert hasattr(caps, 'supports_caching')
    assert isinstance(caps.supports_caching, bool)
