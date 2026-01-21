# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Single embed behavior.

Spec refs:
  • §10.3 embed() — basic contract
  • §10.4 Errors (Embedding-Specific)
  • §10.5 Capabilities Discovery
  • §12 Error Handling — BadRequest, ModelNotAvailable, TextTooLong
  • §10.6 Normalization & Truncation Semantics

Notes:
- Models must be sourced from capabilities().supported_models (protocol surface),
  not from adapter attributes.
- Do not assume embedding determinism across calls for real providers.
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

pytestmark = pytest.mark.asyncio


async def test_core_ops_embed_returns_valid_embedding_structure(adapter: BaseEmbeddingAdapter):
    """§10.3: embed() must return valid EmbedResult with correct structure."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    ctx = OperationContext(request_id="t_embed_structure", tenant="test")
    spec = EmbedSpec(text="hello world", model=model)
    res = await adapter.embed(spec, ctx=ctx)

    assert res.embedding.vector, "Embedding vector cannot be empty"
    assert isinstance(res.embedding.vector, list), "Vector must be a list"
    assert all(isinstance(v, (int, float)) for v in res.embedding.vector), "Vector must contain numbers"

    assert res.embedding.dimensions == len(res.embedding.vector), "Dimensions must match vector length"
    assert res.embedding.dimensions > 0, "Dimensions must be positive"

    assert res.embedding.text == "hello world", "Original text must be preserved"
    assert res.model == spec.model, "Result model must match spec model"

    assert isinstance(res.truncated, bool), "truncated must be boolean"


async def test_core_ops_embed_requires_valid_text(adapter: BaseEmbeddingAdapter):
    """§10.4: embed() must validate text input and raise BadRequest for invalid values."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    invalid_cases = [
        ("", "empty text"),
        (None, "null text"),  # type: ignore[arg-type]
        ("   ", "whitespace-only text"),
    ]

    for text, description in invalid_cases:
        spec = EmbedSpec(text=text, model=model)  # type: ignore[arg-type]
        with pytest.raises(BadRequest) as exc_info:
            await adapter.embed(spec)

        error_msg = str(exc_info.value).lower()
        assert any(term in error_msg for term in ["text", "input", "invalid", "empty"]), (
            f"BadRequest should mention text issue for {description}: {error_msg}"
        )


async def test_core_ops_embed_requires_valid_model(adapter: BaseEmbeddingAdapter):
    """§10.4: embed() must validate model parameter."""
    invalid_models = ["", "invalid-model-12345", "unknown/model/name"]
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
    assert any(term in error_msg for term in ["model", "available", "support", "unknown"]), (
        f"Error should mention model issue: {error_msg}"
    )


async def test_core_ops_embed_truncation_behavior_matches_capabilities(adapter: BaseEmbeddingAdapter):
    """§10.6: Truncation behavior must match declared capabilities."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    # If max_text_length is declared, it must be enforced consistently.
    if caps.max_text_length is not None:
        long_text = "x" * (caps.max_text_length + 100)

        # truncate=False must raise TextTooLong
        with pytest.raises(TextTooLong):
            await adapter.embed(EmbedSpec(text=long_text, model=model, truncate=False))

        # truncate=True behavior depends on supports_truncation
        if caps.supports_truncation:
            result = await adapter.embed(EmbedSpec(text=long_text, model=model, truncate=True))
            assert result.truncated is True
            assert len(result.embedding.text) <= len(long_text)
        else:
            with pytest.raises((TextTooLong, NotSupported)):
                await adapter.embed(EmbedSpec(text=long_text, model=model, truncate=True))
    else:
        # If no max_text_length declared, adapter must not raise TextTooLong for length alone.
        # If it does, capabilities are inconsistent.
        long_text = "x" * 20000
        try:
            await adapter.embed(EmbedSpec(text=long_text, model=model, truncate=False))
        except TextTooLong:
            raise AssertionError("Adapter raised TextTooLong but capabilities.max_text_length is None")


async def test_core_ops_embed_normalization_produces_unit_vectors(adapter: BaseEmbeddingAdapter):
    """§10.6: normalize=True must produce vectors with unit length when supported; else NotSupported."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    if not caps.supports_normalization:
        with pytest.raises(NotSupported):
            await adapter.embed(EmbedSpec(text="normalize this", model=model, normalize=True))
        return

    test_texts = [
        "normalize this text",
        "short",
        "a longer piece of text that should be normalized properly",
    ]

    for text in test_texts:
        result = await adapter.embed(EmbedSpec(text=text, model=model, normalize=True))
        vector = result.embedding.vector
        norm = math.sqrt(sum(v * v for v in vector))
        assert 0.99 <= norm <= 1.01, f"Normalized vector should have unit length, got {norm} for text: '{text}'"


async def test_core_ops_embed_normalization_unsupported_raises_clear_error(adapter: BaseEmbeddingAdapter):
    """§10.4: Normalization requests must raise clear error when unsupported (or succeed if supported)."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    if caps.supports_normalization:
        res = await adapter.embed(EmbedSpec(text="test", model=model, normalize=True))
        assert res.embedding.vector
        return

    with pytest.raises(NotSupported) as exc_info:
        await adapter.embed(EmbedSpec(text="test", model=model, normalize=True))

    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ["normaliz", "support", "implement"]), (
        f"Error should mention normalization: {error_msg}"
    )


async def test_core_ops_embed_vector_quality_and_consistency(adapter: BaseEmbeddingAdapter):
    """§10.3: Enforce contract-safe validity and dimension stability (no determinism assumptions)."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    r1 = await adapter.embed(EmbedSpec(text="consistent embedding", model=model))
    r2 = await adapter.embed(EmbedSpec(text="consistent embedding", model=model))
    r3 = await adapter.embed(EmbedSpec(text="different text", model=model))

    for r in (r1, r2, r3):
        assert isinstance(r.embedding.vector, list) and r.embedding.vector
        assert all(isinstance(v, (int, float)) for v in r.embedding.vector)
        assert r.embedding.dimensions == len(r.embedding.vector)

    assert r1.embedding.dimensions == r2.embedding.dimensions == r3.embedding.dimensions, (
        "Embedding dimensions should be stable for a given model"
    )


async def test_core_ops_embed_special_character_handling(adapter: BaseEmbeddingAdapter):
    """§10.3: embed() should handle special characters and Unicode correctly."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

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
        result = await adapter.embed(EmbedSpec(text=text, model=model))
        assert result.embedding.vector, f"Failed for text: {repr(text)}"
        assert all(isinstance(v, (int, float)) for v in result.embedding.vector)


async def test_core_ops_embed_context_propagation(adapter: BaseEmbeddingAdapter):
    """§6.1: Operation context should be accepted and not affect correctness."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    ctx = OperationContext(
        request_id="test_context_123",
        tenant="test-tenant",
        deadline_ms=int(__import__("time").time() * 1000) + 5000,
        attrs={"k": "v"},
    )

    result = await adapter.embed(EmbedSpec(text="context test", model=model), ctx=ctx)
    assert result.embedding.vector


async def test_core_ops_embed_dimensions_consistent_with_capabilities(adapter: BaseEmbeddingAdapter):
    """§10.5: Embedding dimensions should be consistent with capabilities."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    test_texts = ["short", "medium length text", "a longer piece of text for testing dimensions"]

    for text in test_texts:
        result = await adapter.embed(EmbedSpec(text=text, model=model))
        dimensions = result.embedding.dimensions
        vector_length = len(result.embedding.vector)

        assert dimensions == vector_length
        if caps.max_dimensions:
            assert dimensions <= caps.max_dimensions
        assert dimensions > 0
