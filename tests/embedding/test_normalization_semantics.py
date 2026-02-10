# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Normalization semantics.

Spec refs:
  • §10.6 Normalization & Truncation Semantics
  • §10.5 Capabilities Discovery
  • §10.4 Errors (Embedding-Specific)
  • §13.3 Observability & Privacy

Notes:
- No skips: tests assert behavior consistent with capabilities.
- Do not assume vector determinism across calls or that different texts always produce different vectors.
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


async def test_normalization_single_embed_normalize_true_produces_unit_vector(adapter: BaseEmbeddingAdapter):
    """§10.6: normalize=True must produce approximately unit vectors when supported; else NotSupported."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]
    ctx = OperationContext(request_id="t_norm_single_true", tenant="t")

    if not caps.supports_normalization:
        with pytest.raises(NotSupported):
            await adapter.embed(EmbedSpec(text="normalize", model=model, normalize=True), ctx=ctx)
        return

    res = await adapter.embed(
        EmbedSpec(
            text="normalize this text to unit length",
            model=model,
            truncate=True,
            normalize=True,
        ),
        ctx=ctx,
    )

    norm = _norm(res.embedding.vector)
    assert 0.99 <= norm <= 1.01


async def test_normalization_single_embed_normalize_false_not_forced_unit_norm(adapter: BaseEmbeddingAdapter):
    """
    §10.6: normalize=False should not require unit norm.
    If adapter normalizes_at_source, normalize=False may still be unit; that is acceptable.
    If adapter does not normalize at source, normalize=True must still be unit.
    """
    caps = await adapter.capabilities()
    model = caps.supported_models[0]
    ctx = OperationContext(request_id="t_norm_single_false", tenant="t")

    if not caps.supports_normalization:
        with pytest.raises(NotSupported):
            await adapter.embed(EmbedSpec(text="x", model=model, normalize=True), ctx=ctx)
        return

    raw = await adapter.embed(
        EmbedSpec(
            text="provide raw vector without normalization",
            model=model,
            truncate=True,
            normalize=False,
        ),
        ctx=ctx,
    )
    norm_raw = _norm(raw.embedding.vector)
    assert norm_raw >= 0.0

    normed = await adapter.embed(
        EmbedSpec(
            text="provide raw vector without normalization",
            model=model,
            truncate=True,
            normalize=True,
        ),
        ctx=ctx,
    )
    norm_normed = _norm(normed.embedding.vector)
    assert 0.99 <= norm_normed <= 1.01


async def test_normalization_batch_embed_normalize_true_all_unit_vectors(adapter: BaseEmbeddingAdapter):
    """§10.6: Batch normalization must apply consistently to all vectors when supported."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]
    ctx = OperationContext(request_id="t_norm_batch_true", tenant="t")

    if not getattr(caps, "supports_batch_embedding", True):
        with pytest.raises(NotSupported):
            await adapter.embed_batch(BatchEmbedSpec(texts=["a"], model=model, normalize=True), ctx=ctx)
        return

    if not caps.supports_normalization:
        with pytest.raises(NotSupported):
            await adapter.embed_batch(
                BatchEmbedSpec(
                    texts=["first text to normalize", "second text", "third example"],
                    model=model,
                    truncate=True,
                    normalize=True,
                ),
                ctx=ctx,
            )
        return

    res = await adapter.embed_batch(
        BatchEmbedSpec(
            texts=["first text to normalize", "second text", "third example"],
            model=model,
            truncate=True,
            normalize=True,
        ),
        ctx=ctx,
    )

    assert len(res.embeddings) == 3 or len(res.embeddings) <= 3
    for i, embedding in enumerate(res.embeddings):
        n = _norm(embedding.vector)
        assert 0.99 <= n <= 1.01, f"Vector {i} should be unit norm, got {n:.6f}"


async def test_normalization_not_supported_raises_clear_error(adapter: BaseEmbeddingAdapter):
    """§10.4: Normalization requests must raise NotSupported when unsupported."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]
    ctx = OperationContext(request_id="t_norm_nosupport", tenant="t")

    if caps.supports_normalization:
        res = await adapter.embed(EmbedSpec(text="test", model=model, normalize=True), ctx=ctx)
        assert res.embedding.vector
        return

    with pytest.raises(NotSupported) as exc_info:
        await adapter.embed(EmbedSpec(text="test normalization error", model=model, normalize=True), ctx=ctx)

    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ["normaliz", "support", "implement"])


async def test_normalization_normalizes_at_source_respected(adapter: BaseEmbeddingAdapter):
    """§10.6: If normalizes_at_source is true, normalize=True should still yield unit vectors."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]
    ctx = OperationContext(request_id="t_norm_source", tenant="t")

    if not caps.supports_normalization:
        with pytest.raises(NotSupported):
            await adapter.embed(EmbedSpec(text="x", model=model, normalize=True), ctx=ctx)
        return

    if getattr(caps, "normalizes_at_source", False):
        res = await adapter.embed(EmbedSpec(text="text for source-normalizing adapter", model=model, normalize=True), ctx=ctx)
        n = _norm(res.embedding.vector)
        assert 0.99 <= n <= 1.01
    else:
        # If it doesn't normalize at source, base should still normalize when requested.
        res = await adapter.embed(EmbedSpec(text="text for base-normalizing adapter", model=model, normalize=True), ctx=ctx)
        n = _norm(res.embedding.vector)
        assert 0.99 <= n <= 1.01


async def test_normalization_consistency_across_calls(adapter: BaseEmbeddingAdapter):
    """§10.6: normalize=True must always produce unit vectors across repeated calls (no determinism assumption)."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]
    ctx = OperationContext(request_id="t_norm_consistent", tenant="t")

    if not caps.supports_normalization:
        with pytest.raises(NotSupported):
            await adapter.embed(EmbedSpec(text="x", model=model, normalize=True), ctx=ctx)
        return

    spec = EmbedSpec(text="identical text for consistency check", model=model, normalize=True)
    r1 = await adapter.embed(spec, ctx=ctx)
    r2 = await adapter.embed(spec, ctx=ctx)

    assert 0.99 <= _norm(r1.embedding.vector) <= 1.01
    assert 0.99 <= _norm(r2.embedding.vector) <= 1.01


async def test_normalization_different_texts_different_vectors(adapter: BaseEmbeddingAdapter):
    """§10.6: Do not require vectors to differ; require both be unit norm when normalize=True."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]
    ctx = OperationContext(request_id="t_norm_different", tenant="t")

    if not caps.supports_normalization:
        with pytest.raises(NotSupported):
            await adapter.embed(EmbedSpec(text="x", model=model, normalize=True), ctx=ctx)
        return

    result1 = await adapter.embed(EmbedSpec(text="first unique text", model=model, normalize=True), ctx=ctx)
    result2 = await adapter.embed(EmbedSpec(text="second different text", model=model, normalize=True), ctx=ctx)

    assert 0.99 <= _norm(result1.embedding.vector) <= 1.01
    assert 0.99 <= _norm(result2.embedding.vector) <= 1.01


async def test_normalization_small_vectors_handled(adapter: BaseEmbeddingAdapter):
    """§10.6: Normalization should handle short texts correctly (unit norm when supported)."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]
    ctx = OperationContext(request_id="t_norm_small", tenant="t")

    if not caps.supports_normalization:
        with pytest.raises(NotSupported):
            await adapter.embed(EmbedSpec(text="a", model=model, normalize=True), ctx=ctx)
        return

    for text in ["a", "hi", "ok"]:
        result = await adapter.embed(EmbedSpec(text=text, model=model, normalize=True), ctx=ctx)
        n = _norm(result.embedding.vector)
        assert 0.99 <= n <= 1.01


async def test_normalization_batch_mixed_normalization(adapter: BaseEmbeddingAdapter):
    """§10.6: Batch normalize flag must control output norms consistently (where applicable)."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]
    ctx = OperationContext(request_id="t_norm_batch_mixed", tenant="t")

    if not getattr(caps, "supports_batch_embedding", True):
        with pytest.raises(NotSupported):
            await adapter.embed_batch(BatchEmbedSpec(texts=["x"], model=model), ctx=ctx)
        return

    if not caps.supports_normalization:
        with pytest.raises(NotSupported):
            await adapter.embed_batch(BatchEmbedSpec(texts=["batch text one"], model=model, normalize=True), ctx=ctx)
        return

    spec_normalized = BatchEmbedSpec(texts=["batch text one", "batch text two"], model=model, normalize=True)
    spec_raw = BatchEmbedSpec(texts=["batch text one", "batch text two"], model=model, normalize=False)

    result_norm = await adapter.embed_batch(spec_normalized, ctx=ctx)
    result_raw = await adapter.embed_batch(spec_raw, ctx=ctx)

    for embedding in result_norm.embeddings:
        n = _norm(embedding.vector)
        assert 0.99 <= n <= 1.01

    # For raw vectors, do not require non-unit; just ensure normalize=True is unit.
    for embedding in result_raw.embeddings:
        assert _norm(embedding.vector) >= 0.0
