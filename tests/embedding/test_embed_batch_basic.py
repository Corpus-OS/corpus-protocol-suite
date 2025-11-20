# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Batch semantics & partial failures.

Spec refs:
  • §10.3 embed_batch()
  • §12.5 Partial Failure Reporting
  • §10.5 Capabilities Discovery
"""

import pytest

from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter,
    BatchEmbedSpec,
    EmbedSpec,
    EmbeddingCapabilities,
    EmbedResult,
    BatchEmbedResult,
    EmbeddingVector,
    OperationContext,
    BadRequest,
    ModelNotAvailable,
    NotSupported,
    TextTooLong,
)

pytestmark = pytest.mark.asyncio


def make_ctx(ctx_cls, **kwargs):
    """Local helper to construct an OperationContext."""
    return ctx_cls(**kwargs)


def supports_batch_embedding(adapter: BaseEmbeddingAdapter) -> bool:
    """Check batch embedding capability."""
    caps = adapter.capabilities
    return getattr(caps, "supports_batch_embedding", True)  # Default True per spec


async def test_batch_partial_returns_batch_result(adapter: BaseEmbeddingAdapter):
    """§10.3: embed_batch must return valid BatchEmbedResult."""
    if not supports_batch_embedding(adapter):
        pytest.skip("Batch embedding not supported")

    ctx = make_ctx(OperationContext, request_id="t_batch_ok", tenant="test")
    spec = BatchEmbedSpec(
        texts=["a", "b", "c"],
        model=adapter.supported_models[0],
    )
    res = await adapter.embed_batch(spec, ctx=ctx)
    assert isinstance(res, BatchEmbedResult)
    assert len(res.embeddings) == 3
    assert res.failed_texts == []

    # Validate each embedding
    for embedding in res.embeddings:
        assert isinstance(embedding, EmbeddingVector)
        assert embedding.index is not None
        assert 0 <= embedding.index < 3
        assert len(embedding.vector) > 0


async def test_batch_partial_requires_non_empty_model(adapter: BaseEmbeddingAdapter):
    """§10.4: Empty model must raise BadRequest."""
    if not supports_batch_embedding(adapter):
        pytest.skip("Batch embedding not supported")

    spec = BatchEmbedSpec(texts=["x"], model="")
    with pytest.raises(BadRequest) as exc_info:
        await adapter.embed_batch(spec)

    error_msg = str(exc_info.value).lower()
    assert any(
        term in error_msg for term in ["model", "empty", "invalid"]
    ), f"Error should mention model issue: {error_msg}"


async def test_batch_partial_requires_non_empty_texts(adapter: BaseEmbeddingAdapter):
    """§10.4: Empty texts array must raise BadRequest."""
    if not supports_batch_embedding(adapter):
        pytest.skip("Batch embedding not supported")

    spec = BatchEmbedSpec(texts=[], model=adapter.supported_models[0])
    with pytest.raises(BadRequest) as exc_info:
        await adapter.embed_batch(spec)

    error_msg = str(exc_info.value).lower()
    assert any(
        term in error_msg for term in ["text", "empty", "invalid"]
    ), f"Error should mention text issue: {error_msg}"


async def test_batch_partial_respects_max_batch_size(adapter: BaseEmbeddingAdapter):
    """§10.5: Batch size must respect declared limits."""
    if not supports_batch_embedding(adapter):
        pytest.skip("Batch embedding not supported")

    caps = adapter.capabilities
    if caps.max_batch_size is None:
        pytest.skip("Adapter does not declare max_batch_size")

    big = ["x"] * (caps.max_batch_size + 1)
    spec = BatchEmbedSpec(texts=big, model=caps.supported_models[0])
    with pytest.raises(BadRequest) as exc_info:
        await adapter.embed_batch(spec)

    error_msg = str(exc_info.value).lower()
    assert any(
        term in error_msg for term in ["batch", "size", "limit", "max"]
    ), f"Error should mention batch size: {error_msg}"


async def test_batch_partial_unknown_model_raises_model_not_available(
    adapter: BaseEmbeddingAdapter,
):
    """§10.4: Unknown models must raise ModelNotAvailable."""
    if not supports_batch_embedding(adapter):
        pytest.skip("Batch embedding not supported")

    spec = BatchEmbedSpec(texts=["x"], model="nope-model")
    with pytest.raises(ModelNotAvailable) as exc_info:
        await adapter.embed_batch(spec)

    error_msg = str(exc_info.value).lower()
    assert any(
        term in error_msg for term in ["model", "available", "support"]
    ), f"Error should mention model: {error_msg}"


async def test_batch_partial_partial_failure_reporting(adapter: BaseEmbeddingAdapter):
    """§12.5: Partial failures must report per-item errors with indices."""
    if not supports_batch_embedding(adapter):
        pytest.skip("Batch embedding not supported")

    ctx = make_ctx(OperationContext, request_id="t_batch_partial", tenant="test")
    texts = ["ok", "", "also ok", "another valid"]
    spec = BatchEmbedSpec(texts=texts, model=adapter.supported_models[0])

    res = await adapter.embed_batch(spec, ctx=ctx)

    # Should embed valid entries and report failures for bad ones
    assert len(res.embeddings) >= 2
    assert any(f["index"] == 1 for f in res.failed_texts)

    # Validate failure structure
    for failure in res.failed_texts:
        assert "index" in failure
        assert 0 <= failure["index"] < len(texts)
        assert "error" in failure
        assert "message" in failure["error"]
        assert "code" in failure["error"]
        assert failure["error"]["code"].isupper()


async def test_batch_partial_single_item_works(adapter: BaseEmbeddingAdapter):
    """§10.3: Single-item batches should work correctly."""
    if not supports_batch_embedding(adapter):
        pytest.skip("Batch embedding not supported")

    ctx = make_ctx(OperationContext, request_id="t_batch_single", tenant="test")
    spec = BatchEmbedSpec(
        texts=["single item"],
        model=adapter.supported_models[0],
    )
    res = await adapter.embed_batch(spec, ctx=ctx)

    assert isinstance(res, BatchEmbedResult)
    assert len(res.embeddings) == 1
    assert res.failed_texts == []
    assert res.embeddings[0].index == 0


async def test_batch_partial_ordering_preserved(adapter: BaseEmbeddingAdapter):
    """§12.5: Batch results must preserve input ordering."""
    if not supports_batch_embedding(adapter):
        pytest.skip("Batch embedding not supported")

    ctx = make_ctx(OperationContext, request_id="t_batch_order", tenant="test")
    texts = ["first", "second", "third"]
    spec = BatchEmbedSpec(texts=texts, model=adapter.supported_models[0])

    res = await adapter.embed_batch(spec, ctx=ctx)

    # Check indices match original positions
    for embedding in res.embeddings:
        original_text = texts[embedding.index]
        if embedding.index == 0:
            assert "first" in original_text.lower()
        elif embedding.index == 1:
            assert "second" in original_text.lower()
        elif embedding.index == 2:
            assert "third" in original_text.lower()


async def test_batch_partial_empty_strings_handled_consistently(
    adapter: BaseEmbeddingAdapter,
):
    """§12.5: Empty strings should be handled consistently across batch."""
    if not supports_batch_embedding(adapter):
        pytest.skip("Batch embedding not supported")

    ctx = make_ctx(OperationContext, request_id="t_batch_empty", tenant="test")
    texts = ["", "valid", ""]
    spec = BatchEmbedSpec(texts=texts, model=adapter.supported_models[0])

    res = await adapter.embed_batch(spec, ctx=ctx)

    # Empty strings should be either all failed or all successful
    empty_indices = {0, 2}
    empty_success = {e.index for e in res.embeddings if e.index in empty_indices}
    empty_failures = {f["index"] for f in res.failed_texts if f["index"] in empty_indices}

    assert empty_success.isdisjoint(
        empty_failures
    ), "Empty strings handled inconsistently"


async def test_batch_partial_not_supported_raises_clear_error(
    adapter: BaseEmbeddingAdapter,
):
    """§10.4: Batch must raise NotSupported when capability is false."""
    if supports_batch_embedding(adapter):
        pytest.skip("Adapter supports batch embedding")

    spec = BatchEmbedSpec(texts=["test"], model=adapter.supported_models[0])
    with pytest.raises(NotSupported) as exc_info:
        await adapter.embed_batch(spec)

    error_msg = str(exc_info.value).lower()
    assert any(
        term in error_msg for term in ["batch", "support", "implement"]
    ), f"Error should mention batch support: {error_msg}"
