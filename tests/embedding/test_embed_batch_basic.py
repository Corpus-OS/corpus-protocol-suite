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


async def supports_batch_embedding(adapter: BaseEmbeddingAdapter) -> bool:
    """Check batch embedding capability."""
    caps: EmbeddingCapabilities = await adapter.capabilities()
    # Default True per spec, but adapters may explicitly disable.
    return getattr(caps, "supports_batch_embedding", True)


async def test_batch_partial_returns_batch_result(adapter: BaseEmbeddingAdapter):
    """§10.3: embed_batch must return valid BatchEmbedResult."""
    if not await supports_batch_embedding(adapter):
        pytest.skip("Batch embedding not supported")

    ctx = OperationContext(request_id="t_batch_ok", tenant="test")
    spec = BatchEmbedSpec(
        texts=["a", "b", "c"],
        model=adapter.supported_models[0],
    )
    res = await adapter.embed_batch(spec, ctx=ctx)
    assert isinstance(res, BatchEmbedResult)
    assert len(res.embeddings) == 3
    assert res.failed_texts == []

    # Validate each embedding (index is optional, so do not require it)
    for i, embedding in enumerate(res.embeddings):
        assert isinstance(embedding, EmbeddingVector)
        assert len(embedding.vector) > 0
        # By protocol, embeddings[i] corresponds to texts[i] unless documented otherwise
        assert embedding.text == spec.texts[i]


async def test_batch_partial_requires_non_empty_model(adapter: BaseEmbeddingAdapter):
    """§10.4: Empty model must raise a model-related error."""
    if not await supports_batch_embedding(adapter):
        pytest.skip("Batch embedding not supported")

    spec = BatchEmbedSpec(texts=["x"], model="")
    # BaseEmbeddingAdapter raises BadRequest; MockEmbeddingAdapter may raise ModelNotAvailable.
    with pytest.raises((BadRequest, ModelNotAvailable)) as exc_info:
        await adapter.embed_batch(spec)

    error_msg = str(exc_info.value).lower()
    assert any(
        term in error_msg for term in ["model", "empty", "invalid", "available", "support"]
    ), f"Error should mention model issue: {error_msg}"


async def test_batch_partial_requires_non_empty_texts(adapter: BaseEmbeddingAdapter):
    """§10.4: Empty texts array may reject the batch or return an empty result."""
    if not await supports_batch_embedding(adapter):
        pytest.skip("Batch embedding not supported")

    spec = BatchEmbedSpec(texts=[], model=adapter.supported_models[0])
    try:
        res = await adapter.embed_batch(spec)
    except BadRequest as exc:
        error_msg = str(exc).lower()
        assert any(term in error_msg for term in ["text", "empty", "invalid"]), (
            f"Error should mention text issue: {error_msg}"
        )
        return

    # Also acceptable: adapter chooses to treat empty batch as a valid no-op.
    assert isinstance(res, BatchEmbedResult)
    assert res.embeddings == []
    assert res.failed_texts == []


async def test_batch_partial_respects_max_batch_size(adapter: BaseEmbeddingAdapter):
    """§10.5: Batch size must respect declared limits."""
    if not await supports_batch_embedding(adapter):
        pytest.skip("Batch embedding not supported")

    caps: EmbeddingCapabilities = await adapter.capabilities()
    if caps.max_batch_size is None:
        pytest.skip("Adapter does not declare max_batch_size")

    big = ["x"] * (caps.max_batch_size + 1)
    spec = BatchEmbedSpec(texts=big, model=caps.supported_models[0])
    with pytest.raises(BadRequest) as exc_info:
        await adapter.embed_batch(spec)

    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ["batch", "size", "limit", "max"]), (
        f"Error should mention batch size: {error_msg}"
    )


async def test_batch_partial_unknown_model_raises_model_not_available(adapter: BaseEmbeddingAdapter):
    """§10.4: Unknown models must raise ModelNotAvailable."""
    if not await supports_batch_embedding(adapter):
        pytest.skip("Batch embedding not supported")

    spec = BatchEmbedSpec(texts=["x"], model="nope-model")
    with pytest.raises(ModelNotAvailable) as exc_info:
        await adapter.embed_batch(spec)

    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ["model", "available", "support"]), (
        f"Error should mention model: {error_msg}"
    )


async def test_batch_partial_partial_failure_reporting(adapter: BaseEmbeddingAdapter):
    """§12.5: Partial failures should report per-item errors when supported."""
    if not await supports_batch_embedding(adapter):
        pytest.skip("Batch embedding not supported")

    ctx = OperationContext(request_id="t_batch_partial", tenant="test")
    texts = ["ok", "", "also ok", "another valid"]
    spec = BatchEmbedSpec(texts=texts, model=adapter.supported_models[0])

    try:
        res = await adapter.embed_batch(spec, ctx=ctx)
    except BadRequest:
        # Some adapters reject the entire batch on invalid input.
        return

    assert isinstance(res, BatchEmbedResult)
    total_items = len(texts)
    total_processed = len(res.embeddings) + len(res.failed_texts)
    assert total_processed <= total_items

    # For adapters that support partial failures, we expect an entry for index 1.
    if res.failed_texts:
        assert any(f.get("index") == 1 for f in res.failed_texts)

    # Validate failure structure for adapters that report failures.
    for failure in res.failed_texts:
        assert "index" in failure
        assert 0 <= failure["index"] < len(texts)
        assert "error" in failure
        assert "code" in failure
        assert "message" in failure

        assert isinstance(failure["error"], str)
        assert isinstance(failure["code"], str)
        assert failure["code"].isupper()


async def test_batch_partial_single_item_works(adapter: BaseEmbeddingAdapter):
    """§10.3: Single-item batches should work correctly."""
    if not await supports_batch_embedding(adapter):
        pytest.skip("Batch embedding not supported")

    ctx = OperationContext(request_id="t_batch_single", tenant="test")
    spec = BatchEmbedSpec(
        texts=["single item"],
        model=adapter.supported_models[0],
    )
    res = await adapter.embed_batch(spec, ctx=ctx)

    assert isinstance(res, BatchEmbedResult)
    assert len(res.embeddings) == 1
    assert res.failed_texts == []

    emb = res.embeddings[0]
    assert isinstance(emb, EmbeddingVector)
    # Index is optional; if present it should be 0
    if emb.index is not None:
        assert emb.index == 0
    assert emb.text == "single item"


async def test_batch_partial_ordering_preserved(adapter: BaseEmbeddingAdapter):
    """§12.5: Batch results must preserve input ordering."""
    if not await supports_batch_embedding(adapter):
        pytest.skip("Batch embedding not supported")

    ctx = OperationContext(request_id="t_batch_order", tenant="test")
    texts = ["first", "second", "third"]
    spec = BatchEmbedSpec(texts=texts, model=adapter.supported_models[0])

    res = await adapter.embed_batch(spec, ctx=ctx)

    # By default, embeddings[i] should correspond to texts[i].
    assert isinstance(res, BatchEmbedResult)
    assert len(res.embeddings) == len(texts)

    for i, embedding in enumerate(res.embeddings):
        assert embedding.text == texts[i]


async def test_batch_partial_empty_strings_handled_consistently(adapter: BaseEmbeddingAdapter):
    """§12.5: Empty strings should be handled consistently across batch."""
    if not await supports_batch_embedding(adapter):
        pytest.skip("Batch embedding not supported")

    ctx = OperationContext(request_id="t_batch_empty", tenant="test")
    texts = ["", "valid", ""]
    spec = BatchEmbedSpec(texts=texts, model=adapter.supported_models[0])

    try:
        res = await adapter.embed_batch(spec, ctx=ctx)
    except BadRequest:
        # Adapter chooses to reject batches with empty strings entirely.
        return

    assert isinstance(res, BatchEmbedResult)

    empty_indices = {0, 2}

    # Successful embeddings for empty positions (where index is known)
    empty_success = {
        e.index
        for e in res.embeddings
        if e.index is not None and e.index in empty_indices
    }

    # Failures for empty positions
    empty_failures = {
        f.get("index")
        for f in res.failed_texts
        if f.get("index") in empty_indices
    }

    # No index should be simultaneously considered success and failure.
    assert empty_success.isdisjoint(empty_failures), "Empty strings handled inconsistently"


async def test_batch_partial_not_supported_raises_clear_error(adapter: BaseEmbeddingAdapter):
    """§10.4: Batch must raise NotSupported when capability is false."""
    if await supports_batch_embedding(adapter):
        pytest.skip("Adapter supports batch embedding")

    spec = BatchEmbedSpec(texts=["test"], model=adapter.supported_models[0])
    with pytest.raises(NotSupported) as exc_info:
        await adapter.embed_batch(spec)

    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ["batch", "support", "implement"]), (
        f"Error should mention batch support: {error_msg}"
    )
