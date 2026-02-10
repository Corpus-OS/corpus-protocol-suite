# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Batch semantics & partial failures.

Spec refs:
  • §10.3 embed_batch()
  • §12.5 Partial Failure Reporting
  • §10.5 Capabilities Discovery

Notes:
- No skips: tests assert behavior consistent with capabilities.
- Some adapters reject invalid batches entirely (fail-fast); others return partial failures.
  Both are acceptable if consistent with the adapter’s contract.
"""

import pytest

from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter,
    BatchEmbedSpec,
    BatchEmbedResult,
    EmbeddingVector,
    OperationContext,
    BadRequest,
    ModelNotAvailable,
    NotSupported,
)

pytestmark = pytest.mark.asyncio


async def test_batch_partial_returns_batch_result(adapter: BaseEmbeddingAdapter):
    """§10.3: embed_batch must return valid BatchEmbedResult when supported; else NotSupported."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    ctx = OperationContext(request_id="t_batch_ok", tenant="test")
    spec = BatchEmbedSpec(texts=["a", "b", "c"], model=model)

    if not getattr(caps, "supports_batch_embedding", True):
        with pytest.raises(NotSupported):
            await adapter.embed_batch(spec, ctx=ctx)
        return

    res = await adapter.embed_batch(spec, ctx=ctx)
    assert isinstance(res, BatchEmbedResult)
    assert len(res.embeddings) == 3
    assert res.failed_texts == []

    for i, embedding in enumerate(res.embeddings):
        assert isinstance(embedding, EmbeddingVector)
        assert len(embedding.vector) > 0
        assert embedding.text == spec.texts[i]


async def test_batch_partial_requires_non_empty_model(adapter: BaseEmbeddingAdapter):
    """§10.4: Empty model must raise a model-related error when batch is supported."""
    caps = await adapter.capabilities()
    model_any = caps.supported_models[0]

    spec = BatchEmbedSpec(texts=["x"], model="")
    if not getattr(caps, "supports_batch_embedding", True):
        with pytest.raises(NotSupported):
            await adapter.embed_batch(BatchEmbedSpec(texts=["x"], model=model_any))
        return

    with pytest.raises((BadRequest, ModelNotAvailable)) as exc_info:
        await adapter.embed_batch(spec)

    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ["model", "empty", "invalid", "available", "support"])


async def test_batch_partial_requires_non_empty_texts(adapter: BaseEmbeddingAdapter):
    """§10.4: Empty texts array may reject the batch or return an empty result (must be consistent)."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    spec = BatchEmbedSpec(texts=[], model=model)

    if not getattr(caps, "supports_batch_embedding", True):
        with pytest.raises(NotSupported):
            await adapter.embed_batch(BatchEmbedSpec(texts=["x"], model=model))
        return

    try:
        res = await adapter.embed_batch(spec)
    except BadRequest as exc:
        error_msg = str(exc).lower()
        assert any(term in error_msg for term in ["text", "texts", "empty", "invalid"])
        return

    assert isinstance(res, BatchEmbedResult)
    assert res.embeddings == []
    assert res.failed_texts == []


async def test_batch_partial_respects_max_batch_size(adapter: BaseEmbeddingAdapter):
    """§10.5: If max_batch_size is declared, exceeding it must raise BadRequest; else it must not."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    if not getattr(caps, "supports_batch_embedding", True):
        with pytest.raises(NotSupported):
            await adapter.embed_batch(BatchEmbedSpec(texts=["x"], model=model))
        return

    if caps.max_batch_size is not None:
        big = ["x"] * (caps.max_batch_size + 1)
        with pytest.raises(BadRequest) as exc_info:
            await adapter.embed_batch(BatchEmbedSpec(texts=big, model=model))
        error_msg = str(exc_info.value).lower()
        assert any(term in error_msg for term in ["batch", "size", "limit", "max"])
    else:
        # If no limit declared, a modest batch should succeed and must not raise a size-limit BadRequest.
        res = await adapter.embed_batch(BatchEmbedSpec(texts=["x"] * 8, model=model))
        assert isinstance(res, BatchEmbedResult)


async def test_batch_partial_unknown_model_raises_model_not_available(adapter: BaseEmbeddingAdapter):
    """§10.4: Unknown models must raise ModelNotAvailable (or NotSupported if batch unsupported)."""
    caps = await adapter.capabilities()
    model_any = caps.supported_models[0]

    if not getattr(caps, "supports_batch_embedding", True):
        with pytest.raises(NotSupported):
            await adapter.embed_batch(BatchEmbedSpec(texts=["x"], model=model_any))
        return

    with pytest.raises(ModelNotAvailable) as exc_info:
        await adapter.embed_batch(BatchEmbedSpec(texts=["x"], model="nope-model"))

    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ["model", "available", "support"])


async def test_batch_partial_partial_failure_reporting(adapter: BaseEmbeddingAdapter):
    """§12.5: Invalid items may fail fast or be reported per-item; either is acceptable."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    ctx = OperationContext(request_id="t_batch_partial", tenant="test")
    texts = ["ok", "", "also ok", "another valid"]
    spec = BatchEmbedSpec(texts=texts, model=model)

    if not getattr(caps, "supports_batch_embedding", True):
        with pytest.raises(NotSupported):
            await adapter.embed_batch(spec, ctx=ctx)
        return

    try:
        res = await adapter.embed_batch(spec, ctx=ctx)
    except BadRequest:
        # Fail-fast is acceptable.
        return

    assert isinstance(res, BatchEmbedResult)
    total_items = len(texts)
    total_processed = len(res.embeddings) + len(res.failed_texts)
    assert total_processed <= total_items

    if res.failed_texts:
        assert any(f.get("index") == 1 for f in res.failed_texts)

    for failure in res.failed_texts:
        assert "index" in failure and isinstance(failure["index"], int)
        assert 0 <= failure["index"] < len(texts)
        assert "error" in failure and isinstance(failure["error"], str)
        assert "code" in failure and isinstance(failure["code"], str)
        assert "message" in failure and isinstance(failure["message"], str)
        assert failure["code"].isupper()


async def test_batch_partial_single_item_works(adapter: BaseEmbeddingAdapter):
    """§10.3: Single-item batches should work correctly when supported; else NotSupported."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    ctx = OperationContext(request_id="t_batch_single", tenant="test")
    spec = BatchEmbedSpec(texts=["single item"], model=model)

    if not getattr(caps, "supports_batch_embedding", True):
        with pytest.raises(NotSupported):
            await adapter.embed_batch(spec, ctx=ctx)
        return

    res = await adapter.embed_batch(spec, ctx=ctx)
    assert isinstance(res, BatchEmbedResult)
    assert len(res.embeddings) == 1
    assert res.failed_texts == []

    emb = res.embeddings[0]
    if emb.index is not None:
        assert emb.index == 0
    assert emb.text == "single item"


async def test_batch_partial_ordering_preserved(adapter: BaseEmbeddingAdapter):
    """§12.5: If embeddings are returned, default contract is positional alignment unless documented otherwise."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    ctx = OperationContext(request_id="t_batch_order", tenant="test")
    texts = ["first", "second", "third"]
    spec = BatchEmbedSpec(texts=texts, model=model)

    if not getattr(caps, "supports_batch_embedding", True):
        with pytest.raises(NotSupported):
            await adapter.embed_batch(spec, ctx=ctx)
        return

    res = await adapter.embed_batch(spec, ctx=ctx)

    # If an implementation returns only successes, length may be < len(texts); do not require equality.
    # But if it returns full-length embeddings, ordering must match.
    if len(res.embeddings) == len(texts):
        for i, embedding in enumerate(res.embeddings):
            assert embedding.text == texts[i]


async def test_batch_partial_empty_strings_handled_consistently(adapter: BaseEmbeddingAdapter):
    """§12.5: Empty strings may fail-fast or be recorded as failures; either is acceptable if consistent."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    ctx = OperationContext(request_id="t_batch_empty", tenant="test")
    texts = ["", "valid", ""]
    spec = BatchEmbedSpec(texts=texts, model=model)

    if not getattr(caps, "supports_batch_embedding", True):
        with pytest.raises(NotSupported):
            await adapter.embed_batch(spec, ctx=ctx)
        return

    try:
        res = await adapter.embed_batch(spec, ctx=ctx)
    except BadRequest:
        return

    assert isinstance(res, BatchEmbedResult)

    empty_indices = {0, 2}
    empty_success = {e.index for e in res.embeddings if e.index is not None and e.index in empty_indices}
    empty_failures = {f.get("index") for f in res.failed_texts if f.get("index") in empty_indices}
    assert empty_success.isdisjoint(empty_failures), "Empty strings handled inconsistently"


async def test_batch_partial_not_supported_raises_clear_error(adapter: BaseEmbeddingAdapter):
    """§10.4: Batch must raise NotSupported when capability is false; otherwise it must succeed."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    spec = BatchEmbedSpec(texts=["test"], model=model)

    if getattr(caps, "supports_batch_embedding", True):
        res = await adapter.embed_batch(spec)
        assert isinstance(res, BatchEmbedResult)
        return

    with pytest.raises(NotSupported) as exc_info:
        await adapter.embed_batch(spec)

    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ["batch", "support", "implement"])
