# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Batch semantics & partial failures.

Spec refs:
  • §10.3 embed_batch()
  • §12.5 Partial Failure Reporting
"""

import pytest

from corpus_sdk.embedding.embedding_base import (
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
)
from corpus_sdk.examples.embedding.mock_embedding_adapter import MockEmbeddingAdapter
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_embed_batch_returns_batch_result():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_batch_ok", tenant="test")
    spec = BatchEmbedSpec(
        texts=["a", "b", "c"],
        model=a.supported_models[0],
    )
    res = await a.embed_batch(spec, ctx=ctx)
    assert isinstance(res, BatchEmbedResult)
    assert len(res.embeddings) == 3
    assert res.failed_texts == []


async def test_embed_batch_requires_non_empty_model():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    spec = BatchEmbedSpec(texts=["x"], model="")
    with pytest.raises(BadRequest):
        await a.embed_batch(spec)


async def test_embed_batch_requires_non_empty_texts():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    spec = BatchEmbedSpec(texts=[], model=a.supported_models[0])
    with pytest.raises(BadRequest):
        await a.embed_batch(spec)


async def test_embed_batch_respects_max_batch_size():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    caps = await a.capabilities()
    big = ["x"] * (caps.max_batch_size + 1)
    spec = BatchEmbedSpec(texts=big, model=caps.supported_models[0])
    with pytest.raises(BadRequest):
        await a.embed_batch(spec)


async def test_embed_batch_unknown_model_raises_model_not_available():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    spec = BatchEmbedSpec(texts=["x"], model="nope-model")
    with pytest.raises(ModelNotAvailable):
        await a.embed_batch(spec)


class FallbackBatchAdapter(MockEmbeddingAdapter):
    """Force batch path to NotSupported to exercise per-item fallback."""

    async def _do_embed_batch(self, spec: BatchEmbedSpec, *, ctx: OperationContext = None) -> BatchEmbedResult:
        raise NotSupported("batch not supported")


async def test_embed_batch_partial_failure_reporting_on_fallback():
    a = FallbackBatchAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_batch_fallback", tenant="test")
    texts = ["ok", "", "also ok"]
    spec = BatchEmbedSpec(texts=texts, model=a.supported_models[0])

    res = await a.embed_batch(spec, ctx=ctx)

    # Should embed valid entries and report failures for bad ones
    assert len(res.embeddings) >= 2
    assert any(f["index"] == 1 for f in res.failed_texts)
    assert all("error" in f and "message" in f for f in res.failed_texts)

