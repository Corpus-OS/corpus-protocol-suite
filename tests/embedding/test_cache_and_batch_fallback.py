# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Cache behavior & batch fallback semantics.

Spec refs:
  • §10.3 embed(), embed_batch()
  • §12.5 Partial Success & Caching (informative but enforced here)
"""

import pytest

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
from corpus_sdk.examples.embedding.mock_embedding_adapter import MockEmbeddingAdapter
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


class CountingAdapter(MockEmbeddingAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_calls = 0

    async def _do_embed(self, spec: EmbedSpec, *, ctx: OperationContext = None) -> EmbedResult:
        self.embed_calls += 1
        return await super()._do_embed(spec, ctx=ctx)


async def test_embed_cache_respected_in_standalone_mode():
    a = CountingAdapter(mode="standalone", failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_cache", tenant="t")
    spec = EmbedSpec(text="cache me", model=a.supported_models[0], normalize=False)

    r1 = await a.embed(spec, ctx=ctx)
    r2 = await a.embed(spec, ctx=ctx)

    assert a.embed_calls >= 1
    # If cache works, second call should not increment embed_calls
    assert a.embed_calls == 1
    assert r1.embedding.vector == r2.embedding.vector


async def test_embed_cache_respects_tenant_isolation():
    """
    Cache keys MUST be tenant-aware to avoid cross-tenant leakage.
    Same spec under different tenants should not hit the same cache entry.
    """
    a = CountingAdapter(mode="standalone", failure_rate=0.0)
    spec = EmbedSpec(text="cache-me", model=a.supported_models[0], normalize=False)

    ctx1 = make_ctx(OperationContext, request_id="t_cache_t1", tenant="tenant-1")
    ctx2 = make_ctx(OperationContext, request_id="t_cache_t2", tenant="tenant-2")

    await a.embed(spec, ctx=ctx1)
    await a.embed(spec, ctx=ctx2)

    # Two distinct tenants ⇒ two backend calls (no cross-tenant cache sharing)
    assert a.embed_calls == 2


class FallbackAdapter(MockEmbeddingAdapter):
    async def _do_embed_batch(self, spec: BatchEmbedSpec, *, ctx: OperationContext = None) -> BatchEmbedResult:
        raise NotSupported("batch not supported")


async def test_embed_batch_fallback_uses_per_item_and_reports_failures():
    a = FallbackAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_fallback", tenant="t")
    texts = ["ok", "", "ok2"]
    spec = BatchEmbedSpec(texts=texts, model=a.supported_models[0])

    res = await a.embed_batch(spec, ctx=ctx)
    assert isinstance(res, BatchEmbedResult)
    assert len(res.embeddings) >= 2
    assert any(f["index"] == 1 for f in res.failed_texts)
