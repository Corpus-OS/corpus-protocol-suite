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
from examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_embed_cache_respected_in_standalone_mode(adapter: BaseEmbeddingAdapter):
    """
    Calling embed() twice with the same (tenant, spec) should only trigger a single
    backend embedding call if the adapter implements caching internally.

    We instrument the adapter's internal _do_embed hook to count how many times
    the backend is actually invoked. If the adapter does not expose _do_embed
    (i.e. it does not derive from the expected BaseEmbeddingAdapter pattern),
    this test is skipped rather than failing spuriously.
    """
    # Some adapters (e.g. non-SDK implementations) may not expose _do_embed;
    # in that case we cannot introspect cache behavior and we skip this check.
    if not hasattr(adapter, "_do_embed"):
        pytest.skip("Adapter does not expose _do_embed; cannot introspect cache behavior")

    # Instrument the internal backend call to count invocations.
    embed_calls = 0
    original_do_embed = adapter._do_embed  # type: ignore[attr-defined]

    async def counting_do_embed(spec: EmbedSpec, *, ctx: OperationContext | None = None) -> EmbedResult:
        nonlocal embed_calls
        embed_calls += 1
        return await original_do_embed(spec, ctx=ctx)  # type: ignore[misc]

    # Monkey-patch the adapter's internal hook.
    adapter._do_embed = counting_do_embed  # type: ignore[assignment]

    ctx = make_ctx(OperationContext, request_id="t_cache", tenant="t")
    spec = EmbedSpec(text="cache me", model=adapter.supported_models[0], normalize=False)

    r1 = await adapter.embed(spec, ctx=ctx)
    r2 = await adapter.embed(spec, ctx=ctx)

    # At least one backend call must have happened.
    assert embed_calls >= 1
    # If cache works, the second logical embed call should *not* call _do_embed again.
    assert embed_calls == 1
    assert isinstance(r1, EmbedResult)
    assert isinstance(r2, EmbedResult)
    assert isinstance(r1.embedding, EmbeddingVector)
    assert isinstance(r2.embedding, EmbeddingVector)
    assert r1.embedding.vector == r2.embedding.vector


async def test_embed_cache_respects_tenant_isolation(adapter: BaseEmbeddingAdapter):
    """
    Cache keys MUST be tenant-aware to avoid cross-tenant leakage.
    Same spec under different tenants should not hit the same cache entry.
    That means we expect two backend calls when only the tenant changes.
    """
    if not hasattr(adapter, "_do_embed"):
        pytest.skip("Adapter does not expose _do_embed; cannot introspect cache behavior")

    embed_calls = 0
    original_do_embed = adapter._do_embed  # type: ignore[attr-defined]

    async def counting_do_embed(spec: EmbedSpec, *, ctx: OperationContext | None = None) -> EmbedResult:
        nonlocal embed_calls
        embed_calls += 1
        return await original_do_embed(spec, ctx=ctx)  # type: ignore[misc]

    adapter._do_embed = counting_do_embed  # type: ignore[assignment]

    spec = EmbedSpec(text="cache-me", model=adapter.supported_models[0], normalize=False)

    ctx1 = make_ctx(OperationContext, request_id="t_cache_t1", tenant="tenant-1")
    ctx2 = make_ctx(OperationContext, request_id="t_cache_t2", tenant="tenant-2")

    await adapter.embed(spec, ctx=ctx1)
    await adapter.embed(spec, ctx=ctx2)

    # Two distinct tenants ⇒ two backend calls (no cross-tenant cache sharing).
    assert embed_calls == 2


async def test_embed_batch_fallback_uses_per_item_and_reports_failures(adapter: BaseEmbeddingAdapter):
    """
    If some items in a batch cannot be embedded, the adapter MUST surface partial
    failures with per-item indexing rather than failing the entire batch.

    This test does not care *how* the adapter implements this (native batch,
    internal fallback to per-item, etc.); it only enforces the observable contract:

      • Successful items appear in embeddings with correct ordering.
      • Failed items are reported in failed_texts with a correct 'index' field.
    """
    capabilities: EmbeddingCapabilities = adapter.capabilities  # type: ignore[assignment]
    # If the adapter explicitly declares that batch embedding is unsupported,
    # calling embed_batch SHOULD still behave according to the partial-failure
    # contract via whatever fallback mechanism it uses internally.
    # We do not skip based on capabilities; instead we require the call to work.

    ctx = make_ctx(OperationContext, request_id="t_fallback", tenant="t")
    texts = ["ok", "", "ok2"]
    spec = BatchEmbedSpec(texts=texts, model=adapter.supported_models[0])

    res = await adapter.embed_batch(spec, ctx=ctx)

    assert isinstance(res, BatchEmbedResult)
    # We expect at least two successful embeddings (indices 0 and 2).
    assert len(res.embeddings) >= 2

    # And we expect at least one recorded failure for the middle element (index 1).
    assert any(f["index"] == 1 for f in res.failed_texts)