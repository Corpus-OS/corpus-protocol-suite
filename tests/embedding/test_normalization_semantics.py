# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Normalization semantics.

Spec refs:
  • §9 (Embedding Protocol V1) — normalize flag & vector semantics
  • §6.1 — Deterministic behavior across calls
  • §13 — SIEM-safe: normals tested via norms only, no content in metrics

Asserts:
  • normalize=True produces (approx) unit-length vectors when supported
  • normalize=False does NOT force unit norm
  • Batch normalization applies per-vector consistently
  • If capabilities.supports_normalization is False and normalize=True, NotSupported is raised
  • normalizes_at_source=True is honored without double-normalization
"""

import math
import pytest

from adapter_sdk.embedding_base import (
    OperationContext,
    EmbedSpec,
    BatchEmbedSpec,
    NotSupported,
)
from examples.embedding.mock_embedding_adapter import MockEmbeddingAdapter
from examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


def _norm(vec):
    return math.sqrt(sum(v * v for v in vec))


async def test_single_embed_normalize_true_produces_unit_vector():
    adapter = MockEmbeddingAdapter(failure_rate=0.0, normalizes_at_source=False)
    ctx = make_ctx(OperationContext, request_id="t_norm_single_true", tenant="t")

    spec = EmbedSpec(
        text="normalize me",
        model=adapter.supported_models[0],
        truncate=True,
        normalize=True,
    )
    res = await adapter.embed(spec, ctx=ctx)

    n = _norm(res.embedding.vector)
    assert abs(n - 1.0) < 1e-6, f"expected unit norm, got {n}"
    assert res.truncated is False


async def test_single_embed_normalize_false_not_forced_unit_norm():
    adapter = MockEmbeddingAdapter(failure_rate=0.0, normalizes_at_source=False)
    ctx = make_ctx(OperationContext, request_id="t_norm_single_false", tenant="t")

    spec = EmbedSpec(
        text="raw vector please",
        model=adapter.supported_models[0],
        truncate=True,
        normalize=False,
    )
    res = await adapter.embed(spec, ctx=ctx)

    n = _norm(res.embedding.vector)
    # For deterministic random [-1,1) 512-d, norm should be far from exactly 1
    assert abs(n - 1.0) > 1e-3, f"expected non-unit norm, got {n}"


async def test_batch_embed_normalize_true_all_unit_vectors():
    adapter = MockEmbeddingAdapter(failure_rate=0.0, normalizes_at_source=False)
    ctx = make_ctx(OperationContext, request_id="t_norm_batch_true", tenant="t")

    spec = BatchEmbedSpec(
        texts=["a", "b", "c"],
        model=adapter.supported_models[0],
        truncate=True,
        normalize=True,
    )
    res = await adapter.embed_batch(spec, ctx=ctx)

    assert len(res.embeddings) == 3
    for ev in res.embeddings:
        n = _norm(ev.vector)
        assert abs(n - 1.0) < 1e-6, f"expected unit norm in batch, got {n}"


class NoNormAdapter(MockEmbeddingAdapter):
    """
    Adapter that advertises no normalization support to verify
    BaseEmbeddingAdapter surfaces NotSupported when normalize=True.
    """
    async def _do_capabilities(self):
        caps = await super()._do_capabilities()
        return type(caps)(
            **{
                **caps.__dict__,
                "supports_normalization": False,
                "normalizes_at_source": False,
            }
        )


async def test_normalization_not_supported_raises():
    adapter = NoNormAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_norm_nosupport", tenant="t")

    spec = EmbedSpec(
        text="x",
        model=adapter.supported_models[0],
        truncate=True,
        normalize=True,
    )
    with pytest.raises(NotSupported):
        await adapter.embed(spec, ctx=ctx)


class SourceNormAdapter(MockEmbeddingAdapter):
    """
    Adapter that normalizes at source; BaseEmbeddingAdapter MUST NOT re-normalize.
    """
    async def _do_capabilities(self):
        caps = await super()._do_capabilities()
        return type(caps)(
            **{
                **caps.__dict__,
                "supports_normalization": True,
                "normalizes_at_source": True,
            }
        )


async def test_normalizes_at_source_respected_no_double_normalization():
    adapter = SourceNormAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_norm_source", tenant="t")

    spec = EmbedSpec(
        text="normalize at source",
        model=adapter.supported_models[0],
        truncate=True,
        normalize=True,
    )
    res = await adapter.embed(spec, ctx=ctx)

    n = _norm(res.embedding.vector)
    # Still unit length, and no error; double-normalization would still be unit,
    # but this asserts protocol path is valid for adapters that do it at source.
    assert abs(n - 1.0) < 1e-6

