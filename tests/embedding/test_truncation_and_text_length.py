# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Truncation & max text length.

Spec refs:
  • Embedding Protocol V1 — max_text_length, truncate flag
  • §6.1 — Deterministic truncation behavior
  • §9.5 / Error taxonomy — TEXT_TOO_LONG via TextTooLong

Asserts:
  • Inputs longer than max_text_length are truncated when truncate=True
  • Same condition raises TextTooLong when truncate=False
  • Batch embedding truncates each item consistently
  • Batch with truncate=False and oversize text fails with TextTooLong
  • Short texts pass through unchanged
"""

import pytest

from adapter_sdk.embedding_base import (
    OperationContext,
    EmbedSpec,
    BatchEmbedSpec,
    TextTooLong,
)
from examples.embedding.mock_embedding_adapter import MockEmbeddingAdapter
from examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_embed_truncates_when_allowed_and_sets_flag():
    adapter = MockEmbeddingAdapter(failure_rate=0.0)
    caps = await adapter.capabilities()
    max_len = caps.max_text_length or 1024

    long_text = "x" * (max_len + 10)
    ctx = make_ctx(OperationContext, request_id="t_trunc_single_ok", tenant="t")

    spec = EmbedSpec(
        text=long_text,
        model=adapter.supported_models[0],
        truncate=True,
        normalize=False,
    )
    res = await adapter.embed(spec, ctx=ctx)

    assert len(res.embedding.text) == max_len
    assert res.truncated is True


async def test_embed_raises_when_truncation_disallowed():
    adapter = MockEmbeddingAdapter(failure_rate=0.0)
    caps = await adapter.capabilities()
    max_len = caps.max_text_length or 1024

    long_text = "x" * (max_len + 1)
    ctx = make_ctx(OperationContext, request_id="t_trunc_single_err", tenant="t")

    spec = EmbedSpec(
        text=long_text,
        model=adapter.supported_models[0],
        truncate=False,
        normalize=False,
    )

    with pytest.raises(TextTooLong):
        await adapter.embed(spec, ctx=ctx)


async def test_batch_truncates_all_when_allowed():
    adapter = MockEmbeddingAdapter(failure_rate=0.0)
    caps = await adapter.capabilities()
    max_len = caps.max_text_length or 1024

    long1 = "a" * (max_len + 5)
    long2 = "b" * (max_len + 50)

    ctx = make_ctx(OperationContext, request_id="t_trunc_batch_ok", tenant="t")
    spec = BatchEmbedSpec(
        texts=[long1, long2],
        model=adapter.supported_models[0],
        truncate=True,
        normalize=False,
    )
    res = await adapter.embed_batch(spec, ctx=ctx)

    assert len(res.embeddings) == 2
    assert all(len(ev.text) == max_len for ev in res.embeddings)


async def test_batch_oversize_without_truncation_raises():
    adapter = MockEmbeddingAdapter(failure_rate=0.0)
    caps = await adapter.capabilities()
    max_len = caps.max_text_length or 1024

    long1 = "a" * (max_len + 1)
    ctx = make_ctx(OperationContext, request_id="t_trunc_batch_err", tenant="t")

    spec = BatchEmbedSpec(
        texts=[long1],
        model=adapter.supported_models[0],
        truncate=False,
        normalize=False,
    )

    with pytest.raises(TextTooLong):
        await adapter.embed_batch(spec, ctx=ctx)


async def test_short_texts_unchanged():
    adapter = MockEmbeddingAdapter(failure_rate=0.0)
    caps = await adapter.capabilities()
    max_len = caps.max_text_length or 1024

    text = "short text within limit"
    assert len(text) < max_len

    ctx = make_ctx(OperationContext, request_id="t_trunc_short", tenant="t")

    spec = EmbedSpec(
        text=text,
        model=adapter.supported_models[0],
        truncate=True,
        normalize=False,
    )
    res = await adapter.embed(spec, ctx=ctx)

    assert res.embedding.text == text
    assert res.truncated is False

