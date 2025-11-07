# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Single embed behavior.

Spec refs:
  • §10.3 embed() — basic contract
  • §12 Error Handling — BadRequest, ModelNotAvailable, TextTooLong
"""

import pytest

from corpus_sdk.embedding.embedding_base import (
    EmbedSpec,
    OperationContext,
    BadRequest,
    ModelNotAvailable,
    TextTooLong,
)
from corpus_sdk.examples.embedding.mock_embedding_adapter import MockEmbeddingAdapter
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_embed_returns_embed_result_and_vector():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_embed_ok", tenant="test")
    spec = EmbedSpec(text="hello", model=a.supported_models[0])
    res = await a.embed(spec, ctx=ctx)

    assert res.embedding.vector
    assert res.embedding.dimensions == len(res.embedding.vector)
    assert res.embedding.text == "hello"
    assert res.model == spec.model


async def test_embed_requires_non_empty_text():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    spec = EmbedSpec(text="", model=a.supported_models[0])
    with pytest.raises(BadRequest):
        await a.embed(spec)


async def test_embed_requires_non_empty_model():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    spec = EmbedSpec(text="x", model="")
    with pytest.raises(BadRequest):
        await a.embed(spec)


async def test_embed_unknown_model_raises_model_not_available():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    spec = EmbedSpec(text="hello", model="nope-123")
    with pytest.raises(ModelNotAvailable):
        await a.embed(spec)


async def test_embed_truncates_when_allowed():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    caps = await a.capabilities()
    long_text = "x" * (caps.max_text_length + 10)
    spec = EmbedSpec(text=long_text, model=caps.supported_models[0], truncate=True)
    res = await a.embed(spec)
    assert len(res.embedding.text) <= caps.max_text_length
    assert res.truncated is True


async def test_embed_rejects_when_truncate_false_and_too_long():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    caps = await a.capabilities()
    long_text = "x" * (caps.max_text_length + 10)
    spec = EmbedSpec(text=long_text, model=caps.supported_models[0], truncate=False)
    with pytest.raises(TextTooLong):
        await a.embed(spec)


async def test_embed_normalize_flag_produces_unit_vector():
    a = MockEmbeddingAdapter(failure_rate=0.0, normalizes_at_source=False)
    spec = EmbedSpec(text="normalize me", model=a.supported_models[0], normalize=True)
    res = await a.embed(spec)
    norm = sum(v * v for v in res.embedding.vector) ** 0.5
    assert 0.99 <= norm <= 1.01

