# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — count_tokens behavior.

Spec refs:
  • §10.3 count_tokens()
  • §12 Error Handling — NotSupported, ModelNotAvailable
"""

import pytest

from corpus_sdk.embedding.embedding_base import (
    OperationContext,
    NotSupported,
    ModelNotAvailable,
)
from corpus_sdk.examples.embedding.mock_embedding_adapter import MockEmbeddingAdapter
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_count_tokens_returns_non_negative_int():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_tokens_basic", tenant="t")
    n = await a.count_tokens("hello world", a.supported_models[0], ctx=ctx)
    assert isinstance(n, int)
    assert n >= 0


async def test_count_tokens_monotonic_with_respect_to_length():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    m = a.supported_models[0]
    n1 = await a.count_tokens("hi", m)
    n2 = await a.count_tokens("hi there now", m)
    assert n2 >= n1


async def test_count_tokens_empty_string_zero_or_minimal():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    m = a.supported_models[0]
    n = await a.count_tokens("", m)
    assert n >= 0


async def test_count_tokens_unicode_safe():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    m = a.supported_models[0]
    n = await a.count_tokens("こんにちは 👋🌍", m)
    assert n >= 0


async def test_count_tokens_unknown_model_raises_model_not_available():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    with pytest.raises(ModelNotAvailable):
        await a.count_tokens("x", "does-not-exist")


class NoCountAdapter(MockEmbeddingAdapter):
    async def _do_capabilities(self):
        caps = await super()._do_capabilities()
        return EmbeddingCapabilities(
            **{
                **caps.__dict__,
                "supports_token_counting": False,
            }
        )


async def test_count_tokens_not_supported_raises_not_supported():
    a = NoCountAdapter(failure_rate=0.0)
    with pytest.raises(NotSupported):
        await a.count_tokens("x", a.supported_models[0])

