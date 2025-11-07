# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Error taxonomy & retry hints.

Spec refs:
  • §12.1, §12.4 Error Handling
"""

import pytest

from corpus_sdk.embedding.embedding_base import (
    EmbedSpec,
    BatchEmbedSpec,
    OperationContext,
    ResourceExhausted,
    Unavailable,
    BadRequest,
    ModelNotAvailable,
    EmbeddingAdapterError,
)
from corpus_sdk.examples.embedding.mock_embedding_adapter import MockEmbeddingAdapter
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_bad_request_on_invalid_inputs():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    with pytest.raises(BadRequest):
        await a.embed(EmbedSpec(text="", model=a.supported_models[0]))


async def test_model_not_available_for_unsupported_model():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    with pytest.raises(ModelNotAvailable):
        await a.embed(EmbedSpec(text="x", model="nope"))


class FailingEmbeddingAdapter(MockEmbeddingAdapter):
    async def _do_embed(self, spec: EmbedSpec, *, ctx: OperationContext = None):
        raise ResourceExhausted("rate limited", retry_after_ms=500, resource_scope="rate_limit")

    async def _do_embed_batch(self, spec: BatchEmbedSpec, *, ctx: OperationContext = None):
        raise Unavailable("backend down", retry_after_ms=1000)


async def test_retryable_errors_with_hints_single():
    a = FailingEmbeddingAdapter(failure_rate=0.0)
    spec = EmbedSpec(text="x", model=a.supported_models[0])
    with pytest.raises(ResourceExhausted) as ei:
        await a.embed(spec)
    err = ei.value
    assert err.retry_after_ms == 500
    assert err.resource_scope == "rate_limit"


async def test_retryable_errors_with_hints_batch():
    a = FailingEmbeddingAdapter(failure_rate=0.0)
    spec = BatchEmbedSpec(texts=["x"], model=a.supported_models[0])
    with pytest.raises(Unavailable) as ei:
        await a.embed_batch(spec)
    err = ei.value
    assert err.retry_after_ms == 1000

