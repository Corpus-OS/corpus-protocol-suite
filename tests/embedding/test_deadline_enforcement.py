# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Deadline semantics.

Spec refs:
  • §6.1 Context & Deadlines
  • §12.4 DeadlineExceeded mapping
"""

import time
import pytest

from corpus_sdk.embedding.embedding_base import (
    EmbedSpec,
    BatchEmbedSpec,
    OperationContext,
    DeadlineExceeded,
)
from corpus_sdk.examples.embedding.mock_embedding_adapter import MockEmbeddingAdapter
from corpus_sdk.examples.common.ctx import make_ctx, remaining_budget_ms, clear_time_cache

pytestmark = pytest.mark.asyncio


async def test_deadline_budget_nonnegative():
    clear_time_cache()
    now = int(time.time() * 1000)
    ctx = make_ctx(OperationContext, request_id="t_embed_budget", tenant="t", deadline_ms=now + 50)
    rem = remaining_budget_ms(ctx)
    assert rem is None or rem >= 0


async def test_deadline_exceeded_on_expired_budget_embed():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    clear_time_cache()
    past = int(time.time() * 1000) - 1
    ctx = make_ctx(OperationContext, request_id="t_embed_expired", tenant="t", deadline_ms=past)
    with pytest.raises(DeadlineExceeded):
        await a.embed(EmbedSpec(text="x", model=a.supported_models[0]), ctx=ctx)


async def test_deadline_exceeded_on_expired_budget_embed_batch():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    clear_time_cache()
    past = int(time.time() * 1000) - 1
    ctx = make_ctx(OperationContext, request_id="t_batch_expired", tenant="t", deadline_ms=past)
    with pytest.raises(DeadlineExceeded):
        await a.embed_batch(
            BatchEmbedSpec(texts=["x"], model=a.supported_models[0]),
            ctx=ctx,
        )

