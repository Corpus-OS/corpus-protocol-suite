# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Deadline semantics.

Spec refs:
  • §6.1 Context & Deadlines
  • §12.4 DeadlineExceeded mapping

Covers:
  • Remaining budget helper never negative
  • Pre-expired deadlines fail fast without backend work
  • embed() enforces short deadlines (may raise DeadlineExceeded)
  • embed_batch() enforces short deadlines (may raise DeadlineExceeded)
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
    """Remaining budget helper never returns a negative value."""
    clear_time_cache()
    now = int(time.time() * 1000)
    ctx = make_ctx(
        OperationContext,
        request_id="t_embed_budget",
        tenant="t",
        deadline_ms=now + 50,
    )
    rem = remaining_budget_ms(ctx)
    assert rem is None or rem >= 0


async def test_preexpired_deadline_fails_fast_embed():
    """Pre-expired deadline MUST raise DeadlineExceeded before backend work."""
    a = MockEmbeddingAdapter(failure_rate=0.0, mode="standalone")
    clear_time_cache()
    past = int(time.time() * 1000) - 1
    ctx = make_ctx(
        OperationContext,
        request_id="t_embed_preexpired",
        tenant="t",
        deadline_ms=past,
    )
    with pytest.raises(DeadlineExceeded):
        await a.embed(
            EmbedSpec(text="x", model=a.supported_models[0]),
            ctx=ctx,
        )


async def test_embed_respects_deadline():
    """Short deadline on embed() SHOULD be honored and may trigger DeadlineExceeded."""
    a = MockEmbeddingAdapter(failure_rate=0.0, mode="standalone")
    clear_time_cache()
    now = int(time.time() * 1000)
    # Extremely short budget to force/encourage timeout behavior
    ctx = make_ctx(
        OperationContext,
        request_id="t_embed_deadline_short",
        tenant="t",
        deadline_ms=now + 1,
    )
    with pytest.raises(DeadlineExceeded):
        await a.embed(
            EmbedSpec(text="x", model=a.supported_models[0]),
            ctx=ctx,
        )


async def test_embed_batch_respects_deadline():
    """Short deadline on embed_batch() SHOULD be honored and may trigger DeadlineExceeded."""
    a = MockEmbeddingAdapter(failure_rate=0.0, mode="standalone")
    clear_time_cache()
    now = int(time.time() * 1000)
    ctx = make_ctx(
        OperationContext,
        request_id="t_embed_batch_deadline_short",
        tenant="t",
        deadline_ms=now + 1,
    )
    with pytest.raises(DeadlineExceeded):
        await a.embed_batch(
            BatchEmbedSpec(texts=["x", "y"], model=a.supported_models[0]),
            ctx=ctx,
        )
