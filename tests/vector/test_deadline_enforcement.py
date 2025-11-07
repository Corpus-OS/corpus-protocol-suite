# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Deadline semantics.

Spec refs:
  • SPECIFICATION.md §6.1 (Operation Context)
  • SPECIFICATION.md §12.1, §12.4 (DeadlineExceeded)
"""

import time
import asyncio
import pytest

from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter
from adapter_sdk.vector_base import (
    OperationContext,
    QuerySpec,
    DeadlineExceeded,
)
from corpus_sdk.examples.common.ctx import make_ctx, remaining_budget_ms, clear_time_cache

pytestmark = pytest.mark.asyncio


async def test_deadline_budget_nonnegative():
    clear_time_cache()
    now = int(time.time() * 1000)
    ctx = make_ctx(OperationContext, request_id="v_deadline_budget", tenant="t", deadline_ms=now + 50)
    rem = remaining_budget_ms(ctx)
    assert rem is None or rem >= 0


async def test_deadline_exceeded_on_expired_budget():
    a = MockVectorAdapter()
    clear_time_cache()
    ctx = make_ctx(OperationContext, request_id="v_deadline_expired", tenant="t",
                   deadline_ms=int(time.time() * 1000) - 1)
    with pytest.raises(DeadlineExceeded):
        await a.query(QuerySpec(vector=[0.1], top_k=1, namespace="default"), ctx=ctx)


async def test_preflight_deadline_check_on_upsert():
    from adapter_sdk.vector_base import Vector, VectorID, UpsertSpec
    a = MockVectorAdapter()
    clear_time_cache()
    ctx = make_ctx(OperationContext, request_id="v_deadline_preflight", tenant="t",
                   deadline_ms=int(time.time() * 1000) - 100)
    spec = UpsertSpec(
        namespace="default",
        vectors=[Vector(id=VectorID("x"), vector=[0.1])],
    )
    with pytest.raises(DeadlineExceeded):
        await a.upsert(spec, ctx=ctx)


async def test_query_respects_deadline_mid_operation():
    a = MockVectorAdapter()
    clear_time_cache()
    now = int(time.time() * 1000)
    ctx = make_ctx(OperationContext, request_id="v_deadline_mid", tenant="t", deadline_ms=now + 1)
    await asyncio.sleep(0.01)  # burn budget
    with pytest.raises(DeadlineExceeded):
        await a.query(QuerySpec(vector=[0.1], top_k=1, namespace="default"), ctx=ctx)
