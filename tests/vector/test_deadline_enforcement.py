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
from corpus_sdk.vector.vector_base import (
    OperationContext,
    QuerySpec,
    DeadlineExceeded,
    Vector,
    VectorID,
    UpsertSpec,
)
from examples.common.ctx import make_ctx, remaining_budget_ms, clear_time_cache

pytestmark = pytest.mark.asyncio


async def test_deadline_deadline_budget_nonnegative():
    """Verify remaining budget helper never returns negative values."""
    clear_time_cache()
    now_ms = int(time.time() * 1000)
    ctx = make_ctx(OperationContext, request_id="v_deadline_budget", tenant="test", deadline_ms=now_ms + 50)
    remaining = remaining_budget_ms(ctx)
    assert remaining is None or remaining >= 0


async def test_deadline_deadline_exceeded_on_expired_budget(adapter):
    """Verify DeadlineExceeded is raised when budget is expired."""
    clear_time_cache()
    expired_ms = int(time.time() * 1000) - 1  # 1ms in the past
    ctx = make_ctx(OperationContext, request_id="v_deadline_expired", tenant="test", deadline_ms=expired_ms)
    
    with pytest.raises(DeadlineExceeded) as exc_info:
        await adapter.query(QuerySpec(vector=[0.1], top_k=1, namespace="default"), ctx=ctx)
    
    err = exc_info.value
    assert err.code in ("DEADLINE", "DEADLINE_EXCEEDED")


async def test_deadline_preflight_deadline_check_on_upsert(adapter):
    """Verify upsert operations respect pre-flight deadline checks."""
    clear_time_cache()
    expired_ms = int(time.time() * 1000) - 100  # 100ms in the past
    ctx = make_ctx(OperationContext, request_id="v_deadline_preflight", tenant="test", deadline_ms=expired_ms)
    
    spec = UpsertSpec(
        namespace="default",
        vectors=[Vector(id=VectorID("x"), vector=[0.1])],
    )
    
    with pytest.raises(DeadlineExceeded):
        await adapter.upsert(spec, ctx=ctx)


async def test_deadline_query_respects_deadline_mid_operation(adapter):
    """Verify queries respect deadlines even during mid-operation."""
    clear_time_cache()
    now_ms = int(time.time() * 1000)
    ctx = make_ctx(OperationContext, request_id="v_deadline_mid", tenant="test", deadline_ms=now_ms + 1)
    
    await asyncio.sleep(0.01)  # Intentionally burn budget
    
    with pytest.raises(DeadlineExceeded):
        await adapter.query(QuerySpec(vector=[0.1], top_k=1, namespace="default"), ctx=ctx)