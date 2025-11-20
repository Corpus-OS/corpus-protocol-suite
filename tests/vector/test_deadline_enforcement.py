# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Deadline semantics.

Spec refs:
  • SPECIFICATION.md §6.1 (Operation Context)
  • SPECIFICATION.md §12.1, §12.4 (DeadlineExceeded)
"""

import time
import asyncio
from typing import Optional
import pytest
from corpus_sdk.vector.vector_base import (
    OperationContext,
    QuerySpec,
    DeadlineExceeded,
    Vector,
    VectorID,
    UpsertSpec,
)

pytestmark = pytest.mark.asyncio


def get_remaining_budget_ms(ctx: OperationContext) -> Optional[int]:
    """Calculate remaining budget in milliseconds."""
    if ctx.deadline_ms is None:
        return None
    now_ms = int(time.time() * 1000)
    remaining = ctx.deadline_ms - now_ms
    return max(0, remaining)


async def test_deadline_deadline_budget_nonnegative():
    """Verify remaining budget helper never returns negative values."""
    now_ms = int(time.time() * 1000)
    
    # Test with future deadline
    ctx = OperationContext(request_id="v_deadline_budget", tenant="test", deadline_ms=now_ms + 50)
    remaining = get_remaining_budget_ms(ctx)
    assert remaining is not None
    assert remaining >= 0
    assert remaining <= 50  # Should be roughly 50ms (allowing for minor time passage)

    # Test with past deadline
    past_ctx = OperationContext(request_id="v_deadline_past", tenant="test", deadline_ms=now_ms - 100)
    past_remaining = get_remaining_budget_ms(past_ctx)
    assert past_remaining == 0  # Should clamp to 0, not negative

    # Test with no deadline
    no_deadline_ctx = OperationContext(request_id="v_deadline_none", tenant="test")
    no_deadline_remaining = get_remaining_budget_ms(no_deadline_ctx)
    assert no_deadline_remaining is None


async def test_deadline_deadline_exceeded_on_expired_budget(adapter):
    """Verify DeadlineExceeded is raised when budget is expired."""
    expired_ms = int(time.time() * 1000) - 1  # 1ms in the past
    ctx = OperationContext(request_id="v_deadline_expired", tenant="test", deadline_ms=expired_ms)
    
    with pytest.raises(DeadlineExceeded) as exc_info:
        await adapter.query(QuerySpec(vector=[0.1], top_k=1, namespace="default"), ctx=ctx)
    
    err = exc_info.value
    assert err.code in ("DEADLINE", "DEADLINE_EXCEEDED")


async def test_deadline_preflight_deadline_check_on_upsert(adapter):
    """Verify upsert operations respect pre-flight deadline checks."""
    expired_ms = int(time.time() * 1000) - 100  # 100ms in the past
    ctx = OperationContext(request_id="v_deadline_preflight", tenant="test", deadline_ms=expired_ms)
    
    spec = UpsertSpec(
        namespace="default",
        vectors=[Vector(id=VectorID("x"), vector=[0.1])],
    )
    
    with pytest.raises(DeadlineExceeded):
        await adapter.upsert(spec, ctx=ctx)


async def test_deadline_query_respects_deadline_mid_operation(adapter):
    """Verify queries respect deadlines even during mid-operation."""
    now_ms = int(time.time() * 1000)
    ctx = OperationContext(request_id="v_deadline_mid", tenant="test", deadline_ms=now_ms + 1)
    
    await asyncio.sleep(0.01)  # Intentionally burn budget
    
    with pytest.raises(DeadlineExceeded):
        await adapter.query(QuerySpec(vector=[0.1], top_k=1, namespace="default"), ctx=ctx)