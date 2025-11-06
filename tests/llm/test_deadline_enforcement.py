# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Deadline enforcement plumbing.
Asserts:
  • Remaining budget is computed correctly
  • Calls under short budgets still complete
  • Budget never goes negative in helper
  • DeadlineExceeded raised when budget exhausted (§8.3, §12.4)
"""
import asyncio
import pytest
from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.llm.llm_base import OperationContext, DeadlineExceeded
from corpus_sdk.examples.common.ctx import make_ctx, remaining_budget_ms

pytestmark = pytest.mark.asyncio


async def test_deadline_budget_nonnegative_and_usable():
    """
    Remaining budget helper should compute correctly and never go negative.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(
        OperationContext,
        request_id="t_deadline_plumbing",
        tenant="test",
        timeout_ms=200,  # relative deadline
    )
    
    rem = remaining_budget_ms(ctx)
    assert rem is not None and rem >= 0, \
        "Remaining budget should be non-negative"
    
    # Run a tiny operation under budget
    _ = await adapter.count_tokens("tiny", ctx=ctx)
    
    # Budget helper should never go negative
    rem2 = remaining_budget_ms(ctx)
    assert rem2 is None or rem2 >= 0, \
        "Budget should never be negative after operation"


async def test_deadline_exceeded_on_expired_budget():
    """
    SPECIFICATION.md §8.3, §12.4 — Deadline Enforcement
    
    Adapter MUST raise DeadlineExceeded when budget is exhausted.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, timeout_ms=1)  # 1ms budget
    
    await asyncio.sleep(0.005)  # Burn the budget
    
    with pytest.raises(DeadlineExceeded) as exc_info:
        await adapter.complete(
            messages=[{"role": "user", "content": "test"}],
            ctx=ctx
        )
    
    err = exc_info.value
    assert err.code == "DEADLINE", \
        f"Expected code='DEADLINE', got '{err.code}'"
    assert "remaining_ms" in err.details, \
        "DeadlineExceeded should include remaining_ms in details"


async def test_deadline_exceeded_during_stream():
    """Stream should raise DeadlineExceeded if deadline expires mid-stream."""
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, timeout_ms=10)  # Very short deadline
    
    chunks = []
    with pytest.raises(DeadlineExceeded):
        async for chunk in adapter.stream(
            messages=[{"role": "user", "content": "long stream"}],
            ctx=ctx
        ):
            chunks.append(chunk)
            await asyncio.sleep(0.005)  # Simulate slow processing
