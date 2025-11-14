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
import time
import pytest
from corpus_sdk.llm.llm_base import OperationContext, DeadlineExceeded

pytestmark = pytest.mark.asyncio

# Constants for test stability
GUARANTEED_EXPIRED_MS = 0  # Epoch 0 ensures pre-expired deadline
BUDGET_BUFFER_MS = 50      # Buffer for timing-sensitive operations


def get_remaining_budget_ms(ctx: OperationContext) -> int:
    """Calculate remaining budget in milliseconds."""
    if ctx.deadline_ms is None:
        return None
    now_ms = int(time.time() * 1000)
    remaining = ctx.deadline_ms - now_ms
    return max(0, remaining)  # Never return negative


async def test_deadline_deadline_budget_nonnegative_and_usable(adapter):
    """
    Remaining budget helper should compute correctly and never go negative.
    """
    now_ms = int(time.time() * 1000)
    ctx = OperationContext(
        request_id="t_deadline_plumbing",
        tenant="test",
        deadline_ms=now_ms + 200,  # 200ms absolute deadline
    )

    rem = get_remaining_budget_ms(ctx)
    assert rem is not None and rem >= 0, \
        "Remaining budget should be non-negative"
    assert rem <= 200, "Remaining budget should not exceed initial deadline"

    # Run a tiny operation under budget (if supported)
    caps = await adapter.capabilities()
    if caps.supports_count_tokens:
        _ = await adapter.count_tokens("tiny", ctx=ctx)

    # Budget helper should never go negative, even after operation
    rem2 = get_remaining_budget_ms(ctx)
    assert rem2 is None or rem2 >= 0, \
        "Budget should never be negative after operation"


async def test_deadline_deadline_exceeded_on_expired_budget(adapter):
    """
    SPECIFICATION.md §8.3, §12.4 — Deadline Enforcement

    Adapter MUST raise DeadlineExceeded when budget is exhausted.
    """
    # Use guaranteed expired deadline to avoid timing flakiness
    ctx = OperationContext(deadline_ms=GUARANTEED_EXPIRED_MS, tenant="test")

    with pytest.raises(DeadlineExceeded) as exc_info:
        await adapter.complete(
            messages=[{"role": "user", "content": "test"}],
            ctx=ctx,
        )

    err = exc_info.value
    # Check for either possible error code format
    assert err.code in ("DEADLINE", "DEADLINE_EXCEEDED"), \
        f"Expected code='DEADLINE' or 'DEADLINE_EXCEEDED', got '{err.code}'"
    
    # Verify error includes useful details
    assert "remaining_ms" in (err.details or {}), \
        "DeadlineExceeded should include remaining_ms in details for debugging"


async def test_deadline_deadline_exceeded_during_stream(adapter):
    """Stream should raise DeadlineExceeded if deadline expires mid-stream."""
    caps = await adapter.capabilities()
    if not caps.supports_streaming:
        pytest.skip("Adapter does not support streaming")

    # Use very short but non-zero deadline to test mid-stream expiration
    now_ms = int(time.time() * 1000)
    ctx = OperationContext(deadline_ms=now_ms + 5, tenant="test")  # 5ms deadline

    with pytest.raises(DeadlineExceeded) as exc_info:
        async for _ in adapter.stream(
            messages=[{"role": "user", "content": "long stream that exceeds deadline"}],
            model=caps.supported_models[0],
            ctx=ctx,
        ):
            await asyncio.sleep(0.01)  # Ensure we exceed the 5ms deadline

    err = exc_info.value
    assert err.code in ("DEADLINE", "DEADLINE_EXCEEDED"), \
        f"Stream should raise DEADLINE, got: {err.code}"


async def test_deadline_operations_complete_with_adequate_budget(adapter):
    """
    Operations SHOULD complete successfully when adequate budget is provided.
    """
    now_ms = int(time.time() * 1000)
    ctx = OperationContext(
        request_id="t_adequate_budget",
        tenant="test", 
        deadline_ms=now_ms + 30000,  # 30 seconds - ample time
    )

    caps = await adapter.capabilities()
    
    # Test complete operation with adequate budget
    result = await adapter.complete(
        messages=[{"role": "user", "content": "test with adequate budget"}],
        model=caps.supported_models[0],
        ctx=ctx,
    )
    
    assert result.text, "Should complete successfully with adequate budget"
    
    # Verify budget helper still works after successful operation
    remaining = get_remaining_budget_ms(ctx)
    assert remaining is None or remaining >= 0, \
        "Budget should not go negative after successful operation"


async def test_deadline_budget_calculation_accuracy(adapter):
    """
    Budget calculations should be reasonably accurate (within expected bounds).
    """
    now_ms = int(time.time() * 1000)
    initial_budget = 1000  # 1 second
    ctx = OperationContext(
        request_id="t_budget_accuracy", 
        tenant="test",
        deadline_ms=now_ms + initial_budget,
    )

    # Get initial remaining budget
    initial_remaining = get_remaining_budget_ms(ctx)
    assert initial_remaining is not None
    assert initial_remaining <= initial_budget
    
    # Perform a quick operation
    caps = await adapter.capabilities()
    if caps.supports_count_tokens:
        await adapter.count_tokens("quick operation", ctx=ctx)
    
    # Get remaining budget after operation
    final_remaining = get_remaining_budget_ms(ctx)
    assert final_remaining is not None
    assert final_remaining <= initial_remaining, "Budget should decrease after operation"
    assert final_remaining >= 0, "Budget should never be negative"