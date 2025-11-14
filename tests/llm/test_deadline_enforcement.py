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
from corpus_sdk.llm.llm_base import OperationContext, DeadlineExceeded
from examples.common.ctx import make_ctx, remaining_budget_ms

pytestmark = pytest.mark.asyncio


async def test_deadline_deadline_budget_nonnegative_and_usable(adapter):
    """
    Remaining budget helper should compute correctly and never go negative.
    """
    ctx = make_ctx(
        OperationContext,
        request_id="t_deadline_plumbing",
        tenant="test",
        timeout_ms=200,  # relative deadline
    )

    rem = remaining_budget_ms(ctx)
    assert rem is not None and rem >= 0, \
        "Remaining budget should be non-negative"

    # Run a tiny operation under budget (if supported)
    caps = await adapter.capabilities()
    if caps.supports_count_tokens:
        _ = await adapter.count_tokens("tiny", ctx=ctx)

    # Budget helper should never go negative
    rem2 = remaining_budget_ms(ctx)
    assert rem2 is None or rem2 >= 0, \
        "Budget should never be negative after operation"


async def test_deadline_deadline_exceeded_on_expired_budget(adapter):
    """
    SPECIFICATION.md §8.3, §12.4 — Deadline Enforcement

    Adapter MUST raise DeadlineExceeded when budget is exhausted.
    """
    ctx = make_ctx(OperationContext, timeout_ms=1)

    await asyncio.sleep(0.005)  # Burn the budget

    with pytest.raises(DeadlineExceeded) as exc_info:
        await adapter.complete(
            messages=[{"role": "user", "content": "test"}],
            ctx=ctx,
        )

    err = exc_info.value
    # Check for either possible error code format
    assert err.code in ("DEADLINE", "DEADLINE_EXCEEDED"), \
        f"Expected code='DEADLINE' or 'DEADLINE_EXCEEDED', got '{err.code}'"
    assert "remaining_ms" in (err.details or {}), \
        "DeadlineExceeded should include remaining_ms in details"


async def test_deadline_deadline_exceeded_during_stream(adapter):
    """Stream should raise DeadlineExceeded if deadline expires mid-stream."""
    caps = await adapter.capabilities()
    if not caps.supports_streaming:
        pytest.skip("Adapter does not support streaming")

    ctx = make_ctx(OperationContext, timeout_ms=10)  # Very short deadline

    with pytest.raises(DeadlineExceeded):
        async for _ in adapter.stream(
            messages=[{"role": "user", "content": "long stream"}],
            model=caps.supported_models[0],
            ctx=ctx,
        ):
            await asyncio.sleep(0.005)  # Simulate slow processing