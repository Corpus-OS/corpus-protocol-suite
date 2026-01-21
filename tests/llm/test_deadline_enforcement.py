# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Deadline enforcement plumbing.
Asserts:
  • Remaining budget is computed correctly
  • Calls under short budgets still complete (when not expired)
  • Budget never goes negative in helper
  • DeadlineExceeded raised when budget exhausted (§8.3, §12.4)
"""
import time
import pytest
from typing import Optional
from corpus_sdk.llm.llm_base import OperationContext, DeadlineExceeded, NotSupported

pytestmark = pytest.mark.asyncio

# Constants for test stability
GUARANTEED_EXPIRED_MS = 0  # Epoch 0 ensures pre-expired deadline


def get_remaining_budget_ms(ctx: OperationContext) -> Optional[int]:
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
    assert rem is not None and rem >= 0, "Remaining budget should be non-negative"
    assert rem <= 200, "Remaining budget should not exceed initial deadline"

    caps = await adapter.capabilities()
    if caps.supports_count_tokens:
        _ = await adapter.count_tokens("tiny", ctx=ctx)

    rem2 = get_remaining_budget_ms(ctx)
    assert rem2 is None or rem2 >= 0, "Budget should never be negative after operation"


async def test_deadline_deadline_exceeded_on_expired_budget(adapter):
    """
    SPECIFICATION.md §8.3, §12.4 — Deadline Enforcement

    Capability↔behavior alignment:
      - If supports_deadline is True: expired deadline MUST raise DeadlineExceeded.
      - If supports_deadline is False: expired deadline MUST NOT force DeadlineExceeded.
    """
    caps = await adapter.capabilities()
    ctx = OperationContext(deadline_ms=GUARANTEED_EXPIRED_MS, tenant="test")

    if caps.supports_deadline:
        with pytest.raises(DeadlineExceeded) as exc_info:
            await adapter.complete(
                messages=[{"role": "user", "content": "test"}],
                ctx=ctx,
            )
        err = exc_info.value
        assert err.code == "DEADLINE_EXCEEDED", f"Expected code='DEADLINE_EXCEEDED', got '{err.code}'"
        assert "remaining_ms" in (err.details or {}), "DeadlineExceeded should include remaining_ms in details"
    else:
        # If deadline support is not advertised, expired deadline should not be enforced by the adapter.
        res = await adapter.complete(
            messages=[{"role": "user", "content": "test"}],
            ctx=ctx,
        )
        assert hasattr(res, "finish_reason")


async def test_deadline_deadline_exceeded_during_stream(adapter):
    """
    Stream should raise DeadlineExceeded if deadline is already expired and supports_deadline=True.

    Capability↔behavior alignment:
      - If supports_streaming is False: stream() MUST raise NotSupported.
      - If supports_streaming is True and supports_deadline is True: expired deadline MUST raise DeadlineExceeded.
      - If supports_streaming is True and supports_deadline is False: expired deadline MUST NOT force DeadlineExceeded.
    """
    caps = await adapter.capabilities()
    ctx = OperationContext(deadline_ms=GUARANTEED_EXPIRED_MS, tenant="test")

    if not caps.supports_streaming:
        with pytest.raises(NotSupported):
            agen = adapter.stream(
                messages=[{"role": "user", "content": "test stream"}],
                model=caps.supported_models[0],
                ctx=ctx,
            )
            async for _ in agen:
                pass
        return

    if caps.supports_deadline:
        with pytest.raises(DeadlineExceeded):
            agen = adapter.stream(
                messages=[{"role": "user", "content": "test stream"}],
                model=caps.supported_models[0],
                ctx=ctx,
            )
            async for _ in agen:
                pass
    else:
        # Deadline not supported: should be able to consume at least one chunk (or terminate cleanly).
        got_any = False
        agen = adapter.stream(
            messages=[{"role": "user", "content": "test stream"}],
            model=caps.supported_models[0],
            ctx=ctx,
        )
        async for _ in agen:
            got_any = True
            break
        assert got_any is True, "Expected stream to make progress when deadline is not enforced"


async def test_deadline_operations_complete_with_adequate_budget(adapter):
    """
    Operations SHOULD complete successfully when adequate budget is provided (when deadlines are supported).
    """
    caps = await adapter.capabilities()
    now_ms = int(time.time() * 1000)
    ctx = OperationContext(
        request_id="t_adequate_budget",
        tenant="test",
        deadline_ms=now_ms + 30_000,
    )

    result = await adapter.complete(
        messages=[{"role": "user", "content": "test with adequate budget"}],
        model=caps.supported_models[0],
        ctx=ctx,
    )

    assert isinstance(result.text, str)
    remaining = get_remaining_budget_ms(ctx)
    assert remaining is None or remaining >= 0, "Budget should not go negative after successful operation"


async def test_deadline_budget_calculation_accuracy(adapter):
    """
    Budget calculations should be reasonably accurate (within expected bounds).
    """
    caps = await adapter.capabilities()
    now_ms = int(time.time() * 1000)
    initial_budget = 1000  # 1 second
    ctx = OperationContext(
        request_id="t_budget_accuracy",
        tenant="test",
        deadline_ms=now_ms + initial_budget,
    )

    initial_remaining = get_remaining_budget_ms(ctx)
    assert initial_remaining is not None
    assert initial_remaining <= initial_budget

    if caps.supports_count_tokens:
        await adapter.count_tokens("quick operation", ctx=ctx)

    final_remaining = get_remaining_budget_ms(ctx)
    assert final_remaining is not None
    assert final_remaining <= initial_remaining, "Budget should decrease after operation"
    assert final_remaining >= 0, "Budget should never be negative"
