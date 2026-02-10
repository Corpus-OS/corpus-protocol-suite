# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Deadline enforcement plumbing.
Asserts:
  • Remaining budget helper never returns negative
  • DeadlineExceeded raised when budget exhausted (§8.3, §12.4) IF supports_deadline=True
  • If supports_deadline=False, expired deadlines are not enforced
"""
import time
import pytest
from typing import Optional
from corpus_sdk.llm.llm_base import OperationContext, DeadlineExceeded, NotSupported

pytestmark = pytest.mark.asyncio

GUARANTEED_EXPIRED_MS = 0


def get_remaining_budget_ms(ctx: OperationContext) -> Optional[int]:
    if ctx.deadline_ms is None:
        return None
    now_ms = int(time.time() * 1000)
    return max(0, ctx.deadline_ms - now_ms)


async def test_deadline_deadline_budget_nonnegative_and_usable(adapter):
    now_ms = int(time.time() * 1000)
    ctx = OperationContext(request_id="t_deadline_plumbing", tenant="test", deadline_ms=now_ms + 200)

    rem = get_remaining_budget_ms(ctx)
    assert rem is not None and rem >= 0
    assert rem <= 200

    caps = await adapter.capabilities()
    if caps.supports_count_tokens:
        await adapter.count_tokens("tiny", model=caps.supported_models[0], ctx=ctx)

    rem2 = get_remaining_budget_ms(ctx)
    assert rem2 is None or rem2 >= 0


async def test_deadline_deadline_exceeded_on_expired_budget(adapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(deadline_ms=GUARANTEED_EXPIRED_MS, tenant="test")

    if caps.supports_deadline:
        with pytest.raises(DeadlineExceeded) as exc:
            await adapter.complete(messages=[{"role": "user", "content": "test"}], ctx=ctx)
        assert exc.value.code == "DEADLINE_EXCEEDED"
        assert "remaining_ms" in (exc.value.details or {})
    else:
        res = await adapter.complete(messages=[{"role": "user", "content": "test"}], ctx=ctx)
        assert hasattr(res, "finish_reason")


async def test_deadline_deadline_exceeded_during_stream(adapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(deadline_ms=GUARANTEED_EXPIRED_MS, tenant="test")

    if not caps.supports_streaming:
        with pytest.raises(NotSupported):
            agen = adapter.stream(messages=[{"role": "user", "content": "test"}], model=caps.supported_models[0], ctx=ctx)
            async for _ in agen:
                pass
        return

    if caps.supports_deadline:
        with pytest.raises(DeadlineExceeded):
            agen = adapter.stream(messages=[{"role": "user", "content": "test"}], model=caps.supported_models[0], ctx=ctx)
            async for _ in agen:
                pass
    else:
        got_any = False
        async for _ in adapter.stream(messages=[{"role": "user", "content": "test"}], model=caps.supported_models[0], ctx=ctx):
            got_any = True
            break
        assert got_any is True


async def test_deadline_operations_complete_with_adequate_budget(adapter):
    caps = await adapter.capabilities()
    now_ms = int(time.time() * 1000)
    ctx = OperationContext(request_id="t_adequate_budget", tenant="test", deadline_ms=now_ms + 30_000)

    result = await adapter.complete(
        messages=[{"role": "user", "content": "test with adequate budget"}],
        model=caps.supported_models[0],
        ctx=ctx,
    )
    assert isinstance(result.text, str)
    remaining = get_remaining_budget_ms(ctx)
    assert remaining is None or remaining >= 0


async def test_deadline_budget_calculation_accuracy(adapter):
    caps = await adapter.capabilities()
    now_ms = int(time.time() * 1000)
    initial_budget = 1000
    ctx = OperationContext(request_id="t_budget_accuracy", tenant="test", deadline_ms=now_ms + initial_budget)

    initial_remaining = get_remaining_budget_ms(ctx)
    assert initial_remaining is not None
    assert initial_remaining <= initial_budget

    if caps.supports_count_tokens:
        await adapter.count_tokens("quick operation", model=caps.supported_models[0], ctx=ctx)

    final_remaining = get_remaining_budget_ms(ctx)
    assert final_remaining is not None
    assert final_remaining <= initial_remaining
    assert final_remaining >= 0


async def test_deadline_not_enforced_when_capability_false(adapter):
    """
    Explicit capability↔behavior assertion:
      - supports_deadline=False => expired deadline must NOT be enforced
      - supports_deadline=True  => expired deadline must be enforced
    """
    caps = await adapter.capabilities()
    ctx = OperationContext(deadline_ms=0, tenant="test")

    if caps.supports_deadline:
        with pytest.raises(DeadlineExceeded):
            await adapter.health(ctx=ctx)
    else:
        h = await adapter.health(ctx=ctx)
        assert isinstance(h, dict) and "ok" in h
