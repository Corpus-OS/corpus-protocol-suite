# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Deadline enforcement plumbing.

Asserts:
  • Remaining budget is computed
  • Calls under short budgets still complete
  • Budget never goes negative in helper
"""

import pytest

from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.llm.llm_base import OperationContext
from corpus_sdk.examples.common.ctx import make_ctx, remaining_budget_ms

pytestmark = pytest.mark.asyncio


async def test_deadline_budget_nonnegative_and_usable():
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(
        OperationContext,
        request_id="t_deadline_plumbing",
        tenant="test",
        timeout_ms=200,  # relative deadline
    )

    rem = remaining_budget_ms(ctx)
    assert rem is not None and rem >= 0

    # Run a tiny operation under budget
    _ = await adapter.count_tokens("tiny", ctx=ctx)

    # Budget helper should never go negative
    rem2 = remaining_budget_ms(ctx)
    assert rem2 is None or rem2 >= 0
