# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Token counting consistency.

Asserts:
  • count_tokens returns non-negative ints
  • Longer texts yield counts >= shorter texts (monotonic heuristic)
"""

import pytest

from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.llm.llm_base import OperationContext
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_count_tokens_monotonic():
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_count_tokens", tenant="test")

    s1 = "short text"
    s2 = "short text plus some more words here"
    c1 = await adapter.count_tokens(s1, ctx=ctx)
    c2 = await adapter.count_tokens(s2, ctx=ctx)

    assert isinstance(c1, int) and isinstance(c2, int)
    assert c1 >= 0 and c2 >= 0
    assert c2 >= c1  # longer text shouldn't return fewer tokens under heuristic
