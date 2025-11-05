# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Basic completion behavior.

Asserts:
  • Non-empty text response
  • Usage accounting consistency
  • Model echoed as supported
"""

import pytest

from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.llm.llm_base import OperationContext
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_complete_basic_text_and_usage():
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_complete_basic", tenant="test")

    res = await adapter.complete(
        messages=[{"role": "user", "content": "hello"}],
        model="mock-model",
        ctx=ctx,
    )

    assert isinstance(res.text, str) and res.text.strip()
    assert res.model in ("mock-model", "mock-model-pro")

    # Usage should be internally consistent
    assert res.usage.total_tokens == res.usage.prompt_tokens + res.usage.completion_tokens
    assert res.finish_reason in ("stop", "length", "tool_call", "content_filter")
