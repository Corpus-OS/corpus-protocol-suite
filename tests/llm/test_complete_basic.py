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
    """
    SPECIFICATION.md §8.3 — Complete Operation
    
    Validates basic completion contract: response structure, token accounting,
    and finish reason enumeration.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_complete_basic", tenant="test")
    
    res = await adapter.complete(
        messages=[{"role": "user", "content": "hello"}],
        model="mock-model",
        ctx=ctx,
    )
    
    # Response structure
    assert isinstance(res, LLMCompletion), "Must return LLMCompletion instance"
    assert isinstance(res.text, str) and res.text.strip(), "Text must be non-empty"
    
    # Model validation
    assert res.model in ("mock-model", "mock-model-pro"), \
        f"Model '{res.model}' not in supported list"
    assert isinstance(res.model_family, str) and res.model_family == "mock"
    
    # Token usage validation
    assert isinstance(res.usage, TokenUsage), "Usage must be TokenUsage instance"
    assert res.usage.prompt_tokens >= 0, "prompt_tokens must be non-negative"
    assert res.usage.completion_tokens >= 0, "completion_tokens must be non-negative"
    assert res.usage.total_tokens >= 0, "total_tokens must be non-negative"
    assert res.usage.total_tokens == res.usage.prompt_tokens + res.usage.completion_tokens, \
        "total_tokens must equal prompt_tokens + completion_tokens"
    
    # Finish reason
    assert res.finish_reason in ("stop", "length", "tool_call", "content_filter"), \
        f"Invalid finish_reason: '{res.finish_reason}'"
