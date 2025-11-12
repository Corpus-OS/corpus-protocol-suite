# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Basic completion behavior.

Asserts:
  • Returns LLMCompletion with non-empty text
  • Usage accounting is internally consistent
  • Model & family align with advertised capabilities
  • Finish reason is one of the allowed enums
"""

import pytest

from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.llm.llm_base import (
    OperationContext,
    LLMCompletion,
    TokenUsage,
)
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_complete_basic_text_and_usage():
    """
    SPECIFICATION.md §8.3 — complete()

    Validates:
      - Response shape (LLMCompletion)
      - Non-empty text
      - Usage totals consistent
      - Model/model_family consistent with capabilities
      - finish_reason is a valid enum value
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_complete_basic", tenant="test")

    caps = await adapter.capabilities()

    res = await adapter.complete(
        messages=[{"role": "user", "content": "hello"}],
        model="mock-model",
        ctx=ctx,
    )

    # Response structure
    assert isinstance(res, LLMCompletion), "must return LLMCompletion instance"
    assert isinstance(res.text, str) and res.text.strip(), "text must be non-empty"

    # Model & family consistency with capabilities
    assert isinstance(res.model, str) and res.model, "model must be a non-empty string"

    if caps.supported_models:
        assert res.model in caps.supported_models, (
            f"model '{res.model}' must be one of capabilities.supported_models"
        )

    assert isinstance(res.model_family, str) and res.model_family, \
        "model_family must be non-empty"
    assert res.model_family == caps.model_family, \
        "model_family must match capabilities.model_family"

    # Token usage validation
    assert isinstance(res.usage, TokenUsage), "usage must be TokenUsage instance"
    assert isinstance(res.usage.prompt_tokens, int) and res.usage.prompt_tokens >= 0
    assert isinstance(res.usage.completion_tokens, int) and res.usage.completion_tokens >= 0
    assert isinstance(res.usage.total_tokens, int) and res.usage.total_tokens >= 0
    assert res.usage.total_tokens == (
        res.usage.prompt_tokens + res.usage.completion_tokens
    ), "total_tokens must equal prompt_tokens + completion_tokens"

    # Finish reason: must be a documented enum value
    allowed_finish_reasons = {"stop", "length", "tool_call", "content_filter", "error"}
    assert res.finish_reason in allowed_finish_reasons, \
        f"invalid finish_reason: {res.finish_reason!r}"
