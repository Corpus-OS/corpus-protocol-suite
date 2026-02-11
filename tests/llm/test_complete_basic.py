# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Basic completion behavior.

Asserts:
  • Returns LLMCompletion with non-empty text (non-tool-calling turn)
  • Usage accounting is internally consistent
  • Model & family align with advertised capabilities
  • Finish reason is one of the allowed enums (includes tool_calls)
  • Tool calling paths are validated when supported
"""

import pytest
from corpus_sdk.llm.llm_base import (
    OperationContext,
    LLMCompletion,
    TokenUsage,
    NotSupported,
)

pytestmark = pytest.mark.asyncio

MIN_COMPLETION_LENGTH = 1
ALLOWED_FINISH_REASONS = {"stop", "length", "tool_calls", "tool_call", "content_filter", "error"}


async def test_core_ops_complete_basic_text_and_usage(adapter):
    ctx = OperationContext(request_id="t_complete_basic", tenant="test")
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    res = await adapter.complete(messages=[{"role": "user", "content": "hello"}], model=model, ctx=ctx)

    assert isinstance(res, LLMCompletion)
    assert isinstance(res.text, str) and len(res.text.strip()) >= MIN_COMPLETION_LENGTH
    assert isinstance(res.model, str) and res.model
    assert isinstance(res.model_family, str) and res.model_family
    assert res.model in caps.supported_models
    assert res.model_family == caps.model_family

    assert isinstance(res.usage, TokenUsage)
    assert isinstance(res.usage.prompt_tokens, int) and res.usage.prompt_tokens >= 0
    assert isinstance(res.usage.completion_tokens, int) and res.usage.completion_tokens >= 0
    assert isinstance(res.usage.total_tokens, int) and res.usage.total_tokens >= 0
    assert res.usage.total_tokens == res.usage.prompt_tokens + res.usage.completion_tokens

    assert isinstance(res.finish_reason, str) and res.finish_reason in ALLOWED_FINISH_REASONS


async def test_core_ops_complete_different_message_structures(adapter):
    ctx = OperationContext(request_id="t_complete_varied_msgs", tenant="test")
    caps = await adapter.capabilities()

    test_cases = [
        [{"role": "user", "content": "Hello, how are you?"}],
        [
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "I don't have real-time weather."},
            {"role": "user", "content": "Then what can you help with?"},
        ],
    ]

    for messages in test_cases:
        result = await adapter.complete(messages=messages, model=caps.supported_models[0], ctx=ctx)
        assert isinstance(result, LLMCompletion)
        assert isinstance(result.text, str)
        assert result.usage.total_tokens >= 0


async def test_core_ops_complete_empty_messages_rejected(adapter):
    ctx = OperationContext(request_id="t_complete_empty_msgs", tenant="test")
    caps = await adapter.capabilities()

    with pytest.raises(Exception) as exc_info:
        await adapter.complete(messages=[], model=caps.supported_models[0], ctx=ctx)

    msg = str(exc_info.value).lower()
    assert any(k in msg for k in ["message", "messages", "empty", "required"])


async def test_core_ops_complete_response_contains_expected_fields(adapter):
    ctx = OperationContext(request_id="t_complete_fields", tenant="test")
    caps = await adapter.capabilities()

    result = await adapter.complete(
        messages=[{"role": "user", "content": "test completion fields"}],
        model=caps.supported_models[0],
        ctx=ctx,
    )

    assert hasattr(result, "text") and isinstance(result.text, str)
    assert hasattr(result, "model") and isinstance(result.model, str)
    assert hasattr(result, "model_family") and isinstance(result.model_family, str)
    assert hasattr(result, "usage") and isinstance(result.usage, TokenUsage)
    assert hasattr(result, "finish_reason") and isinstance(result.finish_reason, str)
    assert hasattr(result, "tool_calls") and isinstance(result.tool_calls, list)


async def test_core_ops_complete_usage_accounting_consistent(adapter):
    ctx = OperationContext(request_id="t_complete_usage_consistency", tenant="test")
    caps = await adapter.capabilities()

    for _ in range(3):
        result = await adapter.complete(
            messages=[{"role": "user", "content": "Count the tokens in this message."}],
            model=caps.supported_models[0],
            ctx=ctx,
        )
        assert result.usage.total_tokens == result.usage.prompt_tokens + result.usage.completion_tokens


async def test_core_ops_complete_different_models_produce_results(adapter):
    ctx = OperationContext(request_id="t_complete_all_models", tenant="test")
    caps = await adapter.capabilities()

    for model in caps.supported_models[:3]:
        result = await adapter.complete(
            messages=[{"role": "user", "content": f"Test response from {model}"}],
            model=model,
            ctx=ctx,
        )
        assert isinstance(result, LLMCompletion)
        assert isinstance(result.text, str)


async def test_complete_system_message_gated_by_capability(adapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(request_id="t_system_message_gate", tenant="test")

    if caps.supports_system_message:
        res = await adapter.complete(
            messages=[{"role": "user", "content": "hello"}],
            model=caps.supported_models[0],
            system_message="You are helpful.",
            ctx=ctx,
        )
        assert isinstance(res, LLMCompletion)
    else:
        with pytest.raises(NotSupported):
            await adapter.complete(
                messages=[{"role": "user", "content": "hello"}],
                model=caps.supported_models[0],
                system_message="You are helpful.",
                ctx=ctx,
            )


async def test_complete_tools_happy_path_emits_tool_calls_when_supported(adapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(request_id="t_tools_required", tenant="test")

    tools = [{"type": "function", "function": {"name": "echo", "parameters": {"type": "object"}}}]

    if caps.supports_tools:
        res = await adapter.complete(
            messages=[{"role": "user", "content": "call: please echo this"}],
            model=caps.supported_models[0],
            tools=tools,
            tool_choice="required",
            ctx=ctx,
        )
        assert isinstance(res, LLMCompletion)
        assert res.finish_reason in {"tool_calls", "tool_call"}
        assert isinstance(res.tool_calls, list) and len(res.tool_calls) >= 1
        assert isinstance(res.text, str)  # may be empty
    else:
        with pytest.raises(NotSupported):
            await adapter.complete(
                messages=[{"role": "user", "content": "call: please echo this"}],
                model=caps.supported_models[0],
                tools=tools,
                tool_choice="required",
                ctx=ctx,
            )


async def test_complete_tool_choice_none_does_not_emit_tool_calls(adapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(request_id="t_tools_none", tenant="test")

    tools = [{"type": "function", "function": {"name": "echo", "parameters": {"type": "object"}}}]

    if caps.supports_tools:
        res = await adapter.complete(
            messages=[{"role": "user", "content": "call: please echo this"}],
            model=caps.supported_models[0],
            tools=tools,
            tool_choice="none",
            ctx=ctx,
        )
        assert isinstance(res, LLMCompletion)
        assert isinstance(res.tool_calls, list)
        assert len(res.tool_calls) == 0
        assert res.finish_reason in ALLOWED_FINISH_REASONS
    else:
        with pytest.raises(NotSupported):
            await adapter.complete(
                messages=[{"role": "user", "content": "call: please echo this"}],
                model=caps.supported_models[0],
                tools=tools,
                tool_choice="none",
                ctx=ctx,
            )
