# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Basic completion behavior.

Asserts:
  • Returns LLMCompletion with non-empty text (for non-tool-calling turns)
  • Usage accounting is internally consistent
  • Model & family align with advertised capabilities
  • Finish reason is one of the allowed enums
  • Response structure validation
"""

import pytest
from corpus_sdk.llm.llm_base import (
    OperationContext,
    LLMCompletion,
    TokenUsage,
    NotSupported,
)

pytestmark = pytest.mark.asyncio

# Constants for completion validation
MIN_COMPLETION_LENGTH = 1  # Minimum expected completion length
ALLOWED_FINISH_REASONS = {"stop", "length", "tool_calls", "tool_call", "content_filter", "error"}


async def test_core_ops_complete_basic_text_and_usage(adapter):
    """
    SPECIFICATION.md §8.3 — complete()

    Validates:
      - Response shape (LLMCompletion)
      - Non-empty text (for non-tool-calling turns)
      - Usage totals consistent
      - Model/model_family consistent with capabilities
      - finish_reason is a valid enum value
    """
    ctx = OperationContext(request_id="t_complete_basic", tenant="test")

    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    res = await adapter.complete(
        messages=[{"role": "user", "content": "hello"}],
        model=model,
        ctx=ctx,
    )

    assert isinstance(res, LLMCompletion), "must return LLMCompletion instance"
    assert isinstance(res.text, str) and len(res.text.strip()) >= MIN_COMPLETION_LENGTH, (
        f"text must be non-empty string (min {MIN_COMPLETION_LENGTH} char)"
    )

    assert isinstance(res.model, str) and res.model, "model must be a non-empty string"
    if caps.supported_models:
        assert res.model in caps.supported_models, (
            f"model '{res.model}' must be one of capabilities.supported_models: {caps.supported_models}"
        )

    assert isinstance(res.model_family, str) and res.model_family, "model_family must be non-empty string"
    assert res.model_family == caps.model_family, (
        f"model_family '{res.model_family}' must match capabilities.model_family '{caps.model_family}'"
    )

    assert isinstance(res.usage, TokenUsage), "usage must be TokenUsage instance"
    assert isinstance(res.usage.prompt_tokens, int) and res.usage.prompt_tokens >= 0
    assert isinstance(res.usage.completion_tokens, int) and res.usage.completion_tokens >= 0
    assert isinstance(res.usage.total_tokens, int) and res.usage.total_tokens >= 0
    assert res.usage.total_tokens == (res.usage.prompt_tokens + res.usage.completion_tokens)

    assert res.finish_reason in ALLOWED_FINISH_REASONS, (
        f"invalid finish_reason: {res.finish_reason!r}, must be one of {ALLOWED_FINISH_REASONS}"
    )


async def test_core_ops_complete_different_message_structures(adapter):
    """
    Test complete() with various message structures and roles.
    """
    ctx = OperationContext(request_id="t_complete_varied_msgs", tenant="test")
    caps = await adapter.capabilities()

    test_cases = [
        [{"role": "user", "content": "Hello, how are you?"}],
        [
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "I'm a language model, I don't have real-time weather data."},
            {"role": "user", "content": "Then what can you help with?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Follow your system instructions."},
        ],
    ]

    for messages in test_cases:
        result = await adapter.complete(
            messages=messages,
            model=caps.supported_models[0],
            ctx=ctx,
        )

        assert isinstance(result, LLMCompletion)
        assert isinstance(result.text, str)
        assert result.usage.total_tokens >= 0


async def test_core_ops_complete_empty_messages_rejected(adapter):
    """
    Empty messages list should be rejected with an error that indicates invalid input.
    """
    ctx = OperationContext(request_id="t_complete_empty_msgs", tenant="test")
    caps = await adapter.capabilities()

    with pytest.raises(Exception) as exc_info:
        await adapter.complete(
            messages=[],
            model=caps.supported_models[0],
            ctx=ctx,
        )

    error_msg = str(exc_info.value).lower()
    assert any(keyword in error_msg for keyword in ["message", "messages", "empty", "required"]), (
        f"Error should mention messages requirement, got: {error_msg}"
    )


async def test_core_ops_complete_response_contains_expected_fields(adapter):
    """
    Verify LLMCompletion contains all expected fields with correct types.
    """
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

    # tool_calls is part of the contract (may be empty)
    assert hasattr(result, "tool_calls"), "LLMCompletion must have tool_calls field"
    assert isinstance(result.tool_calls, list), "tool_calls must be a list"


async def test_core_ops_complete_usage_accounting_consistent(adapter):
    """
    Token usage accounting should be internally consistent per call.
    (No cross-call determinism assumptions.)
    """
    ctx = OperationContext(request_id="t_complete_usage_consistency", tenant="test")
    caps = await adapter.capabilities()

    message = "Count the tokens in this message."

    for _ in range(3):
        result = await adapter.complete(
            messages=[{"role": "user", "content": message}],
            model=caps.supported_models[0],
            ctx=ctx,
        )

        assert result.usage.total_tokens >= 0
        assert result.usage.prompt_tokens >= 0
        assert result.usage.completion_tokens >= 0
        assert result.usage.total_tokens == (result.usage.prompt_tokens + result.usage.completion_tokens)


async def test_core_ops_complete_different_models_produce_results(adapter):
    """
    All supported models should produce valid completions.

    Note: Returned res.model may be a concrete model id (alias resolution),
    so conformance validates membership, not strict equality to requested string.
    """
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

        if caps.supported_models:
            assert result.model in caps.supported_models, (
                f"Response model '{result.model}' must be one of supported_models"
            )


async def test_complete_system_message_gated_by_capability(adapter):
    """
    Capability↔behavior alignment for system_message parameter:

      - If supports_system_message is False: system_message MUST raise NotSupported.
      - If True: system_message MUST be accepted.
    """
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
    """
    If tools are supported, tool_choice="required" MUST yield tool_calls and finish_reason tool_calls.
    If tools are not supported, the call MUST raise NotSupported.
    """
    caps = await adapter.capabilities()
    ctx = OperationContext(request_id="t_tools_required", tenant="test")

    tools = [
        {"type": "function", "function": {"name": "echo", "parameters": {"type": "object"}}},
    ]

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
        # Text may be empty for tool-call turns
        assert isinstance(res.text, str)
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
    """
    If tools are supported, tool_choice="none" MUST NOT emit tool_calls.
    If tools are not supported, complete(tools=...) MUST raise NotSupported.
    """
    caps = await adapter.capabilities()
    ctx = OperationContext(request_id="t_tools_none", tenant="test")

    tools = [
        {"type": "function", "function": {"name": "echo", "parameters": {"type": "object"}}},
    ]

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
