# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Basic completion behavior.

Asserts:
  • Returns LLMCompletion with non-empty text
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
)

pytestmark = pytest.mark.asyncio

# Constants for completion validation
MIN_COMPLETION_LENGTH = 1  # Minimum expected completion length
ALLOWED_FINISH_REASONS = {"stop", "length", "tool_call", "content_filter", "error"}


async def test_core_ops_complete_basic_text_and_usage(adapter):
    """
    SPECIFICATION.md §8.3 — complete()

    Validates:
      - Response shape (LLMCompletion)
      - Non-empty text
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

    # Response structure validation
    assert isinstance(res, LLMCompletion), "must return LLMCompletion instance"
    assert isinstance(res.text, str) and len(res.text.strip()) >= MIN_COMPLETION_LENGTH, \
        f"text must be non-empty string (min {MIN_COMPLETION_LENGTH} char)"

    # Model & family consistency with capabilities
    assert isinstance(res.model, str) and res.model, "model must be a non-empty string"

    if caps.supported_models:
        assert res.model in caps.supported_models, (
            f"model '{res.model}' must be one of capabilities.supported_models: {caps.supported_models}"
        )

    assert isinstance(res.model_family, str) and res.model_family, \
        "model_family must be non-empty string"
    assert res.model_family == caps.model_family, \
        f"model_family '{res.model_family}' must match capabilities.model_family '{caps.model_family}'"

    # Token usage validation
    assert isinstance(res.usage, TokenUsage), "usage must be TokenUsage instance"
    assert isinstance(res.usage.prompt_tokens, int) and res.usage.prompt_tokens >= 0, \
        "prompt_tokens must be non-negative integer"
    assert isinstance(res.usage.completion_tokens, int) and res.usage.completion_tokens >= 0, \
        "completion_tokens must be non-negative integer"
    assert isinstance(res.usage.total_tokens, int) and res.usage.total_tokens >= 0, \
        "total_tokens must be non-negative integer"
    assert res.usage.total_tokens == (res.usage.prompt_tokens + res.usage.completion_tokens), \
        f"total_tokens ({res.usage.total_tokens}) must equal prompt_tokens ({res.usage.prompt_tokens}) + completion_tokens ({res.usage.completion_tokens})"

    # Finish reason validation
    assert res.finish_reason in ALLOWED_FINISH_REASONS, \
        f"invalid finish_reason: {res.finish_reason!r}, must be one of {ALLOWED_FINISH_REASONS}"


async def test_core_ops_complete_different_message_structures(adapter):
    """
    Test complete() with various message structures and roles.
    """
    ctx = OperationContext(request_id="t_complete_varied_msgs", tenant="test")
    caps = await adapter.capabilities()

    test_cases = [
        # Basic user message
        [{"role": "user", "content": "Hello, how are you?"}],
        
        # Conversation history
        [
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "I'm a language model, I don't have real-time weather data."},
            {"role": "user", "content": "Then what can you help with?"}
        ],
    ]

    # Add system message test if supported
    if caps.supports_system_message:
        test_cases.append([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Follow your system instructions."}
        ])

    for messages in test_cases:
        result = await adapter.complete(
            messages=messages,
            model=caps.supported_models[0],
            ctx=ctx,
        )
        
        # Basic response validation
        assert isinstance(result, LLMCompletion)
        assert isinstance(result.text, str) and result.text.strip()
        assert result.usage.total_tokens >= 0


async def test_core_ops_complete_empty_messages_rejected(adapter):
    """
    Empty messages list should be rejected with appropriate error.
    """
    ctx = OperationContext(request_id="t_complete_empty_msgs", tenant="test")
    caps = await adapter.capabilities()

    with pytest.raises(Exception) as exc_info:  # Could be BadRequest, ValueError, etc.
        await adapter.complete(
            messages=[],  # Empty messages
            model=caps.supported_models[0],
            ctx=ctx,
        )
    
    # Should provide meaningful error message
    error_msg = str(exc_info.value).lower()
    assert any(keyword in error_msg for keyword in ["message", "empty", "required"]), \
        f"Error should mention messages requirement, got: {error_msg}"


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

    # Required fields
    assert hasattr(result, 'text') and isinstance(result.text, str)
    assert hasattr(result, 'model') and isinstance(result.model, str)
    assert hasattr(result, 'model_family') and isinstance(result.model_family, str)
    assert hasattr(result, 'usage') and isinstance(result.usage, TokenUsage)
    assert hasattr(result, 'finish_reason') and isinstance(result.finish_reason, str)

    # Optional fields (if present, must be correct type)
    if hasattr(result, 'id'):
        assert isinstance(result.id, str)
    if hasattr(result, 'created'):
        assert isinstance(result.created, int)


async def test_core_ops_complete_usage_accounting_consistent(adapter):
    """
    Token usage accounting should be consistent across multiple calls.
    """
    ctx = OperationContext(request_id="t_complete_usage_consistency", tenant="test")
    caps = await adapter.capabilities()

    # Same input should produce similar usage (allowing for small variations)
    message = "Count the tokens in this message."
    
    usages = []
    for i in range(3):
        result = await adapter.complete(
            messages=[{"role": "user", "content": message}],
            model=caps.supported_models[0],
            ctx=ctx,
        )
        usages.append(result.usage)
        
        # Basic usage validation
        assert result.usage.total_tokens >= result.usage.prompt_tokens
        assert result.usage.total_tokens >= result.usage.completion_tokens

    # Usage should be reasonably consistent (within 20% variation)
    total_tokens = [u.total_tokens for u in usages]
    max_variation = max(total_tokens) - min(total_tokens)
    avg_tokens = sum(total_tokens) / len(total_tokens)
    
    # Allow for some variation but not extreme differences
    assert max_variation <= avg_tokens * 0.2, \
        f"Token usage varies too much: {total_tokens}"


async def test_core_ops_complete_different_models_produce_results(adapter):
    """
    All supported models should produce valid completions.
    """
    ctx = OperationContext(request_id="t_complete_all_models", tenant="test")
    caps = await adapter.capabilities()

    for model in caps.supported_models[:3]:  # Test first 3 models to avoid timeout
        result = await adapter.complete(
            messages=[{"role": "user", "content": f"Test response from {model}"}],
            model=model,
            ctx=ctx,
        )
        
        assert isinstance(result, LLMCompletion), f"Model {model} should return LLMCompletion"
        assert result.text and result.text.strip(), f"Model {model} should produce non-empty text"
        assert result.model == model, f"Response model should match requested model: {result.model} != {model}"