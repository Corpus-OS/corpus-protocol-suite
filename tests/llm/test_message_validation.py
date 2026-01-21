# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Message structure validation.

Specification references:
  • §8.3   (LLM Protocol — Message Structure): required fields and content types
  • §8.3.1 (Role Validation): allowed roles and required fields (provider-variant)
  • §8.3.2 (Content Validation): string content requirements (provider-variant)
  • §12.4  (Error Handling): BadRequest for invalid message structures

Contract-first (BaseLLMAdapter) invariants (MUST, conformance-hard):
  • messages must be a non-empty list
  • each message must be a mapping/object
  • each message must include 'role' and 'content' keys
  • role and content MUST be strings
  • messages must be JSON-serializable
  • extra keys are allowed

Provider / adapter variance (SHOULD / MAY, conformance-soft):
  • Role enums (user/assistant/system/tool) may be enforced by some adapters
  • Empty or whitespace-only content may be rejected or accepted depending on provider/policy
  • Content size limits vary widely across providers
  • Tool-role linkage (tool_call_id) may be enforced by orchestration layers, not Base

This file:
  • Drives assertions through adapter.complete() and adapter.capabilities()
  • Does NOT use pytest.skip()
  • Uses capability↔behavior alignment where the Base contract exposes a capability knob
"""

import json
import pytest
from corpus_sdk.llm.llm_base import OperationContext, BadRequest

pytestmark = pytest.mark.asyncio

# Constants for message validation (some are robustness probes, not Base hard requirements)
ALLOWED_ROLES = {"user", "assistant", "system", "tool"}  # advisory; some adapters enforce, Base does not
MAX_CONTENT_LENGTH = 1_000_000  # robustness probe (providers vary)


async def test_message_validation_empty_messages_list_rejected(adapter):
    """
    §8.3 — Empty messages list MUST be rejected with BadRequest (Base contract).
    """
    ctx = OperationContext(request_id="t_msg_empty", tenant="test")
    caps = await adapter.capabilities()

    with pytest.raises(BadRequest) as exc_info:
        await adapter.complete(messages=[], model=caps.supported_models[0], ctx=ctx)

    error_msg = str(exc_info.value).lower()
    assert any(k in error_msg for k in ["message", "messages", "empty", "required", "missing"]), (
        f"Error should mention empty messages, got: {error_msg}"
    )


async def test_message_validation_each_message_must_be_mapping(adapter):
    """
    §8.3 — Each message MUST be a mapping/object (Base contract).
    """
    ctx = OperationContext(request_id="t_msg_not_mapping", tenant="test")
    caps = await adapter.capabilities()

    with pytest.raises(BadRequest) as exc_info:
        await adapter.complete(messages=["not-a-dict"], model=caps.supported_models[0], ctx=ctx)

    msg = str(exc_info.value).lower()
    assert any(k in msg for k in ["mapping", "object", "message"]), f"Expected mapping-related error, got: {msg}"


async def test_message_validation_missing_role_field_rejected(adapter):
    """
    §8.3 — Messages missing 'role' field MUST be rejected (Base contract).
    """
    ctx = OperationContext(request_id="t_msg_no_role", tenant="test")
    caps = await adapter.capabilities()

    with pytest.raises(BadRequest) as exc_info:
        await adapter.complete(messages=[{"content": "message without role"}], model=caps.supported_models[0], ctx=ctx)

    error_msg = str(exc_info.value).lower()
    assert "role" in error_msg, f"Error should mention missing role, got: {error_msg}"


async def test_message_validation_missing_content_field_rejected(adapter):
    """
    §8.3 — Messages missing 'content' field MUST be rejected (Base contract).
    """
    ctx = OperationContext(request_id="t_msg_no_content", tenant="test")
    caps = await adapter.capabilities()

    with pytest.raises(BadRequest) as exc_info:
        await adapter.complete(messages=[{"role": "user"}], model=caps.supported_models[0], ctx=ctx)

    error_msg = str(exc_info.value).lower()
    assert "content" in error_msg, f"Error should mention missing content, got: {error_msg}"


async def test_message_validation_role_and_content_type_enforced(adapter):
    """
    §8.3 — role/content MUST be strings (Base contract).
    """
    ctx = OperationContext(request_id="t_msg_types", tenant="test")
    caps = await adapter.capabilities()

    bad_messages = [
        [{"role": 123, "content": "x"}],
        [{"role": "user", "content": 123}],
        [{"role": None, "content": "x"}],
        [{"role": "user", "content": None}],
        [{"role": {}, "content": "x"}],
        [{"role": "user", "content": {}}],
    ]

    for msgs in bad_messages:
        with pytest.raises(BadRequest):
            await adapter.complete(messages=msgs, model=caps.supported_models[0], ctx=ctx)


async def test_message_validation_valid_roles_accepted(adapter):
    """
    §8.3.1 — Advisory role enums.
    Base does not enforce role enums, but valid/common roles should generally work.
    """
    ctx = OperationContext(request_id="t_msg_valid_roles", tenant="test")
    caps = await adapter.capabilities()

    # Basic roles (most providers)
    for role in ["user", "assistant"]:
        result = await adapter.complete(
            messages=[{"role": role, "content": f"test message with {role} role"}],
            model=caps.supported_models[0],
            ctx=ctx,
        )
        assert isinstance(result.text, str)


async def test_message_validation_invalid_role_rejected_or_descriptive(adapter):
    """
    §8.3.1 — Some adapters enforce role enums; others accept any string role (Base allows any string).
    Conformance rule:
      - If adapter rejects an unknown role string, it MUST raise BadRequest with a descriptive message.
      - If adapter accepts it, completion must still be well-formed.
    """
    ctx = OperationContext(request_id="t_msg_invalid_role", tenant="test")
    caps = await adapter.capabilities()

    invalid_role_strings = ["invalid_role", "admin", "bot", "  "]  # strings only; Base allows
    for r in invalid_role_strings:
        try:
            res = await adapter.complete(messages=[{"role": r, "content": "test"}], model=caps.supported_models[0], ctx=ctx)
            assert isinstance(res.text, str)
        except BadRequest as e:
            msg = str(e).lower()
            assert any(k in msg for k in ["role", "invalid", "message"]), f"Error should mention role invalidity, got: {msg}"


async def test_message_validation_empty_role_string_rejected_or_descriptive(adapter):
    """
    Empty role string is provider-variant. Base allows it (string), but many adapters reject it.
    Conformance rule:
      - If rejected, must be BadRequest with descriptive message.
      - If accepted, must still produce a valid completion.
    """
    ctx = OperationContext(request_id="t_msg_empty_role_str", tenant="test")
    caps = await adapter.capabilities()

    try:
        res = await adapter.complete(messages=[{"role": "", "content": "test"}], model=caps.supported_models[0], ctx=ctx)
        assert isinstance(res.text, str)
    except BadRequest as e:
        msg = str(e).lower()
        assert any(k in msg for k in ["role", "invalid", "empty", "message"]), f"Expected role-related error, got: {msg}"


async def test_message_validation_system_role_requires_capability_best_effort(adapter):
    """
    Prior version tied system-role messages to supports_system_message.
    In llm_base.py, supports_system_message gates the *system_message parameter*,
    not the 'system' role in messages. Therefore:
      - System role in messages MAY be accepted regardless of supports_system_message.
      - If rejected, must be BadRequest with descriptive message.
    """
    ctx = OperationContext(request_id="t_msg_system_role", tenant="test")
    caps = await adapter.capabilities()

    try:
        res = await adapter.complete(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
            model=caps.supported_models[0],
            ctx=ctx,
        )
        assert isinstance(res.text, str)
    except BadRequest as e:
        msg = str(e).lower()
        assert any(k in msg for k in ["system", "role", "message"]), f"Expected system-role related error, got: {msg}"


async def test_message_validation_empty_content_rejected_for_user_role(adapter):
    """
    §8.3.2 — Empty content handling is provider-variant.
    Base requires content is a string, but does not require non-empty.

    Conformance rule:
      - Non-string content MUST raise BadRequest (covered elsewhere).
      - Empty/whitespace-only strings MAY be accepted or rejected.
        If rejected, must be BadRequest with descriptive message.
        If accepted, completion must be well-formed.
    """
    ctx = OperationContext(request_id="t_msg_empty_content", tenant="test")
    caps = await adapter.capabilities()

    empty_content_variants = ["", "   ", "\t\n"]

    for c in empty_content_variants:
        try:
            res = await adapter.complete(messages=[{"role": "user", "content": c}], model=caps.supported_models[0], ctx=ctx)
            assert isinstance(res.text, str)
        except BadRequest as e:
            msg = str(e).lower()
            assert any(k in msg for k in ["content", "empty", "whitespace", "invalid", "message"]), (
                f"Expected descriptive empty-content error, got: {msg}"
            )


async def test_message_validation_whitespace_only_content_rejected(adapter):
    """
    Same as above, kept as a dedicated regression test since many suites include it.

    Conformance rule:
      - May succeed or raise BadRequest; must not crash; if rejected, message should be descriptive.
    """
    ctx = OperationContext(request_id="t_msg_whitespace", tenant="test")
    caps = await adapter.capabilities()

    whitespace_variants = [" ", "  ", "\t", "\n", "\t\n ", " \t \n "]

    for c in whitespace_variants:
        try:
            res = await adapter.complete(messages=[{"role": "user", "content": c}], model=caps.supported_models[0], ctx=ctx)
            assert isinstance(res.text, str)
        except BadRequest as e:
            msg = str(e).lower()
            assert any(k in msg for k in ["content", "empty", "whitespace", "invalid", "message"]), (
                f"Expected descriptive whitespace-content error, got: {msg}"
            )


async def test_message_validation_content_too_large_rejected(adapter):
    """
    §8.3.2 — Max content limits are provider-variant.

    Conformance rule:
      - Must not crash.
      - If rejected, should be BadRequest (or another normalized error) with descriptive message.
      - If accepted, should produce a valid completion.
    """
    ctx = OperationContext(request_id="t_msg_large_content", tenant="test")
    caps = await adapter.capabilities()

    large_content = "x" * (MAX_CONTENT_LENGTH + 1000)

    try:
        res = await adapter.complete(messages=[{"role": "user", "content": large_content}], model=caps.supported_models[0], ctx=ctx)
        assert isinstance(res.text, str)
    except BadRequest as e:
        msg = str(e).lower()
        assert any(k in msg for k in ["content", "large", "length", "limit", "exceed", "too"]), (
            f"Expected descriptive large-content error, got: {msg}"
        )


async def test_message_validation_valid_content_types_accepted(adapter):
    """
    §8.3.2 — Various valid content formats SHOULD be accepted (strings).
    """
    ctx = OperationContext(request_id="t_msg_content_types", tenant="test")
    caps = await adapter.capabilities()

    valid_content_cases = [
        "Simple text message",
        "Text with punctuation: Hello, world! How are you?",
        "Text with numbers: 12345",
        "Text with unicode: Hello 世界 🌍",
        "Multi-line text: Line 1\nLine 2\nLine 3",
        "Text with special chars: @#$%^&*()",
    ]

    for content in valid_content_cases:
        res = await adapter.complete(messages=[{"role": "user", "content": content}], model=caps.supported_models[0], ctx=ctx)
        assert isinstance(res.text, str)


async def test_message_validation_conversation_structure_accepted(adapter):
    """
    §8.3 — Valid conversation structures MUST be accepted (as long as shape is valid).
    """
    ctx = OperationContext(request_id="t_msg_convo_struct", tenant="test")
    caps = await adapter.capabilities()

    valid_conversations = [
        [{"role": "user", "content": "Hello"}],
        [
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "2+2=4"},
            {"role": "user", "content": "Thanks!"},
        ],
        [
            {"role": "user", "content": "First question"},
            {"role": "user", "content": "Second question"},
        ],
        [
            {"role": "system", "content": "Be helpful and concise."},
            {"role": "user", "content": "Follow instructions"},
        ],
    ]

    for msgs in valid_conversations:
        res = await adapter.complete(messages=msgs, model=caps.supported_models[0], ctx=ctx)
        assert isinstance(res.text, str)


async def test_message_validation_tool_role_requires_tool_call_id(adapter):
    """
    §8.3.1 — Tool role linkage is provider/orchestrator variant; Base does not enforce tool_call_id.

    Conformance rule:
      - Must not crash.
      - If rejected, should be BadRequest with descriptive message.
      - If accepted, should return a valid completion.
    """
    ctx = OperationContext(request_id="t_msg_tool_role", tenant="test")
    caps = await adapter.capabilities()

    msgs_with_id = [
        {"role": "user", "content": "Use a tool"},
        {"role": "tool", "content": "Tool result", "tool_call_id": "call_123"},
    ]

    try:
        res = await adapter.complete(messages=msgs_with_id, model=caps.supported_models[0], ctx=ctx)
        assert isinstance(res.text, str)
    except BadRequest:
        # acceptable; ensure descriptive elsewhere
        pass

    msgs_without_id = [
        {"role": "user", "content": "Use a tool"},
        {"role": "tool", "content": "Tool result"},
    ]

    try:
        res2 = await adapter.complete(messages=msgs_without_id, model=caps.supported_models[0], ctx=ctx)
        assert isinstance(res2.text, str)
    except BadRequest as e:
        msg = str(e).lower()
        assert any(k in msg for k in ["tool", "role", "message"]), f"Expected tool-role related error, got: {msg}"


async def test_message_validation_mixed_invalid_and_valid_rejected(adapter):
    """
    If any message violates required shape (Base contract), the call MUST be rejected.
    """
    ctx = OperationContext(request_id="t_msg_mixed_invalid", tenant="test")
    caps = await adapter.capabilities()

    mixed_messages = [
        {"role": "user", "content": "Valid message"},
        {"role": "user"},  # invalid: missing content (hard Base requirement)
        {"role": "user", "content": "Another valid message"},
    ]

    with pytest.raises(BadRequest) as exc_info:
        await adapter.complete(messages=mixed_messages, model=caps.supported_models[0], ctx=ctx)

    msg = str(exc_info.value).lower()
    assert any(k in msg for k in ["content", "missing", "message", "messages"]), f"Expected missing-content error, got: {msg}"


async def test_message_validation_error_messages_are_descriptive(adapter):
    """
    §12.4 — Validation errors SHOULD provide descriptive messages.
    """
    ctx = OperationContext(request_id="t_msg_error_desc", tenant="test")
    caps = await adapter.capabilities()

    test_cases = [
        ([], "empty messages list"),
        ([{"content": "no role"}], "missing role field"),
        ([{"role": "user"}], "missing content field"),
        ([{"role": 1, "content": "x"}], "non-string role"),
        ([{"role": "user", "content": 2}], "non-string content"),
    ]

    for messages, desc in test_cases:
        with pytest.raises(BadRequest) as exc_info:
            await adapter.complete(messages=messages, model=caps.supported_models[0], ctx=ctx)

        error_msg = str(exc_info.value)
        assert error_msg and error_msg.strip(), f"Error message should not be empty for {desc}"
        assert len(error_msg) < 1000, f"Error message should be reasonable length for {desc}"


async def test_message_validation_extra_keys_are_ignored(adapter):
    """
    Extra keys in message dicts MUST NOT cause validation failure (Base contract).
    """
    ctx = OperationContext(request_id="t_msg_extra_keys", tenant="test")
    caps = await adapter.capabilities()

    msgs = [
        {"role": "user", "content": "hello", "foo": "bar", "n": 1},
        {"role": "assistant", "content": "hi", "metadata": {"k": "v"}},
    ]

    json.dumps(msgs)
    res = await adapter.complete(messages=msgs, model=caps.supported_models[0], ctx=ctx)
    assert isinstance(res.text, str)


async def test_message_validation_messages_must_be_json_serializable(adapter):
    """
    Messages must be JSON-serializable (Base contract).
    """
    ctx = OperationContext(request_id="t_msg_json", tenant="test")
    caps = await adapter.capabilities()

    class _X:
        pass

    msgs = [{"role": "user", "content": "ok", "extra": _X()}]

    with pytest.raises(BadRequest):
        await adapter.complete(messages=msgs, model=caps.supported_models[0], ctx=ctx)


async def test_message_validation_max_reasonable_messages_accepted(adapter):
    """
    Conversations with reasonable numbers of messages SHOULD be accepted (robustness).
    """
    ctx = OperationContext(request_id="t_msg_reasonable_count", tenant="test")
    caps = await adapter.capabilities()

    msgs = []
    for i in range(10):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"Message {i+1} from {role}"})

    res = await adapter.complete(messages=msgs, model=caps.supported_models[0], ctx=ctx)
    assert isinstance(res.text, str)
