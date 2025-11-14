# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Message validation.

Covers:
  • Rejection of unknown roles
  • Rejection of empty message arrays
  • Rejection of messages missing required fields
  • Acceptance of standard roles (system/user/assistant)
  • Handling of large-but-reasonable message content
"""

import pytest
from corpus_sdk.llm.llm_base import OperationContext, BadRequest
from examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize(
    "messages, expected_hint",
    [
        ([{"role": "alien", "content": "hello from zeta reticuli"}], "role"),
        ([], "empty"),
        ([{"content": "hi"}], "role"),        # missing role
        ([{"role": "user"}], "content"),      # missing content
    ],
)
async def test_message_validation_invalid_messages_rejected(adapter, messages, expected_hint):
    """
    Verify that the adapter (or base) enforces basic message schema rules.
    """
    caps = await adapter.capabilities()
    ctx = make_ctx(OperationContext, request_id="t_msg_invalid", tenant="test")

    with pytest.raises(BadRequest) as exc_info:
        await adapter.complete(messages=messages, model=caps.supported_models[0], ctx=ctx)

    # Error message should hint at the offending field/condition
    msg = str(getattr(exc_info.value, "message", exc_info.value)).lower()
    assert expected_hint in msg


async def test_message_validation_accepts_standard_roles(adapter):
    """
    The adapter should accept the canonical conversation roles.
    """
    caps = await adapter.capabilities()
    ctx = make_ctx(OperationContext, request_id="t_msg_roles_ok", tenant="test")

    res = await adapter.complete(
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Previous response"},
        ],
        model=caps.supported_models[0],
        ctx=ctx,
    )
    assert isinstance(res.text, str) and len(res.text) > 0


async def test_message_validation_handles_large_message_content(adapter):
    """
    Large-but-reasonable content should be processed without schema failures.
    (This does NOT test model context limits; just schema-level acceptance.)
    """
    caps = await adapter.capabilities()
    ctx = make_ctx(OperationContext, request_id="t_msg_large", tenant="test")

    large_content = "x" * 10_000  # ~10KB payload: reasonable for schema tests
    res = await adapter.complete(
        messages=[{"role": "user", "content": large_content}],
        model=caps.supported_models[0],
        ctx=ctx,
    )
    assert isinstance(res.text, str) and res.text.strip()