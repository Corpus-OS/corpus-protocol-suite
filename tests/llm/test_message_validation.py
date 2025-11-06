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

from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.llm.llm_base import OperationContext
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.errors import BadRequest

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
async def test_invalid_messages_rejected(messages, expected_hint):
    """
    Verify that the adapter (or base) enforces basic message schema rules.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_msg_invalid", tenant="test")

    with pytest.raises(BadRequest) as exc_info:
        await adapter.complete(messages=messages, model="mock-model", ctx=ctx)

    # Error message should hint at the offending field/condition
    msg = str(getattr(exc_info.value, "message", exc_info.value)).lower()
    assert expected_hint in msg


async def test_accepts_standard_roles():
    """
    The adapter should accept the canonical conversation roles.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_msg_roles_ok", tenant="test")

    res = await adapter.complete(
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Previous response"},
        ],
        model="mock-model",
        ctx=ctx,
    )
    assert isinstance(res.text, str) and len(res.text) > 0


async def test_handles_large_message_content():
    """
    Large-but-reasonable content should be processed without schema failures.
    (This does NOT test model context limits; just schema-level acceptance.)
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_msg_large", tenant="test")

    large_content = "x" * 10_000  # ~10KB payload: reasonable for schema tests
    res = await adapter.complete(
        messages=[{"role": "user", "content": large_content}],
        model="mock-model",
        ctx=ctx,
    )
    assert isinstance(res.text, str) and res.text.strip()
