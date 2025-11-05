# SPDX-License-Identifier: Apache-2.0
import pytest

from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.llm.llm_base import OperationContext
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.errors import BadRequest

pytestmark = pytest.mark.asyncio


async def test_rejects_unknown_roles():
    """Verify adapter validates message roles."""
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_msg_role_unknown", tenant="test")

    with pytest.raises(BadRequest) as exc_info:
        await adapter.complete(
            messages=[{"role": "alien", "content": "hello from zeta reticuli"}],
            model="mock-model",
            ctx=ctx,
        )
    assert "role" in str(exc_info.value.message).lower()


async def test_requires_nonempty_messages():
    """Verify adapter rejects empty message lists."""
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_msg_empty", tenant="test")

    with pytest.raises(BadRequest) as exc_info:
        await adapter.complete(messages=[], model="mock-model", ctx=ctx)
    assert "empty" in str(exc_info.value.message).lower()


async def test_requires_role_and_content_fields():
    """Verify adapter validates required message fields."""
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_msg_fields", tenant="test")

    # Missing 'role'
    with pytest.raises(BadRequest) as exc_info:
        await adapter.complete(
            messages=[{"content": "hi"}], 
            model="mock-model", 
            ctx=ctx
        )
    assert "role" in str(exc_info.value.message).lower()

    # Missing 'content'  
    with pytest.raises(BadRequest) as exc_info:
        await adapter.complete(
            messages=[{"role": "user"}],
            model="mock-model",
            ctx=ctx
        )
    assert "content" in str(exc_info.value.message).lower()


async def test_accepts_standard_roles():
    """Verify adapter accepts standard conversation roles."""
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_msg_roles_ok", tenant="test")

    # Test all standard roles
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
    """Verify adapter handles reasonably sized message content."""
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_msg_large", tenant="test")

    # Large but reasonable content
    large_content = "x" * 10000
    res = await adapter.complete(
        messages=[{"role": "user", "content": large_content}],
        model="mock-model", 
        ctx=ctx,
    )
    assert isinstance(res.text, str)
