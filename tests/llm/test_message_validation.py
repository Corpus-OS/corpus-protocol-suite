# SPDX-License-Identifier: Apache-2.0
import pytest

from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.llm.llm_base import OperationContext
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


def _is_bad_request(exc: Exception) -> bool:
    """
    Be tolerant of where BadRequest comes from by checking common signals:
    - class name "BadRequest"
    - or an attribute 'code' == "BAD_REQUEST"
    """
    if exc.__class__.__name__ == "BadRequest":
        return True
    code = getattr(exc, "code", None)
    return code == "BAD_REQUEST"


async def test_rejects_unknown_roles():
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_msg_role_unknown", tenant="test")

    with pytest.raises(Exception) as ei:
        await adapter.complete(
            messages=[{"role": "alien", "content": "hello from zeta reticuli"}],
            model="mock-model",
            ctx=ctx,
        )
    assert _is_bad_request(ei.value), f"Expected BadRequest, got: {type(ei.value).__name__}"


async def test_requires_nonempty_messages():
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_msg_empty", tenant="test")

    with pytest.raises(Exception) as ei:
        await adapter.complete(messages=[], model="mock-model", ctx=ctx)
    assert _is_bad_request(ei.value), f"Expected BadRequest, got: {type(ei.value).__name__}"


async def test_requires_role_and_content_fields():
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_msg_fields", tenant="test")

    # Missing 'role'
    with pytest.raises(Exception) as ei1:
        await adapter.complete(messages=[{"content": "hi"}], model="mock-model", ctx=ctx)
    assert _is_bad_request(ei1.value), f"Expected BadRequest, got: {type(ei1.value).__name__}"

    # Missing 'content'
    with pytest.raises(Exception) as ei2:
        await adapter.complete(messages=[{"role": "user"}], model="mock-model", ctx=ctx)
    assert _is_bad_request(ei2.value), f"Expected BadRequest, got: {type(ei2.value).__name__}"


async def test_accepts_standard_roles():
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_msg_roles_ok", tenant="test")

    # Minimal, valid message set
    res = await adapter.complete(
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Ping"},
        ],
        model="mock-model",
        ctx=ctx,
    )
    assert isinstance(res.text, str) and res.text
