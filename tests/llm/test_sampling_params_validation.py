# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Sampling parameter validation.
Covers:
  • Temperature must be in [0.0, 2.0]
  • top_p must be in (0.0, 1.0]
  • frequency_penalty must be in [-2.0, 2.0]
  • presence_penalty must be in [-2.0, 2.0]
  • Invalid ranges raise BadRequest with informative messages
"""
import pytest
from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.llm.llm_base import OperationContext
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.errors import BadRequest

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("temperature", [-0.1, 2.1, -1.0, 999.0])
async def test_invalid_temperature_rejected(temperature):
    """
    SPECIFICATION.md §8.3 — Temperature Validation
    
    Temperature MUST be in range [0.0, 2.0]. Values outside this range
    MUST raise BadRequest.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_temp_invalid", tenant="test")
    
    with pytest.raises(BadRequest) as exc_info:
        await adapter.complete(
            messages=[{"role": "user", "content": "test"}],
            temperature=temperature,
            model="mock-model",
            ctx=ctx
        )
    
    # Error message should mention temperature
    msg = str(getattr(exc_info.value, "message", exc_info.value)).lower()
    assert "temperature" in msg, \
        f"Error message should mention 'temperature', got: {msg}"


@pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0, 1.5, 2.0])
async def test_valid_temperature_accepted(temperature):
    """
    Temperature values within [0.0, 2.0] MUST be accepted.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, tenant="test")
    
    # Should not raise
    res = await adapter.complete(
        messages=[{"role": "user", "content": "test"}],
        temperature=temperature,
        model="mock-model",
        ctx=ctx
    )
    
    assert isinstance(res.text, str) and res.text.strip()


@pytest.mark.parametrize("top_p", [0.0, -0.1, 1.1, 2.0, -1.0])
async def test_invalid_top_p_rejected(top_p):
    """
    SPECIFICATION.md §8.3 — top_p Validation
    
    top_p MUST be in range (0.0, 1.0]. Values outside this range
    MUST raise BadRequest.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_top_p_invalid", tenant="test")
    
    with pytest.raises(BadRequest) as exc_info:
        await adapter.complete(
            messages=[{"role": "user", "content": "test"}],
            top_p=top_p,
            model="mock-model",
            ctx=ctx
        )
    
    # Error message should mention top_p
    msg = str(getattr(exc_info.value, "message", exc_info.value)).lower()
    assert "top_p" in msg or "top p" in msg, \
        f"Error message should mention 'top_p', got: {msg}"


@pytest.mark.parametrize("top_p", [0.1, 0.5, 0.9, 1.0])
async def test_valid_top_p_accepted(top_p):
    """
    top_p values within (0.0, 1.0] MUST be accepted.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, tenant="test")
    
    # Should not raise
    res = await adapter.complete(
        messages=[{"role": "user", "content": "test"}],
        top_p=top_p,
        model="mock-model",
        ctx=ctx
    )
    
    assert isinstance(res.text, str) and res.text.strip()


@pytest.mark.parametrize("frequency_penalty", [-2.1, 2.1, -3.0, 5.0])
async def test_invalid_frequency_penalty_rejected(frequency_penalty):
    """
    SPECIFICATION.md §8.3 — frequency_penalty Validation
    
    frequency_penalty MUST be in range [-2.0, 2.0]. Values outside this range
    MUST raise BadRequest.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_freq_pen_invalid", tenant="test")
    
    with pytest.raises(BadRequest) as exc_info:
        await adapter.complete(
            messages=[{"role": "user", "content": "test"}],
            frequency_penalty=frequency_penalty,
            model="mock-model",
            ctx=ctx
        )
    
    # Error message should mention frequency_penalty
    msg = str(getattr(exc_info.value, "message", exc_info.value)).lower()
    assert "frequency" in msg, \
        f"Error message should mention 'frequency', got: {msg}"


@pytest.mark.parametrize("frequency_penalty", [-2.0, -1.0, 0.0, 1.0, 2.0])
async def test_valid_frequency_penalty_accepted(frequency_penalty):
    """
    frequency_penalty values within [-2.0, 2.0] MUST be accepted.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, tenant="test")
    
    # Should not raise
    res = await adapter.complete(
        messages=[{"role": "user", "content": "test"}],
        frequency_penalty=frequency_penalty,
        model="mock-model",
        ctx=ctx
    )
    
    assert isinstance(res.text, str) and res.text.strip()


@pytest.mark.parametrize("presence_penalty", [-2.1, 2.1, -3.0, 5.0])
async def test_invalid_presence_penalty_rejected(presence_penalty):
    """
    SPECIFICATION.md §8.3 — presence_penalty Validation
    
    presence_penalty MUST be in range [-2.0, 2.0]. Values outside this range
    MUST raise BadRequest.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_pres_pen_invalid", tenant="test")
    
    with pytest.raises(BadRequest) as exc_info:
        await adapter.complete(
            messages=[{"role": "user", "content": "test"}],
            presence_penalty=presence_penalty,
            model="mock-model",
            ctx=ctx
        )
    
    # Error message should mention presence_penalty
    msg = str(getattr(exc_info.value, "message", exc_info.value)).lower()
    assert "presence" in msg, \
        f"Error message should mention 'presence', got: {msg}"


@pytest.mark.parametrize("presence_penalty", [-2.0, -1.0, 0.0, 1.0, 2.0])
async def test_valid_presence_penalty_accepted(presence_penalty):
    """
    presence_penalty values within [-2.0, 2.0] MUST be accepted.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, tenant="test")
    
    # Should not raise
    res = await adapter.complete(
        messages=[{"role": "user", "content": "test"}],
        presence_penalty=presence_penalty,
        model="mock-model",
        ctx=ctx
    )
    
    assert isinstance(res.text, str) and res.text.strip()


async def test_multiple_invalid_params_error_message():
    """
    When multiple parameters are invalid, error should be informative.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, tenant="test")
    
    with pytest.raises(BadRequest) as exc_info:
        await adapter.complete(
            messages=[{"role": "user", "content": "test"}],
            temperature=999.0,  # Invalid
            top_p=2.0,          # Invalid
            model="mock-model",
            ctx=ctx
        )
    
    # Should catch first invalid parameter
    err = exc_info.value
    assert hasattr(err, 'message') or str(err)
