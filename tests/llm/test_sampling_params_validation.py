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
from corpus_sdk.llm.llm_base import OperationContext, BadRequest

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("temperature", [-0.1, 2.1, -1.0, 999.0])
async def test_sampling_params_invalid_temperature_rejected(adapter, temperature):
    """
    SPECIFICATION.md §8.3 — Temperature Validation

    Temperature MUST be in range [0.0, 2.0]. Values outside this range
    MUST raise BadRequest.
    """
    caps = await adapter.capabilities()
    ctx = OperationContext(request_id="t_temp_invalid", tenant="test")

    with pytest.raises(BadRequest) as exc_info:
        await adapter.complete(
            messages=[{"role": "user", "content": "test"}],
            temperature=temperature,
            model=caps.supported_models[0],
            ctx=ctx,
        )

    msg = str(getattr(exc_info.value, "message", exc_info.value)).lower()
    assert "temperature" in msg, \
        f"Error message should mention 'temperature', got: {msg}"


@pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0, 1.5, 2.0])
async def test_sampling_params_valid_temperature_accepted(adapter, temperature):
    """
    Temperature values within [0.0, 2.0] MUST be accepted.
    """
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test")

    res = await adapter.complete(
        messages=[{"role": "user", "content": "test"}],
        temperature=temperature,
        model=caps.supported_models[0],
        ctx=ctx,
    )

    assert isinstance(res.text, str) and res.text.strip()


@pytest.mark.parametrize("top_p", [0.0, -0.1, 1.1, 2.0, -1.0])
async def test_sampling_params_invalid_top_p_rejected(adapter, top_p):
    """
    SPECIFICATION.md §8.3 — top_p Validation

    top_p MUST be in range (0.0, 1.0]. Values outside this range
    MUST raise BadRequest.
    """
    caps = await adapter.capabilities()
    ctx = OperationContext(request_id="t_top_p_invalid", tenant="test")

    with pytest.raises(BadRequest) as exc_info:
        await adapter.complete(
            messages=[{"role": "user", "content": "test"}],
            top_p=top_p,
            model=caps.supported_models[0],
            ctx=ctx,
        )

    msg = str(getattr(exc_info.value, "message", exc_info.value)).lower()
    assert "top_p" in msg or "top p" in msg, \
        f"Error message should mention 'top_p', got: {msg}"


@pytest.mark.parametrize("top_p", [0.1, 0.5, 0.9, 1.0])
async def test_sampling_params_valid_top_p_accepted(adapter, top_p):
    """
    top_p values within (0.0, 1.0] MUST be accepted.
    """
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test")

    res = await adapter.complete(
        messages=[{"role": "user", "content": "test"}],
        top_p=top_p,
        model=caps.supported_models[0],
        ctx=ctx,
    )

    assert isinstance(res.text, str) and res.text.strip()


@pytest.mark.parametrize("frequency_penalty", [-2.1, 2.1, -3.0, 5.0])
async def test_sampling_params_invalid_frequency_penalty_rejected(adapter, frequency_penalty):
    """
    SPECIFICATION.md §8.3 — frequency_penalty Validation

    frequency_penalty MUST be in range [-2.0, 2.0]. Values outside this range
    MUST raise BadRequest.
    """
    caps = await adapter.capabilities()
    ctx = OperationContext(request_id="t_freq_pen_invalid", tenant="test")

    with pytest.raises(BadRequest) as exc_info:
        await adapter.complete(
            messages=[{"role": "user", "content": "test"}],
            frequency_penalty=frequency_penalty,
            model=caps.supported_models[0],
            ctx=ctx,
        )

    msg = str(getattr(exc_info.value, "message", exc_info.value)).lower()
    assert "frequency" in msg, \
        f"Error message should mention 'frequency', got: {msg}"


@pytest.mark.parametrize("frequency_penalty", [-2.0, -1.0, 0.0, 1.0, 2.0])
async def test_sampling_params_valid_frequency_penalty_accepted(adapter, frequency_penalty):
    """
    frequency_penalty values within [-2.0, 2.0] MUST be accepted.
    """
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test")

    res = await adapter.complete(
        messages=[{"role": "user", "content": "test"}],
        frequency_penalty=frequency_penalty,
        model=caps.supported_models[0],
        ctx=ctx,
    )

    assert isinstance(res.text, str) and res.text.strip()


@pytest.mark.parametrize("presence_penalty", [-2.1, 2.1, -3.0, 5.0])
async def test_sampling_params_invalid_presence_penalty_rejected(adapter, presence_penalty):
    """
    SPECIFICATION.md §8.3 — presence_penalty Validation

    presence_penalty MUST be in range [-2.0, 2.0]. Values outside this range
    MUST raise BadRequest.
    """
    caps = await adapter.capabilities()
    ctx = OperationContext(request_id="t_pres_pen_invalid", tenant="test")

    with pytest.raises(BadRequest) as exc_info:
        await adapter.complete(
            messages=[{"role": "user", "content": "test"}],
            presence_penalty=presence_penalty,
            model=caps.supported_models[0],
            ctx=ctx,
        )

    msg = str(getattr(exc_info.value, "message", exc_info.value)).lower()
    assert "presence" in msg, \
        f"Error message should mention 'presence', got: {msg}"


@pytest.mark.parametrize("presence_penalty", [-2.0, -1.0, 0.0, 1.0, 2.0])
async def test_sampling_params_valid_presence_penalty_accepted(adapter, presence_penalty):
    """
    presence_penalty values within [-2.0, 2.0] MUST be accepted.
    """
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test")

    res = await adapter.complete(
        messages=[{"role": "user", "content": "test"}],
        presence_penalty=presence_penalty,
        model=caps.supported_models[0],
        ctx=ctx,
    )

    assert isinstance(res.text, str) and res.text.strip()


async def test_sampling_params_multiple_invalid_params_error_message(adapter):
    """
    When multiple parameters are invalid, error should be informative.
    """
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test")

    with pytest.raises(BadRequest) as exc_info:
        await adapter.complete(
            messages=[{"role": "user", "content": "test"}],
            temperature=999.0,  # Invalid
            top_p=2.0,          # Invalid
            model=caps.supported_models[0],
            ctx=ctx,
        )

    err = exc_info.value
    msg = str(getattr(err, "message", err)).lower()
    assert msg, "Error message should be non-empty for multiple invalid params"