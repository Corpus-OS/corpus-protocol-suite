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
    assert "temperature" in msg


@pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0, 1.5, 2.0])
async def test_sampling_params_valid_temperature_accepted(adapter, temperature):
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test")

    res = await adapter.complete(
        messages=[{"role": "user", "content": "test"}],
        temperature=temperature,
        model=caps.supported_models[0],
        ctx=ctx,
    )
    assert isinstance(res.text, str)


@pytest.mark.parametrize("top_p", [0.0, -0.1, 1.1, 2.0, -1.0])
async def test_sampling_params_invalid_top_p_rejected(adapter, top_p):
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
    assert ("top_p" in msg) or ("top p" in msg)


@pytest.mark.parametrize("top_p", [0.1, 0.5, 0.9, 1.0])
async def test_sampling_params_valid_top_p_accepted(adapter, top_p):
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test")

    res = await adapter.complete(
        messages=[{"role": "user", "content": "test"}],
        top_p=top_p,
        model=caps.supported_models[0],
        ctx=ctx,
    )
    assert isinstance(res.text, str)


@pytest.mark.parametrize("frequency_penalty", [-2.1, 2.1, -3.0, 5.0])
async def test_sampling_params_invalid_frequency_penalty_rejected(adapter, frequency_penalty):
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
    assert "frequency" in msg


@pytest.mark.parametrize("frequency_penalty", [-2.0, -1.0, 0.0, 1.0, 2.0])
async def test_sampling_params_valid_frequency_penalty_accepted(adapter, frequency_penalty):
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test")

    res = await adapter.complete(
        messages=[{"role": "user", "content": "test"}],
        frequency_penalty=frequency_penalty,
        model=caps.supported_models[0],
        ctx=ctx,
    )
    assert isinstance(res.text, str)


@pytest.mark.parametrize("presence_penalty", [-2.1, 2.1, -3.0, 5.0])
async def test_sampling_params_invalid_presence_penalty_rejected(adapter, presence_penalty):
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
    assert "presence" in msg


@pytest.mark.parametrize("presence_penalty", [-2.0, -1.0, 0.0, 1.0, 2.0])
async def test_sampling_params_valid_presence_penalty_accepted(adapter, presence_penalty):
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test")

    res = await adapter.complete(
        messages=[{"role": "user", "content": "test"}],
        presence_penalty=presence_penalty,
        model=caps.supported_models[0],
        ctx=ctx,
    )
    assert isinstance(res.text, str)


async def test_sampling_params_multiple_invalid_params_error_message(adapter):
    caps = await adapter.capabilities()
    ctx = OperationContext(tenant="test")

    with pytest.raises(BadRequest) as exc_info:
        await adapter.complete(
            messages=[{"role": "user", "content": "test"}],
            temperature=999.0,
            top_p=2.0,
            model=caps.supported_models[0],
            ctx=ctx,
        )

    msg = str(getattr(exc_info.value, "message", exc_info.value)).lower()
    assert msg.strip()
