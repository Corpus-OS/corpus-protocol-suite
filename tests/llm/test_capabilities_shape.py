# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Capabilities shape.

Asserts:
  • Returns LLMCapabilities instance
  • Has core fields populated
  • Has at least one supported model
"""

import pytest

from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.llm.llm_base import LLMCapabilities

pytestmark = pytest.mark.asyncio


async def test_capabilities_shape_and_required_fields():
    adapter = MockLLMAdapter(failure_rate=0.0)
    caps = await adapter.capabilities()

    assert isinstance(caps, LLMCapabilities)
    assert caps.server == "mock"
    assert caps.model_family == "mock"
    assert caps.max_context_length > 0
    assert caps.supports_streaming is True
    assert isinstance(caps.supported_models, tuple)
    assert len(caps.supported_models) >= 1
