# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Health report shape.

Asserts:
  • Health returns a dict with required keys
  • 'ok' is a boolean; server/version exist
"""

import pytest

from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.llm.llm_base import OperationContext
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_health_has_required_fields():
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_health", tenant="test")

    h = await adapter.health(ctx=ctx)
    # Health may be degraded or healthy; we only assert the shape
    assert isinstance(h, dict)
    assert "ok" in h and isinstance(h["ok"], bool)
    assert "server" in h and isinstance(h["server"], str)
    assert "version" in h and isinstance(h["version"], str)
