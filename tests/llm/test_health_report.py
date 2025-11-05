# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Health report shape.
Covers:
  • Health returns dict with required keys (ok, server, version)
  • 'ok' is boolean; server/version are non-empty strings
  • Shape is consistent regardless of health status
"""
import pytest
from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.llm.llm_base import OperationContext
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_health_has_required_fields():
    """
    Health endpoint MUST return dict with 'ok', 'server', 'version'.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_health", tenant="test")
    
    h = await adapter.health(ctx=ctx)
    
    # Must be a dict
    assert isinstance(h, dict), "health() must return a dict"
    
    # Required field: ok (boolean)
    assert "ok" in h, "health() must include 'ok' field"
    assert isinstance(h["ok"], bool), "'ok' must be a boolean"
    
    # Required field: server (non-empty string)
    assert "server" in h, "health() must include 'server' field"
    assert isinstance(h["server"], str), "'server' must be a string"
    assert len(h["server"]) > 0, "'server' must be non-empty"
    
    # Required field: version (non-empty string)
    assert "version" in h, "health() must include 'version' field"
    assert isinstance(h["version"], str), "'version' must be a string"
    assert len(h["version"]) > 0, "'version' must be non-empty"


async def test_health_shape_consistent_when_degraded():
    """
    Health shape MUST be consistent even when status is degraded/unhealthy.
    """
    adapter = MockLLMAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, tenant="test")
    
    # Call health multiple times to potentially catch degraded state
    # MockLLMAdapter returns unhealthy ~20% of the time
    seen_degraded = False
    
    for attempt in range(20):
        h = await adapter.health(ctx=ctx)
        
        # Shape must always be valid
        assert isinstance(h, dict)
        assert "ok" in h and isinstance(h["ok"], bool)
        assert "server" in h and isinstance(h["server"], str)
        assert "version" in h and isinstance(h["version"], str)
        
        if not h["ok"]:
            seen_degraded = True
            # Degraded health may include additional status info
            if "status" in h:
                assert isinstance(h["status"], str)
    
    # Note: Not asserting we saw degraded state since it's probabilistic
    # The important thing is shape validation works for both states
