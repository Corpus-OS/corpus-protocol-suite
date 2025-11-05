# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Error mapping & retryability.
Covers:
  • Retryable errors (Unavailable, ResourceExhausted) include retry hints
  • retry_after_ms is non-negative integer when present
  • Error messages are informative
  • Error classification aligns with SPECIFICATION.md §12.4
"""
import random
import pytest
from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.llm.llm_base import OperationContext
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.errors import (
    ResourceExhausted, 
    Unavailable,
    BadRequest
)

pytestmark = pytest.mark.asyncio


async def test_retryable_errors_with_hints():
    """
    SPECIFICATION.md §12.1, §12.4 — Retryable Error Classification
    
    Retryable errors (Unavailable, ResourceExhausted) SHOULD include
    retry_after_ms hints for backoff guidance.
    """
    # Deterministic path: always trigger failure branch
    random.seed(1337)
    adapter = MockLLMAdapter(failure_rate=1.0)  # force failure path
    ctx = make_ctx(OperationContext, request_id="t_err_retryable", tenant="test")
    
    with pytest.raises((Unavailable, ResourceExhausted)) as excinfo:
        await adapter.complete(
            messages=[{"role": "user", "content": "overload"}], 
            ctx=ctx
        )
    
    err = excinfo.value
    
    # Verify retryable class per taxonomy
    assert isinstance(err, (Unavailable, ResourceExhausted)), \
        f"Expected retryable error, got {type(err).__name__}"
    
    # Error should have informative message
    assert err.message or str(err), \
        "Error should have a descriptive message"
    
    # retry_after_ms should be valid if present
    ra = getattr(err, "retry_after_ms", None)
    assert (ra is None) or (isinstance(ra, int) and ra >= 0), \
        f"retry_after_ms should be non-negative int or None, got {ra}"
    
    # If retry_after_ms is present, it should be reasonable
    if ra is not None:
        assert ra < 300_000, \
            f"retry_after_ms ({ra}ms) seems unreasonably long (>5min)"


async def test_retryable_errors_have_correct_attributes():
    """Retryable errors should have expected error taxonomy attributes."""
    random.seed(1337)
    adapter = MockLLMAdapter(failure_rate=1.0)
    ctx = make_ctx(OperationContext, tenant="test")
    
    with pytest.raises((Unavailable, ResourceExhausted)) as excinfo:
        await adapter.complete(
            messages=[{"role": "user", "content": "overload"}],
            ctx=ctx
        )
    
    err = excinfo.value
    
    # Should have code attribute
    assert hasattr(err, 'code'), "Error should have 'code' attribute"
    assert isinstance(err.code, str), "Error code should be string"
    
    # Should have retryable attribute
    assert hasattr(err, 'retryable'), "Error should have 'retryable' attribute"
    assert err.retryable is True, "Unavailable/ResourceExhausted should be retryable"
