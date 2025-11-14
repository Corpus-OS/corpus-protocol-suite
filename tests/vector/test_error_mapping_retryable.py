# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Error taxonomy & retry hints.

Spec refs:
  • SPECIFICATION.md §6.3 (Error Taxonomy)
  • SPECIFICATION.md §9.5 (Vector-Specific Errors)
  • SPECIFICATION.md §12.1, §12.4 (Retry Semantics, Mapping)
"""

import pytest
from corpus_sdk.vector.vector_base import (
    VectorAdapterError,
    ResourceExhausted,
    Unavailable,
    IndexNotReady,
    BadRequest,
    DimensionMismatch,
    QuerySpec,
)

pytestmark = pytest.mark.asyncio


async def test_error_handling_retryable_errors_with_hints():
    """Verify retryable errors include proper retry hints."""
    rate_limit_error = ResourceExhausted("rate limit exceeded", retry_after_ms=1000)
    assert rate_limit_error.code == "RESOURCE_EXHAUSTED"
    assert rate_limit_error.retry_after_ms == 1000

    unavailable_error = Unavailable("backend unavailable", retry_after_ms=500)
    assert unavailable_error.code == "UNAVAILABLE"
    assert unavailable_error.retry_after_ms == 500


async def test_error_handling_index_not_ready_retryable():
    """Verify IndexNotReady errors are retryable with hints."""
    index_error = IndexNotReady("index warming up", retry_after_ms=200)
    assert index_error.code == "INDEX_NOT_READY"
    assert index_error.retry_after_ms == 200


async def test_error_handling_dimension_mismatch_non_retryable_flag():
    """Verify DimensionMismatch errors are non-retryable."""
    dimension_error = DimensionMismatch("vector dimension mismatch")
    assert dimension_error.code == "DIMENSION_MISMATCH"
    assert dimension_error.retry_after_ms is None


async def test_error_handling_bad_request_on_invalid_top_k(adapter):
    """Verify BadRequest is raised for invalid top_k values."""
    with pytest.raises(BadRequest) as exc_info:
        await adapter.query(QuerySpec(vector=[0.1], top_k=0, namespace="default"))
    
    err = exc_info.value
    assert err.code == "BAD_REQUEST"
    assert "top_k" in str(err).lower() or "invalid" in str(err).lower()


async def test_error_handling_error_has_siem_safe_details():
    """Verify error details are SIEM-safe (JSON serializable)."""
    error = ResourceExhausted("rate limited", details={"scope": "global", "limit": 1000})
    error_dict = error.asdict()
    
    assert "details" in error_dict
    assert isinstance(error_dict["details"], dict)
    # Details should be JSON-serializable (no complex objects)
    import json
    json.dumps(error_dict)  # Should not raise