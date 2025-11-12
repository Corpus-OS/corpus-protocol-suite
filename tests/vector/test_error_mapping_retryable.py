# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Error taxonomy & retry hints.

Spec refs:
  • SPECIFICATION.md §6.3 (Error Taxonomy)
  • SPECIFICATION.md §9.5 (Vector-Specific Errors)
  • SPECIFICATION.md §12.1, §12.4 (Retry Semantics, Mapping)
"""

import pytest

from adapter_sdk.vector_base import (
    VectorAdapterError,
    ResourceExhausted,
    Unavailable,
    IndexNotReady,
    BadRequest,
)

pytestmark = pytest.mark.asyncio


def test_retryable_errors_with_hints():
    e = ResourceExhausted("rate limit", retry_after_ms=1000)
    assert e.code == "RESOURCE_EXHAUSTED"
    assert e.retry_after_ms == 1000

    u = Unavailable("backend down", retry_after_ms=500)
    assert u.code == "UNAVAILABLE"
    assert u.retry_after_ms == 500


def test_index_not_ready_retryable():
    e = IndexNotReady("index warmup", retry_after_ms=200)
    assert e.code == "INDEX_NOT_READY"
    assert e.retry_after_ms == 200


def test_dimension_mismatch_non_retryable_flag():
    from adapter_sdk.vector_base import DimensionMismatch
    e = DimensionMismatch("bad dim")
    assert e.code == "DIMENSION_MISMATCH"
    assert e.retry_after_ms is None


def test_bad_request_on_invalid_top_k():
    from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter
    from adapter_sdk.vector_base import QuerySpec

    a = MockVectorAdapter()
    with pytest.raises(BadRequest):
        await a.query(QuerySpec(vector=[0.1], top_k=0, namespace="default"))  # type: ignore[misc]


def test_error_has_siem_safe_details():
    e = ResourceExhausted("rl", details={"scary": object()})
    d = e.asdict()
    assert "details" in d
    assert isinstance(d["details"], dict)
