# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Upsert basics & partial failures.

Spec refs:
  • SPECIFICATION.md §9.3 (upsert)
  • SPECIFICATION.md §9.5 (Vector-Specific Errors)
  • SPECIFICATION.md §12.5 (Partial Failure Contracts)
"""

import pytest
from corpus_sdk.vector.vector_base import (
    Vector,
    VectorID,
    UpsertSpec,
    UpsertResult,
    DimensionMismatch,
    BadRequest,
    NotSupported,
)

pytestmark = pytest.mark.asyncio


async def test_upsert_upsert_returns_result_with_counts(adapter):
    """Verify upsert returns proper result structure with counts."""
    spec = UpsertSpec(
        namespace="default",
        vectors=[Vector(id=VectorID("test-vector-1"), vector=[0.1, 0.2])],
    )
    result = await adapter.upsert(spec)
    
    assert isinstance(result, UpsertResult)
    assert result.upserted_count == 1
    assert result.failed_count == 0
    assert isinstance(result.failures, list)


async def test_upsert_validates_dimensions(adapter):
    """Verify upsert validates vector dimensions."""
    caps = await adapter.capabilities()
    bad_dimension = (caps.max_dimensions or 8) + 1

    bad_vector = Vector(id=VectorID("bad-dimension"), vector=[0.0] * bad_dimension)
    spec = UpsertSpec(namespace="default", vectors=[bad_vector])

    with pytest.raises(DimensionMismatch) as exc_info:
        await adapter.upsert(spec)
    
    err = exc_info.value
    assert err.code == "DIMENSION_MISMATCH"


async def test_upsert_validates_namespace_exists_or_behavior_documented(adapter):
    """
    Spec: adapters MUST either validate unknown namespaces or define clear behavior.
    This test accepts either:
      - BadRequest / NotSupported on unknown namespace, OR
      - A successful UpsertResult with a well-formed shape.
    """
    spec = UpsertSpec(
        namespace="__completely_unknown_namespace_123__",
        vectors=[Vector(id=VectorID("test-vector"), vector=[0.1, 0.2])],
    )

    try:
        result = await adapter.upsert(spec)
    except (BadRequest, NotSupported):
        return  # Validation behavior is acceptable

    # If no exception, result must be well-formed
    assert isinstance(result, UpsertResult)
    assert isinstance(result.upserted_count, int)
    assert isinstance(result.failed_count, int)
    assert isinstance(result.failures, list)


async def test_upsert_requires_non_empty_vectors(adapter):
    """Verify upsert requires non-empty vectors list."""
    spec = UpsertSpec(namespace="default", vectors=[])
    
    with pytest.raises(BadRequest) as exc_info:
        await adapter.upsert(spec)
    
    err = exc_info.value
    assert "vector" in str(err).lower() or "empty" in str(err).lower()


async def test_upsert_partial_failure_reporting(adapter):
    """
    When partial success is implemented, failures MUST be listed with indices.
    If adapter chooses atomic failure via DimensionMismatch, that is also valid.
    """
    caps = await adapter.capabilities()
    dim = caps.max_dimensions or 8

    good_vector = Vector(id=VectorID("good-vector"), vector=[0.1] * dim)
    bad_vector = Vector(id=VectorID("bad-dimension"), vector=[0.1] * (dim + 1))

    try:
        result = await adapter.upsert(UpsertSpec(namespace="default", vectors=[good_vector, bad_vector]))
    except DimensionMismatch:
        # Atomic failure is allowed by spec
        return

    # If partial success is implemented, verify proper reporting
    assert isinstance(result, UpsertResult)
    assert result.upserted_count >= 1
    assert result.failed_count >= 1
    assert len(result.failures) == result.failed_count