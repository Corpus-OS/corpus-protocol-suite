# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Dimension validation & errors.

Spec refs:
  • SPECIFICATION.md §9.5 (Vector-Specific Errors)
  • SPECIFICATION.md §12.4 (Error Mapping)
"""

import pytest
from corpus_sdk.vector.vector_base import (
    Vector,
    VectorID,
    UpsertSpec,
    QuerySpec,
    DimensionMismatch,
)

pytestmark = pytest.mark.asyncio


async def test_dimension_validation_dimension_mismatch_on_upsert(adapter):
    """Verify DimensionMismatch is raised for invalid vector dimensions on upsert."""
    caps = await adapter.capabilities()
    bad_dimension = (caps.max_dimensions or 8) + 1

    spec = UpsertSpec(
        namespace="default",
        vectors=[Vector(id=VectorID("bad-dimension"), vector=[0.0] * bad_dimension)],
    )

    with pytest.raises(DimensionMismatch) as exc_info:
        await adapter.upsert(spec)
    
    err = exc_info.value
    assert err.code == "DIMENSION_MISMATCH"


async def test_dimension_validation_dimension_mismatch_on_query(adapter):
    """Verify DimensionMismatch is raised for invalid vector dimensions on query."""
    caps = await adapter.capabilities()
    bad_dimension = (caps.max_dimensions or 8) + 1

    with pytest.raises(DimensionMismatch) as exc_info:
        await adapter.query(
            QuerySpec(
                vector=[0.0] * bad_dimension,
                top_k=1,
                namespace="default",
            )
        )
    
    err = exc_info.value
    assert err.code == "DIMENSION_MISMATCH"


async def test_dimension_validation_dimension_mismatch_error_attributes(adapter):
    """Verify DimensionMismatch error has correct attributes and no retry hint."""
    caps = await adapter.capabilities()
    bad_dimension = (caps.max_dimensions or 8) + 1

    with pytest.raises(DimensionMismatch) as exc_info:
        await adapter.query(QuerySpec(vector=[0.0] * bad_dimension, top_k=1, namespace="default"))
    
    err = exc_info.value
    assert err.code == "DIMENSION_MISMATCH"
    assert getattr(err, "retry_after_ms", None) is None
    assert err.message and isinstance(err.message, str)


async def test_dimension_validation_dimension_mismatch_non_retryable(adapter):
    """Verify DimensionMismatch errors are marked as non-retryable."""
    caps = await adapter.capabilities()
    bad_dimension = (caps.max_dimensions or 8) + 1

    with pytest.raises(DimensionMismatch) as exc_info:
        await adapter.query(
            QuerySpec(
                vector=[0.0] * bad_dimension,
                top_k=1,
                namespace="default",
            )
        )
    
    err = exc_info.value
    assert err.retry_after_ms is None
    # Dimension mismatches are client errors and should not be retried