# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Batch size limits & partial failures.

Spec refs:
  • SPECIFICATION.md §9.3 (Batch operations)
  • SPECIFICATION.md §12.5 (Partial Failure Contracts)
"""

import pytest
from corpus_sdk.vector.vector_base import (
    Vector,
    VectorID,
    UpsertSpec,
    DeleteSpec,
    BadRequest,
    UpsertResult,
    DimensionMismatch,
    QuerySpec,
)

pytestmark = pytest.mark.asyncio


async def test_batch_limits_upsert_respects_max_batch_size(adapter):
    """Verify adapter enforces maximum batch size limits."""
    caps = await adapter.capabilities()
    if caps.max_batch_size is None:
        pytest.skip("Adapter does not define max_batch_size")

    too_many = caps.max_batch_size + 1
    spec = UpsertSpec(
        namespace="default",
        vectors=[Vector(id=VectorID(str(i)), vector=[0.1]) for i in range(too_many)],
    )
    with pytest.raises(BadRequest):
        await adapter.upsert(spec)


async def test_batch_limits_batch_size_exceeded_includes_suggestion(adapter):
    """Verify batch size errors include helpful reduction suggestions when provided."""
    caps = await adapter.capabilities()
    if caps.max_batch_size is None:
        pytest.skip("Adapter does not define max_batch_size")

    too_many = caps.max_batch_size + 10
    spec = UpsertSpec(
        namespace="default",
        vectors=[Vector(id=VectorID(str(i)), vector=[0.1]) for i in range(too_many)],
    )

    with pytest.raises(BadRequest) as exc_info:
        await adapter.upsert(spec)
    err = exc_info.value
    # suggestion is optional, but if present must be int-ish
    if getattr(err, "suggested_batch_reduction", None) is not None:
        assert isinstance(err.suggested_batch_reduction, int)


async def test_batch_limits_partial_failure_reporting_shape(adapter):
    """
    When adapter chooses partial behavior (some fail, some succeed),
    failures MUST be listed. If adapter is atomic, it's allowed to raise instead.
    """
    caps = await adapter.capabilities()
    dim = caps.max_dimensions or 8

    good = Vector(id=VectorID("ok"), vector=[0.1] * dim)
    bad = Vector(id=VectorID("bad"), vector=[0.1] * (dim + 1))
    try:
        res = await adapter.upsert(UpsertSpec(namespace="default", vectors=[good, bad]))
    except DimensionMismatch:
        return  # Atomic behavior is acceptable

    assert isinstance(res, UpsertResult)
    assert res.failed_count >= 1
    assert len(res.failures) == res.failed_count


async def test_batch_limits_batch_operations_atomic_per_vector(adapter):
    """
    Confirms that failure for one item does not corrupt others when partial
    semantics are implemented. If adapter chooses atomic failure, this test
    treats that as acceptable via the same DimensionMismatch escape.
    """
    caps = await adapter.capabilities()
    dim = caps.max_dimensions or 8

    v_good = Vector(id=VectorID("ok2"), vector=[0.2] * dim)
    v_bad = Vector(id=VectorID("bad2"), vector=[0.2] * (dim + 1))

    try:
        res = await adapter.upsert(UpsertSpec(namespace="default", vectors=[v_good, v_bad]))
    except DimensionMismatch:
        return  # Atomic behavior is acceptable

    assert res.upserted_count >= 1

    # Good vector should be queryable
    qr = await adapter.query(QuerySpec(vector=[0.2] * dim, top_k=5, namespace="default"))
    ids = [match.vector.id for match in qr.matches]
    assert VectorID("ok2") in ids