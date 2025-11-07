# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Batch size limits & partial failures.

Spec refs:
  • SPECIFICATION.md §9.3 (Batch operations)
  • SPECIFICATION.md §12.5 (Partial Failure Contracts)
"""

import pytest

from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter
from adapter_sdk.vector_base import (
    Vector,
    VectorID,
    UpsertSpec,
    DeleteSpec,
    BadRequest,
)

pytestmark = pytest.mark.asyncio


async def test_upsert_respects_max_batch_size():
    a = MockVectorAdapter()
    caps = await a.capabilities()
    if caps.max_batch_size is None:
        pytest.skip("Adapter does not define max_batch_size")

    too_many = caps.max_batch_size + 1
    spec = UpsertSpec(
        namespace="default",
        vectors=[Vector(id=VectorID(str(i)), vector=[0.1]) for i in range(too_many)],
    )
    with pytest.raises(BadRequest):
        await a.upsert(spec)


async def test_batch_size_exceeded_includes_suggestion():
    a = MockVectorAdapter()
    caps = await a.capabilities()
    if caps.max_batch_size is None:
        pytest.skip("Adapter does not define max_batch_size")

    too_many = caps.max_batch_size + 10
    spec = UpsertSpec(
        namespace="default",
        vectors=[Vector(id=VectorID(str(i)), vector=[0.1]) for i in range(too_many)],
    )

    with pytest.raises(BadRequest) as ei:
        await a.upsert(spec)
    err = ei.value
    # suggestion is optional, but if present must be int-ish
    if getattr(err, "suggested_batch_reduction", None) is not None:
        assert isinstance(err.suggested_batch_reduction, int)


async def test_partial_failure_reporting_shape():
    """
    When adapter chooses partial behavior (some fail, some succeed),
    failures MUST be listed. If adapter is atomic, it's allowed to raise instead.
    """
    a = MockVectorAdapter()
    caps = await a.capabilities()
    dim = caps.max_dimensions or 8

    from adapter_sdk.vector_base import UpsertResult, DimensionMismatch

    good = Vector(id=VectorID("ok"), vector=[0.1] * dim)
    bad = Vector(id=VectorID("bad"), vector=[0.1] * (dim + 1))
    try:
        res = await a.upsert(UpsertSpec(namespace="default", vectors=[good, bad]))
    except DimensionMismatch:
        return

    assert isinstance(res, UpsertResult)
    assert res.failed_count >= 1
    assert len(res.failures) == res.failed_count


async def test_batch_operations_atomic_per_vector():
    """
    Confirms that failure for one item does not corrupt others when partial
    semantics are implemented. If adapter chooses atomic failure, this test
    treats that as acceptable via the same DimensionMismatch escape.
    """
    a = MockVectorAdapter()
    caps = await a.capabilities()
    dim = caps.max_dimensions or 8

    from adapter_sdk.vector_base import DimensionMismatch, QuerySpec

    v_good = Vector(id=VectorID("ok2"), vector=[0.2] * dim)
    v_bad = Vector(id=VectorID("bad2"), vector=[0.2] * (dim + 1))

    try:
        res = await a.upsert(UpsertSpec(namespace="default", vectors=[v_good, v_bad]))
    except DimensionMismatch:
        return

    assert res.upserted_count >= 1

    # Good vector should be queryable
    qr = await a.query(QuerySpec(vector=[0.2] * dim, top_k=5, namespace="default"))
    ids = [m.vector.id for m in qr.matches]
    assert VectorID("ok2") in ids
