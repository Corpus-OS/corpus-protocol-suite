# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Dimension validation & errors.

Spec refs:
  • SPECIFICATION.md §9.5 (Vector-Specific Errors)
  • SPECIFICATION.md §12.4 (Error Mapping)
"""

import pytest

from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter
from adapter_sdk.vector_base import (
    Vector,
    VectorID,
    UpsertSpec,
    QuerySpec,
    DimensionMismatch,
)

pytestmark = pytest.mark.asyncio


async def test_dimension_mismatch_on_upsert():
    a = MockVectorAdapter()
    caps = await a.capabilities()
    bad_dim = (caps.max_dimensions or 8) + 1

    spec = UpsertSpec(
        namespace="default",
        vectors=[Vector(id=VectorID("bad"), vector=[0.0] * bad_dim)],
    )

    with pytest.raises(DimensionMismatch):
        await a.upsert(spec)


async def test_dimension_mismatch_on_query():
    a = MockVectorAdapter()
    caps = await a.capabilities()
    bad_dim = (caps.max_dimensions or 8) + 1

    with pytest.raises(DimensionMismatch):
        await a.query(
            QuerySpec(
                vector=[0.0] * bad_dim,
                top_k=1,
                namespace="default",
            )
        )


async def test_dimension_mismatch_error_attributes():
    a = MockVectorAdapter()
    caps = await a.capabilities()
    bad_dim = (caps.max_dimensions or 8) + 1

    try:
        await a.query(QuerySpec(vector=[0.0] * bad_dim, top_k=1, namespace="default"))
    except DimensionMismatch as e:
        assert e.code == "DIMENSION_MISMATCH"
        assert not getattr(e, "retry_after_ms", None)


async def test_dimension_mismatch_non_retryable():
    a = MockVectorAdapter()
    caps = await a.capabilities()
    bad_dim = (caps.max_dimensions or 8) + 1

    with pytest.raises(DimensionMismatch) as ei:
        await a.query(
            QuerySpec(
                vector=[0.0] * bad_dim,
                top_k=1,
                namespace="default",
            )
        )
    err = ei.value
    assert err.retry_after_ms is None
