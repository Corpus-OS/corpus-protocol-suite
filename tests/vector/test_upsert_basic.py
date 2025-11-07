# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Upsert basics & partial failures.

Spec refs:
  • SPECIFICATION.md §9.3 (upsert)
  • SPECIFICATION.md §9.5 (Vector-Specific Errors)
  • SPECIFICATION.md §12.5 (Partial Failure Contracts)
"""

import pytest

from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter
from adapter_sdk.vector_base import (
    Vector,
    VectorID,
    UpsertSpec,
    UpsertResult,
    DimensionMismatch,
    BadRequest,
)

pytestmark = pytest.mark.asyncio


async def test_upsert_returns_result_with_counts():
    a = MockVectorAdapter()
    spec = UpsertSpec(
        namespace="default",
        vectors=[Vector(id=VectorID("v1"), vector=[0.1, 0.2])],
    )
    res = await a.upsert(spec)
    assert isinstance(res, UpsertResult)
    assert res.upserted_count == 1
    assert res.failed_count == 0
    assert isinstance(res.failures, list)


async def test_upsert_validates_dimensions():
    a = MockVectorAdapter()
    caps = await a.capabilities()
    bad_dim = (caps.max_dimensions or 8) + 1

    bad_vec = Vector(id=VectorID("bad"), vector=[0.0] * bad_dim)
    spec = UpsertSpec(namespace="default", vectors=[bad_vec])

    with pytest.raises(DimensionMismatch):
        await a.upsert(spec)


async def test_upsert_validates_namespace_exists_or_behavior_documented():
    """
    Spec: adapters MUST either validate unknown namespaces or define clear behavior.
    This test accepts either:
      - BadRequest / NotSupported on unknown namespace, OR
      - A successful UpsertResult with a well-formed shape.
    """
    a = MockVectorAdapter()
    spec = UpsertSpec(
        namespace="__no_such_namespace__",
        vectors=[Vector(id=VectorID("v_ns"), vector=[0.1, 0.2])],
    )

    try:
        res = await a.upsert(spec)
    except (BadRequest,):
        return

    assert isinstance(res, UpsertResult)
    assert isinstance(res.upserted_count, int)
    assert isinstance(res.failed_count, int)
    assert isinstance(res.failures, list)


async def test_upsert_requires_non_empty_vectors():
    a = MockVectorAdapter()
    spec = UpsertSpec(namespace="default", vectors=[])
    with pytest.raises(BadRequest):
        await a.upsert(spec)


async def test_upsert_partial_failure_reporting():
    """
    When partial success is implemented, failures MUST be listed with indices.
    If adapter chooses atomic failure via DimensionMismatch, that is also valid.
    """
    a = MockVectorAdapter()
    caps = await a.capabilities()
    dim = caps.max_dimensions or 8

    good = Vector(id=VectorID("ok"), vector=[0.1] * dim)
    bad = Vector(id=VectorID("too_long"), vector=[0.1] * (dim + 1))

    try:
        res = await a.upsert(UpsertSpec(namespace="default", vectors=[good, bad]))
    except DimensionMismatch:
        # Atomic fail is allowed by spec; this still exercises conformance.
        return

    assert isinstance(res, UpsertResult)
    assert res.upserted_count >= 1
    assert res.failed_count >= 1
    assert len(res.failures) == res.failed_count
