# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Delete operations.

Spec refs:
  • SPECIFICATION.md §9.3 (delete)
  • SPECIFICATION.md §12.5 (Partial Failures)
"""

import pytest

from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter
from adapter_sdk.vector_base import (
    Vector,
    VectorID,
    UpsertSpec,
    DeleteSpec,
    DeleteResult,
    BadRequest,
)

pytestmark = pytest.mark.asyncio


async def test_delete_by_ids_returns_counts():
    a = MockVectorAdapter()
    v = Vector(id=VectorID("d1"), vector=[0.1, 0.2])
    await a.upsert(UpsertSpec(namespace="default", vectors=[v]))

    res = await a.delete(DeleteSpec(namespace="default", ids=[VectorID("d1")]))
    assert isinstance(res, DeleteResult)
    assert res.deleted_count >= 1
    assert isinstance(res.failures, list)


async def test_delete_by_filter_returns_counts():
    a = MockVectorAdapter()
    # Rely on mock semantics; at minimum, no crash + valid shape.
    spec = DeleteSpec(namespace="default", ids=[], filter={"foo": "bar"})
    res = await a.delete(spec)
    assert isinstance(res, DeleteResult)
    assert res.deleted_count >= 0


async def test_delete_requires_ids_or_filter():
    a = MockVectorAdapter()
    spec = DeleteSpec(namespace="default", ids=[], filter=None)
    with pytest.raises(BadRequest):
        await a.delete(spec)


async def test_delete_idempotent_for_missing_ids():
    a = MockVectorAdapter()
    spec = DeleteSpec(namespace="default", ids=[VectorID("missing")])
    res1 = await a.delete(spec)
    res2 = await a.delete(spec)
    assert res1.deleted_count >= 0
    assert res2.deleted_count >= 0


async def test_delete_result_structure():
    a = MockVectorAdapter()
    spec = DeleteSpec(namespace="default", ids=[VectorID("x")])
    res = await a.delete(spec)
    assert hasattr(res, "deleted_count")
    assert hasattr(res, "failed_count")
    assert isinstance(res.failures, list)
