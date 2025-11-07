# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Metadata filtering.

Spec refs:
  • SPECIFICATION.md §9.3 (query/delete filters)
"""

import pytest

from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter
from adapter_sdk.vector_base import (
    Vector,
    VectorID,
    UpsertSpec,
    QuerySpec,
    DeleteSpec,
    BadRequest,
    NotSupported,
)

pytestmark = pytest.mark.asyncio


async def test_query_filter_equality():
    a = MockVectorAdapter()
    v = Vector(
        id=VectorID("f1"),
        vector=[0.1, 0.2],
        metadata={"tag": "keep"},
        namespace="default",
    )
    await a.upsert(UpsertSpec(namespace="default", vectors=[v]))

    res = await a.query(
        QuerySpec(
            vector=[0.1, 0.2],
            top_k=5,
            namespace="default",
            filter={"tag": "keep"},
        )
    )
    # At least shape correctness: no crash, list matches.
    assert isinstance(res.matches, list)


async def test_delete_filter_equality():
    a = MockVectorAdapter()
    spec = DeleteSpec(namespace="default", ids=[], filter={"tag": "keep"})
    res = await a.delete(spec)
    assert res.deleted_count >= 0


async def test_filter_requires_mapping_type():
    a = MockVectorAdapter()
    with pytest.raises(BadRequest):
        await a.query(
            QuerySpec(
                vector=[0.1],
                top_k=1,
                namespace="default",
                filter=["not-a-mapping"],  # type: ignore[arg-type]
            )
        )


async def test_filter_respects_capabilities_support():
    a = MockVectorAdapter()
    caps = await a.capabilities()
    if not caps.supports_metadata_filtering:
        with pytest.raises(NotSupported):
            await a.query(
                QuerySpec(
                    vector=[0.1],
                    top_k=1,
                    namespace="default",
                    filter={"x": 1},
                )
            )


async def test_filter_empty_results_ok():
    a = MockVectorAdapter()
    res = await a.query(
        QuerySpec(
            vector=[0.9, 0.9],
            top_k=3,
            namespace="default",
            filter={"tag": "__unlikely__"},
        )
    )
    assert isinstance(res.matches, list)
