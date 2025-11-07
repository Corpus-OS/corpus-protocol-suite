# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Query semantics.

Spec refs:
  • SPECIFICATION.md §9.3 (query)
  • SPECIFICATION.md §9.2 (Data Types)
"""

import pytest

from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter
from adapter_sdk.vector_base import (
    QuerySpec,
    VectorMatch,
    QueryResult,
    DimensionMismatch,
    BadRequest,
)

pytestmark = pytest.mark.asyncio


async def test_query_returns_vector_matches():
    a = MockVectorAdapter()
    spec = QuerySpec(vector=[0.1, 0.2], top_k=5, namespace="default")
    res = await a.query(spec)
    assert isinstance(res, QueryResult)
    assert isinstance(res.matches, list)
    if res.matches:
        assert isinstance(res.matches[0], VectorMatch)


async def test_query_validates_dimensions():
    a = MockVectorAdapter()
    caps = await a.capabilities()
    bad_dim = (caps.max_dimensions or 8) + 1
    spec = QuerySpec(vector=[0.0] * bad_dim, top_k=5, namespace="default")
    with pytest.raises(DimensionMismatch):
        await a.query(spec)


async def test_query_top_k_must_be_positive():
    a = MockVectorAdapter()
    with pytest.raises(BadRequest):
        await a.query(QuerySpec(vector=[0.1], top_k=0, namespace="default"))


async def test_query_respects_max_top_k():
    a = MockVectorAdapter()
    caps = await a.capabilities()
    if caps.max_top_k is None:
        pytest.skip("Adapter does not publish max_top_k")

    with pytest.raises(BadRequest):
        await a.query(
            QuerySpec(
                vector=[0.1],
                top_k=caps.max_top_k + 1,
                namespace="default",
            )
        )


async def test_query_results_sorted_by_score_desc():
    a = MockVectorAdapter()
    spec = QuerySpec(vector=[0.1, 0.2], top_k=10, namespace="default")
    res = await a.query(spec)
    scores = [m.score for m in res.matches]
    assert scores == sorted(scores, reverse=True)


async def test_query_include_flags_respected():
    a = MockVectorAdapter()
    # include metadata only
    spec = QuerySpec(
        vector=[0.1, 0.2],
        top_k=3,
        namespace="default",
        include_metadata=True,
        include_vectors=False,
    )
    res = await a.query(spec)
    for m in res.matches:
        # metadata MAY be present; but when present must be mapping
        if m.vector.metadata is not None:
            assert isinstance(m.vector.metadata, dict)
