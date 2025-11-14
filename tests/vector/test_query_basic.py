# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Query semantics.

Spec refs:
  • SPECIFICATION.md §9.3 (query)
  • SPECIFICATION.md §9.2 (Data Types)
"""

import pytest
from corpus_sdk.vector.vector_base import (
    QuerySpec,
    VectorMatch,
    QueryResult,
    DimensionMismatch,
    BadRequest,
)

pytestmark = pytest.mark.asyncio


async def test_query_query_returns_vector_matches(adapter):
    """Verify query operations return properly structured results."""
    spec = QuerySpec(vector=[0.1, 0.2], top_k=5, namespace="default")
    result = await adapter.query(spec)
    
    assert isinstance(result, QueryResult)
    assert isinstance(result.matches, list)
    if result.matches:  # If there are matches, verify their structure
        assert isinstance(result.matches[0], VectorMatch)


async def test_query_validates_dimensions(adapter):
    """Verify query validates vector dimensions."""
    caps = await adapter.capabilities()
    bad_dimension = (caps.max_dimensions or 8) + 1
    
    spec = QuerySpec(vector=[0.0] * bad_dimension, top_k=5, namespace="default")
    
    with pytest.raises(DimensionMismatch) as exc_info:
        await adapter.query(spec)
    
    err = exc_info.value
    assert err.code == "DIMENSION_MISMATCH"


async def test_query_top_k_must_be_positive(adapter):
    """Verify top_k must be a positive integer."""
    with pytest.raises(BadRequest) as exc_info:
        await adapter.query(QuerySpec(vector=[0.1], top_k=0, namespace="default"))
    
    err = exc_info.value
    assert "top_k" in str(err).lower() or "positive" in str(err).lower()


async def test_query_respects_max_top_k(adapter):
    """Verify query respects adapter's maximum top_k limit."""
    caps = await adapter.capabilities()
    if caps.max_top_k is None:
        pytest.skip("Adapter does not publish max_top_k")

    with pytest.raises(BadRequest) as exc_info:
        await adapter.query(
            QuerySpec(
                vector=[0.1],
                top_k=caps.max_top_k + 1,
                namespace="default",
            )
        )
    
    err = exc_info.value
    assert "top_k" in str(err).lower() or "max" in str(err).lower()


async def test_query_results_sorted_by_score_desc(adapter):
    """Verify query results are sorted by score in descending order."""
    spec = QuerySpec(vector=[0.1, 0.2], top_k=10, namespace="default")
    result = await adapter.query(spec)
    
    scores = [match.score for match in result.matches]
    # Verify scores are in descending order (highest first)
    assert scores == sorted(scores, reverse=True)


async def test_query_include_flags_respected(adapter):
    """Verify query include flags control returned data."""
    # Query with metadata only, no vectors
    spec = QuerySpec(
        vector=[0.1, 0.2],
        top_k=3,
        namespace="default",
        include_metadata=True,
        include_vectors=False,
    )
    result = await adapter.query(spec)
    
    for match in result.matches:
        # When metadata is present, it must be a dictionary
        if match.vector.metadata is not None:
            assert isinstance(match.vector.metadata, dict)
        
        # When vectors are excluded, they may be None or empty
        # The spec allows either behavior for performance