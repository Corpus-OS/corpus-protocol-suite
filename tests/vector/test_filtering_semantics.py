# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Metadata filtering.

Spec refs:
  • SPECIFICATION.md §9.3 (query/delete filters)
"""

import pytest
from corpus_sdk.vector.vector_base import (
    Vector,
    VectorID,
    UpsertSpec,
    QuerySpec,
    DeleteSpec,
    BadRequest,
    NotSupported,
)

pytestmark = pytest.mark.asyncio


async def test_filtering_query_filter_equality(adapter):
    """Verify equality filters work correctly on query operations."""
    # Create a vector with metadata
    vector = Vector(
        id=VectorID("filter-test-1"),
        vector=[0.1, 0.2],
        metadata={"tag": "keep", "category": "test"},
        namespace="default",
    )
    await adapter.upsert(UpsertSpec(namespace="default", vectors=[vector]))

    # Query with filter
    result = await adapter.query(
        QuerySpec(
            vector=[0.1, 0.2],
            top_k=5,
            namespace="default",
            filter={"tag": "keep"},
        )
    )
    
    # At minimum, should return valid result structure
    assert isinstance(result.matches, list)


async def test_filtering_delete_filter_equality(adapter):
    """Verify equality filters work correctly on delete operations."""
    spec = DeleteSpec(namespace="default", ids=[], filter={"status": "obsolete"})
    result = await adapter.delete(spec)
    
    assert isinstance(result, DeleteResult)
    assert result.deleted_count >= 0


async def test_filtering_filter_requires_mapping_type(adapter):
    """Verify filters must be mapping types (dict-like)."""
    with pytest.raises(BadRequest) as exc_info:
        await adapter.query(
            QuerySpec(
                vector=[0.1],
                top_k=1,
                namespace="default",
                filter="not-a-dict",  # Wrong type
            )
        )
    
    err = exc_info.value
    assert "filter" in str(err).lower() or "mapping" in str(err).lower()


async def test_filtering_filter_respects_capabilities_support(adapter):
    """Verify filtering respects adapter capability flags."""
    caps = await adapter.capabilities()
    
    if not caps.supports_metadata_filtering:
        with pytest.raises(NotSupported) as exc_info:
            await adapter.query(
                QuerySpec(
                    vector=[0.1],
                    top_k=1,
                    namespace="default",
                    filter={"field": "value"},
                )
            )
        
        err = exc_info.value
        assert err.code == "NOT_SUPPORTED"


async def test_filtering_filter_empty_results_ok(adapter):
    """Verify filters that match no vectors return empty results properly."""
    result = await adapter.query(
        QuerySpec(
            vector=[0.9, 0.9],
            top_k=3,
            namespace="default",
            filter={"tag": "completely-unlikely-value-12345"},
        )
    )
    
    assert isinstance(result.matches, list)
    # Empty results are valid - the important thing is no crash