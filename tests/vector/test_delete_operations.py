# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Delete operations.

Spec refs:
  • SPECIFICATION.md §9.3 (delete)
  • SPECIFICATION.md §12.5 (Partial Failures)
"""

import pytest
from corpus_sdk.vector.vector_base import (
    Vector,
    VectorID,
    UpsertSpec,
    DeleteSpec,
    DeleteResult,
    BadRequest,
)

pytestmark = pytest.mark.asyncio


async def test_delete_delete_by_ids_returns_counts(adapter):
    """Verify delete by IDs returns proper counts and result structure."""
    # First upsert a vector to delete
    vector = Vector(id=VectorID("d1"), vector=[0.1, 0.2])
    await adapter.upsert(UpsertSpec(namespace="default", vectors=[vector]))

    # Then delete it
    result = await adapter.delete(DeleteSpec(namespace="default", ids=[VectorID("d1")]))
    
    assert isinstance(result, DeleteResult)
    assert result.deleted_count >= 1
    assert isinstance(result.failures, list)


async def test_delete_delete_by_filter_returns_counts(adapter):
    """Verify delete by filter returns proper result structure."""
    spec = DeleteSpec(namespace="default", ids=[], filter={"foo": "bar"})
    result = await adapter.delete(spec)
    
    assert isinstance(result, DeleteResult)
    assert result.deleted_count >= 0
    assert isinstance(result.failures, list)


async def test_delete_requires_ids_or_filter(adapter):
    """Verify delete requires either IDs or filter parameter."""
    spec = DeleteSpec(namespace="default", ids=[], filter=None)
    
    with pytest.raises(BadRequest) as exc_info:
        await adapter.delete(spec)
    
    err = exc_info.value
    assert "ids" in str(err).lower() or "filter" in str(err).lower()


async def test_delete_idempotent_for_missing_ids(adapter):
    """Verify delete operations are idempotent for non-existent IDs."""
    spec = DeleteSpec(namespace="default", ids=[VectorID("non-existent-id")])
    
    result1 = await adapter.delete(spec)
    result2 = await adapter.delete(spec)
    
    assert result1.deleted_count >= 0
    assert result2.deleted_count >= 0
    # Idempotent means subsequent calls should behave the same
    assert result1.deleted_count == result2.deleted_count


async def test_delete_delete_result_structure(adapter):
    """Verify DeleteResult has all required fields with proper types."""
    spec = DeleteSpec(namespace="default", ids=[VectorID("test-vector")])
    result = await adapter.delete(spec)
    
    assert hasattr(result, "deleted_count") and isinstance(result.deleted_count, int)
    assert hasattr(result, "failed_count") and isinstance(result.failed_count, int)
    assert hasattr(result, "failures") and isinstance(result.failures, list)