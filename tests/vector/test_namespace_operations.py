# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Namespace management.

Spec refs:
  • SPECIFICATION.md §9.3 (create_namespace/delete_namespace)
  • SPECIFICATION.md §9.4 (Distance Metrics)
"""

import pytest
from corpus_sdk.vector.vector_base import (
    NamespaceSpec,
    BadRequest,
    NotSupported,
    Vector,
    VectorID,
    UpsertSpec,
    QuerySpec,
)

pytestmark = pytest.mark.asyncio


async def test_namespace_create_namespace_returns_success(adapter):
    """Verify namespace creation returns success result."""
    spec = NamespaceSpec(namespace="test-namespace-1", dimensions=8, distance_metric="cosine")
    result = await adapter.create_namespace(spec)
    
    assert result.success is True
    assert result.namespace == "test-namespace-1"


async def test_namespace_namespace_requires_positive_dimensions(adapter):
    """Verify namespace creation requires positive dimensions."""
    spec = NamespaceSpec(namespace="invalid-namespace", dimensions=0, distance_metric="cosine")
    
    with pytest.raises(BadRequest) as exc_info:
        await adapter.create_namespace(spec)
    
    err = exc_info.value
    assert "dimension" in str(err).lower() or "positive" in str(err).lower()


async def test_namespace_namespace_requires_valid_distance_metric(adapter):
    """Verify namespace creation requires valid distance metrics."""
    spec = NamespaceSpec(namespace="invalid-metric", dimensions=8, distance_metric="invalid-metric")
    
    with pytest.raises(NotSupported) as exc_info:
        await adapter.create_namespace(spec)
    
    err = exc_info.value
    assert "metric" in str(err).lower() or "distance" in str(err).lower()


async def test_namespace_health_exposes_namespaces_dict(adapter):
    """Verify health response includes namespace dictionary."""
    health = await adapter.health()
    
    assert isinstance(health, dict)
    assert "namespaces" in health
    assert isinstance(health["namespaces"], dict)


async def test_namespace_delete_namespace_idempotent(adapter):
    """Verify namespace deletion is idempotent."""
    # First create a namespace
    spec = NamespaceSpec(namespace="temporary-namespace", dimensions=8, distance_metric="cosine")
    await adapter.create_namespace(spec)

    # Delete it twice - both should succeed
    result1 = await adapter.delete_namespace("temporary-namespace")
    result2 = await adapter.delete_namespace("temporary-namespace")

    assert result1.success is True
    assert result2.success is True


async def test_namespace_namespace_isolation(adapter):
    """Verify namespace isolation prevents cross-namespace data leakage."""
    # Create vectors in different namespaces
    vector_a = Vector(id=VectorID("vector-a"), vector=[0.1, 0.2], namespace="namespace-a")
    vector_b = Vector(id=VectorID("vector-b"), vector=[0.9, 0.8], namespace="namespace-b")

    await adapter.upsert(UpsertSpec(vectors=[vector_a], namespace="namespace-a"))
    await adapter.upsert(UpsertSpec(vectors=[vector_b], namespace="namespace-b"))

    # Query each namespace - should only find vectors from that namespace
    result_a = await adapter.query(QuerySpec(vector=[0.1, 0.2], top_k=5, namespace="namespace-a"))
    result_b = await adapter.query(QuerySpec(vector=[0.9, 0.8], top_k=5, namespace="namespace-b"))

    # Verify namespace isolation
    for match in result_a.matches:
        assert match.vector.namespace == "namespace-a"
    
    for match in result_b.matches:
        assert match.vector.namespace == "namespace-b"