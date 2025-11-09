# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Namespace management.

Spec refs:
  • SPECIFICATION.md §9.3 (create_namespace/delete_namespace)
  • SPECIFICATION.md §9.4 (Distance Metrics)
"""

import pytest

from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter
from adapter_sdk.vector_base import (
    NamespaceSpec,
    BadRequest,
    NotSupported,
)

pytestmark = pytest.mark.asyncio


async def test_create_namespace_returns_success():
    a = MockVectorAdapter()
    spec = NamespaceSpec(namespace="ns1", dimensions=8, distance_metric="cosine")
    res = await a.create_namespace(spec)
    assert res.success is True
    assert res.namespace == "ns1"


async def test_namespace_requires_positive_dimensions():
    a = MockVectorAdapter()
    spec = NamespaceSpec(namespace="bad", dimensions=0, distance_metric="cosine")
    with pytest.raises(BadRequest):
        await a.create_namespace(spec)


async def test_namespace_requires_valid_distance_metric():
    a = MockVectorAdapter()
    spec = NamespaceSpec(namespace="ns2", dimensions=8, distance_metric="weird")
    with pytest.raises(NotSupported):
        await a.create_namespace(spec)


async def test_health_exposes_namespaces_dict():
    a = MockVectorAdapter()
    h = await a.health()
    # Shape: { "ok": bool, "server": str, "version": str, "namespaces": {...} }
    assert isinstance(h, dict)
    assert "namespaces" in h
    assert isinstance(h["namespaces"], dict)


async def test_delete_namespace_idempotent():
    a = MockVectorAdapter()
    spec = NamespaceSpec(namespace="gone", dimensions=8, distance_metric="cosine")
    await a.create_namespace(spec)

    res1 = await a.delete_namespace("gone")
    res2 = await a.delete_namespace("gone")

    assert res1.success is True
    assert res2.success is True


async def test_namespace_isolation():
    a = MockVectorAdapter()

    # Two namespaces get distinct vectors / isolation behavior
    from adapter_sdk.vector_base import Vector, VectorID, UpsertSpec

    v1 = Vector(id=VectorID("a"), vector=[0.1, 0.2], namespace="ns_a")
    v2 = Vector(id=VectorID("b"), vector=[0.9, 0.8], namespace="ns_b")

    await a.upsert(UpsertSpec(vectors=[v1], namespace="ns_a"))
    await a.upsert(UpsertSpec(vectors=[v2], namespace="ns_b"))

    from adapter_sdk.vector_base import QuerySpec

    r_a = await a.query(QuerySpec(vector=[0.1, 0.2], top_k=5, namespace="ns_a"))
    r_b = await a.query(QuerySpec(vector=[0.9, 0.8], top_k=5, namespace="ns_b"))

    assert all(m.vector.namespace == "ns_a" for m in r_a.matches)
    assert all(m.vector.namespace == "ns_b" for m in r_b.matches)
