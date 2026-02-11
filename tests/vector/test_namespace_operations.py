# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Namespace management.

Spec refs:
  • SPECIFICATION.md §9.3 (create_namespace/delete_namespace)
  • SPECIFICATION.md §9.4 (Distance Metrics)
"""

import pytest
from typing import Optional, Dict, Any

from corpus_sdk.vector.vector_base import (
    NamespaceSpec,
    BadRequest,
    NotSupported,
    Vector,
    VectorID,
    UpsertSpec,
    QuerySpec,
    BatchQuerySpec,
)

pytestmark = pytest.mark.asyncio


async def _maybe_create_namespace(adapter, *, namespace: str, dimensions: int, distance_metric: str = "cosine") -> None:
    caps = await adapter.capabilities()
    if not getattr(caps, "supports_index_management", False):
        return
    try:
        await adapter.create_namespace(NamespaceSpec(namespace=namespace, dimensions=dimensions, distance_metric=distance_metric))
    except NotSupported as e:
        raise AssertionError("capabilities.supports_index_management=True but create_namespace raised NotSupported") from e
    except BadRequest as e:
        msg = str(e).lower()
        if any(tok in msg for tok in ("already", "exists", "exist")):
            return
        raise


async def _ensure_namespace_ready(
    adapter,
    *,
    namespace: str,
    dimensions: int,
    seed_id: str,
    seed_vector: float,
    seed_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    await _maybe_create_namespace(adapter, namespace=namespace, dimensions=dimensions, distance_metric="cosine")
    v = Vector(
        id=VectorID(seed_id),
        vector=[float(seed_vector)] * int(dimensions),
        metadata=seed_metadata,
        namespace=namespace,
    )
    await adapter.upsert(UpsertSpec(namespace=namespace, vectors=[v]))
    return namespace


async def test_namespace_create_namespace_returns_success(adapter):
    """Verify namespace creation returns success result."""
    caps = await adapter.capabilities()
    spec = NamespaceSpec(namespace="test-namespace-1", dimensions=8, distance_metric="cosine")

    if not caps.supports_index_management:
        with pytest.raises(NotSupported):
            await adapter.create_namespace(spec)
        return

    result = await adapter.create_namespace(spec)
    assert result.success is True
    assert result.namespace == "test-namespace-1"


async def test_namespace_namespace_requires_positive_dimensions(adapter):
    """Verify namespace creation requires positive dimensions."""
    spec = NamespaceSpec(namespace="invalid-namespace", dimensions=0, distance_metric="cosine")

    with pytest.raises(BadRequest) as exc_info:
        await adapter.create_namespace(spec)

    err = exc_info.value
    error_msg = str(err).lower()
    assert any(keyword in error_msg for keyword in ["dimension", "positive", "invalid", "must"])


async def test_namespace_namespace_requires_valid_distance_metric(adapter):
    """Verify namespace creation requires valid distance metrics."""
    spec = NamespaceSpec(namespace="invalid-metric", dimensions=8, distance_metric="invalid-metric")

    with pytest.raises((BadRequest, NotSupported)) as exc_info:
        await adapter.create_namespace(spec)

    err = exc_info.value
    error_msg = str(err).lower()
    assert any(keyword in error_msg for keyword in ["metric", "distance", "invalid", "supported", "must"])


async def test_namespace_health_exposes_namespaces_dict(adapter):
    """Verify health response includes namespace dictionary."""
    health = await adapter.health()

    assert isinstance(health, dict)
    namespace_key = None
    for key in ["namespaces", "namespace", "collections"]:
        if key in health:
            namespace_key = key
            break

    assert namespace_key is not None, f"No namespace key found in health response: {health.keys()}"
    assert isinstance(health[namespace_key], (dict, list))


async def test_namespace_delete_namespace_idempotent(adapter):
    """Verify namespace deletion is idempotent."""
    caps = await adapter.capabilities()
    spec = NamespaceSpec(namespace="temporary-namespace", dimensions=8, distance_metric="cosine")

    if not caps.supports_index_management:
        with pytest.raises(NotSupported):
            await adapter.delete_namespace("temporary-namespace")
        return

    await adapter.create_namespace(spec)

    result1 = await adapter.delete_namespace("temporary-namespace")
    result2 = await adapter.delete_namespace("temporary-namespace")

    assert result1.success is True
    assert result2.success is True


async def test_namespace_namespace_isolation(adapter):
    """Verify namespace isolation prevents cross-namespace data leakage."""
    caps = await adapter.capabilities()

    if not caps.supports_namespaces:
        with pytest.raises(NotSupported):
            await adapter.query(QuerySpec(vector=[0.1, 0.2], top_k=1, namespace="namespace-a"))
        return

    namespace_a = "namespace-a"
    namespace_b = "namespace-b"

    await _maybe_create_namespace(adapter, namespace=namespace_a, dimensions=2, distance_metric="cosine")
    await _maybe_create_namespace(adapter, namespace=namespace_b, dimensions=2, distance_metric="cosine")

    await adapter.upsert(
        UpsertSpec(
            vectors=[Vector(id=VectorID("vector-a"), vector=[0.1, 0.2], namespace=namespace_a)],
            namespace=namespace_a,
        )
    )
    await adapter.upsert(
        UpsertSpec(
            vectors=[Vector(id=VectorID("vector-b"), vector=[0.9, 0.8], namespace=namespace_b)],
            namespace=namespace_b,
        )
    )

    result_a = await adapter.query(QuerySpec(vector=[0.1, 0.2], top_k=5, namespace=namespace_a))
    result_b = await adapter.query(QuerySpec(vector=[0.9, 0.8], top_k=5, namespace=namespace_b))

    for match in result_a.matches:
        if getattr(match.vector, "namespace", None):
            assert match.vector.namespace == namespace_a

    for match in result_b.matches:
        if getattr(match.vector, "namespace", None):
            assert match.vector.namespace == namespace_b


async def test_namespace_ops_respect_supports_index_management_flag(adapter):
    """
    Capabilities ↔ behavior:
      - supports_index_management=False ⇒ create/delete namespace should raise NotSupported
      - supports_index_management=True  ⇒ create/delete should succeed
    """
    caps = await adapter.capabilities()
    ns = "ns-cap-flag"
    spec = NamespaceSpec(namespace=ns, dimensions=2, distance_metric="cosine")

    if caps.supports_index_management:
        c = await adapter.create_namespace(spec)
        assert bool(getattr(c, "success", True)) is True
        d = await adapter.delete_namespace(ns)
        assert bool(getattr(d, "success", True)) is True
    else:
        with pytest.raises(NotSupported):
            await adapter.create_namespace(spec)
        with pytest.raises(NotSupported):
            await adapter.delete_namespace(ns)


async def test_namespace_query_rejects_namespace_when_supports_namespaces_false(adapter):
    """
    If supports_namespaces is False, providing a namespace should not silently succeed.
    """
    caps = await adapter.capabilities()
    if caps.supports_namespaces:
        ns = "ns-supported"
        await _ensure_namespace_ready(adapter, namespace=ns, dimensions=2, seed_id="nq1", seed_vector=0.21)
        res = await adapter.query(QuerySpec(vector=[0.21, 0.21], top_k=1, namespace=ns))
        assert isinstance(res.matches, list)
        return

    with pytest.raises(NotSupported):
        await adapter.query(QuerySpec(vector=[0.1, 0.2], top_k=1, namespace="ns-should-fail"))


async def test_namespace_upsert_rejects_vector_namespace_mismatch(adapter):
    """
    BaseVectorAdapter namespace semantics: UpsertSpec.namespace is authoritative.
    If Vector.namespace is provided and mismatches, upsert must raise BadRequest.
    """
    ns = "ns-upsert-mismatch"
    await _maybe_create_namespace(adapter, namespace=ns, dimensions=2, distance_metric="cosine")

    with pytest.raises(BadRequest):
        await adapter.upsert(
            UpsertSpec(
                namespace=ns,
                vectors=[
                    Vector(
                        id=VectorID("mismatch"),
                        vector=[0.1, 0.2],
                        namespace="different-ns",
                    )
                ],
            )
        )


async def test_namespace_batch_query_rejects_query_namespace_mismatch(adapter):
    """
    BaseVectorAdapter namespace semantics: BatchQuerySpec.namespace is authoritative.
    If any QuerySpec.namespace mismatches, batch_query must raise BadRequest (when supported).
    """
    caps = await adapter.capabilities()
    ns = "ns-batch-mismatch"
    await _ensure_namespace_ready(adapter, namespace=ns, dimensions=2, seed_id="bq_seed", seed_vector=0.31)

    spec = BatchQuerySpec(
        namespace=ns,
        queries=[
            QuerySpec(vector=[0.31, 0.31], top_k=1, namespace="other-ns"),
        ],
    )

    if not caps.supports_batch_queries:
        with pytest.raises(NotSupported):
            await adapter.batch_query(spec)
        return

    with pytest.raises(BadRequest):
        await adapter.batch_query(spec)
