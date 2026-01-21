# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Metadata filtering.

Spec refs:
  • SPECIFICATION.md §9.3 (query/delete filters)
"""

import pytest
from typing import Optional, Dict, Any

from corpus_sdk.vector.vector_base import (
    Vector,
    VectorID,
    UpsertSpec,
    QuerySpec,
    DeleteSpec,
    DeleteResult,
    BadRequest,
    NotSupported,
    NamespaceSpec,
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


async def test_filtering_query_filter_equality(adapter):
    """Verify equality filters work correctly on query operations."""
    caps = await adapter.capabilities()
    ns = "test-filter-query"
    await _ensure_namespace_ready(
        adapter,
        namespace=ns,
        dimensions=2,
        seed_id="fq_seed",
        seed_vector=0.11,
        seed_metadata={"tag": "other"},
    )

    vector = Vector(
        id=VectorID("filter-test-1"),
        vector=[0.1, 0.2],
        metadata={"tag": "keep", "category": "test"},
        namespace=ns,
    )
    await adapter.upsert(UpsertSpec(namespace=ns, vectors=[vector]))

    if not caps.supports_metadata_filtering:
        with pytest.raises(NotSupported):
            await adapter.query(QuerySpec(vector=[0.1, 0.2], top_k=5, namespace=ns, filter={"tag": "keep"}))
        return

    result = await adapter.query(
        QuerySpec(
            vector=[0.1, 0.2],
            top_k=5,
            namespace=ns,
            filter={"tag": "keep"},
        )
    )

    assert isinstance(result.matches, list)


async def test_filtering_delete_filter_equality(adapter):
    """Verify equality filters work correctly on delete operations."""
    caps = await adapter.capabilities()
    ns = "test-filter-delete"
    await _ensure_namespace_ready(
        adapter,
        namespace=ns,
        dimensions=2,
        seed_id="fd1",
        seed_vector=0.12,
        seed_metadata={"status": "obsolete"},
    )

    spec = DeleteSpec(namespace=ns, ids=[], filter={"status": "obsolete"})

    if not caps.supports_metadata_filtering:
        with pytest.raises(NotSupported):
            await adapter.delete(spec)
        return

    result = await adapter.delete(spec)
    assert isinstance(result, DeleteResult)
    assert result.deleted_count >= 0


async def test_filtering_filter_requires_mapping_type(adapter):
    """Verify filters must be mapping types (dict-like)."""
    # This should fail in BaseVectorAdapter validation before reaching backend.
    with pytest.raises(BadRequest) as exc_info:
        await adapter.query(
            QuerySpec(
                vector=[0.1, 0.1],
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
                    vector=[0.1, 0.1],
                    top_k=1,
                    namespace="default",
                    filter={"field": "value"},
                )
            )
        err = exc_info.value
        assert err.code == "NOT_SUPPORTED"


async def test_filtering_filter_empty_results_ok(adapter):
    """Verify filters that match no vectors return empty results properly."""
    caps = await adapter.capabilities()
    ns = "test-filter-empty"
    await _ensure_namespace_ready(
        adapter,
        namespace=ns,
        dimensions=2,
        seed_id="fe1",
        seed_vector=0.13,
        seed_metadata={"tag": "something-else"},
    )

    spec = QuerySpec(
        vector=[0.9, 0.9],
        top_k=3,
        namespace=ns,
        filter={"tag": "completely-unlikely-value-12345"},
    )

    if not caps.supports_metadata_filtering:
        with pytest.raises(NotSupported):
            await adapter.query(spec)
        return

    result = await adapter.query(spec)
    assert isinstance(result.matches, list)


# NEW
async def test_filtering_unknown_operator_rejected_or_accepted_consistently(adapter):
    """
    Unknown operators are adapter-defined. If supported, the adapter may:
      - reject with BadRequest (preferred), OR
      - accept and treat as no-match.
    This test only enforces "no crash + consistent typing" when filtering is supported.
    """
    caps = await adapter.capabilities()
    ns = "test-filter-unknown-op"
    await _ensure_namespace_ready(
        adapter,
        namespace=ns,
        dimensions=2,
        seed_id="fu1",
        seed_vector=0.14,
        seed_metadata={"tag": "keep"},
    )

    spec = QuerySpec(vector=[0.14, 0.14], top_k=5, namespace=ns, filter={"tag": {"$unknown": ["keep"]}})

    if not caps.supports_metadata_filtering:
        with pytest.raises(NotSupported):
            await adapter.query(spec)
        return

    try:
        res = await adapter.query(spec)
        assert isinstance(res.matches, list)
    except BadRequest:
        pass


# NEW
async def test_filtering_filter_complexity_enforced_if_caps_max_filter_terms_declared(adapter):
    """
    If caps.max_filter_terms is declared, filters with > max_filter_terms keys must raise BadRequest
    when filtering is supported.
    """
    caps = await adapter.capabilities()
    ns = "test-filter-complexity"
    await _ensure_namespace_ready(
        adapter,
        namespace=ns,
        dimensions=2,
        seed_id="fc1",
        seed_vector=0.15,
        seed_metadata={"k0": "v0"},
    )

    if not caps.supports_metadata_filtering:
        with pytest.raises(NotSupported):
            await adapter.query(QuerySpec(vector=[0.15, 0.15], top_k=1, namespace=ns, filter={"k": "v"}))
        return

    if caps.max_filter_terms is None:
        res = await adapter.query(QuerySpec(vector=[0.15, 0.15], top_k=1, namespace=ns, filter={"k": "v"}))
        assert isinstance(res.matches, list)
        return

    too_many = int(caps.max_filter_terms) + 1
    flt = {f"k{i}": f"v{i}" for i in range(too_many)}
    with pytest.raises(BadRequest):
        await adapter.query(QuerySpec(vector=[0.15, 0.15], top_k=1, namespace=ns, filter=flt))
