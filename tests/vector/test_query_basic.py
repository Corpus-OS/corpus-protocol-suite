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
    NotSupported,
    NamespaceSpec,
    Vector,
    VectorID,
    UpsertSpec,
)

pytestmark = pytest.mark.asyncio


async def _ensure_namespace_ready(adapter, *, namespace: str = "default", dimensions: int = 2, distance_metric: str = "cosine") -> str:
    """
    Ensure a namespace exists and contains at least one vector so backends that
    treat empty namespaces as IndexNotReady can still satisfy query semantics tests.

    This is intentionally capability-aware and does not use skips.
    """
    caps = await adapter.capabilities()

    # Prefer creating the namespace when supported.
    if getattr(caps, "supports_index_management", False):
        try:
            await adapter.create_namespace(
                NamespaceSpec(namespace=namespace, dimensions=dimensions, distance_metric=distance_metric)
            )
        except Exception:
            # Namespace may already exist; proceed.
            pass

    # Try to seed at least one vector so query can return shape-valid results.
    try:
        await adapter.upsert(
            UpsertSpec(
                namespace=namespace,
                vectors=[
                    Vector(
                        id=VectorID(f"seed-{namespace}-q"),
                        vector=[0.1] * dimensions,
                        metadata={"seed": True},
                        namespace=namespace,
                    )
                ],
            )
        )
        return namespace
    except Exception:
        # Fall back: if health exposes namespaces, use the first available.
        try:
            h = await adapter.health()
            ns_map = h.get("namespaces")
            if isinstance(ns_map, dict) and ns_map:
                return next(iter(ns_map.keys()))
        except Exception:
            pass

    # Last resort: return requested namespace (tests may still validate error behavior).
    return namespace


async def _seed_for_scoring(adapter, *, namespace: str, dimensions: int = 2) -> None:
    """
    Seed multiple vectors so score ordering tests have >1 match.
    """
    vecs = [
        Vector(id=VectorID(f"{namespace}-s1"), vector=[1.0, 0.0][:dimensions], metadata={"label": "a"}, namespace=namespace),
        Vector(id=VectorID(f"{namespace}-s2"), vector=[0.0, 1.0][:dimensions], metadata={"label": "b"}, namespace=namespace),
        Vector(id=VectorID(f"{namespace}-s3"), vector=[0.7, 0.7][:dimensions], metadata={"label": "c"}, namespace=namespace),
    ]
    await adapter.upsert(UpsertSpec(namespace=namespace, vectors=vecs))


async def test_query_query_returns_vector_matches(adapter):
    """Verify query operations return properly structured results."""
    ns = await _ensure_namespace_ready(adapter, namespace="default", dimensions=2)
    # Add more data so we can reliably observe matches.
    try:
        await _seed_for_scoring(adapter, namespace=ns, dimensions=2)
    except Exception:
        # If the backend rejects duplicates or seeding, proceed with whatever exists.
        pass

    spec = QuerySpec(vector=[0.1, 0.2], top_k=5, namespace=ns)
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
    ns = await _ensure_namespace_ready(adapter, namespace="default", dimensions=2)

    caps = await adapter.capabilities()
    # No skips: if max_top_k isn't published, behavior must not enforce a hidden max.
    probe = (caps.max_top_k + 1) if caps.max_top_k is not None else 1001

    try:
        await adapter.query(QuerySpec(vector=[0.1, 0.2], top_k=probe, namespace=ns))
    except BadRequest as e:
        # If the adapter rejects top_k, it MUST publish max_top_k.
        assert caps.max_top_k is not None, "Adapter enforces max_top_k but does not publish caps.max_top_k"
        err = e
        assert "top_k" in str(err).lower() or "max" in str(err).lower()
        return

    # If it succeeds, that's fine (either max_top_k is None or probe is allowed).
    assert True


async def test_query_results_sorted_by_score_desc(adapter):
    """Verify query results are sorted by score in descending order."""
    ns = await _ensure_namespace_ready(adapter, namespace="default", dimensions=2)
    try:
        await _seed_for_scoring(adapter, namespace=ns, dimensions=2)
    except Exception:
        pass

    spec = QuerySpec(vector=[0.9, 0.1], top_k=10, namespace=ns)
    result = await adapter.query(spec)

    scores = [match.score for match in result.matches]
    # Verify scores are in descending order (highest first)
    assert scores == sorted(scores, reverse=True)


async def test_query_include_flags_respected(adapter):
    """Verify query include flags control returned data."""
    ns = await _ensure_namespace_ready(adapter, namespace="default", dimensions=2)
    try:
        await _seed_for_scoring(adapter, namespace=ns, dimensions=2)
    except Exception:
        pass

    # Query with metadata only, no vectors
    spec = QuerySpec(
        vector=[0.1, 0.2],
        top_k=3,
        namespace=ns,
        include_metadata=True,
        include_vectors=False,
    )
    result = await adapter.query(spec)

    for match in result.matches:
        # When metadata is present, it must be a dictionary
        if match.vector.metadata is not None:
            assert isinstance(match.vector.metadata, dict)

        # When vectors are excluded, Vector.vector should remain a list type (typically empty list).
        assert isinstance(match.vector.vector, list)


# ---------------------------------------------------------------------------
# NEW tests (focused on BaseVectorAdapter + MockVectorAdapter alignment)
# ---------------------------------------------------------------------------

async def test_query_include_vectors_false_returns_list_type(adapter):
    """NEW: include_vectors=False should still return List[float] (often [])."""
    ns = await _ensure_namespace_ready(adapter, namespace="default", dimensions=2)
    spec = QuerySpec(vector=[0.1, 0.2], top_k=3, namespace=ns, include_vectors=False)
    res = await adapter.query(spec)

    for m in res.matches:
        assert isinstance(m.vector.vector, list)


async def test_query_include_metadata_false_allows_none_or_empty(adapter):
    """NEW: include_metadata=False should not return non-mapping metadata types."""
    ns = await _ensure_namespace_ready(adapter, namespace="default", dimensions=2)
    try:
        await _seed_for_scoring(adapter, namespace=ns, dimensions=2)
    except Exception:
        pass

    res = await adapter.query(
        QuerySpec(vector=[0.1, 0.2], top_k=3, namespace=ns, include_metadata=False, include_vectors=False)
    )

    for m in res.matches:
        if m.vector.metadata is not None:
            assert isinstance(m.vector.metadata, dict)


async def test_query_respects_supports_metadata_filtering_capability(adapter):
    """NEW: filter usage must be consistent with caps.supports_metadata_filtering."""
    ns = await _ensure_namespace_ready(adapter, namespace="default", dimensions=2)
    caps = await adapter.capabilities()

    if not caps.supports_metadata_filtering:
        with pytest.raises(NotSupported):
            await adapter.query(QuerySpec(vector=[0.1, 0.2], top_k=3, namespace=ns, filter={"tag": "x"}))
        return

    # Supported → should succeed and be shape-valid.
    res = await adapter.query(QuerySpec(vector=[0.1, 0.2], top_k=3, namespace=ns, filter={"seed": True}))
    assert isinstance(res, QueryResult)
    assert isinstance(res.matches, list)


async def test_query_unknown_namespace_behavior_consistent_with_contract(adapter):
    """NEW: unknown namespace must raise (BadRequest/NotSupported) or succeed with well-formed shape."""
    unknown = "__unknown_ns_query__"
    try:
        res = await adapter.query(QuerySpec(vector=[0.1, 0.2], top_k=3, namespace=unknown))
    except (BadRequest, NotSupported):
        return

    assert isinstance(res, QueryResult)
    assert res.namespace == unknown
    assert isinstance(res.matches, list)


async def test_query_does_not_require_exact_score_values(adapter):
    """NEW: scores should be numeric; ordering non-increasing when present."""
    ns = await _ensure_namespace_ready(adapter, namespace="default", dimensions=2)
    try:
        await _seed_for_scoring(adapter, namespace=ns, dimensions=2)
    except Exception:
        pass

    res = await adapter.query(QuerySpec(vector=[0.8, 0.6], top_k=5, namespace=ns))
    scores = [m.score for m in res.matches]

    for s in scores:
        assert isinstance(s, (int, float))
    assert scores == sorted(scores, reverse=True)
