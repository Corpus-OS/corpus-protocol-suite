# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Upsert basics & partial failures.

Spec refs:
  • SPECIFICATION.md §9.3 (upsert)
  • SPECIFICATION.md §9.5 (Vector-Specific Errors)
  • SPECIFICATION.md §12.5 (Partial Failure Contracts)
"""

import pytest
from corpus_sdk.vector.vector_base import (
    Vector,
    VectorID,
    UpsertSpec,
    UpsertResult,
    DimensionMismatch,
    BadRequest,
    NotSupported,
    NamespaceSpec,
)

pytestmark = pytest.mark.asyncio


async def _ensure_namespace(adapter, *, namespace: str = "default", dimensions: int = 2, metric: str = "cosine") -> str:
    """
    Ensure namespace exists when supports_index_management is true.
    If not supported, fall back to any namespace exposed in health (if present).
    """
    caps = await adapter.capabilities()
    if getattr(caps, "supports_index_management", False):
        try:
            await adapter.create_namespace(NamespaceSpec(namespace=namespace, dimensions=dimensions, distance_metric=metric))
        except Exception:
            pass
        return namespace

    # No index management: use health-discovered namespace if possible.
    try:
        h = await adapter.health()
        ns_map = h.get("namespaces")
        if isinstance(ns_map, dict) and ns_map:
            return next(iter(ns_map.keys()))
    except Exception:
        pass

    return namespace


async def test_upsert_upsert_returns_result_with_counts(adapter):
    """Verify upsert returns proper result structure with counts."""
    ns = await _ensure_namespace(adapter, namespace="default", dimensions=2)

    spec = UpsertSpec(
        namespace=ns,
        vectors=[Vector(id=VectorID("test-vector-1"), vector=[0.1, 0.2], namespace=ns)],
    )
    result = await adapter.upsert(spec)

    assert isinstance(result, UpsertResult)
    assert result.upserted_count == 1
    assert result.failed_count == 0
    assert isinstance(result.failures, list)


async def test_upsert_validates_dimensions(adapter):
    """Verify upsert validates vector dimensions."""
    caps = await adapter.capabilities()
    bad_dimension = (caps.max_dimensions or 8) + 1

    bad_vector = Vector(id=VectorID("bad-dimension"), vector=[0.0] * bad_dimension)
    spec = UpsertSpec(namespace="default", vectors=[bad_vector])

    with pytest.raises(DimensionMismatch) as exc_info:
        await adapter.upsert(spec)

    err = exc_info.value
    assert err.code == "DIMENSION_MISMATCH"


async def test_upsert_validates_namespace_exists_or_behavior_documented(adapter):
    """
    Spec: adapters MUST either validate unknown namespaces or define clear behavior.
    This test accepts either:
      - BadRequest / NotSupported on unknown namespace, OR
      - A successful UpsertResult with a well-formed shape.
    """
    spec = UpsertSpec(
        namespace="__completely_unknown_namespace_123__",
        vectors=[Vector(id=VectorID("test-vector"), vector=[0.1, 0.2])],
    )

    try:
        result = await adapter.upsert(spec)
    except (BadRequest, NotSupported):
        return  # Validation behavior is acceptable

    # If no exception, result must be well-formed
    assert isinstance(result, UpsertResult)
    assert isinstance(result.upserted_count, int)
    assert isinstance(result.failed_count, int)
    assert isinstance(result.failures, list)


async def test_upsert_requires_non_empty_vectors(adapter):
    """Verify upsert requires non-empty vectors list."""
    spec = UpsertSpec(namespace="default", vectors=[])

    with pytest.raises(BadRequest) as exc_info:
        await adapter.upsert(spec)

    err = exc_info.value
    assert "vector" in str(err).lower() or "empty" in str(err).lower()


async def test_upsert_partial_failure_reporting(adapter):
    """
    When partial success is implemented, failures MUST be listed with indices.
    If adapter chooses atomic failure via DimensionMismatch, that is also valid.
    """
    ns = await _ensure_namespace(adapter, namespace="default", dimensions=8)

    good_vector = Vector(id=VectorID("good-vector"), vector=[0.1] * 8, namespace=ns)
    bad_vector = Vector(id=VectorID("bad-dimension"), vector=[0.1] * 9, namespace=ns)

    try:
        result = await adapter.upsert(UpsertSpec(namespace=ns, vectors=[good_vector, bad_vector]))
    except DimensionMismatch:
        # Atomic failure is allowed by spec
        return

    # If partial success is implemented, verify proper reporting
    assert isinstance(result, UpsertResult)
    assert result.upserted_count >= 1
    assert result.failed_count >= 1
    assert len(result.failures) == result.failed_count


# ---------------------------------------------------------------------------
# NEW tests (focused on BaseVectorAdapter + MockVectorAdapter alignment)
# ---------------------------------------------------------------------------

async def test_upsert_rejects_vector_namespace_mismatch(adapter):
    """NEW: UpsertSpec.namespace is authoritative; Vector.namespace mismatch must raise BadRequest."""
    ns = await _ensure_namespace(adapter, namespace="ns-a", dimensions=2)

    spec = UpsertSpec(
        namespace=ns,
        vectors=[
            Vector(id=VectorID("ns-mismatch"), vector=[0.1, 0.2], namespace="different-ns")
        ],
    )
    with pytest.raises(BadRequest):
        await adapter.upsert(spec)


async def test_upsert_respects_max_batch_size_if_published(adapter):
    """NEW: If caps.max_batch_size is set, exceeding it must raise BadRequest and optional suggestion must be 0..100."""
    ns = await _ensure_namespace(adapter, namespace="default", dimensions=2)
    caps = await adapter.capabilities()

    if caps.max_batch_size is None:
        # No published limit: do not enforce a hidden limit here.
        # Still verify that a modest batch succeeds.
        small = [Vector(id=VectorID(f"s{i}"), vector=[0.1, 0.2], namespace=ns) for i in range(3)]
        res = await adapter.upsert(UpsertSpec(namespace=ns, vectors=small))
        assert isinstance(res, UpsertResult)
        return

    too_many = caps.max_batch_size + 1
    vectors = [Vector(id=VectorID(str(i)), vector=[0.1, 0.2], namespace=ns) for i in range(too_many)]

    with pytest.raises(BadRequest) as exc_info:
        await adapter.upsert(UpsertSpec(namespace=ns, vectors=vectors))

    err = exc_info.value
    if getattr(err, "suggested_batch_reduction", None) is not None:
        assert isinstance(err.suggested_batch_reduction, int)
        assert 0 <= err.suggested_batch_reduction <= 100


async def test_upsert_text_not_supported_when_text_storage_strategy_none(adapter):
    """NEW: If caps.text_storage_strategy == 'none', providing Vector.text must raise NotSupported."""
    ns = await _ensure_namespace(adapter, namespace="default", dimensions=2)
    caps = await adapter.capabilities()

    if getattr(caps, "text_storage_strategy", None) != "none":
        # If text is supported, this test is not applicable; ensure shape-valid success instead.
        res = await adapter.upsert(
            UpsertSpec(namespace=ns, vectors=[Vector(id=VectorID("t-ok"), vector=[0.1, 0.2], namespace=ns)])
        )
        assert isinstance(res, UpsertResult)
        return

    with pytest.raises(NotSupported):
        await adapter.upsert(
            UpsertSpec(
                namespace=ns,
                vectors=[Vector(id=VectorID("t1"), vector=[0.1, 0.2], namespace=ns, text="hello")],
            )
        )
