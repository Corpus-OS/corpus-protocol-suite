# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Batch size limits & partial failures.

Spec refs:
  • SPECIFICATION.md §9.3 (Batch operations)
  • SPECIFICATION.md §12.5 (Partial Failure Contracts)
"""

import pytest
from corpus_sdk.vector.vector_base import (
    Vector,
    VectorID,
    UpsertSpec,
    DeleteSpec,
    BadRequest,
    UpsertResult,
    DimensionMismatch,
    QuerySpec,
    NotSupported,
    NamespaceSpec,
)

pytestmark = pytest.mark.asyncio


async def _ensure_namespace_for_dims(adapter, *, desired_dims: int = 2) -> str:
    """
    Best-effort creation of a dedicated namespace with known dimensions.

    If supports_index_management is False, returns "default" without attempting creation.
    If supports_index_management is True but create_namespace fails, that is a capabilities mismatch.
    """
    caps = await adapter.capabilities()
    if not getattr(caps, "supports_index_management", False):
        return "default"

    ns = f"test-batch-limits-d{desired_dims}"
    try:
        await adapter.create_namespace(
            NamespaceSpec(namespace=ns, dimensions=int(desired_dims), distance_metric="cosine")
        )
    except NotSupported as e:
        # Capabilities mismatch: claimed supports_index_management but operation is not supported.
        raise AssertionError(
            "capabilities.supports_index_management=True but create_namespace raised NotSupported"
        ) from e
    return ns


async def _prime_namespace(adapter, namespace: str, dims: int) -> None:
    """
    Ensure the namespace is "ready" for query in adapters that require existing data.
    """
    try:
        await adapter.upsert(
            UpsertSpec(
                namespace=namespace,
                vectors=[Vector(id=VectorID("prime"), vector=[0.01] * dims)],
            )
        )
    except Exception:
        # If the adapter enforces a different schema, we can't safely prime.
        # This helper is best-effort; tests that rely on it handle failures.
        return


async def test_batch_limits_upsert_respects_max_batch_size(adapter):
    """Verify adapter enforces maximum batch size limits."""
    caps = await adapter.capabilities()

    # Prefer a known-dimension namespace when possible to avoid guessing schema dims.
    dims = 2
    namespace = await _ensure_namespace_for_dims(adapter, desired_dims=dims)

    # Choose a batch size to probe:
    # - If max_batch_size declared: exceed it by 1.
    # - Else: use a small batch (2) that should be valid for batch-capable adapters.
    probe_n = (caps.max_batch_size + 1) if (caps.max_batch_size is not None) else 2

    spec = UpsertSpec(
        namespace=namespace,
        vectors=[Vector(id=VectorID(str(i)), vector=[0.1] * dims) for i in range(probe_n)],
    )

    try:
        await adapter.upsert(spec)
        # If it succeeded while declaring a max_batch_size smaller than the probe, that's a mismatch.
        if caps.max_batch_size is not None and probe_n > caps.max_batch_size:
            raise AssertionError(
                f"upsert succeeded with batch size {probe_n} but capabilities.max_batch_size={caps.max_batch_size}"
            )
    except NotSupported:
        # If adapter doesn't support batch operations, it should say so in capabilities.
        assert not caps.supports_batch_operations, (
            "Adapter raised NotSupported for multi-item upsert but capabilities.supports_batch_operations=True"
        )
        assert probe_n > 1
    except BadRequest as e:
        # If adapter enforces a batch limit via BadRequest, it must declare max_batch_size.
        assert caps.max_batch_size is not None, (
            "Adapter raised BadRequest for batch sizing but capabilities.max_batch_size is None (mismatch)"
        )
        assert probe_n > caps.max_batch_size, (
            f"Adapter raised BadRequest for batch sizing at {probe_n}, but caps.max_batch_size={caps.max_batch_size}"
        )
        # Avoid asserting message text; just ensure normalized type is used.
        assert isinstance(e, BadRequest)
    except DimensionMismatch:
        # If we couldn't guarantee schema dimensions (non-index-managed adapters),
        # DimensionMismatch is acceptable; this test is specifically about batch gating.
        return


async def test_batch_limits_batch_size_exceeded_includes_suggestion(adapter):
    """Verify batch size errors include helpful reduction suggestions when provided."""
    caps = await adapter.capabilities()

    dims = 2
    namespace = await _ensure_namespace_for_dims(adapter, desired_dims=dims)

    # If max_batch_size is not declared, we can still enforce consistency:
    # - A "batch size" BadRequest without declaring max_batch_size is a mismatch.
    probe_n = (caps.max_batch_size + 10) if (caps.max_batch_size is not None) else 2

    spec = UpsertSpec(
        namespace=namespace,
        vectors=[Vector(id=VectorID(str(i)), vector=[0.1] * dims) for i in range(probe_n)],
    )

    try:
        await adapter.upsert(spec)
        # If it succeeded while declaring a max_batch_size smaller than the probe, mismatch.
        if caps.max_batch_size is not None and probe_n > caps.max_batch_size:
            raise AssertionError(
                f"upsert succeeded with batch size {probe_n} but capabilities.max_batch_size={caps.max_batch_size}"
            )
    except BadRequest as exc_info:
        err = exc_info
        # If the adapter is throwing a batch-size related BadRequest, it must declare the limit.
        assert caps.max_batch_size is not None, (
            "Adapter raised BadRequest for batch sizing but capabilities.max_batch_size is None (mismatch)"
        )

        # suggestion is optional, but if present must be int-ish and aligned with base semantics (percentage).
        if getattr(err, "suggested_batch_reduction", None) is not None:
            assert isinstance(err.suggested_batch_reduction, int)
            # Base semantics: percentage hint (0..100). Keep tolerant if provider uses a different convention.
            assert 0 <= err.suggested_batch_reduction <= 100
    except NotSupported:
        # Allowed only if adapter doesn't support batch operations.
        assert not caps.supports_batch_operations


async def test_batch_limits_partial_failure_reporting_shape(adapter):
    """
    When adapter chooses partial behavior (some fail, some succeed),
    failures MUST be listed. If adapter is atomic, it's allowed to raise instead.
    """
    caps = await adapter.capabilities()

    # Use a known-dimension namespace when possible.
    if caps.supports_index_management:
        dims = 3
        namespace = await _ensure_namespace_for_dims(adapter, desired_dims=dims)
    else:
        # Without index management, we cannot safely infer schema dimensions; use default namespace.
        # This test remains tolerant: atomic DimensionMismatch is acceptable.
        dims = 2
        namespace = "default"

    good = Vector(id=VectorID("ok"), vector=[0.1] * dims)
    bad = Vector(id=VectorID("bad"), vector=[0.1] * (dims + 1))

    try:
        res = await adapter.upsert(UpsertSpec(namespace=namespace, vectors=[good, bad]))
    except (DimensionMismatch, BadRequest):
        return  # Atomic behavior is acceptable

    assert isinstance(res, UpsertResult)
    assert res.failed_count >= 1
    assert len(res.failures) == res.failed_count


async def test_batch_limits_batch_operations_atomic_per_vector(adapter):
    """
    Confirms that failure for one item does not corrupt others when partial
    semantics are implemented. If adapter chooses atomic failure, this test
    treats that as acceptable via the same DimensionMismatch escape.
    """
    caps = await adapter.capabilities()

    # Use a known-dimension namespace when possible.
    if caps.supports_index_management:
        dims = 3
        namespace = await _ensure_namespace_for_dims(adapter, desired_dims=dims)
        await _prime_namespace(adapter, namespace, dims)
    else:
        dims = 2
        namespace = "default"

    v_good = Vector(id=VectorID("ok2"), vector=[0.2] * dims)
    v_bad = Vector(id=VectorID("bad2"), vector=[0.2] * (dims + 1))

    try:
        res = await adapter.upsert(UpsertSpec(namespace=namespace, vectors=[v_good, v_bad]))
    except (DimensionMismatch, BadRequest):
        return  # Atomic behavior is acceptable

    assert res.upserted_count >= 1

    # Good vector should be queryable
    qr = await adapter.query(QuerySpec(vector=[0.2] * dims, top_k=5, namespace=namespace))
    ids = [match.vector.id for match in qr.matches]
    assert VectorID("ok2") in ids


async def test_batch_limits_delete_respects_max_batch_size_or_supports_batch_operations(adapter):
    """NEW: Verify delete enforces max_batch_size consistently with capabilities."""
    caps = await adapter.capabilities()

    namespace = "default"
    probe_n = (caps.max_batch_size + 1) if (caps.max_batch_size is not None) else 2
    spec = DeleteSpec(ids=[VectorID(str(i)) for i in range(probe_n)], namespace=namespace)

    try:
        await adapter.delete(spec)
        # If delete succeeds with declared max_batch_size smaller than probe, mismatch.
        if caps.max_batch_size is not None and probe_n > caps.max_batch_size:
            raise AssertionError(
                f"delete succeeded with batch size {probe_n} but capabilities.max_batch_size={caps.max_batch_size}"
            )
    except NotSupported:
        assert not caps.supports_batch_operations, (
            "Adapter raised NotSupported for multi-id delete but capabilities.supports_batch_operations=True"
        )
        assert probe_n > 1
    except BadRequest:
        assert caps.max_batch_size is not None, (
            "Adapter raised BadRequest for delete batch sizing but capabilities.max_batch_size is None (mismatch)"
        )
        assert probe_n > caps.max_batch_size


async def test_batch_limits_batch_query_respects_supports_batch_queries(adapter):
    """NEW: Verify batch_query honors capabilities.supports_batch_queries."""
    caps = await adapter.capabilities()

    dims = 2
    namespace = await _ensure_namespace_for_dims(adapter, desired_dims=dims)
    await _prime_namespace(adapter, namespace, dims)

    spec = BatchQuerySpec(
        namespace=namespace,
        queries=[
            QuerySpec(vector=[0.01] * dims, top_k=1, namespace=namespace),
            QuerySpec(vector=[0.01] * dims, top_k=1, namespace=namespace),
        ],
    )

    if not getattr(caps, "supports_batch_queries", False):
        with pytest.raises(NotSupported):
            await adapter.batch_query(spec)
    else:
        res = await adapter.batch_query(spec)
        assert isinstance(res, list)
        assert len(res) == 2
