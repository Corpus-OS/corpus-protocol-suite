# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Delete operations.

Spec refs:
  • SPECIFICATION.md §9.3 (delete)
  • SPECIFICATION.md §12.5 (Partial Failures)
"""

import json
import pytest
from typing import Optional, Dict, Any

from corpus_sdk.vector.vector_base import (
    Vector,
    VectorID,
    UpsertSpec,
    DeleteSpec,
    DeleteResult,
    BadRequest,
    NotSupported,
    NamespaceSpec,
)

pytestmark = pytest.mark.asyncio


# ---------------------------- shared test helpers ---------------------------- #
# NOTE: These helpers only use the public protocol surface (capabilities + ops).
# They are designed to be tolerant of provider variance while remaining strict
# about capability ↔ behavior consistency.


async def _maybe_create_namespace(
    adapter,
    *,
    namespace: str,
    dimensions: int,
    distance_metric: str = "cosine",
) -> None:
    """
    Ensure a namespace exists if the adapter advertises index management support.

    Rules:
      - If caps.supports_index_management is True, create_namespace MUST succeed
        (or be idempotent / already-exists).
      - If caps.supports_index_management is False, we do not attempt creation.
    """
    caps = await adapter.capabilities()
    if not getattr(caps, "supports_index_management", False):
        return

    try:
        res = await adapter.create_namespace(
            NamespaceSpec(namespace=namespace, dimensions=dimensions, distance_metric=distance_metric)
        )
        if hasattr(res, "success"):
            assert bool(res.success) is True
    except NotSupported as e:
        raise AssertionError(
            "capabilities.supports_index_management=True but create_namespace raised NotSupported"
        ) from e
    except BadRequest as e:
        # Some backends treat re-creation as a client error; allow "already exists" class errors.
        msg = str(e).lower()
        if any(tok in msg for tok in ("already", "exists", "exist")):
            return
        raise


async def _ensure_namespace_ready(
    adapter,
    *,
    namespace: str,
    dimensions: int,
    distance_metric: str = "cosine",
    seed_id: str = "seed",
    seed_vector: float = 0.1,
    seed_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Ensure a namespace is usable for query/delete by:
      - creating it if supported
      - upserting at least one vector to avoid IndexNotReady behaviors on some adapters
    """
    await _maybe_create_namespace(adapter, namespace=namespace, dimensions=dimensions, distance_metric=distance_metric)

    v = Vector(
        id=VectorID(seed_id),
        vector=[float(seed_vector)] * int(dimensions),
        metadata=seed_metadata,
        namespace=namespace,
    )
    await adapter.upsert(UpsertSpec(namespace=namespace, vectors=[v]))
    return namespace


def _json_safe(obj: object) -> None:
    """Fail-fast SIEM-safety check: the object must be JSON-serializable."""
    json.dumps(obj)


# ------------------------------- delete tests -------------------------------- #


async def test_delete_delete_by_ids_returns_counts(adapter):
    """Verify delete by IDs returns proper counts and result structure."""
    ns = "test-delete-ids"
    await _ensure_namespace_ready(adapter, namespace=ns, dimensions=2, seed_id="d1", seed_vector=0.2)

    result = await adapter.delete(DeleteSpec(namespace=ns, ids=[VectorID("d1")]))

    assert isinstance(result, DeleteResult)
    assert isinstance(result.deleted_count, int) and result.deleted_count >= 0
    assert isinstance(result.failures, list)


async def test_delete_delete_by_filter_returns_counts(adapter):
    """Verify delete by filter returns proper result structure."""
    ns = "test-delete-filter"
    await _ensure_namespace_ready(
        adapter,
        namespace=ns,
        dimensions=2,
        seed_id="df1",
        seed_vector=0.3,
        seed_metadata={"foo": "bar"},
    )

    caps = await adapter.capabilities()
    spec = DeleteSpec(namespace=ns, ids=[], filter={"foo": "bar"})

    if not caps.supports_metadata_filtering:
        with pytest.raises(NotSupported):
            await adapter.delete(spec)
        return

    result = await adapter.delete(spec)
    assert isinstance(result, DeleteResult)
    assert isinstance(result.deleted_count, int) and result.deleted_count >= 0
    assert isinstance(result.failures, list)


async def test_delete_requires_ids_or_filter(adapter):
    """Verify delete requires either IDs or filter parameter."""
    # Use a real namespace to avoid "unknown namespace" masking the real validation.
    ns = "test-delete-requires"
    await _ensure_namespace_ready(adapter, namespace=ns, dimensions=2, seed_id="dr1", seed_vector=0.21)

    spec = DeleteSpec(namespace=ns, ids=[], filter=None)

    with pytest.raises(BadRequest) as exc_info:
        await adapter.delete(spec)

    err = exc_info.value
    assert "ids" in str(err).lower() or "filter" in str(err).lower()


async def test_delete_idempotent_for_missing_ids(adapter):
    """Verify delete operations are idempotent for non-existent IDs."""
    ns = "test-delete-idempotent"
    await _ensure_namespace_ready(adapter, namespace=ns, dimensions=2, seed_id="keep", seed_vector=0.4)

    spec = DeleteSpec(namespace=ns, ids=[VectorID("non-existent-id")])

    result1 = await adapter.delete(spec)
    result2 = await adapter.delete(spec)

    assert result1.deleted_count >= 0
    assert result2.deleted_count >= 0
    assert result1.deleted_count == result2.deleted_count


async def test_delete_delete_result_structure(adapter):
    """Verify DeleteResult has all required fields with proper types."""
    ns = "test-delete-structure"
    await _ensure_namespace_ready(adapter, namespace=ns, dimensions=2, seed_id="ds1", seed_vector=0.5)

    spec = DeleteSpec(namespace=ns, ids=[VectorID("ds1")])
    result = await adapter.delete(spec)

    assert hasattr(result, "deleted_count") and isinstance(result.deleted_count, int)
    assert hasattr(result, "failed_count") and isinstance(result.failed_count, int)
    assert hasattr(result, "failures") and isinstance(result.failures, list)


# NEW
async def test_delete_filter_not_supported_raises_notsupported_if_capability_false(adapter):
    """If caps say filtering unsupported, delete(filter=...) must raise NotSupported."""
    ns = "test-delete-filter-cap"
    await _ensure_namespace_ready(
        adapter,
        namespace=ns,
        dimensions=2,
        seed_id="dfcap",
        seed_vector=0.6,
        seed_metadata={"status": "obsolete"},
    )

    caps = await adapter.capabilities()
    spec = DeleteSpec(namespace=ns, ids=[], filter={"status": "obsolete"})

    if caps.supports_metadata_filtering:
        res = await adapter.delete(spec)
        assert isinstance(res, DeleteResult)
    else:
        with pytest.raises(NotSupported):
            await adapter.delete(spec)


# NEW
async def test_delete_batch_ids_respects_supports_batch_operations(adapter):
    """
    If supports_batch_operations is false, multi-id deletes must raise NotSupported.
    If true, operation must succeed and return a DeleteResult.
    """
    ns = "test-delete-batch-ops"
    await _ensure_namespace_ready(adapter, namespace=ns, dimensions=2, seed_id="b1", seed_vector=0.7)
    await adapter.upsert(
        UpsertSpec(
            namespace=ns,
            vectors=[Vector(id=VectorID("b2"), vector=[0.7, 0.7], namespace=ns)],
        )
    )

    caps = await adapter.capabilities()
    spec = DeleteSpec(namespace=ns, ids=[VectorID("b1"), VectorID("b2")])

    if caps.supports_batch_operations:
        res = await adapter.delete(spec)
        assert isinstance(res, DeleteResult)
    else:
        with pytest.raises(NotSupported):
            await adapter.delete(spec)


# NEW
async def test_delete_exceed_max_batch_size_raises_badrequest_when_declared(adapter):
    """
    If caps.max_batch_size is declared, delete(ids=...) with > max_batch_size must raise BadRequest.
    If max_batch_size is not declared, at least a small delete should succeed.
    """
    ns = "test-delete-max-batch"
    await _ensure_namespace_ready(adapter, namespace=ns, dimensions=2, seed_id="mb0", seed_vector=0.8)

    caps = await adapter.capabilities()
    if caps.max_batch_size is None:
        res = await adapter.delete(DeleteSpec(namespace=ns, ids=[VectorID("mb0")]))
        assert isinstance(res, DeleteResult)
        return

    too_many = int(caps.max_batch_size) + 1
    ids = [VectorID(f"mb{i}") for i in range(too_many)]
    with pytest.raises(BadRequest):
        await adapter.delete(DeleteSpec(namespace=ns, ids=ids))
