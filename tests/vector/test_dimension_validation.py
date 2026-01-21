# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Dimension validation & errors.

Spec refs:
  • SPECIFICATION.md §9.5 (Vector-Specific Errors)
  • SPECIFICATION.md §12.4 (Error Mapping)
"""

import json
import pytest
from typing import Optional

from corpus_sdk.vector.vector_base import (
    Vector,
    VectorID,
    UpsertSpec,
    QuerySpec,
    DimensionMismatch,
    NamespaceSpec,
    NotSupported,
    BadRequest,
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


def _json_safe(obj: object) -> None:
    json.dumps(obj)


async def test_dimension_validation_dimension_mismatch_on_upsert(adapter):
    """Verify DimensionMismatch is raised for invalid vector dimensions on upsert."""
    caps = await adapter.capabilities()
    bad_dimension = (caps.max_dimensions or 8) + 1

    # Namespace doesn't matter here because BaseVectorAdapter gates max_dimensions before backend.
    spec = UpsertSpec(
        namespace="default",
        vectors=[Vector(id=VectorID("bad-dimension"), vector=[0.0] * bad_dimension)],
    )

    with pytest.raises(DimensionMismatch) as exc_info:
        await adapter.upsert(spec)

    err = exc_info.value
    assert err.code == "DIMENSION_MISMATCH"


async def test_dimension_validation_dimension_mismatch_on_query(adapter):
    """Verify DimensionMismatch is raised for invalid vector dimensions on query."""
    caps = await adapter.capabilities()
    bad_dimension = (caps.max_dimensions or 8) + 1

    with pytest.raises(DimensionMismatch) as exc_info:
        await adapter.query(
            QuerySpec(
                vector=[0.0] * bad_dimension,
                top_k=1,
                namespace="default",
            )
        )

    err = exc_info.value
    assert err.code == "DIMENSION_MISMATCH"


async def test_dimension_validation_dimension_mismatch_error_attributes(adapter):
    """Verify DimensionMismatch error has correct attributes and no retry hint."""
    caps = await adapter.capabilities()
    bad_dimension = (caps.max_dimensions or 8) + 1

    with pytest.raises(DimensionMismatch) as exc_info:
        await adapter.query(QuerySpec(vector=[0.0] * bad_dimension, top_k=1, namespace="default"))

    err = exc_info.value
    assert err.code == "DIMENSION_MISMATCH"
    assert getattr(err, "retry_after_ms", None) is None
    assert err.message and isinstance(err.message, str)


async def test_dimension_validation_dimension_mismatch_non_retryable(adapter):
    """Verify DimensionMismatch errors are marked as non-retryable."""
    caps = await adapter.capabilities()
    bad_dimension = (caps.max_dimensions or 8) + 1

    with pytest.raises(DimensionMismatch) as exc_info:
        await adapter.query(QuerySpec(vector=[0.0] * bad_dimension, top_k=1, namespace="default"))

    err = exc_info.value
    assert err.retry_after_ms is None


# NEW
async def test_dimension_validation_exact_namespace_dimension_mismatch(adapter):
    """
    When namespaces have a declared schema dimension, providing a different dimension should raise DimensionMismatch.

    This test avoids max_dimensions gating by using a small, known dimension.
    """
    ns = "test-dim-schema"
    await _maybe_create_namespace(adapter, namespace=ns, dimensions=2, distance_metric="cosine")
    await adapter.upsert(UpsertSpec(namespace=ns, vectors=[Vector(id=VectorID("d_ok"), vector=[0.1, 0.2], namespace=ns)]))

    with pytest.raises(DimensionMismatch):
        await adapter.query(QuerySpec(vector=[0.1, 0.2, 0.3], top_k=1, namespace=ns))


# NEW
async def test_dimension_validation_dimension_mismatch_asdict_is_json_serializable(adapter):
    """DimensionMismatch.asdict() must remain SIEM-safe (JSON serializable)."""
    caps = await adapter.capabilities()
    bad_dimension = (caps.max_dimensions or 8) + 1
    with pytest.raises(DimensionMismatch) as exc_info:
        await adapter.query(QuerySpec(vector=[0.0] * bad_dimension, top_k=1, namespace="default"))
    _json_safe(exc_info.value.asdict())
