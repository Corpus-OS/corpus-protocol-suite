# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Error taxonomy & retry hints.

Spec refs:
  • SPECIFICATION.md §6.3 (Error Taxonomy)
  • SPECIFICATION.md §9.5 (Vector-Specific Errors)
  • SPECIFICATION.md §12.1, §12.4 (Retry Semantics, Mapping)

Goal:
  - Deterministically validate error codes + retry_after_ms semantics
  - Validate raised (runtime) retryable errors preserve retry hints
  - Validate wire error envelopes preserve retry hints
  - Avoid relying on any provider-specific knobs or randomness
"""

import json
import pytest
from typing import Any, Dict, Optional

from corpus_sdk.vector.vector_base import (
    VECTOR_PROTOCOL_ID,
    BaseVectorAdapter,
    WireVectorHandler,
    VectorCapabilities,
    QuerySpec,
    UpsertSpec,
    OperationContext,
    VectorAdapterError,
    BadRequest,
    DimensionMismatch,
    IndexNotReady,
    NotSupported,
    ResourceExhausted,
    TransientNetwork,
    Unavailable,
)

pytestmark = pytest.mark.asyncio


def _json_safe(obj: object) -> None:
    """Fail-fast SIEM-safety check: object must be JSON-serializable."""
    json.dumps(obj)


class RaisingAdapter(BaseVectorAdapter):
    """
    Deterministic adapter that raises a supplied exception from _do_query().
    Used to test "raised error preserves retry_after_ms" and wire mapping.
    """

    def __init__(self, exc: Exception):
        super().__init__()
        self._exc = exc

    async def _do_capabilities(self) -> VectorCapabilities:
        # Keep capabilities minimal but valid for BaseVectorAdapter gating.
        return VectorCapabilities(
            protocol=VECTOR_PROTOCOL_ID,
            server="raising-adapter",
            version="1.0.0",
            max_dimensions=8,
            supported_metrics=("cosine",),
            supports_namespaces=True,
            supports_metadata_filtering=True,
            supports_batch_operations=True,
            supports_batch_queries=True,
            supports_index_management=True,
            max_top_k=1000,
            max_filter_terms=10,
            text_storage_strategy="none",
        )

    async def _do_query(self, spec, *, ctx: Optional[OperationContext] = None):
        raise self._exc

    async def _do_batch_query(self, spec, *, ctx: Optional[OperationContext] = None):
        raise NotSupported("batch_query not implemented for RaisingAdapter")

    async def _do_upsert(self, spec, *, ctx: Optional[OperationContext] = None):
        raise NotSupported("upsert not implemented for RaisingAdapter")

    async def _do_delete(self, spec, *, ctx: Optional[OperationContext] = None):
        raise NotSupported("delete not implemented for RaisingAdapter")

    async def _do_create_namespace(self, spec, *, ctx: Optional[OperationContext] = None):
        raise NotSupported("create_namespace not implemented for RaisingAdapter")

    async def _do_delete_namespace(self, namespace: str, *, ctx: Optional[OperationContext] = None):
        raise NotSupported("delete_namespace not implemented for RaisingAdapter")

    async def _do_health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
        return {"ok": True, "server": "raising-adapter", "version": "1.0.0", "namespaces": {}}


# ---------------------------------------------------------------------------
# Class-level semantics (pure, deterministic)
# ---------------------------------------------------------------------------

async def test_error_handling_retryable_errors_with_hints():
    """Verify retryable errors include proper retry hints."""
    rate_limit_error = ResourceExhausted("rate limit exceeded", retry_after_ms=1000)
    assert rate_limit_error.code == "RESOURCE_EXHAUSTED"
    assert rate_limit_error.retry_after_ms == 1000

    unavailable_error = Unavailable("backend unavailable", retry_after_ms=500)
    assert unavailable_error.code == "UNAVAILABLE"
    assert unavailable_error.retry_after_ms == 500


async def test_error_handling_index_not_ready_retryable():
    """Verify IndexNotReady errors are retryable with hints."""
    index_error = IndexNotReady("index warming up", retry_after_ms=200)
    assert index_error.code == "INDEX_NOT_READY"
    assert index_error.retry_after_ms == 200


async def test_error_handling_dimension_mismatch_non_retryable_flag():
    """Verify DimensionMismatch errors are non-retryable."""
    dimension_error = DimensionMismatch("vector dimension mismatch")
    assert dimension_error.code == "DIMENSION_MISMATCH"
    assert dimension_error.retry_after_ms is None


async def test_error_handling_error_has_siem_safe_details():
    """Verify error details are SIEM-safe (JSON serializable)."""
    error = ResourceExhausted("rate limited", details={"scope": "global", "limit": 1000})
    error_dict = error.asdict()

    assert "details" in error_dict
    assert isinstance(error_dict["details"], dict)
    _json_safe(error_dict)


# ---------------------------------------------------------------------------
# Raised error semantics (deterministic; no provider knobs)
# ---------------------------------------------------------------------------

async def test_error_handling_retry_after_preserved_when_raised_resource_exhausted():
    """NEW: Raising ResourceExhausted preserves retry_after_ms on the exception."""
    exc = ResourceExhausted("rate limited", retry_after_ms=321)
    adapter = RaisingAdapter(exc)

    with pytest.raises(ResourceExhausted) as exc_info:
        await adapter.query(QuerySpec(vector=[0.1, 0.2], top_k=1, namespace="default"))

    err = exc_info.value
    assert err.code == "RESOURCE_EXHAUSTED"
    assert err.retry_after_ms == 321


async def test_error_handling_retry_after_preserved_when_raised_unavailable():
    """NEW: Raising Unavailable preserves retry_after_ms on the exception."""
    exc = Unavailable("overloaded", retry_after_ms=123)
    adapter = RaisingAdapter(exc)

    with pytest.raises(Unavailable) as exc_info:
        await adapter.query(QuerySpec(vector=[0.1, 0.2], top_k=1, namespace="default"))

    err = exc_info.value
    assert err.code == "UNAVAILABLE"
    assert err.retry_after_ms == 123


async def test_error_handling_retry_after_preserved_when_raised_index_not_ready():
    """NEW: Raising IndexNotReady preserves retry_after_ms on the exception."""
    exc = IndexNotReady("warming", retry_after_ms=50)
    adapter = RaisingAdapter(exc)

    with pytest.raises(IndexNotReady) as exc_info:
        await adapter.query(QuerySpec(vector=[0.1, 0.2], top_k=1, namespace="default"))

    err = exc_info.value
    assert err.code == "INDEX_NOT_READY"
    assert err.retry_after_ms == 50


async def test_error_handling_retry_after_preserved_when_raised_transient_network():
    """NEW: Raising TransientNetwork preserves retry_after_ms on the exception."""
    exc = TransientNetwork("network glitch", retry_after_ms=75)
    adapter = RaisingAdapter(exc)

    with pytest.raises(TransientNetwork) as exc_info:
        await adapter.query(QuerySpec(vector=[0.1, 0.2], top_k=1, namespace="default"))

    err = exc_info.value
    assert err.code == "TRANSIENT_NETWORK"
    assert err.retry_after_ms == 75


# ---------------------------------------------------------------------------
# Adapter-path shape checks using the shared fixture adapter
# ---------------------------------------------------------------------------

async def test_error_handling_bad_request_on_invalid_top_k(adapter):
    """Verify BadRequest is raised for invalid top_k values."""
    with pytest.raises(BadRequest) as exc_info:
        await adapter.query(QuerySpec(vector=[0.1], top_k=0, namespace="default"))

    err = exc_info.value
    assert err.code == "BAD_REQUEST"
    assert "top_k" in str(err).lower() or "invalid" in str(err).lower()


async def test_error_handling_retry_after_field_exists_on_adapter_errors(adapter):
    """
    Ensure adapter-raised VectorAdapterError always has retry_after_ms attribute with stable type.
    This is deterministic by triggering a BadRequest (top_k=-1).
    """
    with pytest.raises(VectorAdapterError) as exc_info:
        await adapter.query(QuerySpec(vector=[0.1], top_k=-1, namespace="default"))

    err = exc_info.value
    assert hasattr(err, "retry_after_ms")
    ra = getattr(err, "retry_after_ms", None)
    assert (ra is None) or isinstance(ra, int)


async def test_error_handling_upsert_bad_request_message_siem_safe(adapter):
    """Ensure raised errors have SIEM-safe asdict() output and a string message."""
    with pytest.raises(BadRequest) as exc_info:
        await adapter.upsert(UpsertSpec(namespace="default", vectors=[]))

    err = exc_info.value
    assert isinstance(err.message, str)
    _json_safe(err.asdict())


# ---------------------------------------------------------------------------
# Wire mapping (deterministic)
# ---------------------------------------------------------------------------

async def test_wire_retry_after_propagates_in_error_envelope():
    """NEW: retry_after_ms must be preserved in the wire error envelope for retryable errors."""
    adapter = RaisingAdapter(Unavailable("overloaded", retry_after_ms=444))
    handler = WireVectorHandler(adapter)

    r = await handler.handle(
        {
            "op": "vector.query",
            "ctx": {"request_id": "wire-ra"},
            "args": {"vector": [0.1, 0.2], "top_k": 1, "namespace": "default"},
        }
    )

    assert r["ok"] is False
    assert r["code"] == "UNAVAILABLE"
    assert r["error"] == "Unavailable"
    assert r["retry_after_ms"] == 444
    assert "ms" in r and isinstance(r["ms"], (int, float))
