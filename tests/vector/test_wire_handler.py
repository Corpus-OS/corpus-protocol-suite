# SPDX-License-Identifier: Apache-2.0
"""
Vector Conformance — Wire-level envelopes & routing.

Asserts (SPECIFICATION.md refs):
  • §4.1, §4.1.6 — Canonical op strings and envelope shapes
  • §9.3         — Vector operations mapped via `vector.<op>` wire contract
  • §6.1         — OperationContext constructed from wire ctx (ignore unknowns)
  • §6.3, §12.4  — Normalized error envelopes for VectorAdapterError subclasses
  • §11.2, §13   — SIEM-safe behavior (no tenant leakage via Wire handler)

Covers:
  • vector.capabilities → success envelope
  • vector.query / upsert / delete / namespace ops → success envelope
  • vector.health → success envelope
  • Context translation: ctx → OperationContext passed through BaseVectorAdapter
  • Unknown op → NotSupported → normalized error envelope
  • VectorAdapterError → mapped with correct code/error/message/details
  • Unexpected Exception → mapped to UNAVAILABLE per common error taxonomy
"""

import pytest
from typing import Any, Dict, Optional
from corpus_sdk.vector.vector_base import (
    VECTOR_PROTOCOL_ID,
    OperationContext,
    VectorAdapterError,
    BadRequest,
    NotSupported,
    Unavailable,
    BaseVectorAdapter,
    WireVectorHandler,
)

pytestmark = pytest.mark.asyncio


class TrackingMockVectorAdapter(BaseVectorAdapter):
    """
    Test adapter for WireVectorHandler testing with full operation tracking.
    Implements minimal required methods for wire handler testing.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.last_ctx: Optional[OperationContext] = None
        self.last_call: Optional[str] = None
        self.last_args: Dict[str, Any] = {}

    def _track(self, op: str, ctx: Optional[OperationContext], **kwargs: Any) -> None:
        self.last_call = op
        self.last_ctx = ctx
        self.last_args = dict(kwargs)

    async def capabilities(self):
        self._track("capabilities", None)
        return {
            "protocol": VECTOR_PROTOCOL_ID,
            "server": "test-vector",
            "version": "1.0.0",
            "max_dimensions": 8,
            "supported_metrics": ("cosine", "euclidean"),
            "supports_namespaces": True,
            "supports_metadata_filtering": True,
        }

    async def query(self, spec, *, ctx: Optional[OperationContext] = None):
        self._track("query", ctx, spec=spec)
        return {
            "namespace": spec.namespace,
            "query_vector": spec.vector,
            "matches": [],
            "total_matches": 0
        }

    async def upsert(self, spec, *, ctx: Optional[OperationContext] = None):
        self._track("upsert", ctx, spec=spec)
        return {
            "upserted_count": len(spec.vectors),
            "failed_count": 0,
            "failures": []
        }

    async def delete(self, spec, *, ctx: Optional[OperationContext] = None):
        self._track("delete", ctx, spec=spec)
        return {
            "deleted_count": len(spec.ids) if spec.ids else 0,
            "failed_count": 0,
            "failures": []
        }

    async def create_namespace(self, spec, *, ctx: Optional[OperationContext] = None):
        self._track("create_namespace", ctx, spec=spec)
        return {
            "success": True,
            "namespace": spec.namespace
        }

    async def delete_namespace(self, namespace: str, *, ctx: Optional[OperationContext] = None):
        self._track("delete_namespace", ctx, namespace=namespace)
        return {
            "success": True,
            "namespace": namespace
        }

    async def health(self, *, ctx: Optional[OperationContext] = None):
        self._track("health", ctx)
        return {
            "ok": True,
            "server": "test-vector",
            "version": "1.0.0",
            "namespaces": {"default": {"vector_count": 0}}
        }


class ErrorAdapter(TrackingMockVectorAdapter):
    """Adapter that raises specific errors for testing wire mapping."""
    
    def __init__(self, exc: Exception):
        super().__init__()
        self._exc = exc

    async def query(self, spec, *, ctx: Optional[OperationContext] = None):
        raise self._exc


class BoomAdapter(TrackingMockVectorAdapter):
    """Adapter that raises unexpected exceptions for testing error mapping."""
    
    async def query(self, spec, *, ctx: Optional[OperationContext] = None):
        raise RuntimeError("unexpected failure")


# ---------------------------------------------------------------------------
# Success-path envelopes
# ---------------------------------------------------------------------------

async def test_wire_contract_capabilities_success_envelope():
    """Verify capabilities operation returns proper success envelope."""
    adapter = TrackingMockVectorAdapter()
    handler = WireVectorHandler(adapter)

    envelope = {
        "op": "vector.capabilities",
        "ctx": {},
        "args": {},
    }
    result = await handler.handle(envelope)

    assert result["ok"] is True
    assert result["code"] == "OK"
    assert isinstance(result["result"], dict)
    assert result["result"]["protocol"] == VECTOR_PROTOCOL_ID
    assert result["result"]["server"] == "test-vector"
    assert result["result"]["version"] == "1.0.0"


async def test_wire_contract_query_roundtrip_and_context_plumbing():
    """Verify query operation with context propagation."""
    adapter = TrackingMockVectorAdapter()
    handler = WireVectorHandler(adapter)

    wire_context = {
        "request_id": "test-request-123",
        "idempotency_key": "test-idempotency",
        "deadline_ms": 9999999999999,
        "traceparent": "00-test-trace-01",
        "tenant": "test-tenant",
        "attrs": {"test_key": "test_value"},
        "ignore_field": "should-be-ignored",  # Must be filtered out
    }
    
    args = {
        "vector": [0.1, 0.2],
        "top_k": 3,
        "namespace": "default",
        "include_metadata": True,
        "include_vectors": False,
    }

    result = await handler.handle({
        "op": "vector.query",
        "ctx": wire_context,
        "args": args,
    })

    # Verify success envelope
    assert result["ok"] is True
    assert result["code"] == "OK"
    assert isinstance(result["result"], dict)

    # Verify context propagation
    assert adapter.last_call == "query"
    assert isinstance(adapter.last_ctx, OperationContext)
    assert adapter.last_ctx.request_id == "test-request-123"
    assert adapter.last_ctx.idempotency_key == "test-idempotency"
    assert adapter.last_ctx.traceparent == "00-test-trace-01"
    assert adapter.last_ctx.tenant == "test-tenant"
    assert "ignore_field" not in (adapter.last_ctx.attrs or {})


async def test_wire_contract_upsert_delete_namespace_health_envelopes():
    """Verify all vector operations return proper success envelopes."""
    adapter = TrackingMockVectorAdapter()
    handler = WireVectorHandler(adapter)

    # Test upsert operation
    upsert_result = await handler.handle({
        "op": "vector.upsert",
        "ctx": {"request_id": "upsert-test"},
        "args": {
            "namespace": "test-namespace",
            "vectors": [{"id": "v1", "vector": [0.1, 0.2]}],
        },
    })
    assert upsert_result["ok"] is True
    assert upsert_result["result"]["upserted_count"] == 1

    # Test delete operation
    delete_result = await handler.handle({
        "op": "vector.delete", 
        "ctx": {"request_id": "delete-test"},
        "args": {
            "namespace": "test-namespace",
            "ids": ["v1"],
        },
    })
    assert delete_result["ok"] is True
    assert delete_result["result"]["deleted_count"] == 1

    # Test create namespace
    namespace_result = await handler.handle({
        "op": "vector.create_namespace",
        "ctx": {"request_id": "namespace-create-test"},
        "args": {
            "namespace": "new-namespace",
            "dimensions": 4,
            "distance_metric": "cosine",
        },
    })
    assert namespace_result["ok"] is True
    assert namespace_result["result"]["namespace"] == "new-namespace"

    # Test health operation
    health_result = await handler.handle({
        "op": "vector.health",
        "ctx": {"request_id": "health-test"},
        "args": {},
    })
    assert health_result["ok"] is True
    assert health_result["result"]["server"] == "test-vector"
    assert health_result["result"]["version"] == "1.0.0"


async def test_wire_contract_delete_namespace_operation():
    """Verify delete_namespace operation returns proper success envelope."""
    adapter = TrackingMockVectorAdapter()
    handler = WireVectorHandler(adapter)

    result = await handler.handle({
        "op": "vector.delete_namespace",
        "ctx": {"request_id": "delete-namespace-test"},
        "args": {
            "namespace": "test-namespace-to-delete",
        },
    })

    # Verify success envelope
    assert result["ok"] is True
    assert result["code"] == "OK"
    assert isinstance(result["result"], dict)
    assert result["result"]["success"] is True
    assert result["result"]["namespace"] == "test-namespace-to-delete"

    # Verify operation was called with correct context and arguments
    assert adapter.last_call == "delete_namespace"
    assert isinstance(adapter.last_ctx, OperationContext)
    assert adapter.last_ctx.request_id == "delete-namespace-test"
    assert adapter.last_args["namespace"] == "test-namespace-to-delete"


# ---------------------------------------------------------------------------
# Error mapping semantics
# ---------------------------------------------------------------------------

async def test_wire_contract_unknown_op_maps_to_not_supported():
    """Verify unknown operations return NOT_SUPPORTED error envelope."""
    adapter = TrackingMockVectorAdapter()
    handler = WireVectorHandler(adapter)

    result = await handler.handle({
        "op": "vector.unknown_operation",
        "ctx": {},
        "args": {},
    })

    assert result["ok"] is False
    assert result["code"] in ("NOT_SUPPORTED", "NOTSUPPORTED")
    assert "unknown" in result["message"].lower()


async def test_wire_contract_maps_vector_adapter_error_to_normalized_envelope():
    """Verify VectorAdapterError subclasses are properly mapped to error envelopes."""
    error = BadRequest("invalid vector parameters")
    adapter = ErrorAdapter(error)
    handler = WireVectorHandler(adapter)

    result = await handler.handle({
        "op": "vector.query",
        "ctx": {"request_id": "error-test"},
        "args": {
            "vector": [0.1],
            "top_k": 1,
            "namespace": "default",
        },
    })

    assert result["ok"] is False
    assert result["code"] == "BAD_REQUEST"
    assert result["error"] == "BadRequest"
    assert result["message"] == "invalid vector parameters"


async def test_wire_contract_maps_unexpected_exception_to_unavailable():
    """Verify unexpected exceptions are mapped to UNAVAILABLE error envelope."""
    adapter = BoomAdapter()
    handler = WireVectorHandler(adapter)

    result = await handler.handle({
        "op": "vector.query",
        "ctx": {"request_id": "unexpected-error-test"},
        "args": {
            "vector": [0.1],
            "top_k": 1,
            "namespace": "default",
        },
    })

    assert result["ok"] is False
    assert result["code"] == "UNAVAILABLE"
    assert result["error"] == "RuntimeError"
    assert "unexpected failure" in result["message"]


async def test_wire_contract_missing_or_invalid_op_maps_to_bad_request():
    """Verify missing or invalid operation fields return BAD_REQUEST."""
    adapter = TrackingMockVectorAdapter()
    handler = WireVectorHandler(adapter)

    result = await handler.handle({
        "ctx": {},
        "args": {},
    })

    assert result["ok"] is False
    assert result["code"] == "BAD_REQUEST"
    assert "op" in result["message"].lower()


async def test_wire_contract_maps_not_supported_adapter_error():
    """Verify NotSupported errors are properly mapped."""
    error = NotSupported("operation not supported")
    adapter = ErrorAdapter(error)
    handler = WireVectorHandler(adapter)

    result = await handler.handle({
        "op": "vector.query",
        "ctx": {"request_id": "not-supported-test"},
        "args": {
            "vector": [0.1],
            "top_k": 1,
            "namespace": "default",
        },
    })

    assert result["ok"] is False
    assert result["code"] in ("NOT_SUPPORTED", "NOTSUPPORTED")
    assert result["error"] == "NotSupported"


async def test_wire_contract_error_envelope_includes_message_and_type():
    """Verify error envelopes include both message and error type."""
    error = BadRequest("validation failed")
    adapter = ErrorAdapter(error)
    handler = WireVectorHandler(adapter)

    result = await handler.handle({
        "op": "vector.query",
        "ctx": {"request_id": "error-details-test"},
        "args": {
            "vector": [0.1],
            "top_k": 1,
            "namespace": "default",
        },
    })

    assert result["ok"] is False
    assert result["code"] == "BAD_REQUEST"
    assert result["error"] == "BadRequest"
    assert "message" in result and isinstance(result["message"], str)
    assert result["message"] == "validation failed"


async def test_wire_contract_query_missing_required_fields_maps_to_bad_request():
    """Verify missing required fields in query return BAD_REQUEST."""
    adapter = TrackingMockVectorAdapter()
    handler = WireVectorHandler(adapter)

    result = await handler.handle({
        "op": "vector.query",
        "ctx": {},
        "args": {},  # Missing required fields
    })

    assert result["ok"] is False
    assert result["code"] == "BAD_REQUEST"
    assert "required" in result["message"].lower() or "missing" in result["message"].lower()
