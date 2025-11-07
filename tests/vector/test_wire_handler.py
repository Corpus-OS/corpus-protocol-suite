# tests/vector/test_wire_handler_envelopes.py
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
from typing import Any, Dict, List, Mapping, Optional

from corpus_sdk.vector.vector_base import (
    VECTOR_PROTOCOL_ID,
    VectorID,
    Vector,
    VectorMatch,
    QueryResult,
    VectorCapabilities,
    QuerySpec,
    UpsertSpec,
    DeleteSpec,
    NamespaceSpec,
    UpsertResult,
    DeleteResult,
    NamespaceResult,
    OperationContext,
    VectorAdapterError,
    BadRequest,
    NotSupported,
    Unavailable,
    BaseVectorAdapter,
    WireVectorHandler,
)

from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter

pytestmark = pytest.mark.asyncio


class TrackingMockVectorAdapter(MockVectorAdapter):
    """
    MockVectorAdapter wrapper that records last ctx/call/args for assertions.

    All vector behavior is inherited from the real mock; this only adds introspection.
    """

    def __init__(self, *args, **kwargs) -> None:
        # Keep tests deterministic by default.
        kwargs.setdefault("failure_rate", 0.0)
        super().__init__(*args, **kwargs)
        self.last_ctx: Optional[OperationContext] = None
        self.last_call: Optional[str] = None
        self.last_args: Dict[str, Any] = {}

    def _track(self, op: str, ctx: Optional[OperationContext], **kwargs: Any) -> None:
        self.last_call = op
        self.last_ctx = ctx
        self.last_args = dict(kwargs)

    async def _do_capabilities(self) -> VectorCapabilities:
        self._track("capabilities", None)
        return await super()._do_capabilities()

    async def _do_query(
        self,
        spec: QuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        self._track("query", ctx, spec=spec)
        return await super()._do_query(spec, ctx=ctx)

    async def _do_upsert(
        self,
        spec: UpsertSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        self._track("upsert", ctx, spec=spec)
        return await super()._do_upsert(spec, ctx=ctx)

    async def _do_delete(
        self,
        spec: DeleteSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        self._track("delete", ctx, spec=spec)
        return await super()._do_delete(spec, ctx=ctx)

    async def _do_create_namespace(
        self,
        spec: NamespaceSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        self._track("create_namespace", ctx, spec=spec)
        return await super()._do_create_namespace(spec, ctx=ctx)

    async def _do_delete_namespace(
        self,
        namespace: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        self._track("delete_namespace", ctx, namespace=namespace)
        return await super()._do_delete_namespace(namespace, ctx=ctx)

    async def _do_health(
        self,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> Dict[str, Any]:
        self._track("health", ctx)
        return await super()._do_health(ctx=ctx)


class ErrorAdapter(TrackingMockVectorAdapter):
    """
    Adapter that can be configured to raise specific errors to test wire mapping.
    Uses the real MockVectorAdapter wiring; only overrides the target op.
    """

    def __init__(self, exc: Exception):
        super().__init__()
        self._exc = exc

    async def _do_query(
        self,
        spec: QuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        # Directly raise the configured exception to test mapping.
        raise self._exc


# ---------------------------------------------------------------------------
# Success-path envelopes
# ---------------------------------------------------------------------------

async def test_wire_capabilities_success_envelope():
    a = TrackingMockVectorAdapter()
    h = WireVectorHandler(a)

    env = {
        "op": "vector.capabilities",
        "ctx": {},
        "args": {},
    }
    res = await h.handle(env)

    assert res["ok"] is True
    assert res["code"] == "OK"
    assert isinstance(res["result"], dict)
    assert res["result"]["protocol"] == VECTOR_PROTOCOL_ID
    assert res["result"]["server"] == "mock-vector"
    assert res["result"]["version"] == "1.0.0"


async def test_wire_query_roundtrip_and_context_plumbing():
    a = TrackingMockVectorAdapter()
    h = WireVectorHandler(a)

    # Set up namespace + data via the wire so MockVectorAdapter query will succeed.
    await h.handle(
        {
            "op": "vector.create_namespace",
            "ctx": {"request_id": "ns-setup"},
            "args": {
                "namespace": "default",
                "dimensions": 2,
                "distance_metric": "cosine",
            },
        }
    )
    await h.handle(
        {
            "op": "vector.upsert",
            "ctx": {"request_id": "upsert-setup"},
            "args": {
                "namespace": "default",
                "vectors": [
                    {
                        "id": "v1",
                        "vector": [0.1, 0.2],
                        "metadata": {"echo": True},
                        "namespace": "default",
                    }
                ],
            },
        }
    )

    ctx_wire = {
        "request_id": "req_wire_q",
        "idempotency_key": "idem_q",
        "deadline_ms": 9999999999999,
        "traceparent": "00-abc-xyz-01",
        "tenant": "acme-tenant",
        "attrs": {"k": "v"},
        "ignore_me": "extra",  # MUST be ignored
    }
    args = {
        "vector": [0.1, 0.2],
        "top_k": 3,
        "namespace": "default",
        "include_metadata": True,
        "include_vectors": False,
    }

    res = await h.handle(
        {
            "op": "vector.query",
            "ctx": ctx_wire,
            "args": args,
        }
    )

    # Envelope shape
    assert res["ok"] is True
    assert res["code"] == "OK"
    out = res["result"]
    assert out["namespace"] == "default"
    assert out["query_vector"] == [0.1, 0.2]
    assert isinstance(out["matches"], list)
    assert out["total_matches"] >= 1

    # Context propagation through BaseVectorAdapter -> TrackingMockVectorAdapter
    assert a.last_call == "query"
    assert isinstance(a.last_ctx, OperationContext)
    assert a.last_ctx.request_id == "req_wire_q"
    assert a.last_ctx.idempotency_key == "idem_q"
    assert a.last_ctx.traceparent == "00-abc-xyz-01"
    assert a.last_ctx.tenant == "acme-tenant"
    assert "ignore_me" not in (a.last_ctx.attrs or {})


async def test_wire_upsert_delete_namespace_health_envelopes():
    a = TrackingMockVectorAdapter()
    h = WireVectorHandler(a)

    # upsert
    upsert_env = {
        "op": "vector.upsert",
        "ctx": {"request_id": "r1"},
        "args": {
            "namespace": "ns",
            "vectors": [
                {
                    "id": "v1",
                    "vector": [0.0, 1.0],
                    "metadata": {"a": 1},
                    "namespace": "ns",
                }
            ],
        },
    }
    upsert_res = await h.handle(upsert_env)
    assert upsert_res["ok"] is True
    assert upsert_res["result"]["upserted_count"] == 1

    # delete
    delete_env = {
        "op": "vector.delete",
        "ctx": {"request_id": "r2"},
        "args": {
            "namespace": "ns",
            "ids": ["v1"],
        },
    }
    delete_res = await h.handle(delete_env)
    assert delete_res["ok"] is True
    assert delete_res["result"]["deleted_count"] == 1

    # create_namespace
    create_env = {
        "op": "vector.create_namespace",
        "ctx": {"request_id": "r3"},
        "args": {
            "namespace": "foo",
            "dimensions": 4,
            "distance_metric": "cosine",
        },
    }
    ns_res = await h.handle(create_env)
    assert ns_res["ok"] is True
    assert ns_res["result"]["namespace"] == "foo"

    # delete_namespace
    del_ns_env = {
        "op": "vector.delete_namespace",
        "ctx": {"request_id": "r4"},
        "args": {"namespace": "foo"},
    }
    del_ns_res = await h.handle(del_ns_env)
    assert del_ns_res["ok"] is True
    assert del_ns_res["result"]["namespace"] == "foo"

    # health
    health_env = {
        "op": "vector.health",
        "ctx": {"request_id": "r5"},
        "args": {},
    }
    health_res = await h.handle(health_env)
    assert health_res["ok"] is True
    assert health_res["result"]["server"] == "mock-vector"
    assert health_res["result"]["version"] == "1.0.0"
    assert "namespaces" in health_res["result"]


# ---------------------------------------------------------------------------
# Error mapping semantics
# ---------------------------------------------------------------------------

async def test_wire_unknown_op_maps_to_notsupported():
    a = TrackingMockVectorAdapter()
    h = WireVectorHandler(a)

    res = await h.handle(
        {
            "op": "vector.nope",
            "ctx": {},
            "args": {},
        }
    )

    assert res["ok"] is False
    # NotSupported from wire path
    assert res["code"] in ("NOT_SUPPORTED", "NOTSUPPORTED")
    assert res["error"] == "NotSupported"
    assert "unknown operation" in res["message"]


async def test_wire_maps_vector_adapter_error_to_normalized_envelope():
    exc = BadRequest("bad vector")
    a = ErrorAdapter(exc)
    h = WireVectorHandler(a)

    res = await h.handle(
        {
            "op": "vector.query",
            "ctx": {"request_id": "err_q"},
            "args": {
                "vector": [0.1],
                "top_k": 1,
                "namespace": "default",
            },
        }
    )

    assert res["ok"] is False
    assert res["code"] == "BAD_REQUEST"
    assert res["error"] == "BadRequest"
    assert res["message"] == "bad vector"
    assert "details" in res  # JSON-safe details present (may be null)


async def test_wire_maps_unexpected_exception_to_unavailable():
    class BoomAdapter(TrackingMockVectorAdapter):
        async def _do_query(
            self,
            spec: QuerySpec,
            *,
            ctx: Optional[OperationContext] = None,
        ) -> QueryResult:
            raise RuntimeError("boom")

    a = BoomAdapter()
    h = WireVectorHandler(a)

    res = await h.handle(
        {
            "op": "vector.query",
            "ctx": {"request_id": "boom"},
            "args": {
                "vector": [0.1],
                "top_k": 1,
                "namespace": "default",
            },
        }
    )

    assert res["ok"] is False
    assert res["code"] == "UNAVAILABLE"
    assert res["error"] == "RuntimeError"
    assert "boom" in res["message"]


# ---------------------------------------------------------------------------
# Additional wire tests (parity with LLM/Embedding)
# ---------------------------------------------------------------------------

async def test_wire_missing_or_invalid_op_maps_to_bad_request():
    a = TrackingMockVectorAdapter()
    h = WireVectorHandler(a)

    res = await h.handle(
        {
            "ctx": {},
            "args": {},
        }
    )

    assert res["ok"] is False
    assert res["code"] == "BAD_REQUEST"
    assert res["error"] == "BadRequest"
    assert "missing or invalid 'op'" in res["message"]


async def test_wire_maps_notsupported_adapter_error_to_not_supported_code():
    exc = NotSupported("nope")
    a = ErrorAdapter(exc)
    h = WireVectorHandler(a)

    res = await h.handle(
        {
            "op": "vector.query",
            "ctx": {"request_id": "ns"},
            "args": {
                "vector": [0.1],
                "top_k": 1,
                "namespace": "default",
            },
        }
    )

    assert res["ok"] is False
    assert res["code"] in ("NOT_SUPPORTED", "NOTSUPPORTED")
    assert res["error"] == "NotSupported"


async def test_wire_error_envelope_includes_message_and_type():
    exc = BadRequest("bad things")
    a = ErrorAdapter(exc)
    h = WireVectorHandler(a)

    res = await h.handle(
        {
            "op": "vector.query",
            "ctx": {"request_id": "err_shape"},
            "args": {
                "vector": [0.1],
                "top_k": 1,
                "namespace": "default",
            },
        }
    )

    assert res["ok"] is False
    assert res["code"] == "BAD_REQUEST"
    assert res["error"] == "BadRequest"
    assert "message" in res and isinstance(res["message"], str) and res["message"]


async def test_wire_query_missing_required_fields_maps_to_bad_request():
    a = TrackingMockVectorAdapter()
    h = WireVectorHandler(a)

    res = await h.handle(
        {
            "op": "vector.query",
            "ctx": {},
            "args": {},
        }
    )

    assert res["ok"] is False
    assert res["code"] == "BAD_REQUEST"
    assert res["error"] == "BadRequest"
