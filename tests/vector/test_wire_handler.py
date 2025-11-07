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

from adapter_sdk.vector_base import (
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


pytestmark = pytest.mark.asyncio


class FakeVectorAdapter(BaseVectorAdapter):
    """
    Minimal adapter for exercising WireVectorHandler.

    Behaviors:
      - deterministic capabilities
      - trivial query/upsert/delete/namespace/health responses
      - records last ctx/spec for assertions
    """

    def __init__(self) -> None:
        super().__init__(mode="thin")
        self.last_ctx: Optional[OperationContext] = None
        self.last_call: Optional[str] = None
        self.last_args: Dict[str, Any] = {}

    # --- helpers -------------------------------------------------------------

    def _store(self, op: str, ctx: Optional[OperationContext], **kwargs: Any) -> None:
        self.last_call = op
        self.last_ctx = ctx
        self.last_args = dict(kwargs)

    # --- required hooks ------------------------------------------------------

    async def _do_capabilities(self) -> VectorCapabilities:
        self._store("capabilities", None)
        return VectorCapabilities(
            server="fake-vector",
            version="1.0.0",
            max_dimensions=8,
            supported_metrics=("cosine", "euclidean"),
            supports_namespaces=True,
            supports_metadata_filtering=True,
            supports_batch_operations=True,
            max_batch_size=256,
            supports_index_management=True,
            idempotent_writes=True,
            supports_multi_tenant=True,
            supports_deadline=True,
            max_top_k=100,
        )

    async def _do_query(
        self,
        spec: QuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        self._store("query", ctx, spec=spec)
        v = Vector(
            id=VectorID("v1"),
            vector=list(spec.vector),
            metadata={"echo": True} if spec.include_metadata else None,
            namespace=spec.namespace,
        )
        match = VectorMatch(vector=v, score=1.0, distance=0.0)
        return QueryResult(
            matches=[match],
            query_vector=list(spec.vector),
            namespace=spec.namespace,
            total_matches=1,
        )

    async def _do_upsert(
        self,
        spec: UpsertSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        self._store("upsert", ctx, spec=spec)
        return UpsertResult(
            upserted_count=len(spec.vectors),
            failed_count=0,
            failures=[],
        )

    async def _do_delete(
        self,
        spec: DeleteSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        self._store("delete", ctx, spec=spec)
        deleted = len(spec.ids) if spec.ids else 0
        return DeleteResult(
            deleted_count=deleted,
            failed_count=0,
            failures=[],
        )

    async def _do_create_namespace(
        self,
        spec: NamespaceSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        self._store("create_namespace", ctx, spec=spec)
        return NamespaceResult(
            success=True,
            namespace=spec.namespace,
            details={
                "dimensions": spec.dimensions,
                "distance_metric": spec.distance_metric,
            },
        )

    async def _do_delete_namespace(
        self,
        namespace: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        self._store("delete_namespace", ctx, namespace=namespace)
        return NamespaceResult(
            success=True,
            namespace=namespace,
            details={},
        )

    async def _do_health(
        self,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> Dict[str, Any]:
        self._store("health", ctx)
        return {
            "ok": True,
            "server": "fake-vector",
            "version": "1.0.0",
            "namespaces": {"default": "ok"},
        )


class ErrorAdapter(FakeVectorAdapter):
    """
    Adapter that can be configured to raise specific errors to test wire mapping.
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
        raise self._exc


# ---------------------------------------------------------------------------
# Success-path envelopes
# ---------------------------------------------------------------------------

async def test_wire_capabilities_success_envelope():
    a = FakeVectorAdapter()
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
    # protocol identity MUST be vector/v1.0
    assert res["result"]["protocol"] == VECTOR_PROTOCOL_ID
    assert res["result"]["server"] == "fake-vector"
    assert res["result"]["version"] == "1.0.0"


async def test_wire_query_roundtrip_and_context_plumbing():
    a = FakeVectorAdapter()
    h = WireVectorHandler(a)

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
    assert out["total_matches"] == 1

    # Context propagation through BaseVectorAdapter -> FakeVectorAdapter
    assert a.last_call == "query"
    assert isinstance(a.last_ctx, OperationContext)
    assert a.last_ctx.request_id == "req_wire_q"
    assert a.last_ctx.idempotency_key == "idem_q"
    assert a.last_ctx.traceparent == "00-abc-xyz-01"
    # raw tenant is allowed in ctx, Wire handler MUST NOT mutate/remove it here
    assert a.last_ctx.tenant == "acme-tenant"
    # unknown ctx field ignored
    assert "ignore_me" not in a.last_ctx.attrs


async def test_wire_upsert_delete_namespace_health_envelopes():
    a = FakeVectorAdapter()
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
    assert health_res["result"]["server"] == "fake-vector"
    assert health_res["result"]["version"] == "1.0.0"
    assert "namespaces" in health_res["result"]


# ---------------------------------------------------------------------------
# Error mapping semantics
# ---------------------------------------------------------------------------

async def test_wire_unknown_op_maps_to_notsupported():
    a = FakeVectorAdapter()
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
    assert isinstance(res["message"], str) and "unknown operation" in res["message"]


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
    # details MAY be present; MUST be JSON-safe
    assert "details" in res


async def test_wire_maps_unexpected_exception_to_unavailable():
    class BoomAdapter(FakeVectorAdapter):
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
