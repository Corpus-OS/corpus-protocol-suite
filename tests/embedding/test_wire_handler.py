# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Wire handler canonical envelopes.

Spec refs:
  • §4 Wire Contract (Embedding) — canonical envelope shapes
  • §10.3 Embedding Operations (embed, embed_batch, count_tokens, health)
  • §10.4 Error Mapping — normalized codes and error payloads

Covers:
  • Successful envelopes for all supported ops
  • Canonical {ok, code, result, error, message} shape
  • Argument validation surfaced as BAD_REQUEST via wire
  • Unsupported / unknown ops surfaced as NOT_SUPPORTED
  • Model-not-available mapped to MODEL_NOT_AVAILABLE
  • Batch semantics & failures preserved through envelope
  • Context propagation via OperationContext
  • Unexpected Exception → UNAVAILABLE mapping
"""

import pytest

from corpus_sdk.embedding.embedding_base import (
    WireEmbeddingHandler,
    EmbedSpec,
    BatchEmbedSpec,
    BadRequest,
    NotSupported,
    ModelNotAvailable,
    TextTooLong,
    OperationContext,
)
from tests.mock.mock_embedding_adapter import MockEmbeddingAdapter

pytestmark = pytest.mark.asyncio


def _assert_ok_envelope(out):
    assert isinstance(out, dict)
    assert out.get("ok") is True
    assert isinstance(out.get("code"), str)
    assert out["code"] == "OK"
    # Success envelopes should not carry an error
    assert "error" not in out or out["error"] in (None, {})


def _assert_error_envelope(out, *, code: str | None = None):
    assert isinstance(out, dict)
    assert out.get("ok") is False

    # Should not include a non-empty result payload on errors
    assert "result" not in out or out["result"] in (None, {})

    # code must be a non-empty string
    assert "code" in out and isinstance(out["code"], str) and out["code"]

    # error should be the error type name as a string
    assert "error" in out and isinstance(out["error"], str) and out["error"]

    # message must be present (may be empty string for some errors)
    assert "message" in out and isinstance(out["message"], str)

    if code is not None:
        assert out["code"] == code


# ---------------------------------------------------------------------------
# Tracking adapter for context & call assertions
# ---------------------------------------------------------------------------

class TrackingMockEmbeddingAdapter(MockEmbeddingAdapter):
    """
    MockEmbeddingAdapter wrapper that records last ctx/call/args.

    Uses the real mock implementation; only adds observability and
    forces deterministic behavior via failure_rate=0.0.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("failure_rate", 0.0)
        super().__init__(*args, **kwargs)
        self.last_ctx = None
        self.last_call = None
        self.last_args = None

    def _store(self, op: str, ctx: OperationContext | None, **kwargs):
        self.last_call = op
        self.last_ctx = ctx
        self.last_args = dict(kwargs)

    async def _do_capabilities(self):
        self._store("capabilities", None)
        return await super()._do_capabilities()

    async def _do_embed(
        self,
        spec: EmbedSpec,
        *,
        ctx: OperationContext | None = None,
    ):
        self._store("embed", ctx, spec=spec)
        return await super()._do_embed(spec, ctx=ctx)

    async def _do_embed_batch(
        self,
        spec: BatchEmbedSpec,
        *,
        ctx: OperationContext | None = None,
    ):
        self._store("embed_batch", ctx, spec=spec)
        return await super()._do_embed_batch(spec, ctx=ctx)

    async def _do_count_tokens(
        self,
        text: str,
        model: str,
        *,
        ctx: OperationContext | None = None,
    ) -> int:
        self._store("count_tokens", ctx, text=text, model=model)
        return await super()._do_count_tokens(text, model, ctx=ctx)

    async def _do_health(
        self,
        *,
        ctx: OperationContext | None = None,
    ):
        self._store("health", ctx)
        # Call super to get consistent health response
        return await super()._do_health(ctx=ctx)


# ---------------------------------------------------------------------------
# Success-path envelopes
# ---------------------------------------------------------------------------

async def test_wire_contract_capabilities_envelope_success():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)

    out = await h.handle({"op": "embedding.capabilities", "ctx": {}, "args": {}})

    _assert_ok_envelope(out)
    assert "result" in out
    caps = out["result"]
    assert isinstance(caps, dict)
    assert "supported_models" in caps


async def test_wire_contract_embed_envelope_success():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)

    env = {
        "op": "embedding.embed",
        "ctx": {},
        "args": {
            "text": "hi",
            "model": a.supported_models[0],
            "truncate": True,
            "normalize": False,
        },
    }
    out = await h.handle(env)

    _assert_ok_envelope(out)
    assert "result" in out

    res = out["result"]
    assert isinstance(res, dict)
    assert res["model"] == a.supported_models[0]
    assert "embedding" in res
    ev = res["embedding"]
    assert isinstance(ev, dict)
    assert isinstance(ev.get("vector"), list)
    assert ev.get("text") == "hi"


async def test_wire_contract_embed_batch_envelope_success():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)

    env = {
        "op": "embedding.embed_batch",
        "ctx": {},
        "args": {
            "texts": ["a", "b"],
            "model": a.supported_models[0],
        },
    }
    out = await h.handle(env)

    _assert_ok_envelope(out)
    assert "result" in out

    res = out["result"]
    assert isinstance(res, dict)
    assert "embeddings" in res
    assert isinstance(res["embeddings"], list)
    assert len(res["embeddings"]) == 2
    for ev in res["embeddings"]:
        assert isinstance(ev, dict)
        assert isinstance(ev.get("vector"), list)


async def test_wire_contract_count_tokens_envelope_success():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)

    env = {
        "op": "embedding.count_tokens",
        "ctx": {},
        "args": {
            "text": "hello",
            "model": a.supported_models[0],
        },
    }
    out = await h.handle(env)

    _assert_ok_envelope(out)
    assert isinstance(out.get("result"), int)
    assert out["result"] >= 0


async def test_wire_contract_health_envelope_success():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)

    out = await h.handle({"op": "embedding.health", "ctx": {}, "args": {}})

    _assert_ok_envelope(out)
    assert "result" in out
    res = out["result"]
    assert isinstance(res, dict)
    assert "ok" in res
    assert "server" in res
    assert "version" in res


# ---------------------------------------------------------------------------
# Context propagation (OperationContext plumbing)
# ---------------------------------------------------------------------------

async def test_wire_contract_embed_context_roundtrip_and_context_plumbing():
    a = TrackingMockEmbeddingAdapter()
    h = WireEmbeddingHandler(a)

    ctx_wire = {
        "request_id": "req-embed-ctx",
        "idempotency_key": "idem-embed",
        "deadline_ms": 9999999999999,
        "traceparent": "00-abc-xyz-embed",
        "tenant": "tenant-embed",
        "attrs": {"foo": "bar"},
        "ignore_me": "extra",  # MUST be ignored
    }
    args = {
        "text": "ctx-check",
        "model": a.supported_models[0],
        "truncate": True,
        "normalize": False,
    }

    out = await h.handle(
        {
            "op": "embedding.embed",
            "ctx": ctx_wire,
            "args": args,
        }
    )

    _assert_ok_envelope(out)

    # Check context propagation into adapter
    assert a.last_call == "embed"
    assert isinstance(a.last_ctx, OperationContext)
    assert a.last_ctx.request_id == "req-embed-ctx"
    assert a.last_ctx.idempotency_key == "idem-embed"
    assert a.last_ctx.traceparent == "00-abc-xyz-embed"
    assert a.last_ctx.tenant == "tenant-embed"
    # unknown field removed from attrs
    assert "ignore_me" not in (a.last_ctx.attrs or {})


# ---------------------------------------------------------------------------
# Error mapping semantics
# ---------------------------------------------------------------------------

async def test_wire_contract_missing_op_rejected_with_bad_request():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)

    out = await h.handle({"ctx": {}, "args": {}})

    _assert_error_envelope(out)
    # Wire handler raises BadRequest for missing op
    assert out["code"] == "BAD_REQUEST"


async def test_wire_contract_unknown_op_rejected_with_not_supported():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)

    out = await h.handle(
        {"op": "embedding.unknown_op", "ctx": {}, "args": {}}
    )

    _assert_error_envelope(out)
    # Wire handler raises NotSupported for unknown operation
    assert out["code"] == "NOT_SUPPORTED"


async def test_wire_contract_embed_missing_required_fields_yields_bad_request():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)

    # Missing text
    out1 = await h.handle(
        {
            "op": "embedding.embed",
            "ctx": {},
            "args": {"model": a.supported_models[0]},
        }
    )
    _assert_error_envelope(out1)
    # Wire handler validates args before calling adapter
    assert out1["code"] == "BAD_REQUEST"

    # Missing model
    out2 = await h.handle(
        {
            "op": "embedding.embed",
            "ctx": {},
            "args": {"text": "hi"},
        }
    )
    _assert_error_envelope(out2)
    assert out2["code"] == "BAD_REQUEST"


async def test_wire_contract_embed_unknown_model_maps_model_not_available():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)

    out = await h.handle(
        {
            "op": "embedding.embed",
            "ctx": {},
            "args": {"text": "hi", "model": "nope-model"},
        }
    )

    _assert_error_envelope(out)
    # Mock adapter raises ModelNotAvailable for unknown models
    assert out["code"] == "MODEL_NOT_AVAILABLE"


async def test_wire_contract_embed_batch_missing_texts_yields_bad_request():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)

    out = await h.handle(
        {
            "op": "embedding.embed_batch",
            "ctx": {},
            "args": {"model": a.supported_models[0]},
        }
    )

    _assert_error_envelope(out)
    # Wire handler validates texts is a list
    assert out["code"] == "BAD_REQUEST"


async def test_wire_contract_embed_batch_empty_texts_list_yields_bad_request():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)

    out = await h.handle(
        {
            "op": "embedding.embed_batch",
            "ctx": {},
            "args": {"texts": [], "model": a.supported_models[0]},
        }
    )

    _assert_error_envelope(out)
    # Wire handler passes empty list to adapter, adapter validates
    assert out["code"] == "BAD_REQUEST"


async def test_wire_contract_embed_batch_unknown_model_maps_model_not_available():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)

    out = await h.handle(
        {
            "op": "embedding.embed_batch",
            "ctx": {},
            "args": {"texts": ["a"], "model": "nope-model"},
        }
    )

    _assert_error_envelope(out)
    # Mock adapter raises ModelNotAvailable for unknown models
    assert out["code"] == "MODEL_NOT_AVAILABLE"


async def test_wire_contract_count_tokens_unknown_model_maps_model_not_available():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)

    out = await h.handle(
        {
            "op": "embedding.count_tokens",
            "ctx": {},
            "args": {"text": "hi", "model": "nope-model"},
        }
    )

    _assert_error_envelope(out)
    # Mock adapter raises ModelNotAvailable for unknown models
    assert out["code"] == "MODEL_NOT_AVAILABLE"


async def test_wire_contract_error_envelope_includes_message_and_type():
    """
    Force a BadRequest from the adapter and ensure wire handler surfaces a proper error envelope.
    """

    class BadRequestAdapter(MockEmbeddingAdapter):
        async def _do_embed(self, spec: EmbedSpec, *, ctx=None):
            raise BadRequest("bad things")

    a = BadRequestAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)

    out = await h.handle(
        {
            "op": "embedding.embed",
            "ctx": {},
            "args": {"text": "x", "model": a.supported_models[0]},
        }
    )

    _assert_error_envelope(out)
    # BadRequest has code="BAD_REQUEST"
    assert out["code"] == "BAD_REQUEST"
    # error is the type name
    assert out["error"] == "BadRequest"
    assert isinstance(out["message"], str) and out["message"]


async def test_wire_contract_text_too_long_maps_to_text_too_long_code_when_exposed():
    """
    Ensure adapter TextTooLong propagates as TEXT_TOO_LONG in the wire envelope.
    """

    class TLAdapter(MockEmbeddingAdapter):
        async def _do_embed(self, spec: EmbedSpec, *, ctx=None):
            raise TextTooLong("too long")

    a = TLAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)

    out = await h.handle(
        {
            "op": "embedding.embed",
            "ctx": {},
            "args": {"text": "x" * 10_000, "model": a.supported_models[0]},
        }
    )

    _assert_error_envelope(out)
    # TextTooLong has code="TEXT_TOO_LONG"
    assert out["code"] == "TEXT_TOO_LONG"
    assert out["error"] == "TextTooLong"


# ---------------------------------------------------------------------------
# Unexpected exception → UNAVAILABLE
# ---------------------------------------------------------------------------

async def test_wire_contract_unexpected_exception_maps_to_unavailable():
    class BoomAdapter(TrackingMockEmbeddingAdapter):
        async def _do_embed(
            self,
            spec: EmbedSpec,
            *,
            ctx: OperationContext | None = None,
        ):
            raise RuntimeError("boom")

    a = BoomAdapter()
    h = WireEmbeddingHandler(a)

    out = await h.handle(
        {
            "op": "embedding.embed",
            "ctx": {"request_id": "boom"},
            "args": {
                "text": "hi",
                "model": a.supported_models[0],
            },
        }
    )

    _assert_error_envelope(out)
    # Unexpected exceptions map to UNAVAILABLE
    assert out["code"] == "UNAVAILABLE"
    # error should be the underlying exception type name
    assert out["error"] == "RuntimeError"
    # message should surface the original message
    assert "boom" in out.get("message", "")


async def test_wire_contract_invalid_envelope_structure_rejected():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)

    # ctx not an object
    out1 = await h.handle({"op": "embedding.capabilities", "ctx": "not-an-object", "args": {}})
    _assert_error_envelope(out1)
    assert out1["code"] == "BAD_REQUEST"

    # args not an object
    out2 = await h.handle({"op": "embedding.capabilities", "ctx": {}, "args": "not-an-object"})
    _assert_error_envelope(out2)
    assert out2["code"] == "BAD_REQUEST"

    # Missing both ctx and args
    out3 = await h.handle({"op": "embedding.capabilities"})
    _assert_error_envelope(out3)
    assert out3["code"] == "BAD_REQUEST"


async def test_wire_contract_batch_invalid_texts_type_rejected():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)

    # texts not a list
    out = await h.handle(
        {
            "op": "embedding.embed_batch",
            "ctx": {},
            "args": {"texts": "not-a-list", "model": a.supported_models[0]},
        }
    )
    _assert_error_envelope(out)
    assert out["code"] == "BAD_REQUEST"
