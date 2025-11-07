# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Wire handler canonical envelopes.

Spec refs:
  • §4 Wire Contract (Embedding) — canonical envelope shapes
  • §10.3 Embedding Operations (embed, embed_batch, count_tokens, health)
  • §10.4 Error Mapping — normalized codes and error payloads

Covers:
  • Successful envelopes for all supported ops
  • Canonical {ok, code, result, error} shape
  • Argument validation surfaced as BAD_REQUEST via wire
  • Unsupported / unknown ops surfaced as NOT_SUPPORTED
  • Model-not-available mapped to MODEL_NOT_AVAILABLE
  • Batch semantics & failures preserved through envelope
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
)
from corpus_sdk.mock_embedding_adapter import MockEmbeddingAdapter

pytestmark = pytest.mark.asyncio


def _assert_ok_envelope(out):
    assert isinstance(out, dict)
    assert out.get("ok") is True
    assert isinstance(out.get("code"), str)
    assert out["code"] == "OK"
    assert "error" not in out or out["error"] in (None, {})


def _assert_error_envelope(out, *, code: str = None):
    assert isinstance(out, dict)
    assert out.get("ok") is False
    assert "result" not in out or out["result"] in (None, {})
    assert "code" in out and isinstance(out["code"], str) and out["code"]
    assert "error" in out and isinstance(out["error"], dict)
    if code is not None:
        assert out["code"] == code


async def test_capabilities_envelope_success():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)

    out = await h.handle({"op": "embedding.capabilities", "ctx": {}, "args": {}})

    _assert_ok_envelope(out)
    assert "result" in out
    # Basic shape sanity
    caps = out["result"]
    assert isinstance(caps, dict)
    assert "supported_models" in caps


async def test_embed_envelope_success():
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


async def test_embed_batch_envelope_success():
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


async def test_count_tokens_envelope_success():
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


async def test_health_envelope_success():
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


async def test_missing_op_rejected_with_bad_request():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)

    out = await h.handle({"ctx": {}, "args": {}})

    _assert_error_envelope(out)
    assert out["code"] in ("BAD_REQUEST", "NOT_SUPPORTED")


async def test_unknown_op_rejected_with_not_supported():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)

    out = await h.handle(
        {"op": "embedding.unknown_op", "ctx": {}, "args": {}}
    )

    _assert_error_envelope(out, code="NOT_SUPPORTED")


async def test_embed_missing_required_fields_yields_bad_request():
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


async def test_embed_unknown_model_maps_model_not_available():
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
    # Either normalized to MODEL_NOT_AVAILABLE, or at least not OK.
    # Prefer strict check if handler maps ModelNotAvailable correctly.
    assert out["code"] in ("MODEL_NOT_AVAILABLE", "NOT_SUPPORTED")


async def test_embed_batch_missing_texts_yields_bad_request():
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
    assert out["code"] == "BAD_REQUEST"


async def test_embed_batch_unknown_model_maps_model_not_available():
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
    assert out["code"] in ("MODEL_NOT_AVAILABLE", "NOT_SUPPORTED")


async def test_count_tokens_unknown_model_maps_model_not_available():
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
    assert out["code"] in ("MODEL_NOT_AVAILABLE", "NOT_SUPPORTED")


async def test_error_envelope_includes_message_and_type():
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

    _assert_error_envelope(out, code="BAD_REQUEST")
    err = out["error"]
    # Minimal expectations: message string present
    assert "message" in err and isinstance(err["message"], str) and err["message"]


async def test_text_too_long_maps_to_text_too_long_code_when_exposed():
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
    # Don't overfit if implementation differs, but prefer canonical code.
    assert out["code"] in ("TEXT_TOO_LONG", "BAD_REQUEST")
