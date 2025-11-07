# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Wire handler canonical envelopes.

Spec refs:
  • Wire Contract (Embedding) — envelope shapes
"""

import pytest

from corpus_sdk.embedding.embedding_base import (
    WireEmbeddingHandler,
    EmbedSpec,
    BatchEmbedSpec,
    BadRequest,
    NotSupported,
)
from corpus_sdk.examples.embedding.mock_embedding_adapter import MockEmbeddingAdapter

pytestmark = pytest.mark.asyncio


async def test_capabilities_envelope_success():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)
    out = await h.handle({"op": "embedding.capabilities", "ctx": {}, "args": {}})
    assert out["ok"] is True
    assert out["code"] == "OK"
    assert "result" in out


async def test_embed_envelope_success():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)
    env = {
        "op": "embedding.embed",
        "ctx": {},
        "args": {"text": "hi", "model": a.supported_models[0], "truncate": True, "normalize": False},
    }
    out = await h.handle(env)
    assert out["ok"] is True
    assert "result" in out
    assert out["result"]["model"] == a.supported_models[0]


async def test_embed_batch_envelope_success():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)
    env = {
        "op": "embedding.embed_batch",
        "ctx": {},
        "args": {"texts": ["a", "b"], "model": a.supported_models[0]},
    }
    out = await h.handle(env)
    assert out["ok"] is True
    assert len(out["result"]["embeddings"]) == 2


async def test_count_tokens_envelope_success():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)
    env = {
        "op": "embedding.count_tokens",
        "ctx": {},
        "args": {"text": "hello", "model": a.supported_models[0]},
    }
    out = await h.handle(env)
    assert out["ok"] is True
    assert isinstance(out["result"], int)


async def test_health_envelope_success():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)
    out = await h.handle({"op": "embedding.health", "ctx": {}, "args": {}})
    assert out["ok"] is True
    assert "result" in out


async def test_missing_op_rejected_with_bad_request():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)
    out = await h.handle({"ctx": {}, "args": {}})
    assert out["ok"] is False
    assert out["code"] in ("BAD_REQUEST", "NOT_SUPPORTED")


async def test_unknown_op_rejected_with_not_supported():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = WireEmbeddingHandler(a)
    out = await h.handle({"op": "embedding.unknown_op", "ctx": {}, "args": {}})
    assert out["ok"] is False
    assert out["code"] == "NOT_SUPPORTED"

