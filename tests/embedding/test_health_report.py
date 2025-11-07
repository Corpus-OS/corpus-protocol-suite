# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Health endpoint.

Spec refs:
  • §10.3 health() — Embedding health contract
  • §6.4 Common — Health surfaces MUST be small, stable, and SIEM-safe

Asserts:
  • health() returns canonical shape: {ok, server, version, models}
  • ok is a boolean
  • models is a dict mapping model -> status/string
  • Shape is consistent even when backend reports degraded/odd payloads
"""

import pytest

from corpus_sdk.embedding.embedding_base import OperationContext
from corpus_sdk.examples.embedding.mock_embedding_adapter import MockEmbeddingAdapter
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_health_returns_required_fields():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_health_ok", tenant="t")
    h = await a.health(ctx=ctx)

    assert isinstance(h, dict)
    assert "ok" in h
    assert "server" in h
    assert "version" in h
    assert "models" in h


async def test_health_ok_is_boolean():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = await a.health()
    assert isinstance(h["ok"], bool)


async def test_health_models_dict_shape():
    a = MockEmbeddingAdapter(failure_rate=0.0)
    h = await a.health()
    models = h["models"]

    assert isinstance(models, dict)
    # When mock is healthy/degraded it always reports all supported models
    # This also asserts values are simple + SIEM-safe.
    for k, v in models.items():
        assert isinstance(k, str) and k
        assert isinstance(v, str) and v


class WeirdHealthAdapter(MockEmbeddingAdapter):
    """
    Adapter with odd _do_health payload to verify BaseEmbeddingAdapter
    normalizes shape and preserves keys even in degraded/error-like cases.
    """
    async def _do_health(self, *, ctx: OperationContext = None):
        # Missing server/version/models on purpose
        return {"ok": False}


async def test_health_shape_consistent_on_error_like_response():
    a = WeirdHealthAdapter(failure_rate=0.0)
    h = await a.health()

    # BaseEmbeddingAdapter.health must normalize to canonical shape
    assert isinstance(h, dict)
    assert set(h.keys()) == {"ok", "server", "version", "models"}
    assert isinstance(h["ok"], bool)
    assert isinstance(h["server"], str)
    assert isinstance(h["version"], str)
    assert isinstance(h["models"], dict)

