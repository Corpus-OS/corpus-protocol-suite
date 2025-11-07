# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — SIEM-safe metrics & context propagation.

Spec refs:
  • §13.1-13.3 Observability & Privacy
  • §6.1 Context propagation

Covers:
  • observe() called with component="embedding" and correct op
  • Tenant identifiers hashed; never logged raw
  • No raw text/texts/vectors/embeddings in metrics payloads
  • Metrics emitted on success and error paths
  • Batch metrics include batch_size for batch operations
  • deadline_bucket tag emitted when deadline_ms is set
"""

import pytest
from typing import Optional, Mapping, Any, List

from corpus_sdk.embedding.embedding_base import (
    EmbedSpec,
    BatchEmbedSpec,
    OperationContext,
    MetricsSink,
)
from corpus_sdk.examples.embedding.mock_embedding_adapter import MockEmbeddingAdapter
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


class CaptureMetrics(MetricsSink):
    def __init__(self) -> None:
        self.observations: List[dict] = []
        self.counters: List[dict] = []

    def observe(
        self,
        *,
        component: str,
        op: str,
        ms: float,
        ok: bool,
        code: str = "OK",
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.observations.append(
            {
                "component": component,
                "op": op,
                "ok": ok,
                "code": code,
                "extra": dict(extra or {}),
                "ms": ms,
            }
        )

    def counter(
        self,
        *,
        component: str,
        name: str,
        value: int = 1,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.counters.append(
            {
                "component": component,
                "name": name,
                "value": value,
                "extra": dict(extra or {}),
            }
        )


async def test_context_propagates_to_metrics_siem_safe():
    """
    test_context_propagates_to_metrics_siem_safe —
    observe called with component="embedding", correct op.
    """
    m = CaptureMetrics()
    a = MockEmbeddingAdapter(metrics=m, failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_siem_ctx", tenant="acme")

    await a.embed(EmbedSpec(text="hello", model=a.supported_models[0]), ctx=ctx)

    # At least one observation for embed
    embed_obs = [o for o in m.observations if o["op"] == "embed"]
    assert embed_obs, "Expected observe() call for embed()"
    last = embed_obs[-1]
    assert last["component"] == "embedding"
    assert last["ok"] is True
    assert last["code"] == "OK"


async def test_tenant_hashed_never_raw():
    """
    test_tenant_hashed_never_raw —
    Only tenant hash appears; raw tenant never logged.
    """
    m = CaptureMetrics()
    tenant = "super-secret-tenant"
    a = MockEmbeddingAdapter(metrics=m, failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_siem_hash", tenant=tenant)

    await a.embed(EmbedSpec(text="hi", model=a.supported_models[0]), ctx=ctx)

    embed_extras = [o["extra"] for o in m.observations if o["op"] == "embed"]
    assert embed_extras
    last = embed_extras[-1]

    # Base should store a hashed tenant identifier, never raw
    assert "tenant" not in last or last["tenant"] != tenant
    assert tenant not in str(last)
    # Allow either tenant_hash or similar hashed field; assert at least one hash-like value
    has_hash_like = any(
        isinstance(v, str) and len(v) >= 8 and tenant not in v
        for k, v in last.items()
        if "tenant" in k
    )
    assert has_hash_like, "Expected hashed tenant identifier in metrics"


async def test_no_text_in_metrics():
    """
    test_no_text_in_metrics —
    No raw text/texts/vectors/embeddings appear in metrics extras.
    """
    m = CaptureMetrics()
    a = MockEmbeddingAdapter(metrics=m, failure_rate=0.0)
    secret_text = "do not leak this"
    ctx = make_ctx(OperationContext, request_id="t_siem_notext", tenant="t")

    await a.embed(EmbedSpec(text=secret_text, model=a.supported_models[0]), ctx=ctx)

    embed_extras = [o["extra"] for o in m.observations if o["op"] == "embed"]
    assert embed_extras
    extra = embed_extras[-1]

    # No obvious raw input keys
    for banned_key in ("text", "texts", "vector", "embedding", "embeddings"):
        assert banned_key not in extra, f"Unexpected leaked key '{banned_key}' in metrics extra"

    # No raw prompt content embedded in serialized extras
    serialized = str(extra)
    assert secret_text not in serialized, "Raw text content leaked into metrics"


async def test_metrics_emitted_on_error_path():
    """
    test_metrics_emitted_on_error_path —
    Errors still produce observe + error counters.
    """
    m = CaptureMetrics()

    class ErrorAdapter(MockEmbeddingAdapter):
        async def _do_embed(self, spec, *, ctx=None):
            raise ValueError("boom")

    a = ErrorAdapter(metrics=m, failure_rate=0.0)

    with pytest.raises(ValueError):
        await a.embed(EmbedSpec(text="x", model=a.supported_models[0]))

    # Must have at least one failed observation
    failed_obs = [o for o in m.observations if o["ok"] is False]
    assert failed_obs, "Expected failed observation on error path"

    # Should also increment some error-related counter
    assert m.counters, "Expected counters on error path"
    assert any(
        "error" in c["name"].lower() or "fail" in c["name"].lower()
        for c in m.counters
    ), "Expected at least one error/failure counter"


async def test_batch_metrics_include_batch_size():
    """
    test_batch_metrics_include_batch_size —
    batch_size recorded for batch ops.
    """
    m = CaptureMetrics()
    a = MockEmbeddingAdapter(metrics=m, failure_rate=0.0)
    ctx = make_ctx(OperationContext, request_id="t_batch_metrics", tenant="t")

    texts = ["a", "b", "c"]
    await a.embed_batch(BatchEmbedSpec(texts=texts, model=a.supported_models[0]), ctx=ctx)

    batch_obs = [o for o in m.observations if o["op"] == "embed_batch"]
    assert batch_obs, "Expected observe() call for embed_batch()"
    last = batch_obs[-1]
    extra = last["extra"]
    # Allow base implementation detail, but assert a batch_size-style field exists and is correct
    key = next((k for k in extra.keys() if k in ("batch_size", "size", "n_items")), None)
    assert key is not None, "Expected batch_size metadata in metrics extras"
    assert extra[key] == len(texts), "batch_size in metrics must match number of texts"


async def test_deadline_bucket_tagged_when_present():
    """
    test_deadline_bucket_tagged_when_present —
    deadline_bucket emitted when deadline_ms set.
    """
    m = CaptureMetrics()
    # Use standalone mode so deadline logic is active, if base uses it
    a = MockEmbeddingAdapter(metrics=m, failure_rate=0.0, mode="standalone")

    # Small but not expired deadline; sufficient to run once
    import time as _time
    deadline = int(_time.time() * 1000) + 250
    ctx = make_ctx(
        OperationContext,
        request_id="t_deadline_bucket",
        tenant="t",
        deadline_ms=deadline,
    )

    await a.embed(EmbedSpec(text="ok", model=a.supported_models[0]), ctx=ctx)

    embed_obs = [o for o in m.observations if o["op"] == "embed"]
    assert embed_obs, "Expected observe() for embed() with deadline"
    extra = embed_obs[-1]["extra"]

    # Expect some deadline-related tagging when deadline is set
    has_deadline_bucket = any("deadline" in k for k in extra.keys())
    assert has_deadline_bucket, "Expected deadline bucket/tag in metrics extras when deadline_ms is provided"
