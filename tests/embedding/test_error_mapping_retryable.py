# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Error taxonomy & retry hints.

Spec refs:
  • §12.1, §12.4 Error Handling
"""

import pytest

from corpus_sdk.embedding.embedding_base import (
    EmbedSpec,
    BatchEmbedSpec,
    OperationContext,
    EmbeddingAdapterError,
    TextTooLong,
    ModelNotAvailable,
    ResourceExhausted,
    Unavailable,
    TransientNetwork,
    DeadlineExceeded,
)
from corpus_sdk.examples.embedding.mock_embedding_adapter import MockEmbeddingAdapter
from corpus_sdk.examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_text_too_long_maps_correctly():
    """
    TextTooLong MUST map to code=TEXT_TOO_LONG and be non-retryable
    (i.e. no retry_after_ms hint by default).
    """

    class TextTooLongAdapter(MockEmbeddingAdapter):
        async def _do_embed(self, spec: EmbedSpec, *, ctx: OperationContext = None):
            raise TextTooLong("too long")

    a = TextTooLongAdapter(failure_rate=0.0)
    spec = EmbedSpec(text="x" * 10_000_000, model=a.supported_models[0])

    with pytest.raises(TextTooLong) as ei:
        await a.embed(spec)

    err = ei.value
    assert err.code == "TEXT_TOO_LONG"
    assert err.retry_after_ms is None


async def test_model_not_available_maps_correctly():
    """
    Unsupported model MUST surface as ModelNotAvailable with MODEL_NOT_AVAILABLE code.
    """
    a = MockEmbeddingAdapter(failure_rate=0.0)

    with pytest.raises(ModelNotAvailable) as ei:
        await a.embed(EmbedSpec(text="x", model="nope"))

    err = ei.value
    assert err.code == "MODEL_NOT_AVAILABLE"
    assert err.retry_after_ms is None


async def test_retryable_errors_have_retry_after_ms():
    """
    Retryable errors (ResourceExhausted, Unavailable, TransientNetwork)
    SHOULD expose retry_after_ms hints when appropriate.
    """

    class RetryHintAdapter(MockEmbeddingAdapter):
        async def _do_embed(self, spec: EmbedSpec, *, ctx: OperationContext = None):
            raise ResourceExhausted("rate limited", retry_after_ms=123)

        async def _do_embed_batch(self, spec: BatchEmbedSpec, *, ctx: OperationContext = None):
            raise Unavailable("backend down", retry_after_ms=456)

    a = RetryHintAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, tenant="t", request_id="retry-hints")

    # ResourceExhausted (single)
    with pytest.raises(ResourceExhausted) as ei1:
        await a.embed(EmbedSpec(text="x", model=a.supported_models[0]), ctx=ctx)
    err1 = ei1.value
    assert err1.code == "RESOURCE_EXHAUSTED"
    assert isinstance(err1.retry_after_ms, int) and err1.retry_after_ms >= 0

    # Unavailable (batch)
    with pytest.raises(Unavailable) as ei2:
        await a.embed_batch(BatchEmbedSpec(texts=["x"], model=a.supported_models[0]), ctx=ctx)
    err2 = ei2.value
    assert err2.code == "UNAVAILABLE"
    assert isinstance(err2.retry_after_ms, int) and err2.retry_after_ms >= 0

    # TransientNetwork via direct raise to confirm shape
    class TransientAdapter(MockEmbeddingAdapter):
        async def _do_embed(self, spec: EmbedSpec, *, ctx: OperationContext = None):
            raise TransientNetwork("net flake", retry_after_ms=42)

    t = TransientAdapter(failure_rate=0.0)
    with pytest.raises(TransientNetwork) as ei3:
        await t.embed(EmbedSpec(text="x", model=t.supported_models[0]), ctx=ctx)
    err3 = ei3.value
    assert err3.code == "TRANSIENT_NETWORK"
    assert isinstance(err3.retry_after_ms, int) and err3.retry_after_ms >= 0


async def test_deadline_exceeded_maps_correctly():
    """
    Pre-expired deadlines MUST raise DeadlineExceeded with DEADLINE_EXCEEDED code.
    """
    a = MockEmbeddingAdapter(failure_rate=0.0)
    ctx = OperationContext(deadline_ms=0)

    with pytest.raises(DeadlineExceeded) as ei:
        await a.embed(EmbedSpec(text="x", model=a.supported_models[0]), ctx=ctx)

    err = ei.value
    assert err.code == "DEADLINE_EXCEEDED"


async def test_partial_failure_codes_in_failures():
    """
    Batch failures MUST use normalized error/code entries in details.failures.
    """

    class PartialFailureAdapter(MockEmbeddingAdapter):
        async def _do_embed_batch(self, spec: BatchEmbedSpec, *, ctx: OperationContext = None):
            # Simulate partial failures encoded in details
            raise EmbeddingAdapterError(
                "partial failure",
                details={
                    "failures": [
                        {"index": 0, "code": "TEXT_TOO_LONG", "error": "TextTooLong"},
                        {"index": 1, "code": "MODEL_NOT_AVAILABLE", "error": "ModelNotAvailable"},
                    ]
                },
            )

    a = PartialFailureAdapter(failure_rate=0.0)
    spec = BatchEmbedSpec(texts=["a", "b"], model=a.supported_models[0])

    with pytest.raises(EmbeddingAdapterError) as ei:
        await a.embed_batch(spec)

    err = ei.value
    failures = (err.details or {}).get("failures") or []
    assert len(failures) == 2

    for f in failures:
        assert "index" in f
        assert isinstance(f["index"], int)
        assert "code" in f and isinstance(f["code"], str) and f["code"].isupper()
        assert "error" in f and isinstance(f["er

