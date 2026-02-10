# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Error taxonomy & retry hints.

Spec refs:
  • §12.1, §12.4 Error Handling
  • §10.4 Errors (Embedding-Specific)

Notes:
- No skips: tests assert behavior consistent with capabilities.
- Batch failure records in this SDK are flat dicts:
    {index, text, error, code, message, metadata?}
  (not nested error objects).
"""

import pytest
import time

from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter,
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
    BadRequest,
    NotSupported,
)

# NOTE:
# Do NOT set a global pytestmark=asyncio because this file contains a sync test.
# Instead mark async tests individually.
# This avoids: PytestWarning: marked with '@pytest.mark.asyncio' but it is not an async function.


@pytest.mark.asyncio
async def test_error_handling_text_too_long_maps_correctly(adapter: BaseEmbeddingAdapter):
    """§10.4: If max_text_length is declared, TextTooLong must map to TEXT_TOO_LONG and be non-retryable."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    if caps.max_text_length is not None:
        long_text = "x" * (caps.max_text_length + 1)
        spec = EmbedSpec(text=long_text, model=model, truncate=False)

        with pytest.raises(TextTooLong) as exc_info:
            await adapter.embed(spec)

        err = exc_info.value
        assert err.code == "TEXT_TOO_LONG"
        assert err.retry_after_ms is None

        error_msg = str(err).lower()
        assert any(term in error_msg for term in ["text", "long", "length", "max"])
    else:
        # If no max_text_length declared, raising TextTooLong for length alone is inconsistent.
        long_text = "x" * 20000
        try:
            await adapter.embed(EmbedSpec(text=long_text, model=model, truncate=False))
        except TextTooLong:
            raise AssertionError("Adapter raised TextTooLong but capabilities.max_text_length is None")


@pytest.mark.asyncio
async def test_error_handling_model_not_available_maps_correctly(adapter: BaseEmbeddingAdapter):
    """§10.4: Unsupported models must raise ModelNotAvailable with correct code."""
    with pytest.raises(ModelNotAvailable) as exc_info:
        await adapter.embed(EmbedSpec(text="x", model="nonexistent-model-123"))

    err = exc_info.value
    assert err.code == "MODEL_NOT_AVAILABLE"
    assert err.retry_after_ms is None

    error_msg = str(err).lower()
    assert any(term in error_msg for term in ["model", "available", "support"])


@pytest.mark.asyncio
async def test_error_handling_bad_request_validation(adapter: BaseEmbeddingAdapter):
    """§10.4: Invalid inputs must raise BadRequest with clear messages."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    with pytest.raises(BadRequest) as exc_info:
        await adapter.embed(EmbedSpec(text="", model=model))

    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ["text", "empty", "invalid"])

    with pytest.raises(BadRequest) as exc_info:
        await adapter.embed(EmbedSpec(text="test", model=""))

    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ["model", "empty", "invalid"])


@pytest.mark.asyncio
async def test_error_handling_not_supported_clear_messages(adapter: BaseEmbeddingAdapter):
    """§10.4: NotSupported errors must indicate missing features clearly, consistent with capabilities."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    # Normalization
    if caps.supports_normalization:
        res = await adapter.embed(EmbedSpec(text="test", model=model, normalize=True))
        assert res.embedding.vector
    else:
        with pytest.raises(NotSupported) as exc_info:
            await adapter.embed(EmbedSpec(text="test", model=model, normalize=True))
        error_msg = str(exc_info.value).lower()
        assert any(term in error_msg for term in ["normaliz", "support", "implement"])

    # Batch
    if getattr(caps, "supports_batch_embedding", True):
        res = await adapter.embed_batch(BatchEmbedSpec(texts=["test"], model=model))
        assert res.embeddings
    else:
        with pytest.raises(NotSupported) as exc_info:
            await adapter.embed_batch(BatchEmbedSpec(texts=["test"], model=model))
        error_msg = str(exc_info.value).lower()
        assert any(term in error_msg for term in ["batch", "support", "implement"])


@pytest.mark.asyncio
async def test_error_handling_deadline_exceeded_maps_correctly(adapter: BaseEmbeddingAdapter):
    """§12.4: DeadlineExceeded must have correct code and message."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    past_deadline = int(time.time() * 1000) - 1000
    ctx = OperationContext(deadline_ms=past_deadline)
    spec = EmbedSpec(text="x", model=model)

    with pytest.raises(DeadlineExceeded) as exc_info:
        await adapter.embed(spec, ctx=ctx)

    err = exc_info.value
    assert err.code == "DEADLINE_EXCEEDED"

    error_msg = str(err).lower()
    assert any(term in error_msg for term in ["deadline", "timeout", "exceeded"])


@pytest.mark.asyncio
async def test_error_handling_batch_partial_failure_codes(adapter: BaseEmbeddingAdapter):
    """§12.5: If batch returns failed_texts, they must use normalized flat codes (uppercase)."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    if not getattr(caps, "supports_batch_embedding", True):
        with pytest.raises(NotSupported):
            await adapter.embed_batch(BatchEmbedSpec(texts=["x"], model=model))
        return

    ctx = OperationContext(request_id="t_batch_errors", tenant="test")
    texts = ["valid", "", "another valid", " "]
    spec = BatchEmbedSpec(texts=texts, model=model)

    try:
        result = await adapter.embed_batch(spec, ctx=ctx)
    except BadRequest:
        # Fail-fast is acceptable for invalid batch inputs.
        return

    for failure in result.failed_texts:
        assert "index" in failure and isinstance(failure["index"], int)
        assert 0 <= failure["index"] < len(texts)
        assert "error" in failure and isinstance(failure["error"], str)
        assert "code" in failure and isinstance(failure["code"], str)
        assert "message" in failure and isinstance(failure["message"], str)

        code = failure["code"]
        assert code.isupper(), f"Error code should be uppercase: {code}"
        assert len(code) <= 50, f"Error code too long: {code}"


def test_error_handling_retryable_errors_have_retry_after_ms():
    """§12.4: Retryable errors should expose retry_after_ms hints."""
    err1 = ResourceExhausted("rate limited", retry_after_ms=123)
    assert err1.code == "RESOURCE_EXHAUSTED"
    assert err1.retry_after_ms == 123

    err2 = Unavailable("backend down", retry_after_ms=456)
    assert err2.code == "UNAVAILABLE"
    assert err2.retry_after_ms == 456

    err3 = TransientNetwork("net flake", retry_after_ms=42)
    assert err3.code == "TRANSIENT_NETWORK"
    assert err3.retry_after_ms == 42

    err4 = ResourceExhausted("no retry hint")
    assert err4.retry_after_ms is None


@pytest.mark.asyncio
async def test_error_handling_error_inheritance_hierarchy(adapter: BaseEmbeddingAdapter):
    """§12.1: Error types must follow proper inheritance hierarchy."""
    assert issubclass(TextTooLong, EmbeddingAdapterError)
    assert issubclass(ModelNotAvailable, EmbeddingAdapterError)
    assert issubclass(ResourceExhausted, EmbeddingAdapterError)
    assert issubclass(Unavailable, EmbeddingAdapterError)
    assert issubclass(TransientNetwork, EmbeddingAdapterError)
    assert issubclass(DeadlineExceeded, EmbeddingAdapterError)
    assert issubclass(BadRequest, EmbeddingAdapterError)
    assert issubclass(NotSupported, EmbeddingAdapterError)


@pytest.mark.asyncio
async def test_error_handling_context_preserved_in_errors(adapter: BaseEmbeddingAdapter):
    """§6.1: Context may be provided on error paths; error objects must remain well-formed and SIEM-safe."""
    caps = await adapter.capabilities()
    model = caps.supported_models[0]

    ctx = OperationContext(
        request_id="test_error_ctx",
        tenant="test-tenant",
        attrs={"simulate": "none"},
    )

    with pytest.raises(BadRequest) as exc_info:
        await adapter.embed(EmbedSpec(text="", model=model), ctx=ctx)

    err = exc_info.value
    assert isinstance(err, EmbeddingAdapterError)
    assert isinstance(err.code, str) and err.code
    assert "tenant" not in str(err.details).lower()
