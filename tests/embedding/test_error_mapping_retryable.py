# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Error taxonomy & retry hints.

Spec refs:
  • §12.1, §12.4 Error Handling
  • §10.4 Errors (Embedding-Specific)
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

pytestmark = pytest.mark.asyncio


def make_ctx(ctx_cls, **kwargs):
    """Local helper to construct an OperationContext."""
    return ctx_cls(**kwargs)


async def test_error_handling_text_too_long_maps_correctly(adapter: BaseEmbeddingAdapter):
    """§10.4: TextTooLong must map to TEXT_TOO_LONG and be non-retryable."""
    caps = adapter.capabilities
    if caps.max_text_length is None:
        pytest.skip("Adapter does not declare max_text_length")

    long_text = "x" * (caps.max_text_length + 1)
    spec = EmbedSpec(text=long_text, model=caps.supported_models[0], truncate=False)

    with pytest.raises(TextTooLong) as exc_info:
        await adapter.embed(spec)

    err = exc_info.value
    assert err.code == "TEXT_TOO_LONG"
    assert err.retry_after_ms is None

    # Verify error message is descriptive
    error_msg = str(err).lower()
    assert any(
        term in error_msg for term in ["text", "long", "length", "max"]
    ), f"Error should mention text length: {error_msg}"


async def test_error_handling_model_not_available_maps_correctly(
    adapter: BaseEmbeddingAdapter,
):
    """§10.4: Unsupported models must raise ModelNotAvailable with correct code."""
    with pytest.raises(ModelNotAvailable) as exc_info:
        await adapter.embed(EmbedSpec(text="x", model="nonexistent-model-123"))

    err = exc_info.value
    assert err.code == "MODEL_NOT_AVAILABLE"
    assert err.retry_after_ms is None

    # Verify error message references the model
    error_msg = str(err).lower()
    assert any(
        term in error_msg for term in ["model", "available", "support"]
    ), f"Error should mention model: {error_msg}"


async def test_error_handling_bad_request_validation(adapter: BaseEmbeddingAdapter):
    """§10.4: Invalid inputs must raise BadRequest with clear messages."""
    # Test empty text
    with pytest.raises(BadRequest) as exc_info:
        await adapter.embed(EmbedSpec(text="", model=adapter.supported_models[0]))

    error_msg = str(exc_info.value).lower()
    assert any(
        term in error_msg for term in ["text", "empty", "invalid"]
    ), f"Error should mention text: {error_msg}"

    # Test empty model
    with pytest.raises(BadRequest) as exc_info:
        await adapter.embed(EmbedSpec(text="test", model=""))

    error_msg = str(exc_info.value).lower()
    assert any(
        term in error_msg for term in ["model", "empty", "invalid"]
    ), f"Error should mention model: {error_msg}"


async def test_error_handling_not_supported_clear_messages(adapter: BaseEmbeddingAdapter):
    """§10.4: NotSupported errors must indicate missing features clearly."""
    caps = adapter.capabilities

    # Test normalization if not supported
    if not getattr(caps, "supports_normalization", False):
        spec = EmbedSpec(
            text="test",
            model=caps.supported_models[0],
            normalize=True,
        )
        with pytest.raises(NotSupported) as exc_info:
            await adapter.embed(spec)

        error_msg = str(exc_info.value).lower()
        assert any(
            term in error_msg for term in ["normaliz", "support", "implement"]
        ), f"Error should mention normalization: {error_msg}"

    # Test batch if not supported
    if not getattr(caps, "supports_batch_embedding", True):
        spec = BatchEmbedSpec(
            texts=["test"],
            model=caps.supported_models[0],
        )
        with pytest.raises(NotSupported) as exc_info:
            await adapter.embed_batch(spec)

        error_msg = str(exc_info.value).lower()
        assert any(
            term in error_msg for term in ["batch", "support", "implement"]
        ), f"Error should mention batch: {error_msg}"


async def test_error_handling_deadline_exceeded_maps_correctly(
    adapter: BaseEmbeddingAdapter,
):
    """§12.4: DeadlineExceeded must have correct code and message."""
    # Use a past deadline to force immediate failure
    past_deadline = int(time.time() * 1000) - 1000
    ctx = make_ctx(OperationContext, deadline_ms=past_deadline)
    spec = EmbedSpec(text="x", model=adapter.supported_models[0])

    with pytest.raises(DeadlineExceeded) as exc_info:
        await adapter.embed(spec, ctx=ctx)

    err = exc_info.value
    assert err.code == "DEADLINE_EXCEEDED"

    error_msg = str(err).lower()
    assert any(
        term in error_msg for term in ["deadline", "timeout", "exceeded"]
    ), f"Error should mention deadline: {error_msg}"


async def test_error_handling_batch_partial_failure_codes(adapter: BaseEmbeddingAdapter):
    """§12.5: Batch failures must use normalized error codes."""
    if not getattr(adapter.capabilities, "supports_batch_embedding", True):
        pytest.skip("Batch embedding not supported")

    ctx = make_ctx(OperationContext, request_id="t_batch_errors", tenant="test")

    # Mix of valid and invalid inputs
    texts = ["valid", "", "another valid", " "]
    spec = BatchEmbedSpec(texts=texts, model=adapter.supported_models[0])

    result = await adapter.embed_batch(spec, ctx=ctx)

    # Validate failure structure
    for failure in result.failed_texts:
        assert "index" in failure
        assert 0 <= failure["index"] < len(texts)
        assert "error" in failure
        assert "code" in failure["error"]
        assert "message" in failure["error"]

        # Error codes should be uppercase and descriptive
        code = failure["error"]["code"]
        assert code.isupper(), f"Error code should be uppercase: {code}"
        assert len(code) <= 50, f"Error code too long: {code}"


def test_error_handling_retryable_errors_have_retry_after_ms():
    """§12.4: Retryable errors should expose retry_after_ms hints."""
    # Test ResourceExhausted
    err1 = ResourceExhausted("rate limited", retry_after_ms=123)
    assert err1.code == "RESOURCE_EXHAUSTED"
    assert err1.retry_after_ms == 123

    # Test Unavailable
    err2 = Unavailable("backend down", retry_after_ms=456)
    assert err2.code == "UNAVAILABLE"
    assert err2.retry_after_ms == 456

    # Test TransientNetwork
    err3 = TransientNetwork("net flake", retry_after_ms=42)
    assert err3.code == "TRANSIENT_NETWORK"
    assert err3.retry_after_ms == 42

    # Test that retry_after_ms is optional
    err4 = ResourceExhausted("no retry hint")
    assert err4.retry_after_ms is None


async def test_error_handling_error_inheritance_hierarchy(adapter: BaseEmbeddingAdapter):
    """§12.1: Error types must follow proper inheritance hierarchy."""
    # All embedding errors should inherit from EmbeddingAdapterError
    assert issubclass(TextTooLong, EmbeddingAdapterError)
    assert issubclass(ModelNotAvailable, EmbeddingAdapterError)
    assert issubclass(ResourceExhausted, EmbeddingAdapterError)
    assert issubclass(Unavailable, EmbeddingAdapterError)
    assert issubclass(TransientNetwork, EmbeddingAdapterError)
    assert issubclass(DeadlineExceeded, EmbeddingAdapterError)
    assert issubclass(BadRequest, EmbeddingAdapterError)
    assert issubclass(NotSupported, EmbeddingAdapterError)


async def test_error_handling_context_preserved_in_errors(adapter: BaseEmbeddingAdapter):
    """§6.1: Error context should preserve request information."""
    from unittest.mock import Mock

    mock_metrics = Mock()

    ctx = make_ctx(
        OperationContext,
        request_id="test_error_ctx",
        tenant="test-tenant",
        metrics=mock_metrics,
    )

    try:
        await adapter.embed(EmbedSpec(text="", model=adapter.supported_models[0]), ctx=ctx)
    except BadRequest:
        # If adapter implements metrics, they should capture the error
        if mock_metrics.method_calls:
            error_calls = [
                call
                for call in mock_metrics.method_calls
                if "error" in str(call).lower()
            ]
            assert error_calls, "Expected error metrics for BadRequest"
