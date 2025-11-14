# SPDX-License-Identifier: Apache-2.0
"""
Embedding Conformance — Truncation & max text length.

Spec refs:
  • §10.6 Normalization & Truncation Semantics
  • §10.5 Capabilities Discovery  
  • §10.4 Errors (Embedding-Specific)
  • §12.5 Partial Success & Caching
"""

import pytest

from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter,
    OperationContext,
    EmbedSpec,
    BatchEmbedSpec,
    TextTooLong,
    BadRequest,
)
from examples.common.ctx import make_ctx

pytestmark = pytest.mark.asyncio


async def test_truncation_embed_truncates_when_allowed_and_sets_flag(adapter: BaseEmbeddingAdapter):
    """§10.6: truncate=True should truncate long texts and set truncated flag."""
    caps = adapter.capabilities
    if caps.max_text_length is None:
        pytest.skip("Adapter does not declare max_text_length")

    max_len = caps.max_text_length
    long_text = "x" * (max_len + 10)
    ctx = make_ctx(OperationContext, request_id="t_trunc_single_ok", tenant="t")

    spec = EmbedSpec(
        text=long_text,
        model=caps.supported_models[0],
        truncate=True,
        normalize=False,
    )
    res = await adapter.embed(spec, ctx=ctx)

    assert len(res.embedding.text) <= max_len, f"Text not truncated: {len(res.embedding.text)} > {max_len}"
    assert res.truncated is True, "truncated flag should be True when truncation occurs"


async def test_truncation_embed_raises_when_truncation_disallowed(adapter: BaseEmbeddingAdapter):
    """§10.4: truncate=False with long text must raise TextTooLong."""
    caps = adapter.capabilities
    if caps.max_text_length is None:
        pytest.skip("Adapter does not declare max_text_length")

    max_len = caps.max_text_length
    long_text = "x" * (max_len + 1)
    ctx = make_ctx(OperationContext, request_id="t_trunc_single_err", tenant="t")

    spec = EmbedSpec(
        text=long_text,
        model=caps.supported_models[0],
        truncate=False,
        normalize=False,
    )

    with pytest.raises(TextTooLong) as exc_info:
        await adapter.embed(spec, ctx=ctx)
    
    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ['text', 'long', 'length', 'max']), \
        f"Error should mention text length: {error_msg}"


async def test_truncation_batch_truncates_all_when_allowed(adapter: BaseEmbeddingAdapter):
    """§10.6: Batch should truncate all items when truncate=True."""
    caps = adapter.capabilities
    if caps.max_text_length is None:
        pytest.skip("Adapter does not declare max_text_length")
    if not getattr(caps, "supports_batch_embedding", True):
        pytest.skip("Batch embedding not supported")

    max_len = caps.max_text_length
    long1 = "a" * (max_len + 5)
    long2 = "b" * (max_len + 50)

    ctx = make_ctx(OperationContext, request_id="t_trunc_batch_ok", tenant="t")
    spec = BatchEmbedSpec(
        texts=[long1, long2],
        model=caps.supported_models[0],
        truncate=True,
        normalize=False,
    )
    res = await adapter.embed_batch(spec, ctx=ctx)

    assert len(res.embeddings) == 2, "Batch should process all items"
    for embedding in res.embeddings:
        assert len(embedding.text) <= max_len, f"Batch item not truncated: {len(embedding.text)} > {max_len}"


async def test_truncation_batch_oversize_without_truncation_raises(adapter: BaseEmbeddingAdapter):
    """§10.4: Batch with truncate=False and long text must raise TextTooLong."""
    caps = adapter.capabilities
    if caps.max_text_length is None:
        pytest.skip("Adapter does not declare max_text_length")
    if not getattr(caps, "supports_batch_embedding", True):
        pytest.skip("Batch embedding not supported")

    max_len = caps.max_text_length
    long1 = "a" * (max_len + 1)
    ctx = make_ctx(OperationContext, request_id="t_trunc_batch_err", tenant="t")

    spec = BatchEmbedSpec(
        texts=[long1],
        model=caps.supported_models[0],
        truncate=False,
        normalize=False,
    )

    with pytest.raises(TextTooLong) as exc_info:
        await adapter.embed_batch(spec, ctx=ctx)
    
    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ['text', 'long', 'length', 'max']), \
        f"Error should mention text length: {error_msg}"


async def test_truncation_short_texts_unchanged(adapter: BaseEmbeddingAdapter):
    """§10.6: Short texts within limits should pass through unchanged."""
    caps = adapter.capabilities
    max_len = caps.max_text_length or 1024  # Use default if not specified

    text = "short text well within limit"
    assert len(text) < max_len

    ctx = make_ctx(OperationContext, request_id="t_trunc_short", tenant="t")

    spec = EmbedSpec(
        text=text,
        model=adapter.supported_models[0],
        truncate=True,  # Even with truncate=True, short text should be unchanged
        normalize=False,
    )
    res = await adapter.embed(spec, ctx=ctx)

    assert res.embedding.text == text, "Short text should be unchanged"
    assert res.truncated is False, "truncated should be False for short text"


async def test_truncation_exact_length_text_handled(adapter: BaseEmbeddingAdapter):
    """§10.6: Texts at exact max length should be handled correctly."""
    caps = adapter.capabilities
    if caps.max_text_length is None:
        pytest.skip("Adapter does not declare max_text_length")

    max_len = caps.max_text_length
    exact_text = "x" * max_len

    ctx = make_ctx(OperationContext, request_id="t_trunc_exact", tenant="t")

    spec = EmbedSpec(
        text=exact_text,
        model=caps.supported_models[0],
        truncate=False,  # Should not need truncation
        normalize=False,
    )
    res = await adapter.embed(spec, ctx=ctx)

    assert len(res.embedding.text) == max_len, f"Exact length text should be unchanged: {len(res.embedding.text)} != {max_len}"
    assert res.truncated is False, "truncated should be False for exact length text"


async def test_truncation_batch_mixed_lengths_with_truncation(adapter: BaseEmbeddingAdapter):
    """§12.5: Batch should handle mixed length texts with truncation."""
    caps = adapter.capabilities
    if caps.max_text_length is None:
        pytest.skip("Adapter does not declare max_text_length")
    if not getattr(caps, "supports_batch_embedding", True):
        pytest.skip("Batch embedding not supported")

    max_len = caps.max_text_length
    texts = [
        "short",
        "x" * (max_len + 10),  # Needs truncation
        "medium length text",
        "y" * (max_len + 5),   # Needs truncation  
        "another short"
    ]

    ctx = make_ctx(OperationContext, request_id="t_trunc_batch_mixed", tenant="t")
    spec = BatchEmbedSpec(
        texts=texts,
        model=caps.supported_models[0],
        truncate=True,
        normalize=False,
    )
    res = await adapter.embed_batch(spec, ctx=ctx)

    # Should process all items (either success or failure)
    total_processed = len(res.embeddings) + len(res.failed_texts)
    assert total_processed == len(texts), f"Not all items processed: {total_processed} != {len(texts)}"
    
    # All successful embeddings should be within length limit
    for embedding in res.embeddings:
        assert len(embedding.text) <= max_len, f"Batch embedding exceeds max length: {len(embedding.text)} > {max_len}"


async def test_truncation_unicode_text_truncation(adapter: BaseEmbeddingAdapter):
    """§10.6: Truncation should handle Unicode text correctly."""
    caps = adapter.capabilities
    if caps.max_text_length is None:
        pytest.skip("Adapter does not declare max_text_length")

    # Create Unicode text that exceeds limit
    base_unicode = "Hello 世界 🌍✨"
    long_unicode = base_unicode * (caps.max_text_length // len(base_unicode) + 2)
    
    ctx = make_ctx(OperationContext, request_id="t_trunc_unicode", tenant="t")

    spec = EmbedSpec(
        text=long_unicode,
        model=caps.supported_models[0],
        truncate=True,
        normalize=False,
    )
    res = await adapter.embed(spec, ctx=ctx)

    assert len(res.embedding.text) <= caps.max_text_length, "Unicode text not truncated properly"
    assert res.truncated is True, "truncated should be True for truncated Unicode text"


async def test_truncation_truncation_boundary_consistency(adapter: BaseEmbeddingAdapter):
    """§10.6: Truncation should be consistent around boundary lengths."""
    caps = adapter.capabilities
    if caps.max_text_length is None:
        pytest.skip("Adapter does not declare max_text_length")

    max_len = caps.max_text_length
    boundary_cases = [
        ("x" * (max_len - 1), False),      # Just under
        ("x" * max_len, False),            # Exactly at
        ("x" * (max_len + 1), True),       # Just over
    ]

    for text, should_truncate in boundary_cases:
        ctx = make_ctx(OperationContext, request_id=f"t_trunc_bound_{len(text)}", tenant="t")
        
        spec = EmbedSpec(
            text=text,
            model=caps.supported_models[0],
            truncate=True,
            normalize=False,
        )
        res = await adapter.embed(spec, ctx=ctx)
        
        if should_truncate:
            assert len(res.embedding.text) <= max_len, f"Text should be truncated: {len(text)} > {max_len}"
            assert res.truncated is True, f"truncated should be True for text length {len(text)}"
        else:
            assert res.truncated is False, f"truncated should be False for text length {len(text)}"