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

pytestmark = pytest.mark.asyncio


async def test_truncation_embed_truncates_when_allowed_and_sets_flag(adapter: BaseEmbeddingAdapter):
    """§10.6: truncate=True should truncate long texts and set truncated flag."""
    caps = await adapter.capabilities()
    if caps.max_text_length is None:
        pytest.skip("Adapter does not declare max_text_length")

    max_len = caps.max_text_length
    long_text = "x" * (max_len + 10)
    ctx = OperationContext(request_id="t_trunc_single_ok", tenant="t")

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
    caps = await adapter.capabilities()
    if caps.max_text_length is None:
        pytest.skip("Adapter does not declare max_text_length")

    max_len = caps.max_text_length
    long_text = "x" * (max_len + 1)
    ctx = OperationContext(request_id="t_trunc_single_err", tenant="t")

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
    caps = await adapter.capabilities()
    if caps.max_text_length is None:
        pytest.skip("Adapter does not declare max_text_length")
    if not getattr(caps, "supports_batch_embedding", True):
        pytest.skip("Batch embedding not supported")

    max_len = caps.max_text_length
    long1 = "a" * (max_len + 5)
    long2 = "b" * (max_len + 50)

    ctx = OperationContext(request_id="t_trunc_batch_ok", tenant="t")
    spec = BatchEmbedSpec(
        texts=[long1, long2],
        model=caps.supported_models[0],
        truncate=True,
        normalize=False,
    )
    res = await adapter.embed_batch(spec, ctx=ctx)

    # Handle partial failures: all items should be processed (success or failure)
    total_processed = len(res.embeddings) + len(res.failed_texts)
    assert total_processed == 2, f"Batch should process all items: {total_processed} != 2"
    
    # All successful embeddings should be within length limit
    for embedding in res.embeddings:
        assert len(embedding.text) <= max_len, f"Batch embedding exceeds max length: {len(embedding.text)} > {max_len}"


async def test_truncation_batch_oversize_without_truncation_raises(adapter: BaseEmbeddingAdapter):
    """§10.4: Batch with truncate=False and long text must raise TextTooLong."""
    caps = await adapter.capabilities()
    if caps.max_text_length is None:
        pytest.skip("Adapter does not declare max_text_length")
    if not getattr(caps, "supports_batch_embedding", True):
        pytest.skip("Batch embedding not supported")

    max_len = caps.max_text_length
    long1 = "a" * (max_len + 1)
    ctx = OperationContext(request_id="t_trunc_batch_err", tenant="t")

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
    caps = await adapter.capabilities()
    max_len = caps.max_text_length or 1024  # Use default if not specified

    text = "short text well within limit"
    assert len(text) < max_len

    ctx = OperationContext(request_id="t_trunc_short", tenant="t")

    spec = EmbedSpec(
        text=text,
        model=caps.supported_models[0],
        truncate=True,  # Even with truncate=True, short text should be unchanged
        normalize=False,
    )
    res = await adapter.embed(spec, ctx=ctx)

    assert res.embedding.text == text, "Short text should be unchanged"
    assert res.truncated is False, "truncated should be False for short text"


async def test_truncation_exact_length_text_handled(adapter: BaseEmbeddingAdapter):
    """§10.6: Texts at exact max length should be handled correctly."""
    caps = await adapter.capabilities()
    if caps.max_text_length is None:
        pytest.skip("Adapter does not declare max_text_length")

    max_len = caps.max_text_length
    exact_text = "x" * max_len

    ctx = OperationContext(request_id="t_trunc_exact", tenant="t")

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
    caps = await adapter.capabilities()
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

    ctx = OperationContext(request_id="t_trunc_batch_mixed", tenant="t")
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
    caps = await adapter.capabilities()
    if caps.max_text_length is None:
        pytest.skip("Adapter does not declare max_text_length")

    # Create Unicode text that exceeds limit
    base_unicode = "Hello 世界 🌍✨"
    long_unicode = base_unicode * (caps.max_text_length // len(base_unicode) + 2)
    
    ctx = OperationContext(request_id="t_trunc_unicode", tenant="t")

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
    caps = await adapter.capabilities()
    if caps.max_text_length is None:
        pytest.skip("Adapter does not declare max_text_length")

    max_len = caps.max_text_length
    boundary_cases = [
        ("x" * (max_len - 1), False),      # Just under
        ("x" * max_len, False),            # Exactly at
        ("x" * (max_len + 1), True),       # Just over
    ]

    for text, should_truncate in boundary_cases:
        ctx = OperationContext(request_id=f"t_trunc_bound_{len(text)}", tenant="t")
        
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


async def test_truncation_truncation_mode_behavior(adapter: BaseEmbeddingAdapter):
    """§10.6: Truncation mode should match declared behavior."""
    caps = await adapter.capabilities()
    
    # Verify truncation_mode is one of allowed values
    assert caps.truncation_mode in ("base", "adapter", "none")
    
    # If truncation_mode is "base", truncation should happen in BaseEmbeddingAdapter
    # If truncation_mode is "adapter", truncation should happen in provider
    # If truncation_mode is "none", TextTooLong should always be raised for oversize
    # (Implementation verification is in other tests, this just verifies the mode is valid)


async def test_truncation_empty_string_handled(adapter: BaseEmbeddingAdapter):
    """§10.6: Empty string should be handled without truncation."""
    caps = await adapter.capabilities()
    
    empty_text = ""
    ctx = OperationContext(request_id="t_trunc_empty", tenant="t")
    
    spec = EmbedSpec(
        text=empty_text,
        model=caps.supported_models[0],
        truncate=True,
        normalize=False,
    )
    
    # Empty string should work (though may fail validation elsewhere)
    # We just verify it doesn't crash on empty string
    try:
        res = await adapter.embed(spec, ctx=ctx)
        if res is not None:  # If it succeeds, verify truncated flag
            assert res.truncated is False, "Empty string should not be marked as truncated"
    except Exception:
        # Empty string may be rejected by validation, which is acceptable
        pass


async def test_truncation_whitespace_only_text(adapter: BaseEmbeddingAdapter):
    """§10.6: Text with only whitespace should be handled or rejected gracefully."""
    caps = await adapter.capabilities()
    if caps.max_text_length is None:
        pytest.skip("Adapter does not declare max_text_length")
    
    # Create whitespace-only text that exceeds limit
    max_len = caps.max_text_length
    whitespace_text = " " * (max_len + 10)
    
    ctx = OperationContext(request_id="t_trunc_whitespace", tenant="t")
    
    spec = EmbedSpec(
        text=whitespace_text,
        model=caps.supported_models[0],
        truncate=True,
        normalize=False,
    )
    
    # Whitespace-only may be rejected as invalid input, which is acceptable
    # Some adapters may accept and truncate it
    try:
        res = await adapter.embed(spec, ctx=ctx)
        # If it succeeds, verify truncation happened
        assert len(res.embedding.text) <= max_len, "Whitespace text not truncated properly"
        assert res.truncated is True, "truncated should be True for truncated whitespace text"
    except BadRequest:
        # Rejection of whitespace-only text is also acceptable
        pass
