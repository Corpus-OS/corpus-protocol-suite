# tests/frameworks/embedding/test_llamaindex_adapter.py

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict

import inspect
import pytest

import corpus_sdk.embedding.framework_adapters.llamaindex as llamaindex_adapter_module
from corpus_sdk.embedding.framework_adapters.llamaindex import (
    CorpusLlamaIndexEmbeddings,
    LLAMAINDEX_AVAILABLE,
    configure_llamaindex_embeddings,
    register_with_llamaindex,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_embedding_matrix_shape(result: Any, expected_rows: int) -> None:
    """Validate that a result looks like a 2D embedding matrix."""
    assert isinstance(
        result, Sequence
    ), f"Expected sequence, got {type(result).__name__}"
    assert len(result) == expected_rows, (
        f"Expected {expected_rows} rows, got {len(result)}"
    )

    for row in result:
        assert isinstance(
            row, Sequence
        ), f"Row is not a sequence: {type(row).__name__}"
        for val in row:
            assert isinstance(
                val, (int, float)
            ), f"Embedding value is not numeric: {val!r}"


def _assert_embedding_vector_shape(result: Any) -> None:
    """Validate that a result looks like a 1D embedding vector."""
    assert isinstance(
        result, Sequence
    ), f"Expected sequence, got {type(result).__name__}"
    for val in result:
        assert isinstance(
            val, (int, float)
        ), f"Embedding value is not numeric: {val!r}"


def _make_embeddings(adapter: Any) -> CorpusLlamaIndexEmbeddings:
    """
    Construct a CorpusLlamaIndexEmbeddings instance from the generic adapter.

    If the adapter doesn't implement get_embedding_dimension, we provide
    embedding_dimension explicitly to satisfy the constructor's contract.
    """
    kwargs: dict[str, Any] = {"corpus_adapter": adapter}
    if not hasattr(adapter, "get_embedding_dimension"):
        kwargs["embedding_dimension"] = 8
    return CorpusLlamaIndexEmbeddings(**kwargs)


# ---------------------------------------------------------------------------
# Constructor / validation behavior
# ---------------------------------------------------------------------------


def test_constructor_rejects_adapter_without_embed() -> None:
    """
    CorpusLlamaIndexEmbeddings should enforce that corpus_adapter exposes
    an `embed` method; otherwise __init__ should raise TypeError.
    """

    class BadAdapter:
        # deliberately missing `embed`
        def __init__(self) -> None:
            pass

    with pytest.raises(TypeError) as exc_info:
        CorpusLlamaIndexEmbeddings(corpus_adapter=BadAdapter(), embedding_dimension=8)

    msg = str(exc_info.value)
    assert "must implement an EmbeddingProtocolV1-compatible interface" in msg


def test_embedding_dimension_required_without_get_embedding_dimension() -> None:
    """
    If the corpus_adapter does not implement get_embedding_dimension(),
    the constructor should require embedding_dimension to be provided.
    """

    class NoDimAdapter:
        def embed(self, *args: Any, **kwargs: Any) -> list[list[float]]:
            return [[0.0, 0.0]]

    # Missing embedding_dimension -> error
    with pytest.raises(ValueError) as exc_info:
        CorpusLlamaIndexEmbeddings(corpus_adapter=NoDimAdapter())
    assert "Embedding dimension is unknown" in str(exc_info.value)

    # Providing embedding_dimension -> allowed
    embeddings = CorpusLlamaIndexEmbeddings(
        corpus_adapter=NoDimAdapter(),
        embedding_dimension=16,
    )
    assert isinstance(embeddings, CorpusLlamaIndexEmbeddings)
    assert embeddings.embedding_dimension == 16


def test_embedding_dimension_reads_from_adapter_when_available() -> None:
    """
    If the adapter exposes get_embedding_dimension(), embedding_dimension
    should be derived from it (unless explicitly overridden).
    """

    class DimAdapter:
        def embed(self, *args: Any, **kwargs: Any) -> list[list[float]]:
            return [[0.0] * 4]

        def get_embedding_dimension(self) -> int:
            return 4

    embeddings = CorpusLlamaIndexEmbeddings(corpus_adapter=DimAdapter())
    assert embeddings.embedding_dimension == 4

    # Explicit override should win
    embeddings_override = CorpusLlamaIndexEmbeddings(
        corpus_adapter=DimAdapter(),
        embedding_dimension=12,
    )
    assert embeddings_override.embedding_dimension == 12


def test_configure_and_register_helpers_return_embeddings(adapter: Any) -> None:
    """
    configure_llamaindex_embeddings and register_with_llamaindex should both
    return CorpusLlamaIndexEmbeddings instances wired to the given adapter.
    """
    emb1 = configure_llamaindex_embeddings(
        corpus_adapter=adapter,
        model_name="cfg-model",
    )
    assert isinstance(emb1, CorpusLlamaIndexEmbeddings)
    assert emb1.corpus_adapter is adapter

    emb2 = register_with_llamaindex(
        corpus_adapter=adapter,
        model_name="reg-model",
    )
    assert isinstance(emb2, CorpusLlamaIndexEmbeddings)
    assert emb2.corpus_adapter is adapter


def test_LLAMAINDEX_AVAILABLE_is_bool() -> None:
    """
    LLAMAINDEX_AVAILABLE flag should always be a boolean, regardless of
    whether LlamaIndex is actually installed.
    """
    assert isinstance(LLAMAINDEX_AVAILABLE, bool)


def test_llamaindex_interface_compatibility(adapter: Any) -> None:
    """
    Verify that CorpusLlamaIndexEmbeddings implements the expected LlamaIndex
    BaseEmbedding interface when LlamaIndex is available.
    """
    embeddings = _make_embeddings(adapter)

    # Core methods should always exist
    assert hasattr(embeddings, "_get_query_embedding")
    assert hasattr(embeddings, "_get_text_embedding")
    assert hasattr(embeddings, "_get_text_embeddings")
    assert hasattr(embeddings, "_aget_query_embedding")
    assert hasattr(embeddings, "_aget_text_embedding")
    assert hasattr(embeddings, "_aget_text_embeddings")

    if not LLAMAINDEX_AVAILABLE:
        pytest.skip(
            "LlamaIndex is not available; cannot assert base class compatibility",
        )

    try:
        from llama_index.core.embeddings import BaseEmbedding  # type: ignore[import]
    except Exception:
        pytest.skip(
            "LLAMAINDEX_AVAILABLE is True but importing BaseEmbedding failed",
        )

    assert isinstance(
        embeddings,
        BaseEmbedding,
    ), "CorpusLlamaIndexEmbeddings should subclass LlamaIndex BaseEmbedding when available"


# ---------------------------------------------------------------------------
# Context translation / LlamaIndexContext mapping
# ---------------------------------------------------------------------------


def test_llamaindex_context_passed_to_context_translation(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Verify that kwargs used as LlamaIndexContext are passed through to
    context_from_llamaindex when embedding.
    """
    captured: Dict[str, Any] = {}

    def fake_from_llamaindex(ctx: Dict[str, Any]) -> None:
        captured["ctx"] = ctx
        # Returning None is allowed; adapter will just skip OperationContext.
        return None

    # Patch the imported symbol inside the module under test
    monkeypatch.setattr(
        llamaindex_adapter_module,
        "context_from_llamaindex",
        fake_from_llamaindex,
    )

    embeddings = _make_embeddings(adapter)

    llama_ctx = {
        "node_ids": ["n1", "n2"],
        "index_id": "idx-123",
        "trace_id": "trace-xyz",
        "workflow": "unit-test",
    }

    # Use the public batch embedding implementation
    result = embeddings._get_text_embeddings(["foo", "bar"], **llama_ctx)
    _assert_embedding_matrix_shape(result, expected_rows=2)

    assert captured.get("ctx") is not None
    assert captured["ctx"] == llama_ctx


# ---------------------------------------------------------------------------
# Sync semantics
# ---------------------------------------------------------------------------


def test_sync_query_and_text_embedding_basic(adapter: Any) -> None:
    """
    Basic smoke test for sync _get_query_embedding / _get_text_embedding /
    _get_text_embeddings behavior: they should accept text input and return
    numeric shapes.
    """
    embeddings = _make_embeddings(adapter)

    query = "llama-query"
    text = "llama-text"
    texts = ["llama-text-1", "llama-text-2", "llama-text-3"]

    q_vec = embeddings._get_query_embedding(query)
    _assert_embedding_vector_shape(q_vec)

    t_vec = embeddings._get_text_embedding(text)
    _assert_embedding_vector_shape(t_vec)

    t_mat = embeddings._get_text_embeddings(texts)
    _assert_embedding_matrix_shape(t_mat, expected_rows=len(texts))


def test_single_text_embedding_consistency(adapter: Any) -> None:
    """
    _get_text_embedding should be consistent with _get_text_embeddings
    for a single text input (at least in dimensionality, typically in values).
    """
    embeddings = _make_embeddings(adapter)

    text = "llama-single-text"

    single_result = embeddings._get_text_embedding(text)
    _assert_embedding_vector_shape(single_result)

    batch_result = embeddings._get_text_embeddings([text])
    _assert_embedding_matrix_shape(batch_result, expected_rows=1)

    # Dimensions must match; if either is empty, it's too weak a signal.
    if not batch_result or len(single_result) == 0 or len(batch_result[0]) == 0:
        pytest.skip("Zero-dimension embeddings; cannot assert consistency")

    assert len(single_result) == len(batch_result[0]), (
        "Single-text embedding dimension does not match batch-of-one row dimension"
    )


def test_empty_text_returns_zero_vector(adapter: Any) -> None:
    """
    Empty or whitespace-only texts should be handled via _handle_empty_text
    and return an all-zero vector of the correct dimension.
    """
    embeddings = _make_embeddings(adapter)

    dim = embeddings.embedding_dimension

    q_vec = embeddings._get_query_embedding("")
    t_vec = embeddings._get_text_embedding("   ")

    assert len(q_vec) == dim
    assert len(t_vec) == dim
    assert all(val == 0.0 for val in q_vec)
    assert all(val == 0.0 for val in t_vec)


def test_large_batch_sync_shape(adapter: Any) -> None:
    """
    Larger batches should still produce N rows for N inputs. This lightly
    stresses translator batching behavior.
    """
    embeddings = _make_embeddings(adapter)

    texts = [f"node-text-{i}" for i in range(40)]
    result = embeddings._get_text_embeddings(texts)
    _assert_embedding_matrix_shape(result, expected_rows=len(texts))


# ---------------------------------------------------------------------------
# Async semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_query_and_text_embedding_basic(adapter: Any) -> None:
    """
    Async _aget_query_embedding / _aget_text_embedding / _aget_text_embeddings
    should be coroutine functions and produce shapes compatible with sync API.
    """
    embeddings = _make_embeddings(adapter)

    # Ensure async methods exist and are coroutine functions
    assert hasattr(embeddings, "_aget_query_embedding")
    assert hasattr(embeddings, "_aget_text_embedding")
    assert hasattr(embeddings, "_aget_text_embeddings")

    assert inspect.iscoroutinefunction(embeddings._aget_query_embedding)
    assert inspect.iscoroutinefunction(embeddings._aget_text_embedding)
    assert inspect.iscoroutinefunction(embeddings._aget_text_embeddings)

    query = "async-llama-query"
    text = "async-llama-text"
    texts = ["async-text-1", "async-text-2"]

    q_vec = await embeddings._aget_query_embedding(query)
    _assert_embedding_vector_shape(q_vec)

    t_vec = await embeddings._aget_text_embedding(text)
    _assert_embedding_vector_shape(t_vec)

    t_mat = await embeddings._aget_text_embeddings(texts)
    _assert_embedding_matrix_shape(t_mat, expected_rows=len(texts))


@pytest.mark.asyncio
async def test_async_and_sync_same_dimension(adapter: Any) -> None:
    """
    Check that sync and async embeddings for the same input produce vectors
    of the same dimensionality (not necessarily identical values).
    """
    embeddings = _make_embeddings(adapter)

    texts = ["same-dim-1", "same-dim-2"]
    query = "same-dim-query"

    sync_q = embeddings._get_query_embedding(query)
    async_q = await embeddings._aget_query_embedding(query)

    sync_mat = embeddings._get_text_embeddings(texts)
    async_mat = await embeddings._aget_text_embeddings(texts)

    # Query dimensions
    assert len(sync_q) == len(async_q)

    # Batch row counts
    assert len(sync_mat) == len(async_mat) == len(texts)

    if sync_mat and async_mat:
        sync_dim = len(sync_mat[0])
        async_dim = len(async_mat[0])
        assert sync_dim == async_dim

