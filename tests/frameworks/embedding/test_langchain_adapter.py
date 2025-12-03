# tests/frameworks/embedding/test_langchain_adapter.py

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict

import inspect
import pytest
from pydantic import ValidationError

import corpus_sdk.embedding.framework_adapters.langchain as langchain_adapter_module
from corpus_sdk.embedding.framework_adapters.langchain import (
    CorpusLangChainEmbeddings,
    LANGCHAIN_AVAILABLE,
    configure_langchain_embeddings,
    register_with_langchain,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_embedding_matrix_shape(
    result: Any,
    expected_rows: int,
) -> None:
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


# ---------------------------------------------------------------------------
# Pydantic / construction behavior
# ---------------------------------------------------------------------------


def test_pydantic_rejects_adapter_without_embed() -> None:
    """
    CorpusLangChainEmbeddings should validate that corpus_adapter has an
    `embed` method; otherwise Pydantic validation should fail.
    """

    class BadAdapter:
        # deliberately missing `embed`
        def __init__(self) -> None:
            pass

    with pytest.raises(ValidationError) as exc_info:
        CorpusLangChainEmbeddings(corpus_adapter=BadAdapter())

    # Make sure our custom ValueError propagated into the ValidationError
    msg = str(exc_info.value)
    assert "must implement EmbeddingProtocolV1 with an 'embed' method" in msg


def test_pydantic_accepts_valid_corpus_adapter(adapter: Any) -> None:
    """
    A valid corpus_adapter implementing `embed` should be accepted and
    stored as-is on the model.
    """
    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=adapter,
        model="test-model",
    )

    assert embeddings.corpus_adapter is adapter
    assert embeddings.model == "test-model"


def test_configure_and_register_helpers_return_embeddings(adapter: Any) -> None:
    """
    configure_langchain_embeddings and register_with_langchain should both
    return CorpusLangChainEmbeddings instances wired to the given adapter.
    """
    emb1 = configure_langchain_embeddings(
        corpus_adapter=adapter,
        model="cfg-model",
    )
    assert isinstance(emb1, CorpusLangChainEmbeddings)
    assert emb1.corpus_adapter is adapter

    emb2 = register_with_langchain(
        corpus_adapter=adapter,
        model="reg-model",
    )
    assert isinstance(emb2, CorpusLangChainEmbeddings)
    assert emb2.corpus_adapter is adapter


def test_LANGCHAIN_AVAILABLE_is_bool() -> None:
    """
    LANGCHAIN_AVAILABLE flag should always be a boolean, regardless of
    whether LangChain is actually installed.
    """
    assert isinstance(LANGCHAIN_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# LangChain interface compatibility
# ---------------------------------------------------------------------------


def test_langchain_interface_compatibility(adapter: Any) -> None:
    """
    Verify that CorpusLangChainEmbeddings implements the expected LangChain
    Embeddings interface when LangChain is available, and that the core
    embedding methods are present regardless.
    """
    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=adapter,
        model="iface-model",
    )

    # Core methods should always exist
    assert hasattr(embeddings, "embed_documents")
    assert hasattr(embeddings, "embed_query")
    assert hasattr(embeddings, "aembed_documents")
    assert hasattr(embeddings, "aembed_query")

    if not LANGCHAIN_AVAILABLE:
        pytest.skip("LangChain is not available; cannot assert base class compatibility")

    try:
        from langchain.embeddings.base import Embeddings  # type: ignore[import]
    except Exception:
        pytest.skip("LANGCHAIN_AVAILABLE is True but importing langchain.embeddings.base failed")

    assert isinstance(
        embeddings,
        Embeddings,
    ), "CorpusLangChainEmbeddings should subclass LangChain Embeddings when available"


# ---------------------------------------------------------------------------
# RunnableConfig / context mapping
# ---------------------------------------------------------------------------


def test_runnable_config_passed_to_context_translation(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Verify that the `config` kwarg is passed through to context_from_langchain.

    We patch context_from_langchain inside the langchain adapter module to
    capture the config object that is passed in.
    """
    captured: Dict[str, Any] = {}

    def fake_from_langchain(
        config: Dict[str, Any],
        framework_version: str | None = None,
    ) -> None:
        captured["config"] = config
        captured["framework_version"] = framework_version
        # Returning None is allowed; the adapter will just skip OperationContext usage.
        return None

    # Patch the imported symbol inside the module under test
    monkeypatch.setattr(
        langchain_adapter_module,
        "context_from_langchain",
        fake_from_langchain,
    )

    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=adapter,
        model="cfg-model",
        framework_version="lc-test-version",
    )

    config = {
        "run_id": "run-123",
        "run_name": "test-run",
        "tags": ["tag-a", "tag-b"],
        "metadata": {"pipeline": "unit-test"},
        "configurable": {"tenant": "acme"},
    }

    # Just ensure call succeeds and our fake context translator sees the config
    result = embeddings.embed_documents(["one", "two"], config=config)
    _assert_embedding_matrix_shape(result, expected_rows=2)

    assert captured.get("config") is config
    assert captured.get("framework_version") == "lc-test-version"


# ---------------------------------------------------------------------------
# Sync / async semantics
# ---------------------------------------------------------------------------


def test_sync_embed_documents_and_query_basic(adapter: Any) -> None:
    """
    Basic smoke test for sync embed_documents and embed_query behavior:
    they should accept simple text input and return numeric shapes.
    """
    embeddings = configure_langchain_embeddings(
        corpus_adapter=adapter,
        model="sync-model",
    )

    texts = ["alpha", "beta", "gamma"]
    query_text = "delta"

    docs_result = embeddings.embed_documents(texts)
    _assert_embedding_matrix_shape(docs_result, expected_rows=len(texts))

    query_result = embeddings.embed_query(query_text)
    _assert_embedding_vector_shape(query_result)


@pytest.mark.asyncio
async def test_async_embed_documents_and_query_basic(adapter: Any) -> None:
    """
    Async aembed_documents / aembed_query should be coroutine functions and
    produce shapes compatible with the sync API.
    """
    embeddings = configure_langchain_embeddings(
        corpus_adapter=adapter,
        model="async-model",
    )

    # Ensure we actually have async methods and they are coroutine functions
    assert hasattr(embeddings, "aembed_documents")
    assert hasattr(embeddings, "aembed_query")
    assert inspect.iscoroutinefunction(embeddings.aembed_documents)
    assert inspect.iscoroutinefunction(embeddings.aembed_query)

    texts = ["alpha-async", "beta-async"]
    query_text = "gamma-async"

    docs_result = await embeddings.aembed_documents(texts)
    _assert_embedding_matrix_shape(docs_result, expected_rows=len(texts))

    query_result = await embeddings.aembed_query(query_text)
    _assert_embedding_vector_shape(query_result)


@pytest.mark.asyncio
async def test_async_and_sync_same_dimension(adapter: Any) -> None:
    """
    Check that sync and async embeddings for the same input produce vectors
    of the same dimensionality (not necessarily identical values).
    """
    embeddings = configure_langchain_embeddings(
        corpus_adapter=adapter,
        model="dim-model",
    )

    texts = ["same-dim-1", "same-dim-2"]
    query = "same-dim-query"

    sync_docs = embeddings.embed_documents(texts)
    sync_query = embeddings.embed_query(query)

    async_docs = await embeddings.aembed_documents(texts)
    async_query = await embeddings.aembed_query(query)

    # Compare dimensions (len of row vectors), if non-empty
    assert len(sync_docs) == len(async_docs) == len(texts)

    if sync_docs and async_docs:
        sync_dim = len(sync_docs[0])
        async_dim = len(async_docs[0])
        assert sync_dim == async_dim

    # Query dimensions
    assert len(sync_query) == len(async_query)


def test_large_batch_sync_shape(adapter: Any) -> None:
    """
    Large-ish batches should still produce N rows for N inputs.
    This is a light stress test around translator batching.
    """
    embeddings = configure_langchain_embeddings(
        corpus_adapter=adapter,
        model="large-batch-model",
    )

    texts = [f"text-{i}" for i in range(50)]
    result = embeddings.embed_documents(texts)
    _assert_embedding_matrix_shape(result, expected_rows=len(texts))

