from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict

import asyncio
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
# Error-context decorator behavior
# ---------------------------------------------------------------------------


def test_error_context_includes_langchain_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When an error occurs during LangChain embedding, error context should
    include LangChain-specific metadata via attach_context().
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(
        langchain_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingAdapter:
        def embed(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("test error from langchain adapter")

    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=FailingAdapter(),
        model="err-model",
    )

    config = {
        "run_id": "run-ctx",
        "run_name": "error-test",
    }

    with pytest.raises(RuntimeError, match="test error from langchain adapter"):
        embeddings.embed_documents(["x", "y"], config=config)

    # Verify some context was attached
    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "langchain"
    # Best-effort propagation of config metadata
    if "run_id" in captured_context:
        assert captured_context["run_id"] == "run-ctx"


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


# ---------------------------------------------------------------------------
# Capabilities / health passthrough
# ---------------------------------------------------------------------------


def test_capabilities_and_health_passthrough_when_underlying_provides() -> None:
    """
    When the underlying adapter implements capabilities/acapabilities and
    health/ahealth, CorpusLangChainEmbeddings should surface them.
    """

    class FullAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0, 1.0] for _ in texts]

        def capabilities(self) -> Dict[str, Any]:
            return {"ok": True}

        async def acapabilities(self) -> Dict[str, Any]:
            return {"ok_async": True}

        def health(self) -> Dict[str, Any]:
            return {"status": "healthy"}

        async def ahealth(self) -> Dict[str, Any]:
            return {"status_async": "healthy"}

    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=FullAdapter(),
        model="cap-model",
    )

    # Sync passthrough
    caps = embeddings.capabilities()
    assert isinstance(caps, dict)
    assert caps.get("ok") is True

    health = embeddings.health()
    assert isinstance(health, dict)
    assert health.get("status") == "healthy"

    # Async passthrough via event loop
    acaps = asyncio.run(embeddings.acapabilities())
    assert isinstance(acaps, dict)
    assert acaps.get("ok_async") is True

    ahealth = asyncio.run(embeddings.ahealth())
    assert isinstance(ahealth, dict)
    assert ahealth.get("status_async") == "healthy"


@pytest.mark.asyncio
async def test_async_capabilities_and_health_fallback_to_sync() -> None:
    """
    acapabilities/ahealth should fall back to sync capabilities()/health()
    when only sync methods are implemented on the underlying adapter.
    """

    class CapHealthAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0, 1.0] for _ in texts]

        def capabilities(self) -> Dict[str, Any]:
            return {"via_sync_caps": True}

        def health(self) -> Dict[str, Any]:
            return {"via_sync_health": True}

    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=CapHealthAdapter(),
        model="cap-fallback-model",
    )

    acaps = await embeddings.acapabilities()
    assert isinstance(acaps, dict)
    assert acaps.get("via_sync_caps") is True

    ahealth = await embeddings.ahealth()
    assert isinstance(ahealth, dict)
    assert ahealth.get("via_sync_health") is True


def test_capabilities_and_health_return_empty_when_missing() -> None:
    """
    If the underlying adapter has no capabilities()/health(), the LangChain
    adapter should return an empty dict rather than raising.
    """

    class NoCapHealthAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 3 for _ in texts]

    embeddings = CorpusLangChainEmbeddings(
        corpus_adapter=NoCapHealthAdapter(),
        model="no-cap-health-model",
    )

    caps = embeddings.capabilities()
    assert isinstance(caps, dict)
    assert caps == {}

    health = embeddings.health()
    assert isinstance(health, dict)
    assert health == {}

    # Async variants should also return empty mapping
    acaps = asyncio.run(embeddings.acapabilities())
    assert isinstance(acaps, dict)
    assert acaps == {}

    ahealth = asyncio.run(embeddings.ahealth())
    assert isinstance(ahealth, dict)
    assert ahealth == {}


# ---------------------------------------------------------------------------
# Resource management (context managers)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_closes_underlying_adapter() -> None:
    """
    __enter__/__exit__ and __aenter__/__aexit__ should call close/aclose on
    the underlying adapter when those methods exist.
    """

    class ClosingAdapter:
        def __init__(self) -> None:
            self.closed = False
            self.aclosed = False

        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 2 for _ in texts]

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.aclosed = True

    adapter = ClosingAdapter()

    # Sync context manager
    with CorpusLangChainEmbeddings(corpus_adapter=adapter, model="ctx-model") as emb:
        _ = emb.embed_documents(["x"])  # smoke

    assert adapter.closed is True

    # Async context manager
    adapter2 = ClosingAdapter()
    emb2 = CorpusLangChainEmbeddings(corpus_adapter=adapter2, model="ctx-model-2")

    async with emb2:
        _ = await emb2.aembed_documents(["y"])

    assert adapter2.aclosed is True
