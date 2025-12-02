# tests/frameworks/embedding/test_autogen_adapter.py

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict

import asyncio
import inspect
import sys
import types

import pytest

import corpus_sdk.embedding.framework_adapters.autogen as autogen_adapter_module
from corpus_sdk.embedding.framework_adapters.autogen import (
    CorpusAutoGenEmbeddings,
    create_retriever,
    register_embeddings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_embedding_matrix_shape(result: Any, expected_rows: int) -> None:
    """Validate that a result looks like a 2D embedding matrix."""
    assert isinstance(result, Sequence), f"Expected sequence, got {type(result).__name__}"
    assert len(result) == expected_rows, f"Expected {expected_rows} rows, got {len(result)}"

    for row in result:
        assert isinstance(row, Sequence), f"Row is not a sequence: {type(row).__name__}"
        for val in row:
            assert isinstance(val, (int, float)), f"Embedding value is not numeric: {val!r}"


def _assert_embedding_vector_shape(result: Any) -> None:
    """Validate that a result looks like a 1D embedding vector."""
    assert isinstance(result, Sequence), f"Expected sequence, got {type(result).__name__}"
    for val in result:
        assert isinstance(val, (int, float)), f"Embedding value is not numeric: {val!r}"


def _make_embeddings(adapter: Any, **kwargs: Any) -> CorpusAutoGenEmbeddings:
    """Construct a CorpusAutoGenEmbeddings instance from the generic adapter."""
    return CorpusAutoGenEmbeddings(corpus_adapter=adapter, **kwargs)


# ---------------------------------------------------------------------------
# Constructor / registration behavior
# ---------------------------------------------------------------------------


def test_constructor_rejects_adapter_without_embed() -> None:
    """
    CorpusAutoGenEmbeddings should enforce that corpus_adapter exposes
    an `embed` method; otherwise __init__ should raise TypeError.
    """

    class BadAdapter:
        # deliberately missing `embed`
        def __init__(self) -> None:
            pass

    with pytest.raises(TypeError) as exc_info:
        CorpusAutoGenEmbeddings(corpus_adapter=BadAdapter())

    msg = str(exc_info.value)
    assert "must implement an EmbeddingProtocolV1-compatible interface" in msg


def test_register_embeddings_returns_instance(adapter: Any) -> None:
    """
    register_embeddings should return a CorpusAutoGenEmbeddings instance wired
    to the given corpus adapter and model/framework_version.
    """
    emb = register_embeddings(
        corpus_adapter=adapter,
        model="auto-model",
        framework_version="autogen-fw-1.0",
    )

    assert isinstance(emb, CorpusAutoGenEmbeddings)
    assert emb.corpus_adapter is adapter
    assert emb.model == "auto-model"
    # framework_version is stored on _framework_version attribute
    assert getattr(emb, "_framework_version") == "autogen-fw-1.0"


# ---------------------------------------------------------------------------
# AutoGen interface compatibility
# ---------------------------------------------------------------------------


def test_autogen_interface_compatibility(adapter: Any) -> None:
    """
    Verify that CorpusAutoGenEmbeddings implements the expected AutoGen
    EmbeddingFunction-style interface when AutoGen is available.
    """
    embeddings = _make_embeddings(adapter)

    # Core methods should always exist
    assert hasattr(embeddings, "embed_documents")
    assert hasattr(embeddings, "embed_query")
    assert hasattr(embeddings, "aembed_documents")
    assert hasattr(embeddings, "aembed_query")
    assert callable(embeddings)  # __call__ for EmbeddingFunction protocol

    try:
        from autogen.agentchat.contrib.retrieve_assistant_agent import (  # type: ignore[import]
            EmbeddingFunction,
        )
    except ImportError:
        # AutoGen not installed - nothing more to assert.
        pytest.skip("AutoGen is not installed; cannot assert EmbeddingFunction compatibility")

    # We can't reliably assert isinstance(...) if EmbeddingFunction is a Protocol,
    # but we can assert that our implementation is structurally compatible:
    assert hasattr(embeddings, "__call__")
    # And that calling __call__ with texts works as EmbeddingFunction expects.
    result = embeddings(["if-this-fails-we-are-not-compatible"])
    _assert_embedding_matrix_shape(result, expected_rows=1)


# ---------------------------------------------------------------------------
# Context translation / AutoGenContext mapping
# ---------------------------------------------------------------------------


def test_autogen_context_passed_to_context_translation(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Verify that autogen_context is passed through to context_from_autogen
    when embedding.
    """
    captured: Dict[str, Any] = {}

    def fake_from_autogen(ctx: Dict[str, Any], framework_version: Any = None) -> None:
        captured["ctx"] = ctx
        captured["framework_version"] = framework_version
        # Returning None is allowed; adapter will just skip OperationContext.
        return None

    # Patch the imported symbol inside the module under test
    monkeypatch.setattr(
        autogen_adapter_module,
        "context_from_autogen",
        fake_from_autogen,
    )

    embeddings = _make_embeddings(
        adapter,
        framework_version="autogen-test-version",
    )

    auto_ctx = {
        "conversation_id": "conv-123",
        "agent_name": "agent-x",
        "workflow_type": "chain",
        "retriever_name": "retriever-y",
    }

    result = embeddings.embed_documents(
        ["foo", "bar"],
        autogen_context=auto_ctx,
    )
    _assert_embedding_matrix_shape(result, expected_rows=2)

    assert captured.get("ctx") is not None
    assert captured["ctx"] == auto_ctx
    assert captured["framework_version"] == "autogen-test-version"


def test_error_context_includes_autogen_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When an error occurs during AutoGen embedding, error context should include
    AutoGen-specific metadata via attach_context().
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(
        autogen_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingAdapter:
        def embed(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("test error from autogen adapter")

    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=FailingAdapter())

    auto_ctx = {"conversation_id": "conv-ctx", "agent_name": "tester"}

    with pytest.raises(RuntimeError, match="test error from autogen adapter"):
        embeddings.embed_documents(["text"], autogen_context=auto_ctx)

    # Verify some context was attached
    assert captured_context, "attach_context was not called"
    assert "framework" in captured_context
    assert captured_context.get("framework") == "autogen"
    # AutoGen-specific fields should be present
    assert captured_context.get("conversation_id") == "conv-ctx"
    assert captured_context.get("agent_name") == "tester"


# ---------------------------------------------------------------------------
# Sync semantics
# ---------------------------------------------------------------------------


def test_sync_embed_documents_and_query_basic(adapter: Any) -> None:
    """
    Basic smoke test for sync embed_documents / embed_query behavior:
    they should accept text input and return numeric shapes.
    """
    embeddings = _make_embeddings(adapter, model="sync-model")

    texts = ["alpha", "beta", "gamma"]
    query = "delta"

    doc_vecs = embeddings.embed_documents(texts)
    _assert_embedding_matrix_shape(doc_vecs, expected_rows=len(texts))

    q_vec = embeddings.embed_query(query)
    _assert_embedding_vector_shape(q_vec)


def test_call_aliases_embed_documents(adapter: Any) -> None:
    """
    __call__ should behave like embed_documents for AutoGen's EmbeddingFunction
    protocol: passing a sequence of texts and returning a matrix.
    """
    embeddings = _make_embeddings(adapter)

    texts = ["call-one", "call-two"]
    result = embeddings(texts)  # uses __call__
    _assert_embedding_matrix_shape(result, expected_rows=len(texts))


def test_sync_embed_documents_with_autogen_context(adapter: Any) -> None:
    """
    embed_documents should accept autogen_context kwarg and not raise TypeError.
    """
    embeddings = _make_embeddings(adapter)

    ctx = {
        "conversation_id": "conv-ctx",
        "agent_name": "tester",
    }

    result = embeddings.embed_documents(
        ["ctx-one", "ctx-two"],
        autogen_context=ctx,
    )
    _assert_embedding_matrix_shape(result, expected_rows=2)


# ---------------------------------------------------------------------------
# Async semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_embed_documents_and_query_basic(adapter: Any) -> None:
    """
    Async aembed_documents / aembed_query should exist and produce shapes
    compatible with the sync API.
    """
    embeddings = _make_embeddings(adapter)

    # Ensure async methods exist and are coroutine functions
    assert hasattr(embeddings, "aembed_documents")
    assert hasattr(embeddings, "aembed_query")

    assert inspect.iscoroutinefunction(embeddings.aembed_documents)
    assert inspect.iscoroutinefunction(embeddings.aembed_query)

    texts = ["async-alpha", "async-beta"]
    query = "async-gamma"

    doc_vecs = await embeddings.aembed_documents(texts)
    _assert_embedding_matrix_shape(doc_vecs, expected_rows=len(texts))

    q_vec = await embeddings.aembed_query(query)
    _assert_embedding_vector_shape(q_vec)


@pytest.mark.asyncio
async def test_async_and_sync_same_dimension(adapter: Any) -> None:
    """
    Check that sync and async embeddings for the same input produce vectors
    of the same dimensionality (not necessarily identical values).
    """
    embeddings = _make_embeddings(adapter)

    texts = ["dim-a", "dim-b"]
    query = "dim-q"

    sync_vecs = embeddings.embed_documents(texts)
    async_vecs = await embeddings.aembed_documents(texts)

    sync_q = embeddings.embed_query(query)
    async_q = await embeddings.aembed_query(query)

    # Row counts
    assert len(sync_vecs) == len(async_vecs) == len(texts)

    # Dimensions (if any rows present)
    if sync_vecs and async_vecs:
        assert len(sync_vecs[0]) == len(async_vecs[0])

    # Query dims
    assert len(sync_q) == len(async_q)


# ---------------------------------------------------------------------------
# Capabilities / health passthrough (best-effort)
# ---------------------------------------------------------------------------


def test_capabilities_and_health_passthrough_when_underlying_provides() -> None:
    """
    When the underlying adapter implements capabilities/acapabilities and
    health/ahealth, CorpusAutoGenEmbeddings should surface them.
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

    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=FullAdapter())

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

    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=CapHealthAdapter())

    acaps = await embeddings.acapabilities()
    assert isinstance(acaps, dict)
    assert acaps.get("via_sync_caps") is True

    ahealth = await embeddings.ahealth()
    assert isinstance(ahealth, dict)
    assert ahealth.get("via_sync_health") is True


def test_capabilities_and_health_return_empty_when_missing() -> None:
    """
    If the underlying adapter has no capabilities()/health(), the AutoGen
    adapter should return an empty dict rather than raising.
    """

    class NoCapHealthAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 3 for _ in texts]

    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=NoCapHealthAdapter())

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
    with CorpusAutoGenEmbeddings(corpus_adapter=adapter) as emb:
        _ = emb.embed_documents(["x"])  # smoke

    assert adapter.closed is True

    # Async context manager
    adapter2 = ClosingAdapter()
    emb2 = CorpusAutoGenEmbeddings(corpus_adapter=adapter2)
    async with emb2:
        _ = await emb2.aembed_documents(["y"])

    assert adapter2.aclosed is True


# ---------------------------------------------------------------------------
# create_retriever behavior / AutoGen integration
# ---------------------------------------------------------------------------


def test_create_retriever_raises_runtime_error_when_autogen_not_installed(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    When AutoGen is not installed (or import fails), create_retriever should
    raise RuntimeError with a helpful message.
    """
    import builtins as _builtins

    orig_import = _builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("autogen"):
            raise ImportError("forced by test")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(_builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as exc_info:
        create_retriever(
            corpus_adapter=adapter,
            vector_store=object(),
            model="m",
        )

    msg = str(exc_info.value)
    assert "AutoGen is not installed" in msg


def test_create_retriever_configures_vector_store_embedding_function(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    When AutoGen is "available", create_retriever should:

    - Construct a CorpusAutoGenEmbeddings instance.
    - Attach it to vector_store.embedding_function when present.
    - Return a VectorStoreRetriever wrapping the same vector_store.
    """
    # Stub AutoGen's VectorStoreRetriever via sys.modules
    retrieve_utils_mod = types.ModuleType("autogen.retrieve_utils")

    class DummyVectorStoreRetriever:
        def __init__(self, vectorstore: Any) -> None:
            self.vectorstore = vectorstore

    setattr(retrieve_utils_mod, "VectorStoreRetriever", DummyVectorStoreRetriever)

    autogen_pkg = types.ModuleType("autogen")

    monkeypatch.setitem(sys.modules, "autogen", autogen_pkg)
    monkeypatch.setitem(sys.modules, "autogen.retrieve_utils", retrieve_utils_mod)

    class DummyVectorStore:
        def __init__(self) -> None:
            self.embedding_function: Any | None = None

    vs = DummyVectorStore()

    retriever = create_retriever(
        corpus_adapter=adapter,
        vector_store=vs,
        model="text-embedding-3-large",
        framework_version="auto-fw-2",
    )

    # Retriever should be our dummy type and hold original vector store
    assert isinstance(retriever, DummyVectorStoreRetriever)
    assert retriever.vectorstore is vs

    # embedding_function should be CorpusAutoGenEmbeddings
    assert isinstance(vs.embedding_function, CorpusAutoGenEmbeddings)
    assert vs.embedding_function.corpus_adapter is adapter
    assert vs.embedding_function.model == "text-embedding-3-large"


def test_create_retriever_configures_private_embedding_function_when_only_private_present(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If vector_store does not expose `embedding_function` but does expose
    `_embedding_function`, create_retriever should set that attribute.
    """
    retrieve_utils_mod = types.ModuleType("autogen.retrieve_utils")

    class DummyVectorStoreRetriever:
        def __init__(self, vectorstore: Any) -> None:
            self.vectorstore = vectorstore

    setattr(retrieve_utils_mod, "VectorStoreRetriever", DummyVectorStoreRetriever)
    autogen_pkg = types.ModuleType("autogen")

    monkeypatch.setitem(sys.modules, "autogen", autogen_pkg)
    monkeypatch.setitem(sys.modules, "autogen.retrieve_utils", retrieve_utils_mod)

    class DummyVectorStore:
        def __init__(self) -> None:
            self._embedding_function: Any | None = None

    vs = DummyVectorStore()

    retriever = create_retriever(
        corpus_adapter=adapter,
        vector_store=vs,
        model="text-embedding-3-small",
    )

    assert isinstance(retriever, DummyVectorStoreRetriever)
    assert retriever.vectorstore is vs

    assert isinstance(vs._embedding_function, CorpusAutoGenEmbeddings)
    assert vs._embedding_function.corpus_adapter is adapter
