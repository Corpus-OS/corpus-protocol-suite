# tests/frameworks/embedding/test_autogen_adapter.py

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional

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

from corpus_sdk.embedding.embedding_base import (
    BatchEmbedResult,
    BatchEmbedSpec,
    EmbedResult,
    EmbedSpec,
    EmbeddingCapabilities,
    EmbeddingProtocolV1,
    EmbeddingStats,
    EmbeddingVector,
)


# ---------------------------------------------------------------------------
# Test adapter fixture (Protocol V1-shaped)
# ---------------------------------------------------------------------------


class StubV1Adapter:
    """
    Minimal EmbeddingProtocolV1-shaped adapter used for conformance tests.

    NOTE: We intentionally return typed results (EmbedResult / BatchEmbedResult)
    because the AutoGen adapter routes through EmbeddingTranslator, which expects
    protocol-level results, not raw matrices.
    """

    def __init__(self) -> None:
        self.closed = False
        self.aclosed = False

    async def capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            server="stub",
            version="0",
            supported_models=(
                "auto-model",
                "sync-model",
                "text-embedding-3-large",
                "text-embedding-3-small",
            ),
        )

    async def embed(self, spec: EmbedSpec, *, ctx: Any = None) -> EmbedResult:
        vec = [0.0, 1.0]
        ev = EmbeddingVector(
            vector=vec,
            text=spec.text,
            model=spec.model,
            dimensions=len(vec),
            index=None,
            metadata=None,
        )
        return EmbedResult(
            embedding=ev,
            model=spec.model,
            text=spec.text,
            tokens_used=None,
            truncated=False,
        )

    async def embed_batch(self, spec: BatchEmbedSpec, *, ctx: Any = None) -> BatchEmbedResult:
        evs = [
            EmbeddingVector(
                vector=[0.0, 1.0],
                text=t,
                model=spec.model,
                dimensions=2,
                index=i,
                metadata=None,
            )
            for i, t in enumerate(spec.texts)
        ]
        return BatchEmbedResult(
            embeddings=evs,
            model=spec.model,
            total_texts=len(spec.texts),
            total_tokens=None,
            failed_texts=[],
        )

    async def count_tokens(self, text: str, model: str, *, ctx: Any = None) -> int:
        return len(text)

    async def health(self, *, ctx: Any = None) -> Dict[str, Any]:
        return {"ok": True, "server": "stub", "version": "0", "models": {}}

    async def get_stats(self, *, ctx: Any = None) -> EmbeddingStats:
        return EmbeddingStats()

    # Lifecycle hooks (used by context manager tests)
    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.aclosed = True


@pytest.fixture
def adapter() -> EmbeddingProtocolV1:
    return StubV1Adapter()


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


def _make_embeddings(adapter_obj: Any, **kwargs: Any) -> CorpusAutoGenEmbeddings:
    return CorpusAutoGenEmbeddings(corpus_adapter=adapter_obj, **kwargs)


# ---------------------------------------------------------------------------
# Constructor / registration behavior
# ---------------------------------------------------------------------------


def test_constructor_rejects_adapter_without_embed() -> None:
    """
    CorpusAutoGenEmbeddings enforces that corpus_adapter exposes a callable `embed`.
    """

    class BadAdapter:
        # deliberately missing `embed`
        def __init__(self) -> None:
            pass

    with pytest.raises(TypeError) as exc_info:
        CorpusAutoGenEmbeddings(corpus_adapter=BadAdapter())  # type: ignore[arg-type]

    msg = str(exc_info.value)
    assert "must implement an EmbeddingProtocolV1-compatible interface" in msg
    assert "'embed' method" in msg


def test_register_embeddings_returns_instance(adapter: Any) -> None:
    """
    register_embeddings returns a CorpusAutoGenEmbeddings instance wired to the given
    corpus adapter and model/framework_version.
    """
    emb = register_embeddings(
        corpus_adapter=adapter,
        model="auto-model",
        framework_version="autogen-fw-1.0",
    )

    assert isinstance(emb, CorpusAutoGenEmbeddings)
    assert emb.corpus_adapter is adapter
    assert emb.model == "auto-model"
    assert getattr(emb, "_framework_version") == "autogen-fw-1.0"


# ---------------------------------------------------------------------------
# Translator boundary (conformance seam)
# ---------------------------------------------------------------------------


def test_translator_created_with_expected_args(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """
    Pin the create_embedding_translator call signature/arguments to avoid
    subtle version-skew regressions.
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def embed(self, raw_texts: Any, op_ctx: Any, framework_ctx: Any) -> Any:
            # Return a simple numeric matrix/vector depending on input shape.
            if isinstance(raw_texts, list):
                return [[0.0, 1.0] for _ in raw_texts]
            return [0.0, 1.0]

        async def arun_embed(self, raw_texts: Any, op_ctx: Any, framework_ctx: Any) -> Any:
            return self.embed(raw_texts, op_ctx, framework_ctx)

    def fake_create_embedding_translator(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return DummyTranslator()

    monkeypatch.setattr(autogen_adapter_module, "create_embedding_translator", fake_create_embedding_translator)

    embeddings = CorpusAutoGenEmbeddings(
        corpus_adapter=adapter,
        model="m",
        batch_config=None,
        text_normalization_config=None,
        framework_version="fv",
    )
    _ = embeddings.embed_documents(["x"])

    assert captured["adapter"] is adapter
    assert captured["framework"] == "autogen"
    assert captured["translator"] is None
    assert "batch_config" in captured
    assert "text_normalization_config" in captured


def test_framework_ctx_contains_autogen_metadata(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """
    Ensure framework_ctx contains key observability & AutoGen routing metadata.
    """
    seen: Dict[str, Any] = {}

    class DummyTranslator:
        def embed(self, raw_texts: Any, op_ctx: Any, framework_ctx: Any) -> Any:
            seen["raw_texts"] = raw_texts
            seen["op_ctx"] = op_ctx
            seen["framework_ctx"] = framework_ctx
            return [[0.0, 1.0] for _ in raw_texts]

        async def arun_embed(self, raw_texts: Any, op_ctx: Any, framework_ctx: Any) -> Any:
            return self.embed(raw_texts, op_ctx, framework_ctx)

    monkeypatch.setattr(autogen_adapter_module, "create_embedding_translator", lambda **_: DummyTranslator())

    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter, model="m", framework_version="fv")

    _ = embeddings.embed_documents(
        ["a", "b"],
        autogen_context={
            "conversation_id": "c1",
            "agent_name": "agent1",
            "workflow_type": "chain",
            "retriever_name": "retr1",
        },
    )

    fc = seen["framework_ctx"]
    assert fc["framework"] == "autogen"
    assert fc["model"] == "m"
    assert fc["framework_version"] == "fv"
    assert fc["conversation_id"] == "c1"
    assert fc["agent_name"] == "agent1"
    assert fc["workflow_type"] == "chain"
    assert fc["retriever_name"] == "retr1"
    assert "_operation_context" in fc  # best-effort: may be None, but key is set when built


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
        pytest.skip("AutoGen is not installed; cannot assert EmbeddingFunction compatibility")

    # Structural compatibility checks (EmbeddingFunction may be a Protocol)
    assert hasattr(embeddings, "__call__")
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
    with framework_version.
    """
    captured: Dict[str, Any] = {}

    def fake_from_autogen(ctx: Dict[str, Any], framework_version: Any = None) -> None:
        captured["ctx"] = ctx
        captured["framework_version"] = framework_version
        return None

    monkeypatch.setattr(autogen_adapter_module, "context_from_autogen", fake_from_autogen)

    embeddings = _make_embeddings(adapter, framework_version="autogen-test-version")

    auto_ctx = {
        "conversation_id": "conv-123",
        "agent_name": "agent-x",
        "workflow_type": "chain",
        "retriever_name": "retriever-y",
    }

    result = embeddings.embed_documents(["foo", "bar"], autogen_context=auto_ctx)
    _assert_embedding_matrix_shape(result, expected_rows=2)

    assert captured["ctx"] == auto_ctx
    assert captured["framework_version"] == "autogen-test-version"


def test_invalid_autogen_context_type_is_ignored(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """
    autogen_context is best-effort. If the type is wrong (not a Mapping),
    embeddings should still work.
    """
    class DummyTranslator:
        def embed(self, raw_texts: Any, op_ctx: Any, framework_ctx: Any) -> Any:
            # Always return a valid numeric matrix.
            return [[0.0, 1.0] for _ in raw_texts]

        async def arun_embed(self, raw_texts: Any, op_ctx: Any, framework_ctx: Any) -> Any:
            return self.embed(raw_texts, op_ctx, framework_ctx)

    monkeypatch.setattr(autogen_adapter_module, "create_embedding_translator", lambda **_: DummyTranslator())

    embeddings = _make_embeddings(adapter)
    out = embeddings.embed_documents(["x"], autogen_context="not-a-mapping")  # type: ignore[arg-type]
    _assert_embedding_matrix_shape(out, expected_rows=1)


def test_context_translation_failure_attaches_context_but_does_not_break(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """
    If context_from_autogen raises, embeddings proceed without OperationContext,
    and attach_context is invoked with operation="context_build".
    """
    calls = {"attached": False}

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("ctx boom")

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        if ctx.get("operation") == "context_build":
            calls["attached"] = True

    class DummyTranslator:
        def embed(self, raw_texts: Any, op_ctx: Any, framework_ctx: Any) -> Any:
            return [[0.0, 1.0] for _ in raw_texts]

        async def arun_embed(self, raw_texts: Any, op_ctx: Any, framework_ctx: Any) -> Any:
            return self.embed(raw_texts, op_ctx, framework_ctx)

    monkeypatch.setattr(autogen_adapter_module, "context_from_autogen", boom)
    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)
    monkeypatch.setattr(autogen_adapter_module, "create_embedding_translator", lambda **_: DummyTranslator())

    embeddings = _make_embeddings(adapter, framework_version="fv")
    out = embeddings.embed_documents(["x"], autogen_context={"conversation_id": "c"})
    _assert_embedding_matrix_shape(out, expected_rows=1)
    assert calls["attached"] is True


# ---------------------------------------------------------------------------
# Input validation conformance
# ---------------------------------------------------------------------------


def test_embed_documents_rejects_non_strings(adapter: Any) -> None:
    embeddings = _make_embeddings(adapter)
    with pytest.raises(TypeError, match=r"expects Sequence\[str\]"):
        embeddings.embed_documents(["ok", 123])  # type: ignore[list-item]


@pytest.mark.asyncio
async def test_aembed_documents_rejects_non_strings(adapter: Any) -> None:
    embeddings = _make_embeddings(adapter)
    with pytest.raises(TypeError, match=r"expects Sequence\[str\]"):
        await embeddings.aembed_documents(["ok", object()])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Error context attachment conformance
# ---------------------------------------------------------------------------


def test_error_context_includes_autogen_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """
    When an error occurs, attach_context should be invoked with AutoGen-specific
    metadata and standard fields.
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    class RaisingTranslator:
        def embed(self, raw_texts: Any, op_ctx: Any, framework_ctx: Any) -> Any:
            raise RuntimeError("boom")

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)
    monkeypatch.setattr(autogen_adapter_module, "create_embedding_translator", lambda **_: RaisingTranslator())

    embeddings = _make_embeddings(adapter, model="m", framework_version="fv")
    auto_ctx = {"conversation_id": "conv-ctx", "agent_name": "tester"}

    with pytest.raises(RuntimeError, match="boom"):
        embeddings.embed_documents(["text"], autogen_context=auto_ctx)

    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "autogen"
    assert captured_context.get("operation") == "embedding_documents"
    assert captured_context.get("conversation_id") == "conv-ctx"
    assert captured_context.get("agent_name") == "tester"
    assert captured_context.get("model") == "m"
    assert captured_context.get("framework_version") == "fv"
    assert "error_codes" in captured_context


@pytest.mark.asyncio
async def test_async_error_context_includes_autogen_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    class RaisingTranslator:
        async def arun_embed(self, raw_texts: Any, op_ctx: Any, framework_ctx: Any) -> Any:
            raise RuntimeError("boom-async")

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)
    monkeypatch.setattr(autogen_adapter_module, "create_embedding_translator", lambda **_: RaisingTranslator())

    embeddings = _make_embeddings(adapter, model="m", framework_version="fv")
    auto_ctx = {"conversation_id": "conv-ctx", "agent_name": "tester"}

    with pytest.raises(RuntimeError, match="boom-async"):
        await embeddings.aembed_documents(["text"], autogen_context=auto_ctx)

    assert captured_context.get("framework") == "autogen"
    assert captured_context.get("operation") == "embedding_documents"
    assert captured_context.get("conversation_id") == "conv-ctx"
    assert captured_context.get("agent_name") == "tester"
    assert captured_context.get("model") == "m"
    assert captured_context.get("framework_version") == "fv"


# ---------------------------------------------------------------------------
# Sync semantics
# ---------------------------------------------------------------------------


def test_sync_embed_documents_and_query_basic(adapter: Any) -> None:
    embeddings = _make_embeddings(adapter, model="sync-model")

    texts = ["alpha", "beta", "gamma"]
    query = "delta"

    doc_vecs = embeddings.embed_documents(texts)
    _assert_embedding_matrix_shape(doc_vecs, expected_rows=len(texts))

    q_vec = embeddings.embed_query(query)
    _assert_embedding_vector_shape(q_vec)


def test_call_aliases_embed_documents(adapter: Any) -> None:
    embeddings = _make_embeddings(adapter)

    texts = ["call-one", "call-two"]
    result = embeddings(texts)  # __call__
    _assert_embedding_matrix_shape(result, expected_rows=len(texts))


def test_sync_embed_documents_with_autogen_context(adapter: Any) -> None:
    embeddings = _make_embeddings(adapter)

    ctx = {"conversation_id": "conv-ctx", "agent_name": "tester"}
    result = embeddings.embed_documents(["ctx-one", "ctx-two"], autogen_context=ctx)
    _assert_embedding_matrix_shape(result, expected_rows=2)


# ---------------------------------------------------------------------------
# Async semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_embed_documents_and_query_basic(adapter: Any) -> None:
    embeddings = _make_embeddings(adapter)

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
    embeddings = _make_embeddings(adapter)

    texts = ["dim-a", "dim-b"]
    query = "dim-q"

    sync_vecs = embeddings.embed_documents(texts)
    async_vecs = await embeddings.aembed_documents(texts)

    sync_q = embeddings.embed_query(query)
    async_q = await embeddings.aembed_query(query)

    assert len(sync_vecs) == len(async_vecs) == len(texts)
    if sync_vecs and async_vecs:
        assert len(sync_vecs[0]) == len(async_vecs[0])
    assert len(sync_q) == len(async_q)


# ---------------------------------------------------------------------------
# Capabilities / health passthrough (best-effort, legacy sync adapters)
# ---------------------------------------------------------------------------


def test_capabilities_and_health_passthrough_when_underlying_provides() -> None:
    """
    This is intentionally for legacy adapters that expose sync capabilities/health
    (CorpusAutoGenEmbeddings.capabilities()/health() are sync passthrough).
    """

    class FullAdapter:
        def embed(self, *args: Any, **kwargs: Any) -> Any:
            return None

        def capabilities(self) -> Dict[str, Any]:
            return {"ok": True}

        async def acapabilities(self) -> Dict[str, Any]:
            return {"ok_async": True}

        def health(self) -> Dict[str, Any]:
            return {"status": "healthy"}

        async def ahealth(self) -> Dict[str, Any]:
            return {"status_async": "healthy"}

    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=FullAdapter())  # type: ignore[arg-type]

    caps = embeddings.capabilities()
    assert caps.get("ok") is True

    health = embeddings.health()
    assert health.get("status") == "healthy"

    acaps = asyncio.run(embeddings.acapabilities())
    assert acaps.get("ok_async") is True

    ahealth = asyncio.run(embeddings.ahealth())
    assert ahealth.get("status_async") == "healthy"


@pytest.mark.asyncio
async def test_async_capabilities_and_health_fallback_to_sync() -> None:
    class CapHealthAdapter:
        def embed(self, *args: Any, **kwargs: Any) -> Any:
            return None

        def capabilities(self) -> Dict[str, Any]:
            return {"via_sync_caps": True}

        def health(self) -> Dict[str, Any]:
            return {"via_sync_health": True}

    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=CapHealthAdapter())  # type: ignore[arg-type]

    acaps = await embeddings.acapabilities()
    assert acaps.get("via_sync_caps") is True

    ahealth = await embeddings.ahealth()
    assert ahealth.get("via_sync_health") is True


def test_capabilities_and_health_return_empty_when_missing() -> None:
    class NoCapHealthAdapter:
        def embed(self, *args: Any, **kwargs: Any) -> Any:
            return None

    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=NoCapHealthAdapter())  # type: ignore[arg-type]

    assert embeddings.capabilities() == {}
    assert embeddings.health() == {}

    acaps = asyncio.run(embeddings.acapabilities())
    assert acaps == {}

    ahealth = asyncio.run(embeddings.ahealth())
    assert ahealth == {}


# ---------------------------------------------------------------------------
# Resource management (context managers)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_closes_underlying_adapter() -> None:
    adapter_obj = StubV1Adapter()
    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter_obj)

    with embeddings as emb:
        _ = emb.embed_documents(["x"])  # smoke

    assert adapter_obj.closed is True

    adapter_obj2 = StubV1Adapter()
    embeddings2 = CorpusAutoGenEmbeddings(corpus_adapter=adapter_obj2)

    async with embeddings2:
        _ = await embeddings2.aembed_documents(["y"])

    assert adapter_obj2.aclosed is True


# ---------------------------------------------------------------------------
# create_retriever behavior / AutoGen integration
# ---------------------------------------------------------------------------


def test_create_retriever_raises_runtime_error_when_autogen_not_installed(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    import builtins as _builtins

    orig_import = _builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("autogen"):
            raise ImportError("forced by test")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(_builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as exc_info:
        create_retriever(corpus_adapter=adapter, vector_store=object(), model="m")

    assert "AutoGen is not installed" in str(exc_info.value)


def test_create_retriever_prefers_setter_method(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    retrieve_utils_mod = types.ModuleType("autogen.retrieve_utils")

    class DummyVectorStoreRetriever:
        def __init__(self, vectorstore: Any, **_: Any) -> None:
            self.vectorstore = vectorstore

    setattr(retrieve_utils_mod, "VectorStoreRetriever", DummyVectorStoreRetriever)
    autogen_pkg = types.ModuleType("autogen")

    monkeypatch.setitem(sys.modules, "autogen", autogen_pkg)
    monkeypatch.setitem(sys.modules, "autogen.retrieve_utils", retrieve_utils_mod)

    class DummyVectorStore:
        def __init__(self) -> None:
            self.set_called_with: Any = None
            self.embedding_function: Any = "should-not-win"

        def set_embedding_function(self, fn: Any) -> None:
            self.set_called_with = fn

        # Add one common method to avoid warning-only validation logs
        def similarity_search(self, *args: Any, **kwargs: Any) -> Any:
            return []

    vs = DummyVectorStore()

    retriever = create_retriever(corpus_adapter=adapter, vector_store=vs, model="text-embedding-3-large")

    assert isinstance(retriever, DummyVectorStoreRetriever)
    assert retriever.vectorstore is vs
    assert vs.set_called_with is not None
    assert isinstance(vs.set_called_with, CorpusAutoGenEmbeddings)


def test_create_retriever_configures_vector_store_embedding_function(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    retrieve_utils_mod = types.ModuleType("autogen.retrieve_utils")

    class DummyVectorStoreRetriever:
        def __init__(self, vectorstore: Any, **_: Any) -> None:
            self.vectorstore = vectorstore

    setattr(retrieve_utils_mod, "VectorStoreRetriever", DummyVectorStoreRetriever)

    autogen_pkg = types.ModuleType("autogen")
    monkeypatch.setitem(sys.modules, "autogen", autogen_pkg)
    monkeypatch.setitem(sys.modules, "autogen.retrieve_utils", retrieve_utils_mod)

    class DummyVectorStore:
        def __init__(self) -> None:
            self.embedding_function: Any | None = None

        def query(self, *args: Any, **kwargs: Any) -> Any:
            return []

    vs = DummyVectorStore()

    retriever = create_retriever(
        corpus_adapter=adapter,
        vector_store=vs,
        model="text-embedding-3-large",
        framework_version="auto-fw-2",
    )

    assert isinstance(retriever, DummyVectorStoreRetriever)
    assert retriever.vectorstore is vs

    assert isinstance(vs.embedding_function, CorpusAutoGenEmbeddings)
    assert vs.embedding_function.corpus_adapter is adapter
    assert vs.embedding_function.model == "text-embedding-3-large"
    assert getattr(vs.embedding_function, "_framework_version") == "auto-fw-2"


def test_create_retriever_configures_private_embedding_function_when_only_private_present(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    retrieve_utils_mod = types.ModuleType("autogen.retrieve_utils")

    class DummyVectorStoreRetriever:
        def __init__(self, vectorstore: Any, **_: Any) -> None:
            self.vectorstore = vectorstore

    setattr(retrieve_utils_mod, "VectorStoreRetriever", DummyVectorStoreRetriever)
    autogen_pkg = types.ModuleType("autogen")

    monkeypatch.setitem(sys.modules, "autogen", autogen_pkg)
    monkeypatch.setitem(sys.modules, "autogen.retrieve_utils", retrieve_utils_mod)

    class DummyVectorStore:
        def __init__(self) -> None:
            self._embedding_function: Any | None = None

        def similarity_search(self, *args: Any, **kwargs: Any) -> Any:
            return []

    vs = DummyVectorStore()

    retriever = create_retriever(corpus_adapter=adapter, vector_store=vs, model="text-embedding-3-small")

    assert isinstance(retriever, DummyVectorStoreRetriever)
    assert retriever.vectorstore is vs

    assert isinstance(vs._embedding_function, CorpusAutoGenEmbeddings)
    assert vs._embedding_function.corpus_adapter is adapter
