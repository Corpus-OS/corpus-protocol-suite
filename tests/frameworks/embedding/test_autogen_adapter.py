# tests/frameworks/embedding/test_autogen_adapter.py

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import sys
import types
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional

import pytest

import corpus_sdk.embedding.framework_adapters.autogen as autogen_adapter_module
from corpus_sdk.embedding.framework_adapters.autogen import (
    CorpusAutoGenEmbeddings,
    create_vector_memory,
    register_embeddings,
)
from corpus_sdk.embedding.embedding_base import EmbeddingCapabilities


# ---------------------------------------------------------------------------
# Framework Version Support Matrix
# ---------------------------------------------------------------------------
"""
Framework Version Support:
- AutoGen: modern ecosystem (autogen-core + autogen-ext[chromadb]) for real integration tests
- Python: 3.9+
- Corpus SDK: 1.0.0+

Integration Notes:
- Adapter module must import without AutoGen installed (optional dependency contract).
- create_vector_memory() must wire real AutoGen ChromaDBVectorMemory when available.
- Integration tests are pass/fail (no skip): fail-fast if required deps are absent or incompatible.
"""


# ---------------------------------------------------------------------------
# Test Helpers
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
    """Construct a CorpusAutoGenEmbeddings instance from the adapter."""
    if "model" not in kwargs:
        kwargs["model"] = "mock-embed-512"
    return CorpusAutoGenEmbeddings(corpus_adapter=adapter, **kwargs)


def _unique_collection(prefix: str) -> str:
    """Generate a unique collection name to avoid cross-test contamination."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------------------
# Constructor / Registration Behavior
# ---------------------------------------------------------------------------

def test_constructor_works_with_real_adapter(adapter) -> None:
    """
    CorpusAutoGenEmbeddings should work with any real adapter that implements embed().
    """
    embeddings = _make_embeddings(adapter)
    assert embeddings.corpus_adapter is adapter
    assert hasattr(embeddings.corpus_adapter, "embed")
    assert callable(embeddings.corpus_adapter.embed)


def test_constructor_rejects_common_user_mistakes() -> None:
    """
    CorpusAutoGenEmbeddings should provide clear error messages for common user mistakes.

    IMPORTANT:
    - Assert semantic guidance (what is required) rather than brittle substrings.
    """
    with pytest.raises(TypeError) as exc_info:
        CorpusAutoGenEmbeddings(corpus_adapter=None)  # type: ignore[arg-type]
    msg = str(exc_info.value)
    assert ("EmbeddingProtocolV1-compatible" in msg or "EmbeddingProtocolV1" in msg) and "embed" in msg.lower()

    with pytest.raises(TypeError) as exc_info:
        CorpusAutoGenEmbeddings(corpus_adapter="not an adapter")  # type: ignore[arg-type]
    msg = str(exc_info.value)
    assert ("EmbeddingProtocolV1-compatible" in msg or "EmbeddingProtocolV1" in msg) and "embed" in msg.lower()

    class MockAgent:
        """Looks like an agent but not an embedding adapter."""
        def chat(self) -> None:
            return None

    with pytest.raises(TypeError) as exc_info:
        CorpusAutoGenEmbeddings(corpus_adapter=MockAgent())
    msg = str(exc_info.value)
    assert ("EmbeddingProtocolV1-compatible" in msg or "EmbeddingProtocolV1" in msg) and "embed" in msg.lower()


def test_register_embeddings_returns_instance(adapter) -> None:
    """
    register_embeddings returns a CorpusAutoGenEmbeddings instance.
    """
    emb = register_embeddings(
        corpus_adapter=adapter,
        model="mock-embed-512",
        framework_version="autogen-fw-1.0",
    )
    assert isinstance(emb, CorpusAutoGenEmbeddings)
    assert emb.corpus_adapter is adapter
    assert emb.model == "mock-embed-512"


# ---------------------------------------------------------------------------
# Translator Boundary
# ---------------------------------------------------------------------------

def test_translator_created_with_expected_args(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    """
    Pin the create_embedding_translator call signature/arguments.

    Ensures stable integration with the common embedding translation layer.
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def embed(self, raw_texts: Any, op_ctx: Any, framework_ctx: Any) -> Any:
            if isinstance(raw_texts, list):
                return [[0.0, 1.0] for _ in raw_texts]
            return [0.0, 1.0]

        async def arun_embed(self, raw_texts: Any, op_ctx: Any, framework_ctx: Any) -> Any:
            return self.embed(raw_texts, op_ctx, framework_ctx)

        def close(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

        def capabilities(self) -> Dict[str, Any]:
            return {}

        def health(self) -> Dict[str, Any]:
            return {}

        async def arun_capabilities(self) -> Dict[str, Any]:
            return {}

        async def arun_health(self) -> Dict[str, Any]:
            return {}

    def fake_create_embedding_translator(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return DummyTranslator()

    monkeypatch.setattr(autogen_adapter_module, "create_embedding_translator", fake_create_embedding_translator)

    embeddings = CorpusAutoGenEmbeddings(
        corpus_adapter=adapter,
        model="mock-embed-512",
        batch_config=None,
        text_normalization_config=None,
        framework_version="fv",
    )
    _ = embeddings.embed_documents(["x"])

    assert captured["adapter"] is adapter
    assert captured["framework"] == "autogen"
    assert captured["translator"] is None


def test_framework_ctx_contains_autogen_metadata(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    """
    Ensure framework_ctx contains AutoGen-specific metadata.
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

        def close(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

        def capabilities(self) -> Dict[str, Any]:
            return {}

        def health(self) -> Dict[str, Any]:
            return {}

        async def arun_capabilities(self) -> Dict[str, Any]:
            return {}

        async def arun_health(self) -> Dict[str, Any]:
            return {}

    monkeypatch.setattr(autogen_adapter_module, "create_embedding_translator", lambda **_: DummyTranslator())

    embeddings = _make_embeddings(adapter, model="mock-embed-512", framework_version="fv")

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
    assert fc["model"] == "mock-embed-512"
    assert fc["framework_version"] == "fv"
    assert fc["conversation_id"] == "c1"
    assert fc["agent_name"] == "agent1"
    assert fc["workflow_type"] == "chain"
    assert fc["retriever_name"] == "retr1"


# ---------------------------------------------------------------------------
# AutoGen Interface Compatibility (No skip; protocol-level)
# ---------------------------------------------------------------------------

def test_autogen_interface_compatibility(adapter) -> None:
    """
    Verify that CorpusAutoGenEmbeddings implements the expected embedding callable surface.

    Note:
    - This is an interface conformance test (no AutoGen import probing here).
    - Real AutoGen integration is validated in dedicated integration tests below.
    """
    embeddings = _make_embeddings(adapter)

    assert hasattr(embeddings, "embed_documents")
    assert hasattr(embeddings, "embed_query")
    assert hasattr(embeddings, "aembed_documents")
    assert hasattr(embeddings, "aembed_query")
    assert callable(embeddings)

    required_methods = ["__call__", "embed_documents", "embed_query", "aembed_documents", "aembed_query"]
    for method in required_methods:
        assert hasattr(embeddings, method), f"Missing required method: {method}"
        assert callable(getattr(embeddings, method)), f"Method not callable: {method}"

    result = embeddings(["test"])
    _assert_embedding_matrix_shape(result, expected_rows=1)


def test_module_import_does_not_require_autogen() -> None:
    """
    The adapter module must remain importable regardless of whether AutoGen is installed.

    This validates the optional dependency contract: AutoGen imports occur lazily
    inside integration helpers such as create_vector_memory().
    """
    assert autogen_adapter_module.__name__.endswith("framework_adapters.autogen")


# ---------------------------------------------------------------------------
# Context Translation / AutoGenContext Mapping
# ---------------------------------------------------------------------------

def test_autogen_context_passed_to_context_translation(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    """
    Verify that autogen_context is passed through to context_from_autogen.
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


def test_invalid_autogen_context_type_is_ignored(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    """
    autogen_context is best-effort. If the type is wrong (not a Mapping), embeddings should still work.
    """
    class DummyTranslator:
        def embed(self, raw_texts: Any, op_ctx: Any, framework_ctx: Any) -> Any:
            return [[0.0, 1.0] for _ in raw_texts]

        async def arun_embed(self, raw_texts: Any, op_ctx: Any, framework_ctx: Any) -> Any:
            return self.embed(raw_texts, op_ctx, framework_ctx)

        def close(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

        def capabilities(self) -> Dict[str, Any]:
            return {}

        def health(self) -> Dict[str, Any]:
            return {}

        async def arun_capabilities(self) -> Dict[str, Any]:
            return {}

        async def arun_health(self) -> Dict[str, Any]:
            return {}

    monkeypatch.setattr(autogen_adapter_module, "create_embedding_translator", lambda **_: DummyTranslator())

    embeddings = _make_embeddings(adapter)
    out = embeddings.embed_documents(["x"], autogen_context="not-a-mapping")  # type: ignore[arg-type]
    _assert_embedding_matrix_shape(out, expected_rows=1)


def test_context_translation_failure_attaches_context_but_does_not_break(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
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

        def close(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

        def capabilities(self) -> Dict[str, Any]:
            return {}

        def health(self) -> Dict[str, Any]:
            return {}

        async def arun_capabilities(self) -> Dict[str, Any]:
            return {}

        async def arun_health(self) -> Dict[str, Any]:
            return {}

    monkeypatch.setattr(autogen_adapter_module, "context_from_autogen", boom)
    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)
    monkeypatch.setattr(autogen_adapter_module, "create_embedding_translator", lambda **_: DummyTranslator())

    embeddings = _make_embeddings(adapter, framework_version="fv")
    out = embeddings.embed_documents(["x"], autogen_context={"conversation_id": "c"})
    _assert_embedding_matrix_shape(out, expected_rows=1)
    assert calls["attached"] is True


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------

def test_embed_documents_rejects_non_string_items(adapter) -> None:
    """Clear error for type mismatches."""
    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        embeddings.embed_documents(["ok", 123])  # type: ignore[list-item]

    error_msg = str(exc.value)
    assert "embed_documents expects Sequence[str]" in error_msg
    assert "item 1" in error_msg


def test_embed_query_rejects_non_string(adapter) -> None:
    """Clear error for type mismatches."""
    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        embeddings.embed_query(123)  # type: ignore[arg-type]

    error_msg = str(exc.value)
    assert "embed_query expects str" in error_msg


def test_call_rejects_non_string_items(adapter) -> None:
    """__call__ is part of the public surface and must enforce the same input contract."""
    embeddings = _make_embeddings(adapter)

    with pytest.raises(TypeError):
        embeddings(["ok", 123])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Sync Semantics
# ---------------------------------------------------------------------------

def test_sync_embed_documents_and_query_basic(adapter) -> None:
    """Basic smoke test for sync embed_documents / embed_query behavior."""
    embeddings = _make_embeddings(adapter, model="mock-embed-512")

    texts = ["alpha", "beta", "gamma"]
    query = "delta"

    doc_vecs = embeddings.embed_documents(texts)
    _assert_embedding_matrix_shape(doc_vecs, expected_rows=len(texts))

    q_vec = embeddings.embed_query(query)
    _assert_embedding_vector_shape(q_vec)


def test_call_aliases_embed_documents(adapter) -> None:
    """__call__ should alias embed_documents for embedding_function protocols."""
    embeddings = _make_embeddings(adapter)

    texts = ["call-one", "call-two"]
    result = embeddings(texts)
    _assert_embedding_matrix_shape(result, expected_rows=len(texts))


def test_sync_embed_documents_with_autogen_context(adapter) -> None:
    """Context parameter stable."""
    embeddings = _make_embeddings(adapter)

    ctx = {"conversation_id": "conv-ctx", "agent_name": "tester"}
    result = embeddings.embed_documents(["ctx-one", "ctx-two"], autogen_context=ctx)
    _assert_embedding_matrix_shape(result, expected_rows=2)


# ---------------------------------------------------------------------------
# Async Semantics
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_embed_documents_and_query_basic(adapter) -> None:
    """Async aembed_documents / aembed_query should exist and produce compatible shapes."""
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
async def test_async_and_sync_same_dimension(adapter) -> None:
    """
    Sync/async parity: sync APIs must not be invoked on the running event loop.
    Use to_thread for sync calls inside async test.
    """
    embeddings = _make_embeddings(adapter)

    texts = ["dim-a", "dim-b"]
    query = "dim-q"

    sync_vecs = await asyncio.to_thread(embeddings.embed_documents, texts)
    async_vecs = await embeddings.aembed_documents(texts)

    sync_q = await asyncio.to_thread(embeddings.embed_query, query)
    async_q = await embeddings.aembed_query(query)

    assert len(sync_vecs) == len(async_vecs) == len(texts)
    if sync_vecs and async_vecs:
        assert len(sync_vecs[0]) == len(async_vecs[0])
    assert len(sync_q) == len(async_q)


@pytest.mark.asyncio
async def test_aembed_documents_rejects_non_string_items(adapter) -> None:
    """Consistent error messages for async API."""
    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        await embeddings.aembed_documents(["ok", object()])  # type: ignore[list-item]

    error_msg = str(exc.value)
    assert "aembed_documents expects Sequence[str]" in error_msg
    assert "item 1" in error_msg


@pytest.mark.asyncio
async def test_aembed_query_rejects_non_string(adapter) -> None:
    """Consistent error messages for async API."""
    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        await embeddings.aembed_query(123)  # type: ignore[arg-type]

    error_msg = str(exc.value)
    assert "aembed_query expects str" in error_msg


# ---------------------------------------------------------------------------
# Event-loop Guard (Hard contract: sync APIs must fail in a running loop)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sync_methods_raise_inside_event_loop(adapter) -> None:
    embeddings = _make_embeddings(adapter)

    with pytest.raises(RuntimeError) as exc:
        embeddings.embed_query("x")
    assert autogen_adapter_module.ErrorCodes.SYNC_WRAPPER_CALLED_IN_EVENT_LOOP in str(exc.value)

    with pytest.raises(RuntimeError) as exc:
        embeddings.embed_documents(["x"])
    assert autogen_adapter_module.ErrorCodes.SYNC_WRAPPER_CALLED_IN_EVENT_LOOP in str(exc.value)

    with pytest.raises(RuntimeError) as exc:
        embeddings(["x"])
    assert autogen_adapter_module.ErrorCodes.SYNC_WRAPPER_CALLED_IN_EVENT_LOOP in str(exc.value)


# ---------------------------------------------------------------------------
# Error-context Richness for Failures (Deterministic via translator injection)
# ---------------------------------------------------------------------------

def test_embed_documents_error_context_includes_autogen_fields(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    """Production debugging context is attached on sync failures."""
    class FailingTranslator:
        def embed(self, raw_texts: Any, op_ctx: Any = None, framework_ctx: Any = None) -> Any:
            raise RuntimeError("translator failed: Check model configuration and API limits")

    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)

    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter, model="mock-embed-512", framework_version="fv")
    monkeypatch.setattr(embeddings, "_translator", FailingTranslator())

    autogen_context = {
        "conversation_id": "conv-1",
        "agent_name": "analyst",
        "workflow_type": "chain",
        "retriever_name": "retr1",
    }

    with pytest.raises(RuntimeError) as exc_info:
        embeddings.embed_documents(["doc1", "doc2"], autogen_context=autogen_context)

    error_str = str(exc_info.value)
    assert "translator failed" in error_str
    assert "Check model configuration" in error_str

    assert captured["framework"] == "autogen"
    assert captured["operation"] == "embedding_documents"
    assert captured["error_codes"] == autogen_adapter_module.EMBEDDING_COERCION_ERROR_CODES
    assert captured["conversation_id"] == "conv-1"
    assert captured["agent_name"] == "analyst"
    assert captured["model"] == "mock-embed-512"
    assert captured["framework_version"] == "fv"
    assert captured.get("texts_count") == 2


@pytest.mark.asyncio
async def test_aembed_query_error_context_includes_autogen_fields(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    """Async errors include debugging context."""
    class FailingTranslator:
        async def arun_embed(self, raw_texts: Any, op_ctx: Any = None, framework_ctx: Any = None) -> Any:
            raise RuntimeError("translator failed: Verify API key and model access permissions")

    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)

    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter, model="mock-embed-512", framework_version="fv")
    monkeypatch.setattr(embeddings, "_translator", FailingTranslator())

    autogen_context = {"conversation_id": "conv-2", "agent_name": "tester"}

    with pytest.raises(RuntimeError) as exc_info:
        await embeddings.aembed_query("hello", autogen_context=autogen_context)

    error_str = str(exc_info.value)
    assert "translator failed" in error_str
    assert "Verify API key" in error_str

    assert captured["framework"] == "autogen"
    assert captured["operation"] == "embedding_query"
    assert captured["conversation_id"] == "conv-2"
    assert captured["agent_name"] == "tester"
    assert captured["model"] == "mock-embed-512"
    assert captured["framework_version"] == "fv"
    assert captured.get("text_len") == 5


def test_dim_hint_is_attached_to_later_errors(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    """
    After a successful embed, dim hint should be reflected in later error context as embedding_dim.
    """
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    class OkTranslator:
        def embed(self, raw_texts: Any, op_ctx: Any = None, framework_ctx: Any = None) -> Any:
            if isinstance(raw_texts, list):
                return [[0.0, 1.0, 2.0] for _ in raw_texts]  # dim=3
            return [0.0, 1.0, 2.0]

    class BoomTranslator:
        def embed(self, raw_texts: Any, op_ctx: Any = None, framework_ctx: Any = None) -> Any:
            raise RuntimeError("boom")

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)

    embeddings = _make_embeddings(adapter, model="m", framework_version="fv")
    monkeypatch.setattr(embeddings, "_translator", OkTranslator())

    _ = embeddings.embed_documents(["a", "b"])
    assert embeddings._embedding_dim_hint == 3

    monkeypatch.setattr(embeddings, "_translator", BoomTranslator())
    with pytest.raises(RuntimeError, match="boom"):
        embeddings.embed_documents(["c"], autogen_context={"conversation_id": "c1"})

    assert captured.get("embedding_dim") == 3


# ---------------------------------------------------------------------------
# Capabilities / Health Passthrough (Stable async usage, no asyncio.run)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_capabilities_passthrough_when_underlying_provides(adapter) -> None:
    """
    acapabilities returns either a dict or EmbeddingCapabilities.
    """
    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)
    caps = await embeddings.acapabilities()
    assert isinstance(caps, (dict, EmbeddingCapabilities))


@pytest.mark.asyncio
async def test_async_capabilities_fallback_to_sync(adapter) -> None:
    """
    acapabilities should work regardless of whether the underlying supports async or sync.
    """
    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)
    acaps = await embeddings.acapabilities()
    assert isinstance(acaps, (dict, EmbeddingCapabilities))


def test_capabilities_empty_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    If the underlying adapter has no capabilities()/acapabilities(), the adapter should return {}.
    """
    class NoCapAdapter:
        async def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 3 for _ in texts]

    class DummyTranslator:
        def embed(self, raw_texts: Any, op_ctx: Any, framework_ctx: Any) -> Any:
            return [[0.0, 1.0, 2.0] for _ in raw_texts]

        async def arun_embed(self, raw_texts: Any, op_ctx: Any, framework_ctx: Any) -> Any:
            return self.embed(raw_texts, op_ctx, framework_ctx)

        def capabilities(self) -> Dict[str, Any]:
            return {}

        def health(self) -> Dict[str, Any]:
            return {}

        async def arun_capabilities(self) -> Dict[str, Any]:
            return {}

        async def arun_health(self) -> Dict[str, Any]:
            return {}

        def close(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=NoCapAdapter())
    monkeypatch.setattr(embeddings, "_translator", DummyTranslator())

    caps = embeddings.capabilities()
    assert isinstance(caps, dict)
    assert caps == {}


def test_health_passthrough_and_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    health/ahealth mirrors capabilities: passthrough when available, {} when not.
    """
    class HealthAdapter:
        async def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 2 for _ in texts]

    class DummyTranslator:
        def embed(self, raw_texts: Any, op_ctx: Any, framework_ctx: Any) -> Any:
            return [[0.0, 1.0] for _ in raw_texts]

        async def arun_embed(self, raw_texts: Any, op_ctx: Any, framework_ctx: Any) -> Any:
            return self.embed(raw_texts, op_ctx, framework_ctx)

        def capabilities(self) -> Dict[str, Any]:
            return {}

        def health(self) -> Dict[str, Any]:
            return {"status": "ok"}

        async def arun_capabilities(self) -> Dict[str, Any]:
            return {}

        async def arun_health(self) -> Dict[str, Any]:
            return {"status": "ok"}

        def close(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=HealthAdapter())
    monkeypatch.setattr(embeddings, "_translator", DummyTranslator())

    health = embeddings.health()
    assert isinstance(health, dict)
    assert health.get("status") == "ok"

    class NoHealthAdapter:
        async def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 2 for _ in texts]

    class DummyTranslatorNoHealth(DummyTranslator):
        def health(self) -> Dict[str, Any]:
            return {}

        async def arun_health(self) -> Dict[str, Any]:
            return {}

    embeddings2 = CorpusAutoGenEmbeddings(corpus_adapter=NoHealthAdapter())
    monkeypatch.setattr(embeddings2, "_translator", DummyTranslatorNoHealth())

    health2 = embeddings2.health()
    assert isinstance(health2, dict)
    assert health2 == {}


# ---------------------------------------------------------------------------
# Resource Management (Correct contract: translator is closed)
# ---------------------------------------------------------------------------

def test_context_manager_closes_translator(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    """Sync context manager exit closes translator (not the underlying adapter directly)."""
    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)
    called = {"close": False}

    class DummyTranslator:
        def close(self) -> None:
            called["close"] = True

    monkeypatch.setattr(embeddings, "_translator", DummyTranslator())
    with embeddings:
        pass
    assert called["close"] is True


@pytest.mark.asyncio
async def test_async_context_manager_closes_translator(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    """Async context manager exit closes translator (not the underlying adapter directly)."""
    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)
    called = {"aclose": False}

    class DummyTranslator:
        async def aclose(self) -> None:
            called["aclose"] = True

    monkeypatch.setattr(embeddings, "_translator", DummyTranslator())
    async with embeddings:
        pass
    assert called["aclose"] is True


# ---------------------------------------------------------------------------
# Concurrency Tests
# ---------------------------------------------------------------------------

@pytest.mark.concurrency
def test_shared_embedder_thread_safety(adapter) -> None:
    """Shared embedder is thread-safe for concurrent access."""
    embedder = register_embeddings(adapter, model="mock-embed-512")

    def embed_query(text: str) -> List[float]:
        return embedder.embed_query(text)

    texts = [f"query {i}" for i in range(10)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(embed_query, text) for text in texts]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == len(texts)
    for result in results:
        assert isinstance(result, list)
        assert all(isinstance(x, (int, float)) for x in result)


@pytest.mark.asyncio
@pytest.mark.concurrency
async def test_concurrent_async_embedding(adapter) -> None:
    """Async embedding supports concurrent operations."""
    embedder = register_embeddings(adapter, model="mock-embed-512")

    async def embed_async(text: str) -> List[float]:
        return await embedder.aembed_query(text)

    texts = [f"async query {i}" for i in range(5)]
    results = await asyncio.gather(*[embed_async(text) for text in texts])

    assert len(results) == len(texts)
    for result in results:
        assert isinstance(result, list)
        assert all(isinstance(x, (int, float)) for x in result)


# ---------------------------------------------------------------------------
# Real AutoGen Integration Tests (Pass/Fail; never skip)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def require_autogen_chromadb():
    """
    Hard requirement for real AutoGen integration tests.

    Policy:
    - No skip. If AutoGen or chromadb extras are missing or incompatible, fail-fast.
    - Capability-based checks (feature detection) rather than brittle version comparisons.
    """
    try:
        from autogen_ext.memory import chromadb as chroma_mod  # type: ignore[import-not-found]
        from autogen_core import memory as core_mem_mod  # type: ignore[import-not-found]
    except Exception as exc:
        pytest.fail(
            "AutoGen Chroma memory dependencies are required for integration tests. "
            'Install: pip install -U "autogen-agentchat" "autogen-core" "autogen-ext[chromadb]". '
            f"Import error: {exc!r}",
            pytrace=False,
        )

    required_chroma_attrs = ("ChromaDBVectorMemory", "PersistentChromaDBVectorMemoryConfig", "CustomEmbeddingFunctionConfig")
    for attr in required_chroma_attrs:
        if not hasattr(chroma_mod, attr):
            pytest.fail(
                f"AutoGen chromadb module missing required attribute '{attr}'. "
                "Your autogen-ext[chromadb] installation appears incompatible with this integration.",
                pytrace=False,
            )

    required_core_attrs = ("MemoryContent", "MemoryMimeType")
    for attr in required_core_attrs:
        if not hasattr(core_mem_mod, attr):
            pytest.fail(
                f"AutoGen core memory module missing required attribute '{attr}'. "
                "Your autogen-core installation appears incompatible with this integration.",
                pytrace=False,
            )

    return chroma_mod, core_mem_mod


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_autogen_chromadb_memory_roundtrip_uses_corpus_embeddings(
    require_autogen_chromadb,
    monkeypatch: pytest.MonkeyPatch,
    adapter,
    tmp_path,
) -> None:
    """
    Real integration (pass/fail): create_vector_memory wires into real AutoGen ChromaDBVectorMemory
    and actually uses Corpus embeddings.

    This test:
    - Uses create_vector_memory() to build a real ChromaDBVectorMemory
    - Adds MemoryContent, queries for it, and validates results
    - Wraps adapter embedding entry points to confirm embeddings are computed via the configured
      Corpus adapter during AutoGen memory add/query.

    IMPORTANT:
    - The embedding translation layer may legitimately route to either unary `embed()` or
      batch `embed_batch()` depending on adapter capabilities and batching policy.
    - This test validates the integration contract ("Corpus adapter is used") without
      pinning to a specific internal call path.
    """
    _chroma_mod, core_mem_mod = require_autogen_chromadb
    MemoryContent = core_mem_mod.MemoryContent
    MemoryMimeType = core_mem_mod.MemoryMimeType

    calls: Dict[str, Any] = {"n": 0, "methods": set()}

    def _wrap_counter(method_name: str) -> None:
        """
        Wrap an adapter method (if present/callable) to count invocations.

        Notes:
        - We patch the *instance* attribute, which is sufficient for this test and avoids
          altering global class behavior across the suite.
        - We preserve async vs sync behavior so the integration stack continues to function.
        """
        orig = getattr(adapter, method_name, None)
        if not callable(orig):
            return

        if inspect.iscoroutinefunction(orig):
            async def wrapped(*args: Any, **kwargs: Any) -> Any:
                calls["n"] += 1
                calls["methods"].add(method_name)
                return await orig(*args, **kwargs)

            monkeypatch.setattr(adapter, method_name, wrapped, raising=True)
        else:
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                calls["n"] += 1
                calls["methods"].add(method_name)
                return orig(*args, **kwargs)

            monkeypatch.setattr(adapter, method_name, wrapped, raising=True)

    # Count calls across BOTH common embedding surfaces.
    _wrap_counter("embed")
    _wrap_counter("embed_batch")

    # If neither exists, this adapter cannot support the integration contract this test validates.
    assert callable(getattr(adapter, "embed", None)) or callable(getattr(adapter, "embed_batch", None)), (
        "Adapter must expose embed() and/or embed_batch() for AutoGen embedding integration."
    )

    memory = create_vector_memory(
        corpus_adapter=adapter,
        collection_name=_unique_collection("corpus_autogen_memory_it"),
        persistence_path=str(tmp_path),
        model="mock-embed-512",
        k=3,
        score_threshold=None,
    )

    try:
        await memory.add(
            MemoryContent(
                content="The user prefers temperatures in Celsius",
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "preferences"},
            )
        )

        result = await memory.query("temperature units")
        assert hasattr(result, "results"), "AutoGen MemoryQueryResult must expose .results"
        assert isinstance(result.results, list)
        assert len(result.results) >= 1

        # Integration contract: at least one adapter embedding method must be invoked
        # during add/query. We accept either unary or batch entry points.
        assert calls["n"] >= 1, (
            "Expected the Corpus adapter to be used for embedding via AutoGen memory add/query, "
            "but no embedding method was invoked. "
            f"Observed methods: {sorted(calls['methods'])}"
        )

        assert any(
            getattr(item, "content", None) == "The user prefers temperatures in Celsius"
            for item in result.results
        ), "Expected stored MemoryContent to be returned by query()"
    finally:
        await memory.close()

# ---------------------------------------------------------------------------
# Additional Real Chroma Coverage (6 tests)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_autogen_chromadb_persistence_reload_roundtrip(
    require_autogen_chromadb,
    adapter,
    tmp_path,
) -> None:
    """
    Real integration: persistence_path must actually persist content across memory instances.

    Contract:
    - Add content to a persistent store
    - Close memory
    - Re-open a new memory pointed at the same persistence_path + collection_name
    - Query and confirm stored content is returned
    """
    _chroma_mod, core_mem_mod = require_autogen_chromadb
    MemoryContent = core_mem_mod.MemoryContent
    MemoryMimeType = core_mem_mod.MemoryMimeType

    collection = _unique_collection("corpus_autogen_persist")
    persist_path = tmp_path / "persist_db"

    memory1 = create_vector_memory(
        corpus_adapter=adapter,
        collection_name=collection,
        persistence_path=str(persist_path),
        model="mock-embed-512",
        k=5,
        score_threshold=None,
    )

    try:
        await memory1.add(
            MemoryContent(
                content="Persistent memory: apples are red",
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "facts"},
            )
        )
    finally:
        await memory1.close()

    memory2 = create_vector_memory(
        corpus_adapter=adapter,
        collection_name=collection,
        persistence_path=str(persist_path),
        model="mock-embed-512",
        k=5,
        score_threshold=None,
    )

    try:
        result = await memory2.query("apples color")
        assert hasattr(result, "results")
        assert isinstance(result.results, list)
        assert any(getattr(item, "content", None) == "Persistent memory: apples are red" for item in result.results), (
            "Expected persisted content to be returned after reopening the vector memory"
        )
    finally:
        await memory2.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_autogen_chromadb_k_is_respected_on_query_results(
    require_autogen_chromadb,
    adapter,
    tmp_path,
) -> None:
    """
    Real integration: the configured `k` should cap the number of returned results.

    We do NOT assume perfect ranking stability; we enforce only:
      - len(results) <= k
      - len(results) >= 1 after adding content
    """
    _chroma_mod, core_mem_mod = require_autogen_chromadb
    MemoryContent = core_mem_mod.MemoryContent
    MemoryMimeType = core_mem_mod.MemoryMimeType

    persist_path = tmp_path / "k_respected_db"

    async def _seed(memory) -> None:
        # Add multiple distinct items so a capped k has an opportunity to matter.
        for i in range(10):
            await memory.add(
                MemoryContent(
                    content=f"k-test-item-{i}: fruit facts",
                    mime_type=MemoryMimeType.TEXT,
                    metadata={"i": i},
                )
            )

    # k=1
    col1 = _unique_collection("corpus_autogen_k1")
    mem_k1 = create_vector_memory(
        corpus_adapter=adapter,
        collection_name=col1,
        persistence_path=str(persist_path),
        model="mock-embed-512",
        k=1,
        score_threshold=None,
    )
    try:
        await _seed(mem_k1)
        r1 = await mem_k1.query("fruit")
        assert hasattr(r1, "results")
        assert isinstance(r1.results, list)
        assert 1 <= len(r1.results) <= 1, f"Expected k=1 to cap results at 1, got {len(r1.results)}"
    finally:
        await mem_k1.close()

    # k=5
    col5 = _unique_collection("corpus_autogen_k5")
    mem_k5 = create_vector_memory(
        corpus_adapter=adapter,
        collection_name=col5,
        persistence_path=str(persist_path),
        model="mock-embed-512",
        k=5,
        score_threshold=None,
    )
    try:
        await _seed(mem_k5)
        r5 = await mem_k5.query("fruit")
        assert hasattr(r5, "results")
        assert isinstance(r5.results, list)
        assert 1 <= len(r5.results) <= 5, f"Expected k=5 to cap results at 5, got {len(r5.results)}"
    finally:
        await mem_k5.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_autogen_chromadb_score_threshold_filters_results_in_some_direction(
    require_autogen_chromadb,
    adapter,
    tmp_path,
) -> None:
    """
    Real integration: `score_threshold` should have a filtering effect.

    Because different AutoGen/Chroma versions interpret "score_threshold" differently
    (similarity >= threshold vs distance <= threshold), this test is direction-agnostic:

    - Run a baseline query with score_threshold=None (expect >1 results).
    - Re-open with extremely-low and extremely-high thresholds.
    - Assert that at least one of those extremes reduces result count vs baseline.
      If neither reduces, score_threshold is being ignored and this test fails.
    """
    _chroma_mod, core_mem_mod = require_autogen_chromadb
    MemoryContent = core_mem_mod.MemoryContent
    MemoryMimeType = core_mem_mod.MemoryMimeType

    collection = _unique_collection("corpus_autogen_threshold")
    persist_path = tmp_path / "threshold_db"

    mem_base = create_vector_memory(
        corpus_adapter=adapter,
        collection_name=collection,
        persistence_path=str(persist_path),
        model="mock-embed-512",
        k=10,
        score_threshold=None,
    )

    try:
        for i in range(10):
            await mem_base.add(
                MemoryContent(
                    content=f"threshold-item-{i}: animals and facts",
                    mime_type=MemoryMimeType.TEXT,
                    metadata={"i": i},
                )
            )
        base = await mem_base.query("animals")
        assert hasattr(base, "results")
        assert isinstance(base.results, list)
        assert len(base.results) >= 2, (
            "Baseline query should return multiple results so threshold filtering can be observed"
        )
        base_n = len(base.results)
    finally:
        await mem_base.close()

    mem_low = create_vector_memory(
        corpus_adapter=adapter,
        collection_name=collection,
        persistence_path=str(persist_path),
        model="mock-embed-512",
        k=10,
        score_threshold=0.0,
    )
    try:
        low = await mem_low.query("animals")
        assert hasattr(low, "results")
        assert isinstance(low.results, list)
        low_n = len(low.results)
    finally:
        await mem_low.close()

    mem_high = create_vector_memory(
        corpus_adapter=adapter,
        collection_name=collection,
        persistence_path=str(persist_path),
        model="mock-embed-512",
        k=10,
        score_threshold=1e9,
    )
    try:
        high = await mem_high.query("animals")
        assert hasattr(high, "results")
        assert isinstance(high.results, list)
        high_n = len(high.results)
    finally:
        await mem_high.close()

    assert (low_n < base_n) or (high_n < base_n), (
        "score_threshold did not reduce result count under either extreme (0.0 or 1e9). "
        "This suggests score_threshold is ignored by the installed AutoGen/Chroma integration."
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_autogen_chromadb_batch_embedding_path_is_exercised_when_supported(
    require_autogen_chromadb,
    monkeypatch: pytest.MonkeyPatch,
    adapter,
    tmp_path,
) -> None:
    """
    Real integration: exercise batch embedding behavior.

    Strategy:
    - Wrap adapter embedding entry points and record:
        * total calls
        * largest batch size observed across calls
    - Attempt to call memory.add() with a list of MemoryContent objects (batch add).
      If the installed AutoGen memory only supports single-item add(), we fall back
      to sequential adds and assert that embedding was called multiple times.
    - The test passes when either:
      (A) a batch add is supported and we observe an embedding call with batch size >= 2, OR
      (B) batch add is not supported but the system still performs multiple embedding calls
          (reflecting the actual backend behavior).

    IMPORTANT:
    - The translation layer may invoke either `embed()` or `embed_batch()` depending on adapter
      capabilities and batching decisions. We track both without enforcing internal choices
      beyond the observable batch-size behavior when batch add is supported.
    """
    _chroma_mod, core_mem_mod = require_autogen_chromadb
    MemoryContent = core_mem_mod.MemoryContent
    MemoryMimeType = core_mem_mod.MemoryMimeType

    observed: Dict[str, Any] = {"max_batch": 0, "calls": 0, "methods": set()}

    def _maybe_batch_len_from_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
        # Common patterns:
        #   embed(texts=[...]) or embed([...]) or embed(raw_texts=[...])
        if args:
            first = args[0]
            if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
                try:
                    return len(first)
                except Exception:
                    return 0
        for key in ("texts", "raw_texts", "input", "inputs"):
            v = kwargs.get(key)
            if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
                try:
                    return len(v)
                except Exception:
                    return 0
        return 0

    def _wrap_batch_probe(method_name: str) -> None:
        """
        Wrap an adapter method (if present/callable) to count calls and track max batch size.

        Notes:
        - This wrapper is intentionally tolerant of different adapter signatures.
        - We patch the *instance* attribute for isolation to this test.
        """
        orig = getattr(adapter, method_name, None)
        if not callable(orig):
            return

        if inspect.iscoroutinefunction(orig):
            async def wrapped(*args: Any, **kwargs: Any) -> Any:
                observed["calls"] += 1
                observed["methods"].add(method_name)
                observed["max_batch"] = max(observed["max_batch"], _maybe_batch_len_from_args(args, kwargs))
                return await orig(*args, **kwargs)

            monkeypatch.setattr(adapter, method_name, wrapped, raising=True)
        else:
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                observed["calls"] += 1
                observed["methods"].add(method_name)
                observed["max_batch"] = max(observed["max_batch"], _maybe_batch_len_from_args(args, kwargs))
                return orig(*args, **kwargs)

            monkeypatch.setattr(adapter, method_name, wrapped, raising=True)

    # Probe both potential embedding surfaces.
    _wrap_batch_probe("embed")
    _wrap_batch_probe("embed_batch")

    assert callable(getattr(adapter, "embed", None)) or callable(getattr(adapter, "embed_batch", None)), (
        "Adapter must expose embed() and/or embed_batch() for batch-path integration test."
    )

    memory = create_vector_memory(
        corpus_adapter=adapter,
        collection_name=_unique_collection("corpus_autogen_batch"),
        persistence_path=str(tmp_path / "batch_db"),
        model="mock-embed-512",
        k=5,
        score_threshold=None,
    )

    try:
        items = [
            MemoryContent(content="Batch item one: dogs", mime_type=MemoryMimeType.TEXT, metadata={"i": 1}),
            MemoryContent(content="Batch item two: cats", mime_type=MemoryMimeType.TEXT, metadata={"i": 2}),
            MemoryContent(content="Batch item three: birds", mime_type=MemoryMimeType.TEXT, metadata={"i": 3}),
        ]

        batch_add_supported = True
        try:
            await memory.add(items)  # type: ignore[arg-type]
        except TypeError:
            batch_add_supported = False
            for it in items:
                await memory.add(it)

        _ = await memory.query("pets")

        # Sanity: integration must have caused embedding work.
        assert observed["calls"] >= 1, (
            "Expected at least one embedding call during add/query, but none were observed. "
            f"Observed methods: {sorted(observed['methods'])}"
        )

        if batch_add_supported:
            # Batch add supported: we expect at least one embedding invocation with a batch size >= 2.
            assert observed["max_batch"] >= 2, (
                "memory.add(list[MemoryContent]) succeeded, but no adapter embedding call was observed "
                "with a batch size >= 2. This suggests batching is not flowing through to the embedding "
                "adapter as expected. "
                f"Observed max_batch={observed['max_batch']} methods={sorted(observed['methods'])}"
            )
        else:
            # Batch add not supported: sequential adds should still yield multiple embedding calls.
            assert observed["calls"] >= 2, (
                "AutoGen memory does not support batch add() in this environment, but we still expect multiple "
                "embedding calls across sequential adds and query. "
                f"Observed calls={observed['calls']} methods={sorted(observed['methods'])}"
            )
    finally:
        await memory.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_autogen_chromadb_collection_isolation_same_persistence_path(
    require_autogen_chromadb,
    adapter,
    tmp_path,
) -> None:
    """
    Real integration: two collections under the same persistence_path should be isolated.

    Contract:
    - Add content to collection A
    - Query collection B for the same concept
    - Collection B must NOT return the content from collection A
    """
    _chroma_mod, core_mem_mod = require_autogen_chromadb
    MemoryContent = core_mem_mod.MemoryContent
    MemoryMimeType = core_mem_mod.MemoryMimeType

    persist_path = tmp_path / "isolation_db"
    col_a = _unique_collection("corpus_autogen_iso_a")
    col_b = _unique_collection("corpus_autogen_iso_b")

    mem_a = create_vector_memory(
        corpus_adapter=adapter,
        collection_name=col_a,
        persistence_path=str(persist_path),
        model="mock-embed-512",
        k=5,
        score_threshold=None,
    )
    mem_b = create_vector_memory(
        corpus_adapter=adapter,
        collection_name=col_b,
        persistence_path=str(persist_path),
        model="mock-embed-512",
        k=5,
        score_threshold=None,
    )

    try:
        await mem_a.add(
            MemoryContent(
                content="Isolation test content: Jupiter is a gas giant",
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "astronomy"},
            )
        )

        rb = await mem_b.query("Jupiter gas giant")
        assert hasattr(rb, "results")
        assert isinstance(rb.results, list)

        assert not any(
            getattr(item, "content", None) == "Isolation test content: Jupiter is a gas giant"
            for item in rb.results
        ), (
            "Collection isolation violated: collection B returned content stored only in collection A"
        )
    finally:
        await mem_a.close()
        await mem_b.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_autogen_chromadb_metadata_roundtrip_on_retrieved_items(
    require_autogen_chromadb,
    adapter,
    tmp_path,
) -> None:
    """
    Real integration: metadata must round-trip for stored MemoryContent.

    Contract:
    - Add MemoryContent with metadata
    - Query for it
    - Retrieved result item corresponding to that content must preserve metadata keys/values
    """
    _chroma_mod, core_mem_mod = require_autogen_chromadb
    MemoryContent = core_mem_mod.MemoryContent
    MemoryMimeType = core_mem_mod.MemoryMimeType

    collection = _unique_collection("corpus_autogen_meta")
    memory = create_vector_memory(
        corpus_adapter=adapter,
        collection_name=collection,
        persistence_path=str(tmp_path / "meta_db"),
        model="mock-embed-512",
        k=5,
        score_threshold=None,
    )

    content_text = "Metadata roundtrip: user likes espresso"
    meta = {"category": "preferences", "drink": "espresso"}

    try:
        await memory.add(
            MemoryContent(
                content=content_text,
                mime_type=MemoryMimeType.TEXT,
                metadata=dict(meta),
            )
        )

        result = await memory.query("espresso preference")
        assert hasattr(result, "results")
        assert isinstance(result.results, list)
        assert result.results, "Expected at least one result after storing MemoryContent"

        matched = None
        for item in result.results:
            if getattr(item, "content", None) == content_text:
                matched = item
                break

        assert matched is not None, "Expected stored content to appear in query results"

        retrieved_meta = getattr(matched, "metadata", None)
        assert isinstance(retrieved_meta, Mapping), (
            "Expected retrieved MemoryContent to expose a Mapping metadata attribute"
        )
        for k, v in meta.items():
            assert retrieved_meta.get(k) == v, (
                f"Expected metadata key {k!r} to round-trip with value {v!r}; "
                f"got {retrieved_meta.get(k)!r}"
            )
    finally:
        await memory.close()


# ---------------------------------------------------------------------------
# create_vector_memory Integration Tests (Deterministic, no AutoGen required)
# ---------------------------------------------------------------------------

def test_create_vector_memory_raises_runtime_error_when_autogen_not_installed(
    monkeypatch: pytest.MonkeyPatch,
    adapter,
) -> None:
    """Clear error when AutoGen Chroma dependencies are not installed."""
    import builtins as _builtins

    orig_import = _builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("autogen_ext.memory.chromadb"):
            raise ImportError("autogen-ext[chromadb] not installed")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(_builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as exc_info:
        create_vector_memory(corpus_adapter=adapter, collection_name="test-collection")

    msg = str(exc_info.value)
    assert "AutoGen Chroma memory dependencies are not installed" in msg
    assert "autogen-ext[chromadb]" in msg


def test_create_vector_memory_configures_chroma_with_custom_embedding_function(
    monkeypatch: pytest.MonkeyPatch,
    adapter,
) -> None:
    """create_vector_memory wires up Chroma config and embedding function correctly."""
    chroma_mod = types.ModuleType("autogen_ext.memory.chromadb")

    class DummyEmbeddingFunctionConfig:
        def __init__(self, function: Any, params: Dict[str, Any]) -> None:
            self.function = function
            self.params = params

    class DummyConfig:
        def __init__(
            self,
            collection_name: str,
            persistence_path: Optional[str] = None,
            embedding_function_config: Optional[DummyEmbeddingFunctionConfig] = None,
            k: int = 3,
            score_threshold: Optional[float] = None,
        ) -> None:
            self.collection_name = collection_name
            self.persistence_path = persistence_path
            self.embedding_function_config = embedding_function_config
            self.k = k
            self.score_threshold = score_threshold

    class DummyMemory:
        def __init__(self, config: DummyConfig) -> None:
            self.config = config

    chroma_mod.CustomEmbeddingFunctionConfig = DummyEmbeddingFunctionConfig
    chroma_mod.PersistentChromaDBVectorMemoryConfig = DummyConfig
    chroma_mod.ChromaDBVectorMemory = DummyMemory

    monkeypatch.setitem(sys.modules, "autogen_ext", types.ModuleType("autogen_ext"))
    monkeypatch.setitem(sys.modules, "autogen_ext.memory", types.ModuleType("autogen_ext.memory"))
    monkeypatch.setitem(sys.modules, "autogen_ext.memory.chromadb", chroma_mod)

    memory = create_vector_memory(
        corpus_adapter=adapter,
        collection_name="corpus_autogen_memory",
        persistence_path="/tmp/chroma",
        model="mock-embed-512",
        batch_config=None,
        text_normalization_config=None,
        autogen_config={"foo": "bar"},
        framework_version="fw-1.0",
        k=5,
        score_threshold=0.42,
    )

    assert isinstance(memory, DummyMemory)
    cfg = memory.config
    assert cfg.collection_name == "corpus_autogen_memory"
    assert cfg.persistence_path == "/tmp/chroma"
    assert cfg.k == 5
    assert cfg.score_threshold == 0.42

    emb_cfg = cfg.embedding_function_config
    assert isinstance(emb_cfg, DummyEmbeddingFunctionConfig)

    embedding_fn = emb_cfg.function(**emb_cfg.params)
    assert isinstance(embedding_fn, CorpusAutoGenEmbeddings)
    assert embedding_fn.corpus_adapter is adapter
    assert embedding_fn.model == "mock-embed-512"
    assert embedding_fn.autogen_config == {"foo": "bar"}
    assert embedding_fn._framework_version == "fw-1.0"


def test_create_vector_memory_uses_defaults_when_optional_args_omitted(
    monkeypatch: pytest.MonkeyPatch,
    adapter,
) -> None:
    """create_vector_memory applies sensible defaults when args are omitted."""
    chroma_mod = types.ModuleType("autogen_ext.memory.chromadb")

    class DummyEmbeddingFunctionConfig:
        def __init__(self, function: Any, params: Dict[str, Any]) -> None:
            self.function = function
            self.params = params

    class DummyConfig:
        def __init__(
            self,
            collection_name: str,
            persistence_path: Optional[str] = None,
            embedding_function_config: Optional[DummyEmbeddingFunctionConfig] = None,
            k: int = 3,
            score_threshold: Optional[float] = None,
        ) -> None:
            self.collection_name = collection_name
            self.persistence_path = persistence_path
            self.embedding_function_config = embedding_function_config
            self.k = k
            self.score_threshold = score_threshold

    class DummyMemory:
        def __init__(self, config: DummyConfig) -> None:
            self.config = config

    chroma_mod.CustomEmbeddingFunctionConfig = DummyEmbeddingFunctionConfig
    chroma_mod.PersistentChromaDBVectorMemoryConfig = DummyConfig
    chroma_mod.ChromaDBVectorMemory = DummyMemory

    monkeypatch.setitem(sys.modules, "autogen_ext", types.ModuleType("autogen_ext"))
    monkeypatch.setitem(sys.modules, "autogen_ext.memory", types.ModuleType("autogen_ext.memory"))
    monkeypatch.setitem(sys.modules, "autogen_ext.memory.chromadb", chroma_mod)

    memory = create_vector_memory(corpus_adapter=adapter)

    assert isinstance(memory, DummyMemory)
    cfg = memory.config
    assert cfg.collection_name == "corpus_autogen_memory"
    assert cfg.persistence_path is None
    assert cfg.k == 3
    assert cfg.score_threshold is None

    emb_cfg = cfg.embedding_function_config
    assert isinstance(emb_cfg, DummyEmbeddingFunctionConfig)

    embedding_fn = emb_cfg.function(**emb_cfg.params)
    assert isinstance(embedding_fn, CorpusAutoGenEmbeddings)
    assert embedding_fn.corpus_adapter is adapter
    assert embedding_fn.model is None
    assert embedding_fn.autogen_config == {}
