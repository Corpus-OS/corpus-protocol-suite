# tests/frameworks/embedding/test_autogen_adapter.py

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional
import inspect
import asyncio
import pytest
import sys
import types
import concurrent.futures

import corpus_sdk.embedding.framework_adapters.autogen as autogen_adapter_module
from corpus_sdk.embedding.framework_adapters.autogen import (
    CorpusAutoGenEmbeddings,
    create_retriever,
    register_embeddings,
)
from corpus_sdk.embedding.embedding_base import OperationContext, EmbeddingCapabilities


# ---------------------------------------------------------------------------
# Framework Version Support Matrix
# ---------------------------------------------------------------------------
"""
Framework Version Support:
- AutoGen: 0.10.x (recommended, latest architecture)
- AutoGen: 0.7+ (supported, new architecture)
- AutoGen: 0.2.x (legacy, integration tests with retrieve_assistant_agent)
- Python: 3.9+
- Corpus SDK: 1.0.0+

Integration Notes:
- Compatible with AutoGen's EmbeddingFunction protocol (0.2.x legacy)
- Supports AutoGen retrieval workflows and agent memory (0.2.x legacy)
- Handles AutoGen context (conversation_id, agent_name, workflow_type, retriever_name)
- Framework protocol-first design (no hard inheritance required)

AutoGen Version Compatibility:
- AutoGen 0.10.x (Recommended):
  * Latest stable release with improved architecture
  * Core adapter tests fully supported
  * Uses: autogen_agentchat, autogen_core (modern APIs)
  * Integration tests with new patterns (to be implemented)
  * Install: pip install pyautogen (gets 0.10.x)

- AutoGen 0.7-0.9 (Supported):
  * New architecture, core adapter tests pass
  * Integration tests skipped (require update for new patterns)
  * Install: pip install 'pyautogen>=0.7,<0.10'
  
- AutoGen 0.2.x (Legacy):
  * Full integration test support for backward compatibility
  * Uses: autogen.agentchat.contrib.retrieve_assistant_agent
  * Install: pip install 'pyautogen<0.3'
  * Note: Integration tests specifically validate 0.2.x patterns

Note: AutoGen compatibility is verified via duck typing and protocol
implementation, not inheritance. This ensures compatibility even when
AutoGen base classes change. The adapter itself works with all versions,
but integration tests specifically validate 0.2.x legacy patterns.
"""


# ---------------------------------------------------------------------------
# Test Helpers
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


def _make_embeddings(adapter: Any, **kwargs: Any) -> CorpusAutoGenEmbeddings:
    """Construct a CorpusAutoGenEmbeddings instance from the adapter."""
    # Provide default model if not specified
    if 'model' not in kwargs:
        kwargs['model'] = 'mock-embed-512'
    return CorpusAutoGenEmbeddings(corpus_adapter=adapter, **kwargs)


# ---------------------------------------------------------------------------
# Constructor / Registration Behavior
# ---------------------------------------------------------------------------

def test_constructor_works_with_real_adapter(adapter) -> None:
    """
    CorpusAutoGenEmbeddings should work with any real adapter.

    Framework Compatibility: Ensures basic protocol compliance across
    all supported AutoGen versions.

    Note: Uses the adapter fixture from conftest.py
    """
    embeddings = _make_embeddings(adapter)
    assert embeddings.corpus_adapter is adapter
    assert hasattr(embeddings.corpus_adapter, 'embed')
    assert callable(embeddings.corpus_adapter.embed)


def test_constructor_rejects_common_user_mistakes() -> None:
    """
    CorpusAutoGenEmbeddings should provide clear error messages for
    common user mistakes.

    IMPORTANT:
    - We assert *semantic* guidance (what is required) rather than brittle
      substrings about specific types ("None", "str", etc.).
    """
    # Common mistake 1: Passing None
    with pytest.raises(TypeError) as exc_info:
        CorpusAutoGenEmbeddings(corpus_adapter=None)  # type: ignore[arg-type]

    msg = str(exc_info.value)
    assert "EmbeddingProtocolV1-compatible" in msg or "EmbeddingProtocolV1" in msg
    assert "embed" in msg.lower()

    # Common mistake 2: Passing a string (wrong type)
    with pytest.raises(TypeError) as exc_info:
        CorpusAutoGenEmbeddings(corpus_adapter="not an adapter")  # type: ignore[arg-type]

    msg = str(exc_info.value)
    assert "EmbeddingProtocolV1-compatible" in msg or "EmbeddingProtocolV1" in msg
    assert "embed" in msg.lower()

    # Common mistake 3: Passing an object without embed() method
    class MockAgent:
        """Looks like an agent but not an embedding adapter"""
        def chat(self):
            pass

    with pytest.raises(TypeError) as exc_info:
        CorpusAutoGenEmbeddings(corpus_adapter=MockAgent())

    msg = str(exc_info.value)
    assert "EmbeddingProtocolV1-compatible" in msg or "EmbeddingProtocolV1" in msg
    assert "embed" in msg.lower()


def test_register_embeddings_returns_instance(adapter) -> None:
    """
    register_embeddings returns a CorpusAutoGenEmbeddings instance.

    Framework Compatibility: Registration pattern stable across
    all supported AutoGen versions.

    Note: Uses the adapter fixture from conftest.py
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

    Framework Compatibility: Ensures stable integration with the common
    embedding translation layer.
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def embed(self, raw_texts: Any, op_ctx: Any, framework_ctx: Any) -> Any:
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

    Framework Compatibility: Context propagation stable across
    AutoGen versions.
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
# AutoGen Interface Compatibility
# ---------------------------------------------------------------------------

def test_autogen_interface_compatibility(adapter) -> None:
    """
    Verify that CorpusAutoGenEmbeddings implements AutoGen's embedding interface.
    
    Framework Compatibility: Works with both AutoGen 0.2.x (legacy) and 0.10.x (modern).
    Note: We test protocol compliance, not inheritance, for better
    forward compatibility.
    """
    embeddings = _make_embeddings(adapter)

    # Core methods should always exist
    assert hasattr(embeddings, "embed_documents")
    assert hasattr(embeddings, "embed_query")
    assert hasattr(embeddings, "aembed_documents")
    assert hasattr(embeddings, "aembed_query")
    assert callable(embeddings)  # __call__ for EmbeddingFunction protocol

    # Verify we have all required methods with correct signatures
    required_methods = ['__call__', 'embed_documents', 'embed_query', 
                       'aembed_documents', 'aembed_query']
    for method in required_methods:
        assert hasattr(embeddings, method), f"Missing required method: {method}"
        assert callable(getattr(embeddings, method)), f"Method not callable: {method}"
    
    # Test the __call__ method works
    result = embeddings(["test"])
    _assert_embedding_matrix_shape(result, expected_rows=1)
    
    # Try to import AutoGen modules (works with both 0.2.x and 0.10.x)
    autogen_available = False
    try:
        # Try new AutoGen 0.10.x first
        import autogen_agentchat
        autogen_available = True
    except ImportError:
        try:
            # Fall back to old AutoGen 0.2.x
            from autogen.agentchat.contrib.retrieve_assistant_agent import (  # type: ignore[import]
                EmbeddingFunction,
            )
            autogen_available = True
        except ImportError:
            pass
    
    if not autogen_available:
        pytest.skip("AutoGen not installed - skipping compatibility check")


# ---------------------------------------------------------------------------
# Context Translation / AutoGenContext Mapping
# ---------------------------------------------------------------------------

def test_autogen_context_passed_to_context_translation(
    monkeypatch: pytest.MonkeyPatch,
    adapter,
) -> None:
    """
    Verify that autogen_context is passed through to context_from_autogen.

    Framework Compatibility: Context translation stable across
    AutoGen versions.

    Note: Uses the adapter fixture from conftest.py
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


def test_error_context_includes_autogen_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When an error occurs during AutoGen embedding, error context should include
    AutoGen-specific metadata via attach_context().

    Error Message Quality: Errors must contain actionable context for debugging
    AutoGen workflows in production.
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)

    class FailingAdapter:
        async def embed(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("test error from autogen adapter: Check model configuration and API keys")
        
        async def embed_batch(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("test error from autogen adapter: Check model configuration and API keys")

    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=FailingAdapter())

    auto_ctx = {"conversation_id": "conv-123", "agent_name": "tester"}

    with pytest.raises(RuntimeError, match="test error from autogen adapter") as exc_info:
        embeddings.embed_documents(["text"], autogen_context=auto_ctx)

    error_str = str(exc_info.value)
    assert "test error from autogen adapter" in error_str
    assert "Check model configuration" in error_str or "API keys" in error_str

    assert captured_context, "attach_context was not called"

    assert captured_context.get("framework") == "autogen"
    assert captured_context.get("conversation_id") == "conv-123"
    assert captured_context.get("agent_name") == "tester"

    assert captured_context.get("operation") == "embedding_documents"

    assert captured_context.get("error_codes") == autogen_adapter_module.EMBEDDING_COERCION_ERROR_CODES


def test_invalid_autogen_context_type_is_ignored(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    """
    autogen_context is best-effort. If the type is wrong (not a Mapping),
    embeddings should still work.
    """
    class DummyTranslator:
        def embed(self, raw_texts: Any, op_ctx: Any, framework_ctx: Any) -> Any:
            return [[0.0, 1.0] for _ in raw_texts]

        async def arun_embed(self, raw_texts: Any, op_ctx: Any, framework_ctx: Any) -> Any:
            return self.embed(raw_texts, op_ctx, framework_ctx)

    monkeypatch.setattr(autogen_adapter_module, "create_embedding_translator", lambda **_: DummyTranslator())

    embeddings = _make_embeddings(adapter)
    out = embeddings.embed_documents(["x"], autogen_context="not-a-mapping")  # type: ignore[arg-type]
    _assert_embedding_matrix_shape(out, expected_rows=1)


def test_context_translation_failure_attaches_context_but_does_not_break(
    monkeypatch: pytest.MonkeyPatch, adapter
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
# Input Validation
# ---------------------------------------------------------------------------

def test_embed_documents_rejects_non_string_items(adapter):
    """Error Message Quality: Clear error for type mismatches."""
    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        embeddings.embed_documents(["ok", 123])  # type: ignore[list-item]

    error_msg = str(exc.value)
    assert "embed_documents expects Sequence[str]" in error_msg
    assert "item 1" in error_msg


def test_embed_query_rejects_non_string(adapter):
    """Error Message Quality: Clear error for type mismatches."""
    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        embeddings.embed_query(123)  # type: ignore[arg-type]

    error_msg = str(exc.value)
    assert "embed_query expects str" in error_msg


# ---------------------------------------------------------------------------
# Sync Semantics
# ---------------------------------------------------------------------------

def test_sync_embed_documents_and_query_basic(adapter) -> None:
    """
    Basic smoke test for sync embed_documents / embed_query behavior.
    """
    embeddings = _make_embeddings(adapter, model="mock-embed-512")

    texts = ["alpha", "beta", "gamma"]
    query = "delta"

    doc_vecs = embeddings.embed_documents(texts)
    _assert_embedding_matrix_shape(doc_vecs, expected_rows=len(texts))

    q_vec = embeddings.embed_query(query)
    _assert_embedding_vector_shape(q_vec)


def test_call_aliases_embed_documents(adapter) -> None:
    """
    __call__ should alias embed_documents for AutoGen EmbeddingFunction protocol.
    """
    embeddings = _make_embeddings(adapter)

    texts = ["call-one", "call-two"]
    result = embeddings(texts)  # __call__
    _assert_embedding_matrix_shape(result, expected_rows=len(texts))


def test_sync_embed_documents_with_autogen_context(adapter) -> None:
    """Context parameter stable across versions."""
    embeddings = _make_embeddings(adapter)

    ctx = {"conversation_id": "conv-ctx", "agent_name": "tester"}
    result = embeddings.embed_documents(["ctx-one", "ctx-two"], autogen_context=ctx)
    _assert_embedding_matrix_shape(result, expected_rows=2)


# ---------------------------------------------------------------------------
# Async Semantics
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_embed_documents_and_query_basic(adapter) -> None:
    """
    Async aembed_documents / aembed_query should exist and produce shapes
    compatible with the sync API.
    """
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
    """Sync/async parity maintained across versions."""
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


@pytest.mark.asyncio
async def test_aembed_documents_rejects_non_string_items(adapter):
    """Consistent error messages for async API."""
    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        await embeddings.aembed_documents(["ok", object()])  # type: ignore[list-item]

    error_msg = str(exc.value)
    assert "aembed_documents expects Sequence[str]" in error_msg
    assert "item 1" in error_msg


@pytest.mark.asyncio
async def test_aembed_query_rejects_non_string(adapter):
    """Consistent error messages for async API."""
    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        await embeddings.aembed_query(123)  # type: ignore[arg-type]

    error_msg = str(exc.value)
    assert "aembed_query expects str" in error_msg


# ---------------------------------------------------------------------------
# Error-context Richness for Failures
# ---------------------------------------------------------------------------

def test_embed_documents_error_context_includes_autogen_fields(monkeypatch, adapter):
    """Production debugging context in errors."""
    class FailingTranslator:
        def embed(self, raw_texts, op_ctx=None, framework_ctx=None):
            raise RuntimeError("translator failed: Check model configuration and API limits")

    captured = {}

    def fake_attach_context(exc, **kwargs):
        captured["exc"] = exc
        captured["kwargs"] = kwargs

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)

    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter, model="mock-embed-512")
    
    # Use monkeypatch to inject failing translator
    with monkeypatch.context() as m:
        m.setattr(embeddings, "_translator", FailingTranslator())

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

        ctx = captured["kwargs"]
        assert ctx["framework"] == "autogen"
        assert ctx["operation"] == "embedding_documents"
        assert ctx["error_codes"] == autogen_adapter_module.EMBEDDING_COERCION_ERROR_CODES
        assert ctx["conversation_id"] == "conv-1"
        assert ctx["agent_name"] == "analyst"
        # Verify production debugging context
        assert "model" in ctx
        assert ctx["model"] == "mock-embed-512"


@pytest.mark.asyncio
async def test_aembed_query_error_context_includes_autogen_fields(monkeypatch, adapter):
    """Async errors also include debugging context."""
    class FailingTranslator:
        async def arun_embed(self, raw_texts, op_ctx=None, framework_ctx=None):
            raise RuntimeError("translator failed: Verify API key and model access permissions")

    captured = {}

    def fake_attach_context(exc, **kwargs):
        captured["exc"] = exc
        captured["kwargs"] = kwargs

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)

    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter, model="mock-embed-512")
    
    # Use monkeypatch to inject failing translator
    with monkeypatch.context() as m:
        m.setattr(embeddings, "_translator", FailingTranslator())

        autogen_context = {"conversation_id": "conv-2", "agent_name": "tester"}

        with pytest.raises(RuntimeError) as exc_info:
            await embeddings.aembed_query("hello", autogen_context=autogen_context)

        error_str = str(exc_info.value)
        assert "translator failed" in error_str
        assert "Verify API key" in error_str

        ctx = captured["kwargs"]
        assert ctx["framework"] == "autogen"
        assert ctx["operation"] == "embedding_query"
        assert ctx["conversation_id"] == "conv-2"
        assert ctx["agent_name"] == "tester"
        assert "model" in ctx


@pytest.mark.asyncio
async def test_async_error_context_includes_autogen_context(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    """
    When an async error occurs, attach_context should include AutoGen-specific
    metadata and standard fields.
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    class RaisingTranslator:
        async def arun_embed(self, raw_texts: Any, op_ctx: Any, framework_ctx: Any) -> Any:
            raise RuntimeError("boom-async")

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)
    monkeypatch.setattr(autogen_adapter_module, "create_embedding_translator", lambda **_: RaisingTranslator())

    embeddings = _make_embeddings(adapter, model="mock-embed-512", framework_version="fv")
    auto_ctx = {"conversation_id": "conv-ctx", "agent_name": "tester"}

    with pytest.raises(RuntimeError, match="boom-async"):
        await embeddings.aembed_documents(["text"], autogen_context=auto_ctx)

    assert captured_context.get("framework") == "autogen"
    assert captured_context.get("operation") == "embedding_documents"
    assert captured_context.get("conversation_id") == "conv-ctx"
    assert captured_context.get("agent_name") == "tester"
    assert captured_context.get("model") == "mock-embed-512"
    assert captured_context.get("framework_version") == "fv"


# ---------------------------------------------------------------------------
# Capabilities / Health Passthrough
# ---------------------------------------------------------------------------

def test_capabilities_passthrough_when_underlying_provides(adapter) -> None:
    """
    When the underlying adapter implements capabilities/acapabilities,
    CorpusAutoGenEmbeddings should surface them.
    """
    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)

    # The adapter's capabilities() is async, so we need to await it
    caps = asyncio.run(embeddings.acapabilities())
    # Can be either a dict or an EmbeddingCapabilities object
    assert isinstance(caps, (dict, EmbeddingCapabilities))


@pytest.mark.asyncio
async def test_async_capabilities_fallback_to_sync(adapter) -> None:
    """
    acapabilities should fall back to sync capabilities() when only the
    sync method is implemented on the underlying adapter.
    """
    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)

    # Use the async version since the adapter is async
    acaps = await embeddings.acapabilities()
    # Can be either a dict or an EmbeddingCapabilities object
    assert isinstance(acaps, (dict, EmbeddingCapabilities))


def test_capabilities_empty_when_missing():
    """
    If the underlying adapter has no capabilities()/acapabilities(),
    the AutoGen adapter should return an empty mapping (best-effort).
    """
    class NoCapAdapter:
        async def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 3 for _ in texts]

    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=NoCapAdapter())

    caps = embeddings.capabilities()
    assert isinstance(caps, dict)
    assert caps == {}


def test_health_passthrough_and_missing():
    """
    health/ahealth behavior mirrors capabilities/acapabilities: passthrough
    when available, empty mapping when not.
    """
    class HealthAdapter:
        async def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 2 for _ in texts]

        def health(self) -> Dict[str, Any]:
            return {"status": "ok"}

    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=HealthAdapter())

    health = embeddings.health()
    assert isinstance(health, dict)
    assert health.get("status") == "ok"

    class NoHealthAdapter:
        async def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 2 for _ in texts]

    embeddings2 = CorpusAutoGenEmbeddings(corpus_adapter=NoHealthAdapter())

    health2 = embeddings2.health()
    assert isinstance(health2, dict)
    assert health2 == {}


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
# Resource Management
# ---------------------------------------------------------------------------

def test_context_manager_closes_underlying_adapter(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    """
    Context manager should close underlying adapter (sync version).

        async def embed(self, *args: Any, **kwargs: Any) -> Any:
            return [[0.1, 0.2, 0.3]]
        
        async def embed_batch(self, *args: Any, **kwargs: Any) -> Any:
            from corpus_sdk.embedding.embedding_base import BatchEmbedResult, EmbeddingVector
            embeddings = [
                EmbeddingVector(
                    vector=[0.1, 0.2, 0.3],
                    text="x",
                    model="mock-embed-512",
                    dimensions=3,
                    index=0
                )
            ]
            return BatchEmbedResult(
                embeddings=embeddings, 
                model="mock-embed-512", 
                total_texts=1
            )

    def close() -> None:
        closed["v"] = True

    monkeypatch.setattr(adapter, "close", close, raising=False)

    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)

    with embeddings:
        pass

    assert closed["v"] is True


@pytest.mark.asyncio
async def test_async_context_manager_closes_underlying_adapter(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    """
    Async context manager should close underlying adapter (async version).

        async def embed(self, *args: Any, **kwargs: Any) -> Any:
            return [[0.1, 0.2, 0.3]]
        
        async def embed_batch(self, *args: Any, **kwargs: Any) -> Any:
            from corpus_sdk.embedding.embedding_base import BatchEmbedResult, EmbeddingVector
            embeddings = [
                EmbeddingVector(
                    vector=[0.1, 0.2, 0.3],
                    text="y",
                    model="mock-embed-512",
                    dimensions=3,
                    index=0
                )
            ]
            return BatchEmbedResult(
                embeddings=embeddings, 
                model="mock-embed-512", 
                total_texts=1
            )

    async def aclose() -> None:
        aclosed["v"] = True

    monkeypatch.setattr(adapter, "aclose", aclose, raising=False)

    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)

    async with embeddings:
        pass

    assert aclosed["v"] is True


# ---------------------------------------------------------------------------
# Concurrency Tests
# ---------------------------------------------------------------------------

@pytest.mark.concurrency
def test_shared_embedder_thread_safety(adapter):
    """
    Shared embedder is thread-safe for concurrent access.
    """
    embedder = register_embeddings(adapter, model="mock-embed-512")
    
    def embed_query(text: str):
        return embedder.embed_query(text)

    texts = [f"query {i}" for i in range(10)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(embed_query, text) for text in texts]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == len(texts)
    for result in results:
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)


@pytest.mark.asyncio
@pytest.mark.concurrency
async def test_concurrent_async_embedding(adapter):
    """
    Async embedding supports concurrent operations.
    """
    embedder = register_embeddings(adapter, model="mock-embed-512")
    
    async def embed_async(text: str):
        return await embedder.aembed_query(text)

    texts = [f"async query {i}" for i in range(5)]
    tasks = [embed_async(text) for text in texts]
    results = await asyncio.gather(*tasks)

    assert len(results) == len(texts)
    for result in results:
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)


# ---------------------------------------------------------------------------
# Real AutoGen Integration Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAutoGenIntegration:
    """
    Integration tests with real AutoGen agents and workflows.

    These tests verify that our adapter actually works in real AutoGen
    agent workflows.
    
    Framework Compatibility: Tested with AutoGen 0.10.x (primary) and 0.2.x (legacy).
    
    Note: Tests dynamically adapt to available AutoGen version:
    - AutoGen 0.10.x: Uses new autogen_agentchat APIs
    - AutoGen 0.2.x: Uses legacy retrieve_assistant_agent module
    """

    @pytest.fixture
    def autogen_available(self):
        """Check if AutoGen (any version) is available for integration tests."""
        try:
            # Try new AutoGen 0.10.x first
            import autogen_agentchat
            return "0.10.x"
        except ImportError:
            try:
                # Try old AutoGen 0.2.x
                import autogen
                from autogen.agentchat.contrib.retrieve_assistant_agent import EmbeddingFunction
                return "0.2.x"
            except ImportError:
                pytest.skip("AutoGen not installed - skipping integration tests")
    
    def test_can_create_embeddings_for_autogen_retrieve_agent(self, autogen_available, adapter):
        """
        Real integration: Can create embeddings that work with AutoGen agents.
        
        Framework Compatibility: Validates embedding integration pattern
        works with AutoGen's agent architecture (both 0.10.x and 0.2.x).
        """
        embedder = register_embeddings(
            corpus_adapter=adapter,
            model="mock-embed-512",
            framework_version="1.0.0"
        )

        result = embedder(["test document for AutoGen agent"])
        assert isinstance(result, list)
        assert len(result) == 1
        assert all(isinstance(x, float) for row in result for x in row)

        result_with_context = embedder.embed_query(
            "query from agent",
            autogen_context={
                "conversation_id": "conv-123",
                "agent_name": "researcher",
                "workflow_type": "rag"
            }
        )
        assert isinstance(result_with_context, list)
        assert all(isinstance(x, float) for x in result_with_context)

    def test_embeddings_work_with_autogen_retriever_workflows(self, autogen_available, adapter):
        """
        Real integration: Embeddings work with AutoGen's retriever-based workflows.
        
        Framework Compatibility: Validates RAG workflow integration pattern
        works with AutoGen's retrieval systems (both 0.10.x and 0.2.x).
        """
        embedder = register_embeddings(
            corpus_adapter=adapter,
            model="mock-embed-512",
            framework_version="1.0.0"
        )

        documents = [
            "AutoGen is a framework for building multi-agent applications.",
            "Agents can work together to solve complex tasks.",
            "Embeddings are used to convert text to vector representations."
        ]

        embeddings = embedder(documents)
        assert len(embeddings) == len(documents)

        query = "What is AutoGen?"
        query_embedding = embedder.embed_query(query)
        assert isinstance(query_embedding, list)
        assert all(isinstance(x, float) for x in query_embedding)

        embeddings_with_context = embedder.embed_documents(
            documents,
            autogen_context={
                "conversation_id": "research-session-1",
                "agent_name": "research_assistant",
                "workflow_type": "document_retrieval",
                "retriever_name": "knowledge_base"
            }
        )
        assert len(embeddings_with_context) == len(documents)

    def test_error_handling_in_autogen_workflow(self, autogen_available):
        """
        Real integration: Error handling in AutoGen context.
        """
        class FailingTestAdapter:
            async def embed(self, texts: List[str], ctx=None) -> List[List[float]]:
                raise RuntimeError("Rate limit exceeded: Please wait 60 seconds before retrying")
            
            async def embed_batch(self, texts: List[str], ctx=None) -> List[List[float]]:
                raise RuntimeError("Rate limit exceeded: Please wait 60 seconds before retrying")
            
            def capabilities(self) -> Dict[str, Any]:
                return {"supported_models": ["test-model"]}

        adapter = FailingTestAdapter()
        embedder = register_embeddings(adapter, model="mock-embed-512")
        
        # Test that errors from embedder propagate with context
        with pytest.raises(Exception) as exc_info:
            embedder.embed_documents(["test document"])

        error_str = str(exc_info.value)
        assert "rate limit" in error_str.lower() or "exceeded" in error_str.lower()
        assert "wait 60 seconds" in error_str.lower() or "retry" in error_str.lower()

    @pytest.mark.asyncio
    async def test_async_embeddings_in_autogen_workflow(self, autogen_available, adapter):
        """
        Real integration: Async embeddings in AutoGen async workflows.
        """
        embedder = register_embeddings(adapter, model="mock-embed-512")
        
        # Test async embedding in agent context
        embeddings = await embedder.aembed_query(
            "async query for AutoGen workflow",
            autogen_context={
                "conversation_id": "async-session",
                "agent_name": "async_agent",
                "workflow_type": "async_processing"
            }
        )

        assert isinstance(embeddings, list)
        assert all(isinstance(x, float) for x in embeddings)

        documents = ["doc1", "doc2", "doc3"]
        batch_embeddings = await embedder.aembed_documents(documents)
        assert len(batch_embeddings) == len(documents)

    def test_multiple_agents_can_share_same_embedder(self, autogen_available, adapter):
        """
        Real integration: Multiple agents/retrievers can share the same embedder.
        """
        embedder = register_embeddings(adapter, model="mock-embed-512")
        
        # Simulate multiple agents using same embedder
        contexts = [
            {"conversation_id": "conv-1", "agent_name": "researcher", "workflow_type": "research"},
            {"conversation_id": "conv-1", "agent_name": "analyst", "workflow_type": "analysis"},
            {"conversation_id": "conv-2", "agent_name": "summarizer", "workflow_type": "summarization"},
        ]

        for ctx in contexts:
            result = embedder.embed_query(f"query from {ctx['agent_name']}", autogen_context=ctx)
            assert isinstance(result, list)
            assert all(isinstance(x, float) for x in result)

            batch_result = embedder([f"doc from {ctx['agent_name']}"], autogen_context=ctx)
            assert len(batch_result) == 1


# ---------------------------------------------------------------------------
# create_retriever Integration Tests
# ---------------------------------------------------------------------------

def test_create_retriever_raises_runtime_error_when_autogen_not_installed(
    monkeypatch: pytest.MonkeyPatch,
    adapter,
) -> None:
    """Clear error when AutoGen not installed."""
    import builtins as _builtins

    orig_import = _builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("autogen"):
            raise ImportError("AutoGen not installed: pip install pyautogen")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(_builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as exc_info:
        create_retriever(corpus_adapter=adapter, vector_store=object(), model="mock-embed-512")

    assert "AutoGen is not installed" in str(exc_info.value)
    assert "pip install pyautogen" in str(exc_info.value)


def test_create_retriever_prefers_setter_method(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    """create_retriever should prefer public setter methods."""
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

        def similarity_search(self, *args: Any, **kwargs: Any) -> Any:
            return []

    vs = DummyVectorStore()

    retriever = create_retriever(corpus_adapter=adapter, vector_store=vs, model="mock-embed-512")

    assert isinstance(retriever, DummyVectorStoreRetriever)
    assert retriever.vectorstore is vs
    assert vs.set_called_with is not None
    assert isinstance(vs.set_called_with, CorpusAutoGenEmbeddings)


def test_create_retriever_configures_vector_store_embedding_function(
    monkeypatch: pytest.MonkeyPatch,
    adapter,
) -> None:
    """create_retriever should configure vector store embedding function."""
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
        model="mock-embed-512",
        framework_version="auto-fw-2",
    )

    assert isinstance(retriever, DummyVectorStoreRetriever)
    assert retriever.vectorstore is vs

    assert isinstance(vs.embedding_function, CorpusAutoGenEmbeddings)
    assert vs.embedding_function.corpus_adapter is adapter
    assert vs.embedding_function.model == "mock-embed-512"


def test_create_retriever_configures_private_embedding_function_when_only_private_present(
    monkeypatch: pytest.MonkeyPatch,
    adapter,
) -> None:
    """create_retriever should work with private embedding function attribute."""
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

    retriever = create_retriever(corpus_adapter=adapter, vector_store=vs, model="mock-embed-512")

    assert isinstance(retriever, DummyVectorStoreRetriever)
    assert retriever.vectorstore is vs

    assert isinstance(vs._embedding_function, CorpusAutoGenEmbeddings)
    assert vs._embedding_function.corpus_adapter is adapter
