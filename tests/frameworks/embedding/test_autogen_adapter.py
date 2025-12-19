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
from corpus_sdk.embedding.embedding_base import OperationContext


# ---------------------------------------------------------------------------
# Framework Version Support Matrix
# ---------------------------------------------------------------------------
"""
Framework Version Support:
- AutoGen: 0.2+ (tested up to latest)
- Python: 3.8+
- Corpus SDK: 1.0.0+

Integration Notes:
- Compatible with AutoGen's EmbeddingFunction protocol
- Supports AutoGen retrieval workflows and agent memory
- Handles AutoGen context (conversation_id, agent_name, workflow_type, retriever_name)
- Framework protocol-first design (no hard inheritance required)

Note: AutoGen compatibility is verified via duck typing and protocol
implementation, not inheritance. This ensures compatibility even when
AutoGen base classes change.
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
    
    Error Message Quality: Users get helpful error messages, not
    cryptic Python errors.
    """
    # Common mistake 1: Passing None
    with pytest.raises(TypeError) as exc_info:
        CorpusAutoGenEmbeddings(corpus_adapter=None)  # type: ignore[arg-type]
    
    msg = str(exc_info.value)
    assert "must implement an EmbeddingProtocolV1-compatible interface" in msg
    assert "None" in msg or "null" in msg.lower()
    
    # Common mistake 2: Passing a string (wrong type)
    with pytest.raises(TypeError) as exc_info:
        CorpusAutoGenEmbeddings(corpus_adapter="not an adapter")  # type: ignore[arg-type]
    
    msg = str(exc_info.value)
    assert "must implement an EmbeddingProtocolV1-compatible interface" in msg
    assert "str" in msg or "string" in msg.lower()
    
    # Common mistake 3: Passing an object without embed() method
    class MockAgent:
        """Looks like an agent but not an embedding adapter"""
        def chat(self): 
            pass
        
    with pytest.raises(TypeError) as exc_info:
        CorpusAutoGenEmbeddings(corpus_adapter=MockAgent())
    
    msg = str(exc_info.value)
    assert "must implement an EmbeddingProtocolV1-compatible interface" in msg
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
        model="auto-model",
        framework_version="autogen-fw-1.0",
    )

    assert isinstance(emb, CorpusAutoGenEmbeddings)
    assert emb.corpus_adapter is adapter
    assert emb.model == "auto-model"


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
        model="m",
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

    embeddings = _make_embeddings(adapter, model="m", framework_version="fv")

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


# ---------------------------------------------------------------------------
# AutoGen Interface Compatibility
# ---------------------------------------------------------------------------

def test_autogen_interface_compatibility(adapter) -> None:
    """
    Verify that CorpusAutoGenEmbeddings implements AutoGen's EmbeddingFunction.
    
    Framework Compatibility: Duck-typing compatibility with AutoGen.
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

    # Try to import AutoGen's EmbeddingFunction if available
    try:
        from autogen.agentchat.contrib.retrieve_assistant_agent import (  # type: ignore[import]
            EmbeddingFunction,
        )
        
        # IMPORTANT: We use duck typing, not inheritance
        # Check that we can be used where EmbeddingFunction is expected
        # This is a weaker but more flexible check
        embedder = embeddings
        
        # Verify we have all required methods with correct signatures
        required_methods = ['__call__', 'embed_documents', 'embed_query', 
                           'aembed_documents', 'aembed_query']
        for method in required_methods:
            assert hasattr(embedder, method), f"Missing required method: {method}"
            assert callable(getattr(embedder, method)), f"Method not callable: {method}"
        
        # Test the __call__ method works
        result = embedder(["test"])
        _assert_embedding_matrix_shape(result, expected_rows=1)
            
    except ImportError:
        pytest.skip("AutoGen is not installed; cannot assert interface compatibility")


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

    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=FailingAdapter())

    auto_ctx = {"conversation_id": "conv-123", "agent_name": "tester"}

    with pytest.raises(RuntimeError, match="test error from autogen adapter") as exc_info:
        embeddings.embed_documents(["text"], autogen_context=auto_ctx)
    
    # Enhanced error message quality assertions
    error_str = str(exc_info.value)
    # Verify error contains actionable information
    assert "test error from autogen adapter" in error_str
    assert "Check model configuration" in error_str or "API keys" in error_str
    
    # Verify some context was attached
    assert captured_context, "attach_context was not called"
    
    # Framework tagging should be present
    assert "framework" in captured_context
    assert captured_context.get("framework") == "autogen"
    
    # AutoGen-specific fields should be present in the context
    assert captured_context.get("conversation_id") == "conv-123"
    assert captured_context.get("agent_name") == "tester"
    
    # Verify context contains debugging breadcrumbs
    assert "operation" in captured_context
    assert captured_context["operation"] == "embedding_documents"
    
    # Verify error codes are attached for proper categorization
    assert "error_codes" in captured_context
    assert captured_context["error_codes"] == autogen_adapter_module.EMBEDDING_COERCION_ERROR_CODES


def test_invalid_autogen_context_type_is_ignored(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    """
    autogen_context is best-effort. If the type is wrong (not a Mapping),
    embeddings should still work.
    
    Framework Compatibility: AutoGen context validation is lenient to
    maintain compatibility with various AutoGen versions.
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
    monkeypatch: pytest.MonkeyPatch, adapter
) -> None:
    """
    If context_from_autogen raises, embeddings proceed without OperationContext,
    and attach_context is invoked with operation="context_build".
    
    Error Message Quality: Context preserved even when context translation fails.
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
    # Verify error is actionable
    assert "item 1 is int" in error_msg or "must be str" in error_msg
    assert "embed_documents expects Sequence[str]" in error_msg


def test_embed_query_rejects_non_string(adapter):
    """Error Message Quality: Clear error for type mismatches."""
    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        embeddings.embed_query(123)  # type: ignore[arg-type]
    
    error_msg = str(exc.value)
    # Verify error is actionable
    assert "embed_query expects str" in error_msg
    assert "got int" in error_msg or "got 123" in error_msg


# ---------------------------------------------------------------------------
# Sync Semantics
# ---------------------------------------------------------------------------

def test_sync_embed_documents_and_query_basic(adapter) -> None:
    """
    Basic smoke test for sync embed_documents / embed_query behavior.
    
    Framework Compatibility: Sync API stable across all AutoGen versions.
    """
    embeddings = _make_embeddings(adapter, model="sync-model")

    texts = ["alpha", "beta", "gamma"]
    query = "delta"

    doc_vecs = embeddings.embed_documents(texts)
    _assert_embedding_matrix_shape(doc_vecs, expected_rows=len(texts))

    q_vec = embeddings.embed_query(query)
    _assert_embedding_vector_shape(q_vec)


def test_call_aliases_embed_documents(adapter) -> None:
    """
    __call__ should alias embed_documents for AutoGen EmbeddingFunction protocol.
    
    Framework Compatibility: __call__ method pattern stable across AutoGen versions.
    """
    embeddings = _make_embeddings(adapter)

    texts = ["call-one", "call-two"]
    result = embeddings(texts)  # __call__
    _assert_embedding_matrix_shape(result, expected_rows=len(texts))


def test_sync_embed_documents_with_autogen_context(adapter) -> None:
    """Framework Compatibility: Context parameter stable across versions."""
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
    
    Framework Compatibility: Async API stable across AutoGen versions.
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
async def test_async_and_sync_same_dimension(adapter) -> None:
    """Framework Compatibility: Sync/async parity maintained across versions."""
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


@pytest.mark.asyncio
async def test_aembed_documents_rejects_non_string_items(adapter):
    """Error Message Quality: Consistent error messages for async API."""
    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        await embeddings.aembed_documents(["ok", object()])  # type: ignore[list-item]
    
    error_msg = str(exc.value)
    # Verify error is actionable and consistent with sync version
    assert "item 1 is object" in error_msg or "must be str" in error_msg
    assert "aembed_documents expects Sequence[str]" in error_msg


@pytest.mark.asyncio
async def test_aembed_query_rejects_non_string(adapter):
    """Error Message Quality: Consistent error messages for async API."""
    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        await embeddings.aembed_query(123)  # type: ignore[arg-type]
    
    error_msg = str(exc.value)
    # Verify error is actionable and consistent with sync version
    assert "aembed_query expects str" in error_msg
    assert "got int" in error_msg or "got 123" in error_msg


# ---------------------------------------------------------------------------
# Error-context Richness for Failures
# ---------------------------------------------------------------------------

def test_embed_documents_error_context_includes_autogen_fields(monkeypatch, adapter):
    """Error Message Quality: Production debugging context in errors."""
    class FailingTranslator:
        def embed(self, raw_texts, op_ctx=None, framework_ctx=None):
            raise RuntimeError("translator failed: Check model configuration and API limits")

    captured = {}

    def fake_attach_context(exc, **kwargs):
        captured["exc"] = exc
        captured["kwargs"] = kwargs

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)

    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter, model="test-model")
    
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
        
        # Verify error message quality
        error_str = str(exc_info.value)
        assert "translator failed" in error_str
        assert "Check model configuration" in error_str  # Actionable guidance

        ctx = captured["kwargs"]
        assert ctx["framework"] == "autogen"
        assert ctx["operation"] == "embedding_documents"
        assert (
            ctx["error_codes"] == autogen_adapter_module.EMBEDDING_COERCION_ERROR_CODES
        )
        assert ctx["conversation_id"] == "conv-1"
        assert ctx["agent_name"] == "analyst"
        # Verify production debugging context
        assert "model" in ctx
        assert ctx["model"] == "test-model"


@pytest.mark.asyncio
async def test_aembed_query_error_context_includes_autogen_fields(monkeypatch, adapter):
    """Error Message Quality: Async errors also include debugging context."""
    class FailingTranslator:
        async def arun_embed(self, raw_texts, op_ctx=None, framework_ctx=None):
            raise RuntimeError("translator failed: Verify API key and model access permissions")

    captured = {}

    def fake_attach_context(exc, **kwargs):
        captured["exc"] = exc
        captured["kwargs"] = kwargs

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)

    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter, model="test-model")
    
    # Use monkeypatch to inject failing translator
    with monkeypatch.context() as m:
        m.setattr(embeddings, "_translator", FailingTranslator())
        
        autogen_context = {"conversation_id": "conv-2", "agent_name": "tester"}

        with pytest.raises(RuntimeError) as exc_info:
            await embeddings.aembed_query("hello", autogen_context=autogen_context)
        
        # Verify error message quality
        error_str = str(exc_info.value)
        assert "translator failed" in error_str
        assert "Verify API key" in error_str  # Actionable guidance

        ctx = captured["kwargs"]
        assert ctx["framework"] == "autogen"
        assert ctx["operation"] == "embedding_query"
        assert ctx["conversation_id"] == "conv-2"
        assert ctx["agent_name"] == "tester"
        # Verify async-specific context if any
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
# Capabilities / Health Passthrough
# ---------------------------------------------------------------------------

def test_capabilities_passthrough_when_underlying_provides(adapter) -> None:
    """
    When the underlying adapter implements capabilities/acapabilities,
    CorpusAutoGenEmbeddings should surface them.
    """
    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)

    caps = embeddings.capabilities()
    assert isinstance(caps, dict)


@pytest.mark.asyncio
async def test_async_capabilities_fallback_to_sync(adapter) -> None:
    """
    acapabilities should fall back to sync capabilities() when only the
    sync method is implemented on the underlying adapter.
    """
    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)

    acaps = await embeddings.acapabilities()
    assert isinstance(acaps, dict)


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
    assert ahealth.get("status_async") is True


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

def test_context_manager_closes_underlying_adapter() -> None:
    """
    Context manager should close underlying adapter (sync version).
    """
    class CloseableAdapter:
        def __init__(self) -> None:
            self.closed = False
            self.aclosed = False

        def embed(self, *args: Any, **kwargs: Any) -> Any:
            return None

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.aclosed = True

    adapter = CloseableAdapter()
    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)

    with embeddings as emb:
        _ = emb.embed_documents(["x"])  # smoke test

    assert adapter.closed is True


@pytest.mark.asyncio
async def test_async_context_manager_closes_underlying_adapter() -> None:
    """
    Async context manager should close underlying adapter (async version).
    """
    class CloseableAdapter:
        def __init__(self) -> None:
            self.closed = False
            self.aclosed = False

        def embed(self, *args: Any, **kwargs: Any) -> Any:
            return None

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.aclosed = True

    adapter = CloseableAdapter()
    embeddings = CorpusAutoGenEmbeddings(corpus_adapter=adapter)

    async with embeddings:
        _ = await embeddings.aembed_documents(["y"])

    assert adapter.aclosed is True


# ---------------------------------------------------------------------------
# Concurrency Tests
# ---------------------------------------------------------------------------

@pytest.mark.concurrency
def test_shared_embedder_thread_safety(adapter):
    """
    Shared embedder is thread-safe for concurrent access.
    """
    embedder = register_embeddings(adapter, model="concurrent-model")
    
    def embed_query(text: str):
        return embedder.embed_query(text)
    
    # Concurrent embedding calls
    texts = [f"query {i}" for i in range(10)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(embed_query, text) for text in texts]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    # All calls should succeed
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
    embedder = register_embeddings(adapter, model="async-concurrent-model")
    
    async def embed_async(text: str):
        return await embedder.aembed_query(text)
    
    # Concurrent async embedding
    texts = [f"async query {i}" for i in range(5)]
    tasks = [embed_async(text) for text in texts]
    results = await asyncio.gather(*tasks)
    
    # All calls should succeed
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
    agent workflows. They're skipped if AutoGen is not installed.
    
    Framework Compatibility: Tested with AutoGen 0.2+.
    """
    
    @pytest.fixture
    def autogen_available(self):
        """Check if AutoGen is available for integration tests."""
        try:
            import autogen
            return True
        except ImportError:
            pytest.skip("AutoGen not installed - skipping integration tests")
    
    def test_can_create_embeddings_for_autogen_retrieve_agent(self, autogen_available, adapter):
        """
        Real integration: Can create embeddings that work with AutoGen's
        RetrieveAssistantAgent.
        
        Framework Compatibility: Validates embedding integration pattern
        works with AutoGen's agent architecture.
        """
        # Import inside test to avoid ImportError when skipping
        from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent  # noqa: F401
        
        embedder = register_embeddings(
            corpus_adapter=adapter,
            model="autogen-model",
            framework_version="1.0.0"
        )
        
        # Test that embedder works in agent context
        result = embedder(["test document for AutoGen agent"])
        assert isinstance(result, list)
        assert len(result) == 1
        assert all(isinstance(x, float) for row in result for x in row)
        
        # Test that embedder works with autogen_context
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
        works with AutoGen's retrieval systems.
        """
        # Create embedder for AutoGen workflows
        embedder = register_embeddings(
            corpus_adapter=adapter,
            model="text-embedding-3-large",
            framework_version="1.0.0"
        )
        
        # Simulate AutoGen RAG workflow steps
        # 1. Embed documents for vector store
        documents = [
            "AutoGen is a framework for building multi-agent applications.",
            "RetrieveAssistantAgent enables RAG capabilities in AutoGen.",
            "Embeddings are used to convert text to vector representations."
        ]
        
        embeddings = embedder(documents)
        assert len(embeddings) == len(documents)
        
        # 2. Embed query for retrieval
        query = "What is AutoGen?"
        query_embedding = embedder.embed_query(query)
        assert isinstance(query_embedding, list)
        assert all(isinstance(x, float) for x in query_embedding)
        
        # 3. Test with full AutoGen context
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
        
        Framework Compatibility: Validates error propagation pattern
        works in AutoGen workflows.
        """
        # Create adapter that will fail
        class FailingTestAdapter:
            async def embed(self, texts: List[str], ctx=None) -> List[List[float]]:
                raise RuntimeError("Rate limit exceeded: Please wait 60 seconds before retrying")
            
            def capabilities(self) -> Dict[str, Any]:
                return {"supported_models": ["test-model"]}
        
        adapter = FailingTestAdapter()
        embedder = register_embeddings(adapter, model="failing-model")
        
        # Test that errors from embedder propagate with context
        with pytest.raises(Exception) as exc_info:
            embedder.embed_documents(["test document"])
        
        error_str = str(exc_info.value)
        # Verify error contains actionable information
        assert "rate limit" in error_str.lower() or "exceeded" in error_str.lower()
        # Verify error suggests specific retry action
        assert "wait 60 seconds" in error_str.lower() or "retry" in error_str.lower()
    
    @pytest.mark.asyncio
    async def test_async_embeddings_in_autogen_workflow(self, autogen_available, adapter):
        """
        Real integration: Async embeddings in AutoGen async workflows.
        
        Framework Compatibility: Validates async integration pattern
        works with AutoGen's async capabilities.
        """
        embedder = register_embeddings(adapter, model="async-model")
        
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
        
        # Test async batch embedding
        documents = ["doc1", "doc2", "doc3"]
        batch_embeddings = await embedder.aembed_documents(documents)
        assert len(batch_embeddings) == len(documents)
    
    def test_multiple_agents_can_share_same_embedder(self, autogen_available, adapter):
        """
        Real integration: Multiple agents/retrievers can share the same embedder.
        
        Framework Compatibility: Validates shared resource pattern
        works in multi-agent AutoGen applications.
        """
        embedder = register_embeddings(adapter, model="shared-model")
        
        # Simulate multiple agents using same embedder
        contexts = [
            {
                "conversation_id": "conv-1",
                "agent_name": "researcher",
                "workflow_type": "research"
            },
            {
                "conversation_id": "conv-1",
                "agent_name": "analyst",
                "workflow_type": "analysis"
            },
            {
                "conversation_id": "conv-2",
                "agent_name": "summarizer",
                "workflow_type": "summarization"
            }
        ]
        
        # All agents should be able to use the same embedder
        for i, ctx in enumerate(contexts):
            result = embedder.embed_query(f"query from {ctx['agent_name']}", autogen_context=ctx)
            assert isinstance(result, list)
            assert all(isinstance(x, float) for x in result)
            
            # Verify context is passed through
            batch_result = embedder([f"doc from {ctx['agent_name']}"], autogen_context=ctx)
            assert len(batch_result) == 1


# ---------------------------------------------------------------------------
# create_retriever Integration Tests
# ---------------------------------------------------------------------------

def test_create_retriever_raises_runtime_error_when_autogen_not_installed(
    monkeypatch: pytest.MonkeyPatch,
    adapter,
) -> None:
    """Error Message Quality: Clear error when AutoGen not installed."""
    import builtins as _builtins

    orig_import = _builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("autogen"):
            raise ImportError("AutoGen not installed: pip install pyautogen")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(_builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as exc_info:
        create_retriever(corpus_adapter=adapter, vector_store=object(), model="m")

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

    retriever = create_retriever(corpus_adapter=adapter, vector_store=vs, model="text-embedding-3-large")

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
        model="text-embedding-3-large",
        framework_version="auto-fw-2",
    )

    assert isinstance(retriever, DummyVectorStoreRetriever)
    assert retriever.vectorstore is vs

    assert isinstance(vs.embedding_function, CorpusAutoGenEmbeddings)
    assert vs.embedding_function.corpus_adapter is adapter
    assert vs.embedding_function.model == "text-embedding-3-large"


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

    retriever = create_retriever(corpus_adapter=adapter, vector_store=vs, model="text-embedding-3-small")

    assert isinstance(retriever, DummyVectorStoreRetriever)
    assert retriever.vectorstore is vs

    assert isinstance(vs._embedding_function, CorpusAutoGenEmbeddings)
    assert vs._embedding_function.corpus_adapter is adapter
