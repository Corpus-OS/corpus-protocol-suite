from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Optional
import inspect
import asyncio
import pytest
import time
from unittest.mock import Mock, patch
import concurrent.futures

import corpus_sdk.embedding.framework_adapters.crewai as crewai_adapter_module
from corpus_sdk.embedding.framework_adapters.crewai import (
    CorpusCrewAIEmbeddings,
    create_embedder,
    register_with_crewai,
    CrewAIConfig,
    ErrorCodes,
)
from corpus_sdk.embedding.embedding_base import OperationContext


# ---------------------------------------------------------------------------
# Framework Version Support Matrix
# ---------------------------------------------------------------------------
"""
Framework Version Support:
- CrewAI: 0.28.0+ (tested up to 0.51.0)
- Python: 3.8+
- Corpus SDK: 1.0.0+

Integration Notes:
- Compatible with CrewAI's agent.embedder attribute pattern
- Supports CrewAI knowledge sources and RAG workflows
- Handles CrewAI context (agent_role, task_id, crew_id, workflow)
- Framework protocol-first design (no hard inheritance required)

Note: CrewAI compatibility is verified via duck typing and protocol
implementation, not inheritance. This ensures compatibility even when
CrewAI base classes change.
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


def _make_embeddings(adapter: Any, **kwargs: Any) -> CorpusCrewAIEmbeddings:
    """
    Construct a CorpusCrewAIEmbeddings instance from the generic adapter.
    """
    return CorpusCrewAIEmbeddings(corpus_adapter=adapter, **kwargs)


# ---------------------------------------------------------------------------
# Constructor / config behavior
# ---------------------------------------------------------------------------

def test_constructor_works_with_real_adapter(adapter) -> None:
    """
    CorpusCrewAIEmbeddings should work with any real adapter.
    
    Framework Compatibility: Ensures basic protocol compliance across
    all supported CrewAI versions (0.28.0+).
    
    Note: Uses the adapter fixture from conftest.py
    """
    embeddings = _make_embeddings(adapter)
    assert embeddings.corpus_adapter is adapter
    assert hasattr(embeddings.corpus_adapter, 'embed')
    assert callable(embeddings.corpus_adapter.embed)


def test_constructor_rejects_common_user_mistakes() -> None:
    """
    CorpusCrewAIEmbeddings should provide clear error messages for
    common user mistakes.
    
    Error Message Quality: Users get helpful error messages, not
    cryptic Python errors.
    """
    # Common mistake 1: Passing None
    with pytest.raises(TypeError) as exc_info:
        CorpusCrewAIEmbeddings(corpus_adapter=None)
    
    msg = str(exc_info.value)
    assert "must implement an EmbeddingProtocolV1-compatible interface" in msg
    assert "None" in msg or "null" in msg.lower()
    
    # Common mistake 2: Passing a string (wrong type)
    with pytest.raises(TypeError) as exc_info:
        CorpusCrewAIEmbeddings(corpus_adapter="not an adapter")
    
    msg = str(exc_info.value)
    assert "must implement an EmbeddingProtocolV1-compatible interface" in msg
    assert "str" in msg or "string" in msg.lower()
    
    # Common mistake 3: Passing an object without embed() method
    class MockAgent:
        """Looks like an agent but not an embedding adapter"""
        def chat(self): 
            pass
        
    with pytest.raises(TypeError) as exc_info:
        CorpusCrewAIEmbeddings(corpus_adapter=MockAgent())
    
    msg = str(exc_info.value)
    assert "must implement an EmbeddingProtocolV1-compatible interface" in msg
    assert "embed" in msg.lower()


def test_crewai_config_defaults_and_bool_coercion(adapter) -> None:
    """
    crewai_config should be normalized with defaults and booleans coerced.
    
    Framework Compatibility: Configuration patterns stable across
    CrewAI 0.28.0-0.51.0.
    
    Note: Uses the adapter fixture from conftest.py
    """
    embeddings = _make_embeddings(
        adapter,
        crewai_config={
            "fallback_to_simple_context": 0,  # falsy -> bool
            # leave enable_agent_context_propagation unset
            "task_aware_batching": 1,  # truthy -> bool
        },
    )

    cfg = embeddings.crewai_config
    # Defaults filled in
    assert "fallback_to_simple_context" in cfg
    assert "enable_agent_context_propagation" in cfg
    assert "task_aware_batching" in cfg

    # Bool coercion
    assert isinstance(cfg["fallback_to_simple_context"], bool)
    assert isinstance(cfg["enable_agent_context_propagation"], bool)
    assert isinstance(cfg["task_aware_batching"], bool)

    # Specific values
    assert cfg["fallback_to_simple_context"] is False
    assert cfg["task_aware_batching"] is True


def test_create_embedder_returns_crewai_embeddings(adapter) -> None:
    """
    create_embedder should return a CorpusCrewAIEmbeddings wired to the adapter.
    
    Framework Compatibility: create_embedder pattern stable across
    all supported CrewAI versions.
    
    Note: Uses the adapter fixture from conftest.py
    """
    emb = create_embedder(
        corpus_adapter=adapter,
        model="crewai-model",
        framework_version="1.2.3",
    )
    assert isinstance(emb, CorpusCrewAIEmbeddings)
    assert emb.corpus_adapter is adapter
    assert emb.model == "crewai-model"


# ---------------------------------------------------------------------------
# Context translation / CrewAIContext mapping
# ---------------------------------------------------------------------------

def test_crewai_context_passed_to_context_translation(
    monkeypatch: pytest.MonkeyPatch,
    adapter,
) -> None:
    """
    Verify that crewai_context is passed through to context_from_crewai
    when embedding.
    
    Framework Compatibility: Context translation stable across
    CrewAI 0.28.0-0.51.0.
    
    Note: Uses the adapter fixture from conftest.py
    """
    captured: Dict[str, Any] = {}

    def fake_from_crewai(ctx: Dict[str, Any], framework_version: Any = None) -> None:
        captured["ctx"] = ctx
        captured["framework_version"] = framework_version
        # Returning None is allowed; adapter will just skip OperationContext.
        return None

    # Patch the imported symbol inside the module under test
    monkeypatch.setattr(
        crewai_adapter_module,
        "context_from_crewai",
        fake_from_crewai,
    )

    embeddings = _make_embeddings(
        adapter,
        framework_version="crewai-test-version",
    )

    crew_ctx = {
        "agent_role": "researcher",
        "task_id": "task-123",
        "crew_id": "crew-xyz",
        "workflow": "unit-test",
    }

    result = embeddings.embed_documents(
        ["foo", "bar"],
        crewai_context=crew_ctx,
    )
    _assert_embedding_matrix_shape(result, expected_rows=2)

    assert captured.get("ctx") is not None
    assert captured["ctx"] == crew_ctx
    assert captured["framework_version"] == "crewai-test-version"


def test_error_context_includes_crewai_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When an error occurs during CrewAI embedding, error context should include
    CrewAI-specific metadata (e.g., agent_role, task_id) via attach_context().
    
    Error Message Quality: Errors must contain actionable context for debugging
    CrewAI workflows in production.
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(
        crewai_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingAdapter:
        async def embed(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("test error from crewai adapter: Check model configuration and API keys")

    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=FailingAdapter())

    crew_ctx = {"agent_role": "tester", "task_id": "task-123"}

    with pytest.raises(RuntimeError, match="test error from crewai adapter") as exc_info:
        embeddings.embed_documents(["text"], crewai_context=crew_ctx)
    
    # Enhanced error message quality assertions
    error_str = str(exc_info.value)
    # Verify error contains actionable information
    assert "test error from crewai adapter" in error_str
    assert "Check model configuration" in error_str or "API keys" in error_str
    
    # Verify some context was attached
    assert captured_context, "attach_context was not called"
    
    # Framework tagging should be present
    assert "framework" in captured_context
    assert captured_context.get("framework") == "crewai"
    
    # CrewAI-specific fields should be present in the context
    assert captured_context.get("agent_role") == "tester"
    assert captured_context.get("task_id") == "task-123"
    
    # Verify context contains debugging breadcrumbs
    assert "operation" in captured_context
    assert captured_context["operation"] == "embedding_documents"
    
    # Verify error codes are attached for proper categorization
    assert "error_codes" in captured_context
    assert captured_context["error_codes"] == crewai_adapter_module.EMBEDDING_COERCION_ERROR_CODES


# ---------------------------------------------------------------------------
# Additional config / context nuance tests
# ---------------------------------------------------------------------------

def test_fallback_to_simple_context_true_uses_default_operation_context(monkeypatch, adapter):
    """Framework Compatibility: Fallback behavior stable across CrewAI versions."""
    calls = {}

    def fake_from_crewai(crewai_ctx, framework_version=None):
        class WeirdCtx:
            pass

        calls["ctx"] = crewai_ctx
        return WeirdCtx()

    monkeypatch.setattr(crewai_adapter_module, "context_from_crewai", fake_from_crewai)

    embeddings = CorpusCrewAIEmbeddings(
        corpus_adapter=adapter,
        crewai_config=CrewAIConfig(fallback_to_simple_context=True),
    )

    crewai_context = {"agent_role": "analyst", "task_id": "t1"}
    core_ctx, framework_ctx = embeddings._build_contexts(crewai_context=crewai_context)

    assert isinstance(core_ctx, OperationContext)
    assert framework_ctx["framework"] == "crewai"


def test_fallback_to_simple_context_false_leaves_core_ctx_none(monkeypatch, adapter):
    """Framework Compatibility: Context propagation configurable across versions."""
    def fake_from_crewai(crewai_ctx, framework_version=None):
        class WeirdCtx:
            pass

        return WeirdCtx()

    monkeypatch.setattr(crewai_adapter_module, "context_from_crewai", fake_from_crewai)

    embeddings = CorpusCrewAIEmbeddings(
        corpus_adapter=adapter,
        crewai_config=CrewAIConfig(fallback_to_simple_context=False),
    )

    crewai_context = {"agent_role": "analyst", "task_id": "t1"}
    core_ctx, framework_ctx = embeddings._build_contexts(crewai_context=crewai_context)

    assert core_ctx is None
    assert framework_ctx["framework"] == "crewai"


def test_enable_agent_context_propagation_flag_controls_operation_context_propagation(
    monkeypatch,
    adapter,
):
    """Framework Compatibility: Context propagation stable across CrewAI versions."""
    def fake_from_crewai(crewai_ctx, framework_version=None):
        return OperationContext(request_id="r1")

    monkeypatch.setattr(crewai_adapter_module, "context_from_crewai", fake_from_crewai)

    crewai_context = {"agent_role": "analyst"}

    emb_default = CorpusCrewAIEmbeddings(
        corpus_adapter=adapter,
        crewai_config=CrewAIConfig(enable_agent_context_propagation=True),
    )
    core_ctx, framework_ctx = emb_default._build_contexts(crewai_context=crewai_context)
    assert isinstance(core_ctx, OperationContext)
    assert framework_ctx["_operation_context"] is core_ctx

    emb_disabled = CorpusCrewAIEmbeddings(
        corpus_adapter=adapter,
        crewai_config=CrewAIConfig(enable_agent_context_propagation=False),
    )
    core_ctx2, framework_ctx2 = emb_disabled._build_contexts(
        crewai_context=crewai_context
    )
    assert isinstance(core_ctx2, OperationContext)
    assert "_operation_context" not in framework_ctx2


def test_task_aware_batching_sets_batch_strategy(adapter):
    """Framework Compatibility: Task-aware batching optimization for CrewAI workflows."""
    embeddings = CorpusCrewAIEmbeddings(
        corpus_adapter=adapter,
        crewai_config=CrewAIConfig(task_aware_batching=True),
    )

    crewai_context = {
        "agent_role": "analyst",
        "task_id": "task-123",
    }
    core_ctx, framework_ctx = embeddings._build_contexts(crewai_context=crewai_context)

    assert framework_ctx["batch_strategy"] == "task_aware_task-123"


def test_non_mapping_crewai_context_raises_value_error(adapter):
    """Error Message Quality: Clear error when context has wrong type."""
    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=adapter)

    with pytest.raises(ValueError) as exc:
        embeddings._build_contexts(crewai_context="not-a-mapping")  # type: ignore[arg-type]

    error_msg = str(exc.value)
    assert ErrorCodes.CREWAI_CONTEXT_INVALID in error_msg
    # Verify error is actionable
    assert "must be a mapping" in error_msg.lower() or "must be a dict" in error_msg.lower()


def test_context_from_crewai_failure_attaches_error_context(monkeypatch, adapter):
    """Error Message Quality: Context preserved even when context translation fails."""
    captured = {}

    def fake_from_crewai(crewai_ctx, framework_version=None):
        raise RuntimeError("boom")

    def fake_attach_context(exc, **kwargs):
        captured["exc"] = exc
        captured["kwargs"] = kwargs

    monkeypatch.setattr(crewai_adapter_module, "context_from_crewai", fake_from_crewai)
    monkeypatch.setattr(crewai_adapter_module, "attach_context", fake_attach_context)

    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=adapter)

    crewai_context = {"agent_role": "analyst", "task_id": "t1"}
    core_ctx, framework_ctx = embeddings._build_contexts(crewai_context=crewai_context)

    assert "exc" in captured
    assert "crewai_context_snapshot" in captured["kwargs"]
    assert (
        captured["kwargs"]["error_codes"]
        == crewai_adapter_module.EMBEDDING_COERCION_ERROR_CODES
    )
    # Verify error context contains framework info
    assert captured["kwargs"].get("framework") == "crewai"


# ---------------------------------------------------------------------------
# Sync semantics
# ---------------------------------------------------------------------------

def test_sync_embed_documents_and_query_basic(adapter) -> None:
    """
    Basic smoke test for sync embed_documents / embed_query behavior.
    
    Framework Compatibility: Sync API stable across all CrewAI versions.
    """
    embeddings = _make_embeddings(adapter, model="sync-model")

    texts = ["alpha", "beta", "gamma"]
    query = "delta"

    doc_vecs = embeddings.embed_documents(texts)
    _assert_embedding_matrix_shape(doc_vecs, expected_rows=len(texts))

    q_vec = embeddings.embed_query(query)
    _assert_embedding_vector_shape(q_vec)


def test_sync_embed_documents_with_crewai_context(adapter) -> None:
    """Framework Compatibility: Context parameter stable across versions."""
    embeddings = _make_embeddings(adapter)

    ctx = {
        "agent_role": "tester",
        "task_id": "shape-check",
    }

    result = embeddings.embed_documents(
        ["ctx-one", "ctx-two"],
        crewai_context=ctx,
    )
    _assert_embedding_matrix_shape(result, expected_rows=2)


def test_embed_documents_rejects_non_string_items(adapter):
    """Error Message Quality: Clear error for type mismatches."""
    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        embeddings.embed_documents(["ok", 123])  # type: ignore[list-item]
    
    error_msg = str(exc.value)
    # Verify error is actionable
    assert "item 1 is int" in error_msg or "must be str" in error_msg
    assert "embed_documents expects Sequence[str]" in error_msg


def test_embed_query_rejects_non_string(adapter):
    """Error Message Quality: Clear error for type mismatches."""
    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        embeddings.embed_query(123)  # type: ignore[arg-type]
    
    error_msg = str(exc.value)
    # Verify error is actionable
    assert "embed_query expects str" in error_msg
    assert "got int" in error_msg or "got 123" in error_msg


# ---------------------------------------------------------------------------
# Async semantics
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_embed_documents_and_query_basic(adapter) -> None:
    """
    Async aembed_documents / aembed_query should exist and produce shapes
    compatible with the sync API.
    
    Framework Compatibility: Async API stable across CrewAI versions.
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
    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        await embeddings.aembed_documents(["ok", 123])  # type: ignore[list-item]
    
    error_msg = str(exc.value)
    # Verify error is actionable and consistent with sync version
    assert "item 1 is int" in error_msg or "must be str" in error_msg
    assert "aembed_documents expects Sequence[str]" in error_msg


@pytest.mark.asyncio
async def test_aembed_query_rejects_non_string(adapter):
    """Error Message Quality: Consistent error messages for async API."""
    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        await embeddings.aembed_query(123)  # type: ignore[arg-type]
    
    error_msg = str(exc.value)
    # Verify error is actionable and consistent with sync version
    assert "aembed_query expects str" in error_msg
    assert "got int" in error_msg or "got 123" in error_msg


def test_crewai_interface_compatibility(adapter) -> None:
    """
    Verify that CorpusCrewAIEmbeddings implements the expected CrewAI
    Embeddings interface when CrewAI is available.
    
    Framework Compatibility: Duck-typing compatibility with CrewAI 0.28.0+.
    Note: We test protocol compliance, not inheritance, for better
    forward compatibility.
    """
    embeddings = _make_embeddings(adapter)

    # Core methods should always exist
    assert hasattr(embeddings, "embed_documents")
    assert hasattr(embeddings, "embed_query")
    assert hasattr(embeddings, "aembed_documents")
    assert hasattr(embeddings, "aembed_query")

    # Try to import CrewAI's Embeddings base class if available
    try:
        from crewai.embeddings import Embeddings  # type: ignore[import]
        
        # IMPORTANT: We use duck typing, not inheritance
        # Check that we can be used where Embeddings is expected
        # This is a weaker but more flexible check
        embedder = embeddings
        
        # Verify we have all required methods with correct signatures
        required_methods = ['embed_documents', 'embed_query', 
                           'aembed_documents', 'aembed_query']
        for method in required_methods:
            assert hasattr(embedder, method), f"Missing required method: {method}"
            assert callable(getattr(embedder, method)), f"Method not callable: {method}"
            
        # Note: We don't assert isinstance() because we use protocol pattern
        # for better forward compatibility
        
    except ImportError:
        pytest.skip("CrewAI is not installed; cannot assert interface compatibility")


# ---------------------------------------------------------------------------
# Error-context richness for failures
# ---------------------------------------------------------------------------

def test_embed_documents_error_context_includes_crewai_fields(monkeypatch, adapter):
    """Error Message Quality: Production debugging context in errors."""
    class FailingTranslator:
        def embed(self, raw_texts, op_ctx=None, framework_ctx=None):
            raise RuntimeError("translator failed: Check model configuration and API limits")

    captured = {}

    def fake_attach_context(exc, **kwargs):
        captured["exc"] = exc
        captured["kwargs"] = kwargs

    monkeypatch.setattr(crewai_adapter_module, "attach_context", fake_attach_context)

    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=adapter, model="test-model")
    
    # Use monkeypatch to inject failing translator
    with monkeypatch.context() as m:
        m.setattr(embeddings, "_translator", FailingTranslator())
        
        crewai_context = {
            "agent_role": "analyst",
            "task_id": "t1",
            "workflow": "wf",
            "crew_id": "crew-1",
        }

        with pytest.raises(RuntimeError) as exc_info:
            embeddings.embed_documents(["doc1", "doc2"], crewai_context=crewai_context)
        
        # Verify error message quality
        error_str = str(exc_info.value)
        assert "translator failed" in error_str
        assert "Check model configuration" in error_str  # Actionable guidance

        ctx = captured["kwargs"]
        assert ctx["framework"] == "crewai"
        assert ctx["operation"] == "embedding_documents"
        assert (
            ctx["error_codes"] == crewai_adapter_module.EMBEDDING_COERCION_ERROR_CODES
        )
        assert ctx["agent_role"] == "analyst"
        assert ctx["task_id"] == "t1"
        # Verify production debugging context
        assert "model" in ctx
        assert ctx["model"] == "test-model"


@pytest.mark.asyncio
async def test_aembed_query_error_context_includes_crewai_fields(monkeypatch, adapter):
    """Error Message Quality: Async errors also include debugging context."""
    class FailingTranslator:
        async def arun_embed(self, raw_texts, op_ctx=None, framework_ctx=None):
            raise RuntimeError("translator failed: Verify API key and model access permissions")

    captured = {}

    def fake_attach_context(exc, **kwargs):
        captured["exc"] = exc
        captured["kwargs"] = kwargs

    monkeypatch.setattr(crewai_adapter_module, "attach_context", fake_attach_context)

    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=adapter, model="test-model")
    
    # Use monkeypatch to inject failing translator
    with monkeypatch.context() as m:
        m.setattr(embeddings, "_translator", FailingTranslator())
        
        crewai_context = {"agent_role": "analyst", "task_id": "t1"}

        with pytest.raises(RuntimeError) as exc_info:
            await embeddings.aembed_query("hello", crewai_context=crewai_context)
        
        # Verify error message quality
        error_str = str(exc_info.value)
        assert "translator failed" in error_str
        assert "Verify API key" in error_str  # Actionable guidance

        ctx = captured["kwargs"]
        assert ctx["framework"] == "crewai"
        assert ctx["operation"] == "embedding_query"
        assert ctx["agent_role"] == "analyst"
        assert ctx["task_id"] == "t1"
        # Verify async-specific context if any
        assert "model" in ctx


# ---------------------------------------------------------------------------
# Capabilities / health passthrough
# ---------------------------------------------------------------------------

def test_capabilities_passthrough_when_underlying_provides(adapter) -> None:
    """
    When the underlying adapter implements capabilities/acapabilities,
    CorpusCrewAIEmbeddings should surface them.
    """
    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=adapter)

    caps = embeddings.capabilities()
    assert isinstance(caps, dict)


@pytest.mark.asyncio
async def test_async_capabilities_fallback_to_sync(adapter) -> None:
    """
    acapabilities should fall back to sync capabilities() when only the
    sync method is implemented on the underlying adapter.
    """
    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=adapter)

    acaps = await embeddings.acapabilities()
    assert isinstance(acaps, dict)


def test_capabilities_empty_when_missing():
    """
    If the underlying adapter has no capabilities()/acapabilities(),
    the CrewAI adapter should return an empty mapping (best-effort).
    """
    class NoCapAdapter:
        async def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 3 for _ in texts]

    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=NoCapAdapter())

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

    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=HealthAdapter())

    health = embeddings.health()
    assert isinstance(health, dict)
    assert health.get("status") == "ok"

    class NoHealthAdapter:
        async def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 2 for _ in texts]

    embeddings2 = CorpusCrewAIEmbeddings(corpus_adapter=NoHealthAdapter())

    health2 = embeddings2.health()
    assert isinstance(health2, dict)
    assert health2 == {}


# ---------------------------------------------------------------------------
# Integration Tests with Real CrewAI Objects
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCrewAIIntegration:
    """
    Integration tests with real CrewAI objects.
    
    These tests verify that our adapter actually works in CrewAI workflows.
    They're skipped if CrewAI is not installed.
    
    Framework Compatibility: Tested with CrewAI 0.28.0-0.51.0.
    """
    
    @pytest.fixture
    def crewai_available(self):
        """Check if CrewAI is available for integration tests."""
        try:
            import crewai
            return True
        except ImportError:
            pytest.skip("CrewAI not installed - skipping integration tests")
    
    def test_can_create_embedder_for_crewai_agent(self, crewai_available, adapter):
        """
        Basic integration: Can create embedder that CrewAI agents can use.
        
        Framework Compatibility: Validates basic embedder assignment pattern
        works across CrewAI versions.
        """
        import crewai
        
        embedder = create_embedder(adapter, model="test-model")
        
        # Create a minimal agent that would use embedder
        # This tests the actual interface compatibility
        agent = crewai.Agent(
            role="researcher",
            goal="Research topics",
            backstory="A curious researcher",
            allow_delegation=False,
            verbose=False
        )
        
        # The embedder should be assignable to agent.embedder
        # This is the key integration point
        agent.embedder = embedder
        
        # Verify the assignment worked
        assert agent.embedder is embedder
        # Verify embedder has required methods
        assert hasattr(agent.embedder, 'embed_documents')
        assert hasattr(agent.embedder, 'embed_query')
        
        # Test that embedder actually works
        result = agent.embedder.embed_documents(["test document"])
        assert isinstance(result, list)
        assert len(result) == 1
        assert all(isinstance(x, float) for row in result for x in row)
    
    def test_embedder_works_with_crewai_knowledge_sources(self, crewai_available, adapter):
        """
        Integration: Embedder works with CrewAI knowledge sources.
        
        Framework Compatibility: Validates knowledge source integration
        pattern works across CrewAI versions.
        """
        import crewai
        
        embedder = create_embedder(adapter, model="test-model")
        
        # Create agent with embedder
        agent = crewai.Agent(
            role="researcher",
            goal="Research topics",
            backstory="A curious researcher",
            embedder=embedder,  # Assign at creation
            allow_delegation=False,
            verbose=False
        )
        
        # Create a knowledge source (this would trigger embeddings in real use)
        # Note: Actual RAG would require setting up a vector store
        # This test verifies the integration point exists
        
        # Verify agent can use embedder
        embeddings = agent.embedder.embed_query("test query")
        assert isinstance(embeddings, list)
        assert all(isinstance(x, float) for x in embeddings)
    
    def test_crew_with_multiple_agents_sharing_embedder(self, crewai_available, adapter):
        """
        Integration: Multiple agents in a crew can share the same embedder.
        
        Framework Compatibility: Validates shared resource pattern
        works across CrewAI versions.
        """
        import crewai
        from crewai import Task
        
        embedder = create_embedder(adapter, model="shared-model")
        
        # Create multiple agents sharing the same embedder
        researcher = crewai.Agent(
            role="researcher",
            goal="Research information",
            backstory="Expert researcher",
            embedder=embedder,
            allow_delegation=False,
            verbose=False
        )
        
        analyst = crewai.Agent(
            role="analyst",
            goal="Analyze research",
            backstory="Data analyst",
            embedder=embedder,  # Same embedder
            allow_delegation=False,
            verbose=False
        )
        
        # Create tasks
        research_task = Task(
            description="Research AI trends",
            agent=researcher,
            expected_output="Research report"
        )
        
        analysis_task = Task(
            description="Analyze research findings",
            agent=analyst,
            expected_output="Analysis report"
        )
        
        # Create crew
        crew = crewai.Crew(
            agents=[researcher, analyst],
            tasks=[research_task, analysis_task],
            verbose=False
        )
        
        # Verify both agents have the embedder
        assert researcher.embedder is embedder
        assert analyst.embedder is embedder
        assert researcher.embedder is analyst.embedder  # Same instance
        
        # Test that embedder works for both agents
        for agent in [researcher, analyst]:
            embeddings = agent.embedder.embed_query(f"query from {agent.role}")
            assert isinstance(embeddings, list)
            assert len(embeddings) > 0
    
    def test_error_handling_in_crewai_workflow(self, crewai_available):
        """
        Integration: Error handling in CrewAI context.
        
        Framework Compatibility: Validates error propagation pattern
        works across CrewAI versions.
        """
        import crewai
        
        # Create adapter that will fail
        class FailingTestAdapter:
            async def embed(self, texts: List[str], ctx=None) -> List[List[float]]:
                raise RuntimeError("Rate limit exceeded: Please wait before retrying")
            
            def capabilities(self) -> Dict[str, Any]:
                return {"supported_models": ["test-model"]}
        
        adapter = FailingTestAdapter()
        embedder = create_embedder(adapter, model="failing-model")
        
        agent = crewai.Agent(
            role="researcher",
            goal="Research topics",
            backstory="A curious researcher",
            embedder=embedder,
            allow_delegation=False,
            verbose=False
        )
        
        # Test that errors from embedder propagate with context
        with pytest.raises(Exception) as exc_info:
            agent.embedder.embed_documents(["test document"])
        
        error_str = str(exc_info.value)
        # Verify error contains actionable information
        assert "rate limit" in error_str.lower() or "exceeded" in error_str.lower()
        # Verify error suggests retry action
        assert "wait" in error_str.lower() or "retry" in error_str.lower()
    
    @pytest.mark.asyncio
    async def test_async_embedding_in_crewai_workflow(self, crewai_available, adapter):
        """
        Integration: Async embedding in CrewAI async workflows.
        
        Framework Compatibility: Validates async integration pattern
        works across CrewAI versions.
        """
        import crewai
        
        embedder = create_embedder(adapter, model="async-model")
        
        agent = crewai.Agent(
            role="researcher",
            goal="Research topics",
            backstory="A curious researcher",
            embedder=embedder,
            allow_delegation=False,
            verbose=False
        )
        
        # Test async embedding
        embeddings = await agent.embedder.aembed_query("async query")
        assert isinstance(embeddings, list)
        assert all(isinstance(x, float) for x in embeddings)


# ---------------------------------------------------------------------------
# Concurrency Tests
# ---------------------------------------------------------------------------

@pytest.mark.concurrency
class TestConcurrency:
    """
    Concurrency and thread-safety tests.
    """
    
    def test_shared_embedder_thread_safety(self, adapter):
        """
        Shared embedder is thread-safe for concurrent access.
        """
        import concurrent.futures
        
        embedder = create_embedder(adapter, model="concurrent-model")
        
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
    async def test_concurrent_async_embedding(self, adapter):
        """
        Async embedding supports concurrent operations.
        """
        embedder = create_embedder(adapter, model="async-concurrent-model")
        
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
# Crew registration helpers
# ---------------------------------------------------------------------------

def test_register_with_crewai_attaches_embedder_to_agents(adapter) -> None:
    """Framework Compatibility: Registration helper stable across versions."""
    class DummyAgent:
        def __init__(self) -> None:
            self.embedder: Any | None = None

    class DummyCrew:
        def __init__(self) -> None:
            self._agents = [DummyAgent(), DummyAgent()]
            self.name = "dummy-crew"

        @property
        def agents(self):
            return self._agents

    crew = DummyCrew()

    emb = register_with_crewai(
        crew=crew,
        corpus_adapter=adapter,
        model="crew-model",
        framework_version="fw-1",
    )
    assert isinstance(emb, CorpusCrewAIEmbeddings)

    # All agents should now have embedder set to the returned instance
    for agent in crew.agents:
        assert agent.embedder is emb


def test_register_with_crewai_handles_agents_callable(adapter) -> None:
    """Framework Compatibility: Flexible agent access pattern."""
    class DummyAgent:
        def __init__(self) -> None:
            self.embedder: Any | None = None

    class DummyCrewCallable:
        def __init__(self) -> None:
            self._agents = [DummyAgent()]
            self.name = "callable-crew"

        def agents(self):
            # callable returning list
            return self._agents

    crew = DummyCrewCallable()

    emb = register_with_crewai(
        crew=crew,
        corpus_adapter=adapter,
        model="crew-callable-model",
    )
    assert isinstance(emb, CorpusCrewAIEmbeddings)

    for agent in crew.agents():
        assert agent.embedder is emb


def test_register_with_crewai_no_agents_attribute(adapter) -> None:
    """Framework Compatibility: Graceful handling of missing agents."""
    class CrewNoAgents:
        def __init__(self) -> None:
            self.name = "no-agents-crew"

    crew = CrewNoAgents()

    emb = register_with_crewai(
        crew=crew,
        corpus_adapter=adapter,
    )
    assert isinstance(emb, CorpusCrewAIEmbeddings)


def test_register_with_crewai_crew_none_raises_value_error(adapter):
    """Error Message Quality: Clear error for invalid crew."""
    with pytest.raises(ValueError) as exc:
        register_with_crewai(None, adapter)  # type: ignore[arg-type]
    
    error_msg = str(exc.value)
    assert "crew cannot be None" in error_msg
    # Verify error is actionable
    assert "valid crew instance" in error_msg.lower() or "provide a crew" in error_msg.lower()


def test_register_with_crewai_agents_callable_that_raises_attaches_error_context(
    monkeypatch,
    adapter,
):
    """Error Message Quality: Errors during registration include context."""
    captured = {}

    def fake_attach_context(exc, **kwargs):
        captured["exc"] = exc
        captured["kwargs"] = kwargs

    monkeypatch.setattr(crewai_adapter_module, "attach_context", fake_attach_context)

    class BadCrew:
        name = "bad-crew"

        def agents(self):
            raise RuntimeError("boom: crew agents access failed")

    embedder = register_with_crewai(BadCrew(), adapter)
    assert isinstance(embedder, CorpusCrewAIEmbeddings)

    assert "exc" in captured
    snapshot = captured["kwargs"].get("crew_snapshot") or {}
    assert snapshot.get("type") == "BadCrew"
    assert snapshot.get("name") == "bad-crew"
    # Verify error context includes framework info
    assert captured["kwargs"].get("framework") == "crewai"
