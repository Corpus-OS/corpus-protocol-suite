# tests/frameworks/embedding/test_crewai_adapter.py

from __future__ import annotations

from collections.abc import Sequence, Mapping
from typing import Any, Dict, List, Optional
import inspect
import asyncio
import pytest
import concurrent.futures

import corpus_sdk.embedding.framework_adapters.crewai as crewai_adapter_module
from corpus_sdk.embedding.framework_adapters.crewai import (
    CorpusCrewAIEmbeddings,
    create_embedder,
    register_with_crewai,
    CrewAIConfig,
    ErrorCodes,
)
from corpus_sdk.embedding.embedding_base import OperationContext, EmbeddingCapabilities


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

Policy:
- No pytest.skip in this file. Integration tests are pass/fail:
  if CrewAI is not available, tests fail-fast with actionable install guidance.
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


def _make_embeddings(adapter: Any, **kwargs: Any) -> CorpusCrewAIEmbeddings:
    """
    Construct a CorpusCrewAIEmbeddings instance from the generic adapter.
    """
    # Default to first supported model if no model specified
    if "model" not in kwargs and hasattr(adapter, "supported_models"):
        kwargs["model"] = adapter.supported_models[0]
    return CorpusCrewAIEmbeddings(corpus_adapter=adapter, **kwargs)


@pytest.fixture(scope="session")
def require_crewai():
    """
    Hard requirement for real CrewAI integration tests (pass/fail, never skip).

    This validates CrewAI is importable and exposes the minimal surface we rely on.
    """
    try:
        import crewai  # type: ignore
    except Exception as exc:
        pytest.fail(
            "CrewAI is required for integration tests in this module. Install with:\n"
            "  pip install -U crewai\n"
            f"Import error: {exc!r}",
            pytrace=False,
        )

    missing = [name for name in ("Agent", "Crew", "Task") if not hasattr(crewai, name)]
    if missing:
        pytest.fail(
            f"CrewAI import succeeded but missing required symbols: {missing}. "
            "Your CrewAI installation appears incompatible with these tests.",
            pytrace=False,
        )

    return crewai


# ---------------------------------------------------------------------------
# Constructor / config behavior
# ---------------------------------------------------------------------------

def test_constructor_works_with_real_adapter(adapter) -> None:
    """
    CorpusCrewAIEmbeddings should work with any real adapter.

    Note: Uses the adapter fixture from conftest.py
    """
    embeddings = _make_embeddings(adapter)
    assert embeddings.corpus_adapter is adapter
    assert hasattr(embeddings.corpus_adapter, "embed")
    assert callable(embeddings.corpus_adapter.embed)


def test_constructor_rejects_common_user_mistakes() -> None:
    """
    CorpusCrewAIEmbeddings should provide clear error messages for common user mistakes.
    """
    with pytest.raises(TypeError) as exc_info:
        CorpusCrewAIEmbeddings(corpus_adapter=None)  # type: ignore[arg-type]
    assert "must implement an EmbeddingProtocolV1-compatible interface" in str(exc_info.value)

    with pytest.raises(TypeError) as exc_info:
        CorpusCrewAIEmbeddings(corpus_adapter="not an adapter")  # type: ignore[arg-type]
    assert "must implement an EmbeddingProtocolV1-compatible interface" in str(exc_info.value)

    class MockAgent:
        """Looks like an agent but not an embedding adapter"""
        def chat(self) -> None:
            return None

    with pytest.raises(TypeError) as exc_info:
        CorpusCrewAIEmbeddings(corpus_adapter=MockAgent())
    msg = str(exc_info.value)
    assert "must implement an EmbeddingProtocolV1-compatible interface" in msg
    assert "embed" in msg.lower()


def test_crewai_config_defaults_and_bool_coercion(adapter) -> None:
    """
    crewai_config should be normalized with defaults and booleans coerced.
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
    assert "fallback_to_simple_context" in cfg
    assert "enable_agent_context_propagation" in cfg
    assert "task_aware_batching" in cfg

    assert isinstance(cfg["fallback_to_simple_context"], bool)
    assert isinstance(cfg["enable_agent_context_propagation"], bool)
    assert isinstance(cfg["task_aware_batching"], bool)

    assert cfg["fallback_to_simple_context"] is False
    assert cfg["task_aware_batching"] is True


def test_create_embedder_returns_crewai_embeddings(adapter) -> None:
    """
    create_embedder should return a CorpusCrewAIEmbeddings wired to the adapter.
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

def test_crewai_context_passed_to_context_translation(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    """
    Verify that crewai_context is passed through to context_from_crewai when embedding.
    """
    captured: Dict[str, Any] = {}

    def fake_from_crewai(ctx: Dict[str, Any], framework_version: Any = None) -> None:
        captured["ctx"] = ctx
        captured["framework_version"] = framework_version
        return None

    monkeypatch.setattr(crewai_adapter_module, "context_from_crewai", fake_from_crewai)

    embeddings = _make_embeddings(adapter, framework_version="crewai-test-version")

    crew_ctx = {
        "agent_role": "researcher",
        "task_id": "task-123",
        "crew_id": "crew-xyz",
        "workflow": "unit-test",
    }

    result = embeddings.embed_documents(["foo", "bar"], crewai_context=crew_ctx)
    _assert_embedding_matrix_shape(result, expected_rows=2)

    assert captured.get("ctx") is not None
    assert captured["ctx"] == crew_ctx
    assert captured["framework_version"] == "crewai-test-version"


def test_error_context_includes_crewai_context(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    """
    When an error occurs during CrewAI embedding, error context should include
    CrewAI-specific metadata via attach_context().

    Deterministic: inject a failing translator so we test framework-adapter behavior
    (not backend adapter behavior).
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(crewai_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def embed(self, raw_texts: Any, op_ctx: Any = None, framework_ctx: Any = None) -> Any:
            raise RuntimeError("test error from crewai adapter: Check model configuration and API keys")

    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=adapter, model="test-model")
    monkeypatch.setattr(embeddings, "_translator", FailingTranslator())

    crew_ctx = {"agent_role": "tester", "task_id": "task-123"}

    with pytest.raises(RuntimeError, match="test error from crewai adapter"):
        embeddings.embed_documents(["text"], crewai_context=crew_ctx)

    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "crewai"
    assert captured_context.get("agent_role") == "tester"
    assert captured_context.get("task_id") == "task-123"
    assert captured_context.get("operation") == "embedding_documents"
    assert captured_context.get("error_codes") == crewai_adapter_module.EMBEDDING_COERCION_ERROR_CODES


# ---------------------------------------------------------------------------
# Additional config / context nuance tests
# ---------------------------------------------------------------------------

def test_fallback_to_simple_context_true_uses_default_operation_context(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    """Fallback behavior stable across CrewAI versions."""
    calls: Dict[str, Any] = {}

    def fake_from_crewai(crewai_ctx: Mapping[str, Any], framework_version: Any = None) -> Any:
        class WeirdCtx:
            pass
        calls["ctx"] = dict(crewai_ctx)
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


def test_fallback_to_simple_context_false_leaves_core_ctx_none(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    """Context propagation configurable across versions."""
    def fake_from_crewai(crewai_ctx: Mapping[str, Any], framework_version: Any = None) -> Any:
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
    monkeypatch: pytest.MonkeyPatch,
    adapter,
) -> None:
    """Context propagation stable across CrewAI versions."""
    def fake_from_crewai(crewai_ctx: Mapping[str, Any], framework_version: Any = None) -> OperationContext:
        return OperationContext(request_id="r1")

    monkeypatch.setattr(crewai_adapter_module, "context_from_crewai", fake_from_crewai)

    crewai_context = {"agent_role": "analyst"}

    emb_enabled = CorpusCrewAIEmbeddings(
        corpus_adapter=adapter,
        crewai_config=CrewAIConfig(enable_agent_context_propagation=True),
    )
    core_ctx, framework_ctx = emb_enabled._build_contexts(crewai_context=crewai_context)
    assert isinstance(core_ctx, OperationContext)
    assert framework_ctx["_operation_context"] is core_ctx

    emb_disabled = CorpusCrewAIEmbeddings(
        corpus_adapter=adapter,
        crewai_config=CrewAIConfig(enable_agent_context_propagation=False),
    )
    core_ctx2, framework_ctx2 = emb_disabled._build_contexts(crewai_context=crewai_context)
    assert isinstance(core_ctx2, OperationContext)
    assert "_operation_context" not in framework_ctx2


def test_task_aware_batching_sets_batch_strategy(adapter) -> None:
    """Task-aware batching optimization for CrewAI workflows."""
    embeddings = CorpusCrewAIEmbeddings(
        corpus_adapter=adapter,
        crewai_config=CrewAIConfig(task_aware_batching=True),
    )

    crewai_context = {"agent_role": "analyst", "task_id": "task-123"}
    _core_ctx, framework_ctx = embeddings._build_contexts(crewai_context=crewai_context)

    assert framework_ctx["batch_strategy"] == "task_aware_task-123"


def test_non_mapping_crewai_context_raises_value_error(adapter) -> None:
    """Clear error when context has wrong type."""
    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=adapter)

    with pytest.raises(ValueError) as exc:
        embeddings._build_contexts(crewai_context="not-a-mapping")  # type: ignore[arg-type]

    error_msg = str(exc.value)
    assert ErrorCodes.CREWAI_CONTEXT_INVALID in error_msg
    assert "mapping" in error_msg.lower() or "dict" in error_msg.lower()


def test_context_from_crewai_failure_attaches_error_context(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    """Context preserved even when context translation fails."""
    captured: Dict[str, Any] = {}

    def fake_from_crewai(crewai_ctx: Mapping[str, Any], framework_version: Any = None) -> Any:
        raise RuntimeError("boom")

    def fake_attach_context(exc: BaseException, **kwargs: Any) -> None:
        captured["exc"] = exc
        captured["kwargs"] = kwargs

    monkeypatch.setattr(crewai_adapter_module, "context_from_crewai", fake_from_crewai)
    monkeypatch.setattr(crewai_adapter_module, "attach_context", fake_attach_context)

    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=adapter)

    crewai_context = {"agent_role": "analyst", "task_id": "t1"}
    _core_ctx, _framework_ctx = embeddings._build_contexts(crewai_context=crewai_context)

    assert "exc" in captured
    assert "crewai_context_snapshot" in captured["kwargs"]
    assert captured["kwargs"]["error_codes"] == crewai_adapter_module.EMBEDDING_COERCION_ERROR_CODES
    assert captured["kwargs"].get("framework") == "crewai"


# ---------------------------------------------------------------------------
# Sync semantics
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


def test_sync_embed_documents_with_crewai_context(adapter) -> None:
    """Context parameter stable across versions."""
    embeddings = _make_embeddings(adapter)

    ctx = {"agent_role": "tester", "task_id": "shape-check"}

    result = embeddings.embed_documents(["ctx-one", "ctx-two"], crewai_context=ctx)
    _assert_embedding_matrix_shape(result, expected_rows=2)


def test_embed_documents_rejects_non_string_items(adapter) -> None:
    """Clear error for type mismatches."""
    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        embeddings.embed_documents(["ok", 123])  # type: ignore[list-item]

    error_msg = str(exc.value)
    assert "embed_documents expects Sequence[str]" in error_msg
    assert "item 1" in error_msg


def test_embed_query_rejects_non_string(adapter) -> None:
    """Clear error for type mismatches."""
    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        embeddings.embed_query(123)  # type: ignore[arg-type]

    error_msg = str(exc.value)
    assert "embed_query expects str" in error_msg


# ---------------------------------------------------------------------------
# Async semantics
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_embed_documents_and_query_basic(adapter) -> None:
    """Async aembed_documents / aembed_query produce shapes compatible with sync API."""
    embeddings = _make_embeddings(adapter)

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
    """
    Sync/async parity maintained across versions.

    To avoid deadlocks/loop-guard issues, run sync APIs off the event loop.
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
    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        await embeddings.aembed_documents(["ok", 123])  # type: ignore[list-item]

    error_msg = str(exc.value)
    assert "aembed_documents expects Sequence[str]" in error_msg
    assert "item 1" in error_msg


@pytest.mark.asyncio
async def test_aembed_query_rejects_non_string(adapter) -> None:
    """Consistent error messages for async API."""
    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=adapter)

    with pytest.raises(TypeError) as exc:
        await embeddings.aembed_query(123)  # type: ignore[arg-type]

    error_msg = str(exc.value)
    assert "aembed_query expects str" in error_msg


def test_crewai_interface_compatibility(adapter) -> None:
    """
    Verify that CorpusCrewAIEmbeddings implements the expected CrewAI embedder surface.

    This is a protocol-level test only; real CrewAI object integration is tested below.
    """
    embeddings = _make_embeddings(adapter)

    assert hasattr(embeddings, "embed_documents")
    assert hasattr(embeddings, "embed_query")
    assert callable(embeddings.embed_documents)
    assert callable(embeddings.embed_query)

    assert hasattr(embeddings, "aembed_documents")
    assert hasattr(embeddings, "aembed_query")
    assert callable(embeddings.aembed_documents)
    assert callable(embeddings.aembed_query)

    result_docs = embeddings.embed_documents(["test1", "test2"])
    assert isinstance(result_docs, list)
    assert len(result_docs) == 2
    assert all(isinstance(emb, list) for emb in result_docs)

    result_query = embeddings.embed_query("test query")
    assert isinstance(result_query, list)
    assert all(isinstance(x, (int, float)) for x in result_query)


# ---------------------------------------------------------------------------
# Error-context richness for failures (Deterministic via translator injection)
# ---------------------------------------------------------------------------

def test_embed_documents_error_context_includes_crewai_fields(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    """Production debugging context in errors (sync)."""
    class FailingTranslator:
        def embed(self, raw_texts: Any, op_ctx: Any = None, framework_ctx: Any = None) -> Any:
            raise RuntimeError("translator failed: Check model configuration and API limits")

    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **kwargs: Any) -> None:
        captured["exc"] = exc
        captured["kwargs"] = kwargs

    monkeypatch.setattr(crewai_adapter_module, "attach_context", fake_attach_context)

    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=adapter, model="test-model")
    monkeypatch.setattr(embeddings, "_translator", FailingTranslator())

    crewai_context = {
        "agent_role": "analyst",
        "task_id": "t1",
        "workflow": "wf",
        "crew_id": "crew-1",
    }

    with pytest.raises(RuntimeError) as exc_info:
        embeddings.embed_documents(["doc1", "doc2"], crewai_context=crewai_context)

    error_str = str(exc_info.value)
    assert "translator failed" in error_str
    assert "Check model configuration" in error_str

    ctx = captured["kwargs"]
    assert ctx["framework"] == "crewai"
    assert ctx["operation"] == "embedding_documents"
    assert ctx["error_codes"] == crewai_adapter_module.EMBEDDING_COERCION_ERROR_CODES
    assert ctx["agent_role"] == "analyst"
    assert ctx["task_id"] == "t1"
    assert ctx["model"] == "test-model"


@pytest.mark.asyncio
async def test_aembed_query_error_context_includes_crewai_fields(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    """Production debugging context in errors (async)."""
    class FailingTranslator:
        async def arun_embed(self, raw_texts: Any, op_ctx: Any = None, framework_ctx: Any = None) -> Any:
            raise RuntimeError("translator failed: Verify API key and model access permissions")

    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **kwargs: Any) -> None:
        captured["exc"] = exc
        captured["kwargs"] = kwargs

    monkeypatch.setattr(crewai_adapter_module, "attach_context", fake_attach_context)

    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=adapter, model="test-model")
    monkeypatch.setattr(embeddings, "_translator", FailingTranslator())

    crewai_context = {"agent_role": "analyst", "task_id": "t1"}

    with pytest.raises(RuntimeError) as exc_info:
        await embeddings.aembed_query("hello", crewai_context=crewai_context)

    error_str = str(exc_info.value)
    assert "translator failed" in error_str
    assert "Verify API key" in error_str

    ctx = captured["kwargs"]
    assert ctx["framework"] == "crewai"
    assert ctx["operation"] == "embedding_query"
    assert ctx["agent_role"] == "analyst"
    assert ctx["task_id"] == "t1"
    assert "model" in ctx


# ---------------------------------------------------------------------------
# Capabilities / health passthrough
# ---------------------------------------------------------------------------

def test_capabilities_passthrough_when_underlying_provides(adapter) -> None:
    """
    capabilities() surfaces adapter capabilities (best-effort).

    Parity policy with other framework adapters/tests:
    - capabilities() may return a plain mapping OR a typed EmbeddingCapabilities object.
    """
    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=adapter)
    caps = embeddings.capabilities()
    assert isinstance(caps, (dict, EmbeddingCapabilities))


@pytest.mark.asyncio
async def test_async_capabilities_fallback_to_sync(adapter) -> None:
    """
    acapabilities() works even when only sync capabilities exist.

    Parity policy with other framework adapters/tests:
    - acapabilities() may return a plain mapping OR a typed EmbeddingCapabilities object.
    """
    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=adapter)
    acaps = await embeddings.acapabilities()
    assert isinstance(acaps, (dict, EmbeddingCapabilities))


def test_capabilities_empty_when_missing() -> None:
    """If adapter lacks capabilities methods, return empty mapping."""
    class NoCapAdapter:
        async def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 3 for _ in texts]

    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=NoCapAdapter())
    caps = embeddings.capabilities()
    assert isinstance(caps, dict)
    assert caps == {}


def test_health_passthrough_and_missing() -> None:
    """health() mirrors capabilities behavior."""
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
# Integration Tests with Real CrewAI Objects (Pass/Fail; never skip)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCrewAIIntegration:
    """
    Real integration tests with CrewAI objects.

    Policy: pass/fail only. If CrewAI isn't available, require_crewai fails-fast.
    """

    @pytest.fixture(autouse=True)
    def setup_crewai_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Set up environment for CrewAI tests.

        We set a placeholder API key to satisfy versions that validate presence
        even when no network calls are made during these tests.
        """
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-testing")

    def test_can_create_embedder_for_crewai_agent(self, require_crewai, adapter) -> None:
        import crewai  # type: ignore

        embedder = create_embedder(adapter, model="mock-embed-512")

        agent = crewai.Agent(
            role="researcher",
            goal="Research topics",
            backstory="A curious researcher",
            allow_delegation=False,
            verbose=False,
        )

        agent.embedder = embedder
        assert agent.embedder is embedder
        assert hasattr(agent.embedder, "embed_documents")
        assert hasattr(agent.embedder, "embed_query")

        result = agent.embedder.embed_documents(["test document"])
        assert isinstance(result, list)
        assert len(result) == 1
        assert all(isinstance(x, (int, float)) for row in result for x in row)

    def test_embedder_works_with_crewai_knowledge_sources(self, require_crewai, adapter) -> None:
        import crewai  # type: ignore

        embedder = create_embedder(adapter, model="mock-embed-512")

        agent = crewai.Agent(
            role="researcher",
            goal="Research topics",
            backstory="A curious researcher",
            allow_delegation=False,
            verbose=False,
        )
        agent.embedder = embedder

        embeddings = agent.embedder.embed_query("test query")
        assert isinstance(embeddings, list)
        assert all(isinstance(x, (int, float)) for x in embeddings)

    def test_crew_with_multiple_agents_sharing_embedder(self, require_crewai, adapter) -> None:
        import crewai  # type: ignore
        from crewai import Task  # type: ignore

        embedder = create_embedder(adapter, model="mock-embed-512")

        researcher = crewai.Agent(
            role="researcher",
            goal="Research information",
            backstory="Expert researcher",
            allow_delegation=False,
            verbose=False,
        )
        researcher.embedder = embedder

        analyst = crewai.Agent(
            role="analyst",
            goal="Analyze research",
            backstory="Data analyst",
            allow_delegation=False,
            verbose=False,
        )
        analyst.embedder = embedder

        research_task = Task(
            description="Research AI trends",
            agent=researcher,
            expected_output="Research report",
        )
        analysis_task = Task(
            description="Analyze research findings",
            agent=analyst,
            expected_output="Analysis report",
        )

        crew = crewai.Crew(
            agents=[researcher, analyst],
            tasks=[research_task, analysis_task],
            verbose=False,
        )

        assert crew is not None
        assert researcher.embedder is embedder
        assert analyst.embedder is embedder
        assert researcher.embedder is analyst.embedder

        for agent in [researcher, analyst]:
            vec = agent.embedder.embed_query(f"query from {agent.role}")
            assert isinstance(vec, list)
            assert len(vec) > 0
            assert all(isinstance(x, (int, float)) for x in vec)

    def test_error_handling_in_crewai_workflow(self, require_crewai) -> None:
        import crewai  # type: ignore

        class FailingTestAdapter:
            async def embed(self, texts: List[str], ctx: Any = None) -> List[List[float]]:
                raise RuntimeError("Rate limit exceeded: Please wait before retrying")

            async def embed_batch(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("Rate limit exceeded: Please wait before retrying")

            def capabilities(self) -> Dict[str, Any]:
                return {"supported_models": ["mock-embed-512"]}

        adapter = FailingTestAdapter()
        embedder = create_embedder(adapter, model="mock-embed-512")

        agent = crewai.Agent(
            role="researcher",
            goal="Research topics",
            backstory="A curious researcher",
            allow_delegation=False,
            verbose=False,
        )
        agent.embedder = embedder

        with pytest.raises(Exception) as exc_info:
            agent.embedder.embed_documents(["test document"])

        error_str = str(exc_info.value).lower()
        assert ("rate limit" in error_str) or ("exceeded" in error_str)
        assert ("wait" in error_str) or ("retry" in error_str)

    @pytest.mark.asyncio
    async def test_async_embedding_in_crewai_workflow(self, require_crewai, adapter) -> None:
        import crewai  # type: ignore

        embedder = create_embedder(adapter, model="mock-embed-512")

        agent = crewai.Agent(
            role="researcher",
            goal="Research topics",
            backstory="A curious researcher",
            allow_delegation=False,
            verbose=False,
        )
        agent.embedder = embedder

        embeddings = await agent.embedder.aembed_query("async query")
        assert isinstance(embeddings, list)
        assert all(isinstance(x, (int, float)) for x in embeddings)


# ---------------------------------------------------------------------------
# Concurrency Tests
# ---------------------------------------------------------------------------

@pytest.mark.concurrency
class TestConcurrency:
    """Concurrency and thread-safety tests."""

    def test_shared_embedder_thread_safety(self, adapter) -> None:
        embedder = create_embedder(adapter, model="mock-embed-512")

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
    async def test_concurrent_async_embedding(self, adapter) -> None:
        embedder = create_embedder(adapter, model="mock-embed-512")

        async def embed_async(text: str) -> List[float]:
            return await embedder.aembed_query(text)

        texts = [f"async query {i}" for i in range(5)]
        results = await asyncio.gather(*[embed_async(text) for text in texts])

        assert len(results) == len(texts)
        for result in results:
            assert isinstance(result, list)
            assert all(isinstance(x, (int, float)) for x in result)


# ---------------------------------------------------------------------------
# Crew registration helpers
# ---------------------------------------------------------------------------

def test_register_with_crewai_attaches_embedder_to_agents(adapter) -> None:
    """Registration helper stable across versions."""
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

    for agent in crew.agents:
        assert agent.embedder is emb


def test_register_with_crewai_handles_agents_callable(adapter) -> None:
    """Flexible agent access pattern."""
    class DummyAgent:
        def __init__(self) -> None:
            self.embedder: Any | None = None

    class DummyCrewCallable:
        def __init__(self) -> None:
            self._agents = [DummyAgent()]
            self.name = "callable-crew"

        def agents(self):
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
    """Graceful handling of missing agents."""
    class CrewNoAgents:
        def __init__(self) -> None:
            self.name = "no-agents-crew"

    crew = CrewNoAgents()

    emb = register_with_crewai(
        crew=crew,
        corpus_adapter=adapter,
    )
    assert isinstance(emb, CorpusCrewAIEmbeddings)


def test_register_with_crewai_crew_none_raises_value_error(adapter) -> None:
    """Clear error for invalid crew."""
    with pytest.raises(ValueError) as exc:
        register_with_crewai(None, adapter)  # type: ignore[arg-type]
    assert "crew cannot be None" in str(exc.value)


def test_register_with_crewai_agents_callable_that_raises_attaches_error_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter,
) -> None:
    """Errors during registration include context."""
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **kwargs: Any) -> None:
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
    assert captured["kwargs"].get("framework") == "crewai"
