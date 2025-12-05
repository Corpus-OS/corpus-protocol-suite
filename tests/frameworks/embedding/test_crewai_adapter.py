# tests/frameworks/embedding/test_crewai_adapter.py

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict

import inspect
import pytest

import corpus_sdk.embedding.framework_adapters.crewai as crewai_adapter_module
from corpus_sdk.embedding.framework_adapters.crewai import (
    CorpusCrewAIEmbeddings,
    create_embedder,
    register_with_crewai,
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


def _make_embeddings(adapter: Any, **kwargs: Any) -> CorpusCrewAIEmbeddings:
    """
    Construct a CorpusCrewAIEmbeddings instance from the generic adapter.
    """
    return CorpusCrewAIEmbeddings(corpus_adapter=adapter, **kwargs)


# ---------------------------------------------------------------------------
# Constructor / config behavior
# ---------------------------------------------------------------------------


def test_constructor_rejects_adapter_without_embed() -> None:
    """
    CorpusCrewAIEmbeddings should enforce that corpus_adapter exposes
    an `embed` method; otherwise __init__ should raise TypeError.
    """

    class BadAdapter:
        # deliberately missing `embed`
        def __init__(self) -> None:
            pass

    with pytest.raises(TypeError) as exc_info:
        CorpusCrewAIEmbeddings(corpus_adapter=BadAdapter())

    msg = str(exc_info.value)
    assert "must implement an EmbeddingProtocolV1-compatible interface" in msg


def test_crewai_config_defaults_and_bool_coercion(adapter: Any) -> None:
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


def test_create_embedder_returns_crewai_embeddings(adapter: Any) -> None:
    """
    create_embedder should return a CorpusCrewAIEmbeddings wired to the adapter.
    """
    from corpus_sdk.embedding.framework_adapters.crewai import create_embedder

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
    adapter: Any,
) -> None:
    """
    Verify that crewai_context is passed through to context_from_crewai
    when embedding.
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
        def embed(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("test error from crewai adapter")
        
        async def embed_batch(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("test error from crewai adapter")

    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=FailingAdapter())

    crew_ctx = {"agent_role": "tester", "task_id": "task-123"}

    with pytest.raises(RuntimeError, match="test error from crewai adapter"):
        embeddings.embed_documents(["text"], crewai_context=crew_ctx)

    # Verify some context was attached
    assert captured_context, "attach_context was not called"

    # Framework tagging should be present
    assert "framework" in captured_context
    # If your adapter uses a different tag, tweak this accordingly:
    assert captured_context.get("framework") == "crewai"

    # CrewAI-specific fields should be present in the context
    assert captured_context.get("agent_role") == "tester"
    assert captured_context.get("task_id") == "task-123"


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


def test_sync_embed_documents_with_crewai_context(adapter: Any) -> None:
    """
    embed_documents should accept crewai_context kwarg and not raise TypeError.
    """
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


def test_crewai_interface_compatibility(adapter: Any) -> None:
    """
    Verify that CorpusCrewAIEmbeddings implements the expected CrewAI
    Embeddings interface when CrewAI is available.
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
    except ImportError:
        pytest.skip("CrewAI is not installed; cannot assert interface compatibility")

    assert isinstance(
        embeddings,
        Embeddings,
    ), "CorpusCrewAIEmbeddings should subclass CrewAI Embeddings when available"


# ---------------------------------------------------------------------------
# Capabilities / health passthrough
# ---------------------------------------------------------------------------


def test_capabilities_passthrough_when_underlying_provides() -> None:
    """
    When the underlying adapter implements capabilities/acapabilities,
    CorpusCrewAIEmbeddings should surface them.
    """

    class CapAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0, 1.0] for _ in texts]

        def capabilities(self) -> Dict[str, Any]:
            return {"ok": True}

    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=CapAdapter())

    caps = embeddings.capabilities()
    assert isinstance(caps, dict)
    assert caps.get("ok") is True


@pytest.mark.asyncio
async def test_async_capabilities_fallback_to_sync() -> None:
    """
    acapabilities should fall back to sync capabilities() when only the
    sync method is implemented on the underlying adapter.
    """

    class CapAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0, 1.0] for _ in texts]

        def capabilities(self) -> Dict[str, Any]:
            return {"via_sync": True}

    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=CapAdapter())

    acaps = await embeddings.acapabilities()
    assert isinstance(acaps, dict)
    assert acaps.get("via_sync") is True


def test_capabilities_raises_when_missing() -> None:
    """
    If the underlying adapter has no capabilities()/acapabilities(),
    the CrewAI adapter should raise NotImplementedError.
    """

    class NoCapAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 3 for _ in texts]

    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=NoCapAdapter())

    with pytest.raises(NotImplementedError):
        embeddings.capabilities()

    # acapabilities should also raise
    with pytest.raises(NotImplementedError):
        import asyncio

        asyncio.run(embeddings.acapabilities())


def test_health_passthrough_and_missing() -> None:
    """
    health/ahealth behavior mirrors capabilities/acapabilities.
    """

    class HealthAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 2 for _ in texts]

        def health(self) -> Dict[str, Any]:
            return {"status": "ok"}

    embeddings = CorpusCrewAIEmbeddings(corpus_adapter=HealthAdapter())

    health = embeddings.health()
    assert isinstance(health, dict)
    assert health.get("status") == "ok"

    # Adapter with no health() methods
    class NoHealthAdapter:
        def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
            return [[0.0] * 2 for _ in texts]

    embeddings2 = CorpusCrewAIEmbeddings(corpus_adapter=NoHealthAdapter())

    with pytest.raises(NotImplementedError):
        embeddings2.health()

    with pytest.raises(NotImplementedError):
        import asyncio

        asyncio.run(embeddings2.ahealth())


# ---------------------------------------------------------------------------
# Crew registration helpers
# ---------------------------------------------------------------------------


def test_register_with_crewai_attaches_embedder_to_agents(adapter: Any) -> None:
    """
    register_with_crewai should attach the embedder to each agent that has
    an `embedder` attribute on a crew with `agents`.
    """

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


def test_register_with_crewai_handles_agents_callable(adapter: Any) -> None:
    """
    register_with_crewai should handle crew.agents being callable that
    returns a list of agents.
    """

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


def test_register_with_crewai_no_agents_attribute(adapter: Any) -> None:
    """
    If crew has no 'agents' attribute, register_with_crewai should still
    return an embedder without raising; it just can't auto-attach.
    """

    class CrewNoAgents:
        def __init__(self) -> None:
            self.name = "no-agents-crew"

    crew = CrewNoAgents()

    emb = register_with_crewai(
        crew=crew,
        corpus_adapter=adapter,
    )
    assert isinstance(emb, CorpusCrewAIEmbeddings)
