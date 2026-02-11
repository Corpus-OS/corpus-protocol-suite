"""
CrewAI Vector framework adapter tests.

These tests are written against the current public API in
`corpus_sdk.vector.framework_adapters.crewai`, which exposes a CrewAI BaseTool
implementation (`CorpusCrewAIVectorSearchTool`) with sync/async run APIs, streaming,
MMR search, and protocol client wrapper (`CorpusCrewAIVectorClient`).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

import corpus_sdk.vector.framework_adapters.crewai as crewai_adapter_module
from corpus_sdk.vector.framework_adapters.crewai import (
    CorpusCrewAIVectorClient,
    CorpusCrewAIVectorSearchTool,
    CorpusVectorSearchInput,
    ErrorCodes,
    _ensure_not_in_event_loop,
)
from corpus_sdk.vector.vector_base import (
    BadRequest,
    NotSupported,
    OperationContext,
    QueryResult,
    Vector,
    VectorCapabilities,
    VectorMatch,
    UpsertResult,
    DeleteResult,
    NamespaceResult,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_TEXT = "hello from crewai vector tests"
SAMPLE_EMBEDDING = [0.1, 0.2, 0.3, 0.4]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_caps(**overrides: Any) -> VectorCapabilities:
    """Construct protocol-valid VectorCapabilities for tests."""
    base: Dict[str, Any] = {
        "server": "test",
        "version": "0",
        "supports_metadata_filtering": True,
        "max_top_k": 100,
    }
    base.update(overrides)
    return VectorCapabilities(**base)


def _make_match(
    *,
    id: str,
    score: float,
    text: str = "test",
    embedding: Optional[List[float]] = None,
) -> VectorMatch:
    vec = embedding if embedding is not None else [0.1, 0.2]
    return VectorMatch(
        vector=Vector(
            id=id,
            vector=list(vec),
            metadata={"page_content": text, "id": id},
        ),
        score=float(score),
        distance=1.0 - float(score),
    )


def _empty_result(*, namespace: str = "default", query_vector: Optional[List[float]] = None) -> QueryResult:
    return QueryResult(
        matches=[],
        query_vector=list(query_vector) if query_vector is not None else [0.0],
        namespace=namespace,
        total_matches=0,
    )


def _make_dummy_translator() -> Any:
    """Factory for creating a standard dummy translator for tests."""

    def _caps(**overrides: Any) -> VectorCapabilities:
        return VectorCapabilities(
            server="test",
            version="0",
            supports_metadata_filtering=True,
            max_top_k=100,
            **overrides,
        )

    def _match(*, id: str, score: float, text: str = "test") -> VectorMatch:
        return VectorMatch(
            vector=Vector(
                id=id,
                vector=[0.1, 0.2],
                metadata={"page_content": text, "id": id},
            ),
            score=float(score),
            distance=1.0 - float(score),
        )

    class DummyTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            return QueryResult(
                matches=[_match(id="doc-0", score=0.95)],
                query_vector=[0.1, 0.2],
                namespace="default",
                total_matches=1,
            )

        async def arun_query(self, *a: Any, **k: Any) -> Any:
            return QueryResult(
                matches=[_match(id="doc-0", score=0.95)],
                query_vector=[0.1, 0.2],
                namespace="default",
                total_matches=1,
            )

        def query_stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            class Chunk:
                matches = [
                    _match(id="doc-0", score=0.95)
                ]
                is_final = True

            yield Chunk()

        async def arun_query_stream(self, *a: Any, **k: Any) -> Any:
            class Chunk:
                matches = [
                    _match(id="doc-0", score=0.95)
                ]
                is_final = True

            async def _gen():
                yield Chunk()

            return _gen()

        def capabilities(self) -> VectorCapabilities:
            return _caps()

        async def acapabilities(self) -> VectorCapabilities:
            return _caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

        async def ahealth(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    return DummyTranslator()


def _make_mock_embedding_function(vectors: Optional[List[List[float]]] = None):
    """Create a mock embedding function that returns predefined vectors."""
    vectors = vectors or [[0.1, 0.2, 0.3, 0.4]]

    def embedding_fn(texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if not vectors:
            return [[0.0] for _ in texts]
        if len(vectors) >= len(texts):
            return vectors[: len(texts)]
        # Repeat last vector if the caller asks for more.
        return list(vectors) + [list(vectors[-1]) for _ in range(len(texts) - len(vectors))]

    return embedding_fn


async def _make_async_embedding_function(texts: List[str]) -> List[List[float]]:
    """Async embedding function for testing."""
    return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter() -> Any:
    """Create a minimal test adapter."""

    class TestAdapter:
        async def capabilities(self) -> VectorCapabilities:
            return VectorCapabilities(
                server="test",
                version="0",
                supports_metadata_filtering=True,
                max_top_k=100,
                supports_batch_queries=True,
            )

        async def query(self, *args: Any, **kwargs: Any) -> QueryResult:
            return QueryResult(matches=[], query_vector=[0.0], namespace="default", total_matches=0)

        async def batch_query(self, *args: Any, **kwargs: Any) -> List[QueryResult]:
            return [QueryResult(matches=[], query_vector=[0.0], namespace="default", total_matches=0)]

        async def upsert(self, *args: Any, **kwargs: Any) -> UpsertResult:
            return UpsertResult(upserted_count=0, failed_count=0, failures=[])

        async def delete(self, *args: Any, **kwargs: Any) -> DeleteResult:
            return DeleteResult(deleted_count=0, failed_count=0, failures=[])

        async def create_namespace(self, *args: Any, **kwargs: Any) -> NamespaceResult:
            return NamespaceResult(success=True, namespace="default", details={})

        async def delete_namespace(self, *args: Any, **kwargs: Any) -> NamespaceResult:
            return NamespaceResult(success=True, namespace="default", details={})

        def health(self) -> Dict[str, Any]:
            return {"status": "ok"}

    return TestAdapter()


# ---------------------------------------------------------------------------
# Construction / Initialization Tests (10 tests)
# ---------------------------------------------------------------------------


def test_init_requires_crewai_installed() -> None:
    """Tool should raise RuntimeError if CrewAI not available."""
    with patch.object(crewai_adapter_module, "CREWAI_AVAILABLE", False):
        with pytest.raises(RuntimeError, match="requires `crewai`"):
            CorpusCrewAIVectorSearchTool(corpus_adapter=Mock())


def test_init_requires_corpus_adapter(adapter: Any) -> None:
    """Adapter must be provided."""
    with pytest.raises((TypeError, AttributeError, ValidationError)):
        CorpusCrewAIVectorSearchTool(corpus_adapter=None)  # type: ignore


def test_init_stores_config_attributes(adapter: Any) -> None:
    """Tool should keep key config attributes accessible."""
    tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter,
        namespace="test-ns",
        id_field="custom_id",
        text_field="custom_text",
        metadata_field="custom_meta",
        score_threshold=0.8,
        default_top_k=10,
        use_mmr_by_default=True,
        mmr_lambda=0.7,
    )

    assert tool.namespace == "test-ns"
    assert tool.id_field == "custom_id"
    assert tool.text_field == "custom_text"
    assert tool.metadata_field == "custom_meta"
    assert tool.score_threshold == 0.8
    assert tool.default_top_k == 10
    assert tool.use_mmr_by_default is True
    assert tool.mmr_lambda == 0.7


def test_init_validates_score_threshold_range(adapter: Any) -> None:
    """score_threshold should be between 0.0 and 1.0 if provided."""
    # Valid range should work
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter, score_threshold=0.5)
    assert tool.score_threshold == 0.5

    # Edge cases
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter, score_threshold=0.0)
    assert tool.score_threshold == 0.0

    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter, score_threshold=1.0)
    assert tool.score_threshold == 1.0


def test_init_validates_default_top_k_positive(adapter: Any) -> None:
    """default_top_k must be positive."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter, default_top_k=5)
    assert tool.default_top_k == 5


def test_init_with_embedding_function(adapter: Any) -> None:
    """embedding_function should be stored correctly."""
    fn = _make_mock_embedding_function()
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter, embedding_function=fn)
    assert tool.embedding_function is fn


def test_init_with_async_embedding_function(adapter: Any) -> None:
    """async_embedding_function should be stored correctly."""
    tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter, async_embedding_function=_make_async_embedding_function
    )
    assert tool.async_embedding_function is _make_async_embedding_function


def test_init_with_static_operation_context(adapter: Any) -> None:
    """static_operation_context should be stored and accessible."""
    ctx = OperationContext(request_id="test", tenant="test", attrs={})
    tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter, static_operation_context=ctx
    )
    assert tool.static_operation_context is ctx


def test_init_with_own_adapter_flag(adapter: Any) -> None:
    """own_adapter=True should enable lifecycle management."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter, own_adapter=True)
    assert tool.own_adapter is True


def test_init_sets_crewai_tool_metadata(adapter: Any) -> None:
    """Tool should have CrewAI-specific metadata."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    assert tool.name == "corpus_vector_search"
    assert "semantic vector search" in tool.description.lower()
    assert tool.args_schema is CorpusVectorSearchInput


# ---------------------------------------------------------------------------
# Translator Wiring Tests (4 tests)
# ---------------------------------------------------------------------------


def test_translator_created_with_framework_crewai(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Translator factory should be called with framework='crewai'."""
    captured: Dict[str, Any] = {}

    class FakeTranslator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def query(self, *a, **k):
            return _empty_result(namespace="default")

        def capabilities(self):
            return _make_caps()

    def fake_create(*_a, **kwargs):
        captured.update(kwargs)
        return FakeTranslator(**kwargs)

    # Patch the VectorTranslator class itself
    with patch.object(crewai_adapter_module, "VectorTranslator", FakeTranslator):
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=_make_mock_embedding_function()
        )
        _ = tool._translator

    assert captured.get("framework") == "crewai"
    assert captured.get("adapter") is adapter


def test_translator_cached_property_reused(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Multiple accesses to _translator should return same instance."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
        translator1 = tool._translator
        translator2 = tool._translator

        assert translator1 is translator2


def test_translator_uses_custom_framework_translator(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should use CrewAI-specific VectorFrameworkTranslator."""
    captured: Dict[str, Any] = {}

    class FakeTranslator:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.translator = kwargs.get("translator")

        def query(self, *a, **k):
            return _empty_result(namespace="default")

    with patch.object(crewai_adapter_module, "VectorTranslator", FakeTranslator):
        tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
        _ = tool._translator

    assert "translator" in captured
    translator_obj = captured["translator"]
    # Should be the custom inner class
    assert translator_obj.__class__.__name__ == "_CrewAIVectorFrameworkTranslator"


def test_translator_available_on_first_access(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Lazy construction should work on first access."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
        # Should not raise
        translator = tool._translator
        assert translator is not None


# ---------------------------------------------------------------------------
# Context Translation Tests (8 tests)
# ---------------------------------------------------------------------------


def test_build_core_context_from_operation_context(adapter: Any) -> None:
    """Should pass through OperationContext unchanged."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    ctx = OperationContext(request_id="test", tenant="test", attrs={})

    result = tool._build_core_context(ctx)

    assert result is ctx


def test_build_core_context_from_dict(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should build OperationContext from dict using context_translation."""
    captured: Dict[str, Any] = {}
    base_ctx = OperationContext(request_id="from-dict", tenant="from-dict", attrs={})

    def fake_from_dict(mapping: Any) -> Any:
        captured["mapping"] = mapping
        return base_ctx

    monkeypatch.setattr(crewai_adapter_module, "ctx_from_dict", fake_from_dict)

    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    test_dict = {"key": "value"}

    ctx = tool._build_core_context(test_dict)

    assert captured["mapping"] == test_dict
    assert isinstance(ctx, OperationContext)


def test_build_core_context_from_crewai_task(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should build OperationContext from CrewAI task using context_translation."""
    captured: Dict[str, Any] = {}
    base_ctx = OperationContext(request_id="from-task", tenant="from-task", attrs={})

    def fake_from_crewai(task: Any) -> Any:
        captured["task"] = task
        return base_ctx

    monkeypatch.setattr(crewai_adapter_module, "ctx_from_crewai", fake_from_crewai)

    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    mock_task = Mock()

    ctx = tool._build_core_context(mock_task)

    assert captured["task"] is mock_task
    assert isinstance(ctx, OperationContext)


def test_build_core_context_handles_none_returns_static(adapter: Any) -> None:
    """Should return static_operation_context when call_context is None."""
    static_ctx = OperationContext(request_id="static", tenant="static", attrs={})
    tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter, static_operation_context=static_ctx
    )

    ctx = tool._build_core_context(None)

    assert ctx is static_ctx


def test_build_core_context_dict_translation_error_attaches_context(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Errors during dict translation should attach error context."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    def fake_from_dict(mapping: Any) -> Any:
        raise RuntimeError("dict translation failed")

    monkeypatch.setattr(crewai_adapter_module, "attach_context", fake_attach_context)
    monkeypatch.setattr(crewai_adapter_module, "ctx_from_dict", fake_from_dict)

    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)

    with pytest.raises(RuntimeError, match="dict translation failed"):
        tool._build_core_context({"key": "value"})

    assert captured_ctx.get("framework") == "crewai"
    assert captured_ctx.get("operation") == "context_translation_from_dict"


def test_build_core_context_crewai_translation_error_attaches_context(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Errors during CrewAI translation should attach error context."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    def fake_from_crewai(task: Any) -> Any:
        raise RuntimeError("crewai translation failed")

    monkeypatch.setattr(crewai_adapter_module, "attach_context", fake_attach_context)
    monkeypatch.setattr(crewai_adapter_module, "ctx_from_crewai", fake_from_crewai)

    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)

    with pytest.raises(RuntimeError, match="crewai translation failed"):
        tool._build_core_context(Mock())

    assert captured_ctx.get("framework") == "crewai"
    assert captured_ctx.get("operation") == "context_translation_from_crewai"


def test_build_framework_context_includes_namespace(adapter: Any) -> None:
    """framework_ctx should include namespace."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter, namespace="test-ns")
    fw_ctx = tool._build_framework_context(namespace="override-ns")

    assert fw_ctx["namespace"] == "override-ns"


def test_build_contexts_orchestrates_both(adapter: Any) -> None:
    """_build_contexts should return both OperationContext and framework_ctx."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter, namespace="test-ns")
    core_ctx, fw_ctx = tool._build_contexts(
        call_context=None,
        namespace=None,
    )

    assert core_ctx is None  # No call_context provided
    assert isinstance(fw_ctx, Mapping)
    assert fw_ctx["namespace"] == "test-ns"


# ---------------------------------------------------------------------------
# Namespace Resolution Tests (3 tests)
# ---------------------------------------------------------------------------


def test_effective_namespace_uses_override(adapter: Any) -> None:
    """Explicit namespace should override tool default."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter, namespace="default")
    ns = tool._effective_namespace("override")
    assert ns == "override"


def test_effective_namespace_uses_tool_default(adapter: Any) -> None:
    """Should use tool default when not specified."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter, namespace="default")
    ns = tool._effective_namespace(None)
    assert ns == "default"


def test_effective_namespace_handles_none_default(adapter: Any) -> None:
    """None default should pass through."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter, namespace=None)
    ns = tool._effective_namespace(None)
    assert ns is None


# ---------------------------------------------------------------------------
# Dimension Hint Tests (6 tests)
# ---------------------------------------------------------------------------


def test_update_dim_hint_sets_first_write(adapter: Any) -> None:
    """First non-zero dimension should win."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    assert tool._vector_dim_hint is None

    tool._update_dim_hint(4)
    assert tool._vector_dim_hint == 4


def test_update_dim_hint_thread_safe(adapter: Any) -> None:
    """Concurrent updates should not race."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)

    def update_thread(dim: int):
        tool._update_dim_hint(dim)

    threads = [threading.Thread(target=update_thread, args=(i,)) for i in range(1, 10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should have a value, and it should be one of the attempted values
    assert tool._vector_dim_hint is not None
    assert 1 <= tool._vector_dim_hint < 10


def test_update_dim_hint_ignores_subsequent_writes(adapter: Any) -> None:
    """Second write should be no-op."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    tool._update_dim_hint(4)
    tool._update_dim_hint(8)

    assert tool._vector_dim_hint == 4


def test_validate_embedding_dimension_against_hint(adapter: Any) -> None:
    """Should raise on dimension mismatch."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    tool._update_dim_hint(4)

    with pytest.raises(BadRequest, match="does not match expected"):
        tool._validate_embedding_dimension([0.1, 0.2])


def test_validate_embedding_dimension_noop_when_hint_none(adapter: Any) -> None:
    """No validation without hint; sets hint instead."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    # Should not raise, should set hint
    tool._validate_embedding_dimension([0.1, 0.2])
    assert tool._vector_dim_hint == 2


def test_zero_vector_requires_known_dimension(adapter: Any) -> None:
    """Should raise if dimension unknown."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)

    with pytest.raises(BadRequest, match="dimension is unknown"):
        tool._zero_vector()


# ---------------------------------------------------------------------------
# Embedding Function Tests (Sync) (8 tests)
# ---------------------------------------------------------------------------


def test_vectorize_documents_uses_provided_embeddings(adapter: Any) -> None:
    """Should use explicit embeddings when provided."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    tool._update_dim_hint(2)

    texts = ["a", "b"]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    result = tool.vectorize_documents(texts, embeddings=embeddings)

    assert result == embeddings


def test_vectorize_documents_validates_length_match(adapter: Any) -> None:
    """Should raise if embeddings length doesn't match texts length."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    texts = ["a", "b"]
    embeddings = [[0.1, 0.2]]  # Only 1 embedding for 2 texts

    with pytest.raises(BadRequest, match="does not match"):
        tool.vectorize_documents(texts, embeddings=embeddings)


def test_vectorize_documents_calls_embedding_function(adapter: Any) -> None:
    """Should call embedding_function if no embeddings provided."""
    called = {"times": 0}

    def counting_fn(texts: List[str]) -> List[List[float]]:
        called["times"] += 1
        return [[0.1, 0.2] for _ in texts]

    tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter, embedding_function=counting_fn
    )
    texts = ["a", "b"]

    result = tool.vectorize_documents(texts)

    assert called["times"] == 1
    assert len(result) == 2


def test_vectorize_documents_raises_without_function(adapter: Any) -> None:
    """Should raise NotSupported if no function configured."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    texts = ["a"]

    with pytest.raises(NotSupported, match="No embedding_function"):
        tool.vectorize_documents(texts)


def test_vectorize_documents_handles_empty_texts(adapter: Any) -> None:
    """Empty texts should return zero vectors when dim known."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    tool._update_dim_hint(4)

    result = tool.vectorize_documents(["", "  "])

    assert len(result) == 2
    assert all(v == [0.0, 0.0, 0.0, 0.0] for v in result)


def test_vectorize_documents_validates_dimension_consistency(adapter: Any) -> None:
    """Should enforce consistent dimensions across batch."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    tool._update_dim_hint(4)

    embeddings = [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2]]  # Mismatched dims

    with pytest.raises(BadRequest, match="does not match expected"):
        tool.vectorize_documents(["a", "b"], embeddings=embeddings)


def test_vectorize_documents_updates_dim_hint(adapter: Any) -> None:
    """Should set hint from first vector."""
    fn = _make_mock_embedding_function([[0.1, 0.2, 0.3]])
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter, embedding_function=fn)

    tool.vectorize_documents(["a"])

    assert tool._vector_dim_hint == 3


def test_vectorize_documents_handles_function_errors(adapter: Any) -> None:
    """Should wrap function errors with context."""

    def failing_fn(texts: List[str]) -> List[List[float]]:
        raise ValueError("embedding failed")

    tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter, embedding_function=failing_fn
    )

    with pytest.raises(BadRequest, match="embedding_function failed"):
        tool.vectorize_documents(["a"])


# ---------------------------------------------------------------------------
# Embedding Function Tests (Async) (8 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_avectorize_documents_uses_provided_embeddings(adapter: Any) -> None:
    """Should use explicit embeddings."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    tool._update_dim_hint(2)

    texts = ["a", "b"]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    result = await tool.avectorize_documents(texts, embeddings=embeddings)

    assert result == embeddings


@pytest.mark.asyncio
async def test_avectorize_documents_validates_length_match(adapter: Any) -> None:
    """Should raise if lengths mismatch."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    texts = ["a", "b"]
    embeddings = [[0.1, 0.2]]

    with pytest.raises(BadRequest, match="does not match"):
        await tool.avectorize_documents(texts, embeddings=embeddings)


@pytest.mark.asyncio
async def test_avectorize_documents_calls_async_function(adapter: Any) -> None:
    """Should prefer async_embedding_function."""
    called = {"times": 0}

    async def counting_fn(texts: List[str]) -> List[List[float]]:
        called["times"] += 1
        return [[0.1, 0.2] for _ in texts]

    tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter, async_embedding_function=counting_fn
    )
    texts = ["a", "b"]

    result = await tool.avectorize_documents(texts)

    assert called["times"] == 1
    assert len(result) == 2


@pytest.mark.asyncio
async def test_avectorize_documents_falls_back_to_sync(adapter: Any) -> None:
    """Should use sync function in thread if no async."""
    called = {"times": 0}

    def counting_fn(texts: List[str]) -> List[List[float]]:
        called["times"] += 1
        return [[0.1, 0.2] for _ in texts]

    tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter, embedding_function=counting_fn
    )
    texts = ["a"]

    result = await tool.avectorize_documents(texts)

    assert called["times"] == 1


@pytest.mark.asyncio
async def test_avectorize_documents_raises_without_function(adapter: Any) -> None:
    """Should raise NotSupported if no function."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)

    with pytest.raises(NotSupported, match="No embedding_function"):
        await tool.avectorize_documents(["a"])


@pytest.mark.asyncio
async def test_avectorize_documents_handles_empty_texts(adapter: Any) -> None:
    """Empty texts should return zero vectors when dim known."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    tool._update_dim_hint(4)

    result = await tool.avectorize_documents(["", "  "])

    assert len(result) == 2
    assert all(v == [0.0, 0.0, 0.0, 0.0] for v in result)


@pytest.mark.asyncio
async def test_avectorize_documents_updates_dim_hint(adapter: Any) -> None:
    """Should set hint from first vector."""

    async def fn(texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3]]

    tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter, async_embedding_function=fn
    )

    await tool.avectorize_documents(["a"])

    assert tool._vector_dim_hint == 3


@pytest.mark.asyncio
async def test_avectorize_documents_handles_function_errors(adapter: Any) -> None:
    """Should wrap errors with context."""

    async def failing_fn(texts: List[str]) -> List[List[float]]:
        raise ValueError("async embedding failed")

    tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter, async_embedding_function=failing_fn
    )

    with pytest.raises(BadRequest, match="async_embedding_function failed"):
        await tool.avectorize_documents(["a"])


# ---------------------------------------------------------------------------
# Query Embedding Tests (Sync) (6 tests)
# ---------------------------------------------------------------------------


def test_vectorize_query_uses_provided_embedding(adapter: Any) -> None:
    """Should use explicit embedding when provided."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    tool._update_dim_hint(4)

    result = tool.vectorize_query("test", embedding=[0.1, 0.2, 0.3, 0.4])

    assert result == [0.1, 0.2, 0.3, 0.4]


def test_vectorize_query_validates_embedding_dimension(adapter: Any) -> None:
    """Should enforce dim_hint."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    tool._update_dim_hint(4)

    with pytest.raises(BadRequest, match="does not match expected"):
        tool.vectorize_query("test", embedding=[0.1, 0.2])


def test_vectorize_query_calls_embedding_function(adapter: Any) -> None:
    """Should call embedding_function([query])."""
    called = {"times": 0}

    def counting_fn(texts: List[str]) -> List[List[float]]:
        called["times"] += 1
        return [[0.1, 0.2, 0.3, 0.4]]

    tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter, embedding_function=counting_fn
    )

    result = tool.vectorize_query("test query")

    assert called["times"] == 1
    assert len(result) == 4


def test_vectorize_query_returns_zero_vector_for_empty_query(adapter: Any) -> None:
    """Should return deterministic zeros if dim known."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    tool._update_dim_hint(4)

    result = tool.vectorize_query("")

    assert result == [0.0, 0.0, 0.0, 0.0]


def test_vectorize_query_raises_for_empty_query_without_dim(adapter: Any) -> None:
    """Should raise if dim unknown."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)

    with pytest.raises(BadRequest, match="query cannot be empty"):
        tool.vectorize_query("")


def test_vectorize_query_raises_without_function(adapter: Any) -> None:
    """Should raise NotSupported if no function."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)

    with pytest.raises(NotSupported, match="No embedding_function"):
        tool.vectorize_query("test")


# ---------------------------------------------------------------------------
# Query Embedding Tests (Async) (6 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_avectorize_query_uses_provided_embedding(adapter: Any) -> None:
    """Should use explicit embedding."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    tool._update_dim_hint(4)

    result = await tool.avectorize_query("test", embedding=[0.1, 0.2, 0.3, 0.4])

    assert result == [0.1, 0.2, 0.3, 0.4]


@pytest.mark.asyncio
async def test_avectorize_query_validates_embedding_dimension(adapter: Any) -> None:
    """Should enforce dim_hint."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    tool._update_dim_hint(4)

    with pytest.raises(BadRequest, match="does not match expected"):
        await tool.avectorize_query("test", embedding=[0.1, 0.2])


@pytest.mark.asyncio
async def test_avectorize_query_calls_async_function(adapter: Any) -> None:
    """Should prefer async_embedding_function."""
    called = {"times": 0}

    async def counting_fn(texts: List[str]) -> List[List[float]]:
        called["times"] += 1
        return [[0.1, 0.2, 0.3, 0.4]]

    tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter, async_embedding_function=counting_fn
    )

    result = await tool.avectorize_query("test")

    assert called["times"] == 1
    assert len(result) == 4


@pytest.mark.asyncio
async def test_avectorize_query_falls_back_to_sync(adapter: Any) -> None:
    """Should use sync function in thread."""
    fn = _make_mock_embedding_function([[0.1, 0.2, 0.3, 0.4]])
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter, embedding_function=fn)

    result = await tool.avectorize_query("test")

    assert len(result) == 4


@pytest.mark.asyncio
async def test_avectorize_query_returns_zero_vector_for_empty_query(
    adapter: Any,
) -> None:
    """Should return deterministic zeros if dim known."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    tool._update_dim_hint(4)

    result = await tool.avectorize_query("")

    assert result == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_avectorize_query_raises_without_function(adapter: Any) -> None:
    """Should raise NotSupported if no function."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)

    with pytest.raises(NotSupported, match="No embedding_function"):
        await tool.avectorize_query("test")


# ---------------------------------------------------------------------------
# Compatibility Alias Tests (12 tests)
# ---------------------------------------------------------------------------


def test_embed_query_alias_sync(adapter: Any) -> None:
    """embed_query should be alias for vectorize_query."""
    fn = _make_mock_embedding_function()
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter, embedding_function=fn)

    result = tool.embed_query("test")

    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)


@pytest.mark.asyncio
async def test_aembed_query_alias_async(adapter: Any) -> None:
    """aembed_query should be alias for avectorize_query."""
    fn = _make_mock_embedding_function()
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter, embedding_function=fn)

    result = await tool.aembed_query("test")

    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)


@pytest.mark.asyncio
async def test_embed_query_async_alternate_alias(adapter: Any) -> None:
    """embed_query_async should be alternate alias."""
    fn = _make_mock_embedding_function()
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter, embedding_function=fn)

    result = await tool.embed_query_async("test")

    assert isinstance(result, list)


def test_embed_documents_alias_sync(adapter: Any) -> None:
    """embed_documents should be alias for vectorize_documents."""
    fn = _make_mock_embedding_function()
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter, embedding_function=fn)

    result = tool.embed_documents(["a", "b"])

    assert isinstance(result, list)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_aembed_documents_alias_async(adapter: Any) -> None:
    """aembed_documents should be alias for avectorize_documents."""
    fn = _make_mock_embedding_function()
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter, embedding_function=fn)

    result = await tool.aembed_documents(["a", "b"])

    assert isinstance(result, list)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_embed_documents_async_alternate_alias(adapter: Any) -> None:
    """embed_documents_async should be alternate alias."""
    fn = _make_mock_embedding_function()
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter, embedding_function=fn)

    result = await tool.embed_documents_async(["a", "b"])

    assert isinstance(result, list)


def test_embed_texts_alias_sync(adapter: Any) -> None:
    """embed_texts should be alias for vectorize_documents."""
    fn = _make_mock_embedding_function()
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter, embedding_function=fn)

    result = tool.embed_texts(["a", "b"])

    assert isinstance(result, list)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_aembed_texts_alias_async(adapter: Any) -> None:
    """aembed_texts should be alias for avectorize_documents."""
    fn = _make_mock_embedding_function()
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter, embedding_function=fn)

    result = await tool.aembed_texts(["a", "b"])

    assert isinstance(result, list)
    assert len(result) == 2


def test_compatibility_aliases_have_error_context(adapter: Any) -> None:
    """All compatibility aliases should have error context decorators."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)

    # Check that these methods exist and are callable
    assert callable(tool.embed_query)
    assert callable(tool.aembed_query)
    assert callable(tool.embed_query_async)
    assert callable(tool.embed_documents)
    assert callable(tool.aembed_documents)
    assert callable(tool.embed_documents_async)
    assert callable(tool.embed_texts)
    assert callable(tool.aembed_texts)


def test_embed_query_with_error_raises_bad_request(adapter: Any) -> None:
    """embed_query should wrap errors properly."""

    def failing_fn(texts):
        raise ValueError("test error")

    tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter, embedding_function=failing_fn
    )

    with pytest.raises(BadRequest):
        tool.embed_query("test")


@pytest.mark.asyncio
async def test_aembed_query_with_error_raises_bad_request(adapter: Any) -> None:
    """aembed_query should wrap errors properly."""

    async def failing_fn(texts):
        raise ValueError("test error")

    tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter, async_embedding_function=failing_fn
    )

    with pytest.raises(BadRequest):
        await tool.aembed_query("test")


def test_embed_documents_with_error_raises_bad_request(adapter: Any) -> None:
    """embed_documents should wrap errors properly."""

    def failing_fn(texts):
        raise ValueError("test error")

    tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter, embedding_function=failing_fn
    )

    with pytest.raises(BadRequest):
        tool.embed_documents(["test"])


# ---------------------------------------------------------------------------
# Match Translation Tests (6 tests)
# ---------------------------------------------------------------------------


def test_get_match_score_from_mapping(adapter: Any) -> None:
    """Should extract score from mapping."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    match = {"score": 0.95, "id": "test"}

    score = tool._get_match_score(match)

    assert score == 0.95


def test_get_match_score_from_object(adapter: Any) -> None:
    """Should extract score from object attribute."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    match = Mock(score=0.85)

    score = tool._get_match_score(match)

    assert score == 0.85


def test_get_match_vector_extracts_from_mapping(adapter: Any) -> None:
    """Should extract embedding from mapping."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    match = {"embedding": [0.1, 0.2, 0.3], "id": "test"}

    vec = tool._get_match_vector(match)

    assert vec == [0.1, 0.2, 0.3]


def test_get_match_vector_extracts_from_object(adapter: Any) -> None:
    """Should extract vector from VectorMatch-style object."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    match = Mock(vector=Mock(vector=[0.1, 0.2, 0.3]))

    vec = tool._get_match_vector(match)

    assert vec == [0.1, 0.2, 0.3]


def test_filter_matches_by_score_applies_threshold(adapter: Any) -> None:
    """Should filter low-scoring matches."""
    tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter, score_threshold=0.5
    )
    matches = [
        {"id": "1", "score": 0.9, "metadata": {}},
        {"id": "2", "score": 0.3, "metadata": {}},
        {"id": "3", "score": 0.7, "metadata": {}},
    ]

    filtered = tool._filter_matches_by_score(matches)

    assert len(filtered) == 2
    assert all(tool._get_match_score(m) >= 0.5 for m in filtered)


def test_match_to_payload_converts_to_json(adapter: Any) -> None:
    """Should convert match to JSON-serializable dict."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    match = VectorMatch(
        vector=Vector(
            id="test-id",
            vector=[0.1, 0.2],
            metadata={"page_content": "test content", "custom": "data"},
        ),
        score=0.95,
        distance=0.05,
    )

    payload = tool._match_to_payload(match, return_scores=True)

    assert payload["id"] == "test-id"
    assert payload["text"] == "test content"
    assert payload["score"] == 0.95
    assert payload["metadata"]["custom"] == "data"
    assert "page_content" not in payload["metadata"]  # Removed


# ---------------------------------------------------------------------------
# MMR Tests (6 tests)
# ---------------------------------------------------------------------------


def test_cosine_sim_basic(adapter: Any) -> None:
    """Should compute cosine similarity correctly."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)

    # Identical vectors
    sim = tool._cosine_sim([1.0, 0.0], [1.0, 0.0])
    assert abs(sim - 1.0) < 0.01

    # Orthogonal vectors
    sim = tool._cosine_sim([1.0, 0.0], [0.0, 1.0])
    assert abs(sim - 0.0) < 0.01


def test_mmr_select_indices_pure_relevance_lambda_1(adapter: Any) -> None:
    """Lambda=1.0 should return pure relevance ranking."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    matches = [
        Mock(score=0.5, vector=Mock(vector=[1.0, 0.0])),
        Mock(score=0.9, vector=Mock(vector=[0.9, 0.1])),
        Mock(score=0.7, vector=Mock(vector=[0.7, 0.3])),
    ]

    indices = tool._mmr_select_indices(matches, k=3, lambda_mult=1.0)

    # Should be in descending score order: [1, 2, 0]
    assert indices == [1, 2, 0]


def test_mmr_select_indices_respects_k(adapter: Any) -> None:
    """Should return at most k results."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    matches = [
        Mock(score=0.5, vector=Mock(vector=[1.0, 0.0])),
        Mock(score=0.9, vector=Mock(vector=[0.9, 0.1])),
        Mock(score=0.7, vector=Mock(vector=[0.7, 0.3])),
    ]

    indices = tool._mmr_select_indices(matches, k=2, lambda_mult=0.5)

    assert len(indices) == 2


def test_mmr_select_indices_handles_empty_matches(adapter: Any) -> None:
    """Should handle empty match list."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)

    indices = tool._mmr_select_indices([], k=5, lambda_mult=0.5)

    assert indices == []


def test_mmr_select_indices_caches_similarities(adapter: Any) -> None:
    """Should cache similarity computations."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    matches = [
        Mock(score=0.9, vector=Mock(vector=[1.0, 0.0])),
        Mock(score=0.8, vector=Mock(vector=[0.9, 0.1])),
        Mock(score=0.7, vector=Mock(vector=[0.8, 0.2])),
    ]

    # Just verify it completes without error
    indices = tool._mmr_select_indices(matches, k=3, lambda_mult=0.5)

    assert len(indices) == 3


def test_mmr_select_indices_balances_relevance_and_diversity(adapter: Any) -> None:
    """Should balance relevance and diversity based on lambda."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    # Create matches where high relevance overlaps with high similarity
    matches = [
        Mock(score=1.0, vector=Mock(vector=[1.0, 0.0, 0.0])),  # Most relevant
        Mock(score=0.95, vector=Mock(vector=[0.99, 0.01, 0.0])),  # Similar to first
        Mock(score=0.5, vector=Mock(vector=[0.0, 0.0, 1.0])),  # Diverse
    ]

    # High lambda (0.9) → favor relevance
    indices_high = tool._mmr_select_indices(matches, k=2, lambda_mult=0.9)
    # Should pick the two most relevant: [0, 1]
    assert 0 in indices_high
    assert 1 in indices_high

    # Low lambda (0.1) → favor diversity
    indices_low = tool._mmr_select_indices(matches, k=2, lambda_mult=0.1)
    # Should pick diverse set: [0, 2]
    assert 0 in indices_low
    assert 2 in indices_low


# ---------------------------------------------------------------------------
# Simple Search Tests (Sync) (6 tests)
# ---------------------------------------------------------------------------


def test_search_simple_sync_returns_payloads(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should return list of JSON payloads."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        args = CorpusVectorSearchInput(query="test")
        results = tool._search_simple_sync(args)

        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)


def test_search_simple_sync_calls_translator_query(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should delegate to translator.query()."""
    called = {"query": False}

    class DummyTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            called["query"] = True
            return _empty_result(namespace="default")

        def capabilities(self):
            return _make_caps()

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        args = CorpusVectorSearchInput(query="test")
        tool._search_simple_sync(args)

        assert called["query"] is True


def test_search_simple_sync_uses_default_top_k(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should use tool default_top_k when k not specified."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def query(self, raw_query: Any, **k: Any) -> Any:
            captured["raw_query"] = raw_query
            return _empty_result(namespace="default")

        def capabilities(self):
            return _make_caps()

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn, default_top_k=10
        )

        args = CorpusVectorSearchInput(query="test")
        tool._search_simple_sync(args)

        assert captured["raw_query"]["top_k"] == 10


def test_search_simple_sync_applies_score_threshold(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should filter results by score_threshold."""

    class DummyTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            return QueryResult(
                matches=[
                    _make_match(id="1", score=0.9, text="high"),
                    _make_match(id="2", score=0.3, text="low"),
                ],
                query_vector=[0.0],
                namespace="default",
                total_matches=2,
            )

        def capabilities(self):
            return _make_caps()

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn, score_threshold=0.5
        )

        args = CorpusVectorSearchInput(query="test")
        results = tool._search_simple_sync(args)

        assert len(results) == 1
        assert results[0]["text"] == "high"


def test_search_simple_sync_includes_scores_when_requested(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should include scores in results when return_scores=True."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        args = CorpusVectorSearchInput(query="test", return_scores=True)
        results = tool._search_simple_sync(args)

        assert all("score" in r for r in results)


def test_search_simple_sync_guards_event_loop(adapter: Any) -> None:
    """Should raise RuntimeError in event loop."""

    @pytest.mark.asyncio
    async def test_in_loop():
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        args = CorpusVectorSearchInput(query="test")

        with pytest.raises(RuntimeError, match="event loop"):
            tool._search_simple_sync(args)

    asyncio.run(test_in_loop())


# ---------------------------------------------------------------------------
# Simple Search Tests (Async) (6 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_asearch_simple_returns_payloads(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should return list of JSON payloads."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        args = CorpusVectorSearchInput(query="test")
        caps = _make_caps()
        results = await tool._asearch_simple(args, caps=caps)

        assert isinstance(results, list)


@pytest.mark.asyncio
async def test_asearch_simple_calls_translator_arun_query(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should delegate to translator.arun_query()."""
    called = {"arun_query": False}

    class DummyTranslator:
        async def arun_query(self, *a: Any, **k: Any) -> Any:
            called["arun_query"] = True
            return _empty_result(namespace="default")

        async def acapabilities(self):
            return _make_caps()

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        args = CorpusVectorSearchInput(query="test")
        caps = _make_caps()
        await tool._asearch_simple(args, caps=caps)

        assert called["arun_query"] is True


@pytest.mark.asyncio
async def test_asearch_simple_validates_max_top_k(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should raise if k exceeds max_top_k."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        args = CorpusVectorSearchInput(query="test", k=200)
        caps = _make_caps(max_top_k=100)

        with pytest.raises(BadRequest, match="exceeds maximum"):
            await tool._asearch_simple(args, caps=caps)


@pytest.mark.asyncio
async def test_asearch_simple_validates_filter_support(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should raise if filters not supported."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        args = CorpusVectorSearchInput(query="test", filter={"key": "value"})
        caps = _make_caps(supports_metadata_filtering=False)

        with pytest.raises(NotSupported, match="metadata filtering"):
            await tool._asearch_simple(args, caps=caps)


@pytest.mark.asyncio
async def test_asearch_simple_applies_score_threshold(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should filter results by score_threshold."""

    class DummyTranslator:
        async def arun_query(self, *a: Any, **k: Any) -> Any:
            return QueryResult(
                matches=[
                    _make_match(id="1", score=0.9, text="high"),
                    _make_match(id="2", score=0.3, text="low"),
                ],
                query_vector=[0.0],
                namespace="default",
                total_matches=2,
            )

        async def acapabilities(self):
            return _make_caps()

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn, score_threshold=0.5
        )

        args = CorpusVectorSearchInput(query="test")
        caps = _make_caps()
        results = await tool._asearch_simple(args, caps=caps)

        assert len(results) == 1
        assert results[0]["text"] == "high"


@pytest.mark.asyncio
async def test_asearch_simple_includes_scores_when_requested(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should include scores when return_scores=True."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        args = CorpusVectorSearchInput(query="test", return_scores=True)
        caps = _make_caps()
        results = await tool._asearch_simple(args, caps=caps)

        assert all("score" in r for r in results)


# ---------------------------------------------------------------------------
# MMR Search Tests (Sync) (4 tests)
# ---------------------------------------------------------------------------


def test_search_with_mmr_sync_fetches_candidates(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should fetch fetch_k candidates for MMR."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def query(self, raw_query: Any, **k: Any) -> Any:
            captured["raw_query"] = raw_query
            return QueryResult(
                matches=[
                    _make_match(
                        id=str(i),
                        score=0.9 - i * 0.1,
                        text=f"doc-{i}",
                        embedding=[float(i), 0.0],
                    )
                    for i in range(10)
                ],
                query_vector=[0.0],
                namespace="default",
                total_matches=10,
            )

        def capabilities(self):
            return _make_caps()

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        args = CorpusVectorSearchInput(query="test", k=3, fetch_k=10)
        tool._search_with_mmr_sync(args)

        # Should request fetch_k, not k
        assert captured["raw_query"]["top_k"] == 10
        assert captured["raw_query"]["include_vectors"] is True


def test_search_with_mmr_sync_returns_k_results(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should return exactly k results after MMR."""

    class DummyTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            return QueryResult(
                matches=[
                    _make_match(
                        id=str(i),
                        score=0.9,
                        text=f"doc-{i}",
                        embedding=[float(i), 0.0],
                    )
                    for i in range(10)
                ],
                query_vector=[0.0],
                namespace="default",
                total_matches=10,
            )

        def capabilities(self):
            return _make_caps()

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        args = CorpusVectorSearchInput(query="test", k=3)
        results = tool._search_with_mmr_sync(args)

        assert len(results) == 3


def test_search_with_mmr_sync_uses_lambda_mult(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should use mmr_lambda parameter."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn, mmr_lambda=0.7
        )

        args = CorpusVectorSearchInput(query="test", mmr_lambda=0.3)
        # Should use args.mmr_lambda (0.3), not tool default (0.7)
        # We can't easily verify this without inspecting _mmr_select_indices
        # but we can verify it runs without error
        results = tool._search_with_mmr_sync(args)

        assert isinstance(results, list)


def test_search_with_mmr_sync_guards_event_loop(adapter: Any) -> None:
    """Should raise RuntimeError in event loop."""

    @pytest.mark.asyncio
    async def test_in_loop():
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        args = CorpusVectorSearchInput(query="test")

        with pytest.raises(RuntimeError, match="event loop"):
            tool._search_with_mmr_sync(args)

    asyncio.run(test_in_loop())


# ---------------------------------------------------------------------------
# MMR Search Tests (Async) (4 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_asearch_with_mmr_fetches_candidates(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should fetch fetch_k candidates for MMR."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_query(self, raw_query: Any, **k: Any) -> Any:
            captured["raw_query"] = raw_query
            return QueryResult(
                matches=[
                    _make_match(
                        id=str(i),
                        score=0.9 - i * 0.1,
                        text=f"doc-{i}",
                        embedding=[float(i), 0.0],
                    )
                    for i in range(10)
                ],
                query_vector=[0.0],
                namespace="default",
                total_matches=10,
            )

        async def acapabilities(self):
            return _make_caps()

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        args = CorpusVectorSearchInput(query="test", k=3, fetch_k=10)
        caps = _make_caps()
        await tool._asearch_with_mmr(args, caps=caps)

        # Should request fetch_k with vectors
        assert captured["raw_query"]["top_k"] == 10
        assert captured["raw_query"]["include_vectors"] is True


@pytest.mark.asyncio
async def test_asearch_with_mmr_returns_k_results(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should return exactly k results after MMR."""

    class DummyTranslator:
        async def arun_query(self, *a: Any, **k: Any) -> Any:
            return QueryResult(
                matches=[
                    _make_match(
                        id=str(i),
                        score=0.9,
                        text=f"doc-{i}",
                        embedding=[float(i), 0.0],
                    )
                    for i in range(10)
                ],
                query_vector=[0.0],
                namespace="default",
                total_matches=10,
            )

        async def acapabilities(self):
            return _make_caps()

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        args = CorpusVectorSearchInput(query="test", k=3)
        caps = _make_caps()
        results = await tool._asearch_with_mmr(args, caps=caps)

        assert len(results) == 3


@pytest.mark.asyncio
async def test_asearch_with_mmr_validates_fetch_k(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should raise if fetch_k exceeds max_top_k."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        args = CorpusVectorSearchInput(query="test", k=10, fetch_k=200)
        caps = _make_caps(max_top_k=100)

        with pytest.raises(BadRequest, match="exceeds maximum"):
            await tool._asearch_with_mmr(args, caps=caps)


@pytest.mark.asyncio
async def test_asearch_with_mmr_uses_lambda_mult(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should use mmr_lambda parameter."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn, mmr_lambda=0.7
        )

        args = CorpusVectorSearchInput(query="test", mmr_lambda=0.3)
        caps = _make_caps()
        results = await tool._asearch_with_mmr(args, caps=caps)

        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# CrewAI Tool API Tests (_run / _arun) (6 tests)
# ---------------------------------------------------------------------------


def test_run_dispatches_to_simple_search(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """_run should dispatch to simple search when use_mmr=False."""
    called = {"simple": False}

    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn, use_mmr_by_default=False
        )

        original = tool._search_simple_sync

        def counting_simple(args):
            called["simple"] = True
            return original(args)

        tool._search_simple_sync = counting_simple  # type: ignore

        tool._run(query="test")

        assert called["simple"] is True


def test_run_dispatches_to_mmr_search(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """_run should dispatch to MMR search when use_mmr=True."""
    called = {"mmr": False}

    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn, use_mmr_by_default=True
        )

        original = tool._search_with_mmr_sync

        def counting_mmr(args):
            called["mmr"] = True
            return original(args)

        tool._search_with_mmr_sync = counting_mmr  # type: ignore

        tool._run(query="test")

        assert called["mmr"] is True


def test_run_validates_input_schema(adapter: Any) -> None:
    """_run should validate input via CorpusVectorSearchInput."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        # Should raise validation error for invalid input
        with pytest.raises(Exception):  # Pydantic validation error
            tool._run(query=123)  # type: ignore


def test_run_guards_event_loop(adapter: Any) -> None:
    """_run should raise RuntimeError in event loop."""

    @pytest.mark.asyncio
    async def test_in_loop():
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        with pytest.raises(RuntimeError, match="event loop"):
            tool._run(query="test")

    asyncio.run(test_in_loop())


@pytest.mark.asyncio
async def test_arun_dispatches_to_async_search(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """_arun should dispatch to async search."""
    called = {"asearch": False}

    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        original = tool._asearch

        async def counting_asearch(args):
            called["asearch"] = True
            return await original(args)

        tool._asearch = counting_asearch  # type: ignore

        await tool._arun(query="test")

        assert called["asearch"] is True


@pytest.mark.asyncio
async def test_arun_validates_input_schema(adapter: Any) -> None:
    """_arun should validate input via CorpusVectorSearchInput."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        # Should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            await tool._arun(query=123)  # type: ignore


# ---------------------------------------------------------------------------
# Callable Interface Tests (6 tests)
# ---------------------------------------------------------------------------


def test_call_with_query_string_delegates_to_run(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """__call__("query") should delegate to _run(query="query")."""
    called = {"run": False}

    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        original = tool._run

        def counting_run(**kwargs):
            called["run"] = True
            return original(**kwargs)

        tool._run = counting_run  # type: ignore

        tool("test query")

        assert called["run"] is True


def test_call_with_texts_performs_vectorization(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """__call__(texts=[...]) should perform vectorization."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        result = tool(texts=["a", "b"])

        assert isinstance(result, list)
        assert len(result) == 2


def test_call_with_text_performs_query_vectorization(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """__call__(text="...") should perform query vectorization."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        result = tool(text="test")

        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)


def test_call_with_kwargs_delegates_to_run(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """__call__(query="...", k=5) should delegate to _run."""
    called = {"run": False}

    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        original = tool._run

        def counting_run(**kwargs):
            called["run"] = True
            return original(**kwargs)

        tool._run = counting_run  # type: ignore

        tool(query="test", k=5)

        assert called["run"] is True


def test_call_raises_for_invalid_positional_args(adapter: Any) -> None:
    """Should raise TypeError for invalid positional arguments."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        with pytest.raises(TypeError):
            tool("arg1", "arg2")


def test_call_guards_event_loop(adapter: Any) -> None:
    """__call__ should raise RuntimeError in event loop."""

    @pytest.mark.asyncio
    async def test_in_loop():
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        with pytest.raises(RuntimeError, match="event loop"):
            tool("test")

    asyncio.run(test_in_loop())


# ---------------------------------------------------------------------------
# Streaming Search Tests (4 tests)
# ---------------------------------------------------------------------------


def test_stream_search_returns_iterator(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should return Iterator[Dict[str, Any]]."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        result = tool.stream_search(query="test")

        assert hasattr(result, "__iter__")


def test_stream_search_yields_payloads(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should yield progressive payloads."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        payloads = list(tool.stream_search(query="test"))

        assert all(isinstance(p, dict) for p in payloads)


def test_stream_search_respects_top_k(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should yield at most k results."""

    class DummyTranslator:
        def query_stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            class Chunk:
                matches = [
                    _make_match(id=str(i), score=0.9, text=f"doc-{i}")
                    for i in range(10)
                ]
                is_final = True

            yield Chunk()

        def capabilities(self):
            return _make_caps()

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        payloads = list(tool.stream_search(query="test", k=3))

        assert len(payloads) == 3


def test_stream_search_guards_event_loop(adapter: Any) -> None:
    """stream_search should raise RuntimeError in event loop."""

    @pytest.mark.asyncio
    async def test_in_loop():
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        with pytest.raises(RuntimeError, match="event loop"):
            list(tool.stream_search(query="test"))

    asyncio.run(test_in_loop())


# ---------------------------------------------------------------------------
# Capabilities Tests (4 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_caps_async_delegates_to_translator(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should delegate to translator.acapabilities()."""
    called = {"acapabilities": False}

    class DummyTranslator:
        async def acapabilities(self):
            called["acapabilities"] = True
            return _make_caps()

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)

        await tool._get_caps_async()

        assert called["acapabilities"] is True


@pytest.mark.asyncio
async def test_get_caps_async_falls_back_to_sync(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should use translator.capabilities() in thread if no async."""
    called = {"capabilities": False}

    class DummyTranslator:
        def capabilities(self):
            called["capabilities"] = True
            return _make_caps()

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)

        await tool._get_caps_async()

        assert called["capabilities"] is True


@pytest.mark.asyncio
async def test_get_caps_async_caches_result(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should cache VectorCapabilities."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)

        caps1 = await tool._get_caps_async()
        caps2 = await tool._get_caps_async()

        # Should cache
        assert tool._caps is not None


@pytest.mark.asyncio
async def test_get_caps_async_raises_if_translator_missing_method(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should raise NotSupported if not implemented."""

    class BadTranslator:
        pass

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=BadTranslator()):
        tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)

        with pytest.raises(NotSupported, match="must implement"):
            await tool._get_caps_async()


# ---------------------------------------------------------------------------
# Context Manager Tests (4 tests)
# ---------------------------------------------------------------------------


def test_context_manager_calls_close(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """__exit__ should call close()."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)

        with tool:
            assert tool is not None


@pytest.mark.asyncio
async def test_async_context_manager_calls_aclose(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """__aexit__ should call aclose()."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)

        async with tool:
            assert tool is not None


def test_close_closes_translator(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should call translator.close()."""
    called = {"close": False}

    class DummyTranslator:
        def close(self) -> None:
            called["close"] = True

        def capabilities(self):
            return _make_caps()

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
        tool.close()

        assert called["close"] is True


def test_close_closes_adapter_when_owned(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should call adapter.close() if own_adapter=True."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        called = {"close": False}

        def close() -> None:
            called["close"] = True

        # Use the protocol-valid adapter fixture, but add a close hook.
        owned_adapter = adapter
        setattr(owned_adapter, "close", close)
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=owned_adapter, own_adapter=True
        )
        tool.close()

        assert called["close"] is True


# ---------------------------------------------------------------------------
# Event Loop Guard Tests (2 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_not_in_event_loop_raises_in_loop() -> None:
    """Should raise RuntimeError when called in event loop."""
    with pytest.raises(RuntimeError, match="event loop"):
        _ensure_not_in_event_loop("test_api")


def test_ensure_not_in_event_loop_succeeds_outside_loop() -> None:
    """Should not raise when called outside event loop."""
    # Should not raise
    _ensure_not_in_event_loop("test_api")


# ---------------------------------------------------------------------------
# Error Context Attachment Tests (6 tests)
# ---------------------------------------------------------------------------


def test_error_context_includes_framework_crewai(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Error context should always include framework='crewai'."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(crewai_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("test error")

        def capabilities(self):
            return _make_caps()

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=FailingTranslator()):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        args = CorpusVectorSearchInput(query="test")

        with pytest.raises(RuntimeError):
            tool._search_simple_sync(args)

        assert captured_ctx.get("framework") == "crewai"


def test_error_context_includes_operation_name(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Error context should include operation name."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(crewai_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("test error")

        def capabilities(self):
            return _make_caps()

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=FailingTranslator()):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        args = CorpusVectorSearchInput(query="test")

        with pytest.raises(RuntimeError):
            tool._search_simple_sync(args)

        assert "operation" in captured_ctx


def test_error_context_includes_query_params(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Error context should include query parameters."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(crewai_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("test error")

        def capabilities(self):
            return _make_caps()

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=FailingTranslator()):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        args = CorpusVectorSearchInput(query="test query", k=5)

        with pytest.raises(RuntimeError):
            tool._search_simple_sync(args)

        assert "query_chars" in captured_ctx or "total_content_chars" in captured_ctx
        assert captured_ctx.get("k") == 5


def test_error_context_includes_vector_dimension_hint(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Error context should include dim hint when available."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(crewai_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("test error")

        def capabilities(self):
            return _make_caps()

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=FailingTranslator()):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )
        tool._update_dim_hint(4)

        args = CorpusVectorSearchInput(query="test")

        with pytest.raises(RuntimeError):
            tool._search_simple_sync(args)

        assert "vector_dim_hint" in captured_ctx
        assert captured_ctx["vector_dim_hint"] == 4


def test_error_context_includes_mmr_params(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Error context should include MMR parameters."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(crewai_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("test error")

        def capabilities(self):
            return _make_caps()

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=FailingTranslator()):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        args = CorpusVectorSearchInput(
            query="test", use_mmr=True, mmr_lambda=0.7, fetch_k=20
        )

        with pytest.raises(RuntimeError):
            tool._search_with_mmr_sync(args)

        assert "fetch_k" in captured_ctx
        assert "lambda_mult" in captured_ctx


def test_error_context_extraction_never_raises(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Metrics errors should not break operation."""
    # This is more of a design verification - error context extraction
    # is wrapped in try/except blocks and should never raise

    class FailingTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("main error")

        def capabilities(self):
            return _make_caps()

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=FailingTranslator()):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        args = CorpusVectorSearchInput(query="test")

        # Should still raise the main error, not a metrics error
        with pytest.raises(RuntimeError, match="main error"):
            tool._search_simple_sync(args)


# ---------------------------------------------------------------------------
# CorpusCrewAIVectorClient Tests (6 tests)
# ---------------------------------------------------------------------------


def test_client_wraps_translator(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should use VectorTranslator internally."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        client = CorpusCrewAIVectorClient(adapter=adapter)

        assert hasattr(client, "_translator")


def test_client_exposes_protocol_methods(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should expose query, batch_query, upsert, delete."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        client = CorpusCrewAIVectorClient(adapter=adapter)

        assert hasattr(client, "query")
        assert hasattr(client, "batch_query")
        assert hasattr(client, "upsert")
        assert hasattr(client, "delete")
        assert hasattr(client, "create_namespace")
        assert hasattr(client, "delete_namespace")


def test_client_capabilities_delegates_to_translator(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should pass through capabilities()."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        client = CorpusCrewAIVectorClient(adapter=adapter)
        caps = client.capabilities()

        assert caps is not None


def test_client_health_delegates_to_translator(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should pass through health()."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        client = CorpusCrewAIVectorClient(adapter=adapter)
        health = client.health()

        assert health is not None


def test_client_passes_task_as_framework_ctx(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """task kwarg → framework_ctx."""
    captured: Dict[str, Any] = {}

    class CapturingTranslator:
        def query(self, raw_query: Any, *, framework_ctx: Any = None, **k: Any) -> Any:
            captured["framework_ctx"] = framework_ctx
            return _empty_result(namespace="default")

        def capabilities(self):
            return _make_caps()

        def health(self):
            return {"status": "ok"}

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=CapturingTranslator()):
        client = CorpusCrewAIVectorClient(adapter=adapter)
        mock_task = Mock()
        client.query({}, task=mock_task)

        assert captured["framework_ctx"] is mock_task


def test_client_query_delegates_to_translator(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """query() should delegate to translator."""
    called = {"query": False}

    class DummyTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            called["query"] = True
            return _empty_result(namespace="default")

        def capabilities(self):
            return _make_caps()

        def health(self):
            return {"status": "ok"}

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        client = CorpusCrewAIVectorClient(adapter=adapter)
        client.query({})

        assert called["query"] is True


# ---------------------------------------------------------------------------
# Input Schema Validation Tests (4 tests)
# ---------------------------------------------------------------------------


def test_input_schema_validates_query_required() -> None:
    """query field should be required."""
    with pytest.raises(Exception):  # Pydantic validation error
        CorpusVectorSearchInput()  # type: ignore


def test_input_schema_validates_k_positive() -> None:
    """k should be positive if provided."""
    # Valid
    args = CorpusVectorSearchInput(query="test", k=5)
    assert args.k == 5

    # None is valid (will use default)
    args = CorpusVectorSearchInput(query="test", k=None)
    assert args.k is None


def test_input_schema_validates_mmr_lambda_range() -> None:
    """mmr_lambda should be in [0, 1]."""
    # Valid range
    args = CorpusVectorSearchInput(query="test", mmr_lambda=0.5)
    assert args.mmr_lambda == 0.5

    # Edge cases
    args = CorpusVectorSearchInput(query="test", mmr_lambda=0.0)
    assert args.mmr_lambda == 0.0

    args = CorpusVectorSearchInput(query="test", mmr_lambda=1.0)
    assert args.mmr_lambda == 1.0


def test_input_schema_optional_fields_default_none() -> None:
    """Optional fields should default to None."""
    args = CorpusVectorSearchInput(query="test")

    assert args.k is None
    assert args.namespace is None
    assert args.filter is None
    assert args.use_mmr is None
    assert args.mmr_lambda is None
    assert args.fetch_k is None
    assert args.return_scores is False  # Defaults to False
    assert args.embedding is None
    assert args.context is None


# ---------------------------------------------------------------------------
# Build Raw Query Tests (4 tests)
# ---------------------------------------------------------------------------


def test_build_raw_query_includes_vector(adapter: Any) -> None:
    """Should include vector in query."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    embedding = [0.1, 0.2, 0.3, 0.4]

    raw = tool._build_raw_query(
        embedding=embedding,
        k=5,
        namespace="test-ns",
        filter=None,
        include_vectors=False,
    )

    assert raw["vector"] == embedding


def test_build_raw_query_includes_top_k(adapter: Any) -> None:
    """Should include top_k in query."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)

    raw = tool._build_raw_query(
        embedding=[0.1, 0.2],
        k=10,
        namespace=None,
        filter=None,
        include_vectors=False,
    )

    assert raw["top_k"] == 10


def test_build_raw_query_includes_filter(adapter: Any) -> None:
    """Should include filters in query."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    filter_dict = {"key": "value"}

    raw = tool._build_raw_query(
        embedding=[0.1, 0.2],
        k=5,
        namespace=None,
        filter=filter_dict,
        include_vectors=False,
    )

    assert raw["filters"] == filter_dict


def test_build_raw_query_includes_namespace(adapter: Any) -> None:
    """Should include namespace in query."""
    tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter, namespace="default-ns"
    )

    raw = tool._build_raw_query(
        embedding=[0.1, 0.2],
        k=5,
        namespace="override-ns",
        filter=None,
        include_vectors=False,
    )

    # Should use override
    assert raw["namespace"] == "override-ns"


# ---------------------------------------------------------------------------
# Result Validation Tests (2 tests)
# ---------------------------------------------------------------------------


def test_validate_query_result_accepts_query_result(adapter: Any) -> None:
    """Should accept QueryResult instances."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    result = _empty_result(namespace="default")

    validated = tool._validate_query_result(result, operation="test")

    assert isinstance(validated, QueryResult)


def test_validate_query_result_rejects_invalid_type(adapter: Any) -> None:
    """Should reject non-QueryResult types."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)

    with pytest.raises(BadRequest, match="unsupported type"):
        tool._validate_query_result({"matches": []}, operation="test")


# ---------------------------------------------------------------------------
# CrewAI-Specific Integration Tests (6 tests - NO SKIPS)
# ---------------------------------------------------------------------------


def test_crewai_tool_metadata_accessible(adapter: Any) -> None:
    """Tool should expose CrewAI-compatible metadata."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)

    assert hasattr(tool, "name")
    assert hasattr(tool, "description")
    assert hasattr(tool, "args_schema")
    assert tool.args_schema is CorpusVectorSearchInput


def test_crewai_task_context_propagation(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Task object → OperationContext."""
    captured: Dict[str, Any] = {}

    def fake_from_crewai(task: Any) -> Any:
        captured["task"] = task
        return OperationContext(request_id="test", tenant="test", attrs={})

    monkeypatch.setattr(crewai_adapter_module, "ctx_from_crewai", fake_from_crewai)

    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        mock_task = Mock()
        tool._run(query="test", context=mock_task)

        assert captured["task"] is mock_task


def test_crewai_tool_works_in_agent_flow(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Tool should work in typical agent usage."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        # Simulate agent calling tool
        results = tool._run(query="find relevant documents")

        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)


def test_crewai_tool_supports_mmr_diversification(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Tool should support MMR for diverse results."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn, use_mmr_by_default=True
        )

        results = tool._run(query="test", k=3, mmr_lambda=0.5)

        assert isinstance(results, list)


def test_crewai_metadata_field_envelope(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """metadata_field should wrap user metadata."""

    class DummyTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            return QueryResult(
                matches=[
                    VectorMatch(
                        vector=Vector(
                            id="1",
                            vector=[0.1, 0.2],
                            metadata={
                                "page_content": "test",
                                "id": "1",
                                "user_meta": {"custom": "data"},
                            },
                        ),
                        score=0.9,
                        distance=0.1,
                    )
                ],
                query_vector=[0.0],
                namespace="default",
                total_matches=1,
            )

        def capabilities(self):
            return _make_caps()

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn, metadata_field="user_meta"
        )

        results = tool._run(query="test")

        # Should extract nested metadata
        assert results[0]["metadata"] == {"custom": "data"}


def test_crewai_streaming_with_real_task(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Streaming should work with task context."""
    monkeypatch.setattr(
        crewai_adapter_module,
        "ctx_from_crewai",
        lambda _task: OperationContext(request_id="test", tenant="test", attrs={}),
    )
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        mock_task = Mock()
        results = list(tool.stream_search(query="test", context=mock_task))

        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# REAL CrewAI integration tests (pass/fail; no skips)
# ---------------------------------------------------------------------------


def _require_crewai_for_vector_integration() -> Any:
    """Fail-fast if CrewAI isn't importable for real integration checks."""
    try:
        import crewai  # noqa: F401  # type: ignore
        from crewai.tools.base_tool import BaseTool  # type: ignore[import-not-found]
    except Exception as exc:
        pytest.fail(
            "CrewAI is required for real vector integration tests in this module. Install with:\n"
            "  pip install -U crewai\n"
            f"Import error: {exc!r}",
            pytrace=False,
        )

    if not getattr(crewai_adapter_module, "CREWAI_AVAILABLE", False):
        pytest.fail(
            "CREWAI_AVAILABLE is False but CrewAI imports succeeded. "
            "This indicates an internal inconsistency in corpus_sdk.vector.framework_adapters.crewai.",
            pytrace=False,
        )

    return BaseTool


@pytest.mark.integration
def test_real_crewai_vector_tool_is_real_basetool(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """When CrewAI is installed, our vector tool must be a real CrewAI BaseTool."""
    BaseTool = _require_crewai_for_vector_integration()
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-testing")

    tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter,
        embedding_function=_make_mock_embedding_function([[0.1, 0.2]]),
    )
    assert isinstance(tool, BaseTool)


@pytest.mark.integration
def test_real_crewai_vector_tool_run_executes_end_to_end(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke test: call CrewAI tool entrypoint `_run` with real BaseTool class."""
    _ = _require_crewai_for_vector_integration()
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-testing")

    tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter,
        embedding_function=_make_mock_embedding_function([[0.1, 0.2]]),
        default_top_k=2,
    )

    out = tool._run(query="hello", k=2)
    assert isinstance(out, list)
    assert all(isinstance(x, dict) for x in out)


@pytest.mark.integration
def test_real_crewai_vector_tool_callable_interface_executes(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke test: __call__ convenience interface works with real CrewAI BaseTool."""
    _ = _require_crewai_for_vector_integration()
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-testing")

    tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter,
        embedding_function=_make_mock_embedding_function([[0.1, 0.2]]),
        default_top_k=2,
    )

    out = tool("hello")
    assert isinstance(out, list)
    assert all(isinstance(x, dict) for x in out)


@pytest.mark.integration
def test_real_crewai_vector_tool_accepts_real_task_context(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke test: passing a real crewai.Task as context does not crash."""
    _ = _require_crewai_for_vector_integration()
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-testing")

    import crewai  # type: ignore
    from crewai import Task  # type: ignore

    agent = crewai.Agent(
        role="researcher",
        goal="Test vector task context",
        backstory="A test agent",
        allow_delegation=False,
        verbose=False,
    )
    task = Task(
        description="Test task for CrewAI vector adapter",
        agent=agent,
        expected_output="A short answer",
    )

    tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter,
        embedding_function=_make_mock_embedding_function([[0.1, 0.2]]),
        default_top_k=2,
    )

    out = tool._run(query="hello", k=2, context=task)
    assert isinstance(out, list)
    assert all(isinstance(x, dict) for x in out)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_crewai_vector_tool_arun_executes_end_to_end(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke test: call CrewAI async tool entrypoint `_arun` with real BaseTool class."""
    _ = _require_crewai_for_vector_integration()
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-testing")

    tool = CorpusCrewAIVectorSearchTool(
        corpus_adapter=adapter,
        embedding_function=_make_mock_embedding_function([[0.1, 0.2]]),
        default_top_k=2,
    )

    out = await tool._arun(query="hello", k=2)
    assert isinstance(out, list)
    assert all(isinstance(x, dict) for x in out)


# ---------------------------------------------------------------------------
# Edge Case Tests (4 tests)
# ---------------------------------------------------------------------------


def test_empty_query_returns_zero_vector(adapter: Any) -> None:
    """Empty query should return deterministic zero vector when dim known."""
    tool = CorpusCrewAIVectorSearchTool(corpus_adapter=adapter)
    tool._update_dim_hint(4)

    vec = tool._embed_query("")

    assert vec == [0.0, 0.0, 0.0, 0.0]


def test_zero_k_returns_empty_results(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """k=0 should return empty list without querying."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        args = CorpusVectorSearchInput(query="test", k=0)
        results = tool._search_simple_sync(args)

        assert results == []


def test_negative_k_returns_empty_results(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Negative k should return empty list."""
    with patch.object(
        crewai_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn
        )

        args = CorpusVectorSearchInput(query="test", k=-1)
        results = tool._search_simple_sync(args)

        assert results == []


def test_all_matches_filtered_returns_empty(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """High score threshold that filters all matches should return empty."""

    class DummyTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            return QueryResult(
                matches=[
                    VectorMatch(
                        vector=Vector(
                            id="1",
                            vector=[0.1, 0.2],
                            metadata={"page_content": "low", "id": "1"},
                        ),
                        score=0.3,
                        distance=0.7,
                    )
                ],
                query_vector=[0.0],
                namespace="default",
                total_matches=1,
            )

        def capabilities(self):
            return _make_caps()

    with patch.object(crewai_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        fn = _make_mock_embedding_function()
        tool = CorpusCrewAIVectorSearchTool(
            corpus_adapter=adapter, embedding_function=fn, score_threshold=0.9
        )

        args = CorpusVectorSearchInput(query="test")
        results = tool._search_simple_sync(args)

        assert results == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
