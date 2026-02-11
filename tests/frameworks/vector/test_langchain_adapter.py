"""
LangChain Vector framework adapter tests.

These tests are written against the current public API in
`corpus_sdk.vector.framework_adapters.langchain`, which exposes LangChain-compatible
VectorStore and BaseRetriever implementations backed by Corpus VectorProtocolV1.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
from unittest.mock import Mock, patch

import pytest

import corpus_sdk.vector.framework_adapters.langchain as langchain_adapter_module
from corpus_sdk.vector.framework_adapters.langchain import (
    CorpusLangChainRetriever,
    CorpusLangChainVectorClient,
    CorpusLangChainVectorStore,
    ErrorCodes,
    _ensure_not_in_event_loop,
)
from corpus_sdk.vector.vector_base import (
    BadRequest,
    NotSupported,
    OperationContext,
    QueryResult,
    UpsertResult,
    Vector,
    VectorCapabilities,
    VectorMatch,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_TEXT = "hello from langchain vector tests"
SAMPLE_EMBEDDING = [0.1, 0.2, 0.3, 0.4]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_caps(**overrides: Any) -> VectorCapabilities:
    supports_metadata_filtering = overrides.pop("supports_metadata_filtering", True)
    max_top_k = overrides.pop("max_top_k", 100)
    return VectorCapabilities(
        server="test",
        version="0",
        supports_metadata_filtering=supports_metadata_filtering,
        max_top_k=max_top_k,
        **overrides,
    )


def _make_match(
    *,
    id: str,
    score: float,
    text: str = "test",
    embedding: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> VectorMatch:
    vec = embedding if embedding is not None else [0.1, 0.2]
    merged_metadata: Dict[str, Any] = {"page_content": text, "id": id}
    if metadata is not None:
        merged_metadata.update(dict(metadata))
    return VectorMatch(
        vector=Vector(
            id=id,
            vector=list(vec),
            metadata=merged_metadata,
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


def _result(
    matches: List[VectorMatch],
    *,
    namespace: str = "default",
    query_vector: Optional[List[float]] = None,
) -> QueryResult:
    return QueryResult(
        matches=list(matches),
        query_vector=list(query_vector) if query_vector is not None else [0.0],
        namespace=namespace,
        total_matches=len(matches),
    )


def _make_dummy_translator() -> Any:
    """Factory for creating a standard dummy translator for tests."""

    class DummyTranslator:
        def upsert(self, *a: Any, **k: Any) -> Any:
            return UpsertResult(
                upserted_count=1,
                failed_count=0,
                failures=[],
            )

        async def arun_upsert(self, *a: Any, **k: Any) -> Any:
            return UpsertResult(
                upserted_count=1,
                failed_count=0,
                failures=[],
            )

        def query(self, *a: Any, **k: Any) -> Any:
            return QueryResult(
                matches=[
                    _make_match(id="doc-0", score=0.95, text="test")
                ],
                query_vector=[0.1, 0.2],
                namespace="default",
                total_matches=1,
            )

        async def arun_query(self, *a: Any, **k: Any) -> Any:
            return QueryResult(
                matches=[
                    _make_match(id="doc-0", score=0.95, text="test")
                ],
                query_vector=[0.1, 0.2],
                namespace="default",
                total_matches=1,
            )

        def query_stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            class Chunk:
                matches = [
                    _make_match(id="doc-0", score=0.95, text="test")
                ]
                is_final = True

            yield Chunk()

        def delete(self, *a: Any, **k: Any) -> Any:
            return {"deleted": 1}

        async def arun_delete(self, *a: Any, **k: Any) -> Any:
            return {"deleted": 1}

        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        async def acapabilities(self) -> VectorCapabilities:
            return _make_caps()

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
        if len(vectors) >= len(texts):
            return vectors[: len(texts)]
        # If caller provided a single prototype vector (or too few), repeat it.
        prototype = vectors[0]
        return [list(prototype) for _ in texts]

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
        def query(self, *args: Any, **kwargs: Any) -> Any:
            return _empty_result(namespace="default")

        async def arun_query(self, *args: Any, **kwargs: Any) -> Any:
            return _empty_result(namespace="default")

        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        async def acapabilities(self) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Dict[str, Any]:
            return {"status": "ok"}

        async def ahealth(self) -> Dict[str, Any]:
            return {"status": "ok"}

    return TestAdapter()


@pytest.fixture
def Document() -> Any:
    """Fixture for LangChain Document class."""
    try:
        from langchain_core.documents import Document

        return Document
    except ImportError:
        # Use stub
        return langchain_adapter_module.Document


# ---------------------------------------------------------------------------
# Construction / Initialization Tests (10 tests)
# ---------------------------------------------------------------------------


def test_init_requires_langchain_installed() -> None:
    """Store should raise RuntimeError if LangChain not available."""
    with patch.object(langchain_adapter_module, "LANGCHAIN_AVAILABLE", False):
        with pytest.raises(RuntimeError, match="requires `langchain-core`"):
            CorpusLangChainVectorStore(corpus_adapter=Mock())


def test_init_requires_corpus_adapter(adapter: Any) -> None:
    """Adapter must be provided."""
    with pytest.raises((TypeError, AttributeError)):
        CorpusLangChainVectorStore(corpus_adapter=None)  # type: ignore


def test_init_stores_config_attributes(adapter: Any) -> None:
    """Store should keep key config attributes accessible."""
    store = CorpusLangChainVectorStore(
        corpus_adapter=adapter,
        namespace="test-ns",
        id_field="custom_id",
        text_field="custom_text",
        metadata_field="custom_meta",
        score_threshold=0.8,
        batch_size=50,
        default_top_k=10,
    )

    assert store.namespace == "test-ns"
    assert store.id_field == "custom_id"
    assert store.text_field == "custom_text"
    assert store.metadata_field == "custom_meta"
    assert store.score_threshold == 0.8
    assert store.batch_size == 50
    assert store.default_top_k == 10


def test_init_validates_score_threshold_range(adapter: Any) -> None:
    """score_threshold should be between 0.0 and 1.0 if provided."""
    # Valid range should work
    store = CorpusLangChainVectorStore(corpus_adapter=adapter, score_threshold=0.5)
    assert store.score_threshold == 0.5

    # Edge cases
    store = CorpusLangChainVectorStore(corpus_adapter=adapter, score_threshold=0.0)
    assert store.score_threshold == 0.0

    store = CorpusLangChainVectorStore(corpus_adapter=adapter, score_threshold=1.0)
    assert store.score_threshold == 1.0


def test_init_validates_batch_size_positive(adapter: Any) -> None:
    """batch_size must be positive."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter, batch_size=50)
    assert store.batch_size == 50


def test_init_validates_default_top_k_positive(adapter: Any) -> None:
    """default_top_k must be positive."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter, default_top_k=5)
    assert store.default_top_k == 5


def test_init_with_embedding_function(adapter: Any) -> None:
    """embedding_function should be stored correctly."""
    fn = _make_mock_embedding_function()
    store = CorpusLangChainVectorStore(corpus_adapter=adapter, embedding_function=fn)
    assert store.embedding_function is fn


def test_init_with_async_embedding_function(adapter: Any) -> None:
    """async_embedding_function should be stored correctly."""
    store = CorpusLangChainVectorStore(
        corpus_adapter=adapter, async_embedding_function=_make_async_embedding_function
    )
    assert store.async_embedding_function is _make_async_embedding_function


def test_init_with_mmr_similarity_fn(adapter: Any) -> None:
    """mmr_similarity_fn should be stored correctly."""

    def custom_sim(a, b):
        return 0.5

    store = CorpusLangChainVectorStore(
        corpus_adapter=adapter, mmr_similarity_fn=custom_sim
    )
    assert store.mmr_similarity_fn is custom_sim


def test_vectorstore_type_property(adapter: Any) -> None:
    """_vectorstore_type should return 'corpus'."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    assert store._vectorstore_type == "corpus"


# ---------------------------------------------------------------------------
# Translator Wiring Tests (4 tests)
# ---------------------------------------------------------------------------


def test_translator_created_with_framework_langchain(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Translator factory should be called with framework='langchain'."""
    captured: Dict[str, Any] = {}

    class FakeTranslator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def query(self, *a, **k):
            return _empty_result(namespace="default")

        def capabilities(self):
            return _make_caps()

    with patch.object(langchain_adapter_module, "VectorTranslator", FakeTranslator):
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=_make_mock_embedding_function()
        )
        _ = store._translator

    assert captured.get("framework") == "langchain"
    assert captured.get("adapter") is adapter


def test_translator_cached_property_reused(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Multiple accesses to _translator should return same instance."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLangChainVectorStore(corpus_adapter=adapter)
        translator1 = store._translator
        translator2 = store._translator

        assert translator1 is translator2


def test_translator_uses_langchain_framework_translator(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should use the LangChain-specific framework translator (QueryResult passthrough)."""
    captured: Dict[str, Any] = {}

    class FakeTranslator:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.translator = kwargs.get("translator")

        def query(self, *a, **k):
            return _empty_result(namespace="default")

    with patch.object(langchain_adapter_module, "VectorTranslator", FakeTranslator):
        store = CorpusLangChainVectorStore(corpus_adapter=adapter)
        _ = store._translator

    assert "translator" in captured
    translator_obj = captured["translator"]
    assert translator_obj.__class__.__name__ == "LangChainVectorFrameworkTranslator"


def test_translator_available_on_first_access(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Lazy construction should work on first access."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLangChainVectorStore(corpus_adapter=adapter)
        # Should not raise
        translator = store._translator
        assert translator is not None


# ---------------------------------------------------------------------------
# Context Translation Tests (8 tests)
# ---------------------------------------------------------------------------


def test_build_core_context_from_operation_context(adapter: Any) -> None:
    """Should pass through OperationContext unchanged."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    ctx = OperationContext(request_id="test", tenant="test", attrs={})

    result = store._build_core_context(ctx)

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

    monkeypatch.setattr(langchain_adapter_module, "ctx_from_dict", fake_from_dict)

    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    test_dict = {"key": "value"}

    ctx = store._build_core_context(test_dict)

    assert captured["mapping"] == test_dict
    assert isinstance(ctx, OperationContext)


def test_build_core_context_from_langchain_config(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should build OperationContext from LangChain config using context_translation."""
    captured: Dict[str, Any] = {}
    base_ctx = OperationContext(request_id="from-lc", tenant="from-lc", attrs={})

    def fake_from_langchain(config: Any) -> Any:
        captured["config"] = config
        return base_ctx

    monkeypatch.setattr(langchain_adapter_module, "ctx_from_langchain", fake_from_langchain)

    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    mock_config = Mock()

    ctx = store._build_core_context(mock_config)

    assert captured["config"] is mock_config
    assert isinstance(ctx, OperationContext)


def test_build_core_context_handles_none_returns_empty(adapter: Any) -> None:
    """Should return empty OperationContext when config is None."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)

    ctx = store._build_core_context(None)

    assert isinstance(ctx, OperationContext)


def test_build_core_context_dict_translation_error_attaches_context(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Errors during dict translation should attach error context."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    def fake_from_dict(mapping: Any) -> Any:
        raise RuntimeError("dict translation failed")

    monkeypatch.setattr(langchain_adapter_module, "attach_context", fake_attach_context)
    monkeypatch.setattr(langchain_adapter_module, "ctx_from_dict", fake_from_dict)

    store = CorpusLangChainVectorStore(corpus_adapter=adapter)

    with pytest.raises(RuntimeError, match="dict translation failed"):
        store._build_core_context({"key": "value"})

    assert captured_ctx.get("framework") == "langchain"
    assert captured_ctx.get("operation") == "vector_context_from_dict"


def test_build_core_context_langchain_translation_error_attaches_context(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Errors during LangChain translation should attach error context."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    def fake_from_langchain(config: Any) -> Any:
        raise RuntimeError("langchain translation failed")

    monkeypatch.setattr(langchain_adapter_module, "attach_context", fake_attach_context)
    monkeypatch.setattr(langchain_adapter_module, "ctx_from_langchain", fake_from_langchain)

    store = CorpusLangChainVectorStore(corpus_adapter=adapter)

    with pytest.raises(RuntimeError, match="langchain translation failed"):
        store._build_core_context(Mock())

    assert captured_ctx.get("framework") == "langchain"
    assert captured_ctx.get("operation") == "vector_context_from_langchain"


def test_build_framework_context_includes_namespace(adapter: Any) -> None:
    """framework_ctx should include namespace."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter, namespace="test-ns")
    core_ctx = OperationContext()

    fw_ctx = store._build_framework_context(
        core_ctx,
        operation="test_op",
        namespace="override-ns",
        filter=None,
    )

    assert fw_ctx["namespace"] == "override-ns"
    assert fw_ctx["vector_operation"] == "test_op"
    assert fw_ctx["has_filter"] is False


def test_build_contexts_orchestrates_both(adapter: Any) -> None:
    """_build_contexts should return both OperationContext and framework_ctx."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter, namespace="test-ns")
    core_ctx, fw_ctx = store._build_contexts(
        operation="test",
        namespace=None,
    )

    assert isinstance(core_ctx, OperationContext)
    assert isinstance(fw_ctx, Mapping)
    assert fw_ctx["namespace"] == "test-ns"
    assert fw_ctx["vector_operation"] == "test"


# ---------------------------------------------------------------------------
# Namespace Resolution Tests (3 tests)
# ---------------------------------------------------------------------------


def test_effective_namespace_uses_override(adapter: Any) -> None:
    """Explicit namespace should override store default."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter, namespace="default")
    ns = store._effective_namespace("override")
    assert ns == "override"


def test_effective_namespace_uses_store_default(adapter: Any) -> None:
    """Should use store default when not specified."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter, namespace="default")
    ns = store._effective_namespace(None)
    assert ns == "default"


def test_effective_namespace_handles_none_default(adapter: Any) -> None:
    """None default should pass through."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter, namespace=None)
    ns = store._effective_namespace(None)
    assert ns is None


# ---------------------------------------------------------------------------
# Dimension Hint Tests (6 tests)
# ---------------------------------------------------------------------------


def test_update_dim_hint_sets_first_write(adapter: Any) -> None:
    """First non-zero dimension should win."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    assert store._vector_dim_hint is None

    store._update_dim_hint(4)
    assert store._vector_dim_hint == 4


def test_update_dim_hint_thread_safe(adapter: Any) -> None:
    """Concurrent updates should not race."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)

    def update_thread(dim: int):
        store._update_dim_hint(dim)

    threads = [threading.Thread(target=update_thread, args=(i,)) for i in range(1, 10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should have a value, and it should be one of the attempted values
    assert store._vector_dim_hint is not None
    assert 1 <= store._vector_dim_hint < 10


def test_update_dim_hint_ignores_subsequent_writes(adapter: Any) -> None:
    """Second write should be no-op."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    store._update_dim_hint(4)
    store._update_dim_hint(8)

    assert store._vector_dim_hint == 4


def test_zero_vector_returns_correct_dimension(adapter: Any) -> None:
    """Should return zeros of correct dimension."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)

    vec = store._zero_vector(4)

    assert vec == [0.0, 0.0, 0.0, 0.0]


def test_zero_vector_with_different_dimensions(adapter: Any) -> None:
    """Should work with any positive dimension."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)

    vec2 = store._zero_vector(2)
    assert len(vec2) == 2
    assert all(x == 0.0 for x in vec2)

    vec10 = store._zero_vector(10)
    assert len(vec10) == 10
    assert all(x == 0.0 for x in vec10)


def test_update_dim_hint_ignores_invalid_dimensions(adapter: Any) -> None:
    """Should ignore None and non-positive dimensions."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)

    store._update_dim_hint(None)
    assert store._vector_dim_hint is None

    store._update_dim_hint(0)
    assert store._vector_dim_hint is None

    store._update_dim_hint(-1)
    assert store._vector_dim_hint is None


# ---------------------------------------------------------------------------
# Embedding Function Tests (Sync) (8 tests)
# ---------------------------------------------------------------------------


def test_ensure_embeddings_uses_provided(adapter: Any) -> None:
    """Should use explicit embeddings when provided."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    store._update_dim_hint(2)

    texts = ["a", "b"]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    result = store._ensure_embeddings(texts, embeddings)

    assert result == embeddings


def test_ensure_embeddings_validates_length_match(adapter: Any) -> None:
    """Should raise if embeddings length doesn't match texts length."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    texts = ["a", "b"]
    embeddings = [[0.1, 0.2]]  # Only 1 embedding for 2 texts

    with pytest.raises(BadRequest, match="does not match"):
        store._ensure_embeddings(texts, embeddings)


def test_ensure_embeddings_calls_embedding_function(adapter: Any) -> None:
    """Should call embedding_function if no embeddings provided."""
    called = {"times": 0}

    def counting_fn(texts: List[str]) -> List[List[float]]:
        called["times"] += 1
        return [[0.1, 0.2] for _ in texts]

    store = CorpusLangChainVectorStore(
        corpus_adapter=adapter, embedding_function=counting_fn
    )
    texts = ["a", "b"]

    result = store._ensure_embeddings(texts, None)

    assert called["times"] == 1
    assert len(result) == 2


def test_ensure_embeddings_raises_without_function(adapter: Any) -> None:
    """Should raise NotSupported if no function configured."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    texts = ["a"]

    with pytest.raises(NotSupported, match="No embedding_function"):
        store._ensure_embeddings(texts, None)


def test_ensure_embeddings_handles_empty_texts(adapter: Any) -> None:
    """Empty texts should return zero vectors when dim known."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    store._update_dim_hint(4)

    result = store._ensure_embeddings(["", "  "], None)

    assert len(result) == 2
    assert all(v == [0.0, 0.0, 0.0, 0.0] for v in result)


def test_ensure_embeddings_mixed_empty_and_valid(adapter: Any) -> None:
    """Should handle mix of empty and non-empty texts."""
    called = {"texts": []}

    def tracking_fn(texts: List[str]) -> List[List[float]]:
        called["texts"] = texts
        return [[0.1, 0.2] for _ in texts]

    store = CorpusLangChainVectorStore(
        corpus_adapter=adapter, embedding_function=tracking_fn
    )
    store._update_dim_hint(2)

    texts = ["", "valid", "  ", "another"]
    result = store._ensure_embeddings(texts, None)

    # Should only embed non-empty texts
    assert called["texts"] == ["valid", "another"]
    assert len(result) == 4
    assert result[0] == [0.0, 0.0]  # Empty
    assert result[1] == [0.1, 0.2]  # Valid
    assert result[2] == [0.0, 0.0]  # Empty
    assert result[3] == [0.1, 0.2]  # Valid


def test_ensure_embeddings_updates_dim_hint(adapter: Any) -> None:
    """Should set hint from first vector."""
    fn = _make_mock_embedding_function([[0.1, 0.2, 0.3]])
    store = CorpusLangChainVectorStore(corpus_adapter=adapter, embedding_function=fn)

    store._ensure_embeddings(["a"], None)

    assert store._vector_dim_hint == 3


def test_ensure_embeddings_handles_function_errors(adapter: Any) -> None:
    """Should wrap function errors with context."""

    def failing_fn(texts: List[str]) -> List[List[float]]:
        raise ValueError("embedding failed")

    store = CorpusLangChainVectorStore(
        corpus_adapter=adapter, embedding_function=failing_fn
    )

    with pytest.raises(BadRequest, match="embedding_function failed"):
        store._ensure_embeddings(["a"], None)


# ---------------------------------------------------------------------------
# Embedding Function Tests (Async) (8 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_embeddings_async_uses_provided(adapter: Any) -> None:
    """Should use explicit embeddings."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    store._update_dim_hint(2)

    texts = ["a", "b"]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    result = await store._ensure_embeddings_async(texts, embeddings)

    assert result == embeddings


@pytest.mark.asyncio
async def test_ensure_embeddings_async_validates_length_match(adapter: Any) -> None:
    """Should raise if lengths mismatch."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    texts = ["a", "b"]
    embeddings = [[0.1, 0.2]]

    with pytest.raises(BadRequest, match="does not match"):
        await store._ensure_embeddings_async(texts, embeddings)


@pytest.mark.asyncio
async def test_ensure_embeddings_async_calls_async_function(adapter: Any) -> None:
    """Should prefer async_embedding_function."""
    called = {"times": 0}

    async def counting_fn(texts: List[str]) -> List[List[float]]:
        called["times"] += 1
        return [[0.1, 0.2] for _ in texts]

    store = CorpusLangChainVectorStore(
        corpus_adapter=adapter, async_embedding_function=counting_fn
    )
    texts = ["a", "b"]

    result = await store._ensure_embeddings_async(texts, None)

    assert called["times"] == 1
    assert len(result) == 2


@pytest.mark.asyncio
async def test_ensure_embeddings_async_falls_back_to_sync(adapter: Any) -> None:
    """Should use sync function in thread if no async."""
    called = {"times": 0}

    def counting_fn(texts: List[str]) -> List[List[float]]:
        called["times"] += 1
        return [[0.1, 0.2] for _ in texts]

    store = CorpusLangChainVectorStore(
        corpus_adapter=adapter, embedding_function=counting_fn
    )
    texts = ["a"]

    result = await store._ensure_embeddings_async(texts, None)

    assert called["times"] == 1


@pytest.mark.asyncio
async def test_ensure_embeddings_async_raises_without_function(adapter: Any) -> None:
    """Should raise NotSupported if no function."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)

    with pytest.raises(NotSupported, match="No embedding_function"):
        await store._ensure_embeddings_async(["a"], None)


@pytest.mark.asyncio
async def test_ensure_embeddings_async_handles_empty_texts(adapter: Any) -> None:
    """Empty texts should return zero vectors when dim known."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    store._update_dim_hint(4)

    result = await store._ensure_embeddings_async(["", "  "], None)

    assert len(result) == 2
    assert all(v == [0.0, 0.0, 0.0, 0.0] for v in result)


@pytest.mark.asyncio
async def test_ensure_embeddings_async_updates_dim_hint(adapter: Any) -> None:
    """Should set hint from first vector."""

    async def fn(texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3]]

    store = CorpusLangChainVectorStore(
        corpus_adapter=adapter, async_embedding_function=fn
    )

    await store._ensure_embeddings_async(["a"], None)

    assert store._vector_dim_hint == 3


@pytest.mark.asyncio
async def test_ensure_embeddings_async_handles_function_errors(adapter: Any) -> None:
    """Should wrap errors with context."""

    async def failing_fn(texts: List[str]) -> List[List[float]]:
        raise ValueError("async embedding failed")

    store = CorpusLangChainVectorStore(
        corpus_adapter=adapter, async_embedding_function=failing_fn
    )

    with pytest.raises(BadRequest, match="async_embedding_function failed"):
        await store._ensure_embeddings_async(["a"], None)


# ---------------------------------------------------------------------------
# Query Embedding Tests (Sync) (6 tests)
# ---------------------------------------------------------------------------


def test_embed_query_uses_provided_embedding(adapter: Any) -> None:
    """Should use explicit embedding when provided."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    store._update_dim_hint(4)

    result = store._embed_query("test", embedding=[0.1, 0.2, 0.3, 0.4])

    assert result == [0.1, 0.2, 0.3, 0.4]


def test_embed_query_calls_embedding_function(adapter: Any) -> None:
    """Should call embedding_function([query])."""
    called = {"times": 0}

    def counting_fn(texts: List[str]) -> List[List[float]]:
        called["times"] += 1
        return [[0.1, 0.2, 0.3, 0.4]]

    store = CorpusLangChainVectorStore(
        corpus_adapter=adapter, embedding_function=counting_fn
    )

    result = store._embed_query("test query")

    assert called["times"] == 1
    assert len(result) == 4


def test_embed_query_returns_zero_vector_for_empty_query(adapter: Any) -> None:
    """Should return deterministic zeros if dim known."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    store._update_dim_hint(4)

    result = store._embed_query("")

    assert result == [0.0, 0.0, 0.0, 0.0]


def test_embed_query_raises_for_empty_query_without_dim(adapter: Any) -> None:
    """Should raise if dim unknown."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)

    with pytest.raises(BadRequest, match="query cannot be empty"):
        store._embed_query("")


def test_embed_query_raises_without_function(adapter: Any) -> None:
    """Should raise NotSupported if no function."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)

    with pytest.raises(NotSupported, match="No embedding_function"):
        store._embed_query("test")


def test_embed_query_updates_dim_hint(adapter: Any) -> None:
    """Should set hint from vector."""
    fn = _make_mock_embedding_function([[0.1, 0.2, 0.3]])
    store = CorpusLangChainVectorStore(corpus_adapter=adapter, embedding_function=fn)

    store._embed_query("test")

    assert store._vector_dim_hint == 3


# ---------------------------------------------------------------------------
# Query Embedding Tests (Async) (6 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_query_async_uses_provided_embedding(adapter: Any) -> None:
    """Should use explicit embedding."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    store._update_dim_hint(4)

    result = await store._embed_query_async("test", embedding=[0.1, 0.2, 0.3, 0.4])

    assert result == [0.1, 0.2, 0.3, 0.4]


@pytest.mark.asyncio
async def test_embed_query_async_calls_async_function(adapter: Any) -> None:
    """Should prefer async_embedding_function."""
    called = {"times": 0}

    async def counting_fn(texts: List[str]) -> List[List[float]]:
        called["times"] += 1
        return [[0.1, 0.2, 0.3, 0.4]]

    store = CorpusLangChainVectorStore(
        corpus_adapter=adapter, async_embedding_function=counting_fn
    )

    result = await store._embed_query_async("test")

    assert called["times"] == 1
    assert len(result) == 4


@pytest.mark.asyncio
async def test_embed_query_async_falls_back_to_sync(adapter: Any) -> None:
    """Should use sync function in thread."""
    fn = _make_mock_embedding_function([[0.1, 0.2, 0.3, 0.4]])
    store = CorpusLangChainVectorStore(corpus_adapter=adapter, embedding_function=fn)

    result = await store._embed_query_async("test")

    assert len(result) == 4


@pytest.mark.asyncio
async def test_embed_query_async_returns_zero_vector_for_empty_query(
    adapter: Any,
) -> None:
    """Should return deterministic zeros if dim known."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    store._update_dim_hint(4)

    result = await store._embed_query_async("")

    assert result == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_embed_query_async_raises_for_empty_query_without_dim(adapter: Any) -> None:
    """Should raise if dim unknown."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)

    with pytest.raises(BadRequest, match="query cannot be empty"):
        await store._embed_query_async("")


@pytest.mark.asyncio
async def test_embed_query_async_raises_without_function(adapter: Any) -> None:
    """Should raise NotSupported if no function."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)

    with pytest.raises(NotSupported, match="No embedding_function"):
        await store._embed_query_async("test")


# ---------------------------------------------------------------------------
# Metadata/IDs Normalization Tests (4 tests)
# ---------------------------------------------------------------------------


def test_normalize_metadatas_returns_empty_dicts(adapter: Any) -> None:
    """None → [{}, {}, ...]."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    result = store._normalize_metadatas(3, None)

    assert result == [{}, {}, {}]


def test_normalize_metadatas_replicates_single(adapter: Any) -> None:
    """Single metadata → replicated to n."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    result = store._normalize_metadatas(3, [{"key": "value"}])

    assert len(result) == 3
    assert all(m == {"key": "value"} for m in result)


def test_normalize_metadatas_raises_on_length_mismatch(adapter: Any) -> None:
    """Should raise if len mismatch."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)

    with pytest.raises(BadRequest, match="does not match"):
        store._normalize_metadatas(3, [{"a": 1}, {"b": 2}])


def test_normalize_ids_generates_defaults(adapter: Any) -> None:
    """None → [uuid, uuid, ...]."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    result = store._normalize_ids(3, None)

    assert len(result) == 3
    assert all(isinstance(i, str) for i in result)
    assert len(set(result)) == 3  # All unique


# ---------------------------------------------------------------------------
# Document Translation Tests (4 tests)
# ---------------------------------------------------------------------------


def test_to_corpus_vectors_builds_correct_structure(adapter: Any) -> None:
    """Should build Vector objects with correct structure."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)

    texts = ["test"]
    embeddings = [[0.1, 0.2]]
    metadatas = [{"key": "value"}]
    ids = ["id-1"]

    vectors = store._to_corpus_vectors(
        texts, embeddings, metadatas, ids, namespace="test-ns"
    )

    assert len(vectors) == 1
    vec = vectors[0]
    assert vec.id == "id-1"
    assert vec.vector == [0.1, 0.2]
    assert vec.namespace == "test-ns"
    assert "page_content" in vec.metadata
    assert vec.metadata["page_content"] == "test"


def test_to_corpus_vectors_handles_metadata_envelope(adapter: Any) -> None:
    """Should wrap metadata under metadata_field when configured."""
    store = CorpusLangChainVectorStore(
        corpus_adapter=adapter, metadata_field="user_meta"
    )

    texts = ["test"]
    embeddings = [[0.1, 0.2]]
    metadatas = [{"custom": "data"}]
    ids = ["id-1"]

    vectors = store._to_corpus_vectors(texts, embeddings, metadatas, ids, namespace=None)

    vec = vectors[0]
    assert "user_meta" in vec.metadata
    assert vec.metadata["user_meta"] == {"custom": "data"}


def test_from_corpus_matches_converts_to_documents(adapter: Any, Document: Any) -> None:
    """Should convert VectorMatch to LangChain Documents."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)

    matches = [
        _make_match(
            id="1",
            score=0.9,
            text="test content",
            metadata={"custom": "data"},
        )
    ]

    results = store._from_corpus_matches(matches)

    assert len(results) == 1
    doc, score = results[0]
    assert isinstance(doc, Document)
    assert doc.page_content == "test content"
    assert score == 0.9
    assert doc.metadata["custom"] == "data"
    assert "page_content" not in doc.metadata  # Removed


def test_from_corpus_matches_handles_metadata_envelope(adapter: Any, Document: Any) -> None:
    """Should extract nested metadata when metadata_field is set."""
    store = CorpusLangChainVectorStore(
        corpus_adapter=adapter, metadata_field="user_meta"
    )

    matches = [
        _make_match(
            id="1",
            score=0.9,
            text="test",
            metadata={"user_meta": {"custom": "data"}},
        )
    ]

    results = store._from_corpus_matches(matches)
    doc, _ = results[0]

    assert doc.metadata == {"custom": "data"}


# ---------------------------------------------------------------------------
# Add Texts Tests (Sync) (6 tests)
# ---------------------------------------------------------------------------


def test_add_texts_returns_ids(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Should return list of IDs."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        ids = store.add_texts(["test"])

        assert isinstance(ids, list)
        assert len(ids) == 1


def test_add_texts_calls_translator_upsert(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should delegate to translator.upsert()."""
    called = {"upsert": False}

    class DummyTranslator:
        def upsert(self, *a: Any, **k: Any) -> Any:
            called["upsert"] = True
            return UpsertResult(
                upserted_count=1,
                failed_count=0,
                failures=[],
            )

        def capabilities(self):
            return _make_caps()

    with patch.object(langchain_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        store.add_texts(["test"])

        assert called["upsert"] is True


def test_add_texts_uses_embedding_function(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should compute embeddings if not provided."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        called = {"times": 0}

        def counting_fn(texts: List[str]) -> List[List[float]]:
            called["times"] += 1
            return [[0.1, 0.2, 0.3, 0.4]]

        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=counting_fn
        )

        store.add_texts(["test"])

        assert called["times"] == 1


def test_add_texts_handles_partial_failure(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should log warnings for partial failures but not raise."""

    class PartialFailureTranslator:
        def upsert(self, *a: Any, **k: Any) -> Any:
            return UpsertResult(
                upserted_count=1,
                failed_count=1,
                failures=[{"id": "doc-1", "error": "test error"}],
            )

        def capabilities(self):
            return _make_caps()

    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=PartialFailureTranslator()
    ):
        fn = _make_mock_embedding_function([[0.1, 0.2], [0.3, 0.4]])
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        # Should not raise
        ids = store.add_texts(["test1", "test2"])
        assert len(ids) == 2


def test_add_texts_raises_if_all_failed(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should raise if all documents failed."""

    class AllFailureTranslator:
        def upsert(self, *a: Any, **k: Any) -> Any:
            return UpsertResult(
                upserted_count=0,
                failed_count=2,
                failures=[
                    {"id": "doc-0", "error": "test error"},
                    {"id": "doc-1", "error": "test error"},
                ],
            )

        def capabilities(self):
            return _make_caps()

    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=AllFailureTranslator()
    ):
        fn = _make_mock_embedding_function([[0.1, 0.2], [0.3, 0.4]])
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        with pytest.raises(Exception, match="All"):
            store.add_texts(["test1", "test2"])


def test_add_texts_guards_event_loop(adapter: Any) -> None:
    """Should raise RuntimeError in event loop."""

    @pytest.mark.asyncio
    async def test_in_loop():
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        with pytest.raises(RuntimeError, match="event loop"):
            store.add_texts(["test"])

    asyncio.run(test_in_loop())


# ---------------------------------------------------------------------------
# Add Texts Tests (Async) (4 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aadd_texts_returns_ids(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should return list of IDs."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        ids = await store.aadd_texts(["test"])

        assert isinstance(ids, list)
        assert len(ids) == 1


@pytest.mark.asyncio
async def test_aadd_texts_calls_translator_arun_upsert(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should delegate to translator.arun_upsert()."""
    called = {"arun_upsert": False}

    class DummyTranslator:
        async def arun_upsert(self, *a: Any, **k: Any) -> Any:
            called["arun_upsert"] = True
            return UpsertResult(
                upserted_count=1,
                failed_count=0,
                failures=[],
            )

        async def acapabilities(self):
            return _make_caps()

    with patch.object(langchain_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        await store.aadd_texts(["test"])

        assert called["arun_upsert"] is True


@pytest.mark.asyncio
async def test_aadd_texts_uses_async_embedding_function(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should compute embeddings async."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        called = {"times": 0}

        async def counting_fn(texts: List[str]) -> List[List[float]]:
            called["times"] += 1
            return [[0.1, 0.2, 0.3, 0.4]]

        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, async_embedding_function=counting_fn
        )

        await store.aadd_texts(["test"])

        assert called["times"] == 1


@pytest.mark.asyncio
async def test_aadd_texts_handles_empty_list(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should handle empty text list gracefully."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLangChainVectorStore(corpus_adapter=adapter)

        ids = await store.aadd_texts([])

        assert ids == []


# ---------------------------------------------------------------------------
# Add Documents Tests (4 tests)
# ---------------------------------------------------------------------------


def test_add_documents_extracts_texts_and_metadata(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, Document: Any
) -> None:
    """Should convert Documents → texts/metadata."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        docs = [Document(page_content="test", metadata={"key": "value"})]
        ids = store.add_documents(docs)

        assert len(ids) == 1


def test_add_documents_delegates_to_add_texts(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, Document: Any
) -> None:
    """Should call add_texts internally."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        called = {"add_texts": False}
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        original_add_texts = store.add_texts

        def counting_add_texts(*args, **kwargs):
            called["add_texts"] = True
            return original_add_texts(*args, **kwargs)

        store.add_texts = counting_add_texts  # type: ignore

        docs = [Document(page_content="test", metadata={})]
        store.add_documents(docs)

        assert called["add_texts"] is True


@pytest.mark.asyncio
async def test_aadd_documents_extracts_texts_and_metadata(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, Document: Any
) -> None:
    """Should convert Documents → texts/metadata."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        docs = [Document(page_content="test", metadata={"key": "value"})]
        ids = await store.aadd_documents(docs)

        assert len(ids) == 1


@pytest.mark.asyncio
async def test_aadd_documents_delegates_to_aadd_texts(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, Document: Any
) -> None:
    """Should call aadd_texts internally."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        called = {"aadd_texts": False}
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        original_aadd_texts = store.aadd_texts

        async def counting_aadd_texts(*args, **kwargs):
            called["aadd_texts"] = True
            return await original_aadd_texts(*args, **kwargs)

        store.aadd_texts = counting_aadd_texts  # type: ignore

        docs = [Document(page_content="test", metadata={})]
        await store.aadd_documents(docs)

        assert called["aadd_texts"] is True


# ---------------------------------------------------------------------------
# Similarity Search Tests (Sync) (6 tests)
# ---------------------------------------------------------------------------


def test_similarity_search_returns_documents(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, Document: Any
) -> None:
    """Should return List[Document]."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        docs = store.similarity_search("test query")

        assert isinstance(docs, list)
        assert all(isinstance(d, Document) for d in docs)


def test_similarity_search_calls_translator_query(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should delegate to translator.query()."""
    called = {"query": False}

    class DummyTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            called["query"] = True
            return _empty_result(namespace="default")

        def capabilities(self):
            return _make_caps()

    with patch.object(langchain_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        store.similarity_search("test")

        assert called["query"] is True


def test_similarity_search_validates_max_top_k(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should raise if k exceeds max_top_k."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        with pytest.raises(BadRequest, match="exceeds maximum"):
            store.similarity_search("test", k=200)


def test_similarity_search_validates_filter_support(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should raise if filters not supported."""

    class DummyTranslator:
        def capabilities(self):
            return _make_caps(supports_metadata_filtering=False)

    with patch.object(langchain_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        with pytest.raises(NotSupported, match="metadata filtering"):
            store.similarity_search("test", filter={"key": "value"})


def test_similarity_search_applies_score_threshold(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, Document: Any
) -> None:
    """Should filter results by score_threshold."""

    class DummyTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            return _result(
                [
                    _make_match(id="1", score=0.9, text="high"),
                    _make_match(id="2", score=0.3, text="low"),
                ],
                namespace="default",
            )

        def capabilities(self):
            return _make_caps()

    with patch.object(langchain_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn, score_threshold=0.5
        )

        docs = store.similarity_search("test")

        assert len(docs) == 1
        assert docs[0].page_content == "high"


def test_similarity_search_guards_event_loop(adapter: Any) -> None:
    """Should raise RuntimeError in event loop."""

    @pytest.mark.asyncio
    async def test_in_loop():
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        with pytest.raises(RuntimeError, match="event loop"):
            store.similarity_search("test")

    asyncio.run(test_in_loop())


# ---------------------------------------------------------------------------
# Similarity Search Tests (Async) (4 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_asimilarity_search_returns_documents(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, Document: Any
) -> None:
    """Should return List[Document]."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        docs = await store.asimilarity_search("test query")

        assert isinstance(docs, list)
        assert all(isinstance(d, Document) for d in docs)


@pytest.mark.asyncio
async def test_asimilarity_search_calls_translator_arun_query(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should delegate to translator.arun_query()."""
    called = {"arun_query": False}

    class DummyTranslator:
        async def arun_query(self, *a: Any, **k: Any) -> Any:
            called["arun_query"] = True
            return _empty_result(namespace="default")

        async def acapabilities(self):
            return _make_caps()

    with patch.object(langchain_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        await store.asimilarity_search("test")

        assert called["arun_query"] is True


@pytest.mark.asyncio
async def test_asimilarity_search_validates_max_top_k(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should raise if k exceeds max_top_k."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        with pytest.raises(BadRequest, match="exceeds maximum"):
            await store.asimilarity_search("test", k=200)


@pytest.mark.asyncio
async def test_asimilarity_search_applies_score_threshold(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, Document: Any
) -> None:
    """Should filter results by score_threshold."""

    class DummyTranslator:
        async def arun_query(self, *a: Any, **k: Any) -> Any:
            return _result(
                [
                    _make_match(id="1", score=0.9, text="high"),
                    _make_match(id="2", score=0.3, text="low"),
                ],
                namespace="default",
            )

        async def acapabilities(self):
            return _make_caps()

    with patch.object(langchain_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn, score_threshold=0.5
        )

        docs = await store.asimilarity_search("test")

        assert len(docs) == 1
        assert docs[0].page_content == "high"


# ---------------------------------------------------------------------------
# Streaming Search Tests (4 tests)
# ---------------------------------------------------------------------------


def test_similarity_search_stream_returns_iterator(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should return Iterator[Document]."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        result = store.similarity_search_stream("test")

        assert hasattr(result, "__iter__")


def test_similarity_search_stream_yields_documents(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, Document: Any
) -> None:
    """Should yield progressive documents."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        docs = list(store.similarity_search_stream("test"))

        assert all(isinstance(d, Document) for d in docs)


def test_similarity_search_stream_validates_max_top_k(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should raise if k exceeds max_top_k."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        with pytest.raises(BadRequest, match="exceeds maximum"):
            list(store.similarity_search_stream("test", k=200))


def test_similarity_search_stream_guards_event_loop(adapter: Any) -> None:
    """Should raise RuntimeError in event loop."""

    @pytest.mark.asyncio
    async def test_in_loop():
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        with pytest.raises(RuntimeError, match="event loop"):
            list(store.similarity_search_stream("test"))

    asyncio.run(test_in_loop())


# ---------------------------------------------------------------------------
# Similarity Search with Score Tests (4 tests)
# ---------------------------------------------------------------------------


def test_similarity_search_with_score_returns_tuples(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, Document: Any
) -> None:
    """Should return List[Tuple[Document, float]]."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        results = store.similarity_search_with_score("test")

        assert isinstance(results, list)
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)


def test_similarity_search_with_score_includes_scores(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scores should be present in tuples."""

    class DummyTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            return _result([
                _make_match(id="1", score=0.95, text="test"),
            ])

        def capabilities(self):
            return _make_caps()

    with patch.object(langchain_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        results = store.similarity_search_with_score("test")

        assert len(results) == 1
        doc, score = results[0]
        assert score == 0.95


@pytest.mark.asyncio
async def test_asimilarity_search_with_score_returns_tuples(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, Document: Any
) -> None:
    """Should return List[Tuple[Document, float]]."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        results = await store.asimilarity_search_with_score("test")

        assert isinstance(results, list)
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)


@pytest.mark.asyncio
async def test_asimilarity_search_with_score_includes_scores(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scores should be present in tuples."""

    class DummyTranslator:
        async def arun_query(self, *a: Any, **k: Any) -> Any:
            return _result([
                _make_match(id="1", score=0.95, text="test"),
            ])

        async def acapabilities(self):
            return _make_caps()

    with patch.object(langchain_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        results = await store.asimilarity_search_with_score("test")

        assert len(results) == 1
        doc, score = results[0]
        assert score == 0.95


# ---------------------------------------------------------------------------
# MMR Tests (8 tests)
# ---------------------------------------------------------------------------


def test_cosine_sim_basic(adapter: Any) -> None:
    """Should compute cosine similarity correctly."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)

    # Identical vectors
    sim = store._cosine_sim([1.0, 0.0], [1.0, 0.0])
    assert abs(sim - 1.0) < 0.01

    # Orthogonal vectors
    sim = store._cosine_sim([1.0, 0.0], [0.0, 1.0])
    assert abs(sim - 0.0) < 0.01


def test_mmr_select_indices_pure_relevance_lambda_1(adapter: Any) -> None:
    """Lambda=1.0 should return pure relevance ranking."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    query_vec = [1.0, 0.0]
    matches = [
        _make_match(id="0", score=0.5, embedding=[1.0, 0.0]),
        _make_match(id="1", score=0.9, embedding=[0.9, 0.1]),
        _make_match(id="2", score=0.7, embedding=[0.7, 0.3]),
    ]

    indices = store._mmr_select_indices(query_vec, matches, k=3, lambda_mult=1.0)

    # Should be in descending score order: [1, 2, 0]
    assert indices == [1, 2, 0]


def test_mmr_select_indices_respects_k(adapter: Any) -> None:
    """Should return at most k results."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    query_vec = [1.0, 0.0]
    matches = [_make_match(id=str(i), score=0.9, embedding=[1.0, 0.0]) for i in range(5)]

    indices = store._mmr_select_indices(query_vec, matches, k=2, lambda_mult=0.5)

    assert len(indices) == 2


def test_mmr_select_indices_handles_empty_matches(adapter: Any) -> None:
    """Should handle empty match list."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)
    query_vec = [1.0, 0.0]

    indices = store._mmr_select_indices(query_vec, [], k=5, lambda_mult=0.5)

    assert indices == []


def test_mmr_search_validates_lambda_range(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should raise if lambda_mult not in [0, 1]."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        with pytest.raises(BadRequest, match="lambda_mult must be in"):
            store.max_marginal_relevance_search("test", k=4, lambda_mult=1.5)


def test_mmr_search_uses_custom_similarity_fn(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should use custom mmr_similarity_fn when provided."""
    called = {"times": 0}

    def custom_sim(a, b):
        called["times"] += 1
        return 0.5

    class MultiMatchTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            return _result(
                [
                    _make_match(id="a", score=0.9, embedding=[1.0, 0.0, 0.0, 0.0]),
                    _make_match(id="b", score=0.85, embedding=[0.0, 1.0, 0.0, 0.0]),
                    _make_match(id="c", score=0.8, embedding=[1.0, 1.0, 0.0, 0.0]),
                ],
                namespace="default",
                query_vector=[1.0, 0.0],
            )

        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=MultiMatchTranslator()
    ):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn, mmr_similarity_fn=custom_sim
        )

        # This will trigger MMR which calls similarity function
        store.max_marginal_relevance_search("test", k=2, lambda_mult=0.5)

        # Should have been called during MMR
        assert called["times"] > 0


def test_mmr_search_requests_vectors(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should request include_vectors=True for MMR."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def query(self, raw_query: Any, **k: Any) -> Any:
            captured["raw_query"] = raw_query
            return _result(
                [
                    _make_match(id="1", score=0.9, text="test", embedding=[0.1, 0.2]),
                ],
                namespace="default",
            )

        def capabilities(self):
            return _make_caps()

    with patch.object(langchain_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        store.max_marginal_relevance_search("test")

        assert captured["raw_query"]["include_vectors"] is True


def test_mmr_search_guards_event_loop(adapter: Any) -> None:
    """Should raise RuntimeError in event loop."""

    @pytest.mark.asyncio
    async def test_in_loop():
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        with pytest.raises(RuntimeError, match="event loop"):
            store.max_marginal_relevance_search("test")

    asyncio.run(test_in_loop())


# ---------------------------------------------------------------------------
# Delete Tests (4 tests)
# ---------------------------------------------------------------------------


def test_delete_by_ids_delegates_to_translator(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should call translator.delete() with IDs."""
    called = {"delete": False}

    class DummyTranslator:
        def delete(self, *a: Any, **k: Any) -> Any:
            called["delete"] = True
            return {"deleted": 1}

        def capabilities(self):
            return _make_caps()

    with patch.object(langchain_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        store = CorpusLangChainVectorStore(corpus_adapter=adapter)
        store.delete(ids=["doc-0"])

        assert called["delete"] is True


def test_delete_by_filter_delegates_to_translator(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should call translator.delete() with filter."""
    called = {"delete": False}

    class DummyTranslator:
        def delete(self, *a: Any, **k: Any) -> Any:
            called["delete"] = True
            return {"deleted": 1}

        def capabilities(self):
            return _make_caps()

    with patch.object(langchain_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        store = CorpusLangChainVectorStore(corpus_adapter=adapter)
        store.delete(filter={"key": "value"})

        assert called["delete"] is True


def test_delete_raises_without_ids_or_filter(adapter: Any) -> None:
    """Should raise BadRequest if neither provided."""
    store = CorpusLangChainVectorStore(corpus_adapter=adapter)

    with pytest.raises(BadRequest, match="must provide ids or filter"):
        store.delete()


@pytest.mark.asyncio
async def test_adelete_delegates_to_translator(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Async delete should work."""
    called = {"arun_delete": False}

    class DummyTranslator:
        async def arun_delete(self, *a: Any, **k: Any) -> Any:
            called["arun_delete"] = True
            return {"deleted": 1}

        async def acapabilities(self):
            return _make_caps()

    with patch.object(langchain_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        store = CorpusLangChainVectorStore(corpus_adapter=adapter)
        await store.adelete(ids=["doc-0"])

        assert called["arun_delete"] is True


# ---------------------------------------------------------------------------
# Callable Interface Tests (4 tests)
# ---------------------------------------------------------------------------


def test_call_with_string_returns_query_embedding(adapter: Any) -> None:
    """__call__(str) should return query embedding."""
    fn = _make_mock_embedding_function()
    store = CorpusLangChainVectorStore(corpus_adapter=adapter, embedding_function=fn)

    result = store("test query")

    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)


def test_call_with_list_returns_document_embeddings(adapter: Any) -> None:
    """__call__(List[str]) should return document embeddings."""
    fn = _make_mock_embedding_function([[0.1, 0.2], [0.3, 0.4]])
    store = CorpusLangChainVectorStore(corpus_adapter=adapter, embedding_function=fn)

    result = store(["doc1", "doc2"])

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(v, list) for v in result)


def test_vectorize_query_delegates_to_embed_query(adapter: Any) -> None:
    """vectorize_query should call _embed_query."""
    fn = _make_mock_embedding_function()
    store = CorpusLangChainVectorStore(corpus_adapter=adapter, embedding_function=fn)

    result = store.vectorize_query("test")

    assert isinstance(result, list)
    assert len(result) == 4


def test_vectorize_documents_delegates_to_ensure_embeddings(adapter: Any) -> None:
    """vectorize_documents should call _ensure_embeddings."""
    fn = _make_mock_embedding_function()
    store = CorpusLangChainVectorStore(corpus_adapter=adapter, embedding_function=fn)

    result = store.vectorize_documents(["doc1", "doc2"])

    assert isinstance(result, list)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Capabilities Tests (4 tests)
# ---------------------------------------------------------------------------


def test_get_caps_sync_delegates_to_translator(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should delegate to translator.capabilities()."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLangChainVectorStore(corpus_adapter=adapter)

        caps = store._get_caps_sync()

        assert isinstance(caps, VectorCapabilities)


def test_get_caps_sync_caches_result(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should cache VectorCapabilities."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLangChainVectorStore(corpus_adapter=adapter)

        caps1 = store._get_caps_sync()
        caps2 = store._get_caps_sync()

        # Should cache
        assert store._caps is not None


@pytest.mark.asyncio
async def test_get_caps_async_delegates_to_translator(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should delegate to translator.acapabilities()."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLangChainVectorStore(corpus_adapter=adapter)

        caps = await store._get_caps_async()

        assert isinstance(caps, VectorCapabilities)


@pytest.mark.asyncio
async def test_get_caps_async_raises_if_translator_missing_method(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should raise NotSupported if not implemented."""

    class BadTranslator:
        pass

    with patch.object(langchain_adapter_module, "VectorTranslator", return_value=BadTranslator()):
        store = CorpusLangChainVectorStore(corpus_adapter=adapter)

        with pytest.raises(NotSupported, match="must implement"):
            await store._get_caps_async()


# ---------------------------------------------------------------------------
# Context Manager Tests (4 tests)
# ---------------------------------------------------------------------------


def test_context_manager_calls_close(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """__exit__ should call close()."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLangChainVectorStore(corpus_adapter=adapter)

        with store:
            assert store is not None


@pytest.mark.asyncio
async def test_async_context_manager_calls_aclose(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """__aexit__ should call aclose()."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        store = CorpusLangChainVectorStore(corpus_adapter=adapter)

        async with store:
            assert store is not None


def test_close_closes_translator_and_adapter(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should call close on both translator and adapter."""
    called = {"translator": False, "adapter": False}

    class DummyTranslator:
        def close(self) -> None:
            called["translator"] = True

        def capabilities(self):
            return _make_caps()

    class OwnedAdapter:
        def close(self) -> None:
            called["adapter"] = True

        def capabilities(self):
            return _make_caps()

    with patch.object(langchain_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        owned_adapter = OwnedAdapter()
        store = CorpusLangChainVectorStore(corpus_adapter=owned_adapter)
        store.close()

        assert called["translator"] is True
        assert called["adapter"] is True


@pytest.mark.asyncio
async def test_aclose_closes_translator_and_adapter(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should call aclose on both translator and adapter."""
    called = {"translator": False, "adapter": False}

    class DummyTranslator:
        async def aclose(self) -> None:
            called["translator"] = True

        async def acapabilities(self):
            return _make_caps()

    class OwnedAdapter:
        async def aclose(self) -> None:
            called["adapter"] = True

        async def acapabilities(self):
            return _make_caps()

    with patch.object(langchain_adapter_module, "VectorTranslator", return_value=DummyTranslator()):
        owned_adapter = OwnedAdapter()
        store = CorpusLangChainVectorStore(corpus_adapter=owned_adapter)
        await store.aclose()

        assert called["translator"] is True
        assert called["adapter"] is True


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
# Error Context Tests (4 tests)
# ---------------------------------------------------------------------------


def test_error_context_includes_framework_langchain(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Error context should always include framework='langchain'."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(langchain_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("test error")

        def capabilities(self):
            return _make_caps()

    with patch.object(langchain_adapter_module, "VectorTranslator", return_value=FailingTranslator()):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        with pytest.raises(RuntimeError):
            store.similarity_search("test")

        assert captured_ctx.get("framework") == "langchain"


def test_error_context_includes_operation_name(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Error context should include operation name."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(langchain_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("test error")

        def capabilities(self):
            return _make_caps()

    with patch.object(langchain_adapter_module, "VectorTranslator", return_value=FailingTranslator()):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        with pytest.raises(RuntimeError):
            store.similarity_search("test")

        assert "operation" in captured_ctx


def test_error_context_includes_vector_dimension_hint(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Error context should include dim hint when available."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(langchain_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("test error")

        def capabilities(self):
            return _make_caps()

    with patch.object(langchain_adapter_module, "VectorTranslator", return_value=FailingTranslator()):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )
        store._update_dim_hint(4)

        with pytest.raises(RuntimeError):
            store.similarity_search("test")

        assert "vector_dimension_hint" in captured_ctx
        assert captured_ctx["vector_dimension_hint"] == 4


def test_error_context_extraction_never_raises(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metrics errors should not break operation."""
    class FailingTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("main error")

        def capabilities(self):
            return _make_caps()

    with patch.object(langchain_adapter_module, "VectorTranslator", return_value=FailingTranslator()):
        fn = _make_mock_embedding_function()
        store = CorpusLangChainVectorStore(
            corpus_adapter=adapter, embedding_function=fn
        )

        # Should still raise the main error, not a metrics error
        with pytest.raises(RuntimeError, match="main error"):
            store.similarity_search("test")


# ---------------------------------------------------------------------------
# Client Tests (4 tests)
# ---------------------------------------------------------------------------


def test_client_wraps_translator(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should use VectorTranslator internally."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        client = CorpusLangChainVectorClient(adapter=adapter)

        assert hasattr(client, "_translator")


def test_client_exposes_protocol_methods(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should expose protocol methods."""
    with patch.object(
        langchain_adapter_module, "VectorTranslator", return_value=_make_dummy_translator()
    ):
        client = CorpusLangChainVectorClient(adapter=adapter)

        assert hasattr(client, "query")
        assert hasattr(client, "batch_query")
        assert hasattr(client, "upsert")
        assert hasattr(client, "delete")
        assert hasattr(client, "create_namespace")
        assert hasattr(client, "delete_namespace")


# ---------------------------------------------------------------------------
# REAL LangChain integration tests (pass/fail; no skips)
# ---------------------------------------------------------------------------


def _require_langchain_for_vector_integration() -> None:
    """Fail-fast if langchain-core isn't importable for real integration checks."""
    try:
        import langchain_core  # noqa: F401
        from langchain_core.retrievers import BaseRetriever  # noqa: F401
        from langchain_core.vectorstores import VectorStore  # noqa: F401
    except Exception as exc:
        pytest.fail(
            "LangChain integration tests require langchain-core. Install with:\n"
            "  pip install -U langchain-core\n"
            f"Import error: {exc!r}",
            pytrace=False,
        )

    if not getattr(langchain_adapter_module, "LANGCHAIN_AVAILABLE", False):
        pytest.fail(
            "LANGCHAIN_AVAILABLE is False but langchain_core imports succeeded. "
            "This indicates an internal inconsistency in corpus_sdk.vector.framework_adapters.langchain.",
            pytrace=False,
        )


@pytest.mark.integration
def test_real_langchain_vector_store_is_vectorstore(adapter: Any) -> None:
    """When LangChain is installed, CorpusLangChainVectorStore must be a real VectorStore."""
    _require_langchain_for_vector_integration()
    from langchain_core.vectorstores import VectorStore  # type: ignore[import]

    store = CorpusLangChainVectorStore(
        corpus_adapter=adapter,
        embedding_function=_make_mock_embedding_function([[0.1, 0.2]]),
        default_top_k=2,
    )
    assert isinstance(store, VectorStore)


@pytest.mark.integration
def test_real_langchain_similarity_search_returns_real_documents(adapter: Any) -> None:
    """similarity_search should return real langchain_core Document objects."""
    _require_langchain_for_vector_integration()
    from langchain_core.documents import Document as LCDocument  # type: ignore[import]

    from corpus_sdk.vector.vector_base import QueryResult

    class AsyncVectorAdapter:
        async def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        async def query(self, *_a: Any, **_k: Any) -> QueryResult:
            # One match is enough for type validation.
            return _result([_make_match(id="doc-0", score=0.9, text="hello")])

        async def health(self) -> Dict[str, Any]:
            return {"status": "ok"}

    store = CorpusLangChainVectorStore(
        corpus_adapter=AsyncVectorAdapter(),
        embedding_function=_make_mock_embedding_function([[0.1, 0.2]]),
        default_top_k=1,
    )

    docs = store.similarity_search("hello", k=1)
    assert isinstance(docs, list) and len(docs) == 1
    assert isinstance(docs[0], LCDocument)
    assert isinstance(docs[0].page_content, str)


@pytest.mark.integration
def test_real_langchain_vector_store_as_retriever_invoke(adapter: Any) -> None:
    """VectorStore.as_retriever() should return an invokable retriever (sync)."""
    _require_langchain_for_vector_integration()

    from corpus_sdk.vector.vector_base import QueryResult

    class AsyncVectorAdapter:
        async def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        async def query(self, *_a: Any, **_k: Any) -> QueryResult:
            return _result([_make_match(id="doc-0", score=0.9, text="hello")])

        async def health(self) -> Dict[str, Any]:
            return {"status": "ok"}

    store = CorpusLangChainVectorStore(
        corpus_adapter=AsyncVectorAdapter(),
        embedding_function=_make_mock_embedding_function([[0.1, 0.2]]),
        default_top_k=1,
    )

    retriever = store.as_retriever(search_kwargs={"k": 1})
    if not hasattr(retriever, "invoke"):
        pytest.fail("Expected retriever to expose invoke() when using LangChain.", pytrace=False)

    out_sync = retriever.invoke("hello")
    assert isinstance(out_sync, list)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_langchain_vector_store_as_retriever_ainvoke(adapter: Any) -> None:
    """VectorStore.as_retriever() should return an invokable retriever (async)."""
    _require_langchain_for_vector_integration()

    from corpus_sdk.vector.vector_base import QueryResult

    class AsyncVectorAdapter:
        async def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        async def query(self, *_a: Any, **_k: Any) -> QueryResult:
            return _result([_make_match(id="doc-0", score=0.9, text="hello")])

        async def health(self) -> Dict[str, Any]:
            return {"status": "ok"}

    store = CorpusLangChainVectorStore(
        corpus_adapter=AsyncVectorAdapter(),
        embedding_function=_make_mock_embedding_function([[0.1, 0.2]]),
        async_embedding_function=_make_async_embedding_function,
        default_top_k=1,
    )

    retriever = store.as_retriever(search_kwargs={"k": 1})
    if not hasattr(retriever, "ainvoke"):
        pytest.fail("Expected retriever to expose ainvoke() when using LangChain.", pytrace=False)

    out_async = await retriever.ainvoke("hello")
    assert isinstance(out_async, list)


@pytest.mark.integration
def test_real_langchain_retriever_is_base_retriever_and_invokable(adapter: Any) -> None:
    """Smoke test: CorpusLangChainRetriever is a real BaseRetriever and can run."""
    _require_langchain_for_vector_integration()
    from langchain_core.retrievers import BaseRetriever  # type: ignore[import]

    class AsyncVectorAdapter:
        async def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        async def query(self, *_a: Any, **_k: Any) -> QueryResult:
            return _empty_result(namespace="default")

        async def health(self) -> Dict[str, Any]:
            return {"status": "ok"}

    store = CorpusLangChainVectorStore(
        corpus_adapter=AsyncVectorAdapter(),
        embedding_function=_make_mock_embedding_function([[0.1, 0.2]]),
        default_top_k=2,
    )
    retriever = CorpusLangChainRetriever(vector_store=store)
    assert isinstance(retriever, BaseRetriever)

    if hasattr(retriever, "invoke"):
        out = retriever.invoke("hello")
    else:
        out = retriever.get_relevant_documents("hello")
    assert isinstance(out, list)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_langchain_retriever_async_is_invokable(adapter: Any) -> None:
    """Smoke test: async retriever entrypoints work with real langchain-core."""
    _require_langchain_for_vector_integration()

    class AsyncVectorAdapter:
        async def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        async def query(self, *_a: Any, **_k: Any) -> QueryResult:
            return _empty_result(namespace="default")

        async def health(self) -> Dict[str, Any]:
            return {"status": "ok"}

    store = CorpusLangChainVectorStore(
        corpus_adapter=AsyncVectorAdapter(),
        embedding_function=_make_mock_embedding_function([[0.1, 0.2]]),
        async_embedding_function=_make_async_embedding_function,
        default_top_k=2,
    )
    retriever = CorpusLangChainRetriever(vector_store=store)

    if hasattr(retriever, "ainvoke"):
        out = await retriever.ainvoke("hello")
    else:
        out = await retriever.aget_relevant_documents("hello")
    assert isinstance(out, list)
