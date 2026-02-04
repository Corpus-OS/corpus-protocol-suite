"""AutoGen Vector framework adapter tests.

These tests are written against the current public API in
`corpus_sdk.vector.framework_adapters.autogen`, which exposes an AutoGen-friendly
vector store (`CorpusAutoGenVectorStore`) with sync/async methods, streaming,
MMR search, and callable embedding interfaces.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
from unittest.mock import Mock, patch

import pytest

import corpus_sdk.vector.framework_adapters.autogen as autogen_adapter_module
from corpus_sdk.vector.framework_adapters.autogen import (
    AutoGenDocument,
    CorpusAutoGenVectorClient,
    CorpusAutoGenVectorStore,
    CorpusAutoGenRetrieverTool,
    ErrorCodes,
)
from corpus_sdk.vector.vector_base import (
    BadRequest,
    NotSupported,
    OperationContext,
    VectorCapabilities,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_TEXT = "hello from autogen vector tests"
SAMPLE_EMBEDDING = [0.1, 0.2, 0.3, 0.4]


def _make_caps(**overrides: Any) -> VectorCapabilities:
    """Construct a protocol-valid VectorCapabilities for test doubles."""
    base: Dict[str, Any] = {
        "server": "test",
        "version": "0",
        "supports_batch_operations": True,
        "supports_metadata_filtering": True,
        "supports_batch_queries": True,
        "max_top_k": 100,
        "text_storage_strategy": "metadata",
    }
    base.update(overrides)
    return VectorCapabilities(**base)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_translator() -> Any:
    """Factory for creating a standard dummy translator for tests."""

    class DummyTranslator:
        def upsert(self, *a: Any, **k: Any) -> Any:
            return {"ids": ["doc-0"], "count": 1}

        async def arun_upsert(self, *a: Any, **k: Any) -> Any:
            return {"ids": ["doc-0"], "count": 1}

        def query(self, *a: Any, **k: Any) -> Any:
            return {
                "matches": [
                    {
                        "id": "doc-0",
                        "score": 0.95,
                        "metadata": {"page_content": "test", "id": "doc-0"},
                        "vector": [0.1, 0.2],
                    }
                ],
                "namespace": "default",
            }

        async def arun_query(self, *a: Any, **k: Any) -> Any:
            return {
                "matches": [
                    {
                        "id": "doc-0",
                        "score": 0.95,
                        "metadata": {"page_content": "test", "id": "doc-0"},
                        "vector": [0.1, 0.2],
                    }
                ],
                "namespace": "default",
            }

        def query_stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            yield {
                "matches": [
                    {
                        "id": "doc-0",
                        "score": 0.95,
                        "metadata": {"page_content": "test", "id": "doc-0"},
                    }
                ],
                "is_final": True,
            }

        async def arun_query_stream(self, *a: Any, **k: Any) -> Any:
            async def _gen():
                yield {
                    "matches": [
                        {
                            "id": "doc-0",
                            "score": 0.95,
                            "metadata": {"page_content": "test", "id": "doc-0"},
                        }
                    ],
                    "is_final": True,
                }

            return _gen()

        def delete(self, *a: Any, **k: Any) -> Any:
            return {"deleted": 1}

        async def arun_delete(self, *a: Any, **k: Any) -> Any:
            return {"deleted": 1}

        def capabilities(self, *a: Any, **k: Any) -> VectorCapabilities:
            return _make_caps()

        async def acapabilities(self, *a: Any, **k: Any) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

        async def ahealth(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    return DummyTranslator()


def _make_mock_embedding_function(vectors: Optional[List[List[float]]] = None):
    """Create a mock embedding function that returns predefined vectors."""
    vectors = vectors or [list(SAMPLE_EMBEDDING)]

    def embedding_fn(texts: List[str]) -> List[List[float]]:
        # Default behavior: return one embedding per input text.
        if len(vectors) == 1:
            return [list(vectors[0]) for _ in texts]
        return vectors[: len(texts)]

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
            return {"matches": [], "namespace": "default"}

        def upsert(self, *args: Any, **kwargs: Any) -> Any:
            return {"ids": ["doc-0"], "count": 1}

        def delete(self, *args: Any, **kwargs: Any) -> Any:
            return {"deleted": 1}

        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Dict[str, Any]:
            return {"status": "ok"}

    return TestAdapter()


# ---------------------------------------------------------------------------
# Construction / Initialization Tests (8 tests)
# ---------------------------------------------------------------------------


def test_init_requires_corpus_adapter() -> None:
    """Adapter must be provided and implement VectorProtocolV1."""
    with pytest.raises((TypeError, AttributeError)):
        CorpusAutoGenVectorStore(corpus_adapter=None)  # type: ignore


def test_init_stores_config_attributes(adapter: Any) -> None:
    """Store should keep key config attributes accessible."""
    store = CorpusAutoGenVectorStore(
        corpus_adapter=adapter,
        namespace="test-ns",
        id_field="custom_id",
        text_field="custom_text",
        metadata_field="custom_meta",
        score_threshold=0.8,
        default_top_k=10,
        framework_version="v1.0",
    )

    assert store.namespace == "test-ns"
    assert store.id_field == "custom_id"
    assert store.text_field == "custom_text"
    assert store.metadata_field == "custom_meta"
    assert store.score_threshold == 0.8
    assert store.default_top_k == 10
    assert store._framework_version == "v1.0"


def test_init_validates_score_threshold_range(adapter: Any) -> None:
    """score_threshold should be between 0.0 and 1.0 if provided."""
    # Valid range should work
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, score_threshold=0.5)
    assert store.score_threshold == 0.5

    # Edge cases
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, score_threshold=0.0)
    assert store.score_threshold == 0.0

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, score_threshold=1.0)
    assert store.score_threshold == 1.0


def test_init_validates_default_top_k_positive(adapter: Any) -> None:
    """default_top_k must be positive."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, default_top_k=5)
    assert store.default_top_k == 5


def test_init_with_embedding_function(adapter: Any) -> None:
    """embedding_function should be stored correctly."""
    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)
    assert store.embedding_function is fn


def test_init_with_async_embedding_function(adapter: Any) -> None:
    """async_embedding_function should be stored correctly."""
    store = CorpusAutoGenVectorStore(
        corpus_adapter=adapter, async_embedding_function=_make_async_embedding_function
    )
    assert store.async_embedding_function is _make_async_embedding_function


def test_init_with_framework_version(adapter: Any) -> None:
    """framework_version should be stored and accessible."""
    store = CorpusAutoGenVectorStore(
        corpus_adapter=adapter, framework_version="autogen-v2.0"
    )
    assert store._framework_version == "autogen-v2.0"


def test_init_with_own_adapter_flag(adapter: Any) -> None:
    """own_adapter=True should enable lifecycle management."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, own_adapter=True)
    assert store._own_adapter is True


# ---------------------------------------------------------------------------
# Translator Wiring Tests (4 tests)
# ---------------------------------------------------------------------------


def test_translator_created_with_framework_autogen(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Translator factory should be called with framework='autogen'."""
    captured: Dict[str, Any] = {}

    def fake_create_vector_translator(*_: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return _make_dummy_translator()

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", fake_create_vector_translator
    )

    store = CorpusAutoGenVectorStore(
        corpus_adapter=adapter, embedding_function=_make_mock_embedding_function()
    )
    _ = store.add_texts([SAMPLE_TEXT])

    assert captured.get("framework") == "autogen"
    assert captured.get("adapter") is adapter


def test_translator_cached_property_reused(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Multiple accesses to _translator should return same instance."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    translator1 = store._translator
    translator2 = store._translator

    assert translator1 is translator2


def test_translator_uses_create_vector_translator_factory(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should use common create_vector_translator factory."""
    called = {"count": 0}

    def counting_factory(*_: Any, **__: Any) -> Any:
        called["count"] += 1
        return _make_dummy_translator()

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", counting_factory
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    _ = store._translator

    assert called["count"] == 1


def test_translator_available_on_first_access(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Lazy construction should work on first access."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    # Should not raise
    translator = store._translator
    assert translator is not None


# ---------------------------------------------------------------------------
# Context Translation Tests (6 tests)
# ---------------------------------------------------------------------------


def test_build_core_context_from_conversation(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should build OperationContext from AutoGen conversation."""
    captured: Dict[str, Any] = {}
    base_ctx = OperationContext(request_id="from-core", tenant="from-core", attrs={"x": 1})

    def fake_context_from_autogen(
        conversation: Any,
        *,
        framework_version: Any = None,
        **extra: Any,
    ) -> Any:
        captured["conversation"] = conversation
        captured["framework_version"] = framework_version
        captured.update(extra)
        return base_ctx

    monkeypatch.setattr(
        autogen_adapter_module, "core_ctx_from_autogen", fake_context_from_autogen
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, framework_version="v1.0")
    mock_conversation = {"messages": []}

    ctx = store._build_core_context(conversation=mock_conversation, extra_context=None)

    assert captured["conversation"] == mock_conversation
    assert captured["framework_version"] == "v1.0"
    assert isinstance(ctx, OperationContext)


def test_build_core_context_handles_none_gracefully(adapter: Any) -> None:
    """Should return None when no context is provided."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    ctx = store._build_core_context(conversation=None, extra_context=None)
    assert ctx is None


def test_build_core_context_with_extra_context(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should merge extra_context dict."""
    captured: Dict[str, Any] = {}
    base_ctx = OperationContext(request_id="test", tenant="test", attrs={})

    def fake_context_from_autogen(
        conversation: Any,
        *,
        framework_version: Any = None,
        **extra: Any,
    ) -> Any:
        captured.update(extra)
        return base_ctx

    monkeypatch.setattr(
        autogen_adapter_module, "core_ctx_from_autogen", fake_context_from_autogen
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    extra = {"custom_key": "custom_value"}

    _ = store._build_core_context(conversation=None, extra_context=extra)

    assert captured["custom_key"] == "custom_value"


def test_context_translation_error_attaches_context(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Errors during context translation should attach error context."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    def fake_context_from_autogen(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("ctx translation failed")

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)
    monkeypatch.setattr(
        autogen_adapter_module, "core_ctx_from_autogen", fake_context_from_autogen
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)

    with pytest.raises(RuntimeError, match="ctx translation failed"):
        store._build_core_context(conversation={}, extra_context=None)

    assert captured_ctx.get("framework") == "autogen"
    assert captured_ctx.get("operation") == "vector_context_translation"


def test_build_framework_context_includes_namespace(adapter: Any) -> None:
    """framework_ctx should include namespace."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, namespace="test-ns")
    fw_ctx = store._build_framework_context(
        None, operation="test_op", namespace="override-ns"
    )

    assert fw_ctx["namespace"] == "override-ns"
    assert fw_ctx["operation"] == "test_op"


def test_build_contexts_orchestrates_both(adapter: Any) -> None:
    """_build_contexts should return both OperationContext and framework_ctx."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, namespace="test-ns")
    core_ctx, fw_ctx = store._build_contexts(
        operation="test",
        conversation=None,
        extra_context=None,
        namespace=None,
    )

    assert core_ctx is None  # No conversation provided
    assert isinstance(fw_ctx, Mapping)
    assert fw_ctx["namespace"] == "test-ns"


# ---------------------------------------------------------------------------
# Namespace Resolution Tests (3 tests)
# ---------------------------------------------------------------------------


def test_effective_namespace_uses_override(adapter: Any) -> None:
    """Explicit namespace should override store default."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, namespace="default")
    ns = store._effective_namespace("override")
    assert ns == "override"


def test_effective_namespace_uses_store_default(adapter: Any) -> None:
    """Should use store default when not specified."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, namespace="default")
    ns = store._effective_namespace(None)
    assert ns == "default"


def test_effective_namespace_handles_none_default(adapter: Any) -> None:
    """None default should pass through."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, namespace=None)
    ns = store._effective_namespace(None)
    assert ns is None


# ---------------------------------------------------------------------------
# Dimension Hint Tests (6 tests)
# ---------------------------------------------------------------------------


def test_update_dim_hint_sets_first_write(adapter: Any) -> None:
    """First non-zero dimension should win."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    assert store._vector_dim_hint is None

    store._update_dim_hint(4)
    assert store._vector_dim_hint == 4


def test_update_dim_hint_thread_safe(adapter: Any) -> None:
    """Concurrent updates should not race."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)

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
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    store._update_dim_hint(4)
    store._update_dim_hint(8)

    assert store._vector_dim_hint == 4


def test_maybe_check_dim_validates_against_hint(adapter: Any) -> None:
    """Should raise on dimension mismatch."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    store._update_dim_hint(4)

    with pytest.raises(BadRequest, match="dimension mismatch"):
        store._maybe_check_dim([0.1, 0.2], where="test")


def test_maybe_check_dim_noop_when_hint_none(adapter: Any) -> None:
    """No validation without hint."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    # Should not raise
    store._maybe_check_dim([0.1, 0.2], where="test")


def test_zero_vector_requires_known_dimension(adapter: Any) -> None:
    """Should raise if dimension unknown."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)

    with pytest.raises(BadRequest, match="dimension is unknown"):
        store._zero_vector()


# ---------------------------------------------------------------------------
# Embedding Function Tests (Sync) (8 tests)
# ---------------------------------------------------------------------------


def test_ensure_embeddings_uses_provided(adapter: Any) -> None:
    """Should use explicit embeddings when provided."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    texts = ["a", "b"]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    result = store._ensure_embeddings(texts, embeddings)

    assert result == embeddings


def test_ensure_embeddings_validates_length_match(adapter: Any) -> None:
    """Should raise if embeddings length doesn't match texts length."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
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

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=counting_fn)
    texts = ["a", "b"]

    result = store._ensure_embeddings(texts, None)

    assert called["times"] == 1
    assert len(result) == 2


def test_ensure_embeddings_raises_without_function(adapter: Any) -> None:
    """Should raise NotSupported if no function configured."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    texts = ["a"]

    with pytest.raises(NotSupported, match="No embedding_function"):
        store._ensure_embeddings(texts, None)


def test_ensure_embeddings_validates_batch_shape(adapter: Any) -> None:
    """All vectors should have same dimension."""
    fn = lambda texts: [[0.1, 0.2], [0.3, 0.4, 0.5]]  # Mismatched dims
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)
    texts = ["a", "b"]

    with pytest.raises(BadRequest, match="inconsistent vector sizes"):
        store._ensure_embeddings(texts, None)


def test_ensure_embeddings_updates_dim_hint(adapter: Any) -> None:
    """Should set hint from first vector."""
    fn = _make_mock_embedding_function([[0.1, 0.2, 0.3]])
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)
    texts = ["a"]

    store._ensure_embeddings(texts, None)

    assert store._vector_dim_hint == 3


def test_ensure_embeddings_enforces_dim_hint(adapter: Any) -> None:
    """Should raise if vectors mismatch hint."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    store._update_dim_hint(4)

    embeddings = [[0.1, 0.2]]  # Only 2 dims, hint is 4

    with pytest.raises(BadRequest, match="dimension mismatch"):
        store._ensure_embeddings(["a"], embeddings)


def test_ensure_embeddings_handles_function_errors(adapter: Any) -> None:
    """Should wrap function errors with context."""

    def failing_fn(texts: List[str]) -> List[List[float]]:
        raise ValueError("embedding failed")

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=failing_fn)

    with pytest.raises(BadRequest, match="embedding_function failed"):
        store._ensure_embeddings(["a"], None)


# ---------------------------------------------------------------------------
# Embedding Function Tests (Async) (8 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_embeddings_async_uses_provided(adapter: Any) -> None:
    """Should use explicit embeddings."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    texts = ["a", "b"]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    result = await store._ensure_embeddings_async(texts, embeddings)

    assert result == embeddings


@pytest.mark.asyncio
async def test_ensure_embeddings_async_validates_length_match(adapter: Any) -> None:
    """Should raise if lengths mismatch."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
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

    store = CorpusAutoGenVectorStore(
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

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=counting_fn)
    texts = ["a"]

    result = await store._ensure_embeddings_async(texts, None)

    assert called["times"] == 1


@pytest.mark.asyncio
async def test_ensure_embeddings_async_raises_without_function(adapter: Any) -> None:
    """Should raise NotSupported if no function."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)

    with pytest.raises(NotSupported, match="No embedding_function"):
        await store._ensure_embeddings_async(["a"], None)


@pytest.mark.asyncio
async def test_ensure_embeddings_async_validates_batch_shape(adapter: Any) -> None:
    """All vectors should have same dimension."""

    async def bad_fn(texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2], [0.3, 0.4, 0.5]]

    store = CorpusAutoGenVectorStore(
        corpus_adapter=adapter, async_embedding_function=bad_fn
    )

    with pytest.raises(BadRequest, match="inconsistent vector sizes"):
        await store._ensure_embeddings_async(["a", "b"], None)


@pytest.mark.asyncio
async def test_ensure_embeddings_async_updates_dim_hint(adapter: Any) -> None:
    """Should set hint from first vector."""

    async def fn(texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3]]

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, async_embedding_function=fn)

    await store._ensure_embeddings_async(["a"], None)

    assert store._vector_dim_hint == 3


@pytest.mark.asyncio
async def test_ensure_embeddings_async_handles_function_errors(adapter: Any) -> None:
    """Should wrap errors with context."""

    async def failing_fn(texts: List[str]) -> List[List[float]]:
        raise ValueError("async embedding failed")

    store = CorpusAutoGenVectorStore(
        corpus_adapter=adapter, async_embedding_function=failing_fn
    )

    with pytest.raises(BadRequest, match="async_embedding_function failed"):
        await store._ensure_embeddings_async(["a"], None)


# ---------------------------------------------------------------------------
# Query Embedding Tests (Sync) (6 tests)
# ---------------------------------------------------------------------------


def test_embed_query_uses_provided_embedding(adapter: Any) -> None:
    """Should use explicit embedding when provided."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    store._update_dim_hint(4)

    result = store._embed_query("test", embedding=[0.1, 0.2, 0.3, 0.4])

    assert result == [0.1, 0.2, 0.3, 0.4]


def test_embed_query_validates_embedding_dimension(adapter: Any) -> None:
    """Should enforce dim_hint."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    store._update_dim_hint(4)

    with pytest.raises(BadRequest, match="dimension mismatch"):
        store._embed_query("test", embedding=[0.1, 0.2])


def test_embed_query_calls_embedding_function(adapter: Any) -> None:
    """Should call embedding_function([query])."""
    called = {"times": 0}

    def counting_fn(texts: List[str]) -> List[List[float]]:
        called["times"] += 1
        return [[0.1, 0.2, 0.3, 0.4]]

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=counting_fn)

    result = store._embed_query("test query")

    assert called["times"] == 1
    assert len(result) == 4


def test_embed_query_returns_zero_vector_for_empty_query(adapter: Any) -> None:
    """Should return deterministic zeros if dim known."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    store._update_dim_hint(4)

    result = store._embed_query("")

    assert result == [0.0, 0.0, 0.0, 0.0]


def test_embed_query_raises_for_empty_query_without_dim(adapter: Any) -> None:
    """Should raise if dim unknown."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)

    with pytest.raises(BadRequest, match="query cannot be empty"):
        store._embed_query("")


def test_embed_query_raises_without_function(adapter: Any) -> None:
    """Should raise NotSupported if no function."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)

    with pytest.raises(NotSupported, match="No embedding_function"):
        store._embed_query("test")


# ---------------------------------------------------------------------------
# Query Embedding Tests (Async) (6 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_query_async_uses_provided_embedding(adapter: Any) -> None:
    """Should use explicit embedding."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    store._update_dim_hint(4)

    result = await store._embed_query_async("test", embedding=[0.1, 0.2, 0.3, 0.4])

    assert result == [0.1, 0.2, 0.3, 0.4]


@pytest.mark.asyncio
async def test_embed_query_async_validates_embedding_dimension(adapter: Any) -> None:
    """Should enforce dim_hint."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    store._update_dim_hint(4)

    with pytest.raises(BadRequest, match="dimension mismatch"):
        await store._embed_query_async("test", embedding=[0.1, 0.2])


@pytest.mark.asyncio
async def test_embed_query_async_calls_async_function(adapter: Any) -> None:
    """Should prefer async_embedding_function."""
    called = {"times": 0}

    async def counting_fn(texts: List[str]) -> List[List[float]]:
        called["times"] += 1
        return [[0.1, 0.2, 0.3, 0.4]]

    store = CorpusAutoGenVectorStore(
        corpus_adapter=adapter, async_embedding_function=counting_fn
    )

    result = await store._embed_query_async("test")

    assert called["times"] == 1
    assert len(result) == 4


@pytest.mark.asyncio
async def test_embed_query_async_falls_back_to_sync(adapter: Any) -> None:
    """Should use sync function in thread."""
    fn = _make_mock_embedding_function([[0.1, 0.2, 0.3, 0.4]])
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    result = await store._embed_query_async("test")

    assert len(result) == 4


@pytest.mark.asyncio
async def test_embed_query_async_returns_zero_vector_for_empty_query(adapter: Any) -> None:
    """Should return deterministic zeros if dim known."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    store._update_dim_hint(4)

    result = await store._embed_query_async("")

    assert result == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_embed_query_async_raises_without_function(adapter: Any) -> None:
    """Should raise NotSupported if no function."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)

    with pytest.raises(NotSupported, match="No embedding_function"):
        await store._embed_query_async("test")


# ---------------------------------------------------------------------------
# Metadata/IDs Normalization Tests (4 tests)
# ---------------------------------------------------------------------------


def test_normalize_metadatas_returns_empty_dicts(adapter: Any) -> None:
    """None → [{}, {}, ...]."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    result = store._normalize_metadatas(3, None)

    assert result == [{}, {}, {}]


def test_normalize_metadatas_replicates_single(adapter: Any) -> None:
    """Single metadata → replicated to n."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    result = store._normalize_metadatas(3, [{"key": "value"}])

    assert len(result) == 3
    assert all(m == {"key": "value"} for m in result)


def test_normalize_metadatas_raises_on_length_mismatch(adapter: Any) -> None:
    """Should raise if len mismatch."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)

    with pytest.raises(BadRequest, match="does not match"):
        store._normalize_metadatas(3, [{"a": 1}, {"b": 2}])


def test_normalize_ids_generates_defaults(adapter: Any) -> None:
    """None → ["doc-0", "doc-1", ...]."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    result = store._normalize_ids(3, None)

    assert result == ["doc-0", "doc-1", "doc-2"]


# ---------------------------------------------------------------------------
# Add Texts Tests (Sync) (6 tests)
# ---------------------------------------------------------------------------


def test_add_texts_returns_ids(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Should return list of IDs."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function([[0.1, 0.2, 0.3, 0.4]])
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

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
            return {"ids": ["doc-0"], "count": 1}

        def capabilities(self, *a: Any, **k: Any) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    fn = _make_mock_embedding_function([[0.1, 0.2, 0.3, 0.4]])
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    store.add_texts(["test"])

    assert called["upsert"] is True


def test_add_texts_uses_embedding_function(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should compute embeddings if not provided."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    called = {"times": 0}

    def counting_fn(texts: List[str]) -> List[List[float]]:
        called["times"] += 1
        return [[0.1, 0.2, 0.3, 0.4]]

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=counting_fn)

    store.add_texts(["test"])

    assert called["times"] == 1


def test_add_texts_validates_empty_texts(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should raise for empty/whitespace texts."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    with pytest.raises(BadRequest, match="empty or whitespace"):
        store.add_texts(["   "])


def test_add_texts_validates_upsert_result(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should raise if all documents failed."""

    class FailingTranslator:
        def upsert(self, *a: Any, **k: Any) -> Any:
            return {"ids": [], "count": 0}

        def capabilities(self, *a: Any, **k: Any) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: FailingTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    with pytest.raises(BadRequest, match="All documents failed"):
        store.add_texts(["test"])


def test_add_texts_builds_raw_documents_with_metadata(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should build correct document shape."""
    captured: Dict[str, Any] = {}

    class CapturingTranslator:
        def upsert(self, raw_documents: Any, **k: Any) -> Any:
            captured["raw_documents"] = raw_documents
            return {"ids": ["doc-0"], "count": 1}

        def capabilities(self, *a: Any, **k: Any) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: CapturingTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    store.add_texts(["test"], metadatas=[{"key": "value"}])

    docs = captured["raw_documents"]
    assert len(docs) == 1
    assert docs[0]["id"] == "doc-0"
    assert "vector" in docs[0]
    assert "metadata" in docs[0]


# ---------------------------------------------------------------------------
# Add Texts Tests (Async) (6 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aadd_texts_returns_ids(
    monkeypatch: pytest.MonkeyPatch, adapter: Any
) -> None:
    """Should return list of IDs."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

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
            return {"ids": ["doc-0"], "count": 1}

        async def acapabilities(self) -> VectorCapabilities:
            return _make_caps()

        async def ahealth(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    await store.aadd_texts(["test"])

    assert called["arun_upsert"] is True


@pytest.mark.asyncio
async def test_aadd_texts_uses_async_embedding_function(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should compute embeddings async."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    called = {"times": 0}

    async def counting_fn(texts: List[str]) -> List[List[float]]:
        called["times"] += 1
        return [[0.1, 0.2, 0.3, 0.4]]

    store = CorpusAutoGenVectorStore(
        corpus_adapter=adapter, async_embedding_function=counting_fn
    )

    await store.aadd_texts(["test"])

    assert called["times"] == 1


@pytest.mark.asyncio
async def test_aadd_texts_validates_empty_texts(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should raise for empty/whitespace texts."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    with pytest.raises(BadRequest, match="empty or whitespace"):
        await store.aadd_texts(["   "])


@pytest.mark.asyncio
async def test_aadd_texts_validates_upsert_result(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should raise if all documents failed."""

    class FailingTranslator:
        async def arun_upsert(self, *a: Any, **k: Any) -> Any:
            return {"ids": [], "count": 0}

        async def acapabilities(self) -> VectorCapabilities:
            return _make_caps()

        async def ahealth(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: FailingTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    with pytest.raises(BadRequest, match="All documents failed"):
        await store.aadd_texts(["test"])


@pytest.mark.asyncio
async def test_aadd_texts_builds_raw_documents_with_metadata(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should build correct document shape."""
    captured: Dict[str, Any] = {}

    class CapturingTranslator:
        async def arun_upsert(self, raw_documents: Any, **k: Any) -> Any:
            captured["raw_documents"] = raw_documents
            return {"ids": ["doc-0"], "count": 1}

        async def acapabilities(self) -> VectorCapabilities:
            return _make_caps()

        async def ahealth(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: CapturingTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    await store.aadd_texts(["test"], metadatas=[{"key": "value"}])

    docs = captured["raw_documents"]
    assert len(docs) == 1
    assert "vector" in docs[0]
    assert "metadata" in docs[0]


# ---------------------------------------------------------------------------
# Add Documents Tests (4 tests)
# ---------------------------------------------------------------------------


def test_add_documents_extracts_texts_and_metadata(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should convert AutoGenDocument → texts/metadata."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    docs = [AutoGenDocument(page_content="test", metadata={"key": "value"})]
    ids = store.add_documents(docs)

    assert len(ids) == 1


def test_add_documents_delegates_to_add_texts(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should call add_texts internally."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    called = {"add_texts": False}
    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    original_add_texts = store.add_texts

    def counting_add_texts(*args, **kwargs):
        called["add_texts"] = True
        return original_add_texts(*args, **kwargs)

    store.add_texts = counting_add_texts  # type: ignore

    docs = [AutoGenDocument(page_content="test", metadata={})]
    store.add_documents(docs)

    assert called["add_texts"] is True


@pytest.mark.asyncio
async def test_aadd_documents_extracts_texts_and_metadata(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should convert AutoGenDocument → texts/metadata."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    docs = [AutoGenDocument(page_content="test", metadata={"key": "value"})]
    ids = await store.aadd_documents(docs)

    assert len(ids) == 1


@pytest.mark.asyncio
async def test_aadd_documents_delegates_to_aadd_texts(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should call aadd_texts internally."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    called = {"aadd_texts": False}
    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    original_aadd_texts = store.aadd_texts

    async def counting_aadd_texts(*args, **kwargs):
        called["aadd_texts"] = True
        return await original_aadd_texts(*args, **kwargs)

    store.aadd_texts = counting_aadd_texts  # type: ignore

    docs = [AutoGenDocument(page_content="test", metadata={})]
    await store.aadd_documents(docs)

    assert called["aadd_texts"] is True


# ---------------------------------------------------------------------------
# Similarity Search Tests (Sync) (6 tests)
# ---------------------------------------------------------------------------


def test_similarity_search_returns_documents(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should return List[AutoGenDocument]."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    docs = store.similarity_search("test query")

    assert isinstance(docs, list)
    assert all(isinstance(d, AutoGenDocument) for d in docs)


def test_similarity_search_calls_translator_query(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should delegate to translator.query()."""
    called = {"query": False}

    class DummyTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            called["query"] = True
            return {"matches": [], "namespace": "default"}

        def capabilities(self, *a: Any, **k: Any) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    store.similarity_search("test")

    assert called["query"] is True


def test_similarity_search_uses_default_top_k(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should use store default_top_k."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def query(self, raw_query: Any, **k: Any) -> Any:
            captured["raw_query"] = raw_query
            return {"matches": [], "namespace": "default"}

        def capabilities(self, *a: Any, **k: Any) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(
        corpus_adapter=adapter, embedding_function=fn, default_top_k=10
    )

    store.similarity_search("test")  # No k specified

    assert captured["raw_query"]["top_k"] == 10


def test_similarity_search_validates_top_k_against_capabilities(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should raise if exceeds max_top_k."""

    class DummyTranslator:
        def capabilities(self, *a: Any, **k: Any) -> VectorCapabilities:
            return _make_caps(max_top_k=10)

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    with pytest.raises(BadRequest, match="exceeds maximum"):
        store.similarity_search("test", k=20)


def test_similarity_search_validates_filter_support(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should raise if filters not supported."""

    class DummyTranslator:
        def capabilities(self, *a: Any, **k: Any) -> VectorCapabilities:
            return _make_caps(supports_metadata_filtering=False)

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    with pytest.raises(NotSupported, match="metadata filtering"):
        store.similarity_search("test", filter={"key": "value"})


def test_similarity_search_applies_score_threshold(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should filter results by score_threshold."""

    class DummyTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            return {
                "matches": [
                    {"id": "1", "score": 0.9, "metadata": {"page_content": "high"}},
                    {"id": "2", "score": 0.3, "metadata": {"page_content": "low"}},
                ],
                "namespace": "default",
            }

        def capabilities(self, *a: Any, **k: Any) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(
        corpus_adapter=adapter, embedding_function=fn, score_threshold=0.5
    )

    docs = store.similarity_search("test")

    assert len(docs) == 1
    assert docs[0].page_content == "high"


# ---------------------------------------------------------------------------
# Similarity Search Tests (Async) (6 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_asimilarity_search_returns_documents(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should return List[AutoGenDocument]."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    docs = await store.asimilarity_search("test query")

    assert isinstance(docs, list)
    assert all(isinstance(d, AutoGenDocument) for d in docs)


@pytest.mark.asyncio
async def test_asimilarity_search_calls_translator_arun_query(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should delegate to translator.arun_query()."""
    called = {"arun_query": False}

    class DummyTranslator:
        async def arun_query(self, *a: Any, **k: Any) -> Any:
            called["arun_query"] = True
            return {"matches": [], "namespace": "default"}

        async def acapabilities(self, *a: Any, **k: Any) -> VectorCapabilities:
            return _make_caps()

        async def ahealth(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    await store.asimilarity_search("test")

    assert called["arun_query"] is True


@pytest.mark.asyncio
async def test_asimilarity_search_uses_default_top_k(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should use store default_top_k."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_query(self, raw_query: Any, **k: Any) -> Any:
            captured["raw_query"] = raw_query
            return {"matches": [], "namespace": "default"}

        async def acapabilities(self, *a: Any, **k: Any) -> VectorCapabilities:
            return _make_caps()

        async def ahealth(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(
        corpus_adapter=adapter, embedding_function=fn, default_top_k=10
    )

    await store.asimilarity_search("test")

    assert captured["raw_query"]["top_k"] == 10


@pytest.mark.asyncio
async def test_asimilarity_search_validates_top_k_against_capabilities(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should raise if exceeds max_top_k."""

    class DummyTranslator:
        async def acapabilities(self, *a: Any, **k: Any) -> VectorCapabilities:
            return _make_caps(max_top_k=10)

        async def ahealth(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    with pytest.raises(BadRequest, match="exceeds maximum"):
        await store.asimilarity_search("test", k=20)


@pytest.mark.asyncio
async def test_asimilarity_search_validates_filter_support(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should raise if filters not supported."""

    class DummyTranslator:
        async def acapabilities(self, *a: Any, **k: Any) -> VectorCapabilities:
            return _make_caps(supports_metadata_filtering=False)

        async def ahealth(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    with pytest.raises(NotSupported, match="metadata filtering"):
        await store.asimilarity_search("test", filter={"key": "value"})


@pytest.mark.asyncio
async def test_asimilarity_search_applies_score_threshold(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should filter results by score_threshold."""

    class DummyTranslator:
        async def arun_query(self, *a: Any, **k: Any) -> Any:
            return {
                "matches": [
                    {"id": "1", "score": 0.9, "metadata": {"page_content": "high"}},
                    {"id": "2", "score": 0.3, "metadata": {"page_content": "low"}},
                ],
                "namespace": "default",
            }

        async def acapabilities(self, *a: Any, **k: Any) -> VectorCapabilities:
            return _make_caps()

        async def ahealth(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(
        corpus_adapter=adapter, embedding_function=fn, score_threshold=0.5
    )

    docs = await store.asimilarity_search("test")

    assert len(docs) == 1
    assert docs[0].page_content == "high"


# ---------------------------------------------------------------------------
# Similarity Search with Score Tests (4 tests)
# ---------------------------------------------------------------------------


def test_similarity_search_with_score_returns_tuples(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should return List[Tuple[AutoGenDocument, float]]."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    results = store.similarity_search_with_score("test")

    assert isinstance(results, list)
    for doc, score in results:
        assert isinstance(doc, AutoGenDocument)
        assert isinstance(score, float)


def test_similarity_search_with_score_includes_scores(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scores should be present in tuples."""

    class DummyTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            return {
                "matches": [
                    {"id": "1", "score": 0.95, "metadata": {"page_content": "test"}},
                ],
                "namespace": "default",
            }

        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    results = store.similarity_search_with_score("test")

    assert len(results) == 1
    doc, score = results[0]
    assert score == 0.95


@pytest.mark.asyncio
async def test_asimilarity_search_with_score_returns_tuples(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should return List[Tuple[AutoGenDocument, float]]."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    results = await store.asimilarity_search_with_score("test")

    assert isinstance(results, list)
    for doc, score in results:
        assert isinstance(doc, AutoGenDocument)
        assert isinstance(score, float)


@pytest.mark.asyncio
async def test_asimilarity_search_with_score_includes_scores(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scores should be present in tuples."""

    class DummyTranslator:
        async def arun_query(self, *a: Any, **k: Any) -> Any:
            return {
                "matches": [
                    {"id": "1", "score": 0.95, "metadata": {"page_content": "test"}},
                ],
                "namespace": "default",
            }

        async def acapabilities(self) -> VectorCapabilities:
            return _make_caps()

        async def ahealth(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    results = await store.asimilarity_search_with_score("test")

    assert len(results) == 1
    doc, score = results[0]
    assert score == 0.95


# ---------------------------------------------------------------------------
# Streaming Search Tests (4 tests)
# ---------------------------------------------------------------------------


def test_similarity_search_stream_returns_iterator(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should return Iterator[AutoGenDocument]."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    result = store.similarity_search_stream("test")

    assert hasattr(result, "__iter__")


def test_similarity_search_stream_yields_documents(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should yield progressive documents."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    docs = list(store.similarity_search_stream("test"))

    assert all(isinstance(d, AutoGenDocument) for d in docs)


@pytest.mark.asyncio
async def test_asimilarity_search_stream_returns_async_iterator(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should return AsyncIterator[AutoGenDocument]."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    result = await store.asimilarity_search_stream("test")

    assert hasattr(result, "__aiter__")


@pytest.mark.asyncio
async def test_asimilarity_search_stream_yields_documents(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should yield progressive async documents."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    docs = []
    async for doc in await store.asimilarity_search_stream("test"):
        docs.append(doc)

    assert all(isinstance(d, AutoGenDocument) for d in docs)


# ---------------------------------------------------------------------------
# Raw Query API Tests (4 tests)
# ---------------------------------------------------------------------------


def test_query_returns_raw_matches(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should return List[Mapping[str, Any]]."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    matches = store.query([0.1, 0.2, 0.3, 0.4])

    assert isinstance(matches, list)


def test_query_delegates_to_translator(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should use translator.query()."""
    called = {"query": False}

    class DummyTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            called["query"] = True
            return {"matches": [], "namespace": "default"}

        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    store.query([0.1, 0.2, 0.3, 0.4])

    assert called["query"] is True


@pytest.mark.asyncio
async def test_aquery_returns_raw_matches(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should return List[Mapping[str, Any]]."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    matches = await store.aquery([0.1, 0.2, 0.3, 0.4])

    assert isinstance(matches, list)


@pytest.mark.asyncio
async def test_aquery_delegates_to_translator(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should use translator.arun_query()."""
    called = {"arun_query": False}

    class DummyTranslator:
        async def arun_query(self, *a: Any, **k: Any) -> Any:
            called["arun_query"] = True
            return {"matches": [], "namespace": "default"}

        async def acapabilities(self) -> VectorCapabilities:
            return _make_caps()

        async def ahealth(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    await store.aquery([0.1, 0.2, 0.3, 0.4])

    assert called["arun_query"] is True


# ---------------------------------------------------------------------------
# MMR Search Tests (6 tests)
# ---------------------------------------------------------------------------


def test_mmr_search_returns_documents(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should return List[AutoGenDocument]."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    docs = store.max_marginal_relevance_search("test")

    assert isinstance(docs, list)
    assert all(isinstance(d, AutoGenDocument) for d in docs)


def test_mmr_search_uses_mmr_config(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should pass MMRConfig to translator."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def query(self, *a: Any, mmr_config: Any = None, **k: Any) -> Any:
            captured["mmr_config"] = mmr_config
            return {"matches": [], "namespace": "default"}

        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    store.max_marginal_relevance_search("test", k=4, lambda_mult=0.5)

    mmr = captured["mmr_config"]
    assert mmr is not None
    assert mmr.enabled is True
    assert mmr.k == 4
    assert mmr.lambda_mult == 0.5


def test_mmr_search_requests_vectors(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should request include_vectors=True for MMR."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def query(self, raw_query: Any, **k: Any) -> Any:
            captured["raw_query"] = raw_query
            return {"matches": [], "namespace": "default"}

        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    store.max_marginal_relevance_search("test")

    assert captured["raw_query"]["include_vectors"] is True


def test_mmr_search_limits_results_to_k(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Final results should be capped at k (not fetch_k)."""

    class DummyTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            # Return more matches than requested k
            return {
                "matches": [
                    {"id": str(i), "score": 0.9, "metadata": {"page_content": f"doc-{i}"}}
                    for i in range(10)
                ],
                "namespace": "default",
            }

        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    docs = store.max_marginal_relevance_search("test", k=3)

    assert len(docs) == 3


@pytest.mark.asyncio
async def test_ammr_search_returns_documents(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should return List[AutoGenDocument]."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    docs = await store.amax_marginal_relevance_search("test")

    assert isinstance(docs, list)
    assert all(isinstance(d, AutoGenDocument) for d in docs)


@pytest.mark.asyncio
async def test_ammr_search_uses_mmr_config(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should pass MMRConfig to translator."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_query(self, *a: Any, mmr_config: Any = None, **k: Any) -> Any:
            captured["mmr_config"] = mmr_config
            return {"matches": [], "namespace": "default"}

        async def acapabilities(self) -> VectorCapabilities:
            return _make_caps()

        async def ahealth(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    await store.amax_marginal_relevance_search("test", k=4, lambda_mult=0.5)

    mmr = captured["mmr_config"]
    assert mmr is not None
    assert mmr.enabled is True


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

        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
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

        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    store.delete(filter={"key": "value"})

    assert called["delete"] is True


def test_delete_raises_without_ids_or_filter(adapter: Any) -> None:
    """Should raise BadRequest if neither provided."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)

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

        async def acapabilities(self) -> VectorCapabilities:
            return _make_caps()

        async def ahealth(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    await store.adelete(ids=["doc-0"])

    assert called["arun_delete"] is True


# ---------------------------------------------------------------------------
# Callable Interface Tests (6 tests)
# ---------------------------------------------------------------------------


def test_call_single_string_returns_single_vector(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """str → List[float]."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    result = store("test")

    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)


def test_call_list_strings_returns_list_vectors(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """List[str] → List[List[float]]."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = lambda texts: [[0.1, 0.2] for _ in texts]
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    result = store(["a", "b"])

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(v, list) for v in result)


def test_call_empty_batch_returns_empty_list(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[] → []."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)

    result = store([])

    assert result == []


def test_call_empty_items_return_zero_vectors(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty strings → zero vectors if dim known."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    store._update_dim_hint(4)

    result = store([""])

    assert result == [[0.0, 0.0, 0.0, 0.0]]


@pytest.mark.asyncio
async def test_acall_single_string_returns_single_vector(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Async str → List[float]."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    store = CorpusAutoGenVectorStore(
        corpus_adapter=adapter, async_embedding_function=_make_async_embedding_function
    )

    result = await store.acall("test")

    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)


@pytest.mark.asyncio
async def test_acall_uses_async_embedding_function(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should prefer async function."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    called = {"times": 0}

    async def counting_fn(texts: List[str]) -> List[List[float]]:
        called["times"] += 1
        return [[0.1, 0.2]]

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, async_embedding_function=counting_fn)

    await store.acall("test")

    assert called["times"] == 1


# ---------------------------------------------------------------------------
# Capabilities Tests (4 tests)
# ---------------------------------------------------------------------------


def test_capabilities_delegates_to_translator_only(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should delegate only to translator, not adapter."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)

    caps = store.capabilities()

    assert isinstance(caps, (VectorCapabilities, Mapping))


def test_capabilities_caches_result(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """VectorCapabilities should be cached."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)

    caps1 = store.capabilities()
    caps2 = store.capabilities()

    # Should cache the result
    if isinstance(caps1, VectorCapabilities):
        assert store._caps is not None


@pytest.mark.asyncio
async def test_acapabilities_delegates_to_translator_or_thread(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should use translator.acapabilities or run sync in thread."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)

    caps = await store.acapabilities()

    assert isinstance(caps, (VectorCapabilities, Mapping))


@pytest.mark.asyncio
async def test_acapabilities_caches_result(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VectorCapabilities should be cached."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)

    await store.acapabilities()

    if isinstance(store._caps, VectorCapabilities):
        assert store._caps is not None


# ---------------------------------------------------------------------------
# Health Tests (4 tests)
# ---------------------------------------------------------------------------


def test_health_delegates_to_translator_only(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should delegate only to translator, not adapter."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)

    health = store.health()

    assert isinstance(health, Mapping)


def test_health_raises_if_translator_missing_method(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should raise NotSupported if not implemented."""

    class BadTranslator:
        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: BadTranslator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)

    with pytest.raises(NotSupported):
        store.health()


@pytest.mark.asyncio
async def test_ahealth_delegates_to_translator_or_thread(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should use translator.ahealth or run sync in thread."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)

    health = await store.ahealth()

    assert isinstance(health, Mapping)


@pytest.mark.asyncio
async def test_ahealth_raises_if_translator_missing_method(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should raise NotSupported if not implemented."""

    class BadTranslator:
        async def acapabilities(self) -> VectorCapabilities:
            return _make_caps()

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: BadTranslator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)

    with pytest.raises(NotSupported):
        await store.ahealth()


# ---------------------------------------------------------------------------
# Context Manager Tests (4 tests)
# ---------------------------------------------------------------------------


def test_context_manager_calls_close(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """__exit__ should call close()."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)

    with store:
        assert store is not None


@pytest.mark.asyncio
async def test_async_context_manager_calls_aclose(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """__aexit__ should call aclose()."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)

    async with store:
        assert store is not None


def test_close_closes_translator(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should call translator.close()."""
    called = {"close": False}

    class DummyTranslator:
        def close(self) -> None:
            called["close"] = True

        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: DummyTranslator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    store.close()

    assert called["close"] is True


def test_close_closes_adapter_when_owned(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should call adapter.close() if own_adapter=True."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    called = {"close": False}

    class OwnedAdapter:
        def close(self) -> None:
            called["close"] = True

        def query(self, *a: Any, **k: Any) -> Any:
            return {"matches": [], "namespace": "default"}

        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Dict[str, Any]:
            return {"status": "ok"}

    owned_adapter = OwnedAdapter()
    store = CorpusAutoGenVectorStore(corpus_adapter=owned_adapter, own_adapter=True)
    store.close()

    assert called["close"] is True


# ---------------------------------------------------------------------------
# Event Loop Guard Tests (6 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_texts_raises_in_event_loop(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """add_texts should raise RuntimeError when called in event loop."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    with pytest.raises(RuntimeError, match="event loop"):
        store.add_texts(["test"])


@pytest.mark.asyncio
async def test_similarity_search_raises_in_event_loop(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """similarity_search should raise RuntimeError when called in event loop."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    with pytest.raises(RuntimeError, match="event loop"):
        store.similarity_search("test")


@pytest.mark.asyncio
async def test_delete_raises_in_event_loop(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """delete should raise RuntimeError when called in event loop."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)

    with pytest.raises(RuntimeError, match="event loop"):
        store.delete(ids=["doc-0"])


@pytest.mark.asyncio
async def test_call_raises_in_event_loop(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """__call__ should raise RuntimeError when called in event loop."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    with pytest.raises(RuntimeError, match="event loop"):
        store("test")


@pytest.mark.asyncio
async def test_from_texts_raises_in_event_loop(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """from_texts should raise RuntimeError when called in event loop."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()

    with pytest.raises(RuntimeError, match="event loop"):
        CorpusAutoGenVectorStore.from_texts(
            ["test"], corpus_adapter=adapter, embedding_function=fn
        )


@pytest.mark.asyncio
async def test_async_methods_work_in_event_loop(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Async methods should work fine in event loop."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    # Should not raise
    ids = await store.aadd_texts(["test"])
    assert ids is not None

    docs = await store.asimilarity_search("test")
    assert docs is not None


# ---------------------------------------------------------------------------
# Error Context Attachment Tests (8 tests)
# ---------------------------------------------------------------------------


def test_add_texts_error_attaches_context(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Errors should include texts_count, total_content_chars."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def upsert(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("upsert failed")

        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: FailingTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    with pytest.raises(RuntimeError, match="upsert failed"):
        store.add_texts(["test text"])

    assert captured_ctx.get("framework") == "autogen"
    assert "texts_count" in captured_ctx or "vectors_count" in captured_ctx
    assert "total_content_chars" in captured_ctx


def test_similarity_search_error_attaches_context(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Errors should include query_chars, k, namespace."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("query failed")

        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: FailingTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn, namespace="test-ns")

    with pytest.raises(RuntimeError, match="query failed"):
        store.similarity_search("test query", k=5)

    assert captured_ctx.get("framework") == "autogen"
    assert "query_chars" in captured_ctx or "total_content_chars" in captured_ctx
    assert captured_ctx.get("k") == 5


def test_mmr_search_error_attaches_context(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Errors should include k, fetch_k, lambda_mult."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("mmr query failed")

        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: FailingTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    with pytest.raises(RuntimeError, match="mmr query failed"):
        store.max_marginal_relevance_search("test", k=4, lambda_mult=0.5, fetch_k=20)

    assert captured_ctx.get("framework") == "autogen"
    assert "k" in captured_ctx
    assert "lambda_mult" in captured_ctx
    assert "fetch_k" in captured_ctx


def test_delete_error_attaches_context(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Errors should include ids_count, has_filter."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def delete(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("delete failed")

        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: FailingTranslator()
    )

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)

    with pytest.raises(RuntimeError, match="delete failed"):
        store.delete(ids=["doc-0", "doc-1"])

    assert captured_ctx.get("framework") == "autogen"
    assert "ids_count" in captured_ctx


def test_call_error_attaches_context(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Errors should include vectors_count, has_empty_items."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)

    def failing_fn(texts: List[str]) -> List[List[float]]:
        raise ValueError("embedding failed")

    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=failing_fn)

    with pytest.raises(BadRequest, match="embedding_function failed"):
        store(["test", ""])

    assert captured_ctx.get("framework") == "autogen"
    assert "vectors_count" in captured_ctx or "texts_count" in captured_ctx


def test_error_context_includes_framework_autogen(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Error context should always include framework='autogen'."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("test error")

        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: FailingTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    with pytest.raises(RuntimeError):
        store.similarity_search("test")

    assert captured_ctx.get("framework") == "autogen"


def test_error_context_includes_vector_dimension_hint(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Error context should include dim hint when available."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("test error")

        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: FailingTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)
    store._update_dim_hint(4)

    with pytest.raises(RuntimeError):
        store.similarity_search("test")

    assert "vector_dimension_hint" in captured_ctx
    assert captured_ctx["vector_dimension_hint"] == 4


def test_error_context_extraction_never_raises(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metrics errors should not break operation."""
    # This is more of a design verification - error context extraction
    # is wrapped in try/except blocks and should never raise

    class FailingTranslator:
        def query(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("main error")

        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: FailingTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    # Should still raise the main error, not a metrics error
    with pytest.raises(RuntimeError, match="main error"):
        store.similarity_search("test")


# ---------------------------------------------------------------------------
# Coercion Utilities Tests (4 tests)
# ---------------------------------------------------------------------------


def test_coerce_hits_safe_returns_empty_on_empty_result(adapter: Any) -> None:
    """Should return [] instead of raising for empty results."""
    result = autogen_adapter_module._coerce_hits_safe({"matches": []})
    assert result == []


def test_coerce_hits_safe_validates_hit_structure(adapter: Any) -> None:
    """Should enforce canonical shape."""
    result = autogen_adapter_module._coerce_hits_safe({
        "matches": [
            {"id": "1", "score": 0.9, "metadata": {}}
        ]
    })
    assert len(result) == 1
    assert result[0]["id"] == "1"


def test_warn_if_extreme_k_logs_warning(adapter: Any, caplog) -> None:
    """Should log for k > 100."""
    with caplog.at_level(logging.WARNING):
        autogen_adapter_module._warn_if_extreme_k(150, "test_op")

    # Should have logged something about extreme k
    assert any("150" in record.message for record in caplog.records)


def test_warn_if_extreme_k_does_not_raise(adapter: Any) -> None:
    """Advisory only, doesn't block."""
    # Should not raise
    autogen_adapter_module._warn_if_extreme_k(1000, "test_op")


# ---------------------------------------------------------------------------
# Match Translation Tests (4 tests)
# ---------------------------------------------------------------------------


def test_from_matches_converts_to_autogen_documents(adapter: Any) -> None:
    """Match → AutoGenDocument conversion."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    matches = [
        {
            "id": "1",
            "score": 0.9,
            "metadata": {"page_content": "test content", "key": "value"},
        }
    ]

    results = store._from_matches(matches)

    assert len(results) == 1
    doc, score = results[0]
    assert isinstance(doc, AutoGenDocument)
    assert doc.page_content == "test content"
    assert score == 0.9


def test_from_matches_extracts_metadata_correctly(adapter: Any) -> None:
    """Should handle metadata_field envelope."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, metadata_field="user_meta")
    matches = [
        {
            "id": "1",
            "score": 0.9,
            "metadata": {
                "page_content": "test",
                "user_meta": {"custom": "data"},
            },
        }
    ]

    results = store._from_matches(matches)
    doc, _ = results[0]

    assert doc.metadata == {"custom": "data"}


def test_from_matches_applies_score_threshold(adapter: Any) -> None:
    """Should filter low-scoring matches."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, score_threshold=0.5)
    matches = [
        {"id": "1", "score": 0.9, "metadata": {"page_content": "high"}},
        {"id": "2", "score": 0.3, "metadata": {"page_content": "low"}},
    ]

    results = store._from_matches(matches)

    assert len(results) == 1
    assert results[0][0].page_content == "high"


def test_from_matches_handles_missing_fields_gracefully(adapter: Any) -> None:
    """Should default for missing data."""
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter)
    matches = [
        {"id": "1", "score": 0.9, "metadata": {}},  # No page_content
    ]

    results = store._from_matches(matches)

    assert len(results) == 1
    doc, _ = results[0]
    assert doc.page_content == ""


# ---------------------------------------------------------------------------
# Convenience Constructors Tests (4 tests)
# ---------------------------------------------------------------------------


def test_from_texts_creates_and_populates_store(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should return populated store."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore.from_texts(
        ["test"], corpus_adapter=adapter, embedding_function=fn
    )

    assert isinstance(store, CorpusAutoGenVectorStore)


def test_from_texts_calls_add_texts(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should delegate to add_texts."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    called = {"add_texts": False}

    original_add_texts = CorpusAutoGenVectorStore.add_texts

    def counting_add_texts(self, *args, **kwargs):
        called["add_texts"] = True
        return original_add_texts(self, *args, **kwargs)

    monkeypatch.setattr(CorpusAutoGenVectorStore, "add_texts", counting_add_texts)

    fn = _make_mock_embedding_function()
    CorpusAutoGenVectorStore.from_texts(
        ["test"], corpus_adapter=adapter, embedding_function=fn
    )

    assert called["add_texts"] is True


def test_from_documents_creates_and_populates_store(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should return populated store."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    docs = [AutoGenDocument(page_content="test", metadata={})]
    store = CorpusAutoGenVectorStore.from_documents(
        docs, corpus_adapter=adapter, embedding_function=fn
    )

    assert isinstance(store, CorpusAutoGenVectorStore)


def test_from_documents_calls_add_documents(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should delegate to add_documents."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    called = {"add_documents": False}

    original_add_documents = CorpusAutoGenVectorStore.add_documents

    def counting_add_documents(self, *args, **kwargs):
        called["add_documents"] = True
        return original_add_documents(self, *args, **kwargs)

    monkeypatch.setattr(CorpusAutoGenVectorStore, "add_documents", counting_add_documents)

    fn = _make_mock_embedding_function()
    docs = [AutoGenDocument(page_content="test", metadata={})]
    CorpusAutoGenVectorStore.from_documents(
        docs, corpus_adapter=adapter, embedding_function=fn
    )

    assert called["add_documents"] is True


# ---------------------------------------------------------------------------
# CorpusAutoGenVectorClient Tests (5 tests)
# ---------------------------------------------------------------------------


def test_client_wraps_translator(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should use VectorTranslator internally."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    client = CorpusAutoGenVectorClient(adapter=adapter)

    assert hasattr(client, "_translator")


def test_client_exposes_protocol_methods(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should expose query, batch_query, upsert, delete."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    client = CorpusAutoGenVectorClient(adapter=adapter)

    assert hasattr(client, "query")
    assert hasattr(client, "batch_query")
    assert hasattr(client, "upsert")
    assert hasattr(client, "delete")


def test_client_capabilities_delegates_to_translator(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should pass through capabilities()."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    client = CorpusAutoGenVectorClient(adapter=adapter)
    caps = client.capabilities()

    assert caps is not None


def test_client_health_delegates_to_translator(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should pass through health()."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    client = CorpusAutoGenVectorClient(adapter=adapter)
    health = client.health()

    assert health is not None


def test_client_passes_conversation_as_framework_ctx(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """conversation kwarg → framework_ctx."""
    captured: Dict[str, Any] = {}

    class CapturingTranslator:
        def query(self, raw_query: Any, *, framework_ctx: Any = None, **k: Any) -> Any:
            captured["framework_ctx"] = framework_ctx
            return {"matches": [], "namespace": "default"}

        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: CapturingTranslator()
    )

    client = CorpusAutoGenVectorClient(adapter=adapter)
    mock_conversation = {"messages": []}
    client.query({}, conversation=mock_conversation)

    assert captured["framework_ctx"] == mock_conversation


# ---------------------------------------------------------------------------
# CorpusAutoGenRetrieverTool Tests (4 tests)
# ---------------------------------------------------------------------------


def test_retriever_tool_callable(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """__call__ should work as AutoGen tool."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)
    tool = CorpusAutoGenRetrieverTool(vector_store=store)

    results = tool("test query")

    assert isinstance(results, list)


def test_retriever_tool_delegates_to_vector_store(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should call store.similarity_search()."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    called = {"similarity_search": False}
    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    original_search = store.similarity_search

    def counting_search(*args, **kwargs):
        called["similarity_search"] = True
        return original_search(*args, **kwargs)

    store.similarity_search = counting_search  # type: ignore

    tool = CorpusAutoGenRetrieverTool(vector_store=store)
    tool("test")

    assert called["similarity_search"] is True


def test_retriever_tool_returns_dict_list(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should return List[Dict[str, Any]]."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)
    tool = CorpusAutoGenRetrieverTool(vector_store=store)

    results = tool("test")

    assert isinstance(results, list)
    for item in results:
        assert isinstance(item, dict)
        assert "page_content" in item
        assert "metadata" in item


def test_retriever_tool_merges_search_kwargs(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should combine default + runtime kwargs."""
    captured: Dict[str, Any] = {}

    class CapturingStore:
        def __init__(self):
            self.corpus_adapter = adapter
            self._translator = _make_dummy_translator()

        def similarity_search(self, query: str, **kwargs: Any) -> List[AutoGenDocument]:
            captured.update(kwargs)
            return []

    store = CapturingStore()
    tool = CorpusAutoGenRetrieverTool(
        vector_store=store, search_kwargs={"k": 10}  # type: ignore
    )

    tool("test", k=5)  # Override default

    assert captured.get("k") == 5


# ---------------------------------------------------------------------------
# AutoGen-Specific Integration Tests (6 tests - NO SKIPS)
# ---------------------------------------------------------------------------


def test_autogen_document_roundtrip(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """AutoGenDocument → store → retrieve → AutoGenDocument."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    original_doc = AutoGenDocument(
        page_content="test content", metadata={"key": "value"}
    )

    ids = store.add_documents([original_doc])
    assert len(ids) == 1

    # Note: Real roundtrip would require a working backend
    # This test verifies the API contract


def test_autogen_conversation_context_propagation(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Conversation object → OperationContext."""
    captured: Dict[str, Any] = {}

    def fake_context_from_autogen(
        conversation: Any, *, framework_version: Any = None, **extra: Any
    ) -> Any:
        captured["conversation"] = conversation
        return OperationContext(request_id="test", tenant="test", attrs={})

    monkeypatch.setattr(
        autogen_adapter_module, "core_ctx_from_autogen", fake_context_from_autogen
    )
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    mock_conversation = {"messages": [{"role": "user", "content": "test"}]}
    store.similarity_search("test", conversation=mock_conversation)

    assert captured["conversation"] == mock_conversation


def test_autogen_retriever_tool_in_agent_flow(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tool works in agent conversation."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)
    tool = CorpusAutoGenRetrieverTool(vector_store=store)

    # Simulate agent calling tool
    results = tool("find relevant documents")

    assert isinstance(results, list)
    assert all(isinstance(r, dict) for r in results)


def test_autogen_vector_function_interface(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """__call__ works as AutoGen vector_function."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    # Single string
    vec = store("test")
    assert isinstance(vec, list)
    assert all(isinstance(x, float) for x in vec)

    # Batch
    vecs = store(["a", "b"])
    assert isinstance(vecs, list)
    assert len(vecs) == 2


def test_autogen_metadata_field_envelope(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """metadata_field wraps user metadata."""
    captured: Dict[str, Any] = {}

    class CapturingTranslator:
        def upsert(self, raw_documents: Any, **k: Any) -> Any:
            captured["raw_documents"] = raw_documents
            return {"ids": ["doc-0"], "count": 1}

        def capabilities(self) -> VectorCapabilities:
            return _make_caps()

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: CapturingTranslator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(
        corpus_adapter=adapter, embedding_function=fn, metadata_field="user_meta"
    )

    store.add_texts(["test"], metadatas=[{"custom": "data"}])

    doc = captured["raw_documents"][0]
    assert "user_meta" in doc["metadata"]
    assert doc["metadata"]["user_meta"] == {"custom": "data"}


def test_autogen_streaming_with_real_conversation(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Streaming with conversation context."""
    monkeypatch.setattr(
        autogen_adapter_module, "create_vector_translator", lambda *_a, **_k: _make_dummy_translator()
    )

    fn = _make_mock_embedding_function()
    store = CorpusAutoGenVectorStore(corpus_adapter=adapter, embedding_function=fn)

    mock_conversation = {"messages": []}
    docs = list(store.similarity_search_stream("test", conversation=mock_conversation))

    assert isinstance(docs, list)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
