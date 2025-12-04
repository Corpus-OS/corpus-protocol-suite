# tests/vector/frameworks/test_semantickernel_vector_adapter.py

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Mapping as TypingMapping, Sequence

import inspect

import pytest

import corpus_sdk.vector.framework_adapters.semantic_kernel as sk_vector_module
from corpus_sdk.vector.framework_adapters.semantic_kernel import (
    CorpusSemanticKernelVectorPlugin,
    CorpusSemanticKernelVectorStore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(adapter: Any, **kwargs: Any) -> CorpusSemanticKernelVectorStore:
    """Construct a CorpusSemanticKernelVectorStore from a generic adapter."""
    return CorpusSemanticKernelVectorStore(corpus_adapter=adapter, **kwargs)


def _make_plugin(
    store: CorpusSemanticKernelVectorStore,
    **kwargs: Any,
) -> CorpusSemanticKernelVectorPlugin:
    """Construct a CorpusSemanticKernelVectorPlugin from a store."""
    return CorpusSemanticKernelVectorPlugin(vector_store=store, **kwargs)


def _mock_translator_with_capture(
    captured: Dict[str, Any],
    method_name: str,
    return_value: Any,
) -> Any:
    """Sync translator that captures args/kwargs of a single method."""

    class MockTranslator:
        def __getattr__(self, name: str) -> Any:
            if name == method_name:

                def method(*args: Any, **kwargs: Any) -> Any:
                    if args:
                        captured["args"] = args
                    captured.update(kwargs)
                    return return_value

                return method
            raise AttributeError(name)

    return MockTranslator()


def _mock_async_translator_with_capture(
    captured: Dict[str, Any],
    method_name: str,
    return_value: Any,
) -> Any:
    """Async translator that captures args/kwargs of a single async method."""

    class MockTranslator:
        def __getattr__(self, name: str) -> Any:
            if name == method_name:

                async def method(*args: Any, **kwargs: Any) -> Any:
                    if args:
                        captured["args"] = args
                    captured.update(kwargs)
                    return return_value

                return method
            raise AttributeError(name)

    return MockTranslator()


# ---------------------------------------------------------------------------
# Translator construction / framework wiring
# ---------------------------------------------------------------------------


def test_default_translator_uses_semantickernel_framework(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    _translator cached_property should construct a VectorTranslator with:
    - framework='semantic_kernel'
    - translator instance of DefaultVectorFrameworkTranslator
    - adapter=the underlying corpus_adapter
    """
    captured: Dict[str, Any] = {}

    class DummyVectorTranslator:
        def __init__(self, *, adapter: Any, framework: str, translator: Any) -> None:
            captured["adapter"] = adapter
            captured["framework"] = framework
            captured["translator"] = translator

    class DummyFrameworkTranslator:
        pass

    monkeypatch.setattr(
        sk_vector_module,
        "VectorTranslator",
        DummyVectorTranslator,
    )
    monkeypatch.setattr(
        sk_vector_module,
        "DefaultVectorFrameworkTranslator",
        DummyFrameworkTranslator,
    )

    store = _make_store(adapter)
    _ = store._translator  # noqa: SLF001

    assert captured["adapter"] is adapter
    assert captured["framework"] == "semantic_kernel"
    assert isinstance(captured["translator"], DummyFrameworkTranslator)


def test_translator_is_cached_property(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    _translator is a cached_property; accessing it twice should return the same instance.
    """
    created: List[Any] = []

    class DummyVectorTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            created.append(self)

    monkeypatch.setattr(
        sk_vector_module,
        "VectorTranslator",
        DummyVectorTranslator,
    )

    store = _make_store(adapter)
    first = store._translator  # noqa: SLF001
    second = store._translator  # noqa: SLF001
    assert first is second
    assert len(created) == 1


# ---------------------------------------------------------------------------
# Context translation / from_semantic_kernel & from_dict mapping
# ---------------------------------------------------------------------------


def test_semantickernel_context_and_context_dict_passed_to_core_ctx(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Verify that sk_context/sk_settings and context_dict are passed through to
    context_from_semantic_kernel / context_from_dict respectively.
    """
    captured: Dict[str, Any] = {}

    class DummyOperationContext:
        pass

    monkeypatch.setattr(
        sk_vector_module,
        "OperationContext",
        DummyOperationContext,
    )

    def fake_context_from_semantic_kernel(
        context: Any,
        *,
        settings: Any = None,
    ) -> Any:
        captured["sk_context"] = context
        captured["sk_settings"] = settings
        return DummyOperationContext()

    def fake_context_from_dict(d: TypingMapping[str, Any]) -> Any:
        captured["context_dict"] = dict(d)
        return DummyOperationContext()

    monkeypatch.setattr(
        sk_vector_module,
        "context_from_semantic_kernel",
        fake_context_from_semantic_kernel,
    )
    monkeypatch.setattr(
        sk_vector_module,
        "context_from_dict",
        fake_context_from_dict,
    )

    store = _make_store(adapter)

    # 1) SK context path
    ctx = store._build_ctx(
        sk_context={"user": "u1"},
        sk_settings={"temperature": 0.1},
    )
    assert isinstance(ctx, DummyOperationContext)
    assert captured["sk_context"] == {"user": "u1"}
    assert captured["sk_settings"] == {"temperature": 0.1"}

    # 2) Dict path (no SK args)
    captured.clear()
    ctx2 = store._build_ctx(context_dict={"request_id": "abc"})
    assert isinstance(ctx2, DummyOperationContext)
    assert captured["context_dict"] == {"request_id": "abc"}


def test_build_ctx_returns_existing_operation_context(adapter: Any) -> None:
    """
    If ctx is already an OperationContext instance, _build_ctx returns it unchanged.
    """
    store = _make_store(adapter)

    class DummyOperationContext:
        pass

    # Override OperationContext for isinstance checks
    sk_vector_module.OperationContext = DummyOperationContext  # type: ignore[assignment]

    existing = DummyOperationContext()
    out = store._build_ctx(ctx=existing)
    assert out is existing


# ---------------------------------------------------------------------------
# Configuration validation helpers
# ---------------------------------------------------------------------------


def test_validate_batch_size_bounds(adapter: Any) -> None:
    store = _make_store(adapter, batch_size=10)
    assert store.batch_size == 10

    with pytest.raises(ValueError):
        _make_store(adapter, batch_size=0)


def test_validate_default_top_k_bounds(adapter: Any) -> None:
    store = _make_store(adapter, default_top_k=5)
    assert store.default_top_k == 5

    with pytest.raises(ValueError):
        _make_store(adapter, default_top_k=0)


def test_validate_score_threshold_bounds(adapter: Any) -> None:
    store = _make_store(adapter, score_threshold=0.5)
    assert store.score_threshold == 0.5

    with pytest.raises(ValueError):
        _make_store(adapter, score_threshold=1.5)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def test_ensure_embeddings_uses_passed_embeddings(adapter: Any) -> None:
    store = _make_store(adapter)
    embs = [[0.1, 0.2], [0.3, 0.4]]
    out = store._ensure_embeddings(["a", "b"], embs)
    assert out is embs


def test_ensure_embeddings_length_mismatch_attaches_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    _ensure_embeddings should raise BAD_EMBEDDINGS and call attach_context
    when lengths mismatch.
    """
    store = _make_store(adapter)
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured.update(ctx)

    monkeypatch.setattr(
        sk_vector_module,
        "attach_context",
        fake_attach_context,
    )

    with pytest.raises(Exception) as exc_info:
        store._ensure_embeddings(["a", "b"], embeddings=[[0.1, 0.2]])

    err = exc_info.value
    assert getattr(err, "code", None) == "BAD_EMBEDDINGS"
    assert captured.get("framework") == "semantic_kernel"
    assert captured.get("operation") == "ensure_embeddings"


@pytest.mark.asyncio
async def test_ensure_embeddings_async_prefers_async_embedding(
    adapter: Any,
) -> None:
    async def aembed(texts: List[str]) -> List[List[float]]:
        return [[float(len(t))] for t in texts]

    store = _make_store(adapter, async_embedding_function=aembed)
    embs = await store._ensure_embeddings_async(["hi", "there"], embeddings=None)
    assert embs == [[2.0], [5.0]]


def test_embed_query_uses_embedding_function_when_no_embedding_given(adapter: Any) -> None:
    def embed(texts: List[str]) -> List[List[float]]:
        return [[1.0, 2.0] for _ in texts]

    store = _make_store(adapter, embedding_function=embed)
    vec = store._embed_query("hello")
    assert vec == [1.0, 2.0]


@pytest.mark.asyncio
async def test_embed_query_async_uses_async_function_when_available(adapter: Any) -> None:
    async def aembed(texts: List[str]) -> List[List[float]]:
        return [[3.0] for _ in texts]

    store = _make_store(adapter, async_embedding_function=aembed)
    vec = await store._embed_query_async("hi")
    assert vec == [3.0]


# ---------------------------------------------------------------------------
# Capabilities helpers
# ---------------------------------------------------------------------------


class DummyCaps:
    def __init__(
        self,
        max_top_k: int | None = None,
        supports_metadata_filtering: bool = True,
        supports_namespaces: bool = True,
        max_batch_size: int | None = None,
    ) -> None:
        self.max_top_k = max_top_k
        self.max_batch_size = max_batch_size
        self.supports_metadata_filtering = supports_metadata_filtering
        self.supports_namespaces = supports_namespaces


def test_get_capabilities_uses_translator_and_caches(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    translator = _mock_translator_with_capture(
        captured,
        method_name="capabilities",
        return_value=DummyCaps(max_top_k=42, max_batch_size=256),
    )

    def fake_vector_translator(*_: Any, **__: Any) -> Any:
        return translator

    monkeypatch.setattr(
        sk_vector_module,
        "VectorTranslator",
        fake_vector_translator,
    )

    store = _make_store(adapter)
    caps1 = store.get_capabilities()
    caps2 = store.get_capabilities()
    assert caps1 is caps2
    assert caps1.max_top_k == 42
    assert caps1.max_batch_size == 256


@pytest.mark.asyncio
async def test_aget_capabilities_uses_async_translator_and_caches(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    translator = _mock_async_translator_with_capture(
        captured,
        method_name="arun_capabilities",
        return_value=DummyCaps(max_top_k=10),
    )

    def fake_vector_translator(*_: Any, **__: Any) -> Any:
        return translator

    monkeypatch.setattr(
        sk_vector_module,
        "VectorTranslator",
        fake_vector_translator,
    )

    store = _make_store(adapter)
    caps1 = await store.aget_capabilities()
    caps2 = await store.aget_capabilities()
    assert caps1 is caps2
    assert caps1.max_top_k == 10


# ---------------------------------------------------------------------------
# Query / delete parameter validation
# ---------------------------------------------------------------------------


def test_validate_query_params_respects_max_top_k(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    translator = _mock_translator_with_capture(
        {},
        method_name="capabilities",
        return_value=DummyCaps(max_top_k=5),
    )

    monkeypatch.setattr(
        sk_vector_module,
        "VectorTranslator",
        lambda *_, **__: translator,
    )

    store = _make_store(adapter)
    store._get_caps_sync()  # prime cache

    with pytest.raises(Exception) as exc_info:
        store._validate_query_params_sync(
            top_k=10,
            namespace=None,
            filter=None,
        )

    err = exc_info.value
    assert getattr(err, "code", None) == "BAD_TOP_K"


def test_validate_delete_params_requires_ids_or_filter(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    translator = _mock_translator_with_capture(
        {},
        method_name="capabilities",
        return_value=DummyCaps(),
    )

    monkeypatch.setattr(
        sk_vector_module,
        "VectorTranslator",
        lambda *_, **__: translator,
    )

    store = _make_store(adapter)
    store._get_caps_sync()

    with pytest.raises(Exception) as exc_info:
        store._validate_delete_params_sync(
            ids=None,
            namespace=None,
            filter=None,
        )

    err = exc_info.value
    assert getattr(err, "code", None) == "BAD_DELETE"


# ---------------------------------------------------------------------------
# Core vector dataclasses used in tests
# ---------------------------------------------------------------------------


class DummyVector:
    def __init__(
        self,
        metadata: Dict[str, Any],
        vector: Sequence[float] | None = None,
    ) -> None:
        self.metadata = metadata
        self.vector = list(vector or [])


class DummyMatch:
    def __init__(
        self,
        metadata: Dict[str, Any],
        score: float,
        vector: Sequence[float] | None = None,
    ) -> None:
        self.vector = DummyVector(metadata=metadata, vector=vector)
        self.score = score


class DummyQueryResult:
    def __init__(self, matches: List[DummyMatch]) -> None:
        self.matches = matches


class DummyQueryChunk:
    def __init__(self, matches: List[DummyMatch]) -> None:
        self.matches = matches


# ---------------------------------------------------------------------------
# Upsert wiring / partial failure handling
# ---------------------------------------------------------------------------


class DummyUpsertResult:
    def __init__(
        self,
        upserted_count: int,
        failed_count: int,
        failures: List[Any] | None = None,
    ) -> None:
        self.upserted_count = upserted_count
        self.failed_count = failed_count
        self.failures = failures or []


def test_add_texts_builds_vectors_and_calls_translator_upsert(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    add_texts() should build a raw request mapping for upsert and pass it to
    translator.upsert along with framework_ctx.
    """
    captured: Dict[str, Any] = {}

    translator = _mock_translator_with_capture(
        captured,
        method_name="upsert",
        return_value=DummyUpsertResult(upserted_count=2, failed_count=0),
    )

    def fake_vector_translator(*_: Any, **__: Any) -> Any:
        return translator

    monkeypatch.setattr(
        sk_vector_module,
        "VectorTranslator",
        fake_vector_translator,
    )

    def embed(texts: List[str]) -> List[List[float]]:
        return [[float(len(t))] for t in texts]

    store = _make_store(
        adapter,
        embedding_function=embed,
        namespace="ns-add",
    )

    ids = store.add_texts(
        ["a", "bc"],
        metadatas=[{"x": 1}, {"y": 2}],
        ids=["id1", "id2"],
    )
    assert ids == ["id1", "id2"]

    assert "args" in captured
    raw_request = captured["args"][0]
    assert raw_request["namespace"] == "ns-add"
    vectors = raw_request["vectors"]
    assert len(vectors) == 2
    assert vectors[0].id == "id1"
    assert vectors[0].metadata["page_content"] == "a"
    assert vectors[0].metadata["id"] == "id1"


def test_handle_partial_upsert_failure_raises_when_all_fail(
    adapter: Any,
) -> None:
    store = _make_store(adapter)

    result = DummyUpsertResult(
        upserted_count=0,
        failed_count=3,
        failures=[{"reason": "bad"}],
    )
    with pytest.raises(Exception) as exc_info:
        store._handle_partial_upsert_failure(
            result,
            total_texts=3,
            namespace="ns",
        )

    err = exc_info.value
    assert getattr(err, "code", None) == "BATCH_UPSERT_FAILED"


# ---------------------------------------------------------------------------
# Similarity search (sync / async) + streaming
# ---------------------------------------------------------------------------


def test_similarity_search_basic(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Basic smoke test: similarity_search() should return AI-optimized dicts and
    respect score_threshold.
    """
    matches = [
        DummyMatch({"page_content": "hello", "id": "1"}, score=0.9),
        DummyMatch({"page_content": "world", "id": "2"}, score=0.1),
    ]
    result = DummyQueryResult(matches)

    captured: Dict[str, Any] = {}

    translator = _mock_translator_with_capture(
        captured,
        method_name="query",
        return_value=result,
    )

    def fake_vector_translator(*_: Any, **__: Any) -> Any:
        return translator

    monkeypatch.setattr(
        sk_vector_module,
        "VectorTranslator",
        fake_vector_translator,
    )
    # Make QueryResult isinstance checks pass
    monkeypatch.setattr(
        sk_vector_module,
        "QueryResult",
        DummyQueryResult,
    )

    def embed(texts: List[str]) -> List[List[float]]:
        return [[0.0] * 3 for _ in texts]

    store = _make_store(
        adapter,
        embedding_function=embed,
        score_threshold=0.5,
    )

    docs = store.similarity_search("hi", k=4)
    assert len(docs) == 1
    doc = docs[0]
    assert doc["content"] == "hello"
    # id/text stripped from metadata
    assert doc["metadata"] == {}
    assert "confidence" in doc
    assert doc["source"] == "vector_database"


@pytest.mark.asyncio
async def test_asimilarity_search_basic(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    matches = [
        DummyMatch({"page_content": "foo", "id": "1"}, score=0.7),
    ]
    result = DummyQueryResult(matches)

    captured: Dict[str, Any] = {}

    translator = _mock_async_translator_with_capture(
        captured,
        method_name="arun_query",
        return_value=result,
    )

    def fake_vector_translator(*_: Any, **__: Any) -> Any:
        return translator

    monkeypatch.setattr(
        sk_vector_module,
        "VectorTranslator",
        fake_vector_translator,
    )
    monkeypatch.setattr(
        sk_vector_module,
        "QueryResult",
        DummyQueryResult,
    )

    async def aembed(texts: List[str]) -> List[List[float]]:
        return [[1.0] * 2 for _ in texts]

    store = _make_store(
        adapter,
        async_embedding_function=aembed,
    )

    docs = await store.asimilarity_search("q", k=2)
    assert len(docs) == 1
    assert docs[0]["content"] == "foo"


def test_similarity_search_stream_invalid_chunk_attaches_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    similarity_search_stream() should treat non-QueryChunk chunks as an error
    and attach semantic_kernel error context.
    """
    invalid_chunk = object()
    captured_ctx: Dict[str, Any] = {}

    class DummyTranslator:
        def query_stream(
            self,
            *args: Any,
            **kwargs: Any,  # noqa: ARG002
        ):
            yield invalid_chunk

    def fake_vector_translator(*_: Any, **__: Any) -> Any:
        return DummyTranslator()

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured_ctx.update(ctx)

    monkeypatch.setattr(
        sk_vector_module,
        "VectorTranslator",
        fake_vector_translator,
    )
    monkeypatch.setattr(
        sk_vector_module,
        "QueryChunk",
        DummyQueryChunk,
    )
    monkeypatch.setattr(
        sk_vector_module,
        "attach_context",
        fake_attach_context,
    )

    def embed(texts: List[str]) -> List[List[float]]:
        return [[0.0] for _ in texts]

    store = _make_store(
        adapter,
        embedding_function=embed,
    )

    with pytest.raises(Exception):
        list(store.similarity_search_stream("hi"))

    assert captured_ctx.get("framework") == "semantic_kernel"
    assert "similarity_search_stream" in str(captured_ctx.get("operation", ""))


@pytest.mark.asyncio
async def test_asimilarity_search_stream_invalid_chunk_attaches_context_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    asimilarity_search_stream() should also validate/non-QueryChunk chunks and
    attach Semantic Kernel error context.
    """
    invalid_chunk = object()
    captured_ctx: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_query_stream(
            self,
            *args: Any,
            **kwargs: Any,  # noqa: ARG002
        ):
            async def gen():
                yield invalid_chunk

            return gen()

    def fake_vector_translator(*_: Any, **__: Any) -> Any:
        return DummyTranslator()

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured_ctx.update(ctx)

    monkeypatch.setattr(
        sk_vector_module,
        "VectorTranslator",
        fake_vector_translator,
    )
    monkeypatch.setattr(
        sk_vector_module,
        "QueryChunk",
        DummyQueryChunk,
    )
    monkeypatch.setattr(
        sk_vector_module,
        "attach_context",
        fake_attach_context,
    )

    async def aembed(texts: List[str]) -> List[List[float]]:
        return [[0.0] for _ in texts]

    store = _make_store(
        adapter,
        async_embedding_function=aembed,
    )

    aiter = store.asimilarity_search_stream("hi", k=2)
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    with pytest.raises(Exception):
        async for _ in aiter:  # noqa: B007
            pass

    assert captured_ctx.get("framework") == "semantic_kernel"
    assert "similarity_search_stream" in str(captured_ctx.get("operation", ""))


# ---------------------------------------------------------------------------
# MMR search
# ---------------------------------------------------------------------------


def test_mmr_search_rejects_invalid_lambda(adapter: Any) -> None:
    store = _make_store(adapter)

    with pytest.raises(Exception) as exc_info:
        store.max_marginal_relevance_search(
            "q",
            k=2,
            lambda_mult=1.5,
        )

    err = exc_info.value
    assert getattr(err, "code", None) == "BAD_MMR_LAMBDA"


def test_mmr_search_calls_mmr_selector(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    matches = [
        DummyMatch({"page_content": "a", "id": "1"}, score=0.9, vector=[1.0, 0.0]),
        DummyMatch({"page_content": "b", "id": "2"}, score=0.8, vector=[0.0, 1.0]),
    ]
    result = DummyQueryResult(matches)

    translator = _mock_translator_with_capture(
        {},
        method_name="query",
        return_value=result,
    )

    def fake_vector_translator(*_: Any, **__: Any) -> Any:
        return translator

    monkeypatch.setattr(
        sk_vector_module,
        "VectorTranslator",
        fake_vector_translator,
    )
    monkeypatch.setattr(
        sk_vector_module,
        "QueryResult",
        DummyQueryResult,
    )

    selected_indices: Dict[str, Any] = {}

    def fake_mmr_select_indices(
        self: CorpusSemanticKernelVectorStore,  # noqa: ARG001
        query_vec: Sequence[float],
        candidate_matches: List[Any],
        k: int,
        lambda_mult: float,
    ) -> List[int]:
        selected_indices["query_vec"] = list(query_vec)
        selected_indices["k"] = k
        selected_indices["lambda_mult"] = lambda_mult
        selected_indices["count"] = len(candidate_matches)
        return [1, 0]  # reverse order

    monkeypatch.setattr(
        CorpusSemanticKernelVectorStore,
        "_mmr_select_indices",
        fake_mmr_select_indices,
    )

    def embed(texts: List[str]) -> List[List[float]]:
        return [[1.0, 0.0] for _ in texts]

    store = _make_store(
        adapter,
        embedding_function=embed,
    )

    docs = store.max_marginal_relevance_search(
        "q",
        k=2,
        lambda_mult=0.7,
    )
    assert [d["content"] for d in docs] == ["b", "a"]
    assert selected_indices["k"] == 2
    assert selected_indices["lambda_mult"] == 0.7
    assert selected_indices["count"] == 2


# ---------------------------------------------------------------------------
# Delete wiring
# ---------------------------------------------------------------------------


def test_delete_builds_raw_request_and_calls_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    delete() should build the correct raw request mapping and call translator.delete.
    """
    captured: Dict[str, Any] = {}

    translator = _mock_translator_with_capture(
        captured,
        method_name="delete",
        return_value=None,
    )

    def fake_vector_translator(*_: Any, **__: Any) -> Any:
        return translator

    monkeypatch.setattr(
        sk_vector_module,
        "VectorTranslator",
        fake_vector_translator,
    )

    store = _make_store(adapter)
    store._caps = DummyCaps()  # bypass capabilities fetch

    store.delete(
        ids=["1", "2"],
        namespace="ns-del",
        filter=None,
    )

    assert "args" in captured
    raw_request = captured["args"][0]
    assert raw_request == {
        "namespace": "ns-del",
        "ids": ["1", "2"],
        "filter": None,
    }


@pytest.mark.asyncio
async def test_adelete_builds_raw_request_and_calls_translator_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    translator = _mock_async_translator_with_capture(
        captured,
        method_name="arun_delete",
        return_value=None,
    )

    def fake_vector_translator(*_: Any, **__: Any) -> Any:
        return translator

    monkeypatch.setattr(
        sk_vector_module,
        "VectorTranslator",
        fake_vector_translator,
    )

    store = _make_store(adapter)
    store._caps = DummyCaps()

    await store.adelete(
        ids=["x"],
        namespace="ns-adel",
        filter=None,
    )

    raw_request = captured["args"][0]
    assert raw_request == {
        "namespace": "ns-adel",
        "ids": ["x"],
        "filter": None,
    }


# ---------------------------------------------------------------------------
# Plugin: capabilities wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plugin_get_capabilities_uses_store_and_shapes_result(
    adapter: Any,
) -> None:
    caps = DummyCaps(
        max_top_k=50,
        max_batch_size=128,
        supports_metadata_filtering=True,
        supports_namespaces=True,
    )

    store = _make_store(adapter)
    store._caps = caps

    async def fake_aget_capabilities() -> Any:
        return caps

    store.aget_capabilities = fake_aget_capabilities  # type: ignore[assignment]

    plugin = _make_plugin(store)

    result = await plugin.get_capabilities()
    assert result["max_batch_size"] == 128
    assert result["max_top_k"] == 50
    assert result["supports_metadata_filtering"] is True
    assert result["supports_namespaces"] is True
    assert "description" in result


# ---------------------------------------------------------------------------
# Plugin: vector_search / MMR / store_document wiring + error mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plugin_vector_search_calls_store_and_returns_docs(
    adapter: Any,
) -> None:
    store = _make_store(adapter)

    async def fake_asimilarity_search(
        query: str,
        k: int,
        **kwargs: Any,  # noqa: ARG001
    ) -> List[Dict[str, Any]]:
        return [
            {
                "content": f"doc-{query}-{k}",
                "metadata": {},
                "confidence": 0.9,
                "source": "vector_database",
            }
        ]

    store.asimilarity_search = fake_asimilarity_search  # type: ignore[assignment]

    plugin = _make_plugin(
        store,
        framework_version="sk-fw-test",
    )

    docs = await plugin.vector_search(
        "what is corpus?",
        k=3,
        filter={"tag": "x"},
    )
    assert docs
    assert docs[0]["content"] == "doc-what is corpus?-3"


@pytest.mark.asyncio
async def test_plugin_vector_mmr_search_calls_store(
    adapter: Any,
) -> None:
    store = _make_store(adapter)

    async def fake_amax_marginal_relevance_search(
        query: str,
        k: int,
        lambda_mult: float,
        **kwargs: Any,  # noqa: ARG001
    ) -> List[Dict[str, Any]]:
        return [
            {
                "content": f"mmr-{query}-{k}-{lambda_mult}",
                "metadata": {},
                "confidence": 0.8,
                "source": "vector_database",
            }
        ]

    store.amax_marginal_relevance_search = fake_amax_marginal_relevance_search  # type: ignore[assignment]

    plugin = _make_plugin(store)

    docs = await plugin.vector_mmr_search(
        "q",
        k=2,
        lambda_mult=0.7,
    )
    assert docs[0]["content"] == "mmr-q-2-0.7"


@pytest.mark.asyncio
async def test_plugin_store_document_calls_aadd_texts_and_returns_id(
    adapter: Any,
) -> None:
    store = _make_store(adapter)

    async def fake_aadd_texts(
        texts: Iterable[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str] | None = None,
        **kwargs: Any,  # noqa: ARG001
    ) -> List[str]:
        assert list(texts) == ["hello"]
        assert metadatas == [{"foo": "bar"}]
        return ids or ["generated-id"]

    store.aadd_texts = fake_aadd_texts  # type: ignore[assignment]

    plugin = _make_plugin(store)
    doc_id = await plugin.store_document(
        content="hello",
        metadata={"foo": "bar"},
        document_id="explicit-id",
    )
    assert doc_id == "explicit-id"


@pytest.mark.asyncio
async def test_plugin_errors_wrapped_as_kernelfunctionexception(
    adapter: Any,
) -> None:
    store = _make_store(adapter)
    plugin = _make_plugin(store)

    class DummyBadRequest(Exception):
        def __init__(self, msg: str) -> None:
            super().__init__(msg)
            self.code = "BAD_REQUEST"

    async def failing_search(*_: Any, **__: Any) -> Any:
        raise DummyBadRequest("oops")

    store.asimilarity_search = failing_search  # type: ignore[assignment]

    KernelFunctionException = sk_vector_module.KernelFunctionException

    with pytest.raises(KernelFunctionException) as exc_info:
        await plugin.vector_search("q")

    msg = str(exc_info.value)
    # Error message should indicate a vector-related failure
    assert "Vector search failed" in msg or "Invalid vector operation" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
