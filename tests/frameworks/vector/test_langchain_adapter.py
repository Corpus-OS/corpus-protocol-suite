# tests/frameworks/vector/test_langchain_vector_adapter.py

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

import inspect
import math
import pytest

import corpus_sdk.vector.framework_adapters.langchain as langchain_vector_module
from corpus_sdk.vector.framework_adapters.langchain import (
    CorpusLangChainRetriever,
    CorpusLangChainVectorStore,
)
from corpus_sdk.vector.vector_base import (
    BadRequest,
    NotSupported,
    OperationContext,
    QueryChunk,
    QueryResult,
    UpsertResult,
    Vector,
    VectorAdapterError,
    VectorMatch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_embedding_fn(texts: List[str]) -> List[List[float]]:
    """Deterministic toy embedding function: 4-dim vector based on index."""
    embs: List[List[float]] = []
    for i, _ in enumerate(texts):
        embs.append([float(i), 0.0, 0.0, 1.0])
    return embs


async def _simple_async_embedding_fn(texts: List[str]) -> List[List[float]]:
    return _simple_embedding_fn(texts)


def _make_store(
    adapter: Any,
    *,
    with_embeddings: bool = True,
    **kwargs: Any,
) -> CorpusLangChainVectorStore:
    """
    Construct a CorpusLangChainVectorStore with a simple embedding function.
    """
    store_kwargs: Dict[str, Any] = {"corpus_adapter": adapter}
    if with_embeddings:
        store_kwargs["embedding_function"] = _simple_embedding_fn
    store_kwargs.update(kwargs)
    return CorpusLangChainVectorStore(**store_kwargs)


def _make_match(
    text: str,
    *,
    score: float = 0.9,
    doc_id: str = "doc-1",
    extra_meta: Optional[Dict[str, Any]] = None,
    vector_dim: int = 4,
) -> VectorMatch:
    """
    Helper to build a VectorMatch with metadata containing text/id and extras.
    """
    meta: Dict[str, Any] = {
        "page_content": text,
        "id": doc_id,
    }
    if extra_meta:
        meta.update(extra_meta)

    v = Vector(
        id=doc_id,
        vector=[float(i) for i in range(vector_dim)],
        metadata=meta,
        namespace="ns",
        text=None,
    )
    return VectorMatch(score=score, vector=v)


# ---------------------------------------------------------------------------
# Translator wiring
# ---------------------------------------------------------------------------


def test_default_translator_uses_langchain_framework_label(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    _translator cached_property should construct VectorTranslator with
    framework='langchain' and DefaultVectorFrameworkTranslator instance.
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def __init__(self, adapter_arg: Any, framework: str, translator: Any) -> None:
            captured["adapter"] = adapter_arg
            captured["framework"] = framework
            captured["translator"] = translator

        # minimal surface so calls don't crash if they sneak through
        def query(self, *args: Any, **kwargs: Any) -> QueryResult:  # noqa: ARG002
            return QueryResult(matches=[])

    monkeypatch.setattr(
        langchain_vector_module,
        "VectorTranslator",
        DummyTranslator,
    )

    store = _make_store(adapter)
    _ = store._translator  # noqa: SLF001

    assert captured["adapter"] is adapter
    assert captured["framework"] == "langchain"
    assert isinstance(
        captured["translator"],
        langchain_vector_module.DefaultVectorFrameworkTranslator,
    )


# ---------------------------------------------------------------------------
# Context translation / _build_ctx
# ---------------------------------------------------------------------------


def test_build_ctx_accepts_operation_context_passthrough(
    adapter: Any,
) -> None:
    """
    If config is already an OperationContext, _build_ctx should return it as-is.
    """
    store = _make_store(adapter)
    ctx = OperationContext()
    built = store._build_ctx(config=ctx)  # noqa: SLF001
    assert built is ctx


def test_build_ctx_prefers_ctx_from_dict_over_langchain(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    _build_ctx should first try ctx_from_dict and only fall back to
    ctx_from_langchain if that fails.
    """
    captured: Dict[str, Any] = {"dict_called": False, "lc_called": False}

    class DummyOC(OperationContext):
        pass

    def fake_from_dict(config: Mapping[str, Any]) -> OperationContext:
        captured["dict_called"] = True
        captured["dict_config"] = config
        return DummyOC()

    def fake_from_langchain(_: Mapping[str, Any]) -> OperationContext:
        captured["lc_called"] = True
        raise AssertionError("ctx_from_langchain should not be called in this path")

    monkeypatch.setattr(
        langchain_vector_module,
        "ctx_from_dict",
        fake_from_dict,
    )
    monkeypatch.setattr(
        langchain_vector_module,
        "ctx_from_langchain",
        fake_from_langchain,
    )

    translator_captured: Dict[str, Any] = {}

    class DummyTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        def query(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> QueryResult:
            translator_captured["raw_query"] = dict(raw_query)
            translator_captured["op_ctx"] = op_ctx
            translator_captured["framework_ctx"] = dict(framework_ctx)
            return QueryResult(matches=[])

    monkeypatch.setattr(
        langchain_vector_module,
        "VectorTranslator",
        DummyTranslator,
    )

    store = _make_store(adapter)
    cfg = {"run_id": "lc-run"}

    docs = store.similarity_search("hello", k=1, config=cfg)
    assert docs == []
    assert captured["dict_called"] is True
    assert captured["lc_called"] is False
    assert captured["dict_config"] is cfg
    assert isinstance(translator_captured["op_ctx"], OperationContext)


def test_build_ctx_falls_back_to_ctx_from_langchain_when_dict_fails(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If ctx_from_dict raises, _build_ctx should fall back to ctx_from_langchain.
    """
    captured: Dict[str, Any] = {"dict_called": False, "lc_called": False}

    class DummyOC(OperationContext):
        pass

    def fake_from_dict(config: Mapping[str, Any]) -> OperationContext:
        captured["dict_called"] = True
        raise RuntimeError("dict path failed")

    def fake_from_langchain(config: Mapping[str, Any]) -> OperationContext:
        captured["lc_called"] = True
        captured["lc_config"] = config
        return DummyOC()

    monkeypatch.setattr(
        langchain_vector_module,
        "ctx_from_dict",
        fake_from_dict,
    )
    monkeypatch.setattr(
        langchain_vector_module,
        "ctx_from_langchain",
        fake_from_langchain,
    )

    translator_captured: Dict[str, Any] = {}

    class DummyTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        def query(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> QueryResult:
            translator_captured["raw_query"] = dict(raw_query)
            translator_captured["op_ctx"] = op_ctx
            translator_captured["framework_ctx"] = dict(framework_ctx)
            return QueryResult(matches=[])

    monkeypatch.setattr(
        langchain_vector_module,
        "VectorTranslator",
        DummyTranslator,
    )

    store = _make_store(adapter)
    cfg = {"run_id": "lc-run-2"}

    docs = store.similarity_search("hello", k=1, config=cfg)
    assert docs == []
    assert captured["dict_called"] is True
    assert captured["lc_called"] is True
    assert captured["lc_config"] is cfg
    assert isinstance(translator_captured["op_ctx"], OperationContext)


# ---------------------------------------------------------------------------
# Error-context decorators (sync + async)
# ---------------------------------------------------------------------------


def test_sync_errors_include_langchain_metadata_in_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    When a sync vector op fails, with_error_context should call attach_context
    with framework='langchain' and operation='similarity_search_sync' (or similar).
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(
        langchain_vector_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        def query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from langchain vector adapter")

    monkeypatch.setattr(
        langchain_vector_module,
        "VectorTranslator",
        FailingTranslator,
    )

    store = _make_store(adapter)

    with pytest.raises(RuntimeError, match="test error from langchain vector adapter"):
        store.similarity_search("oops", k=1)

    assert captured_ctx
    assert captured_ctx.get("framework") == "langchain"
    op = str(captured_ctx.get("operation", ""))
    assert op.startswith("similarity_search")


@pytest.mark.asyncio
async def test_async_errors_include_langchain_metadata_in_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Same as sync error-context test but for asimilarity_search.
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(
        langchain_vector_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        async def arun_query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test async error from langchain vector adapter")

    monkeypatch.setattr(
        langchain_vector_module,
        "VectorTranslator",
        FailingTranslator,
    )

    store = _make_store(adapter)

    with pytest.raises(
        RuntimeError,
        match="test async error from langchain vector adapter",
    ):
        await store.asimilarity_search("oops-async", k=1)

    assert captured_ctx
    assert captured_ctx.get("framework") == "langchain"
    op = str(captured_ctx.get("operation", ""))
    assert op.startswith("similarity_search")


# ---------------------------------------------------------------------------
# Embedding helpers (_ensure_embeddings / _embed_query)
# ---------------------------------------------------------------------------


def test_ensure_embeddings_uses_provided_embeddings(adapter: Any) -> None:
    store = _make_store(adapter)
    texts = ["a", "b"]
    embs = [[1.0, 0.0], [0.0, 1.0]]

    out = store._ensure_embeddings(texts, embeddings=embs)  # noqa: SLF001
    assert out is embs


def test_ensure_embeddings_length_mismatch_raises_badrequest(adapter: Any) -> None:
    store = _make_store(adapter)
    texts = ["a", "b"]
    embs = [[1.0, 0.0]]

    with pytest.raises(BadRequest) as exc_info:
        store._ensure_embeddings(texts, embeddings=embs)  # noqa: SLF001

    msg = str(exc_info.value)
    assert "embeddings length" in msg
    assert getattr(exc_info.value, "code", None) == "BAD_EMBEDDINGS"


def test_ensure_embeddings_requires_embedding_function_when_missing(
    adapter: Any,
) -> None:
    store = _make_store(adapter, with_embeddings=False)
    texts = ["x"]

    with pytest.raises(NotSupported) as exc_info:
        store._ensure_embeddings(texts, embeddings=None)  # noqa: SLF001

    assert getattr(exc_info.value, "code", None) == "NO_EMBEDDING_FUNCTION"


@pytest.mark.asyncio
async def test_ensure_embeddings_async_uses_async_embedding_function(
    adapter: Any,
) -> None:
    store = _make_store(
        adapter,
        with_embeddings=False,
        async_embedding_function=_simple_async_embedding_fn,
    )
    texts = ["a", "b", "c"]
    out = await store._ensure_embeddings_async(texts, embeddings=None)  # noqa: SLF001
    assert len(out) == len(texts)


@pytest.mark.asyncio
async def test_ensure_embeddings_async_falls_back_to_sync_embedding_function(
    adapter: Any,
) -> None:
    store = _make_store(adapter)  # only sync embedding_function
    texts = ["a"]
    out = await store._ensure_embeddings_async(texts, embeddings=None)  # noqa: SLF001
    assert len(out) == 1


def test_embed_query_uses_provided_embedding(adapter: Any) -> None:
    store = _make_store(adapter)
    emb = store._embed_query("q", embedding=[1, 2, 3])  # noqa: SLF001
    assert emb == [1.0, 2.0, 3.0]


def test_embed_query_uses_embedding_function(adapter: Any) -> None:
    store = _make_store(adapter)
    emb = store._embed_query("q")  # noqa: SLF001
    assert len(emb) == 4


def test_embed_query_without_embedding_function_raises(adapter: Any) -> None:
    store = _make_store(adapter, with_embeddings=False)

    with pytest.raises(NotSupported) as exc_info:
        store._embed_query("q")  # noqa: SLF001

    assert getattr(exc_info.value, "code", None) == "NO_EMBEDDING_FUNCTION"


@pytest.mark.asyncio
async def test_embed_query_async_uses_async_embedding_function(
    adapter: Any,
) -> None:
    store = _make_store(
        adapter,
        with_embeddings=False,
        async_embedding_function=_simple_async_embedding_fn,
    )
    emb = await store._embed_query_async("q")  # noqa: SLF001
    assert len(emb) == 4


# ---------------------------------------------------------------------------
# Metadata / ID normalization helpers
# ---------------------------------------------------------------------------


def test_normalize_metadatas_matches_text_length(adapter: Any) -> None:
    store = _make_store(adapter)
    out = store._normalize_metadatas(2, [{"a": 1}, {"b": 2}])  # noqa: SLF001
    assert out == [{"a": 1}, {"b": 2}]


def test_normalize_metadatas_broadcast_single_metadata(adapter: Any) -> None:
    store = _make_store(adapter)
    out = store._normalize_metadatas(3, [{"x": 1}])  # noqa: SLF001
    assert out == [{"x": 1}, {"x": 1}, {"x": 1}]


def test_normalize_metadatas_length_mismatch_raises(adapter: Any) -> None:
    store = _make_store(adapter)
    with pytest.raises(BadRequest) as exc_info:
        store._normalize_metadatas(2, [{"x": 1}, {"y": 2}, {"z": 3}])  # noqa: SLF001
    assert getattr(exc_info.value, "code", None) == "BAD_METADATA"


def test_normalize_ids_generates_ids_when_missing(adapter: Any) -> None:
    store = _make_store(adapter)
    out = store._normalize_ids(3, ids=None)  # noqa: SLF001
    assert len(out) == 3
    assert len(set(out)) == 3  # uuid-ish uniqueness


def test_normalize_ids_length_mismatch_raises(adapter: Any) -> None:
    store = _make_store(adapter)
    with pytest.raises(BadRequest) as exc_info:
        store._normalize_ids(2, ids=["a"])  # noqa: SLF001
    assert getattr(exc_info.value, "code", None) == "BAD_IDS"


# ---------------------------------------------------------------------------
# Vector translation helpers (_to_corpus_vectors / _from_corpus_matches)
# ---------------------------------------------------------------------------


def test_to_corpus_vectors_wraps_text_and_metadata(adapter: Any) -> None:
    store = _make_store(adapter)
    texts = ["d1", "d2"]
    embs = [[0.0, 1.0], [1.0, 0.0]]
    metadatas = [{"a": 1}, {"b": 2}]
    ids = ["id-1", "id-2"]

    vectors = store._to_corpus_vectors(  # noqa: SLF001
        texts=texts,
        embeddings=embs,
        metadatas=metadatas,
        ids=ids,
        namespace="ns-x",
    )

    assert len(vectors) == 2
    v0 = vectors[0]
    assert v0.id == "id-1"
    assert v0.namespace == "ns-x"
    assert v0.metadata["page_content"] == "d1"
    assert v0.metadata["id"] == "id-1"
    assert v0.metadata["a"] == 1


def test_from_corpus_matches_builds_documents_and_strips_internal_keys(
    adapter: Any,
) -> None:
    store = _make_store(adapter)
    matches = [
        _make_match(
            "hello",
            doc_id="d-1",
            score=0.9,
            extra_meta={"topic": "t"},
        ),
    ]

    docs_scores = store._from_corpus_matches(matches)  # noqa: SLF001
    assert len(docs_scores) == 1
    doc, score = docs_scores[0]
    assert doc.page_content == "hello"
    assert doc.metadata == {"topic": "t"}
    assert math.isclose(score, 0.9)


def test_apply_score_threshold_filters_matches(adapter: Any) -> None:
    store = _make_store(adapter)
    store.score_threshold = 0.8

    matches = [
        _make_match("low", score=0.5),
        _make_match("high", score=0.9),
    ]
    out = store._apply_score_threshold(matches)  # noqa: SLF001
    assert len(out) == 1
    assert out[0].vector.metadata["page_content"] == "high"


# ---------------------------------------------------------------------------
# Similarity search (sync + async + streaming)
# ---------------------------------------------------------------------------


def test_similarity_search_uses_translator_and_returns_documents(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    match = _make_match("hello", score=0.9)

    class DummyTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        def query(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> QueryResult:
            captured["raw_query"] = dict(raw_query)
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx)
            return QueryResult(matches=[match])

    monkeypatch.setattr(
        langchain_vector_module,
        "VectorTranslator",
        DummyTranslator,
    )

    store = _make_store(adapter)

    docs = store.similarity_search(
        "q-text",
        k=3,
        filter={"tag": "v"},
        namespace="ns-q",
        config={"run_id": "run-sync"},
    )
    assert len(docs) == 1
    assert docs[0].page_content == "hello"

    raw = captured["raw_query"]
    assert raw["top_k"] == 3
    assert raw["filters"] == {"tag": "v"}
    assert raw["namespace"] == "ns-q"
    assert raw["include_metadata"] is True
    assert raw["include_vectors"] is False


@pytest.mark.asyncio
async def test_async_similarity_search_uses_translator_and_returns_documents(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    match = _make_match("hello-async", score=0.8)

    class DummyTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        async def arun_query(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> QueryResult:
            captured["raw_query"] = dict(raw_query)
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx)
            return QueryResult(matches=[match])

    monkeypatch.setattr(
        langchain_vector_module,
        "VectorTranslator",
        DummyTranslator,
    )

    store = _make_store(adapter)

    docs = await store.asimilarity_search(
        "async-q",
        k=2,
        filter={"k": 1},
        namespace="ns-async",
        config={"run_id": "run-async"},
    )
    assert len(docs) == 1
    assert docs[0].page_content == "hello-async"

    raw = captured["raw_query"]
    assert raw["top_k"] == 2
    assert raw["filters"] == {"k": 1}


def test_similarity_search_stream_yields_documents(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    match = _make_match("stream", score=0.99)

    class DummyTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        def query_stream(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> Iterator[QueryChunk]:
            captured["raw_query"] = dict(raw_query)
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx)
            yield QueryChunk(matches=[match])

    monkeypatch.setattr(
        langchain_vector_module,
        "VectorTranslator",
        DummyTranslator,
    )

    store = _make_store(adapter)

    iterator = store.similarity_search_stream("stream-q", k=2)
    docs = list(iterator)
    assert docs
    assert docs[0].page_content == "stream"


@pytest.mark.asyncio
async def test_async_similarity_search_stream_yields_documents(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    match = _make_match("astream", score=0.77)

    class DummyTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        async def arun_query_stream(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ):
            async def gen():
                captured["raw_query"] = dict(raw_query)
                captured["op_ctx"] = op_ctx
                captured["framework_ctx"] = dict(framework_ctx)
                yield QueryChunk(matches=[match])

            return gen()

    monkeypatch.setattr(
        langchain_vector_module,
        "VectorTranslator",
        DummyTranslator,
    )

    store = _make_store(adapter)

    aiter = store.asimilarity_search_stream("async-stream-q", k=2)
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    seen = []
    async for doc in aiter:
        seen.append(doc)
        break

    assert seen
    assert seen[0].page_content == "astream"


def test_similarity_search_with_score_returns_tuples(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    match1 = _make_match("lo", score=0.5)
    match2 = _make_match("hi", score=0.9)

    class DummyTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        def query(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> QueryResult:
            return QueryResult(matches=[match1, match2])

    monkeypatch.setattr(
        langchain_vector_module,
        "VectorTranslator",
        DummyTranslator,
    )

    store = _make_store(adapter)

    results = store.similarity_search_with_score("q", k=4)
    assert len(results) == 2
    doc, score = results[0]
    assert hasattr(doc, "page_content")
    assert isinstance(score, float)


@pytest.mark.asyncio
async def test_async_similarity_search_with_score_returns_tuples(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    match = _make_match("async-lo", score=0.42)

    class DummyTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        async def arun_query(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> QueryResult:
            return QueryResult(matches=[match])

    monkeypatch.setattr(
        langchain_vector_module,
        "VectorTranslator",
        DummyTranslator,
    )

    store = _make_store(adapter)

    results = await store.asimilarity_search_with_score("q-async", k=3)
    assert len(results) == 1
    doc, score = results[0]
    assert hasattr(doc, "page_content")
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# MMR search (sync + async)
# ---------------------------------------------------------------------------


def test_mmr_search_validates_lambda_and_k(adapter: Any) -> None:
    """
    lambda_mult outside [0, 1] should raise BadRequest; k<=0 returns [].
    """
    store = _make_store(adapter)

    assert store.max_marginal_relevance_search("q", k=0) == []

    with pytest.raises(BadRequest) as exc_info:
        store.max_marginal_relevance_search("q", lambda_mult=1.5)

    assert getattr(exc_info.value, "code", None) == "BAD_MMR_LAMBDA"


def test_mmr_search_calls_translator_and_respects_k(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    max_marginal_relevance_search should:
    - request include_vectors=True
    - use returned matches and internal MMR selector to choose up to k docs.
    """
    captured: Dict[str, Any] = {}

    matches = [
        _make_match("a", score=0.9),
        _make_match("b", score=0.85),
        _make_match("c", score=0.8),
    ]

    class DummyTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        def query(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> QueryResult:
            captured["raw_query"] = dict(raw_query)
            captured["framework_ctx"] = dict(framework_ctx)
            return QueryResult(matches=matches)

    monkeypatch.setattr(
        langchain_vector_module,
        "VectorTranslator",
        DummyTranslator,
    )

    store = _make_store(adapter)

    docs = store.max_marginal_relevance_search("mmr-q", k=2, lambda_mult=0.5)
    assert 0 < len(docs) <= 2

    raw = captured["raw_query"]
    assert raw["include_vectors"] is True
    assert raw["top_k"] >= 2  # fetch_k >= k


@pytest.mark.asyncio
async def test_async_mmr_search_calls_translator_and_respects_k(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    matches = [
        _make_match("aa", score=0.6),
        _make_match("bb", score=0.7),
    ]

    class DummyTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        async def arun_query(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> QueryResult:
            captured["raw_query"] = dict(raw_query)
            captured["framework_ctx"] = dict(framework_ctx)
            return QueryResult(matches=matches)

    monkeypatch.setattr(
        langchain_vector_module,
        "VectorTranslator",
        DummyTranslator,
    )

    store = _make_store(adapter)

    docs = await store.amax_marginal_relevance_search(
        "ammr-q",
        k=2,
        lambda_mult=0.3,
    )
    assert 0 < len(docs) <= 2

    raw = captured["raw_query"]
    assert raw["include_vectors"] is True


# ---------------------------------------------------------------------------
# Delete API (sync + async)
# ---------------------------------------------------------------------------


def test_delete_requires_ids_or_filter(adapter: Any) -> None:
    store = _make_store(adapter)

    with pytest.raises(BadRequest) as exc_info:
        store.delete()

    assert getattr(exc_info.value, "code", None) == "BAD_DELETE"


def test_delete_builds_raw_request_and_calls_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        def delete(
            self,
            raw_request: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> None:
            captured["raw_request"] = dict(raw_request)
            captured["framework_ctx"] = dict(framework_ctx)

    monkeypatch.setattr(
        langchain_vector_module,
        "VectorTranslator",
        DummyTranslator,
    )

    store = _make_store(adapter)

    store.delete(ids=["x", "y"], namespace="ns-del")
    raw = captured["raw_request"]
    assert raw["ids"] == ["x", "y"]
    assert raw["namespace"] == "ns-del"


@pytest.mark.asyncio
async def test_async_delete_builds_raw_request_and_calls_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        async def arun_delete(
            self,
            raw_request: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> None:
            captured["raw_request"] = dict(raw_request)
            captured["framework_ctx"] = dict(framework_ctx)

    monkeypatch.setattr(
        langchain_vector_module,
        "VectorTranslator",
        DummyTranslator,
    )

    store = _make_store(adapter)

    await store.adelete(filter={"tag": "v"}, namespace="ns-adel")
    raw = captured["raw_request"]
    assert raw["filter"] == {"tag": "v"}
    assert raw["ids"] is None
    assert raw["namespace"] == "ns-adel"


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def test_from_texts_constructs_store_and_adds_texts(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        def upsert(
            self,
            raw_request: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> UpsertResult:
            captured["raw_request"] = dict(raw_request)
            captured["framework_ctx"] = dict(framework_ctx)
            # pretend everything succeeded
            return UpsertResult(upserted_count=len(raw_request.get("vectors", [])))

    monkeypatch.setattr(
        langchain_vector_module,
        "VectorTranslator",
        DummyTranslator,
    )

    texts = ["t1", "t2"]
    metadatas = [{"m": 1}, {"m": 2}]
    ids = ["id-1", "id-2"]

    store = CorpusLangChainVectorStore.from_texts(
        texts,
        corpus_adapter=adapter,
        metadatas=metadatas,
        ids=ids,
        embedding_function=_simple_embedding_fn,
    )

    assert isinstance(store, CorpusLangChainVectorStore)
    raw = captured["raw_request"]
    vectors = raw["vectors"]
    assert len(vectors) == 2
    assert vectors[0].id == "id-1"
    assert vectors[0].metadata["page_content"] == "t1"


def test_from_documents_constructs_store_and_adds_documents(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    from langchain_core.documents import Document  # type: ignore[import]

    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        def upsert(
            self,
            raw_request: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> UpsertResult:
            captured["raw_request"] = dict(raw_request)
            captured["framework_ctx"] = dict(framework_ctx)
            return UpsertResult(upserted_count=len(raw_request.get("vectors", [])))

    monkeypatch.setattr(
        langchain_vector_module,
        "VectorTranslator",
        DummyTranslator,
    )

    docs = [
        Document(page_content="d1", metadata={"x": 1}),
        Document(page_content="d2", metadata={"x": 2}),
    ]

    store = CorpusLangChainVectorStore.from_documents(
        docs,
        corpus_adapter=adapter,
        embedding_function=_simple_embedding_fn,
    )

    assert isinstance(store, CorpusLangChainVectorStore)
    raw = captured["raw_request"]
    vectors = raw["vectors"]
    assert len(vectors) == 2
    assert vectors[0].metadata["page_content"] == "d1"


# ---------------------------------------------------------------------------
# CorpusLangChainRetriever behavior
# ---------------------------------------------------------------------------


def test_retriever_sync_delegates_to_vector_store(adapter: Any) -> None:
    from langchain_core.documents import Document  # type: ignore[import]

    captured: Dict[str, Any] = {}

    class DummyStore(CorpusLangChainVectorStore):
        def __init__(self) -> None:
            # avoid real adapter
            pass

        def similarity_search(
            self,
            query: str,
            k: int = 4,
            **kwargs: Any,
        ) -> List[Document]:
            captured["query"] = query
            captured["k"] = k
            captured["kwargs"] = dict(kwargs)
            return [Document(page_content="R", metadata={"source": "dummy"})]

    store = DummyStore()
    retriever = CorpusLangChainRetriever(
        vector_store=store,
        search_kwargs={"k": 5, "namespace": "ret-ns"},
    )

    docs = retriever.get_relevant_documents("retr-q")
    assert len(docs) == 1
    assert captured["query"] == "retr-q"
    assert captured["k"] == 5
    assert captured["kwargs"]["namespace"] == "ret-ns"


@pytest.mark.asyncio
async def test_retriever_async_delegates_to_vector_store(adapter: Any) -> None:
    from langchain_core.documents import Document  # type: ignore[import]

    captured: Dict[str, Any] = {}

    class DummyStore(CorpusLangChainVectorStore):
        def __init__(self) -> None:
            pass

        async def asimilarity_search(
            self,
            query: str,
            k: int = 4,
            **kwargs: Any,
        ) -> List[Document]:
            captured["query"] = query
            captured["k"] = k
            captured["kwargs"] = dict(kwargs)
            return [Document(page_content="AR", metadata={})]

    store = DummyStore()
    retriever = CorpusLangChainRetriever(
        vector_store=store,
        search_kwargs={"k": 2},
    )

    docs = await retriever.aget_relevant_documents("async-retr-q")
    assert len(docs) == 1
    assert captured["query"] == "async-retr-q"
    assert captured["k"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
