 # tests/frameworks/vector/test_langchain_adapter.py

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
from langchain_core.documents import Document

import corpus_sdk.vector.framework_adapters.langchain as langchain_module
from corpus_sdk.vector.framework_adapters.langchain import (
    CorpusLangChainRetriever,
    CorpusLangChainVectorStore,
)
from corpus_sdk.vector.vector_base import (
    BadRequest,
    NotSupported,
    OperationContext,
    Vector,
    VectorAdapterError,
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


def _make_store(
    adapter: Any,
    *,
    with_embeddings: bool = True,
    namespace: Optional[str] = "default",
) -> CorpusLangChainVectorStore:
    kwargs: Dict[str, Any] = {"corpus_adapter": adapter, "namespace": namespace}
    if with_embeddings:
        kwargs["embedding_function"] = _simple_embedding_fn
    return CorpusLangChainVectorStore(**kwargs)


class _FakeVectorMatch:
    """Minimal stand-in for VectorMatch used in internal helpers."""

    def __init__(self, vector: Vector, score: float) -> None:
        self.vector = vector
        self.score = score


# ---------------------------------------------------------------------------
# Translator wiring
# ---------------------------------------------------------------------------


def test_default_translator_uses_langchain_framework(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    The lazy _translator property should construct a VectorTranslator with
    framework='langchain' and a DefaultVectorFrameworkTranslator instance.
    """
    captured: Dict[str, Any] = {}

    class DummyFrameworkTranslator:
        pass

    def fake_default_ft(*args: Any, **kwargs: Any) -> DummyFrameworkTranslator:  # noqa: ARG001
        captured["default_ft_called"] = True
        return DummyFrameworkTranslator()

    def fake_vector_translator(*args: Any, **kwargs: Any) -> Any:
        captured["args"] = args
        captured["kwargs"] = kwargs

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(
        langchain_module,
        "DefaultVectorFrameworkTranslator",
        fake_default_ft,
    )
    monkeypatch.setattr(
        langchain_module,
        "VectorTranslator",
        fake_vector_translator,
    )

    store = _make_store(adapter)

    # Trigger lazy construction
    _ = store._translator  # noqa: SLF001

    assert captured.get("default_ft_called") is True

    args = captured.get("args") or ()
    kwargs = captured.get("kwargs") or {}

    # First positional arg should be the underlying adapter
    assert args[0] is adapter
    assert kwargs.get("framework") == "langchain"
    # translator should be the DefaultVectorFrameworkTranslator instance
    assert isinstance(kwargs.get("translator"), DummyFrameworkTranslator)


# ---------------------------------------------------------------------------
# Context translation / _build_ctx
# ---------------------------------------------------------------------------


def test_build_ctx_uses_ctx_from_dict_then_langchain(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    _build_ctx should try ctx_from_dict first, then fall back to ctx_from_langchain,
    and only return an OperationContext instance.
    """
    captured: Dict[str, Any] = {}

    class DummyOperationContext(OperationContext):  # type: ignore[misc]
        def __init__(self, **kwargs: Any) -> None:  # noqa: D401
            # We don't care about real OperationContext plumbing here.
            self.attrs = kwargs

    monkeypatch.setattr(
        langchain_module,
        "OperationContext",
        DummyOperationContext,
    )

    def fake_ctx_from_dict(config: Mapping[str, Any]) -> DummyOperationContext:
        captured["from_dict"] = config
        return DummyOperationContext(source="dict", config=config)

    def fake_ctx_from_langchain(config: Mapping[str, Any]) -> DummyOperationContext:
        captured["from_langchain"] = config
        return DummyOperationContext(source="langchain", config=config)

    monkeypatch.setattr(
        langchain_module,
        "ctx_from_dict",
        fake_ctx_from_dict,
    )
    monkeypatch.setattr(
        langchain_module,
        "ctx_from_langchain",
        fake_ctx_from_langchain,
    )

    store = _make_store(adapter)

    config = {"run_id": "r-1", "tags": ["t1"]}
    ctx = store._build_ctx(config=config)  # noqa: SLF001

    assert isinstance(ctx, DummyOperationContext)
    # ctx_from_dict should have been used, but not ctx_from_langchain
    assert captured.get("from_dict") is config
    assert "from_langchain" not in captured


def test_build_ctx_accepts_operation_context_passthrough(
    adapter: Any,
) -> None:
    """
    If config is already an OperationContext instance, _build_ctx should return
    it unchanged.
    """
    store = _make_store(adapter)

    class DummyOperationContext(OperationContext):  # type: ignore[misc]
        pass

    ctx = DummyOperationContext()
    out = store._build_ctx(config=ctx)  # noqa: SLF001
    assert out is ctx


# ---------------------------------------------------------------------------
# Embedding helpers + normalization
# ---------------------------------------------------------------------------


def test_ensure_embeddings_uses_embedding_function_when_not_provided(adapter: Any) -> None:
    store = _make_store(adapter, with_embeddings=True)

    texts = ["a", "b"]
    embs = store._ensure_embeddings(texts, embeddings=None)  # noqa: SLF001
    assert len(embs) == len(texts)
    assert all(len(row) == 4 for row in embs)


def test_ensure_embeddings_raises_when_no_embedding_function(adapter: Any) -> None:
    store = _make_store(adapter, with_embeddings=False)

    with pytest.raises(NotSupported) as exc_info:
        store._ensure_embeddings(["x"], embeddings=None)  # noqa: SLF001

    msg = str(exc_info.value)
    assert "No embedding_function configured" in msg


def test_normalize_metadatas_and_ids_happy_and_error_paths(adapter: Any) -> None:
    store = _make_store(adapter)

    # metadatas: None -> [{}] * n
    metas = store._normalize_metadatas(3, None)  # noqa: SLF001
    assert len(metas) == 3
    assert all(isinstance(m, dict) for m in metas)

    # metadatas: single entry replicated
    metas2 = store._normalize_metadatas(2, [{"a": 1}])  # noqa: SLF001
    assert metas2 == [{"a": 1}, {"a": 1}]

    # ids: None -> generated
    ids = store._normalize_ids(2, None)  # noqa: SLF001
    assert len(ids) == 2
    assert all(isinstance(i, str) for i in ids)

    # Error: mismatched lengths
    with pytest.raises(BadRequest):
        store._normalize_metadatas(2, [{"a": 1}, {"b": 2}, {"c": 3}])  # noqa: SLF001

    with pytest.raises(BadRequest):
        store._normalize_ids(1, ["x", "y"])  # noqa: SLF001


def test_to_and_from_corpus_vectors_and_score_threshold(adapter: Any) -> None:
    """
    _to_corpus_vectors should envelope id/text into metadata, and
    _from_corpus_matches should strip those internal keys and respect score_threshold.
    """
    store = _make_store(adapter)
    store.score_threshold = 0.8

    texts = ["t1", "t2"]
    embs = _simple_embedding_fn(texts)
    metadatas = [{"foo": 1}, {"foo": 2}]
    ids = ["id1", "id2"]

    vectors = store._to_corpus_vectors(  # noqa: SLF001
        texts=texts,
        embeddings=embs,
        metadatas=metadatas,
        ids=ids,
        namespace="ns-1",
    )
    assert len(vectors) == 2
    v0 = vectors[0]
    assert isinstance(v0, Vector)
    assert v0.metadata.get(store.text_field) == "t1"
    assert v0.metadata.get(store.id_field) == "id1"

    matches = [
        _FakeVectorMatch(vector=vectors[0], score=0.5),
        _FakeVectorMatch(vector=vectors[1], score=0.9),
    ]

    filtered = store._apply_score_threshold(matches)  # noqa: SLF001
    assert len(filtered) == 1
    assert filtered[0].score == 0.9

    docs_scores = store._from_corpus_matches(filtered)  # noqa: SLF001
    assert len(docs_scores) == 1
    doc, score = docs_scores[0]
    assert isinstance(doc, Document)
    assert doc.page_content == "t2"
    # internal keys removed
    assert store.id_field not in doc.metadata
    assert store.text_field not in doc.metadata
    assert score == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Error-context decorators (sync + async)
# ---------------------------------------------------------------------------


def test_sync_errors_include_langchain_metadata_in_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    When similarity_search fails, with_error_context should call attach_context
    with framework='langchain' and operation='similarity_search_sync'.
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(
        langchain_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingTranslator:
        def query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from langchain vector adapter")

    store = _make_store(adapter)
    store._translator = FailingTranslator()  # type: ignore[assignment]  # noqa: SLF001

    with pytest.raises(RuntimeError, match="test error from langchain vector adapter"):
        store.similarity_search("oops")

    assert captured_ctx
    assert captured_ctx.get("framework") == "langchain"
    assert captured_ctx.get("operation") == "similarity_search_sync"


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
        langchain_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingTranslator:
        async def arun_query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test async error from langchain vector adapter")

    store = _make_store(adapter)
    store._translator = FailingTranslator()  # type: ignore[assignment]  # noqa: SLF001

    with pytest.raises(
        RuntimeError,
        match="test async error from langchain vector adapter",
    ):
        await store.asimilarity_search("oops-async")

    assert captured_ctx
    assert captured_ctx.get("framework") == "langchain"
    assert captured_ctx.get("operation") == "similarity_search_async"


# ---------------------------------------------------------------------------
# Similarity search (sync + async)
# ---------------------------------------------------------------------------


def test_similarity_search_builds_raw_query_and_coerces_documents(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    similarity_search() should:

    - Build a raw query mapping with top_k, filters, include flags.
    - Call translator.query(raw_query, op_ctx, framework_ctx).
    - Treat result as QueryResult and convert matches to Documents.
    """
    captured: Dict[str, Any] = {}

    class FakeQueryResult:
        def __init__(self, matches: Sequence[Any]) -> None:
            self.matches = list(matches)

    # Patch QueryResult used inside the adapter
    monkeypatch.setattr(
        langchain_module,
        "QueryResult",
        FakeQueryResult,
    )

    def make_match(text: str, score: float) -> _FakeVectorMatch:
        vec = Vector(id="v1", vector=[0.0], metadata={"page_content": text, "id": "v1"})
        return _FakeVectorMatch(vector=vec, score=score)

    class DummyTranslator:
        def query(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> FakeQueryResult:
            captured["raw_query"] = dict(raw_query)
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx)
            return FakeQueryResult(matches=[make_match("hello", 0.9)])

    store = _make_store(adapter)
    store._translator = DummyTranslator()  # type: ignore[assignment]  # noqa: SLF001

    docs = store.similarity_search("q-text", k=2, filter={"tag": "t"})
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "hello"

    raw = captured["raw_query"]
    assert raw["top_k"] == 2
    assert raw["filters"] == {"tag": "t"}
    assert raw["include_metadata"] is True
    assert raw["include_vectors"] is False


@pytest.mark.asyncio
async def test_async_similarity_search_builds_raw_query_and_coerces_documents(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class FakeQueryResult:
        def __init__(self, matches: Sequence[Any]) -> None:
            self.matches = list(matches)

    monkeypatch.setattr(
        langchain_module,
        "QueryResult",
        FakeQueryResult,
    )

    def make_match(text: str, score: float) -> _FakeVectorMatch:
        vec = Vector(id="v2", vector=[0.0], metadata={"page_content": text, "id": "v2"})
        return _FakeVectorMatch(vector=vec, score=score)

    class DummyTranslator:
        async def arun_query(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> FakeQueryResult:
            captured["raw_query"] = dict(raw_query)
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx)
            return FakeQueryResult(matches=[make_match("async-hello", 0.85)])

    store = _make_store(adapter)
    store._translator = DummyTranslator()  # type: ignore[assignment]  # noqa: SLF001

    docs = await store.asimilarity_search("async-q", k=3)
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "async-hello"

    raw = captured["raw_query"]
    assert raw["top_k"] == 3
    assert raw["include_vectors"] is False


def test_similarity_search_stream_yields_documents(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    similarity_search_stream should accept the same params as similarity_search
    and yield Documents from QueryChunk.matches.
    """
    captured: Dict[str, Any] = {}

    class FakeQueryChunk:
        def __init__(self, matches: Sequence[Any]) -> None:
            self.matches = list(matches)

    monkeypatch.setattr(
        langchain_module,
        "QueryChunk",
        FakeQueryChunk,
    )

    def make_match(text: str, score: float) -> _FakeVectorMatch:
        vec = Vector(id="s1", vector=[0.0], metadata={"page_content": text, "id": "s1"})
        return _FakeVectorMatch(vector=vec, score=score)

    class StreamTranslator:
        def query_stream(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ):
            captured["raw_query"] = dict(raw_query)
            captured["framework_ctx"] = dict(framework_ctx)
            yield FakeQueryChunk(matches=[make_match("stream-doc", 0.99)])

    store = _make_store(adapter)
    store._translator = StreamTranslator()  # type: ignore[assignment]  # noqa: SLF001

    iterator = store.similarity_search_stream("stream-q", k=2)
    assert hasattr(iterator, "__iter__")
    docs = list(iterator)
    assert docs
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "stream-doc"


# ---------------------------------------------------------------------------
# Similarity-with-score (sync + async)
# ---------------------------------------------------------------------------


def test_similarity_search_with_score_returns_document_score_pairs(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class FakeQueryResult:
        def __init__(self, matches: Sequence[Any]) -> None:
            self.matches = list(matches)

    monkeypatch.setattr(
        langchain_module,
        "QueryResult",
        FakeQueryResult,
    )

    def make_match(text: str, score: float) -> _FakeVectorMatch:
        vec = Vector(id=text, vector=[0.0], metadata={"page_content": text, "id": text})
        return _FakeVectorMatch(vector=vec, score=score)

    class DummyTranslator:
        def query(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> FakeQueryResult:
            captured["raw_query"] = dict(raw_query)
            return FakeQueryResult(
                matches=[make_match("a", 0.5), make_match("b", 0.8)],
            )

    store = _make_store(adapter)
    store._translator = DummyTranslator()  # type: ignore[assignment]  # noqa: SLF001

    results = store.similarity_search_with_score("q", k=4)
    assert len(results) == 2
    doc, score = results[0]
    assert isinstance(doc, Document)
    assert isinstance(score, float)


@pytest.mark.asyncio
async def test_async_similarity_search_with_score_returns_document_score_pairs(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class FakeQueryResult:
        def __init__(self, matches: Sequence[Any]) -> None:
            self.matches = list(matches)

    monkeypatch.setattr(
        langchain_module,
        "QueryResult",
        FakeQueryResult,
    )

    def make_match(text: str, score: float) -> _FakeVectorMatch:
        vec = Vector(id=text, vector=[0.0], metadata={"page_content": text, "id": text})
        return _FakeVectorMatch(vector=vec, score=score)

    class DummyTranslator:
        async def arun_query(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> FakeQueryResult:
            captured["raw_query"] = dict(raw_query)
            return FakeQueryResult(
                matches=[make_match("x", 0.6), make_match("y", 0.9)],
            )

    store = _make_store(adapter)
    store._translator = DummyTranslator()  # type: ignore[assignment]  # noqa: SLF001

    results = await store.asimilarity_search_with_score("q-async", k=3)
    assert len(results) == 2
    doc, score = results[0]
    assert isinstance(doc, Document)
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# MMR search (sync + async)
# ---------------------------------------------------------------------------


def test_mmr_search_builds_query_with_vectors_and_calls_mmr_selector(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    max_marginal_relevance_search should request include_vectors=True and
    delegate candidate_matches to _mmr_select_indices.
    """
    captured: Dict[str, Any] = {}

    class FakeQueryResult:
        def __init__(self, matches: Sequence[Any]) -> None:
            self.matches = list(matches)

    monkeypatch.setattr(
        langchain_module,
        "QueryResult",
        FakeQueryResult,
    )

    def make_match(idx: int) -> _FakeVectorMatch:
        vec = Vector(
            id=f"m{idx}",
            vector=[float(idx), 0.0],
            metadata={"page_content": f"doc-{idx}", "id": f"m{idx}"},
        )
        return _FakeVectorMatch(vector=vec, score=0.5 + idx * 0.1)

    class DummyTranslator:
        def query(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> FakeQueryResult:
            captured["raw_query"] = dict(raw_query)
            return FakeQueryResult(matches=[make_match(0), make_match(1), make_match(2)])

    def fake_mmr_indices(
        self,
        query_vec: Sequence[float],
        candidate_matches: List[Any],
        k: int,
        lambda_mult: float,
    ) -> List[int]:
        captured["mmr_query_vec"] = list(query_vec)
        captured["mmr_k"] = k
        captured["mmr_lambda"] = lambda_mult
        # Just pick the first k items deterministically
        return list(range(min(k, len(candidate_matches))))

    monkeypatch.setattr(
        langchain_module,
        "CorpusLangChainVectorStore._mmr_select_indices",
        fake_mmr_indices,
    )

    store = _make_store(adapter)
    store._translator = DummyTranslator()  # type: ignore[assignment]  # noqa: SLF001

    docs = store.max_marginal_relevance_search("mmr-q", k=2, lambda_mult=0.7, fetch_k=5)
    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)

    raw = captured["raw_query"]
    assert raw["include_vectors"] is True
    assert raw["top_k"] == 5
    assert captured["mmr_k"] == 2
    assert captured["mmr_lambda"] == 0.7


@pytest.mark.asyncio
async def test_async_mmr_search_builds_query_with_vectors_and_calls_mmr_selector(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class FakeQueryResult:
        def __init__(self, matches: Sequence[Any]) -> None:
            self.matches = list(matches)

    monkeypatch.setattr(
        langchain_module,
        "QueryResult",
        FakeQueryResult,
    )

    def make_match(idx: int) -> _FakeVectorMatch:
        vec = Vector(
            id=f"a{idx}",
            vector=[float(idx), 1.0],
            metadata={"page_content": f"adoc-{idx}", "id": f"a{idx}"},
        )
        return _FakeVectorMatch(vector=vec, score=0.4 + idx * 0.2)

    class DummyTranslator:
        async def arun_query(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> FakeQueryResult:
            captured["raw_query"] = dict(raw_query)
            return FakeQueryResult(matches=[make_match(0), make_match(1)])

    def fake_mmr_indices(
        self,
        query_vec: Sequence[float],
        candidate_matches: List[Any],
        k: int,
        lambda_mult: float,
    ) -> List[int]:
        captured["mmr_k"] = k
        captured["mmr_lambda"] = lambda_mult
        return [1]  # just pick the second doc

    monkeypatch.setattr(
        langchain_module,
        "CorpusLangChainVectorStore._mmr_select_indices",
        fake_mmr_indices,
    )

    store = _make_store(adapter)
    store._translator = DummyTranslator()  # type: ignore[assignment]  # noqa: SLF001

    docs = await store.amax_marginal_relevance_search(
        "ammr-q",
        k=1,
        lambda_mult=0.3,
        fetch_k=4,
    )
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert captured["mmr_k"] == 1
    assert captured["mmr_lambda"] == 0.3


def test_mmr_search_rejects_invalid_lambda(adapter: Any) -> None:
    store = _make_store(adapter)

    with pytest.raises(BadRequest) as exc_info:
        store.max_marginal_relevance_search("q", k=2, lambda_mult=1.5)

    assert "BAD_MMR_LAMBDA" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_mmr_search_rejects_invalid_lambda(adapter: Any) -> None:
    store = _make_store(adapter)

    with pytest.raises(BadRequest) as exc_info:
        await store.amax_marginal_relevance_search("q", k=2, lambda_mult=-0.1)

    assert "BAD_MMR_LAMBDA" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Delete API
# ---------------------------------------------------------------------------


def test_delete_requires_ids_or_filter(adapter: Any) -> None:
    """
    delete() should raise BadRequest when neither ids nor filter is provided.
    """
    store = _make_store(adapter)

    with pytest.raises(BadRequest) as exc_info:
        store.delete()

    assert "BAD_DELETE" in str(exc_info.value)


def test_delete_builds_raw_request_and_calls_translator(
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def delete(
            self,
            raw_request: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> None:
            captured["raw_request"] = dict(raw_request)
            captured["framework_ctx"] = dict(framework_ctx)
            captured["op_ctx"] = op_ctx

    store = _make_store(adapter)
    store._translator = DummyTranslator()  # type: ignore[assignment]  # noqa: SLF001

    store.delete(ids=["1", "2"], filter={"tag": "v"}, namespace="ns-del")

    raw = captured["raw_request"]
    assert raw["namespace"] == "ns-del"
    assert raw["ids"] == ["1", "2"]
    assert raw["filter"] == {"tag": "v"}


@pytest.mark.asyncio
async def test_async_delete_builds_raw_request_and_calls_translator(
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_delete(
            self,
            raw_request: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> None:
            captured["raw_request"] = dict(raw_request)
            captured["framework_ctx"] = dict(framework_ctx)

    store = _make_store(adapter)
    store._translator = DummyTranslator()  # type: ignore[assignment]  # noqa: SLF001

    await store.adelete(ids=["x"], filter={"foo": 1}, namespace="ns-adel")

    raw = captured["raw_request"]
    assert raw["ids"] == ["x"]
    assert raw["filter"] == {"foo": 1}
    assert captured["framework_ctx"]["namespace"] == "ns-adel"


# ---------------------------------------------------------------------------
# add_texts / add_documents (sync + async)
# ---------------------------------------------------------------------------


def test_add_texts_uses_translator_upsert_and_returns_ids(
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def upsert(
            self,
            raw_request: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> None:
            captured["raw_request"] = dict(raw_request)
            captured["framework_ctx"] = dict(framework_ctx)

    store = _make_store(adapter)
    store._translator = DummyTranslator()  # type: ignore[assignment]  # noqa: SLF001

    texts = ["alpha", "beta"]
    ids = ["id-a", "id-b"]

    returned_ids = store.add_texts(texts, ids=ids, metadatas=[{"m": 1}, {"m": 2}])
    assert returned_ids == ids

    raw = captured["raw_request"]
    assert raw["namespace"] == "default"
    assert len(raw["vectors"]) == 2


@pytest.mark.asyncio
async def test_async_add_texts_uses_translator_upsert_and_returns_ids(
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_upsert(
            self,
            raw_request: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> None:
            captured["raw_request"] = dict(raw_request)
            captured["framework_ctx"] = dict(framework_ctx)

    store = _make_store(adapter)
    store._translator = DummyTranslator()  # type: ignore[assignment]  # noqa: SLF001

    texts = ["x", "y"]
    returned_ids = await store.aadd_texts(texts)
    assert len(returned_ids) == len(texts)

    raw = captured["raw_request"]
    assert len(raw["vectors"]) == 2


def test_add_documents_delegates_to_add_texts(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        captured["texts"] = list(texts)
        captured["metadatas"] = metadatas
        captured["ids"] = ids
        return ["id-1", "id-2"]

    monkeypatch.setattr(
        langchain_module.CorpusLangChainVectorStore,
        "add_texts",
        fake_add_texts,
    )

    store = _make_store(adapter)
    docs = [
        Document(page_content="d1", metadata={"x": 1}),
        Document(page_content="d2", metadata={"x": 2}),
    ]
    ids = store.add_documents(docs)

    assert ids == ["id-1", "id-2"]
    assert captured["texts"] == ["d1", "d2"]
    assert captured["metadatas"] == [{"x": 1}, {"x": 2}]


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def test_from_texts_constructs_store_and_calls_add_texts(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    def fake_add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        captured["self"] = self
        captured["texts"] = list(texts)
        captured["metadatas"] = metadatas
        captured["ids"] = ids
        return ids or []

    monkeypatch.setattr(
        langchain_module.CorpusLangChainVectorStore,
        "add_texts",
        fake_add_texts,
    )

    texts = ["t1", "t2"]
    metas = [{"m": 1}, {"m": 2}]
    ids = ["i1", "i2"]

    store = CorpusLangChainVectorStore.from_texts(
        texts,
        corpus_adapter=adapter,
        metadatas=metas,
        ids=ids,
    )

    assert isinstance(store, CorpusLangChainVectorStore)
    assert captured["self"] is store
    assert captured["texts"] == texts
    assert captured["metadatas"] == metas
    assert captured["ids"] == ids


def test_from_documents_constructs_store_and_calls_add_documents(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    def fake_add_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        captured["self"] = self
        captured["documents"] = documents
        return ["x"]

    monkeypatch.setattr(
        langchain_module.CorpusLangChainVectorStore,
        "add_documents",
        fake_add_documents,
    )

    docs = [
        Document(page_content="a", metadata={"m": 1}),
        Document(page_content="b", metadata={"m": 2}),
    ]

    store = CorpusLangChainVectorStore.from_documents(
        docs,
        corpus_adapter=adapter,
    )

    assert isinstance(store, CorpusLangChainVectorStore)
    assert captured["self"] is store
    assert captured["documents"] == docs


# ---------------------------------------------------------------------------
# CorpusLangChainRetriever behavior
# ---------------------------------------------------------------------------


def test_retriever_delegates_to_vector_store_and_merges_kwargs() -> None:
    captured: Dict[str, Any] = {}

    class FakeStore(CorpusLangChainVectorStore):
        def __init__(self) -> None:
            # Avoid needing a real adapter here
            pass

        def similarity_search(
            self,
            query: str,
            k: int = 4,
            filter: Mapping[str, Any] | None = None,
            **kwargs: Any,
        ) -> List[Document]:
            captured["query"] = query
            captured["k"] = k
            captured["filter"] = dict(filter or {})
            captured["kwargs"] = dict(kwargs)
            return [Document(page_content="doc", metadata={"foo": "bar"})]

    store = FakeStore()
    retriever = CorpusLangChainRetriever(
        vector_store=store,
        search_kwargs={"k": 4, "filter": {"tag": "t"}},
    )

    docs = retriever.get_relevant_documents(
        "q",
        namespace="ns-ret",
    )
    assert docs and isinstance(docs[0], Document)

    assert captured["query"] == "q"
    # Call-time k should override search_kwargs if provided
    assert captured["k"] == 4
    assert captured["filter"] == {"tag": "t"}
    assert captured["kwargs"]["namespace"] == "ns-ret"


@pytest.mark.asyncio
async def test_async_retriever_delegates_to_vector_store_and_merges_kwargs() -> None:
    captured: Dict[str, Any] = {}

    class FakeStore(CorpusLangChainVectorStore):
        def __init__(self) -> None:
            pass

        async def asimilarity_search(
            self,
            query: str,
            k: int = 4,
            filter: Mapping[str, Any] | None = None,
            **kwargs: Any,
        ) -> List[Document]:
            captured["query"] = query
            captured["k"] = k
            captured["filter"] = dict(filter or {})
            captured["kwargs"] = dict(kwargs)
            return [Document(page_content="adoc", metadata={"baz": 1})]

    store = FakeStore()
    retriever = CorpusLangChainRetriever(
        vector_store=store,
        search_kwargs={"k": 2, "filter": {"tag": "x"}},
    )

    docs = await retriever.aget_relevant_documents(
        "async-q",
        namespace="ns-ret-async",
    )
    assert docs and isinstance(docs[0], Document)
    assert captured["query"] == "async-q"
    assert captured["k"] == 2
    assert captured["filter"] == {"tag": "x"}
    assert captured["kwargs"]["namespace"] == "ns-ret-async"


# ---------------------------------------------------------------------------
# Bad translator result validation
# ---------------------------------------------------------------------------


def test_similarity_search_rejects_non_queryresult_from_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    similarity_search should raise VectorAdapterError when translator.query
    returns a non-QueryResult type.
    """
    # Ensure QueryResult is a distinct type we can check isinstance against
    class FakeQueryResult:
        def __init__(self, matches: Sequence[Any]) -> None:
            self.matches = list(matches)

    monkeypatch.setattr(
        langchain_module,
        "QueryResult",
        FakeQueryResult,
    )

    class BadTranslator:
        def query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return {"not": "a-query-result"}

    store = _make_store(adapter)
    store._translator = BadTranslator()  # type: ignore[assignment]  # noqa: SLF001

    with pytest.raises(VectorAdapterError) as exc_info:
        store.similarity_search("bad")

    assert "BAD_TRANSLATED_RESULT" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
