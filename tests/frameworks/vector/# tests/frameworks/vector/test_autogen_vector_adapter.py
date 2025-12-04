from __future__ import annotations

from collections.abc import Mapping, AsyncIterator
from typing import Any, Dict, List, Mapping as TMapping, Sequence, Tuple
import inspect

import pytest

import corpus_sdk.vector.framework_adapters.autogen as autogen_module
from corpus_sdk.vector.framework_adapters.autogen import (
    AutoGenDocument,
    CorpusAutoGenRetrieverTool,
    CorpusAutoGenVectorStore,
    ErrorCodes,
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
    framework_version: str | None = "autogen-fw-test",
) -> CorpusAutoGenVectorStore:
    """Construct a CorpusAutoGenVectorStore with a simple embedding function."""
    kwargs: Dict[str, Any] = {"corpus_adapter": adapter, "framework_version": framework_version}
    if with_embeddings:
        kwargs["embedding_function"] = _simple_embedding_fn
    return CorpusAutoGenVectorStore(**kwargs)


def _mock_translator_with_capture(
    captured: Dict[str, Any],
    method_name: str,
    return_value: Any,
) -> Any:
    """Sync translator stub that captures args/kwargs for a single method."""

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
    """Async translator stub that captures args/kwargs for a single method."""

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
# Translator wiring
# ---------------------------------------------------------------------------


def test_default_translator_uses_autogen_framework_label(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    The lazy _translator property should call create_vector_translator with
    framework='autogen'.
    """
    captured: Dict[str, Any] = {}

    def fake_create_vector_translator(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["args"] = args
        captured["kwargs"] = kwargs

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(
        autogen_module,
        "create_vector_translator",
        fake_create_vector_translator,
    )

    store = _make_store(adapter)
    _ = store._translator  # noqa: SLF001

    kwargs = captured["kwargs"]
    assert kwargs.get("framework") == "autogen"
    # translator=None is expected (DefaultVectorFrameworkTranslator fallback)
    assert "translator" in kwargs


# ---------------------------------------------------------------------------
# Context translation / from_autogen mapping
# ---------------------------------------------------------------------------


def test_autogen_conversation_and_extra_context_passed_to_core_ctx(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Verify that conversation + extra_context are passed to from_autogen
    with framework_version, and that resulting OperationContext is used
    as op_ctx for translator calls.
    """
    captured_ctx: Dict[str, Any] = {}
    captured_translator: Dict[str, Any] = {}

    # Patch OperationContext so isinstance(ctx, OperationContext) succeeds.
    class DummyOperationContext:
        def __init__(self, **attrs: Any) -> None:
            self.attrs = attrs

    monkeypatch.setattr(
        autogen_module,
        "OperationContext",
        DummyOperationContext,
    )

    def fake_core_ctx_from_autogen(
        conversation: Any,
        *,
        framework_version: Any = None,
        **extra: Any,
    ) -> Any:
        captured_ctx["conversation"] = conversation
        captured_ctx["framework_version"] = framework_version
        captured_ctx["extra"] = extra
        return DummyOperationContext(conversation=conversation, **extra)

    monkeypatch.setattr(
        autogen_module,
        "core_ctx_from_autogen",
        fake_core_ctx_from_autogen,
    )

    translator = _mock_translator_with_capture(
        captured_translator,
        method_name="query",
        return_value=[],
    )

    monkeypatch.setattr(
        autogen_module,
        "create_vector_translator",
        lambda *a, **k: translator,
    )

    store = _make_store(adapter, framework_version="autogen-fw-1.2.3")

    conversation = {"run_id": "r-123", "messages": ["hi"]}
    extra_ctx = {"request_id": "req-xyz", "tenant": "t-1"}

    # Needs embedding_function, which _make_store already provides
    result = store.similarity_search(
        "hello world",
        k=3,
        conversation=conversation,
        extra_context=extra_ctx,
    )
    assert isinstance(result, list)

    # Context translation inputs
    assert captured_ctx["conversation"] is conversation
    assert captured_ctx["framework_version"] == "autogen-fw-1.2.3"
    assert captured_ctx["extra"] == extra_ctx

    # Translator op_ctx should be our DummyOperationContext
    op_ctx = captured_translator.get("op_ctx")
    assert isinstance(op_ctx, DummyOperationContext)
    fw_ctx = captured_translator.get("framework_ctx", {})
    assert isinstance(fw_ctx, dict)
    # Namespace defaults to "default"
    assert fw_ctx.get("namespace") == "default"
    assert fw_ctx.get("framework_version") == "autogen-fw-1.2.3"


def test_build_ctx_rejects_non_operation_context_with_error_code(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If from_autogen returns a non-OperationContext, _build_ctx should raise
    BadRequest with ErrorCodes.BAD_OPERATION_CONTEXT.
    """
    # OperationContext type we expect
    class DummyOperationContext:
        ...

    monkeypatch.setattr(
        autogen_module,
        "OperationContext",
        DummyOperationContext,
    )

    # Return something *not* an OperationContext instance
    monkeypatch.setattr(
        autogen_module,
        "core_ctx_from_autogen",
        lambda *a, **k: object(),
    )

    store = _make_store(adapter)

    with pytest.raises(Exception) as exc_info:
        store._build_ctx(conversation={"x": 1})  # noqa: SLF001

    err = exc_info.value
    assert getattr(err, "code", None) == ErrorCodes.BAD_OPERATION_CONTEXT


def test_build_ctx_translation_failure_attaches_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If from_autogen raises, _build_ctx should attach context with
    operation='vector_context_translation'.
    """
    captured: Dict[str, Any] = {}

    def fake_core_ctx_from_autogen(*_: Any, **__: Any) -> Any:
        raise RuntimeError("boom!")

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured.update(ctx)

    monkeypatch.setattr(
        autogen_module,
        "core_ctx_from_autogen",
        fake_core_ctx_from_autogen,
    )
    monkeypatch.setattr(
        autogen_module,
        "attach_context",
        fake_attach_context,
    )

    store = _make_store(adapter)

    with pytest.raises(RuntimeError, match="boom!"):
        store._build_ctx(conversation={"run_id": "ctx-fail"})  # noqa: SLF001

    assert captured.get("framework") == "autogen"
    assert captured.get("operation") == "vector_context_translation"


# ---------------------------------------------------------------------------
# Error-context decorators (sync + async)
# ---------------------------------------------------------------------------


def test_sync_errors_include_autogen_metadata_in_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    When a sync vector op fails, with_error_context should call attach_context
    with framework='autogen' and operation starting with 'vector_'.
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(
        autogen_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingTranslator:
        def query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from autogen vector adapter")

    monkeypatch.setattr(
        autogen_module,
        "create_vector_translator",
        lambda *a, **k: FailingTranslator(),
    )

    store = _make_store(adapter)

    with pytest.raises(RuntimeError, match="test error from autogen vector adapter"):
        store.similarity_search("oops")

    assert captured_ctx
    assert captured_ctx.get("framework") == "autogen"
    assert str(captured_ctx.get("operation", "")).startswith("vector_")


@pytest.mark.asyncio
async def test_async_errors_include_autogen_metadata_in_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Same as sync error-context test but for the async query path.
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(
        autogen_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingTranslator:
        async def arun_query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test async error from autogen vector adapter")

    monkeypatch.setattr(
        autogen_module,
        "create_vector_translator",
        lambda *a, **k: FailingTranslator(),
    )

    store = _make_store(adapter)

    with pytest.raises(RuntimeError, match="test async error from autogen vector adapter"):
        await store.asimilarity_search("oops-async")

    assert captured_ctx
    assert captured_ctx.get("framework") == "autogen"
    assert str(captured_ctx.get("operation", "")).startswith("vector_")


# ---------------------------------------------------------------------------
# _from_matches / score-threshold behavior
# ---------------------------------------------------------------------------


def test_from_matches_converts_hits_to_documents_and_respects_score_threshold(
    adapter: Any,
) -> None:
    """
    _from_matches should map hit dicts into (AutoGenDocument, score) tuples
    and strip id/text fields from user metadata. score_threshold should filter.
    """
    store = _make_store(adapter)
    store.score_threshold = 0.8

    matches = [
        {
            "id": "1",
            "score": 0.75,
            "metadata": {
                store.text_field: "too low",
                store.id_field: "1",
                "foo": "bar",
            },
        },
        {
            "id": "2",
            "score": 0.95,
            "metadata": {
                store.text_field: "kept",
                store.id_field: "2",
                "baz": 42,
            },
        },
    ]

    docs_scores = store._from_matches(matches)  # noqa: SLF001
    assert len(docs_scores) == 1

    doc, score = docs_scores[0]
    assert isinstance(doc, AutoGenDocument)
    assert doc.page_content == "kept"
    # id/text removed from metadata
    assert "id" not in doc.metadata
    assert store.text_field not in doc.metadata
    assert doc.metadata == {"baz": 42}
    assert score == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Similarity search (sync + async)
# ---------------------------------------------------------------------------


def test_similarity_search_uses_translator_and_coerce_hits(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    similarity_search() should:

    - Build a raw query mapping,
    - Pass it into translator.query,
    - Use _coerce_hits_safe/_extract_matches_from_result,
    - Convert to AutoGenDocument instances.
    """
    captured: Dict[str, Any] = {}
    raw_result = {"matches": "ignored-by-patch"}

    translator = _mock_translator_with_capture(
        captured,
        method_name="query",
        return_value=raw_result,
    )

    def fake_coerce_hits_safe(result: Any) -> List[TMapping[str, Any]]:
        assert result is raw_result
        return [
            {
                "id": "doc-1",
                "score": 0.9,
                "metadata": {
                    "page_content": "hello",
                    "id": "doc-1",
                    "tag": "t1",
                },
            }
        ]

    monkeypatch.setattr(
        autogen_module,
        "create_vector_translator",
        lambda *a, **k: translator,
    )
    monkeypatch.setattr(
        autogen_module,
        "_coerce_hits_safe",
        fake_coerce_hits_safe,
    )

    store = _make_store(adapter)

    docs = store.similarity_search("q-text", k=1, filter={"tag": "t1"})
    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, AutoGenDocument)
    assert doc.page_content == "hello"
    assert doc.metadata == {"tag": "t1"}

    # Raw query shape sanity
    assert "args" in captured
    raw_query = captured["args"][0]
    assert isinstance(raw_query, Mapping)
    assert raw_query["top_k"] == 1
    assert raw_query["filters"] == {"tag": "t1"}
    assert raw_query["include_metadata"] is True
    assert raw_query["include_vectors"] is False


@pytest.mark.asyncio
async def test_async_similarity_search_uses_translator_and_coerce_hits(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    asimilarity_search() should mirror similarity_search but via arun_query.
    """
    captured: Dict[str, Any] = {}
    raw_result = {"matches": "ignored-by-patch"}

    translator = _mock_async_translator_with_capture(
        captured,
        method_name="arun_query",
        return_value=raw_result,
    )

    def fake_coerce_hits_safe(result: Any) -> List[TMapping[str, Any]]:
        assert result is raw_result
        return [
            {
                "id": "doc-2",
                "score": 0.88,
                "metadata": {
                    "page_content": "async-hello",
                    "id": "doc-2",
                    "tag": "t2",
                },
            }
        ]

    monkeypatch.setattr(
        autogen_module,
        "create_vector_translator",
        lambda *a, **k: translator,
    )
    monkeypatch.setattr(
        autogen_module,
        "_coerce_hits_safe",
        fake_coerce_hits_safe,
    )

    store = _make_store(adapter)

    docs = await store.asimilarity_search("async-q", k=2, filter={"tag": "t2"})
    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, AutoGenDocument)
    assert doc.page_content == "async-hello"
    assert doc.metadata == {"tag": "t2"}

    raw_query = captured["args"][0]
    assert raw_query["top_k"] == 2
    assert raw_query["filters"] == {"tag": "t2"}


def test_similarity_search_with_score_returns_tuples(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    similarity_search_with_score() should return (AutoGenDocument, score) tuples.
    """
    raw_result = {"matches": "ignored"}
    captured: Dict[str, Any] = {}

    translator = _mock_translator_with_capture(
        captured,
        method_name="query",
        return_value=raw_result,
    )

    def fake_coerce_hits_safe(result: Any) -> List[TMapping[str, Any]]:
        assert result is raw_result
        return [
            {
                "id": "doc-1",
                "score": 0.5,
                "metadata": {"page_content": "lo", "id": "doc-1"},
            },
            {
                "id": "doc-2",
                "score": 0.8,
                "metadata": {"page_content": "hi", "id": "doc-2"},
            },
        ]

    monkeypatch.setattr(
        autogen_module,
        "create_vector_translator",
        lambda *a, **k: translator,
    )
    monkeypatch.setattr(
        autogen_module,
        "_coerce_hits_safe",
        fake_coerce_hits_safe,
    )

    store = _make_store(adapter)

    results = store.similarity_search_with_score("q", k=4)
    assert len(results) == 2
    doc, score = results[0]
    assert isinstance(doc, AutoGenDocument)
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Streaming similarity search (sync + async)
# ---------------------------------------------------------------------------


def test_similarity_search_stream_yields_documents(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    similarity_search_stream() should yield AutoGenDocument instances
    based on streaming chunks from translator.query_stream.
    """
    captured: Dict[str, Any] = {}

    class StreamTranslator:
        def query_stream(self, *args: Any, **kwargs: Any):  # noqa: ARG002
            captured["args"] = args
            captured.update(kwargs)
            yield {"matches": "ignored"}

    def fake_coerce_hits_safe(chunk: Any) -> List[TMapping[str, Any]]:
        assert isinstance(chunk, dict)
        return [
            {
                "id": "doc-s",
                "score": 0.99,
                "metadata": {"page_content": "stream", "id": "doc-s"},
            },
        ]

    monkeypatch.setattr(
        autogen_module,
        "create_vector_translator",
        lambda *a, **k: StreamTranslator(),
    )
    monkeypatch.setattr(
        autogen_module,
        "_coerce_hits_safe",
        fake_coerce_hits_safe,
    )

    store = _make_store(adapter)

    iterator = store.similarity_search_stream("stream-q", k=3)
    assert inspect.isgenerator(iterator) or hasattr(iterator, "__iter__")

    docs = list(iterator)
    assert docs
    assert isinstance(docs[0], AutoGenDocument)
    assert docs[0].page_content == "stream"


@pytest.mark.asyncio
async def test_async_similarity_search_stream_yields_documents(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    asimilarity_search_stream() should yield AutoGenDocument instances
    from translator.arun_query_stream.
    """
    captured: Dict[str, Any] = {}

    class AsyncStreamTranslator:
        async def arun_query_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:  # noqa: ARG002
            async def gen():
                captured["args"] = args
                captured.update(kwargs)
                yield {"matches": "ignored-async"}
            return gen()

    def fake_coerce_hits_safe(chunk: Any) -> List[TMapping[str, Any]]:
        assert isinstance(chunk, dict)
        return [
            {
                "id": "doc-as",
                "score": 0.77,
                "metadata": {"page_content": "astream", "id": "doc-as"},
            },
        ]

    monkeypatch.setattr(
        autogen_module,
        "create_vector_translator",
        lambda *a, **k: AsyncStreamTranslator(),
    )
    monkeypatch.setattr(
        autogen_module,
        "_coerce_hits_safe",
        fake_coerce_hits_safe,
    )

    store = _make_store(adapter)

    aiter = store.asimilarity_search_stream("async-stream-q", k=2)
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    seen: List[AutoGenDocument] = []
    async for doc in aiter:
        seen.append(doc)
        break

    assert seen
    assert isinstance(seen[0], AutoGenDocument)
    assert seen[0].page_content == "astream"


# ---------------------------------------------------------------------------
# Low-level query (precomputed embeddings)
# ---------------------------------------------------------------------------


def test_low_level_query_returns_raw_hit_mappings(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    query() should accept a precomputed embedding and return canonical hit mappings.
    """
    captured: Dict[str, Any] = {}
    raw_result = {"matches": "ignored"}

    translator = _mock_translator_with_capture(
        captured,
        method_name="query",
        return_value=raw_result,
    )

    def fake_coerce_hits_safe(result: Any) -> List[TMapping[str, Any]]:
        assert result is raw_result
        return [{"id": "raw-1", "score": 0.1, "metadata": {}}]

    monkeypatch.setattr(
        autogen_module,
        "create_vector_translator",
        lambda *a, **k: translator,
    )
    monkeypatch.setattr(
        autogen_module,
        "_coerce_hits_safe",
        fake_coerce_hits_safe,
    )

    store = _make_store(adapter, with_embeddings=False)

    hits = store.query([0.0, 1.0, 2.0, 3.0], k=5, include_vectors=True)
    assert isinstance(hits, list)
    assert hits and isinstance(hits[0], Mapping)

    raw_query = captured["args"][0]
    assert raw_query["include_vectors"] is True
    assert raw_query["top_k"] == 5


# ---------------------------------------------------------------------------
# MMR search (sync + async)
# ---------------------------------------------------------------------------


def test_mmr_search_builds_mmr_config_and_calls_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    max_marginal_relevance_search() should:

    - Build raw_query with include_vectors=True,
    - Construct MMRConfig with k and lambda_mult,
    - Pass MMRConfig to translator.query,
    - Convert matches to AutoGenDocument results.
    """
    captured: Dict[str, Any] = {}
    raw_result = {"matches": "ignored"}

    translator = _mock_translator_with_capture(
        captured,
        method_name="query",
        return_value=raw_result,
    )

    def fake_coerce_hits_safe(result: Any) -> List[TMapping[str, Any]]:
        assert result is raw_result
        # already MMR-selected matches
        return [
            {
                "id": "mmr-1",
                "score": 0.9,
                "metadata": {"page_content": "mmr-doc", "id": "mmr-1"},
            }
        ]

    monkeypatch.setattr(
        autogen_module,
        "create_vector_translator",
        lambda *a, **k: translator,
    )
    monkeypatch.setattr(
        autogen_module,
        "_coerce_hits_safe",
        fake_coerce_hits_safe,
    )

    store = _make_store(adapter)

    docs = store.max_marginal_relevance_search(
        "mmr-q",
        k=3,
        lambda_mult=0.7,
        fetch_k=10,
    )
    assert len(docs) <= 3
    assert docs and isinstance(docs[0], AutoGenDocument)

    raw_query = captured["args"][0]
    assert raw_query["include_vectors"] is True
    assert raw_query["top_k"] == 10

    mmr_config = captured.get("mmr_config")
    assert mmr_config is not None
    assert getattr(mmr_config, "enabled", False) is True
    assert getattr(mmr_config, "k", None) == 3
    assert getattr(mmr_config, "lambda_mult", None) == 0.7


@pytest.mark.asyncio
async def test_async_mmr_search_builds_mmr_config_and_calls_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    amax_marginal_relevance_search() should mirror the sync MMR wiring via arun_query.
    """
    captured: Dict[str, Any] = {}
    raw_result = {"matches": "ignored-async"}

    translator = _mock_async_translator_with_capture(
        captured,
        method_name="arun_query",
        return_value=raw_result,
    )

    def fake_coerce_hits_safe(result: Any) -> List[TMapping[str, Any]]:
        assert result is raw_result
        return [
            {
                "id": "ammr-1",
                "score": 0.5,
                "metadata": {"page_content": "ammr-doc", "id": "ammr-1"},
            }
        ]

    monkeypatch.setattr(
        autogen_module,
        "create_vector_translator",
        lambda *a, **k: translator,
    )
    monkeypatch.setattr(
        autogen_module,
        "_coerce_hits_safe",
        fake_coerce_hits_safe,
    )

    store = _make_store(adapter)

    docs = await store.amax_marginal_relevance_search(
        "ammr-q",
        k=2,
        lambda_mult=0.3,
        fetch_k=9,
    )
    assert docs and isinstance(docs[0], AutoGenDocument)

    raw_query = captured["args"][0]
    assert raw_query["top_k"] == 9
    assert raw_query["include_vectors"] is True

    mmr_config = captured.get("mmr_config")
    assert getattr(mmr_config, "k", None) == 2
    assert getattr(mmr_config, "lambda_mult", None) == 0.3


# ---------------------------------------------------------------------------
# Delete API (sync + async)
# ---------------------------------------------------------------------------


def test_delete_requires_ids_or_filter(
    adapter: Any,
) -> None:
    """
    delete() should raise BadRequest with BAD_DELETE_REQUEST when neither
    ids nor filter is provided.
    """
    store = _make_store(adapter)

    with pytest.raises(Exception) as exc_info:
        store.delete()

    err = exc_info.value
    assert getattr(err, "code", None) == ErrorCodes.BAD_DELETE_REQUEST


def test_delete_prefers_filter_over_ids_and_calls_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    When filter is provided, it should take precedence over ids and be passed
    as a mapping to translator.delete.
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def delete(self, raw_filter_or_ids: Any, *, op_ctx: Any, framework_ctx: Mapping[str, Any]) -> None:
            captured["arg"] = raw_filter_or_ids
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx)

    monkeypatch.setattr(
        autogen_module,
        "create_vector_translator",
        lambda *a, **k: DummyTranslator(),
    )

    store = _make_store(adapter)

    store.delete(
        ids=["id-1", "id-2"],
        filter={"tag": "v"},
        namespace="ns-del",
    )

    assert captured["arg"] == {"tag": "v"}
    fw_ctx = captured["framework_ctx"]
    assert fw_ctx.get("namespace") == "ns-del"


@pytest.mark.asyncio
async def test_async_delete_prefers_filter_and_calls_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    adelete() should mirror delete() but via translator.arun_delete.
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_delete(self, raw_filter_or_ids: Any, *, op_ctx: Any, framework_ctx: Mapping[str, Any]) -> None:
            captured["arg"] = raw_filter_or_ids
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx)

    monkeypatch.setattr(
        autogen_module,
        "create_vector_translator",
        lambda *a, **k: DummyTranslator(),
    )

    store = _make_store(adapter)

    await store.adelete(
        ids=["x"],
        filter={"foo": 1},
        namespace="ns-adel",
    )

    assert captured["arg"] == {"foo": 1}
    assert captured["framework_ctx"]["namespace"] == "ns-adel"


# ---------------------------------------------------------------------------
# Capabilities / health passthrough
# ---------------------------------------------------------------------------


def test_capabilities_and_health_basic() -> None:
    """
    capabilities() and health() should thinly delegate to the underlying adapter.
    """

    class DummyAdapter:
        def capabilities(self) -> Dict[str, Any]:
            return {"features": ["x", "y"]}

        def health(self) -> Dict[str, Any]:
            return {"status": "ok"}

    store = _make_store(DummyAdapter(), with_embeddings=False)

    caps = store.capabilities()
    health = store.health()

    assert isinstance(caps, Mapping)
    assert caps.get("features") == ["x", "y"]
    assert isinstance(health, Mapping)
    assert health.get("status") == "ok"


@pytest.mark.asyncio
async def test_async_capabilities_and_health_basic() -> None:
    """
    acapabilities() and ahealth() should use async adapter methods when available,
    otherwise fall back to wrapping sync methods.
    """

    class DummyAdapter:
        async def acapabilities(self) -> Dict[str, Any]:
            return {"async_features": True}

        async def ahealth(self) -> Dict[str, Any]:
            return {"async_status": "ok"}

    store = _make_store(DummyAdapter(), with_embeddings=False)

    acaps = await store.acapabilities()
    ahealth = await store.ahealth()

    assert acaps.get("async_features") is True
    assert ahealth.get("async_status") == "ok"


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def test_from_texts_constructs_store_and_adds_texts(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    from_texts() should construct a store and call add_texts with the given
    texts/metadatas/ids.
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def upsert(self, *, raw_documents: List[Mapping[str, Any]], op_ctx: Any, framework_ctx: Mapping[str, Any]) -> None:
            captured["raw_documents"] = list(raw_documents)
            captured["framework_ctx"] = dict(framework_ctx)

    monkeypatch.setattr(
        autogen_module,
        "create_vector_translator",
        lambda *a, **k: DummyTranslator(),
    )

    texts = ["t1", "t2"]
    metadatas = [{"m": 1}, {"m": 2}]
    ids = ["id-1", "id-2"]

    store = CorpusAutoGenVectorStore.from_texts(
        texts,
        corpus_adapter=adapter,
        metadatas=metadatas,
        ids=ids,
        embedding_function=_simple_embedding_fn,
    )

    assert isinstance(store, CorpusAutoGenVectorStore)
    raw_docs = captured["raw_documents"]
    assert len(raw_docs) == 2
    assert raw_docs[0]["id"] == "id-1"
    assert raw_docs[0]["metadata"]["page_content"] == "t1"
    assert raw_docs[0]["metadata"]["id"] == "id-1"


def test_from_documents_constructs_store_and_adds_documents(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    from_documents() should construct a store and add AutoGenDocument instances.
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def upsert(self, *, raw_documents: List[Mapping[str, Any]], op_ctx: Any, framework_ctx: Mapping[str, Any]) -> None:
            captured["raw_documents"] = list(raw_documents)
            captured["framework_ctx"] = dict(framework_ctx)

    monkeypatch.setattr(
        autogen_module,
        "create_vector_translator",
        lambda *a, **k: DummyTranslator(),
    )

    docs = [
        AutoGenDocument(page_content="d1", metadata={"x": 1}),
        AutoGenDocument(page_content="d2", metadata={"x": 2}),
    ]

    store = CorpusAutoGenVectorStore.from_documents(
        docs,
        corpus_adapter=adapter,
        embedding_function=_simple_embedding_fn,
    )

    assert isinstance(store, CorpusAutoGenVectorStore)
    raw_docs = captured["raw_documents"]
    assert len(raw_docs) == 2
    assert raw_docs[0]["metadata"]["page_content"] == "d1"


# ---------------------------------------------------------------------------
# CorpusAutoGenRetrieverTool behavior
# ---------------------------------------------------------------------------


def test_retriever_tool_delegates_to_vector_store_and_shapes_output(
    adapter: Any,
) -> None:
    """
    CorpusAutoGenRetrieverTool.__call__ should:

    - Forward query, k, filter, conversation, extra_context to vector_store.similarity_search,
    - Merge search_kwargs with call-time kwargs,
    - Return list of dicts with page_content/metadata.
    """
    captured: Dict[str, Any] = {}

    class FakeStore(CorpusAutoGenVectorStore):
        def __init__(self) -> None:
            # Avoid needing a real adapter/translator here
            pass

        def similarity_search(
            self,
            query: str,
            k: int = 4,
            filter: TMapping[str, Any] | None = None,
            *,
            conversation: Any | None = None,
            extra_context: TMapping[str, Any] | None = None,
            **kwargs: Any,
        ) -> List[AutoGenDocument]:
            captured["query"] = query
            captured["k"] = k
            captured["filter"] = dict(filter or {})
            captured["conversation"] = conversation
            captured["extra_context"] = dict(extra_context or {})
            captured["kwargs"] = dict(kwargs)
            return [
                AutoGenDocument(page_content="doc-1", metadata={"foo": "bar"}),
            ]

    store = FakeStore()
    tool = CorpusAutoGenRetrieverTool(
        vector_store=store,
        name="corpus_vector_search",
        description="desc",
        search_kwargs={"k": 4, "some_flag": True},
    )

    conv = {"run_id": "tool-run"}
    result = tool(
        "tool-query",
        k=2,  # override search_kwargs
        conversation=conv,
        extra_context={"rid": "123"},
        filter={"tag": "v"},
        extra_kw="value",
    )

    assert captured["query"] == "tool-query"
    assert captured["k"] == 2
    assert captured["filter"] == {"tag": "v"}
    assert captured["conversation"] is conv
    assert captured["extra_context"] == {"rid": "123"}
    # merged kwargs: search_kwargs + call-time kwargs
    assert captured["kwargs"]["some_flag"] is True
    assert captured["kwargs"]["extra_kw"] == "value"

    # Output should be list of dicts, not AutoGenDocument
    assert isinstance(result, list)
    assert result and isinstance(result[0], dict)
    assert result[0]["page_content"] == "doc-1"
    assert result[0]["metadata"] == {"foo": "bar"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
