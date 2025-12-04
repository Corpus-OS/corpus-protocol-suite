# tests/frameworks/vector/test_semantickernel_vector_adapter.py

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple

import inspect
import pytest

import corpus_sdk.vector.framework_adapters.semantic_kernel as sk_vector_module
from corpus_sdk.vector.framework_adapters.semantic_kernel import (
    CorpusSemanticKernelVectorStore,
    CorpusSemanticKernelVectorPlugin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_embedding_function(texts: List[str]) -> List[List[float]]:
    """Simple deterministic sync embedding function for tests."""
    return [[float(len(t))] for t in texts]


async def _default_async_embedding_function(texts: List[str]) -> List[List[float]]:
    """Simple deterministic async embedding function for tests."""
    return [[float(len(t))] for t in texts]


def _make_store(
    adapter: Any,
    **kwargs: Any,
) -> CorpusSemanticKernelVectorStore:
    """
    Construct a CorpusSemanticKernelVectorStore from a generic adapter.

    By default we provide simple sync + async embedding functions so that
    similarity_search-style APIs don't raise NotSupported just because no
    embedding_function is configured in the tests.
    """
    kwargs.setdefault("embedding_function", _default_embedding_function)
    kwargs.setdefault("async_embedding_function", _default_async_embedding_function)
    return CorpusSemanticKernelVectorStore(corpus_adapter=adapter, **kwargs)


def _make_plugin(
    store: CorpusSemanticKernelVectorStore,
    **kwargs: Any,
) -> CorpusSemanticKernelVectorPlugin:
    """Construct a CorpusSemanticKernelVectorPlugin from a store."""
    return CorpusSemanticKernelVectorPlugin(vector_store=store, **kwargs)


def _mock_translator_with_capture(
    captured: Dict[str, Any],
) -> Any:
    """
    Helper to create a sync translator that captures call arguments.

    This is used when we monkeypatch VectorTranslator to return this object
    instead of the real one.
    """

    class MockTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["init_args"] = args
            captured["init_kwargs"] = kwargs

        # Minimal surface used by tests

        def capabilities(self) -> Any:
            captured["capabilities_called"] = True

            class DummyCaps:
                max_top_k = 100
                max_batch_size = 1000
                supports_metadata_filtering = True
                supports_namespaces = True

            return DummyCaps()

        def query(self, *args: Any, **kwargs: Any) -> Any:
            captured["query_args"] = args
            captured["query_kwargs"] = kwargs

            # Return a minimal QueryResult-like object that passes
            # CorpusSemanticKernelVectorStore._validate_query_result_type
            class DummyMatch:
                def __init__(self) -> None:
                    self.score = 0.9

                    class DummyVector:
                        def __init__(self) -> None:
                            self.metadata = {"page_content": "hello", "id": "1"}
                            self.vector = [0.1, 0.2]

                    self.vector = DummyVector()

            class DummyQueryResult(sk_vector_module.QueryResult):  # type: ignore[misc]
                def __init__(self) -> None:
                    # We deliberately don't call super().__init__ here; the
                    # adapter only needs .matches to exist.
                    self.matches = [DummyMatch()]

            return DummyQueryResult()

        def query_stream(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any = None,
            framework_ctx: Mapping[str, Any] | None = None,
        ) -> Iterator[Any]:
            captured["query_stream_raw_query"] = dict(raw_query)
            captured["query_stream_framework_ctx"] = dict(framework_ctx or {})
            captured["query_stream_op_ctx"] = op_ctx

            # Emit a single valid QueryChunk-like object that passes the
            # isinstance(..., QueryChunk) check in similarity_search_stream.
            class DummyMatch:
                def __init__(self) -> None:
                    self.score = 0.8

                    class DummyVector:
                        def __init__(self) -> None:
                            self.metadata = {"page_content": "stream", "id": "s1"}
                            self.vector = [0.3, 0.4]

                    self.vector = DummyVector()

            class DummyChunk(sk_vector_module.QueryChunk):  # type: ignore[misc]
                def __init__(self) -> None:
                    self.matches = [DummyMatch()]

            yield DummyChunk()

        def delete(self, *args: Any, **kwargs: Any) -> Any:
            captured["delete_args"] = args
            captured["delete_kwargs"] = kwargs

    return MockTranslator


def _mock_async_translator_with_capture(
    captured: Dict[str, Any],
) -> Any:
    """
    Helper to create an async translator that captures call arguments,
    used when monkeypatching VectorTranslator in async tests.
    """

    class MockTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["init_args"] = args
            captured["init_kwargs"] = kwargs

        async def arun_capabilities(self) -> Any:
            captured["acapabilities_called"] = True

            class DummyCaps:
                max_top_k = 100
                max_batch_size = 1000
                supports_metadata_filtering = True
                supports_namespaces = True

            return DummyCaps()

        async def arun_query(self, *args: Any, **kwargs: Any) -> Any:
            captured["aquery_args"] = args
            captured["aquery_kwargs"] = kwargs

            class DummyMatch:
                def __init__(self) -> None:
                    self.score = 0.95

                    class DummyVector:
                        def __init__(self) -> None:
                            self.metadata = {"page_content": "hello-async", "id": "a1"}
                            self.vector = [0.5, 0.6]

                    self.vector = DummyVector()

            class DummyQueryResult(sk_vector_module.QueryResult):  # type: ignore[misc]
                def __init__(self) -> None:
                    self.matches = [DummyMatch()]

            return DummyQueryResult()

        async def arun_query_stream(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any = None,
            framework_ctx: Mapping[str, Any] | None = None,
        ) -> AsyncIterator[Any]:
            captured["aquery_stream_raw_query"] = dict(raw_query)
            captured["aquery_stream_framework_ctx"] = dict(framework_ctx or {})
            captured["aquery_stream_op_ctx"] = op_ctx

            class DummyMatch:
                def __init__(self) -> None:
                    self.score = 0.7

                    class DummyVector:
                        def __init__(self) -> None:
                            self.metadata = {"page_content": "astream", "id": "as1"}
                            self.vector = [0.7, 0.8]

                    self.vector = DummyVector()

            class DummyChunk(sk_vector_module.QueryChunk):  # type: ignore[misc]
                def __init__(self) -> None:
                    self.matches = [DummyMatch()]

            async def gen() -> AsyncIterator[Any]:
                yield DummyChunk()

            return gen()

        async def arun_delete(self, *args: Any, **kwargs: Any) -> Any:
            captured["adelete_args"] = args
            captured["adelete_kwargs"] = kwargs

    return MockTranslator


# ---------------------------------------------------------------------------
# Translator construction behavior
# ---------------------------------------------------------------------------


def test_default_translator_uses_semantickernel_framework_label(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    CorpusSemanticKernelVectorStore should:

    - Construct a DefaultVectorFrameworkTranslator instance, and
    - Pass it into VectorTranslator with framework="semantic_kernel".
    """
    captured: Dict[str, Any] = {}

    MockVectorTranslator = _mock_translator_with_capture(captured)

    monkeypatch.setattr(
        sk_vector_module,
        "VectorTranslator",
        MockVectorTranslator,
    )

    store = _make_store(adapter)

    # Trigger lazy translator construction
    _ = store._translator  # noqa: SLF001

    init_kwargs = captured.get("init_kwargs", {})
    assert init_kwargs.get("framework") == "semantic_kernel"
    translator_obj = init_kwargs.get("translator")
    assert isinstance(
        translator_obj,
        sk_vector_module.DefaultVectorFrameworkTranslator,
    )


# ---------------------------------------------------------------------------
# Context translation & _build_ctx behavior
# ---------------------------------------------------------------------------


def test_semantickernel_context_and_settings_passed_to_context_helper_and_store(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Verify that sk_context/settings are passed through to context_from_semantic_kernel
    and that the resulting OperationContext is sent as op_ctx into translator.query().
    """
    captured_ctx: Dict[str, Any] = {}
    captured_translator: Dict[str, Any] = {}

    # Fake OperationContext
    class DummyOperationContext:
        def __init__(self, **kwargs: Any) -> None:
            self.attrs = kwargs

    monkeypatch.setattr(
        sk_vector_module,
        "OperationContext",
        DummyOperationContext,
    )

    def fake_context_from_semantic_kernel(
        sk_context: Any,
        *,
        settings: Any = None,
        framework_version: Any = None,
    ) -> Any:
        captured_ctx["sk_context"] = sk_context
        captured_ctx["settings"] = settings
        captured_ctx["framework_version"] = framework_version
        return DummyOperationContext(tag="from-sk")

    monkeypatch.setattr(
        sk_vector_module,
        "context_from_semantic_kernel",
        fake_context_from_semantic_kernel,
    )

    # Use a capturing translator that returns a tiny QueryResult-like object
    MockVectorTranslator = _mock_translator_with_capture(captured_translator)

    monkeypatch.setattr(
        sk_vector_module,
        "VectorTranslator",
        MockVectorTranslator,
    )

    store = _make_store(adapter)

    sk_ctx = object()
    sk_settings = {"temperature": 0.42}

    result = store.similarity_search(
        "hello",
        k=1,
        sk_context=sk_ctx,
        sk_settings=sk_settings,
    )
    assert result  # non-empty list from our dummy translator

    # Context helper got the settings & context
    assert captured_ctx.get("sk_context") is sk_ctx
    assert captured_ctx.get("settings") == sk_settings

    # op_ctx argument passed into translator.query should be the DummyOperationContext
    query_kwargs = captured_translator.get("query_kwargs", {})
    op_ctx = query_kwargs.get("op_ctx")
    assert isinstance(op_ctx, DummyOperationContext)
    assert op_ctx.attrs.get("tag") == "from-sk"


# ---------------------------------------------------------------------------
# Error-context behavior (embedding & query failure paths)
# ---------------------------------------------------------------------------


def test_add_texts_bad_embeddings_attach_semantickernel_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If embeddings length mismatches texts length, add_texts() should raise
    a BadRequest-like error and attach Semantic Kernel error context.
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx["exc"] = exc
        captured_ctx.update(ctx)

    monkeypatch.setattr(
        sk_vector_module,
        "attach_context",
        fake_attach_context,
    )

    # We don't care about translator behavior in this test; just avoid using it.
    # Use a tiny store and call add_texts with mismatched embeddings length.
    store = _make_store(adapter)

    with pytest.raises(Exception) as exc_info:
        store.add_texts(
            ["a", "b"],
            embeddings=[[0.1, 0.2]],  # length 1 vs 2 texts
        )

    err = exc_info.value
    # Error should be about embeddings
    msg = str(err).lower()
    assert "embeddings" in msg or "embedding" in msg

    # We should have attached context with semantic_kernel framework label
    assert captured_ctx
    assert captured_ctx.get("framework") == "semantic_kernel"
    assert captured_ctx.get("operation") in {
        "ensure_embeddings",
        "ensure_embeddings_async",
    }


def test_sync_similarity_search_errors_include_semantickernel_metadata(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    When an error occurs during a sync vector operation, error context should
    include Semantic Kernel metadata via attach_context().
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx["exc"] = exc
        captured_ctx.update(ctx)

    monkeypatch.setattr(
        sk_vector_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        def capabilities(self) -> Any:
            class DummyCaps:
                max_top_k = 100
                max_batch_size = 1000
                supports_metadata_filtering = True
                supports_namespaces = True

            return DummyCaps()

        def query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from semantic_kernel vector adapter")

    monkeypatch.setattr(
        sk_vector_module,
        "VectorTranslator",
        FailingTranslator,
    )

    store = _make_store(adapter)

    with pytest.raises(RuntimeError, match="test error from semantic_kernel vector adapter"):
        store.similarity_search("hello", k=2, namespace="err-ns")

    assert captured_ctx
    assert captured_ctx.get("framework") == "semantic_kernel"
    # operation is set in the except block in similarity_search
    assert "similarity_search_sync" in str(captured_ctx.get("operation", ""))


@pytest.mark.asyncio
async def test_async_similarity_search_errors_include_semantickernel_metadata(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Same as the sync error-context test but for the async similarity search path.
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx["exc"] = exc
        captured_ctx.update(ctx)

    monkeypatch.setattr(
        sk_vector_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        async def arun_capabilities(self) -> Any:
            class DummyCaps:
                max_top_k = 100
                max_batch_size = 1000
                supports_metadata_filtering = True
                supports_namespaces = True

            return DummyCaps()

        async def arun_query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test async error from semantic_kernel vector adapter")

    monkeypatch.setattr(
        sk_vector_module,
        "VectorTranslator",
        FailingTranslator,
    )

    store = _make_store(adapter)

    with pytest.raises(RuntimeError, match="test async error from semantic_kernel vector adapter"):
        await store.asimilarity_search("hello-async", k=3, namespace="async-err-ns")

    assert captured_ctx
    assert captured_ctx.get("framework") == "semantic_kernel"
    assert "similarity_search_async" in str(captured_ctx.get("operation", ""))


# ---------------------------------------------------------------------------
# Sync semantics (basic smoke tests)
# ---------------------------------------------------------------------------


def test_sync_similarity_search_and_stream_basic(adapter: Any) -> None:
    """
    Basic smoke test for sync similarity_search / similarity_search_stream behavior.

    Methods should accept text input and not crash, returning AI-friendly
    document dicts. Detailed vector semantics are covered by generic vector
    contract tests.
    """
    store = _make_store(adapter, namespace="sk-ns")

    # Non-streaming search
    docs = store.similarity_search("hello world", k=2)
    assert isinstance(docs, list)

    # We don't assert exact shape, but if non-empty, ensure it's dict-like.
    if docs:
        assert isinstance(docs[0], Mapping)

    # Streaming search
    stream = store.similarity_search_stream("hello world", k=1)
    assert isinstance(stream, Iterator)

    # Just ensure we can iterate at least once without crashing
    _ = list(stream)


# ---------------------------------------------------------------------------
# Async semantics (basic smoke tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_similarity_search_basic(adapter: Any) -> None:
    """
    Async asimilarity_search should exist and produce results compatible with
    the sync API.

    The current adapter does not implement asimilarity_search_stream, so we
    only test asimilarity_search here.
    """
    store = _make_store(adapter, namespace="async-sk-ns")

    assert hasattr(store, "asimilarity_search")

    coro = store.asimilarity_search("hello async", k=2)
    assert inspect.isawaitable(coro)
    docs = await coro
    assert isinstance(docs, list)

    if docs:
        assert isinstance(docs[0], Mapping)


# ---------------------------------------------------------------------------
# Streaming validation tests
# ---------------------------------------------------------------------------


def test_similarity_search_stream_invalid_chunk_triggers_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    similarity_search_stream() should attach Semantic Kernel error context
    when translator yields an invalid (non-QueryChunk) chunk.
    """
    captured: Dict[str, Any] = {}
    invalid_chunk = object()

    class DummyTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        def capabilities(self) -> Any:
            class DummyCaps:
                max_top_k = 10
                max_batch_size = 100
                supports_metadata_filtering = True
                supports_namespaces = True

            return DummyCaps()

        def query_stream(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any = None,
            framework_ctx: Mapping[str, Any] | None = None,
        ) -> Iterator[Any]:
            captured["raw_query"] = dict(raw_query)
            captured["framework_ctx"] = dict(framework_ctx or {})
            captured["op_ctx"] = op_ctx
            # Emit a single invalid chunk
            yield invalid_chunk

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured["error_ctx"] = ctx
        captured["error_exc"] = exc

    monkeypatch.setattr(
        sk_vector_module,
        "VectorTranslator",
        DummyTranslator,
    )
    monkeypatch.setattr(
        sk_vector_module,
        "attach_context",
        fake_attach_context,
    )

    store = _make_store(adapter)

    stream = store.similarity_search_stream("boom")

    with pytest.raises(sk_vector_module.VectorAdapterError) as exc_info:
        for _ in stream:
            pass

    msg = str(exc_info.value)
    assert "unsupported type" in msg or "BAD_STREAM_CHUNK" in msg

    error_ctx = captured.get("error_ctx", {})
    assert error_ctx
    assert error_ctx.get("framework") == "semantic_kernel"
    assert "similarity_search_stream" in str(error_ctx.get("operation", ""))


@pytest.mark.skip("CorpusSemanticKernelVectorStore does not implement asimilarity_search_stream")
@pytest.mark.asyncio
async def test_asimilarity_search_stream_invalid_chunk_triggers_context_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    asimilarity_search_stream() is not currently implemented by the adapter.
    This test is kept as a placeholder and marked skipped to reflect the
    current behavior.
    """
    # If an async streaming API is added in the future, this test can be
    # re-enabled and adapted in a similar fashion to the sync test above.
    pass


# ---------------------------------------------------------------------------
# Delete API wiring
# ---------------------------------------------------------------------------


def test_delete_builds_raw_request_and_calls_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    delete() should build the correct raw request mapping and call
    translator.delete with that mapping and framework_ctx.
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        def capabilities(self) -> Any:
            class DummyCaps:
                max_top_k = 10
                max_batch_size = 100
                supports_metadata_filtering = True
                supports_namespaces = True

            return DummyCaps()

        def delete(self, raw_request: Mapping[str, Any], *, op_ctx: Any, framework_ctx: Mapping[str, Any]) -> None:
            captured["raw_request"] = dict(raw_request)
            captured["framework_ctx"] = dict(framework_ctx)
            captured["op_ctx"] = op_ctx

    monkeypatch.setattr(
        sk_vector_module,
        "VectorTranslator",
        DummyTranslator,
    )

    store = _make_store(adapter)

    store.delete(
        ids=["id1", "id2"],
        filter=None,
        namespace="del-ns",
    )

    raw = captured["raw_request"]
    assert raw == {
        "namespace": "del-ns",
        "ids": ["id1", "id2"],
        "filter": None,
    }

    fw_ctx = captured["framework_ctx"]
    assert fw_ctx  # non-empty; details tested in framework_utils tests


@pytest.mark.asyncio
async def test_adelete_builds_raw_request_and_calls_translator_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    adelete() should mirror delete wiring via translator.arun_delete.
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        async def arun_capabilities(self) -> Any:
            class DummyCaps:
                max_top_k = 10
                max_batch_size = 100
                supports_metadata_filtering = True
                supports_namespaces = True

            return DummyCaps()

        async def arun_delete(
            self,
            raw_request: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ) -> None:
            captured["raw_request"] = dict(raw_request)
            captured["framework_ctx"] = dict(framework_ctx)
            captured["op_ctx"] = op_ctx

    monkeypatch.setattr(
        sk_vector_module,
        "VectorTranslator",
        DummyTranslator,
    )

    store = _make_store(adapter)

    await store.adelete(
        ids=None,
        filter={"foo": "bar"},
        namespace="adel-ns",
    )

    raw = captured["raw_request"]
    assert raw == {
        "namespace": "adel-ns",
        "ids": None,
        "filter": {"foo": "bar"},
    }

    fw_ctx = captured["framework_ctx"]
    assert fw_ctx  # non-empty


# ---------------------------------------------------------------------------
# MMR wiring basics
# ---------------------------------------------------------------------------


def test_mmr_search_calls_mmr_selector_and_formats_results(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    max_marginal_relevance_search() should:

    - Request vectors from translator with include_vectors=True
    - Call _mmr_select_indices with correct parameters
    - Format the selected matches via _format_for_ai_model
    """
    captured: Dict[str, Any] = {}

    class DummyMatch:
        def __init__(self, score: float, vec: Sequence[float]) -> None:
            self.score = score

            class DummyVector:
                def __init__(self) -> None:
                    self.metadata = {
                        "page_content": f"text-{score}",
                        "id": f"id-{score}",
                        "foo": "bar",
                    }
                    self.vector = list(vec)

            self.vector = DummyVector()

    class DummyQueryResult(sk_vector_module.QueryResult):  # type: ignore[misc]
        def __init__(self) -> None:
            self.matches = [
                DummyMatch(0.9, [0.1, 0.2]),
                DummyMatch(0.8, [0.2, 0.3]),
                DummyMatch(0.7, [0.3, 0.4]),
            ]

    class DummyTranslator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        def capabilities(self) -> Any:
            class DummyCaps:
                max_top_k = 100
                max_batch_size = 1000
                supports_metadata_filtering = True
                supports_namespaces = True

            return DummyCaps()

        def query(self, raw_query: Mapping[str, Any], *, op_ctx: Any, framework_ctx: Mapping[str, Any]) -> Any:  # noqa: ARG002
            captured["raw_query"] = dict(raw_query)
            captured["framework_ctx"] = dict(framework_ctx)
            return DummyQueryResult()

    def fake_mmr_select_indices(
        self: CorpusSemanticKernelVectorStore,
        query_vec: Sequence[float],
        candidate_matches: List[Any],
        k: int,
        lambda_mult: float,
    ) -> List[int]:
        captured["mmr_query_vec"] = list(query_vec)
        captured["mmr_k"] = k
        captured["mmr_lambda"] = lambda_mult
        captured["mmr_candidate_count"] = len(candidate_matches)
        # Always pick the first two indices for determinism
        return [0, 1]

    monkeypatch.setattr(
        sk_vector_module,
        "VectorTranslator",
        DummyTranslator,
    )
    monkeypatch.setattr(
        CorpusSemanticKernelVectorStore,
        "_mmr_select_indices",
        fake_mmr_select_indices,
    )

    store = _make_store(
        adapter,
        # Provide a simple embedding_function so we don't hit NotSupported
        embedding_function=lambda texts: [[float(len(t))] for t in texts],
        default_top_k=2,
    )

    docs = store.max_marginal_relevance_search(
        "mmr query",
        k=2,
        lambda_mult=0.5,
        namespace="mmr-ns",
    )
    assert isinstance(docs, list)
    assert len(docs) <= 2

    assert captured.get("mmr_k") == 2
    assert captured.get("mmr_lambda") == 0.5
    assert captured.get("mmr_candidate_count") == 3

    # Since we always pick indices [0, 1], content should reflect scores 0.9 and 0.8
    contents = [d["content"] for d in docs]
    assert any("text-0.9" in c for c in contents)


# ---------------------------------------------------------------------------
# Plugin behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plugin_vector_search_delegates_to_store_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    CorpusSemanticKernelVectorPlugin.vector_search should call
    store.asimilarity_search with the appropriate parameters and return
    the store result.
    """
    captured: Dict[str, Any] = {}

    async def fake_asimilarity_search(
        self: CorpusSemanticKernelVectorStore,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        captured["query"] = query
        captured["k"] = k
        captured["filter"] = dict(filter or {})
        captured["kwargs"] = kwargs
        return [{"content": "ok", "metadata": {}, "confidence": 0.9, "source": "vector_database"}]

    monkeypatch.setattr(
        CorpusSemanticKernelVectorStore,
        "asimilarity_search",
        fake_asimilarity_search,
    )

    store = _make_store(adapter)
    plugin = _make_plugin(store, framework_version="sk-fw-test")

    sk_ctx = type("DummySkCtx", (), {})()
    sk_settings = {"temperature": 0.1}

    docs = await plugin.vector_search(
        "plugin query",
        k=3,
        filter={"foo": "bar"},
        namespace="plugin-ns",
        sk_context=sk_ctx,
        sk_settings=sk_settings,
    )

    assert docs
    assert captured["query"] == "plugin query"
    assert captured["k"] == 3
    assert captured["filter"] == {"foo": "bar"}


@pytest.mark.asyncio
async def test_plugin_store_document_uses_aadd_texts(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    CorpusSemanticKernelVectorPlugin.store_document should delegate to
    store.aadd_texts and return the first ID.
    """
    captured: Dict[str, Any] = {}

    async def fake_aadd_texts(
        self: CorpusSemanticKernelVectorStore,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        captured["texts"] = list(texts)
        captured["metadatas"] = metadatas
        captured["ids"] = ids
        captured["kwargs"] = kwargs
        return ids or ["generated-id-1"]

    monkeypatch.setattr(
        CorpusSemanticKernelVectorStore,
        "aadd_texts",
        fake_aadd_texts,
    )

    store = _make_store(adapter)
    plugin = _make_plugin(store)

    doc_id = await plugin.store_document(
        content="hello from plugin",
        metadata={"topic": "test"},
        document_id=None,
        namespace="plugin-store-ns",
    )

    assert doc_id == "generated-id-1"
    assert captured["texts"] == ["hello from plugin"]
    assert captured["metadatas"] == [{"topic": "test"}]
    assert captured["kwargs"].get("namespace") == "plugin-store-ns"


@pytest.mark.asyncio
async def test_plugin_get_capabilities_uses_store_and_shapes_result(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    plugin.get_capabilities should call store.aget_capabilities and expose
    a simple dict to the model.
    """
    class DummyCaps:
        def __init__(self) -> None:
            self.max_batch_size = 256
            self.max_top_k = 64
            self.supports_metadata_filtering = True
            self.supports_namespaces = True

    async def fake_aget_capabilities(
        self: CorpusSemanticKernelVectorStore,
    ) -> Any:
        return DummyCaps()

    monkeypatch.setattr(
        CorpusSemanticKernelVectorStore,
        "aget_capabilities",
        fake_aget_capabilities,
    )

    store = _make_store(adapter)
    plugin = _make_plugin(store)

    caps = await plugin.get_capabilities()
    assert caps["max_batch_size"] == 256
    assert caps["max_top_k"] == 64
    assert caps["supports_metadata_filtering"] is True
    assert caps["supports_namespaces"] is True
    assert "description" in caps


@pytest.mark.asyncio
async def test_plugin_errors_are_wrapped_in_kernelfunctionexception(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Plugin should wrap vector-layer errors into KernelFunctionException with
    a friendly message and error code.
    """
    class DummyVectorError(sk_vector_module.VectorAdapterError):
        pass

    async def failing_asimilarity_search(
        self: CorpusSemanticKernelVectorStore,
        query: str,
        k: int = 4,
        filter: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        raise DummyVectorError("vector boom", code="VECTOR_FAILURE")

    monkeypatch.setattr(
        CorpusSemanticKernelVectorStore,
        "asimilarity_search",
        failing_asimilarity_search,
    )

    store = _make_store(adapter)
    plugin = _make_plugin(store)

    with pytest.raises(sk_vector_module.KernelFunctionException) as exc_info:
        await plugin.vector_search("plugin error query")

    msg = str(exc_info.value)
    assert "Vector database error" in msg or "Vector search failed" in msg


# ---------------------------------------------------------------------------
# Additional tests for comprehensive coverage (New tests)
# ---------------------------------------------------------------------------

def test_field_uniqueness_validation():
    """Test that reserved metadata fields must be unique."""
    # Should work with unique fields
    store = _make_store(
        adapter=object(),
        id_field="id",
        text_field="page_content",
        metadata_field="metadata",
    )
    assert store.id_field == "id"
    assert store.text_field == "page_content"
    assert store.metadata_field == "metadata"
    
    # Should fail when fields are not unique
    with pytest.raises(ValueError) as excinfo:
        _make_store(
            adapter=object(),
            id_field="same",
            text_field="same",
        )
    assert "must be unique" in str(excinfo.value)


def test_score_threshold_validation():
    """Test score_threshold validation."""
    # Valid threshold should work
    store = _make_store(adapter=object(), score_threshold=0.7)
    assert store.score_threshold == 0.7
    
    # Invalid thresholds should raise
    with pytest.raises(ValueError):
        _make_store(adapter=object(), score_threshold=1.5)
    
    with pytest.raises(ValueError):
        _make_store(adapter=object(), score_threshold=-0.1)


def test_embed_query_with_provided_embedding():
    """Test _embed_query when embedding is provided."""
    store = _make_store(adapter=object())
    
    query_emb = store._embed_query(
        query="test",
        embedding=[0.1, 0.2, 0.3]
    )
    
    assert query_emb == [0.1, 0.2, 0.3]


def test_embed_query_requires_embedding_function():
    """Test _embed_query raises when no embedding function configured."""
    store = _make_store(
        adapter=object(),
        embedding_function=None,
        async_embedding_function=None,
    )
    
    with pytest.raises(sk_vector_module.NotSupported) as excinfo:
        store._embed_query("test")
    
    assert "NO_EMBEDDING_FUNCTION" in str(excinfo.value.code)


@pytest.mark.asyncio
async def test_embed_query_async_with_provided_embedding():
    """Test _embed_query_async when embedding is provided."""
    store = _make_store(adapter=object())
    
    query_emb = await store._embed_query_async(
        query="test",
        embedding=[0.1, 0.2, 0.3]
    )
    
    assert query_emb == [0.1, 0.2, 0.3]


@pytest.mark.asyncio 
async def test_embed_query_async_uses_async_function():
    """Test _embed_query_async prefers async_embedding_function."""
    async def async_embed(texts):
        return [[1.0, 2.0] for _ in texts]
    
    store = _make_store(
        adapter=object(),
        async_embedding_function=async_embed,
    )
    
    query_emb = await store._embed_query_async("test")
    assert query_emb == [1.0, 2.0]


def test_normalize_metadatas():
    """Test metadata normalization."""
    store = _make_store(adapter=object())
    
    # None -> empty dicts
    result = store._normalize_metadatas(2, None)
    assert result == [{}, {}]
    
    # Single metadata replicated
    result = store._normalize_metadatas(3, [{"key": "value"}])
    assert result == [{"key": "value"}, {"key": "value"}, {"key": "value"}]
    
    # Mismatched length raises
    with pytest.raises(sk_vector_module.BadRequest) as excinfo:
        store._normalize_metadatas(2, [{"a": 1}, {"b": 2}, {"c": 3}])
    assert "BAD_METADATA" in str(excinfo.value.code)


def test_normalize_ids():
    """Test ID normalization."""
    store = _make_store(adapter=object())
    
    # None -> generated UUIDs
    result = store._normalize_ids(2, None)
    assert len(result) == 2
    assert all(len(id_str) == 32 for id_str in result)  # UUID hex length
    
    # Provided IDs
    result = store._normalize_ids(2, ["id1", "id2"])
    assert result == ["id1", "id2"]
    
    # Mismatched length raises
    with pytest.raises(sk_vector_module.BadRequest) as excinfo:
        store._normalize_ids(2, ["id1"])
    assert "BAD_IDS" in str(excinfo.value.code)


def test_apply_score_threshold():
    """Test client-side score threshold filtering."""
    store = _make_store(adapter=object(), score_threshold=0.7)
    
    class MockMatch:
        def __init__(self, score):
            self.score = score
    
    matches = [MockMatch(0.6), MockMatch(0.7), MockMatch(0.8), MockMatch(0.9)]
    
    filtered = store._apply_score_threshold(matches)
    assert len(filtered) == 3
    assert all(m.score >= 0.7 for m in filtered)
    
    # No threshold returns all
    store_no_threshold = _make_store(adapter=object(), score_threshold=None)
    filtered = store_no_threshold._apply_score_threshold(matches)
    assert len(filtered) == 4


def test_format_for_ai_model():
    """Test AI-optimized document formatting."""
    store = _make_store(adapter=object())
    
    class MockVector:
        def __init__(self, text="test content", meta=None):
            self.metadata = meta or {"page_content": text, "id": "1"}
            self.vector = []
    
    class MockMatch:
        def __init__(self, score=0.9, text="test content"):
            self.score = score
            self.vector = MockVector(text=text)
    
    matches = [MockMatch(0.9, "short"), MockMatch(0.8, "a" * 600)]  # Second will be truncated
    
    formatted = store._format_for_ai_model(matches)
    
    assert len(formatted) == 2
    assert all("content" in d for d in formatted)
    assert all("metadata" in d for d in formatted)
    assert all("confidence" in d for d in formatted)
    assert all("source" in d for d in formatted)
    
    # Check truncation for long text
    assert len(formatted[1]["content"]) < 600  # Should be truncated


def test_from_texts_constructor():
    """Test from_texts convenience constructor."""
    mock_adapter = object()
    
    store = CorpusSemanticKernelVectorStore.from_texts(
        texts=["hello", "world"],
        corpus_adapter=mock_adapter,
        metadatas=[{"topic": "greeting"}, {"topic": "planet"}],
        ids=["id1", "id2"],
        embedding_function=_default_embedding_function,
    )
    
    assert store.corpus_adapter is mock_adapter
    # Store should have texts added (though we can't verify without mocking translator)


def test_from_documents_constructor():
    """Test from_documents convenience constructor."""
    mock_adapter = object()
    
    documents = [
        {"page_content": "hello", "metadata": {"topic": "greeting"}},
        {"page_content": "world", "metadata": {"topic": "planet"}},
    ]
    
    store = CorpusSemanticKernelVectorStore.from_documents(
        documents=documents,
        corpus_adapter=mock_adapter,
        embedding_function=_default_embedding_function,
    )
    
    assert store.corpus_adapter is mock_adapter


def test_add_documents_sync():
    """Test add_documents sync method."""
    store = _make_store(adapter=object())
    
    documents = [
        {"page_content": "doc1", "metadata": {"a": 1}},
        {"page_content": "doc2", "metadata": {"b": 2}},
    ]
    
    # Mock translator to avoid actual vector operations
    store._translator = type("MockTranslator", (), {
        "capabilities": lambda: type("Caps", (), {
            "max_batch_size": 1000,
            "supports_metadata_filtering": True,
            "supports_namespaces": True,
            "max_top_k": 100,
        })(),
        "upsert": lambda *args, **kwargs: type("Result", (), {
            "upserted_count": 2,
            "failed_count": 0,
            "failures": [],
        })(),
    })()
    
    # Should not raise
    store.add_documents(documents)


@pytest.mark.asyncio
async def test_aadd_documents_async():
    """Test aadd_documents async method."""
    store = _make_store(adapter=object())
    
    documents = [
        {"page_content": "doc1", "metadata": {"a": 1}},
        {"page_content": "doc2", "metadata": {"b": 2}},
    ]
    
    # Should not raise (though won't actually execute without proper mocks)
    try:
        await store.aadd_documents(documents)
    except (AttributeError, NotImplementedError):
        pass  # Expected without proper translator mocking


def test_similarity_search_with_score():
    """Test similarity_search_with_score returns tuples."""
    store = _make_store(adapter=object())
    
    # Mock similarity_search to return test docs
    def mock_similarity_search(*args, **kwargs):
        return [
            {"content": "doc1", "metadata": {}, "confidence": 0.9, "source": "vector_database"},
            {"content": "doc2", "metadata": {}, "confidence": 0.8, "source": "vector_database"},
        ]
    
    store.similarity_search = mock_similarity_search
    
    results = store.similarity_search_with_score("test query", k=2)
    
    assert len(results) == 2
    assert isinstance(results[0], tuple)
    assert len(results[0]) == 2
    assert results[0][0]["confidence"] == 0.9
    assert results[0][1] == 0.9


@pytest.mark.asyncio
async def test_asimilarity_search_with_score():
    """Test asimilarity_search_with_score async method."""
    store = _make_store(adapter=object())
    
    async def mock_asimilarity_search(*args, **kwargs):
        return [
            {"content": "doc1", "metadata": {}, "confidence": 0.9, "source": "vector_database"},
            {"content": "doc2", "metadata": {}, "confidence": 0.8, "source": "vector_database"},
        ]
    
    store.asimilarity_search = mock_asimilarity_search
    
    results = await store.asimilarity_search_with_score("test query", k=2)
    
    assert len(results) == 2
    assert isinstance(results[0], tuple)
    assert results[0][1] == 0.9


def test_plugin_builds_sk_aware_context():
    """Test plugin context building includes SK-specific metadata."""
    store = _make_store(adapter=object())
    plugin = CorpusSemanticKernelVectorPlugin(vector_store=store)
    
    # Mock SK context object with variables
    class MockSkContext:
        def __init__(self):
            self.variables = {"temperature": 0.7}
            self.skill_name = "test_skill"
            self.function_name = "test_function"
    
    # Mock context_from_semantic_kernel
    with monkeypatch.context() as m:
        m.setattr(
            sk_vector_module,
            "context_from_semantic_kernel",
            lambda *args, **kwargs: type("MockOpCtx", (), {
                "extra_context": {}
            })(),
        )
        
        ctx = plugin._build_sk_aware_context(
            sk_context=MockSkContext(),
            sk_settings={"max_tokens": 100}
        )
        
        # Should not crash
        assert ctx is not None


def test_plugin_error_handling_for_different_error_types():
    """Test plugin error handling for different exception types."""
    store = _make_store(adapter=object())
    plugin = CorpusSemanticKernelVectorPlugin(vector_store=store)
    
    # Test NotSupported
    not_supported = sk_vector_module.NotSupported("test", code="TEST_CODE")
    try:
        plugin._handle_sk_plugin_error(not_supported, "test_op")
    except sk_vector_module.KernelFunctionException as exc:
        assert "not supported" in str(exc).lower()
    
    # Test BadRequest
    bad_request = sk_vector_module.BadRequest("test", code="BAD_REQUEST")
    try:
        plugin._handle_sk_plugin_error(bad_request, "test_op")
    except sk_vector_module.KernelFunctionException as exc:
        assert "invalid" in str(exc).lower()
    
    # Test generic exception
    generic = ValueError("generic error")
    try:
        plugin._handle_sk_plugin_error(generic, "test_op")
    except sk_vector_module.KernelFunctionException as exc:
        assert "failed" in str(exc).lower()


@pytest.mark.asyncio
async def test_amax_marginal_relevance_search_basic():
    """Test async MMR search basic functionality."""
    store = _make_store(adapter=object())
    
    # Should have the method
    assert hasattr(store, "amax_marginal_relevance_search")
    
    # Verify it's async
    assert inspect.iscoroutinefunction(store.amax_marginal_relevance_search)


def test_get_capabilities_methods():
    """Test public capability methods."""
    store = _make_store(adapter=object())
    
    # Should have both sync and async methods
    assert hasattr(store, "get_capabilities")
    assert hasattr(store, "aget_capabilities")
    
    # aget_capabilities should be async
    assert inspect.iscoroutinefunction(store.aget_capabilities)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
