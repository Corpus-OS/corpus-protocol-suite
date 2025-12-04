# tests/vector/frameworks/test_crewai_vector_adapter.py

from __future__ import annotations

import inspect
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import pytest

import corpus_sdk.vector.framework_adapters.crewai as crewai_module
from corpus_sdk.vector.framework_adapters.crewai import (
    CorpusCrewAIVectorSearchTool,
    CorpusVectorSearchInput,
    ErrorCodes,
    with_error_context,
    with_async_error_context,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_embedding_fn(texts: List[str]) -> List[List[float]]:
    """Deterministic toy embedding function: 2-dim vector based on index."""
    return [[float(i), 1.0] for i, _ in enumerate(texts)]


def _make_tool(adapter: Any, **kwargs: Any) -> CorpusCrewAIVectorSearchTool:
    """Construct a CorpusCrewAIVectorSearchTool instance."""
    return CorpusCrewAIVectorSearchTool(corpus_adapter=adapter, **kwargs)


class _DummyCaps:
    """Simple stand-in for VectorCapabilities."""

    def __init__(
        self,
        *,
        supports_metadata_filtering: bool = True,
        max_top_k: Optional[int] = None,
    ) -> None:
        self.supports_metadata_filtering = supports_metadata_filtering
        self.max_top_k = max_top_k


# ---------------------------------------------------------------------------
# Error-context decorators
# ---------------------------------------------------------------------------


def test_with_error_context_attaches_framework_and_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured.update(ctx)

    monkeypatch.setattr(crewai_module, "attach_context", fake_attach_context)

    @with_error_context("test_sync_op", foo="bar")
    def failing() -> None:
        raise RuntimeError("sync boom")

    with pytest.raises(RuntimeError, match="sync boom"):
        failing()

    assert captured.get("framework") == "crewai"
    assert captured.get("operation") == "test_sync_op"
    assert captured.get("foo") == "bar"


@pytest.mark.asyncio
async def test_with_async_error_context_attaches_framework_and_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured.update(ctx)

    monkeypatch.setattr(crewai_module, "attach_context", fake_attach_context)

    @with_async_error_context("test_async_op", extra="val")
    async def failing() -> None:
        raise RuntimeError("async boom")

    with pytest.raises(RuntimeError, match="async boom"):
        await failing()

    assert captured.get("framework") == "crewai"
    assert captured.get("operation") == "test_async_op"
    assert captured.get("extra") == "val"


# ---------------------------------------------------------------------------
# OperationContext resolution
# ---------------------------------------------------------------------------


def test_resolve_operation_context_prefers_operation_context_instance(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    class DummyOperationContext:
        def __init__(self, **attrs: Any) -> None:
            self.attrs = attrs

    monkeypatch.setattr(crewai_module, "OperationContext", DummyOperationContext)

    tool = _make_tool(adapter)
    ctx = DummyOperationContext(request_id="r1")

    resolved = tool._resolve_operation_context(ctx)
    assert resolved is ctx


def test_resolve_operation_context_uses_ctx_from_dict_for_mapping(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    class DummyOperationContext:
        def __init__(self, **attrs: Any) -> None:
            self.attrs = attrs

    monkeypatch.setattr(crewai_module, "OperationContext", DummyOperationContext)

    captured: Dict[str, Any] = {}

    def fake_from_dict(data: Mapping[str, Any]) -> Any:
        captured["input"] = dict(data)
        return DummyOperationContext(source="dict", **data)

    def fake_from_crewai(_: Any) -> Any:
        raise AssertionError("from_crewai should not be called for mapping input")

    monkeypatch.setattr(crewai_module, "ctx_from_dict", fake_from_dict)
    monkeypatch.setattr(crewai_module, "ctx_from_crewai", fake_from_crewai)

    tool = _make_tool(adapter)
    ctx_mapping = {"run_id": "123", "foo": "bar"}

    resolved = tool._resolve_operation_context(ctx_mapping)
    assert isinstance(resolved, DummyOperationContext)
    assert resolved.attrs["source"] == "dict"
    assert captured["input"] == ctx_mapping


def test_resolve_operation_context_raises_on_bad_mapping_context_type(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    class DummyOperationContext:
        ...

    monkeypatch.setattr(crewai_module, "OperationContext", DummyOperationContext)

    def fake_from_dict(data: Mapping[str, Any]) -> Any:  # noqa: ARG001
        # Return non-OperationContext to trigger BAD_OPERATION_CONTEXT
        return object()

    monkeypatch.setattr(crewai_module, "ctx_from_dict", fake_from_dict)

    tool = _make_tool(adapter)

    with pytest.raises(Exception) as exc_info:
        tool._resolve_operation_context({"x": 1})

    err = exc_info.value
    assert getattr(err, "code", None) == ErrorCodes.BAD_OPERATION_CONTEXT


def test_resolve_operation_context_uses_ctx_from_crewai_for_non_mapping(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    class DummyOperationContext:
        def __init__(self, **attrs: Any) -> None:
            self.attrs = attrs

    monkeypatch.setattr(crewai_module, "OperationContext", DummyOperationContext)

    captured: Dict[str, Any] = {}

    def fake_from_dict(_: Mapping[str, Any]) -> Any:
        raise AssertionError("from_dict should not be called for non-mapping input")

    def fake_from_crewai(task_like: Any) -> Any:
        captured["input"] = task_like
        return DummyOperationContext(source="crewai")

    monkeypatch.setattr(crewai_module, "ctx_from_dict", fake_from_dict)
    monkeypatch.setattr(crewai_module, "ctx_from_crewai", fake_from_crewai)

    tool = _make_tool(adapter)
    class DummyTask:
        ...

    task = DummyTask()
    resolved = tool._resolve_operation_context(task)
    assert isinstance(resolved, DummyOperationContext)
    assert resolved.attrs["source"] == "crewai"
    assert captured["input"] is task


def test_resolve_operation_context_falls_back_to_static_operation_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    class DummyOperationContext:
        def __init__(self, tag: str) -> None:
            self.tag = tag

    monkeypatch.setattr(crewai_module, "OperationContext", DummyOperationContext)

    tool = _make_tool(adapter)
    static_ctx = DummyOperationContext(tag="static")
    tool.static_operation_context = static_ctx

    resolved = tool._resolve_operation_context(None)
    assert resolved is static_ctx


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def test_embed_query_uses_provided_embedding(adapter: Any) -> None:
    tool = _make_tool(adapter)
    tool.embedding_function = None

    emb = tool._embed_query("hello", embedding=[1, 2, 3])
    assert emb == [1.0, 2.0, 3.0]


def test_embed_query_raises_when_no_function_or_embedding(adapter: Any) -> None:
    tool = _make_tool(adapter)
    tool.embedding_function = None

    with pytest.raises(Exception) as exc_info:
        tool._embed_query("hello")

    err = exc_info.value
    assert getattr(err, "code", None) == ErrorCodes.NO_EMBEDDING_FUNCTION


def test_embed_query_calls_embedding_function_and_validates_length(adapter: Any) -> None:
    tool = _make_tool(adapter)

    def good_embed(texts: List[str]) -> List[List[float]]:
        return [[0.5] * 4 for _ in texts]

    tool.embedding_function = good_embed
    emb = tool._embed_query("hello")
    assert emb == [0.5, 0.5, 0.5, 0.5]

    def bad_embed(texts: List[str]) -> List[List[float]]:
        # Return empty list to trigger BAD_EMBEDDINGS
        return []

    tool.embedding_function = bad_embed

    with pytest.raises(Exception) as exc_info:
        tool._embed_query("hello")

    err = exc_info.value
    assert getattr(err, "code", None) == ErrorCodes.BAD_EMBEDDINGS


def test_embed_query_wraps_embedding_errors(adapter: Any) -> None:
    tool = _make_tool(adapter)

    def failing_embed(texts: List[str]) -> List[List[float]]:  # noqa: ARG001
        raise ValueError("embedding failure")

    tool.embedding_function = failing_embed

    with pytest.raises(Exception) as exc_info:
        tool._embed_query("hello")

    err = exc_info.value
    assert getattr(err, "code", None) == ErrorCodes.EMBEDDING_ERROR


@pytest.mark.asyncio
async def test_embed_query_async_uses_provided_embedding(adapter: Any) -> None:
    tool = _make_tool(adapter)
    emb = await tool._embed_query_async("hello", embedding=[0.1, 0.2])
    assert emb == [0.1, 0.2]


@pytest.mark.asyncio
async def test_embed_query_async_uses_async_embedding_function(
    adapter: Any,
) -> None:
    tool = _make_tool(adapter)

    async def async_embed(texts: List[str]) -> List[List[float]]:
        return [[0.9, 0.8] for _ in texts]

    tool.async_embedding_function = async_embed
    tool.embedding_function = None

    emb = await tool._embed_query_async("hello")
    assert emb == [0.9, 0.8]


@pytest.mark.asyncio
async def test_embed_query_async_falls_back_to_sync_embedding_function(
    adapter: Any,
) -> None:
    tool = _make_tool(adapter)
    tool.async_embedding_function = None
    tool.embedding_function = _simple_embedding_fn

    emb = await tool._embed_query_async("hello")
    # index 0 → [0.0, 1.0]
    assert emb == [0.0, 1.0]


@pytest.mark.asyncio
async def test_embed_query_async_raises_when_no_embedding_functions(
    adapter: Any,
) -> None:
    tool = _make_tool(adapter)
    tool.async_embedding_function = None
    tool.embedding_function = None

    with pytest.raises(Exception) as exc_info:
        await tool._embed_query_async("hello")

    err = exc_info.value
    assert getattr(err, "code", None) == ErrorCodes.NO_EMBEDDING_FUNCTION


@pytest.mark.asyncio
async def test_embed_query_async_wraps_async_embed_errors(adapter: Any) -> None:
    tool = _make_tool(adapter)

    async def failing_async_embed(texts: List[str]) -> List[List[float]]:  # noqa: ARG001
        raise RuntimeError("async fail")

    tool.async_embedding_function = failing_async_embed
    tool.embedding_function = None

    with pytest.raises(Exception) as exc_info:
        await tool._embed_query_async("hello")

    err = exc_info.value
    assert getattr(err, "code", None) == ErrorCodes.EMBEDDING_ERROR


@pytest.mark.asyncio
async def test_embed_query_async_validates_length(adapter: Any) -> None:
    tool = _make_tool(adapter)

    async def bad_async_embed(texts: List[str]) -> List[List[float]]:  # noqa: ARG001
        return []

    tool.async_embedding_function = bad_async_embed

    with pytest.raises(Exception) as exc_info:
        await tool._embed_query_async("hello")

    err = exc_info.value
    assert getattr(err, "code", None) == ErrorCodes.BAD_EMBEDDINGS


# ---------------------------------------------------------------------------
# Match helpers: score / vector / payload / thresholding
# ---------------------------------------------------------------------------


def test_get_match_score_handles_mapping_and_object_and_non_numeric() -> None:
    class ObjMatch:
        def __init__(self, score: Any) -> None:
            self.score = score

    m1 = {"score": 0.7}
    m2 = ObjMatch(0.3)
    m3 = {"score": "not-a-number"}
    m4 = {"no_score": True}

    assert crewai_module.CorpusCrewAIVectorSearchTool._get_match_score(m1) == pytest.approx(
        0.7
    )
    assert crewai_module.CorpusCrewAIVectorSearchTool._get_match_score(m2) == pytest.approx(
        0.3
    )
    assert crewai_module.CorpusCrewAIVectorSearchTool._get_match_score(m3) == 0.0
    assert crewai_module.CorpusCrewAIVectorSearchTool._get_match_score(m4) == 0.0


def test_get_match_vector_handles_mapping_and_vector_object() -> None:
    class VecObj:
        def __init__(self, vector: Sequence[float]) -> None:
            self.vector = vector

    class MatchObj:
        def __init__(self, vec: Sequence[float]) -> None:
            self.vector = VecObj(vec)

    m1 = {"embedding": [0.1, 0.2]}
    m2 = MatchObj([0.3, 0.4])
    m3 = {"embedding": "bad"}
    m4 = MatchObj([])

    vec1 = crewai_module.CorpusCrewAIVectorSearchTool._get_match_vector(m1)
    vec2 = crewai_module.CorpusCrewAIVectorSearchTool._get_match_vector(m2)
    vec3 = crewai_module.CorpusCrewAIVectorSearchTool._get_match_vector(m3)
    vec4 = crewai_module.CorpusCrewAIVectorSearchTool._get_match_vector(m4)

    assert vec1 == [0.1, 0.2]
    assert vec2 == [0.3, 0.4]
    assert vec3 == []
    assert vec4 == []


def test_filter_matches_by_score_applies_threshold(adapter: Any) -> None:
    tool = _make_tool(adapter)
    matches = [
        {"score": 0.3, "id": "low"},
        {"score": 0.6, "id": "mid"},
        {"score": 0.9, "id": "high"},
        {"score": "bad", "id": "bad"},
    ]

    # No threshold → all kept
    tool.score_threshold = None
    kept = tool._filter_matches_by_score(matches)
    assert kept == matches

    # Threshold 0.5 → only >= 0.5 and numeric
    tool.score_threshold = 0.5
    kept = tool._filter_matches_by_score(matches)
    ids = {m["id"] for m in kept}
    assert ids == {"mid", "high"}


def test_match_to_payload_for_mapping_and_vector_match(adapter: Any) -> None:
    tool = _make_tool(adapter)

    # Mapping-style match
    match_mapping = {
        "id": "m1",
        "score": 0.9,
        "metadata": {
            tool.text_field: "hello",
            tool.id_field: "m1",
            "topic": "t",
        },
    }

    payload = tool._match_to_payload(match_mapping, return_scores=True)
    assert payload["text"] == "hello"
    assert payload["id"] == "m1"
    assert payload["metadata"] == {"topic": "t"}
    assert payload["score"] == pytest.approx(0.9)

    # VectorMatch-style object with metadata envelope
    class DummyVector:
        def __init__(self) -> None:
            self.id = "v1"
            self.metadata = {
                tool.metadata_field or "meta": {"a": 1},
                tool.text_field: "inner",
                tool.id_field: "v1",
            }

    class DummyMatch:
        def __init__(self) -> None:
            self.vector = DummyVector()
            self.score = 0.42

    tool.metadata_field = "meta"
    match_obj = DummyMatch()

    payload2 = tool._match_to_payload(match_obj, return_scores=True)
    assert payload2["text"] == "inner"
    assert payload2["id"] == "v1"
    assert payload2["metadata"] == {"a": 1}
    assert payload2["score"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Raw query builder
# ---------------------------------------------------------------------------


def test_build_raw_query_shapes_request(adapter: Any) -> None:
    tool = _make_tool(adapter)
    emb = [0.1, 0.2, 0.3]

    raw = tool._build_raw_query(
        embedding=emb,
        k=5,
        namespace="ns-1",
        filter={"tag": "v"},
        include_vectors=True,
    )

    assert raw["vector"] == [0.1, 0.2, 0.3]
    assert raw["top_k"] == 5
    assert raw["namespace"] == "ns-1"
    assert raw["filters"] == {"tag": "v"}
    assert raw["include_metadata"] is True
    assert raw["include_vectors"] is True


# ---------------------------------------------------------------------------
# MMR selector
# ---------------------------------------------------------------------------


def test_mmr_select_indices_respects_scores_when_lambda_one(adapter: Any) -> None:
    tool = _make_tool(adapter)

    matches = [
        {"id": "a", "score": 0.2, "embedding": [0.1, 0.0]},
        {"id": "b", "score": 0.9, "embedding": [0.0, 1.0]},
        {"id": "c", "score": 0.5, "embedding": [1.0, 0.0]},
    ]

    indices = tool._mmr_select_indices(matches, k=2, lambda_mult=1.0)
    assert indices == [1, 2]  # sorted by score desc


def test_mmr_select_indices_returns_at_most_k_and_handles_empty_vectors(
    adapter: Any,
) -> None:
    tool = _make_tool(adapter)

    matches = [
        {"id": "a", "score": 0.4, "embedding": []},
        {"id": "b", "score": 0.6, "embedding": [0.1]},  # inconsistent dim
        {"id": "c", "score": 0.8, "embedding": [0.2]},
    ]

    indices = tool._mmr_select_indices(matches, k=10, lambda_mult=0.5)
    assert len(indices) <= 3
    assert len(set(indices)) == len(indices)


# ---------------------------------------------------------------------------
# Async capabilities helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_caps_async_uses_adapter_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyAdapter:
        def __init__(self) -> None:
            self.calls = 0

        async def capabilities(self) -> Any:
            self.calls += 1
            return _DummyCaps(max_top_k=50)

    adapter = DummyAdapter()
    tool = _make_tool(adapter, with_embeddings=False)  # type: ignore[arg-type]

    caps1 = await tool._get_caps_async()
    caps2 = await tool._get_caps_async()

    assert caps1 is caps2
    assert adapter.calls == 1


@pytest.mark.asyncio
async def test_get_caps_async_attaches_error_context_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    monkeypatch.setattr(crewai_module, "attach_context", fake_attach_context)

    class FailingAdapter:
        async def capabilities(self) -> Any:
            raise RuntimeError("caps fail")

    tool = _make_tool(FailingAdapter())

    with pytest.raises(RuntimeError, match="caps fail"):
        await tool._get_caps_async()

    assert captured.get("framework") == "crewai"
    assert captured.get("operation") == "capabilities"


# ---------------------------------------------------------------------------
# Query result / stream chunk validation
# ---------------------------------------------------------------------------


def test_validate_query_result_accepts_dummy_queryresult(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyQueryResult:
        def __init__(self, matches: Optional[Sequence[Any]] = None) -> None:
            self.matches = matches or []

    monkeypatch.setattr(crewai_module, "QueryResult", DummyQueryResult)

    result = DummyQueryResult()
    validated = crewai_module.CorpusCrewAIVectorSearchTool._validate_query_result(
        result,
        operation="op",
    )
    assert validated is result


def test_validate_query_result_rejects_wrong_type(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyQueryResult:
        ...

    monkeypatch.setattr(crewai_module, "QueryResult", DummyQueryResult)

    with pytest.raises(Exception) as exc_info:
        crewai_module.CorpusCrewAIVectorSearchTool._validate_query_result(
            object(),
            operation="op",
        )

    err = exc_info.value
    assert getattr(err, "code", None) == ErrorCodes.BAD_QUERY_RESULT


def test_validate_stream_chunk_accepts_dummy_querychunk(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyQueryChunk:
        def __init__(self, matches: Optional[Sequence[Any]] = None) -> None:
            self.matches = matches or []

    monkeypatch.setattr(crewai_module, "QueryChunk", DummyQueryChunk)

    chunk = DummyQueryChunk()
    validated = crewai_module.CorpusCrewAIVectorSearchTool._validate_stream_chunk(
        chunk,
        operation="stream_op",
    )
    assert validated is chunk


def test_validate_stream_chunk_rejects_wrong_type(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyQueryChunk:
        ...

    monkeypatch.setattr(crewai_module, "QueryChunk", DummyQueryChunk)

    with pytest.raises(Exception) as exc_info:
        crewai_module.CorpusCrewAIVectorSearchTool._validate_stream_chunk(
            object(),
            operation="stream_op",
        )

    err = exc_info.value
    assert getattr(err, "code", None) == ErrorCodes.BAD_STREAM_CHUNK


# ---------------------------------------------------------------------------
# Async search: _asearch_simple
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_asearch_simple_builds_raw_query_and_shapes_results(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    # Patch QueryResult type in module
    class DummyQueryResult:
        def __init__(self, matches: Sequence[Any]) -> None:
            self.matches = list(matches)

    monkeypatch.setattr(crewai_module, "QueryResult", DummyQueryResult)

    captured: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_query(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
            mmr_config: Any = None,  # noqa: ARG002
        ) -> Any:
            captured["raw_query"] = dict(raw_query)
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx)
            return DummyQueryResult(
                matches=[
                    {
                        "id": "d1",
                        "score": 0.9,
                        "metadata": {
                            "page_content": "hello",
                            "id": "d1",
                            "tag": "t1",
                        },
                    },
                    {
                        "id": "d2",
                        "score": 0.4,
                        "metadata": {
                            "page_content": "bye",
                            "id": "d2",
                            "tag": "t2",
                        },
                    },
                ]
            )

    tool = _make_tool(adapter)
    tool.async_embedding_function = lambda texts: [  # type: ignore[assignment]
        [0.1, 0.2] for _ in texts
    ]
    tool.embedding_function = None
    tool.score_threshold = 0.5  # filter out low-score match

    # Override the cached_property for this instance
    tool._translator = DummyTranslator()  # type: ignore[attr-defined]

    args = CorpusVectorSearchInput(
        query="q-text",
        k=3,
        namespace="ns-search",
        filter={"tag": "t1"},
        return_scores=True,
    )

    caps = _DummyCaps(supports_metadata_filtering=True, max_top_k=10)

    results = await tool._asearch_simple(args, caps=caps)
    assert isinstance(results, list)
    assert len(results) == 1
    payload = results[0]
    assert payload["text"] == "hello"
    assert payload["metadata"] == {"tag": "t1"}
    assert payload["id"] == "d1"
    assert isinstance(payload["score"], float)

    raw_query = captured["raw_query"]
    assert raw_query["top_k"] == 3
    assert raw_query["filters"] == {"tag": "t1"}
    assert raw_query["include_vectors"] is False

    fw_ctx = captured["framework_ctx"]
    assert fw_ctx.get("namespace") == "ns-search"


@pytest.mark.asyncio
async def test_asearch_simple_rejects_top_k_over_cap(
    adapter: Any,
) -> None:
    tool = _make_tool(adapter)
    tool.async_embedding_function = lambda texts: [  # type: ignore[assignment]
        [0.0, 1.0] for _ in texts
    ]

    args = CorpusVectorSearchInput(
        query="q",
        k=100,
    )
    caps = _DummyCaps(supports_metadata_filtering=True, max_top_k=10)

    with pytest.raises(Exception) as exc_info:
        await tool._asearch_simple(args, caps=caps)

    err = exc_info.value
    assert getattr(err, "code", None) == ErrorCodes.BAD_TOP_K


@pytest.mark.asyncio
async def test_asearch_simple_rejects_filter_when_capability_missing(
    adapter: Any,
) -> None:
    tool = _make_tool(adapter)
    tool.async_embedding_function = lambda texts: [  # type: ignore[assignment]
        [0.0, 1.0] for _ in texts
    ]

    args = CorpusVectorSearchInput(
        query="q",
        k=2,
        filter={"tag": "v"},
    )
    caps = _DummyCaps(supports_metadata_filtering=False, max_top_k=None)

    with pytest.raises(Exception) as exc_info:
        await tool._asearch_simple(args, caps=caps)

    err = exc_info.value
    assert getattr(err, "code", None) == ErrorCodes.FILTER_NOT_SUPPORTED


# ---------------------------------------------------------------------------
# Async search: _asearch_with_mmr
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_asearch_with_mmr_builds_fetch_k_and_uses_mmr(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    class DummyQueryResult:
        def __init__(self, matches: Sequence[Any]) -> None:
            self.matches = list(matches)

    monkeypatch.setattr(crewai_module, "QueryResult", DummyQueryResult)

    captured: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_query(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any,  # noqa: ARG002
            framework_ctx: Mapping[str, Any],
            mmr_config: Any = None,  # noqa: ARG002
        ) -> Any:
            captured["raw_query"] = dict(raw_query)
            captured["framework_ctx"] = dict(framework_ctx)
            return DummyQueryResult(
                matches=[
                    {"id": "a", "score": 0.9, "embedding": [0.1, 0.2], "metadata": {}},
                    {"id": "b", "score": 0.8, "embedding": [0.2, 0.3], "metadata": {}},
                    {"id": "c", "score": 0.7, "embedding": [0.3, 0.4], "metadata": {}},
                ]
            )

    tool = _make_tool(adapter)
    tool.async_embedding_function = lambda texts: [  # type: ignore[assignment]
        [0.0, 1.0] for _ in texts
    ]
    tool.embedding_function = None
    tool._translator = DummyTranslator()  # type: ignore[attr-defined]
    tool.score_threshold = None

    called: Dict[str, Any] = {}

    def fake_mmr_select_indices(
        self: CorpusCrewAIVectorSearchTool,
        candidate_matches: List[Any],
        k: int,
        lambda_mult: float,
    ) -> List[int]:
        called["k"] = k
        called["lambda_mult"] = lambda_mult
        called["count"] = len(candidate_matches)
        # Just pick first k by construction
        return list(range(min(k, len(candidate_matches))))

    monkeypatch.setattr(
        CorpusCrewAIVectorSearchTool,
        "_mmr_select_indices",
        fake_mmr_select_indices,
    )

    args = CorpusVectorSearchInput(
        query="mmr-q",
        k=2,
        mmr_lambda=0.7,
        fetch_k=5,
        return_scores=False,
    )
    caps = _DummyCaps(supports_metadata_filtering=True, max_top_k=10)

    results = await tool._asearch_with_mmr(args, caps=caps)
    assert len(results) == 2

    raw_query = captured["raw_query"]
    assert raw_query["top_k"] == 5
    assert raw_query["include_vectors"] is True

    assert called["k"] == 2
    assert called["lambda_mult"] == pytest.approx(0.7)
    assert called["count"] == 3


@pytest.mark.asyncio
async def test_asearch_with_mmr_rejects_fetch_k_over_cap(
    adapter: Any,
) -> None:
    tool = _make_tool(adapter)
    tool.async_embedding_function = lambda texts: [  # type: ignore[assignment]
        [0.0, 1.0] for _ in texts
    ]

    args = CorpusVectorSearchInput(
        query="q",
        k=5,
        fetch_k=100,
    )
    caps = _DummyCaps(supports_metadata_filtering=True, max_top_k=10)

    with pytest.raises(Exception) as exc_info:
        await tool._asearch_with_mmr(args, caps=caps)

    err = exc_info.value
    assert getattr(err, "code", None) == ErrorCodes.BAD_TOP_K


# ---------------------------------------------------------------------------
# Async dispatch: _asearch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_asearch_dispatches_between_simple_and_mmr(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    tool = _make_tool(adapter)

    async def fake_caps() -> Any:
        return _DummyCaps()

    monkeypatch.setattr(tool, "_get_caps_async", fake_caps)  # type: ignore[arg-type]

    calls: List[str] = []

    async def fake_simple(args: CorpusVectorSearchInput, *, caps: Any) -> List[Dict[str, Any]]:  # noqa: ARG001
        calls.append("simple")
        return [{"mode": "simple"}]

    async def fake_mmr(args: CorpusVectorSearchInput, *, caps: Any) -> List[Dict[str, Any]]:  # noqa: ARG001
        calls.append("mmr")
        return [{"mode": "mmr"}]

    # Bind overrides on the instance
    tool._asearch_simple = fake_simple  # type: ignore[assignment]
    tool._asearch_with_mmr = fake_mmr  # type: ignore[assignment]

    # 1) use_mmr=True → mmr path
    args = CorpusVectorSearchInput(query="q1", use_mmr=True)
    res1 = await tool._asearch(args)
    assert calls[-1] == "mmr"
    assert res1[0]["mode"] == "mmr"

    # 2) use_mmr=False → simple path
    args2 = CorpusVectorSearchInput(query="q2", use_mmr=False)
    res2 = await tool._asearch(args2)
    assert calls[-1] == "simple"
    assert res2[0]["mode"] == "simple"

    # 3) use_mmr is None and tool.use_mmr_by_default=True → mmr
    tool.use_mmr_by_default = True
    args3 = CorpusVectorSearchInput(query="q3", use_mmr=None)
    res3 = await tool._asearch(args3)
    assert calls[-1] == "mmr"
    assert res3[0]["mode"] == "mmr"


# ---------------------------------------------------------------------------
# Sync search: _search_simple_sync and _search_with_mmr_sync
# ---------------------------------------------------------------------------


def test_search_simple_sync_builds_query_and_returns_payloads(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    class DummyQueryResult:
        def __init__(self, matches: Sequence[Any]) -> None:
            self.matches = list(matches)

    monkeypatch.setattr(crewai_module, "QueryResult", DummyQueryResult)

    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def query(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
            mmr_config: Any = None,  # noqa: ARG002
        ) -> Any:
            captured["raw_query"] = dict(raw_query)
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx)
            return DummyQueryResult(
                matches=[
                    {
                        "id": "m1",
                        "score": 0.9,
                        "metadata": {
                            "page_content": "text",
                            "id": "m1",
                            "tag": "x",
                        },
                    }
                ]
            )

    tool = _make_tool(adapter)
    tool.embedding_function = _simple_embedding_fn
    tool._translator = DummyTranslator()  # type: ignore[attr-defined]

    args = CorpusVectorSearchInput(
        query="hello",
        k=4,
        namespace="sync-ns",
        filter={"tag": "x"},
        return_scores=False,
    )

    results = tool._search_simple_sync(args)
    assert len(results) == 1
    payload = results[0]
    assert payload["text"] == "text"
    assert payload["metadata"] == {"tag": "x"}
    assert "score" not in payload

    raw_query = captured["raw_query"]
    assert raw_query["top_k"] == 4
    assert raw_query["filters"] == {"tag": "x"}
    assert raw_query["include_vectors"] is False


def test_search_with_mmr_sync_uses_mmr_and_fetch_k(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    class DummyQueryResult:
        def __init__(self, matches: Sequence[Any]) -> None:
            self.matches = list(matches)

    monkeypatch.setattr(crewai_module, "QueryResult", DummyQueryResult)

    class DummyTranslator:
        def query(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any,  # noqa: ARG002
            framework_ctx: Mapping[str, Any],
            mmr_config: Any = None,  # noqa: ARG002
        ) -> Any:
            return DummyQueryResult(
                matches=[
                    {"id": "a", "score": 0.9, "embedding": [0.1, 0.2], "metadata": {}},
                    {"id": "b", "score": 0.8, "embedding": [0.2, 0.3], "metadata": {}},
                ]
            )

    tool = _make_tool(adapter)
    tool.embedding_function = _simple_embedding_fn
    tool._translator = DummyTranslator()  # type: ignore[attr-defined]
    tool.score_threshold = None

    called: Dict[str, Any] = {}

    def fake_mmr_select_indices(
        self: CorpusCrewAIVectorSearchTool,
        candidate_matches: List[Any],
        k: int,
        lambda_mult: float,
    ) -> List[int]:
        called["k"] = k
        called["lambda_mult"] = lambda_mult
        called["n_matches"] = len(candidate_matches)
        return [0]

    monkeypatch.setattr(
        CorpusCrewAIVectorSearchTool,
        "_mmr_select_indices",
        fake_mmr_select_indices,
    )

    args = CorpusVectorSearchInput(
        query="q",
        k=1,
        mmr_lambda=0.4,
        fetch_k=3,
        return_scores=True,
    )

    results = tool._search_with_mmr_sync(args)
    assert len(results) == 1
    assert "score" in results[0]
    assert called["k"] == 1
    assert called["lambda_mult"] == pytest.approx(0.4)
    assert called["n_matches"] == 2


# ---------------------------------------------------------------------------
# Tool entrypoints: _run and _arun
# ---------------------------------------------------------------------------


def test_run_dispatches_to_sync_search_variants(
    adapter: Any,
) -> None:
    tool = _make_tool(adapter)

    called: List[str] = []

    def fake_simple(self: CorpusCrewAIVectorSearchTool, args: CorpusVectorSearchInput) -> List[Dict[str, Any]]:  # noqa: ARG001
        called.append("simple")
        return [{"mode": "simple"}]

    def fake_mmr(self: CorpusCrewAIVectorSearchTool, args: CorpusVectorSearchInput) -> List[Dict[str, Any]]:  # noqa: ARG001
        called.append("mmr")
        return [{"mode": "mmr"}]

    # Bind methods on class so decorators remain
    setattr(CorpusCrewAIVectorSearchTool, "_search_simple_sync", fake_simple)
    setattr(CorpusCrewAIVectorSearchTool, "_search_with_mmr_sync", fake_mmr)

    # use_mmr=True → mmr
    res1 = tool._run(query="q1", use_mmr=True)
    assert called[-1] == "mmr"
    assert res1[0]["mode"] == "mmr"

    # use_mmr=False → simple
    res2 = tool._run(query="q2", use_mmr=False)
    assert called[-1] == "simple"
    assert res2[0]["mode"] == "simple"

    # use_mmr=None + use_mmr_by_default=True → mmr
    tool.use_mmr_by_default = True
    res3 = tool._run(query="q3")
    assert called[-1] == "mmr"
    assert res3[0]["mode"] == "mmr"


@pytest.mark.asyncio
async def test_arun_delegates_to_asearch(adapter: Any) -> None:
    tool = _make_tool(adapter)

    async def fake_asearch(args: CorpusVectorSearchInput) -> List[Dict[str, Any]]:  # noqa: ARG001
        return [{"ok": True}]

    tool._asearch = fake_asearch  # type: ignore[assignment]

    res = await tool._arun(query="q")
    assert res == [{"ok": True}]


# ---------------------------------------------------------------------------
# Streaming search
# ---------------------------------------------------------------------------


def test_stream_search_yields_payloads_and_respects_top_k(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    class DummyQueryChunk:
        def __init__(self, matches: Sequence[Any]) -> None:
            self.matches = list(matches)

    monkeypatch.setattr(crewai_module, "QueryChunk", DummyQueryChunk)

    class DummyTranslator:
        def query_stream(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any,
            framework_ctx: Mapping[str, Any],
        ):
            # Single chunk with three matches
            yield DummyQueryChunk(
                matches=[
                    {
                        "id": "m1",
                        "score": 0.9,
                        "metadata": {
                            "page_content": "t1",
                            "id": "m1",
                            "topic": "a",
                        },
                    },
                    {
                        "id": "m2",
                        "score": 0.8,
                        "metadata": {
                            "page_content": "t2",
                            "id": "m2",
                            "topic": "b",
                        },
                    },
                    {
                        "id": "m3",
                        "score": 0.7,
                        "metadata": {
                            "page_content": "t3",
                            "id": "m3",
                            "topic": "c",
                        },
                    },
                ]
            )

    tool = _make_tool(adapter)
    tool.embedding_function = _simple_embedding_fn
    tool.score_threshold = 0.0
    tool._translator = DummyTranslator()  # type: ignore[attr-defined]

    iterator = tool.stream_search(query="q", k=2, return_scores=True)
    assert hasattr(iterator, "__iter__")

    docs = list(iterator)
    # top_k=2 → at most 2 yielded
    assert len(docs) == 2
    assert docs[0]["text"] == "t1"
    assert "score" in docs[0]


def test_stream_search_validates_chunk_type(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    # Patch QueryChunk to a dummy type so the wrong type triggers validation
    class DummyQueryChunk:
        ...

    monkeypatch.setattr(crewai_module, "QueryChunk", DummyQueryChunk)

    class BadTranslator:
        def query_stream(self, *args: Any, **kwargs: Any):  # noqa: ARG002
            # Yield wrong chunk type
            yield object()

    tool = _make_tool(adapter)
    tool.embedding_function = _simple_embedding_fn
    tool._translator = BadTranslator()  # type: ignore[attr-defined]

    with pytest.raises(Exception) as exc_info:
        list(tool.stream_search(query="q", k=1))

    err = exc_info.value
    assert getattr(err, "code", None) == ErrorCodes.BAD_STREAM_CHUNK


# ---------------------------------------------------------------------------
# Error-context behavior for vector operations
# ---------------------------------------------------------------------------


def test_sync_vector_search_errors_include_crewai_metadata(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    monkeypatch.setattr(crewai_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("sync vector failure")

    tool = _make_tool(adapter)
    tool.embedding_function = _simple_embedding_fn
    tool._translator = FailingTranslator()  # type: ignore[attr-defined]

    args = CorpusVectorSearchInput(query="q")

    with pytest.raises(RuntimeError, match="sync vector failure"):
        tool._search_simple_sync(args)

    assert captured.get("framework") == "crewai"
    assert captured.get("operation") == "vector_search_sync"


@pytest.mark.asyncio
async def test_async_vector_search_errors_include_crewai_metadata(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    monkeypatch.setattr(crewai_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        async def arun_query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("async vector failure")

    class DummyQueryResult:
        def __init__(self, matches: Sequence[Any]) -> None:
            self.matches = list(matches)

    monkeypatch.setattr(crewai_module, "QueryResult", DummyQueryResult)

    tool = _make_tool(adapter)
    tool.async_embedding_function = lambda texts: [  # type: ignore[assignment]
        [0.0, 1.0] for _ in texts
    ]
    tool._translator = FailingTranslator()  # type: ignore[attr-defined]

    args = CorpusVectorSearchInput(query="q")

    caps = _DummyCaps(supports_metadata_filtering=True, max_top_k=5)

    with pytest.raises(RuntimeError, match="async vector failure"):
        await tool._asearch_simple(args, caps=caps)

    assert captured.get("framework") == "crewai"
    assert captured.get("operation") == "vector_search_async"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
