# tests/frameworks/graph/test_langchain_graph_adapter.py

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Mapping
from typing import Any, Dict, List, Sequence, Type

import pytest

import corpus_sdk.graph.framework_adapters.langchain as langchain_adapter_module
from corpus_sdk.graph.framework_adapters.langchain import (
    CorpusLangChainGraphClient,
    LangChainGraphFrameworkTranslator,
    create_corpus_graph_tool,
    create_langchain_graph_tools,
)
from corpus_sdk.graph.graph_base import BadRequest, NotSupported


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(adapter: Any, **kwargs: Any) -> CorpusLangChainGraphClient:
    """Construct a CorpusLangChainGraphClient instance from the generic adapter."""
    return CorpusLangChainGraphClient(adapter=adapter, **kwargs)


def _patch_create_graph_translator(
    monkeypatch: pytest.MonkeyPatch,
    translator_cls: Type[Any],
) -> None:
    """
    Patch create_graph_translator to always return an instance of translator_cls.

    translator_cls is expected to be a class; instances are created with no args.
    """

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return translator_cls()

    monkeypatch.setattr(
        langchain_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )


def _patch_validate_graph_result_type_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch validate_graph_result_type to simply return the result unchanged."""

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    monkeypatch.setattr(
        langchain_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )


def _tool_sync_call(tool: Any, tool_input: Any) -> Any:
    """
    Invoke a LangChain tool using the highest-level public API available.

    We intentionally do NOT import LangChain in this test suite. Instead, we call
    whichever standard entrypoint exists on the BaseTool instance:

    - invoke(...) (preferred in newer LangChain)
    - run(...)    (common across many versions)
    - _run(...)   (last-resort fallback for compatibility)

    This keeps tests "real integration" while remaining resilient to minor
    LangChain API surface differences.
    """
    if hasattr(tool, "invoke") and callable(getattr(tool, "invoke")):
        return tool.invoke(tool_input)
    if hasattr(tool, "run") and callable(getattr(tool, "run")):
        return tool.run(tool_input)
    # Fallback: direct internal call only if no public tool API exists.
    return tool._run(tool_input)  # type: ignore[attr-defined]


async def _tool_async_call(tool: Any, tool_input: Any) -> Any:
    """
    Async invoke a LangChain tool using the highest-level async API available.

    - ainvoke(...) (preferred in newer LangChain)
    - arun(...)    (common across many versions)
    - _arun(...)   (last-resort fallback for compatibility)

    As above, we do not import LangChain here; the adapter owns the soft-import.
    """
    if hasattr(tool, "ainvoke") and callable(getattr(tool, "ainvoke")):
        return await tool.ainvoke(tool_input)
    if hasattr(tool, "arun") and callable(getattr(tool, "arun")):
        return await tool.arun(tool_input)
    # Fallback: direct internal call only if no public async tool API exists.
    return await tool._arun(tool_input)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Constructor / translator behavior
# ---------------------------------------------------------------------------


def test_default_translator_uses_langchain_framework_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    By default, CorpusLangChainGraphClient should:

    - Construct a LangChainGraphFrameworkTranslator instance, and
    - Pass it into create_graph_translator with framework="langchain".
    """
    captured: Dict[str, Any] = {}

    def fake_create_graph_translator(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["args"] = args
        captured["kwargs"] = kwargs

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(
        langchain_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(adapter)

    # Trigger lazy translator construction
    _ = client._translator  # noqa: SLF001

    assert "kwargs" in captured
    kwargs = captured["kwargs"]

    assert kwargs.get("framework") == "langchain"
    translator = kwargs.get("translator")
    assert isinstance(translator, LangChainGraphFrameworkTranslator)


def test_framework_translator_override_is_respected(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If framework_translator is provided, CorpusLangChainGraphClient should pass
    it through to create_graph_translator instead of constructing its own
    LangChainGraphFrameworkTranslator.
    """
    captured: Dict[str, Any] = {}

    class CustomTranslator(LangChainGraphFrameworkTranslator):
        pass

    custom = CustomTranslator()

    def fake_create_graph_translator(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["args"] = args
        captured["kwargs"] = kwargs

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(
        langchain_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(
        adapter,
        framework_translator=custom,
        framework_version="lc-fw-1.2.3",
    )

    _ = client._translator  # noqa: SLF001

    kwargs = captured["kwargs"]
    assert kwargs.get("framework") == "langchain"
    assert kwargs.get("translator") is custom


# ---------------------------------------------------------------------------
# Context translation / core_ctx_from_langchain mapping
# ---------------------------------------------------------------------------


def test_langchain_config_and_extra_context_passed_to_core_ctx(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Verify that config and extra_context are passed through to
    core_ctx_from_langchain with the configured framework_version.
    """
    captured: Dict[str, Any] = {}

    class DummyOperationContext:
        def __init__(self) -> None:
            self.attrs: Dict[str, Any] = {}

    # Patch OperationContext inside the module so our dummy passes isinstance() checks.
    monkeypatch.setattr(langchain_adapter_module, "OperationContext", DummyOperationContext)

    def fake_core_ctx_from_langchain(
        config: Any,
        *,
        framework_version: Any = None,
        **extra: Any,
    ) -> Any:
        captured["config"] = config
        captured["framework_version"] = framework_version
        captured["extra"] = extra
        return DummyOperationContext()

    monkeypatch.setattr(
        langchain_adapter_module,
        "core_ctx_from_langchain",
        fake_core_ctx_from_langchain,
    )

    client = _make_client(adapter, framework_version="langchain-test-version")

    lc_config = {
        "configurable": {"user_id": "user-123", "run_id": "run-xyz"},
        "tags": ["foo", "bar"],
    }
    extra_ctx = {"request_id": "req-xyz", "tenant": "tenant-1"}

    result = client.query("MATCH (n) RETURN n LIMIT 1", config=lc_config, extra_context=extra_ctx)
    assert result is not None

    assert captured.get("config") == lc_config
    assert captured.get("framework_version") == "langchain-test-version"
    assert captured.get("extra") == extra_ctx


def test_build_ctx_translation_failure_returns_none_and_attaches_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    On core_ctx_from_langchain failure, _build_ctx should:

    - attach error context (framework='langchain', operation='context_translation')
    - return None (best-effort fallback, does not block graph calls)
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured_ctx.update(ctx)

    def fake_core_ctx_from_langchain(*_: Any, **__: Any) -> Any:
        raise RuntimeError("boom from ctx builder")

    monkeypatch.setattr(langchain_adapter_module, "attach_context", fake_attach_context)
    monkeypatch.setattr(langchain_adapter_module, "core_ctx_from_langchain", fake_core_ctx_from_langchain)

    client = _make_client(adapter, framework_version="ctx-fw")

    ctx = client._build_ctx(config={"x": 1}, extra_context={"foo": "bar"})  # noqa: SLF001
    assert ctx is None

    assert captured_ctx.get("framework") == "langchain"
    assert captured_ctx.get("operation") == "context_translation"
    assert captured_ctx.get("error_code") == langchain_adapter_module.ErrorCodes.BAD_OPERATION_CONTEXT


def test_build_ctx_non_operation_context_returns_none(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If from_langchain returns an unsupported type, _build_ctx should return None.
    """

    def fake_core_ctx_from_langchain(*_: Any, **__: Any) -> Any:
        return object()

    monkeypatch.setattr(langchain_adapter_module, "core_ctx_from_langchain", fake_core_ctx_from_langchain)

    client = _make_client(adapter)
    ctx = client._build_ctx(config={"x": 1}, extra_context={"foo": "bar"})  # noqa: SLF001
    assert ctx is None


# ---------------------------------------------------------------------------
# Error-context decorator behavior
# ---------------------------------------------------------------------------


def test_error_context_includes_langchain_metadata_sync(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    When an error occurs during a sync graph operation, error context should
    include LangChain-specific metadata via attach_context().
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured_context.update(ctx)

    monkeypatch.setattr(langchain_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from langchain graph adapter")

    monkeypatch.setattr(
        langchain_adapter_module,
        "create_graph_translator",
        lambda *_args, **_kwargs: FailingTranslator(),
    )

    client = _make_client(adapter)

    with pytest.raises(RuntimeError, match="test error from langchain graph adapter"):
        client.query("MATCH (n) RETURN n", config={"configurable": {"user_id": "u-sync"}})

    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "langchain"
    assert str(captured_context.get("operation", "")).startswith("graph_")


@pytest.mark.asyncio
async def test_error_context_includes_langchain_metadata_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Same as the sync error-context test but for the async query path.
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured_context.update(ctx)

    monkeypatch.setattr(langchain_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        async def arun_query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from langchain graph adapter")

    monkeypatch.setattr(
        langchain_adapter_module,
        "create_graph_translator",
        lambda *_args, **_kwargs: FailingTranslator(),
    )

    client = _make_client(adapter)

    with pytest.raises(RuntimeError, match="test error from langchain graph adapter"):
        await client.aquery("MATCH (n) RETURN n", config={"configurable": {"user_id": "u-async"}})

    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "langchain"
    assert str(captured_context.get("operation", "")).startswith("graph_")


# ---------------------------------------------------------------------------
# Sync semantics (basic smoke tests)
# ---------------------------------------------------------------------------


def test_sync_query_and_stream_basic(adapter: Any) -> None:
    """
    Basic smoke test for sync query / stream_query behavior: methods should
    accept text input and not crash, returning protocol-level shapes.

    Detailed QueryResult / QueryChunk semantics are covered by the generic
    graph contract tests.
    """
    client = _make_client(adapter, default_namespace="lc-ns")

    result = client.query("MATCH (n) RETURN n LIMIT 1")
    assert result is not None

    chunks = list(client.stream_query("MATCH (n) RETURN n LIMIT 2"))
    assert isinstance(chunks, list)


def test_sync_query_accepts_optional_params_and_config(adapter: Any) -> None:
    """
    query() should accept params, dialect, namespace, timeout_ms, and
    config/extra_context kwargs without raising.
    """
    client = _make_client(adapter, default_dialect="cypher")

    result = client.query(
        "MATCH (n) RETURN n LIMIT $limit",
        params={"limit": 5},
        dialect="cypher",
        namespace="ctx-ns",
        timeout_ms=5000,
        config={"configurable": {"user_id": "u-sync"}},
        extra_context={"request_id": "req-sync"},
    )
    assert result is not None


def test_sync_stream_query_accepts_optional_params_and_config(adapter: Any) -> None:
    """
    stream_query() should accept the same optional parameters as query().
    """
    client = _make_client(adapter, default_dialect="cypher")
    chunks = list(
        client.stream_query(
            "MATCH (n) RETURN n LIMIT $limit",
            params={"limit": 2},
            dialect="cypher",
            namespace="ctx-ns",
            timeout_ms=2500,
            config={"configurable": {"user_id": "u-sync"}},
            extra_context={"request_id": "req-sync"},
        )
    )
    assert isinstance(chunks, list)


# ---------------------------------------------------------------------------
# Async semantics (basic smoke tests + sync/async parity)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_query_and_stream_basic(adapter: Any) -> None:
    """
    Async aquery / astream_query should exist and produce results compatible
    with the sync API (non-None result / async-iterable of chunks).

    Also asserts:
    - aquery() returns the same *type* as query()
    - astream_query() returns an async-iterable object.
    """
    client = _make_client(adapter)

    sync_result = client.query("MATCH (n) RETURN n LIMIT 1")
    assert sync_result is not None

    async_result = await client.aquery("MATCH (n) RETURN n LIMIT 1")
    assert async_result is not None
    assert type(sync_result) is type(async_result)  # noqa: E721

    aiter = client.astream_query("MATCH (n) RETURN n LIMIT 2")
    assert hasattr(aiter, "__aiter__"), "astream_query must return an async-iterable"

    seen_any = False
    async for _ in aiter:  # noqa: B007
        seen_any = True
        break
    assert isinstance(seen_any, bool)


@pytest.mark.asyncio
async def test_async_query_accepts_optional_params_and_config(adapter: Any) -> None:
    """
    aquery() should accept the same optional params and context as query().
    """
    client = _make_client(adapter, default_namespace="async-ns")

    result = await client.aquery(
        "MATCH (n) RETURN n LIMIT $limit",
        params={"limit": 3},
        dialect="cypher",
        namespace="async-ns",
        timeout_ms=2500,
        config={"configurable": {"user_id": "u-async"}},
        extra_context={"request_id": "req-async"},
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Sync-in-event-loop safety guards (must raise on sync APIs)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_query_guard_raises_in_event_loop(adapter: Any) -> None:
    client = _make_client(adapter)
    with pytest.raises(RuntimeError, match="active asyncio event loop"):
        client.query("MATCH (n) RETURN n LIMIT 1")


@pytest.mark.asyncio
async def test_sync_stream_query_guard_raises_in_event_loop(adapter: Any) -> None:
    client = _make_client(adapter)
    with pytest.raises(RuntimeError, match="active asyncio event loop"):
        list(client.stream_query("MATCH (n) RETURN n LIMIT 1"))


@pytest.mark.asyncio
async def test_sync_capabilities_guard_raises_in_event_loop(adapter: Any) -> None:
    client = _make_client(adapter)
    with pytest.raises(RuntimeError, match="active asyncio event loop"):
        client.capabilities()


@pytest.mark.asyncio
async def test_sync_get_schema_guard_raises_in_event_loop(adapter: Any) -> None:
    client = _make_client(adapter)
    with pytest.raises(RuntimeError, match="active asyncio event loop"):
        client.get_schema()


@pytest.mark.asyncio
async def test_sync_health_guard_raises_in_event_loop(adapter: Any) -> None:
    client = _make_client(adapter)
    with pytest.raises(RuntimeError, match="active asyncio event loop"):
        client.health()


@pytest.mark.asyncio
async def test_sync_upsert_nodes_guard_raises_in_event_loop(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    client = _make_client(adapter)

    # Avoid depending on UpsertNodesSpec shape: the guard triggers before validation.
    class DummySpec:
        nodes: List[Any] = []

    with pytest.raises(RuntimeError, match="active asyncio event loop"):
        client.upsert_nodes(DummySpec())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_sync_upsert_edges_guard_raises_in_event_loop(adapter: Any) -> None:
    client = _make_client(adapter)

    class DummySpec:
        edges: List[Any] = []

    with pytest.raises(RuntimeError, match="active asyncio event loop"):
        client.upsert_edges(DummySpec())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_sync_delete_nodes_guard_raises_in_event_loop(adapter: Any) -> None:
    client = _make_client(adapter)

    class DummySpec:
        filter = {"x": 1}
        ids = None
        namespace = None

    with pytest.raises(RuntimeError, match="active asyncio event loop"):
        client.delete_nodes(DummySpec())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_sync_delete_edges_guard_raises_in_event_loop(adapter: Any) -> None:
    client = _make_client(adapter)

    class DummySpec:
        filter = {"x": 1}
        ids = None
        namespace = None

    with pytest.raises(RuntimeError, match="active asyncio event loop"):
        client.delete_edges(DummySpec())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_sync_bulk_vertices_guard_raises_in_event_loop(adapter: Any) -> None:
    client = _make_client(adapter)

    class DummySpec:
        namespace = None
        limit = 10
        cursor = None
        filter = None

    with pytest.raises(RuntimeError, match="active asyncio event loop"):
        client.bulk_vertices(DummySpec())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_sync_batch_guard_raises_in_event_loop(adapter: Any) -> None:
    client = _make_client(adapter)
    with pytest.raises(RuntimeError, match="active asyncio event loop"):
        client.batch([])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Dialect fallback (NotSupported) behavior
# ---------------------------------------------------------------------------


def test_query_dialect_fallback_sync_when_not_supported(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """
    If the translator raises NotSupported and a dialect was explicitly provided,
    the adapter should retry without dialect (best-effort).
    """
    calls: List[Dict[str, Any]] = []

    class Translator:
        def query(self, raw_query: Mapping[str, Any], **_: Any) -> Any:
            calls.append(dict(raw_query))
            if "dialect" in raw_query:
                raise NotSupported("dialect not supported")
            return "ok"

    _patch_create_graph_translator(monkeypatch, Translator)
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    client = _make_client(adapter)
    out = client.query("MATCH (n) RETURN n", dialect="cypher")
    assert out == "ok"
    assert len(calls) == 2
    assert "dialect" in calls[0]
    assert "dialect" not in calls[1]


@pytest.mark.asyncio
async def test_query_dialect_fallback_async_when_not_supported(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """
    Async dialect fallback mirrors sync behavior.
    """
    calls: List[Dict[str, Any]] = []

    class Translator:
        async def arun_query(self, raw_query: Mapping[str, Any], **_: Any) -> Any:
            calls.append(dict(raw_query))
            if "dialect" in raw_query:
                raise NotSupported("dialect not supported")
            return "ok-async"

    _patch_create_graph_translator(monkeypatch, Translator)
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    client = _make_client(adapter)
    out = await client.aquery("MATCH (n) RETURN n", dialect="cypher")
    assert out == "ok-async"
    assert len(calls) == 2
    assert "dialect" in calls[0]
    assert "dialect" not in calls[1]


# ---------------------------------------------------------------------------
# Query parameter validation
# ---------------------------------------------------------------------------


def test_query_params_rejects_non_mapping_sync(adapter: Any) -> None:
    client = _make_client(adapter)
    with pytest.raises(TypeError, match="params must be a mapping"):
        client.query("MATCH (n) RETURN n", params="not-a-mapping")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_query_params_rejects_non_mapping_async(adapter: Any) -> None:
    client = _make_client(adapter)
    with pytest.raises(TypeError, match="params must be a mapping"):
        await client.aquery("MATCH (n) RETURN n", params="not-a-mapping")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Delete spec validation (filter OR non-empty ids required)
# ---------------------------------------------------------------------------


def test_delete_nodes_requires_filter_or_ids_sync(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    class DummyTranslator:
        def delete_nodes(self, *_: Any, **__: Any) -> Any:
            return "ok"

    _patch_create_graph_translator(monkeypatch, DummyTranslator)
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    client = _make_client(adapter)

    class Spec:
        filter = None
        ids: List[str] = []
        namespace = None

    with pytest.raises(BadRequest, match="either filter or non-empty ids"):
        client.delete_nodes(Spec())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_delete_nodes_requires_filter_or_ids_async(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    class DummyTranslator:
        async def arun_delete_nodes(self, *_: Any, **__: Any) -> Any:
            return "ok"

    _patch_create_graph_translator(monkeypatch, DummyTranslator)
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    client = _make_client(adapter)

    class Spec:
        filter = None
        ids: List[str] = []
        namespace = None

    with pytest.raises(BadRequest, match="either filter or non-empty ids"):
        await client.adelete_nodes(Spec())  # type: ignore[arg-type]


def test_delete_edges_requires_filter_or_ids_sync(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    class DummyTranslator:
        def delete_edges(self, *_: Any, **__: Any) -> Any:
            return "ok"

    _patch_create_graph_translator(monkeypatch, DummyTranslator)
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    client = _make_client(adapter)

    class Spec:
        filter = None
        ids: List[str] = []
        namespace = None

    with pytest.raises(BadRequest, match="either filter or non-empty ids"):
        client.delete_edges(Spec())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_delete_edges_requires_filter_or_ids_async(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    class DummyTranslator:
        async def arun_delete_edges(self, *_: Any, **__: Any) -> Any:
            return "ok"

    _patch_create_graph_translator(monkeypatch, DummyTranslator)
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    client = _make_client(adapter)

    class Spec:
        filter = None
        ids: List[str] = []
        namespace = None

    with pytest.raises(BadRequest, match="either filter or non-empty ids"):
        await client.adelete_edges(Spec())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Upsert edges validation (no mutation, list returned)
# ---------------------------------------------------------------------------


def test_validate_upsert_edges_returns_list_without_mutating_spec(adapter: Any) -> None:
    client = _make_client(adapter)

    class Edge:
        def __init__(self, id: str) -> None:
            self.id = id

    class Spec:
        def __init__(self) -> None:
            self.edges = (Edge("e1"), Edge("e2"))

    spec = Spec()
    original_edges_obj = spec.edges

    edges = client._validate_upsert_edges_spec(spec)  # noqa: SLF001
    assert isinstance(edges, list)
    assert len(edges) == 2
    # Ensure the original spec attribute is not mutated/reassigned.
    assert spec.edges is original_edges_obj


def test_upsert_edges_uses_validated_edges_list(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def upsert_edges(self, raw_edges: Any, **_: Any) -> Any:
            captured["raw_edges"] = raw_edges
            return "ok"

    _patch_create_graph_translator(monkeypatch, DummyTranslator)
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    client = _make_client(adapter)

    class Edge:
        def __init__(self, id: str) -> None:
            self.id = id

    class Spec:
        def __init__(self) -> None:
            self.edges = (Edge("e1"), Edge("e2"))
            self.namespace = None

    out = client.upsert_edges(Spec())  # type: ignore[arg-type]
    assert out == "ok"
    assert isinstance(captured["raw_edges"], list)
    assert len(captured["raw_edges"]) == 2


# ---------------------------------------------------------------------------
# Streaming chunk validation + awaitable stream normalization
# ---------------------------------------------------------------------------


def test_stream_query_invalid_chunk_triggers_validation(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If the translator yields invalid chunks, stream_query should pass them
    through validate_graph_result_type, and failures there should surface
    to the caller.
    """
    captured: Dict[str, Any] = {}

    class BadChunkTranslator:
        def query_stream(self, *_: Any, **__: Any):
            yield "not-a-chunk"

    _patch_create_graph_translator(monkeypatch, BadChunkTranslator)

    def fake_validate_graph_result_type(result: Any, **kwargs: Any) -> Any:
        captured["result"] = result
        captured["kwargs"] = kwargs
        raise RuntimeError("forced validation failure for chunk")

    monkeypatch.setattr(langchain_adapter_module, "validate_graph_result_type", fake_validate_graph_result_type)

    client = _make_client(adapter)

    iterator = client.stream_query("MATCH (n) RETURN n")
    with pytest.raises(RuntimeError, match="forced validation failure for chunk"):
        next(iterator)

    assert captured.get("result") == "not-a-chunk"
    assert "expected_type" in captured.get("kwargs", {})


@pytest.mark.asyncio
async def test_astream_query_handles_awaitable_return_from_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    astream_query should tolerate translators that return an awaitable resolving
    to an async iterator (defensive normalization).
    """

    class Translator:
        async def arun_query_stream(self, *_: Any, **__: Any) -> Any:
            async def gen() -> Any:
                yield {"records": [1], "is_final": True}

            return gen()

    _patch_create_graph_translator(monkeypatch, Translator)

    # Make validation a pass-through so we don't depend on QueryChunk shapes here.
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    client = _make_client(adapter)

    seen = 0
    async for _ in client.astream_query("MATCH (n) RETURN n LIMIT 1"):
        seen += 1
    assert seen >= 1


@pytest.mark.asyncio
async def test_astream_query_invalid_chunk_triggers_validation_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Async streaming path should also exercise validate_graph_result_type when
    chunks are invalid.
    """
    captured: Dict[str, Any] = {}

    class BadChunkTranslator:
        async def arun_query_stream(self, *_: Any, **__: Any) -> Any:
            async def gen() -> Any:
                yield "not-a-chunk-async"

            return gen()

    _patch_create_graph_translator(monkeypatch, BadChunkTranslator)

    def fake_validate_graph_result_type(result: Any, **kwargs: Any) -> Any:
        captured["result"] = result
        captured["kwargs"] = kwargs
        raise RuntimeError("forced validation failure for async chunk")

    monkeypatch.setattr(langchain_adapter_module, "validate_graph_result_type", fake_validate_graph_result_type)

    client = _make_client(adapter)

    with pytest.raises(RuntimeError, match="forced validation failure for async chunk"):
        async for _ in client.astream_query("MATCH (n) RETURN n"):
            break

    assert captured.get("result") == "not-a-chunk-async"
    assert "expected_type" in captured.get("kwargs", {})


# ---------------------------------------------------------------------------
# Bulk vertices / batch wiring (LangChain adapter)
# ---------------------------------------------------------------------------


def test_bulk_vertices_builds_raw_request_and_calls_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def bulk_vertices(self, raw_request: Mapping[str, Any], **kwargs: Any) -> Any:
            captured["raw_request"] = dict(raw_request)
            captured["framework_ctx"] = dict(kwargs.get("framework_ctx") or {})
            return "bulk-result"

    _patch_create_graph_translator(monkeypatch, DummyTranslator)
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    client = _make_client(adapter, framework_version="fw-bulk-1")

    class Spec:
        namespace = "ns-bulk"
        limit = 42
        cursor = "cursor-token"
        filter = {"foo": "bar"}

    result = client.bulk_vertices(Spec())  # type: ignore[arg-type]
    assert result == "bulk-result"

    assert captured["raw_request"] == {
        "namespace": "ns-bulk",
        "limit": 42,
        "cursor": "cursor-token",
        "filter": {"foo": "bar"},
    }
    fw_ctx = captured["framework_ctx"]
    assert fw_ctx.get("framework") == "langchain"
    assert fw_ctx.get("operation") == "bulk_vertices"
    assert fw_ctx.get("namespace") == "ns-bulk"
    assert fw_ctx.get("framework_version") == "fw-bulk-1"


@pytest.mark.asyncio
async def test_abulk_vertices_builds_raw_request_and_calls_translator_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_bulk_vertices(self, raw_request: Mapping[str, Any], **kwargs: Any) -> Any:
            captured["raw_request"] = dict(raw_request)
            captured["framework_ctx"] = dict(kwargs.get("framework_ctx") or {})
            return "bulk-result-async"

    _patch_create_graph_translator(monkeypatch, DummyTranslator)
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    client = _make_client(adapter, framework_version="fw-abulk-1")

    class Spec:
        namespace = "ns-abulk"
        limit = 7
        cursor = None
        filter = {"bar": 1}

    result = await client.abulk_vertices(Spec())  # type: ignore[arg-type]
    assert result == "bulk-result-async"

    assert captured["raw_request"] == {
        "namespace": "ns-abulk",
        "limit": 7,
        "cursor": None,
        "filter": {"bar": 1},
    }
    fw_ctx = captured["framework_ctx"]
    assert fw_ctx.get("framework") == "langchain"
    assert fw_ctx.get("operation") == "bulk_vertices"
    assert fw_ctx.get("namespace") == "ns-abulk"
    assert fw_ctx.get("framework_version") == "fw-abulk-1"


def test_batch_builds_raw_batch_ops_and_calls_translator(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def batch(self, raw_batch_ops: List[Mapping[str, Any]], **kwargs: Any) -> Any:
            captured["raw_batch_ops"] = [dict(op) for op in raw_batch_ops]
            captured["framework_ctx"] = dict(kwargs.get("framework_ctx") or {})
            return "batch-result"

    _patch_create_graph_translator(monkeypatch, DummyTranslator)
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    # Stub validation for wiring-level test only.
    monkeypatch.setattr(langchain_adapter_module, "validate_batch_operations", lambda *_a, **_k: None)

    client = _make_client(adapter, framework_version="fw-batch-1")

    class DummyBatchOp:
        def __init__(self, op: str, args: Mapping[str, Any]) -> None:
            self.op = op
            self.args = dict(args)

    ops = [DummyBatchOp("upsert_nodes", {"id": "1"}), DummyBatchOp("delete_nodes", {"ids": ["1", "2"]})]

    result = client.batch(ops)  # type: ignore[arg-type]
    assert result == "batch-result"

    assert captured["raw_batch_ops"] == [
        {"op": "upsert_nodes", "args": {"id": "1"}},
        {"op": "delete_nodes", "args": {"ids": ["1", "2"]}},
    ]
    fw_ctx = captured["framework_ctx"]
    assert fw_ctx.get("framework") == "langchain"
    assert fw_ctx.get("operation") == "batch"
    assert fw_ctx.get("framework_version") == "fw-batch-1"


@pytest.mark.asyncio
async def test_abatch_builds_raw_batch_ops_and_calls_translator_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_batch(self, raw_batch_ops: List[Mapping[str, Any]], **kwargs: Any) -> Any:
            captured["raw_batch_ops"] = [dict(op) for op in raw_batch_ops]
            captured["framework_ctx"] = dict(kwargs.get("framework_ctx") or {})
            return "batch-result-async"

    _patch_create_graph_translator(monkeypatch, DummyTranslator)
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    monkeypatch.setattr(langchain_adapter_module, "validate_batch_operations", lambda *_a, **_k: None)

    client = _make_client(adapter, framework_version="fw-abatch-1")

    class DummyBatchOp:
        def __init__(self, op: str, args: Mapping[str, Any]) -> None:
            self.op = op
            self.args = dict(args)

    ops = [DummyBatchOp("upsert_edges", {"id": "e-1"}), DummyBatchOp("delete_edges", {"ids": ["e-1", "e-2"]})]

    result = await client.abatch(ops)  # type: ignore[arg-type]
    assert result == "batch-result-async"

    assert captured["raw_batch_ops"] == [
        {"op": "upsert_edges", "args": {"id": "e-1"}},
        {"op": "delete_edges", "args": {"ids": ["e-1", "e-2"]}},
    ]
    fw_ctx = captured["framework_ctx"]
    assert fw_ctx.get("framework") == "langchain"
    assert fw_ctx.get("operation") == "batch"
    assert fw_ctx.get("framework_version") == "fw-abatch-1"


# ---------------------------------------------------------------------------
# Capabilities / health passthrough (basic + sync/async parity)
# ---------------------------------------------------------------------------


def test_capabilities_and_health_basic(adapter: Any) -> None:
    client = _make_client(adapter)
    assert isinstance(client.capabilities(), Mapping)
    assert isinstance(client.health(), Mapping)


@pytest.mark.asyncio
async def test_async_capabilities_and_health_basic_parity(adapter: Any) -> None:
    client = _make_client(adapter)

    caps = client.capabilities()
    health = client.health()

    acaps = await client.acapabilities()
    ahealth = await client.ahealth()

    assert isinstance(acaps, Mapping)
    assert isinstance(ahealth, Mapping)
    assert set(acaps.keys()) == set(caps.keys())
    assert set(ahealth.keys()) == set(health.keys())


# ---------------------------------------------------------------------------
# Resource management (context managers + executor lifecycle)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_closes_underlying_adapter() -> None:
    """
    __enter__/__exit__ and __aenter__/__aexit__ should call close/aclose on
    the underlying graph adapter when those methods exist.
    """

    class ClosingGraphAdapter:
        def __init__(self) -> None:
            self.closed = False
            self.aclosed = False

        def capabilities(self) -> Dict[str, Any]:
            return {}

        def health(self) -> Dict[str, Any]:
            return {}

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.aclosed = True

    adapter = ClosingGraphAdapter()

    with CorpusLangChainGraphClient(adapter=adapter) as client:
        assert client is not None
    assert adapter.closed is True

    adapter2 = ClosingGraphAdapter()
    client2 = CorpusLangChainGraphClient(adapter=adapter2)

    async with client2:
        assert client2 is not None
    assert adapter2.aclosed is True


def test_close_idempotent_and_shuts_down_tool_executor(adapter: Any) -> None:
    """
    close() should be idempotent and also stop the optional bounded tool executor.

    We avoid importing LangChain here; tool creation itself exercises the soft import.
    """
    client = _make_client(adapter)

    # First close should succeed.
    client.close()
    # Second close should be a no-op.
    client.close()

    # The executor is a module-level optional; after close it should be cleared.
    assert getattr(langchain_adapter_module, "_LANGCHAIN_TOOL_BRIDGE_EXECUTOR") is None


# ---------------------------------------------------------------------------
# Real LangChain tool integration tests (no skips; hard-fail if LangChain missing)
# ---------------------------------------------------------------------------


def test_create_langchain_graph_tools_returns_real_tools(adapter: Any) -> None:
    """
    Tool creation should return a list of tool objects.

    This is a *real integration* test:
    - It will fail if LangChain is not installed (by design; no skips).
    - It does not import LangChain in the test; the adapter owns soft import.
    """
    client = _make_client(adapter)
    tools = create_langchain_graph_tools(client, name_prefix="graph", description_prefix="Corpus graph tool")
    assert isinstance(tools, list)
    assert len(tools) >= 4
    assert all(hasattr(t, "name") for t in tools)
    assert all(isinstance(getattr(t, "name"), str) for t in tools)


def test_langchain_query_tool_runs_and_returns_json(adapter: Any) -> None:
    """
    Query tool should be invokable via LangChain tool surface and return JSON.

    This is a real tool-runtime integration check.
    """
    client = _make_client(adapter)
    tools = create_langchain_graph_tools(client, name_prefix="graph", description_prefix="Corpus graph tool")

    query_tool = next(t for t in tools if str(getattr(t, "name", "")).endswith("_query"))

    out = _tool_sync_call(query_tool, "MATCH (n) RETURN n LIMIT 1")
    assert isinstance(out, str)

    parsed = json.loads(out)
    assert "result" in parsed


@pytest.mark.asyncio
async def test_langchain_query_tool_async_runs_and_returns_json(adapter: Any) -> None:
    """
    Query tool should be async-invokable and return JSON.

    This verifies that tool async entrypoints delegate to client.aquery(...).
    """
    client = _make_client(adapter)
    tools = create_langchain_graph_tools(client, name_prefix="graph", description_prefix="Corpus graph tool")

    query_tool = next(t for t in tools if str(getattr(t, "name", "")).endswith("_query"))

    out = await _tool_async_call(query_tool, "MATCH (n) RETURN n LIMIT 1")
    assert isinstance(out, str)

    parsed = json.loads(out)
    assert "result" in parsed


@pytest.mark.asyncio
async def test_langchain_sync_tool_called_in_event_loop_uses_thread_bridge(adapter: Any) -> None:
    """
    In an async runtime, invoking the *sync* tool entrypoint must not trip
    the client's sync-in-event-loop guard because the tool bridges work
    to a bounded worker thread.
    """
    client = _make_client(adapter)
    tools = create_langchain_graph_tools(client, name_prefix="graph", description_prefix="Corpus graph tool")

    query_tool = next(t for t in tools if str(getattr(t, "name", "")).endswith("_query"))

    # This call happens while the pytest-asyncio event loop is running.
    out = _tool_sync_call(query_tool, "MATCH (n) RETURN n LIMIT 1")
    parsed = json.loads(out)
    assert "result" in parsed


def test_langchain_stream_tool_max_chunks_validation_and_clamp(adapter: Any) -> None:
    """
    Streaming tool should:
    - reject max_chunks <= 0
    - clamp extremely large max_chunks to a safe bound
    """
    client = _make_client(adapter)
    tools = create_langchain_graph_tools(client, name_prefix="graph", description_prefix="Corpus graph tool")

    stream_tool = next(t for t in tools if str(getattr(t, "name", "")).endswith("_stream_query"))

    with pytest.raises(Exception, match="max_chunks must be > 0"):
        _tool_sync_call(stream_tool, {"query": "MATCH (n) RETURN n", "max_chunks": 0})

    out = _tool_sync_call(stream_tool, {"query": "MATCH (n) RETURN n", "max_chunks": 10_000})
    parsed = json.loads(out)
    assert "chunks" in parsed
    assert isinstance(parsed["chunks"], list)


def test_langchain_batch_tool_rejects_malformed_ops(adapter: Any) -> None:
    """
    Batch tool should reject malformed ops items (non-mapping, missing keys, wrong types).
    """
    client = _make_client(adapter)
    tools = create_langchain_graph_tools(client, name_prefix="graph", description_prefix="Corpus graph tool")

    batch_tool = next(t for t in tools if str(getattr(t, "name", "")).endswith("_batch"))

    with pytest.raises(TypeError, match="must be a mapping"):
        _tool_sync_call(batch_tool, {"ops": ["not-a-mapping"]})

    with pytest.raises(TypeError, match=r"\['op'\] must be a non-empty string"):
        _tool_sync_call(batch_tool, {"ops": [{"op": "", "args": {}}]})


def test_create_corpus_graph_tool_uses_defaults_via_translator_capture(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    create_corpus_graph_tool should build an internal client with defaults and
    drive those defaults into raw_query when the tool is invoked.

    We verify this by patching create_graph_translator to capture raw_query.
    """
    captured: Dict[str, Any] = {}

    class CapturingTranslator:
        def query(self, raw_query: Mapping[str, Any], **_: Any) -> Any:
            captured["raw_query"] = dict(raw_query)
            return {"ok": True}

    monkeypatch.setattr(
        langchain_adapter_module,
        "create_graph_translator",
        lambda *_a, **_k: CapturingTranslator(),
    )
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    tool = create_corpus_graph_tool(
        graph_adapter=adapter,
        default_dialect="cypher",
        default_namespace="prod",
        default_timeout_ms=9999,
        framework_version="lc-fw-2.0",
        name="my_graph_tool",
        description="My graph tool for LangChain",
    )

    # Invoke the tool so it executes the captured translator.query(...)
    out = _tool_sync_call(tool, "MATCH (n) RETURN n LIMIT 1")
    assert isinstance(out, str)
    assert captured["raw_query"]["dialect"] == "cypher"
    assert captured["raw_query"]["namespace"] == "prod"
    assert captured["raw_query"]["timeout_ms"] == 9999


# ---------------------------------------------------------------------------
# validate_batch_operations behavior (direct adapter call, not tool)
# ---------------------------------------------------------------------------


def test_batch_validation_rejects_empty_ops(adapter: Any) -> None:
    """
    Adapter batch() should reject empty operations via validate_batch_operations.
    """
    client = _make_client(adapter)
    with pytest.raises((ValueError, TypeError)):
        client.batch([])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_abatch_validation_rejects_empty_ops(adapter: Any) -> None:
    client = _make_client(adapter)
    with pytest.raises((ValueError, TypeError)):
        await client.abatch([])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Main entrypoint for local runs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
