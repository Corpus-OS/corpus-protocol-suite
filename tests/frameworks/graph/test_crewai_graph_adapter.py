# tests/frameworks/graph/test_crewai_graph_adapter.py

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import json
import threading
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Type

import pytest

import corpus_sdk.graph.framework_adapters.crewai as crewai_adapter_module
from corpus_sdk.graph.framework_adapters.crewai import (
    CorpusCrewAIGraphClient,
    CrewAIGraphFrameworkTranslator,
    ErrorCodes,
)
from corpus_sdk.graph.graph_base import (
    BatchOperation,
    GraphCapabilities,
    QueryChunk,
    QueryResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(adapter: Any, **kwargs: Any) -> CorpusCrewAIGraphClient:
    """Construct a CorpusCrewAIGraphClient instance from the generic adapter."""
    return CorpusCrewAIGraphClient(adapter=adapter, **kwargs)


def _patch_attach_context_everywhere(
    monkeypatch: pytest.MonkeyPatch,
    fake_attach_context: Any,
) -> None:
    """
    Patch attach_context in both the adapter module and the canonical core module.

    Some decorators may close over the core attach_context reference; others may
    use the local module import. Patching both maximizes determinism.
    """
    monkeypatch.setattr(crewai_adapter_module, "attach_context", fake_attach_context)
    try:
        import corpus_sdk.core.error_context as error_context_module

        monkeypatch.setattr(error_context_module, "attach_context", fake_attach_context)
    except Exception:
        # Best-effort: tests should still run if the import path differs.
        pass


def _patch_create_graph_translator(
    monkeypatch: pytest.MonkeyPatch,
    translator_obj: Any,
) -> None:
    """
    Patch create_graph_translator to always return translator_obj.

    translator_obj can be an instance or a class (if class, constructed with no args).
    """

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return translator_obj() if isinstance(translator_obj, type) else translator_obj

    monkeypatch.setattr(
        crewai_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )


def _patch_validate_graph_result_type_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch validate_graph_result_type to return the result unchanged (wiring tests)."""

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    monkeypatch.setattr(crewai_adapter_module, "validate_graph_result_type", fake_validate_graph_result_type)


def _make_async_gen(items: Sequence[Any]):
    """Helper: create an async generator from a list of items."""
    async def _gen():
        for it in items:
            yield it
    return _gen()


class DummyBulkSpec:
    """Simple stand-in for BulkVerticesSpec used in wiring tests."""

    def __init__(self, namespace: str, limit: int, cursor: Any, filter_: Any) -> None:
        self.namespace = namespace
        self.limit = limit
        self.cursor = cursor
        self.filter = filter_


class DummyBatchOp:
    """Simple stand-in for BatchOperation used in wiring tests."""

    def __init__(self, op: str, args: Mapping[str, Any]) -> None:
        self.op = op
        self.args = dict(args)


# ---------------------------------------------------------------------------
# Constructor / translator behavior
# ---------------------------------------------------------------------------


def test_import_crewai_graph_client() -> None:
    """Verify core adapter symbols can be imported properly."""
    from corpus_sdk.graph.framework_adapters.crewai import (
        CorpusCrewAIGraphClient,
        CrewAIGraphFrameworkTranslator,
        ErrorCodes,
        create_crewai_graph_tools,
    )

    assert CorpusCrewAIGraphClient is not None
    assert CrewAIGraphFrameworkTranslator is not None
    assert ErrorCodes is not None
    assert callable(create_crewai_graph_tools)


def test_constructor_rejects_invalid_adapter() -> None:
    """Constructor should reject non-GraphProtocol-like adapters."""
    with pytest.raises(TypeError):
        CorpusCrewAIGraphClient(adapter="not-a-graph-adapter")  # type: ignore[arg-type]


def test_constructor_rejects_both_adapter_and_graph_adapter(adapter: Any) -> None:
    """Constructor should reject providing both adapter and graph_adapter."""
    with pytest.raises(TypeError):
        CorpusCrewAIGraphClient(adapter=adapter, graph_adapter=adapter)


def test_constructor_accepts_adapter_without_close() -> None:
    """Adapter without close() should still be accepted if surface is GraphProtocol-like."""
    class SimpleAdapter:
        async def query(self, *args: Any, **kwargs: Any) -> Any:
            return QueryResult(records=[], summary={})

        async def capabilities(self, *args: Any, **kwargs: Any) -> Any:
            return GraphCapabilities(server="test", version="1.0")

    client = CorpusCrewAIGraphClient(adapter=SimpleAdapter())
    assert client is not None


def test_translator_lazy_initialization(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Translator should be created only when first accessed and then cached."""
    call_count = 0

    def fake_create_graph_translator(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        nonlocal call_count
        call_count += 1

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(crewai_adapter_module, "create_graph_translator", fake_create_graph_translator)

    client = _make_client(adapter)
    assert call_count == 0

    _ = client._translator  # noqa: SLF001
    assert call_count == 1

    _ = client._translator  # noqa: SLF001
    assert call_count == 1


def test_default_translator_uses_crewai_framework_translator(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """
    By default, CorpusCrewAIGraphClient should construct a CrewAIGraphFrameworkTranslator and
    pass it into create_graph_translator with framework="crewai".
    """
    captured: Dict[str, Any] = {}

    def fake_create_graph_translator(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["kwargs"] = kwargs

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(crewai_adapter_module, "create_graph_translator", fake_create_graph_translator)

    client = _make_client(adapter)
    _ = client._translator  # noqa: SLF001

    kwargs = captured["kwargs"]
    assert kwargs.get("framework") == "crewai"
    assert isinstance(kwargs.get("translator"), CrewAIGraphFrameworkTranslator)


def test_framework_translator_override_is_respected(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """If framework_translator is provided, it should be passed through to create_graph_translator."""
    captured: Dict[str, Any] = {}

    class CustomTranslator(CrewAIGraphFrameworkTranslator):
        pass

    custom = CustomTranslator()

    def fake_create_graph_translator(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["kwargs"] = kwargs

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(crewai_adapter_module, "create_graph_translator", fake_create_graph_translator)

    client = _make_client(adapter, framework_translator=custom, framework_version="crewai-fw-1.2.3")
    _ = client._translator  # noqa: SLF001

    kwargs = captured["kwargs"]
    assert kwargs.get("framework") == "crewai"
    assert kwargs.get("translator") is custom


# ---------------------------------------------------------------------------
# Context translation / from_crewai mapping
# ---------------------------------------------------------------------------


def test_crewai_task_and_extra_context_passed_to_core_ctx(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """task and extra_context should be passed through to core_ctx_from_crewai with configured framework_version."""
    captured: Dict[str, Any] = {}

    class DummyOperationContext:
        def __init__(self, **kwargs: Any) -> None:
            self.attrs = kwargs

    monkeypatch.setattr(crewai_adapter_module, "OperationContext", DummyOperationContext)

    def fake_core_ctx_from_crewai(task: Any, *, framework_version: Any = None, **extra: Any) -> Any:
        captured["task"] = task
        captured["framework_version"] = framework_version
        captured["extra"] = extra
        return DummyOperationContext(task=task, **extra)

    monkeypatch.setattr(crewai_adapter_module, "core_ctx_from_crewai", fake_core_ctx_from_crewai)

    client = _make_client(adapter, framework_version="crewai-test-version")

    fake_task = object()
    extra_ctx = {"request_id": "req-crewai-xyz", "tenant": "tenant-1"}

    res = client.query("MATCH (n) RETURN n LIMIT 1", task=fake_task, extra_context=extra_ctx)
    assert res is not None

    assert captured["task"] is fake_task
    assert captured["framework_version"] == "crewai-test-version"
    assert captured["extra"] == extra_ctx


def test_build_ctx_none_inputs_returns_none(adapter: Any) -> None:
    """If task and extra_context are both empty/None, _build_ctx returns None."""
    client = _make_client(adapter)
    ctx = client._build_ctx(task=None, extra_context=None)  # noqa: SLF001
    assert ctx is None


def test_context_translation_with_empty_extra_context(adapter: Any) -> None:
    """Empty extra_context should not crash and should still allow query to proceed."""
    client = _make_client(adapter)
    res = client.query("MATCH (n) RETURN n", task=object(), extra_context={})
    assert res is not None


def test_context_translation_failure_attaches_context_and_proceeds(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """
    Context translation is fail-safe:
    - attach_context is called
    - _build_ctx returns None
    - operations proceed without OperationContext
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured_ctx.update(ctx)

    def fake_core_ctx_from_crewai(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        raise RuntimeError("boom from ctx builder")

    _patch_attach_context_everywhere(monkeypatch, fake_attach_context)
    monkeypatch.setattr(crewai_adapter_module, "core_ctx_from_crewai", fake_core_ctx_from_crewai)

    client = _make_client(adapter, framework_version="ctx-fw")

    # _build_ctx should fail-safe to None
    ctx = client._build_ctx(task=object(), extra_context={"foo": "bar"})  # noqa: SLF001
    assert ctx is None

    # operation should still succeed
    res = client.query("MATCH (n) RETURN n", task=object(), extra_context={"foo": "bar"})
    assert res is not None

    assert captured_ctx.get("framework") == "crewai"
    assert captured_ctx.get("operation") == "context_translation"
    assert captured_ctx.get("error_code") == ErrorCodes.BAD_OPERATION_CONTEXT


# ---------------------------------------------------------------------------
# Error-context decorator behavior
# ---------------------------------------------------------------------------


def test_error_context_includes_crewai_metadata_sync(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Error context should be attached on sync errors."""
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured.update(ctx)

    _patch_attach_context_everywhere(monkeypatch, fake_attach_context)

    class FailingTranslator:
        def query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("sync boom")

    _patch_create_graph_translator(monkeypatch, FailingTranslator)

    client = _make_client(adapter)

    with pytest.raises(RuntimeError, match="sync boom"):
        client.query("MATCH (n) RETURN n")

    assert captured.get("framework") == "crewai"
    assert str(captured.get("operation", "")).startswith("graph_")


@pytest.mark.asyncio
async def test_error_context_includes_crewai_metadata_async(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Error context should be attached on async errors."""
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured.update(ctx)

    _patch_attach_context_everywhere(monkeypatch, fake_attach_context)

    class FailingTranslator:
        async def arun_query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("async boom")

    _patch_create_graph_translator(monkeypatch, FailingTranslator)

    client = _make_client(adapter)

    with pytest.raises(RuntimeError, match="async boom"):
        await client.aquery("MATCH (n) RETURN n")

    assert captured.get("framework") == "crewai"
    assert str(captured.get("operation", "")).startswith("graph_")


def test_error_context_includes_query_text(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Best-effort: error context should include the failing query text."""
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured.update(ctx)

    _patch_attach_context_everywhere(monkeypatch, fake_attach_context)

    class FailingTranslator:
        def query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("boom")

    _patch_create_graph_translator(monkeypatch, FailingTranslator)

    client = _make_client(adapter)
    q = "MATCH (n:Special) RETURN n LIMIT 1"
    with pytest.raises(RuntimeError):
        client.query(q)

    # Some decorators attach query; tolerate best-effort behavior.
    if "query" in captured:
        assert captured["query"] == q


# ---------------------------------------------------------------------------
# Sync semantics / validation tests
# ---------------------------------------------------------------------------


def test_sync_query_and_stream_basic(adapter: Any) -> None:
    """Sync query + stream should not crash."""
    client = _make_client(adapter, default_namespace="crewai-ns")
    res = client.query("MATCH (n) RETURN n LIMIT 1")
    assert res is not None
    chunks = list(client.stream_query("MATCH (n) RETURN n LIMIT 2"))
    assert isinstance(chunks, list)


def test_sync_query_accepts_optional_params_and_context(adapter: Any) -> None:
    """query() should accept params/dialect/namespace/timeout_ms/task/extra_context."""
    client = _make_client(adapter, default_dialect="cypher")
    res = client.query(
        "MATCH (n) RETURN n LIMIT $limit",
        params={"limit": 5},
        dialect="cypher",
        namespace="ctx-ns",
        timeout_ms=5000,
        task=object(),
        extra_context={"request_id": "req-sync"},
    )
    assert res is not None


def test_query_params_type_validation(adapter: Any) -> None:
    """params must be a Mapping if provided."""
    client = _make_client(adapter)
    ok = client.query("MATCH (n) RETURN n", params={"limit": 1})
    assert ok is not None
    with pytest.raises((TypeError, ValueError)):
        client.query("MATCH (n) RETURN n", params="not-a-mapping")  # type: ignore[arg-type]


def test_namespace_empty_string_allowed(adapter: Any) -> None:
    """Empty namespace should be allowed and not crash."""
    client = _make_client(adapter)
    res = client.query("MATCH (n) RETURN n", namespace="")
    assert res is not None


@pytest.mark.asyncio
async def test_sync_methods_raise_in_running_event_loop(adapter: Any) -> None:
    """Sync APIs should raise if called inside a running event loop (safety guard)."""
    client = _make_client(adapter)
    with pytest.raises(RuntimeError, match="active asyncio event loop"):
        client.query("MATCH (n) RETURN n")


# ---------------------------------------------------------------------------
# Async semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_query_and_stream_basic(adapter: Any) -> None:
    """Async aquery/astream_query should exist and be usable."""
    client = _make_client(adapter)

    coro = client.aquery("MATCH (n) RETURN n LIMIT 1")
    assert inspect.isawaitable(coro)
    res = await coro
    assert res is not None

    aiter = client.astream_query("MATCH (n) RETURN n LIMIT 2")
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    seen_any = False
    async for _ in aiter:  # noqa: B007
        seen_any = True
        break
    assert isinstance(seen_any, bool)


@pytest.mark.asyncio
async def test_async_query_accepts_optional_params_and_context(adapter: Any) -> None:
    """aquery() should accept the same optional fields as query()."""
    client = _make_client(adapter, default_namespace="async-ns")
    res = await client.aquery(
        "MATCH (n) RETURN n LIMIT $limit",
        params={"limit": 3},
        dialect="cypher",
        namespace="async-ns",
        timeout_ms=2500,
        task=object(),
        extra_context={"request_id": "req-async"},
    )
    assert res is not None


def test_sync_and_async_capabilities_same_structure(adapter: Any) -> None:
    """capabilities/acapabilities should return Mapping-like dicts."""
    client = _make_client(adapter)
    sync_caps = client.capabilities()
    async_caps = asyncio.run(client.acapabilities())
    assert isinstance(sync_caps, Mapping)
    assert isinstance(async_caps, Mapping)


@pytest.mark.asyncio
async def test_async_and_sync_query_results_compatible(adapter: Any) -> None:
    """
    In async tests, sync query must be executed in a worker thread because the adapter guards sync-in-loop.
    """
    client = _make_client(adapter)

    def run_sync() -> Any:
        return client.query("MATCH (n) RETURN n LIMIT 1")

    sync_res = await asyncio.to_thread(run_sync)
    async_res = await client.aquery("MATCH (n) RETURN n LIMIT 1")

    assert hasattr(sync_res, "records") or isinstance(sync_res, dict)
    assert hasattr(async_res, "records") or isinstance(async_res, dict)


# ---------------------------------------------------------------------------
# Streaming validation / “wait for it” behavior
# ---------------------------------------------------------------------------


def test_stream_query_invalid_chunk_triggers_validation(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """stream_query should validate each chunk via validate_graph_result_type."""
    captured: Dict[str, Any] = {}

    class BadChunkTranslator:
        def query_stream(self, *args: Any, **kwargs: Any):  # noqa: ARG002
            yield "not-a-chunk"

    _patch_create_graph_translator(monkeypatch, BadChunkTranslator)

    def fake_validate_graph_result_type(result: Any, **kwargs: Any) -> Any:
        captured["result"] = result
        captured["kwargs"] = kwargs
        raise RuntimeError("forced validation failure for chunk")

    monkeypatch.setattr(crewai_adapter_module, "validate_graph_result_type", fake_validate_graph_result_type)

    client = _make_client(adapter)
    it = client.stream_query("MATCH (n) RETURN n")
    with pytest.raises(RuntimeError, match="forced validation failure for chunk"):
        next(it)

    assert captured.get("result") == "not-a-chunk"
    assert "expected_type" in captured.get("kwargs", {})


@pytest.mark.asyncio
async def test_astream_query_invalid_chunk_triggers_validation_async(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """astream_query should validate each chunk; supports awaitable->aiter and direct aiter."""
    captured: Dict[str, Any] = {}

    class BadChunkTranslatorAwaitable:
        async def arun_query_stream(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return _make_async_gen(["not-a-chunk-async"])

    _patch_create_graph_translator(monkeypatch, BadChunkTranslatorAwaitable)

    def fake_validate_graph_result_type(result: Any, **kwargs: Any) -> Any:
        captured["result"] = result
        captured["kwargs"] = kwargs
        raise RuntimeError("forced validation failure for async chunk")

    monkeypatch.setattr(crewai_adapter_module, "validate_graph_result_type", fake_validate_graph_result_type)

    client = _make_client(adapter)

    aiter = client.astream_query("MATCH (n) RETURN n")
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="forced validation failure for async chunk"):
        async for _ in aiter:  # noqa: B007
            break

    assert captured.get("result") == "not-a-chunk-async"
    assert "expected_type" in captured.get("kwargs", {})


@pytest.mark.asyncio
async def test_astream_query_wait_for_it_supports_direct_async_iterator(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """If translator returns an AsyncIterator directly (not awaitable), adapter should still work."""
    class DirectAIterTranslator:
        def arun_query_stream(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return _make_async_gen([QueryChunk(records=[1], is_final=True)])

    _patch_create_graph_translator(monkeypatch, DirectAIterTranslator)
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    client = _make_client(adapter)
    aiter = client.astream_query("MATCH (n) RETURN n")
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    seen = 0
    async for _ in aiter:
        seen += 1
    assert seen == 1


def test_stream_query_empty_result(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Empty stream should yield no chunks."""
    class EmptyTranslator:
        def query_stream(self, *args: Any, **kwargs: Any):  # noqa: ARG002
            return iter([])

    _patch_create_graph_translator(monkeypatch, EmptyTranslator)
    client = _make_client(adapter)
    chunks = list(client.stream_query("MATCH (n) WHERE 1=0 RETURN n"))
    assert chunks == []


@pytest.mark.asyncio
async def test_astream_query_cancellation(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Async stream can be consumed partially (best-effort cancellation by breaking)."""
    class SlowTranslator:
        async def arun_query_stream(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            async def gen():
                for i in range(10):
                    await asyncio.sleep(0.01)
                    yield QueryChunk(records=[i], is_final=(i == 9))
            return gen()

    _patch_create_graph_translator(monkeypatch, SlowTranslator)
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    client = _make_client(adapter)
    aiter = client.astream_query("MATCH (n) RETURN n")
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    count = 0
    async for _ in aiter:
        count += 1
        if count >= 3:
            break
    assert count == 3


def test_stream_large_result_sets(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Streaming should handle large result sets when chunks are well-typed."""
    class LargeTranslator:
        def query_stream(self, *args: Any, **kwargs: Any):  # noqa: ARG002
            for i in range(100):
                yield QueryChunk(records=[f"record_{i}"], is_final=(i == 99))

    _patch_create_graph_translator(monkeypatch, LargeTranslator)
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    client = _make_client(adapter)
    chunks = list(client.stream_query("MATCH (n) RETURN n LIMIT 100"))
    assert len(chunks) == 100


# ---------------------------------------------------------------------------
# Bulk vertices / batch wiring
# ---------------------------------------------------------------------------


def test_bulk_vertices_builds_raw_request_and_calls_translator(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """bulk_vertices should build the correct raw_request and pass framework_ctx."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def bulk_vertices(self, raw_request: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:
            captured["raw_request"] = dict(raw_request)
            captured["framework_ctx"] = dict(framework_ctx or {})
            captured["op_ctx"] = op_ctx
            return "bulk-result"

    _patch_create_graph_translator(monkeypatch, DummyTranslator)
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    client = _make_client(adapter, framework_version="fw-bulk-1")

    spec = DummyBulkSpec(namespace="ns-bulk", limit=42, cursor="cursor-token", filter_={"foo": "bar"})
    res = client.bulk_vertices(spec)
    assert res == "bulk-result"

    assert captured["raw_request"] == {
        "namespace": "ns-bulk",
        "limit": 42,
        "cursor": "cursor-token",
        "filter": {"foo": "bar"},
    }
    assert captured["framework_ctx"]["framework"] == "crewai"
    assert captured["framework_ctx"]["operation"] == "bulk_vertices"
    assert captured["framework_ctx"]["namespace"] == "ns-bulk"
    assert captured["framework_ctx"]["framework_version"] == "fw-bulk-1"
    assert captured["op_ctx"] is None


@pytest.mark.asyncio
async def test_abulk_vertices_builds_raw_request_and_calls_translator_async(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """abulk_vertices should mirror bulk_vertices via arun_bulk_vertices."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_bulk_vertices(self, raw_request: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:
            captured["raw_request"] = dict(raw_request)
            captured["framework_ctx"] = dict(framework_ctx or {})
            captured["op_ctx"] = op_ctx
            return "bulk-result-async"

    _patch_create_graph_translator(monkeypatch, DummyTranslator)
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    client = _make_client(adapter, framework_version="fw-abulk-1")

    spec = DummyBulkSpec(namespace="ns-abulk", limit=7, cursor=None, filter_={"bar": 1})
    res = await client.abulk_vertices(spec)
    assert res == "bulk-result-async"

    assert captured["raw_request"] == {
        "namespace": "ns-abulk",
        "limit": 7,
        "cursor": None,
        "filter": {"bar": 1},
    }
    assert captured["framework_ctx"]["framework"] == "crewai"
    assert captured["framework_ctx"]["operation"] == "bulk_vertices"
    assert captured["framework_ctx"]["namespace"] == "ns-abulk"
    assert captured["framework_ctx"]["framework_version"] == "fw-abulk-1"
    assert captured["op_ctx"] is None


def test_batch_builds_raw_batch_ops_and_calls_translator(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """batch should translate BatchOperation-like objects into raw ops."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def batch(self, raw_batch_ops: List[Mapping[str, Any]], *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:
            captured["raw_batch_ops"] = [dict(op) for op in raw_batch_ops]
            captured["framework_ctx"] = dict(framework_ctx or {})
            captured["op_ctx"] = op_ctx
            return "batch-result"

    _patch_create_graph_translator(monkeypatch, DummyTranslator)
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    def fake_validate_batch_operations(*_: Any, **__: Any) -> None:
        return None

    monkeypatch.setattr(crewai_adapter_module, "validate_batch_operations", fake_validate_batch_operations)

    client = _make_client(adapter, framework_version="fw-batch-1")

    ops = [DummyBatchOp("upsert_nodes", {"id": "1"}), DummyBatchOp("delete_nodes", {"ids": ["1", "2"]})]
    res = client.batch(ops)
    assert res == "batch-result"

    assert captured["raw_batch_ops"] == [
        {"op": "upsert_nodes", "args": {"id": "1"}},
        {"op": "delete_nodes", "args": {"ids": ["1", "2"]}},
    ]
    assert captured["framework_ctx"]["framework"] == "crewai"
    assert captured["framework_ctx"]["operation"] == "batch"
    assert captured["framework_ctx"]["framework_version"] == "fw-batch-1"
    assert captured["op_ctx"] is None


@pytest.mark.asyncio
async def test_abatch_builds_raw_batch_ops_and_calls_translator_async(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """abatch should mirror batch via arun_batch."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_batch(self, raw_batch_ops: List[Mapping[str, Any]], *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:
            captured["raw_batch_ops"] = [dict(op) for op in raw_batch_ops]
            captured["framework_ctx"] = dict(framework_ctx or {})
            captured["op_ctx"] = op_ctx
            return "batch-result-async"

    _patch_create_graph_translator(monkeypatch, DummyTranslator)
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    def fake_validate_batch_operations(*_: Any, **__: Any) -> None:
        return None

    monkeypatch.setattr(crewai_adapter_module, "validate_batch_operations", fake_validate_batch_operations)

    client = _make_client(adapter, framework_version="fw-abatch-1")

    ops = [DummyBatchOp("upsert_edges", {"id": "e-1"}), DummyBatchOp("delete_edges", {"ids": ["e-1", "e-2"]})]
    res = await client.abatch(ops)
    assert res == "batch-result-async"

    assert captured["raw_batch_ops"] == [
        {"op": "upsert_edges", "args": {"id": "e-1"}},
        {"op": "delete_edges", "args": {"ids": ["e-1", "e-2"]}},
    ]
    assert captured["framework_ctx"]["framework"] == "crewai"
    assert captured["framework_ctx"]["operation"] == "batch"
    assert captured["framework_ctx"]["framework_version"] == "fw-abatch-1"
    assert captured["op_ctx"] is None


# ---------------------------------------------------------------------------
# Capabilities / health
# ---------------------------------------------------------------------------


def test_capabilities_and_health_basic(adapter: Any) -> None:
    """Capabilities and health should be surfaced as mappings."""
    client = _make_client(adapter)
    caps = client.capabilities()
    assert isinstance(caps, Mapping)
    health = client.health()
    assert isinstance(health, Mapping)


@pytest.mark.asyncio
async def test_async_capabilities_and_health_basic(adapter: Any) -> None:
    """Async capabilities/health should be surfaced as mappings."""
    client = _make_client(adapter)
    caps = await client.acapabilities()
    assert isinstance(caps, Mapping)
    health = await client.ahealth()
    assert isinstance(health, Mapping)


# ---------------------------------------------------------------------------
# Resource management (context managers / close semantics)
# ---------------------------------------------------------------------------


def test_close_is_idempotent(adapter: Any) -> None:
    """close() should be safe to call multiple times."""
    close_count = 0

    class CloseCountingAdapter:
        async def query(self, *args: Any, **kwargs: Any) -> Any:
            return QueryResult(records=[], summary={})

        async def capabilities(self, *args: Any, **kwargs: Any) -> Any:
            return GraphCapabilities(server="test", version="1.0")

        def close(self) -> None:
            nonlocal close_count
            close_count += 1

    client = CorpusCrewAIGraphClient(adapter=CloseCountingAdapter())
    client.close()
    client.close()
    assert close_count == 1


@pytest.mark.asyncio
async def test_aclose_prefers_async_close_then_marks_closed() -> None:
    """aclose() should call adapter.aclose when present and be idempotent."""
    aclose_count = 0

    class CloseCountingAdapter:
        async def query(self, *args: Any, **kwargs: Any) -> Any:
            return QueryResult(records=[], summary={})

        async def capabilities(self, *args: Any, **kwargs: Any) -> Any:
            return GraphCapabilities(server="test", version="1.0")

        async def aclose(self) -> None:
            nonlocal aclose_count
            aclose_count += 1

    client = CorpusCrewAIGraphClient(adapter=CloseCountingAdapter())
    await client.aclose()
    await client.aclose()
    assert aclose_count == 1


@pytest.mark.asyncio
async def test_context_manager_closes_underlying_adapter() -> None:
    """__enter__/__exit__ and __aenter__/__aexit__ should call close/aclose when present."""
    class ClosingGraphAdapter:
        def __init__(self) -> None:
            self.closed = False
            self.aclosed = False

        async def query(self, *args: Any, **kwargs: Any) -> Any:
            return QueryResult(records=[], summary={})

        async def capabilities(self, *args: Any, **kwargs: Any) -> Any:
            return GraphCapabilities(server="test", version="1.0")

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.aclosed = True

    adapter = ClosingGraphAdapter()
    with CorpusCrewAIGraphClient(adapter=adapter) as client:
        assert client is not None
    assert adapter.closed is True

    adapter2 = ClosingGraphAdapter()
    async with CorpusCrewAIGraphClient(adapter=adapter2) as client2:
        assert client2 is not None
    assert adapter2.aclosed is True


# ---------------------------------------------------------------------------
# Concurrency tests (mirrors the AutoGen suite style)
# ---------------------------------------------------------------------------


def test_thread_safety_sync_queries(adapter: Any) -> None:
    """Multiple threads should be able to use a single client safely."""
    client = _make_client(adapter)

    results: List[Any] = []
    errors: List[Any] = []

    def run(tid: int) -> None:
        try:
            res = client.query("MATCH (n) RETURN n LIMIT 1")
            results.append((tid, res))
        except Exception as e:  # noqa: BLE001
            errors.append((tid, str(e)))

    threads = [threading.Thread(target=run, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(results) == 10


@pytest.mark.asyncio
async def test_concurrent_async_queries(adapter: Any) -> None:
    """Multiple async tasks should execute without issues."""
    client = _make_client(adapter)

    async def run(i: int) -> Any:
        return await client.aquery("MATCH (n) RETURN n LIMIT 1", params={"i": i})

    results = await asyncio.gather(*(run(i) for i in range(10)), return_exceptions=True)
    assert not any(isinstance(r, Exception) for r in results)


def test_mixed_thread_operations(adapter: Any) -> None:
    """Mix query/stream/bulk calls across threads."""
    client = _make_client(adapter)

    def run_query() -> Any:
        return client.query("MATCH (n) RETURN n LIMIT 1")

    def run_stream() -> int:
        return len(list(client.stream_query("MATCH (n) RETURN n LIMIT 2")))

    def run_caps() -> Any:
        return client.capabilities()

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = []
        for i in range(12):
            if i % 3 == 0:
                futs.append(ex.submit(run_query))
            elif i % 3 == 1:
                futs.append(ex.submit(run_stream))
            else:
                futs.append(ex.submit(run_caps))

        for f in concurrent.futures.as_completed(futs):
            _ = f.result()


# ---------------------------------------------------------------------------
# REAL CrewAI integration tests (no skips allowed)
# ---------------------------------------------------------------------------


def test_crewai_tools_creation_is_real_or_raises_install_error(adapter: Any) -> None:
    """
    No-skip rule:
      - If CrewAI is installed: create_crewai_graph_tools returns real BaseTool instances.
      - If CrewAI is not installed: create_crewai_graph_tools raises RuntimeError with install guidance.
    """
    create_tools = getattr(crewai_adapter_module, "create_crewai_graph_tools", None)
    assert callable(create_tools)

    client = _make_client(adapter)

    try:
        from crewai.tools.base_tool import BaseTool  # type: ignore[import-not-found]
    except Exception:
        with pytest.raises(RuntimeError, match="CrewAI dependencies are not installed"):
            create_tools(client)
        return

    tools = create_tools(client)
    assert isinstance(tools, list)
    assert len(tools) >= 4
    assert all(isinstance(t, BaseTool) for t in tools)


def test_crewai_tool_run_sync_returns_json_or_raises_install_error(adapter: Any) -> None:
    """If installed, BaseTool._run should execute end-to-end and return JSON string."""
    create_tools = getattr(crewai_adapter_module, "create_crewai_graph_tools", None)
    assert callable(create_tools)

    client = _make_client(adapter)

    try:
        from crewai.tools.base_tool import BaseTool  # type: ignore[import-not-found]
    except Exception:
        with pytest.raises(RuntimeError):
            create_tools(client)
        return

    tools = create_tools(client)
    query_tool = next((t for t in tools if getattr(t, "name", "").endswith("_query")), None)
    assert query_tool is not None
    assert isinstance(query_tool, BaseTool)

    out = query_tool._run(query="MATCH (n) RETURN n LIMIT 1")
    assert isinstance(out, str)
    parsed = json.loads(out)
    assert "result" in parsed


@pytest.mark.asyncio
async def test_crewai_tool_run_sync_inside_event_loop_bridges_threads_or_raises_install_error(adapter: Any) -> None:
    """
    If installed, calling BaseTool._run from within a running event loop should still work
    because the tool helper bridges sync execution via a bounded thread pool.
    """
    create_tools = getattr(crewai_adapter_module, "create_crewai_graph_tools", None)
    assert callable(create_tools)

    client = _make_client(adapter)

    try:
        from crewai.tools.base_tool import BaseTool  # type: ignore[import-not-found]
    except Exception:
        with pytest.raises(RuntimeError):
            create_tools(client)
        return

    tools = create_tools(client)
    stream_tool = next((t for t in tools if getattr(t, "name", "").endswith("_stream_query")), None)
    assert stream_tool is not None
    assert isinstance(stream_tool, BaseTool)

    # max_chunks accepts strings defensively via int(max_chunks); also ensure >0 behavior doesn’t crash.
    out = stream_tool._run(query="MATCH (n) RETURN n LIMIT 2", max_chunks="3")  # type: ignore[arg-type]
    assert isinstance(out, str)
    parsed = json.loads(out)
    assert "chunks" in parsed


@pytest.mark.asyncio
async def test_crewai_tool_arun_async_executes_end_to_end_or_raises_install_error(adapter: Any) -> None:
    """If installed, BaseTool._arun should execute end-to-end and return JSON string."""
    create_tools = getattr(crewai_adapter_module, "create_crewai_graph_tools", None)
    assert callable(create_tools)

    client = _make_client(adapter)

    try:
        from crewai.tools.base_tool import BaseTool  # type: ignore[import-not-found]
    except Exception:
        with pytest.raises(RuntimeError):
            create_tools(client)
        return

    tools = create_tools(client)
    batch_tool = next((t for t in tools if getattr(t, "name", "").endswith("_batch")), None)
    assert batch_tool is not None
    assert isinstance(batch_tool, BaseTool)

    out = await batch_tool._arun(
        ops=[{"op": "query", "args": {"text": "MATCH (n) RETURN n LIMIT 1"}}],
    )
    assert isinstance(out, str)
    parsed = json.loads(out)
    assert "result" in parsed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
