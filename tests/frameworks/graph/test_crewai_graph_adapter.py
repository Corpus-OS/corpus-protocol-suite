# tests/frameworks/graph/test_crewai_graph_adapter.py

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List

import inspect

import pytest

import corpus_sdk.graph.framework_adapters.crewai as crewai_adapter_module
from corpus_sdk.graph.framework_adapters.crewai import (
    CorpusCrewAIGraphClient,
    CrewAIGraphFrameworkTranslator,
    ErrorCodes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(graph_adapter: Any, **kwargs: Any) -> CorpusCrewAIGraphClient:
    """Construct a CorpusCrewAIGraphClient instance from the generic adapter."""
    return CorpusCrewAIGraphClient(graph_adapter=graph_adapter, **kwargs)


def _make_dummy_task(**attrs: Any) -> Any:
    """Create a simple CrewAI-like task object with arbitrary attributes."""

    class DummyTask:
        def __init__(self, **kw: Any) -> None:
            self.__dict__.update(kw)

    return DummyTask(**attrs)


# ---------------------------------------------------------------------------
# Constructor / translator behavior
# ---------------------------------------------------------------------------


def test_default_translator_uses_crewai_framework_translator(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    By default, CorpusCrewAIGraphClient should:

    - Construct a CrewAIGraphFrameworkTranslator instance, and
    - Pass it into create_graph_translator with framework="crewai".
    """
    captured: Dict[str, Any] = {}

    def fake_create_graph_translator(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["args"] = args
        captured["kwargs"] = kwargs

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(
        crewai_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(graph_adapter)

    # Trigger lazy translator construction
    _ = client._translator  # noqa: SLF001

    assert "kwargs" in captured
    kwargs = captured["kwargs"]
    assert kwargs.get("framework") == "crewai"
    translator = kwargs.get("translator")
    assert isinstance(translator, CrewAIGraphFrameworkTranslator)


def test_framework_translator_override_is_respected(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    If framework_translator is provided, CorpusCrewAIGraphClient should pass
    it through to create_graph_translator instead of constructing its own
    CrewAIGraphFrameworkTranslator.
    """
    captured: Dict[str, Any] = {}

    class CustomTranslator(CrewAIGraphFrameworkTranslator):
        pass

    custom = CustomTranslator()

    def fake_create_graph_translator(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["args"] = args
        captured["kwargs"] = kwargs

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(
        crewai_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(
        graph_adapter,
        framework_translator=custom,
        framework_version="crewai-fw-1.2.3",
    )

    _ = client._translator  # noqa: SLF001

    kwargs = captured["kwargs"]
    assert kwargs.get("framework") == "crewai"
    assert kwargs.get("translator") is custom


# ---------------------------------------------------------------------------
# Context translation / from_crewai mapping
# ---------------------------------------------------------------------------


def test_crewai_task_and_extra_context_passed_to_core_ctx(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    Verify that task and extra_context are passed through to core_ctx_from_crewai
    with the configured framework_version.
    """
    captured: Dict[str, Any] = {}

    # Patch OperationContext so our fake ctx passes isinstance() check.
    class DummyOperationContext:
        def __init__(self, **kwargs: Any) -> None:
            self.attrs = kwargs

    monkeypatch.setattr(crewai_adapter_module, "OperationContext", DummyOperationContext)

    def fake_core_ctx_from_crewai(
        task: Any,
        *,
        framework_version: Any = None,
        **extra: Any,
    ) -> Any:
        captured["task"] = task
        captured["framework_version"] = framework_version
        captured["extra"] = extra
        return DummyOperationContext()

    monkeypatch.setattr(
        crewai_adapter_module,
        "core_ctx_from_crewai",
        fake_core_ctx_from_crewai,
    )

    client = _make_client(
        graph_adapter,
        framework_version="crewai-test-version",
    )

    task = _make_dummy_task(task_id="task-123")
    extra_ctx = {
        "request_id": "req-xyz",
        "tenant": "tenant-1",
    }

    result = client.query(
        "MATCH (n) RETURN n LIMIT 1",
        task=task,
        extra_context=extra_ctx,
    )
    assert result is not None

    assert captured.get("task") is task
    assert captured.get("framework_version") == "crewai-test-version"
    assert captured.get("extra") == extra_ctx


def test_build_ctx_returns_none_when_no_task_and_no_extra(
    graph_adapter: Any,
) -> None:
    """
    _build_ctx should return None when both task and extra_context are absent.
    """
    client = _make_client(graph_adapter)
    ctx = client._build_ctx(task=None, extra_context=None)  # noqa: SLF001
    assert ctx is None


# ---------------------------------------------------------------------------
# Context translation failure path
# ---------------------------------------------------------------------------


def test_build_ctx_failure_raises_bad_request_like_error_and_attaches_context(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    _build_ctx should wrap failures from core_ctx_from_crewai in an error
    that behaves like BadRequest (has .code == ErrorCodes.BAD_OPERATION_CONTEXT)
    and call attach_context.

    The test deliberately does *not* import BadRequest directly.
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured_ctx.update(ctx)

    def fake_core_ctx_from_crewai(
        task: Any,  # noqa: ARG001
        *,
        framework_version: Any = None,  # noqa: ARG001
        **extra: Any,  # noqa: ARG001
    ) -> Any:
        raise RuntimeError("boom from ctx builder")

    monkeypatch.setattr(crewai_adapter_module, "attach_context", fake_attach_context)
    monkeypatch.setattr(
        crewai_adapter_module,
        "core_ctx_from_crewai",
        fake_core_ctx_from_crewai,
    )

    client = _make_client(graph_adapter, framework_version="ctx-fw")

    with pytest.raises(Exception) as exc_info:
        client._build_ctx(  # noqa: SLF001
            task=_make_dummy_task(),
            extra_context={"foo": "bar"},
        )

    err = exc_info.value
    # We don't import BadRequest, but we still assert semantics.
    assert type(err).__name__ == "BadRequest"
    assert getattr(err, "code", None) == ErrorCodes.BAD_OPERATION_CONTEXT
    assert captured_ctx.get("framework") == "crewai"
    assert captured_ctx.get("operation") == "context_translation"


def test_build_ctx_rejects_non_operation_context_type(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    If from_crewai returns something that is not an OperationContext, the
    client should raise a BadRequest-like error with the same BAD_OPERATION_CONTEXT
    code.
    """

    class NotAnOperationContext:
        pass

    def fake_core_ctx_from_crewai(
        task: Any,  # noqa: ARG001
        *,
        framework_version: Any = None,  # noqa: ARG001
        **extra: Any,  # noqa: ARG001
    ) -> Any:
        return NotAnOperationContext()

    monkeypatch.setattr(
        crewai_adapter_module,
        "core_ctx_from_crewai",
        fake_core_ctx_from_crewai,
    )

    client = _make_client(graph_adapter)

    with pytest.raises(Exception) as exc_info:
        client._build_ctx(  # noqa: SLF001
            task=_make_dummy_task(),
            extra_context={},
        )

    err = exc_info.value
    assert type(err).__name__ == "BadRequest"
    assert getattr(err, "code", None) == ErrorCodes.BAD_OPERATION_CONTEXT


# ---------------------------------------------------------------------------
# Error-context decorator behavior
# ---------------------------------------------------------------------------


def test_error_context_includes_crewai_metadata_sync(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    When an error occurs during a sync graph operation, error context should
    include CrewAI-specific metadata via attach_context().
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured_context.update(ctx)

    monkeypatch.setattr(
        crewai_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingTranslator:
        def query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from crewai graph adapter")

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return FailingTranslator()

    monkeypatch.setattr(
        crewai_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(graph_adapter)

    with pytest.raises(RuntimeError, match="test error from crewai graph adapter"):
        client.query("MATCH (n) RETURN n")

    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "crewai"
    # We don't over-specify the operation name; just require it to exist.
    assert "operation" in captured_context


@pytest.mark.asyncio
async def test_error_context_includes_crewai_metadata_async(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    Same as the sync error-context test but for the async query path.
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured_context.update(ctx)

    monkeypatch.setattr(
        crewai_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingTranslator:
        async def arun_query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from crewai graph adapter")

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return FailingTranslator()

    monkeypatch.setattr(
        crewai_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(graph_adapter)

    with pytest.raises(RuntimeError, match="test error from crewai graph adapter"):
        await client.aquery("MATCH (n) RETURN n")

    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "crewai"
    assert "operation" in captured_context


# ---------------------------------------------------------------------------
# Sync semantics (basic smoke tests)
# ---------------------------------------------------------------------------


def test_sync_query_and_stream_basic(graph_adapter: Any) -> None:
    """
    Basic smoke test for sync query / stream_query behavior: methods should
    accept text input and not crash, returning protocol-level shapes.

    Detailed QueryResult / QueryChunk semantics are covered by the generic
    graph contract tests.
    """
    client = _make_client(graph_adapter, default_namespace="crewai-ns")

    # Non-streaming query
    result = client.query("MATCH (n) RETURN n LIMIT 1")
    assert result is not None

    # Streaming query
    chunks = list(client.stream_query("MATCH (n) RETURN n LIMIT 2"))
    assert isinstance(chunks, list)


def test_sync_query_accepts_optional_params_and_context(graph_adapter: Any) -> None:
    """
    query() should accept params, dialect, namespace, timeout_ms, and
    task/extra_context kwargs without raising.
    """
    client = _make_client(graph_adapter, default_dialect="cypher")

    result = client.query(
        "MATCH (n) RETURN n LIMIT $limit",
        params={"limit": 5},
        dialect="cypher",
        namespace="ctx-ns",
        timeout_ms=5000,
        task=_make_dummy_task(task_id="task-sync"),
        extra_context={"request_id": "req-sync"},
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Async semantics (basic smoke tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_query_and_stream_basic(graph_adapter: Any) -> None:
    """
    Async aquery / astream_query should exist and produce results compatible
    with the sync API (non-None result / async-iterable of chunks).
    """
    client = _make_client(graph_adapter)

    assert hasattr(client, "aquery")
    assert hasattr(client, "astream_query")

    coro = client.aquery("MATCH (n) RETURN n LIMIT 1")
    assert inspect.isawaitable(coro)
    result = await coro
    assert result is not None

    aiter = client.astream_query("MATCH (n) RETURN n LIMIT 2")
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    seen_any = False
    async for _ in aiter:  # noqa: B007
        seen_any = True
        break

    assert isinstance(seen_any, bool)


@pytest.mark.asyncio
async def test_async_query_accepts_optional_params_and_context(
    graph_adapter: Any,
) -> None:
    """
    aquery() should accept the same optional params and context as query().
    """
    client = _make_client(graph_adapter, default_namespace="async-ns")

    result = await client.aquery(
        "MATCH (n) RETURN n LIMIT $limit",
        params={"limit": 3},
        dialect="cypher",
        namespace="async-ns",
        timeout_ms=2500,
        task=_make_dummy_task(task_id="task-async"),
        extra_context={"request_id": "req-async"},
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Bulk vertices / batch semantics (CrewAI wiring)
# ---------------------------------------------------------------------------


def test_bulk_vertices_builds_raw_request_and_calls_translator(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    bulk_vertices() should:

    - Build the correct raw_request mapping from the spec, and
    - Call the underlying translator.bulk_vertices with that mapping and
      appropriate framework_ctx (namespace, framework, operation).
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def bulk_vertices(
            self,
            raw_request: Mapping[str, Any],
            *,
            op_ctx: Any = None,
            framework_ctx: Mapping[str, Any] | None = None,
        ) -> Any:
            captured["raw_request"] = dict(raw_request)
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx or {})
            return "bulk-result"

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return DummyTranslator()

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    monkeypatch.setattr(
        crewai_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        crewai_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )

    client = _make_client(graph_adapter, framework_version="fw-bulk-1")

    class DummyBulkSpec:
        def __init__(self) -> None:
            self.namespace = "ns-bulk"
            self.limit = 42
            self.cursor = "cursor-token"
            self.filter = {"foo": "bar"}

    spec = DummyBulkSpec()

    result = client.bulk_vertices(spec)
    assert result == "bulk-result"

    raw = captured["raw_request"]
    assert raw == {
        "namespace": "ns-bulk",
        "limit": 42,
        "cursor": "cursor-token",
        "filter": {"foo": "bar"},
    }

    fw_ctx = captured["framework_ctx"]
    assert fw_ctx.get("framework") == "crewai"
    assert fw_ctx.get("operation") == "bulk_vertices"
    assert fw_ctx.get("namespace") == "ns-bulk"
    assert fw_ctx.get("framework_version") == "fw-bulk-1"
    assert captured["op_ctx"] is None


@pytest.mark.asyncio
async def test_abulk_vertices_builds_raw_request_and_calls_translator_async(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    abulk_vertices() should mirror bulk_vertices wiring but via the async
    translator.arun_bulk_vertices surface.
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_bulk_vertices(
            self,
            raw_request: Mapping[str, Any],
            *,
            op_ctx: Any = None,
            framework_ctx: Mapping[str, Any] | None = None,
        ) -> Any:
            captured["raw_request"] = dict(raw_request)
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx or {})
            return "bulk-result-async"

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return DummyTranslator()

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    monkeypatch.setattr(
        crewai_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        crewai_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )

    client = _make_client(graph_adapter, framework_version="fw-abulk-1")

    class DummyBulkSpec:
        def __init__(self) -> None:
            self.namespace = "ns-abulk"
            self.limit = 7
            self.cursor = None
            self.filter = {"bar": 1}

    spec = DummyBulkSpec()

    result = await client.abulk_vertices(spec)
    assert result == "bulk-result-async"

    raw = captured["raw_request"]
    assert raw == {
        "namespace": "ns-abulk",
        "limit": 7,
        "cursor": None,
        "filter": {"bar": 1},
    }

    fw_ctx = captured["framework_ctx"]
    assert fw_ctx.get("framework") == "crewai"
    assert fw_ctx.get("operation") == "bulk_vertices"
    assert fw_ctx.get("namespace") == "ns-abulk"
    assert fw_ctx.get("framework_version") == "fw-abulk-1"
    assert captured["op_ctx"] is None


def test_batch_builds_raw_batch_ops_and_calls_translator(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    batch() should:

    - Validate batch operations (we stub validation here), and
    - Translate BatchOperation-like objects into raw_batch_ops mappings
      passed to translator.batch().
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def batch(
            self,
            raw_batch_ops: List[Mapping[str, Any]],
            *,
            op_ctx: Any = None,
            framework_ctx: Mapping[str, Any] | None = None,
        ) -> Any:
            captured["raw_batch_ops"] = [dict(op) for op in raw_batch_ops]
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx or {})
            return "batch-result"

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return DummyTranslator()

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    def fake_validate_batch_operations(*_: Any, **__: Any) -> None:
        return None

    monkeypatch.setattr(
        crewai_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        crewai_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )
    monkeypatch.setattr(
        crewai_adapter_module,
        "validate_batch_operations",
        fake_validate_batch_operations,
    )

    client = _make_client(graph_adapter, framework_version="fw-batch-1")

    class DummyBatchOp:
        def __init__(self, op: str, args: Mapping[str, Any]) -> None:
            self.op = op
            self.args = dict(args)

    ops = [
        DummyBatchOp("upsert_nodes", {"id": "1"}),
        DummyBatchOp("delete_nodes", {"ids": ["1", "2"]}),
    ]

    result = client.batch(ops)
    assert result == "batch-result"

    raw_ops = captured["raw_batch_ops"]
    assert raw_ops == [
        {"op": "upsert_nodes", "args": {"id": "1"}},
        {"op": "delete_nodes", "args": {"ids": ["1", "2"]}},
    ]

    fw_ctx = captured["framework_ctx"]
    assert fw_ctx.get("framework") == "crewai"
    assert fw_ctx.get("operation") == "batch"
    assert fw_ctx.get("framework_version") == "fw-batch-1"
    assert captured["op_ctx"] is None


@pytest.mark.asyncio
async def test_abatch_builds_raw_batch_ops_and_calls_translator_async(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    abatch() should mirror batch wiring but via translator.arun_batch().
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_batch(
            self,
            raw_batch_ops: List[Mapping[str, Any]],
            *,
            op_ctx: Any = None,
            framework_ctx: Mapping[str, Any] | None = None,
        ) -> Any:
            captured["raw_batch_ops"] = [dict(op) for op in raw_batch_ops]
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx or {})
            return "batch-result-async"

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return DummyTranslator()

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    def fake_validate_batch_operations(*_: Any, **__: Any) -> None:
        return None

    monkeypatch.setattr(
        crewai_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        crewai_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )
    monkeypatch.setattr(
        crewai_adapter_module,
        "validate_batch_operations",
        fake_validate_batch_operations,
    )

    client = _make_client(graph_adapter, framework_version="fw-abatch-1")

    class DummyBatchOp:
        def __init__(self, op: str, args: Mapping[str, Any]) -> None:
            self.op = op
            self.args = dict(args)

    ops = [
        DummyBatchOp("upsert_edges", {"id": "e-1"}),
        DummyBatchOp("delete_edges", {"ids": ["e-1", "e-2"]}),
    ]

    result = await client.abatch(ops)
    assert result == "batch-result-async"

    raw_ops = captured["raw_batch_ops"]
    assert raw_ops == [
        {"op": "upsert_edges", "args": {"id": "e-1"}},
        {"op": "delete_edges", "args": {"ids": ["e-1", "e-2"]}},
    ]

    fw_ctx = captured["framework_ctx"]
    assert fw_ctx.get("framework") == "crewai"
    assert fw_ctx.get("operation") == "batch"
    assert fw_ctx.get("framework_version") == "fw-abatch-1"
    assert captured["op_ctx"] is None


# ---------------------------------------------------------------------------
# Upsert / delete wiring (CrewAI-specific)
# ---------------------------------------------------------------------------


def test_upsert_nodes_uses_raw_nodes_and_framework_ctx(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    upsert_nodes() should:

    - Validate spec (we stub validation),
    - Pass spec.nodes into translator.upsert_nodes,
    - Build framework_ctx with framework='crewai', operation='upsert_nodes',
      and the spec namespace.
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def upsert_nodes(
            self,
            raw_nodes: List[Mapping[str, Any]],
            *,
            op_ctx: Any = None,
            framework_ctx: Mapping[str, Any] | None = None,
        ) -> Any:
            captured["raw_nodes"] = list(raw_nodes)
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx or {})
            return "upsert-nodes-result"

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return DummyTranslator()

    def fake_validate_upsert_nodes_spec(*_: Any, **__: Any) -> None:
        return None

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    monkeypatch.setattr(
        crewai_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        crewai_adapter_module,
        "validate_upsert_nodes_spec",
        fake_validate_upsert_nodes_spec,
    )
    monkeypatch.setattr(
        crewai_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )

    client = _make_client(graph_adapter, framework_version="fw-upsert-nodes")

    class DummyNodesSpec:
        def __init__(self) -> None:
            self.namespace = "ns-upsert"
            self.nodes = [{"id": "1"}, {"id": "2"}]

    spec = DummyNodesSpec()

    result = client.upsert_nodes(spec)
    assert result == "upsert-nodes-result"

    assert captured["raw_nodes"] == [{"id": "1"}, {"id": "2"}]
    fw_ctx = captured["framework_ctx"]
    assert fw_ctx.get("framework") == "crewai"
    assert fw_ctx.get("operation") == "upsert_nodes"
    assert fw_ctx.get("namespace") == "ns-upsert"
    assert fw_ctx.get("framework_version") == "fw-upsert-nodes"
    assert captured["op_ctx"] is None


@pytest.mark.asyncio
async def test_aupsert_nodes_uses_raw_nodes_and_framework_ctx_async(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    aupsert_nodes() should mirror upsert_nodes wiring via arun_upsert_nodes().
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_upsert_nodes(
            self,
            raw_nodes: List[Mapping[str, Any]],
            *,
            op_ctx: Any = None,
            framework_ctx: Mapping[str, Any] | None = None,
        ) -> Any:
            captured["raw_nodes"] = list(raw_nodes)
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx or {})
            return "aupsert-nodes-result"

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return DummyTranslator()

    def fake_validate_upsert_nodes_spec(*_: Any, **__: Any) -> None:
        return None

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    monkeypatch.setattr(
        crewai_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        crewai_adapter_module,
        "validate_upsert_nodes_spec",
        fake_validate_upsert_nodes_spec,
    )
    monkeypatch.setattr(
        crewai_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )

    client = _make_client(graph_adapter, framework_version="fw-aupsert-nodes")

    class DummyNodesSpec:
        def __init__(self) -> None:
            self.namespace = "ns-aupsert"
            self.nodes = [{"id": "3"}]

    spec = DummyNodesSpec()

    result = await client.aupsert_nodes(spec)
    assert result == "aupsert-nodes-result"

    assert captured["raw_nodes"] == [{"id": "3"}]
    fw_ctx = captured["framework_ctx"]
    assert fw_ctx.get("framework") == "crewai"
    assert fw_ctx.get("operation") == "upsert_nodes"
    assert fw_ctx.get("namespace") == "ns-aupsert"
    assert fw_ctx.get("framework_version") == "fw-aupsert-nodes"
    assert captured["op_ctx"] is None


def test_upsert_edges_uses_raw_edges_and_framework_ctx(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    upsert_edges() should:

    - Call _validate_upsert_edges_spec,
    - Pass spec.edges into translator.upsert_edges,
    - Build framework_ctx with operation='upsert_edges' and namespace.
    """
    captured: Dict[str, Any] = {}
    validated_spec: Dict[str, Any] = {}

    class DummyTranslator:
        def upsert_edges(
            self,
            raw_edges: List[Mapping[str, Any]],
            *,
            op_ctx: Any = None,
            framework_ctx: Mapping[str, Any] | None = None,
        ) -> Any:
            captured["raw_edges"] = list(raw_edges)
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx or {})
            return "upsert-edges-result"

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return DummyTranslator()

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    def fake_validate_edges(self: Any, spec: Any) -> None:  # noqa: ARG001
        validated_spec["spec"] = spec

    monkeypatch.setattr(
        crewai_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        crewai_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )
    monkeypatch.setattr(
        crewai_adapter_module.CorpusCrewAIGraphClient,
        "_validate_upsert_edges_spec",
        fake_validate_edges,
    )

    client = _make_client(graph_adapter, framework_version="fw-upsert-edges")

    class DummyEdgesSpec:
        def __init__(self) -> None:
            self.namespace = "ns-edges"
            self.edges = [{"id": "e1"}, {"id": "e2"}]

    spec = DummyEdgesSpec()

    result = client.upsert_edges(spec)
    assert result == "upsert-edges-result"
    assert validated_spec.get("spec") is spec

    assert captured["raw_edges"] == [{"id": "e1"}, {"id": "e2"}]
    fw_ctx = captured["framework_ctx"]
    assert fw_ctx.get("framework") == "crewai"
    assert fw_ctx.get("operation") == "upsert_edges"
    assert fw_ctx.get("namespace") == "ns-edges"
    assert fw_ctx.get("framework_version") == "fw-upsert-edges"
    assert captured["op_ctx"] is None


@pytest.mark.asyncio
async def test_aupsert_edges_uses_raw_edges_and_framework_ctx_async(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    aupsert_edges() should mirror upsert_edges wiring via arun_upsert_edges().
    """
    captured: Dict[str, Any] = {}
    validated_spec: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_upsert_edges(
            self,
            raw_edges: List[Mapping[str, Any]],
            *,
            op_ctx: Any = None,
            framework_ctx: Mapping[str, Any] | None = None,
        ) -> Any:
            captured["raw_edges"] = list(raw_edges)
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx or {})
            return "aupsert-edges-result"

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return DummyTranslator()

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    def fake_validate_edges(self: Any, spec: Any) -> None:  # noqa: ARG001
        validated_spec["spec"] = spec

    monkeypatch.setattr(
        crewai_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        crewai_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )
    monkeypatch.setattr(
        crewai_adapter_module.CorpusCrewAIGraphClient,
        "_validate_upsert_edges_spec",
        fake_validate_edges,
    )

    client = _make_client(graph_adapter, framework_version="fw-aupsert-edges")

    class DummyEdgesSpec:
        def __init__(self) -> None:
            self.namespace = "ns-aupsert-edges"
            self.edges = [{"id": "e3"}]

    spec = DummyEdgesSpec()

    result = await client.aupsert_edges(spec)
    assert result == "aupsert-edges-result"
    assert validated_spec.get("spec") is spec

    assert captured["raw_edges"] == [{"id": "e3"}]
    fw_ctx = captured["framework_ctx"]
    assert fw_ctx.get("framework") == "crewai"
    assert fw_ctx.get("operation") == "upsert_edges"
    assert fw_ctx.get("namespace") == "ns-aupsert-edges"
    assert fw_ctx.get("framework_version") == "fw-aupsert-edges"
    assert captured["op_ctx"] is None


def test_upsert_edges_validation_raises_bad_request_like_error(
    graph_adapter: Any,
) -> None:
    """
    The CrewAI adapter's _validate_upsert_edges_spec should raise a
    BadRequest-like error when edges list is missing/invalid.

    We don't import BadRequest; we assert on type name and error message/code.
    """
    client = _make_client(graph_adapter)

    class BadEdgesSpec:
        def __init__(self) -> None:
            self.namespace = "ns-bad"
            self.edges = None

    spec = BadEdgesSpec()

    with pytest.raises(Exception) as exc_info:
        client._validate_upsert_edges_spec(spec)  # noqa: SLF001

    err = exc_info.value
    assert type(err).__name__ == "BadRequest"
    assert "edges" in str(err)


def test_delete_nodes_uses_filter_or_ids_and_framework_ctx(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    delete_nodes() should:

    - Use filter when provided, else ids list,
    - Pass the selected value into translator.delete_nodes,
    - Build framework_ctx with operation='delete_nodes' and namespace.
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def delete_nodes(
            self,
            raw_filter_or_ids: Any,
            *,
            op_ctx: Any = None,
            framework_ctx: Mapping[str, Any] | None = None,
        ) -> Any:
            captured["arg"] = raw_filter_or_ids
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx or {})
            return "delete-nodes-result"

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return DummyTranslator()

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    monkeypatch.setattr(
        crewai_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        crewai_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )

    client = _make_client(graph_adapter, framework_version="fw-del-nodes")

    class DummyDeleteSpec:
        def __init__(self) -> None:
            self.namespace = "ns-del"
            self.filter = None
            self.ids = ["1", "2"]

    spec = DummyDeleteSpec()

    result = client.delete_nodes(spec)
    assert result == "delete-nodes-result"

    assert captured["arg"] == ["1", "2"]
    fw_ctx = captured["framework_ctx"]
    assert fw_ctx.get("framework") == "crewai"
    assert fw_ctx.get("operation") == "delete_nodes"
    assert fw_ctx.get("namespace") == "ns-del"
    assert fw_ctx.get("framework_version") == "fw-del-nodes"
    assert captured["op_ctx"] is None


def test_delete_edges_uses_filter_or_ids_and_framework_ctx(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    delete_edges() should:

    - Use filter when provided, else ids list,
    - Pass the selected value into translator.delete_edges,
    - Build framework_ctx with operation='delete_edges' and namespace.
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def delete_edges(
            self,
            raw_filter_or_ids: Any,
            *,
            op_ctx: Any = None,
            framework_ctx: Mapping[str, Any] | None = None,
        ) -> Any:
            captured["arg"] = raw_filter_or_ids
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx or {})
            return "delete-edges-result"

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return DummyTranslator()

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    monkeypatch.setattr(
        crewai_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        crewai_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )

    client = _make_client(graph_adapter, framework_version="fw-del-edges")

    class DummyDeleteSpec:
        def __init__(self) -> None:
            self.namespace = "ns-del-edges"
            self.filter = {"foo": "bar"}
            self.ids = None

    spec = DummyDeleteSpec()

    result = client.delete_edges(spec)
    assert result == "delete-edges-result"

    assert captured["arg"] == {"foo": "bar"}
    fw_ctx = captured["framework_ctx"]
    assert fw_ctx.get("framework") == "crewai"
    assert fw_ctx.get("operation") == "delete_edges"
    assert fw_ctx.get("namespace") == "ns-del-edges"
    assert fw_ctx.get("framework_version") == "fw-del-edges"
    assert captured["op_ctx"] is None


@pytest.mark.asyncio
async def test_adelete_nodes_and_edges_use_framework_ctx_async(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    adelete_nodes()/adelete_edges() should mirror delete_* wiring via
    arun_delete_nodes/arun_delete_edges.
    """
    captured_nodes: Dict[str, Any] = {}
    captured_edges: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_delete_nodes(
            self,
            raw_filter_or_ids: Any,
            *,
            op_ctx: Any = None,
            framework_ctx: Mapping[str, Any] | None = None,
        ) -> Any:
            captured_nodes["arg"] = raw_filter_or_ids
            captured_nodes["op_ctx"] = op_ctx
            captured_nodes["framework_ctx"] = dict(framework_ctx or {})
            return "adelete-nodes-result"

        async def arun_delete_edges(
            self,
            raw_filter_or_ids: Any,
            *,
            op_ctx: Any = None,
            framework_ctx: Mapping[str, Any] | None = None,
        ) -> Any:
            captured_edges["arg"] = raw_filter_or_ids
            captured_edges["op_ctx"] = op_ctx
            captured_edges["framework_ctx"] = dict(framework_ctx or {})
            return "adelete-edges-result"

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return DummyTranslator()

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    monkeypatch.setattr(
        crewai_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        crewai_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )

    client = _make_client(graph_adapter, framework_version="fw-adelete")

    class DummyDeleteNodesSpec:
        def __init__(self) -> None:
            self.namespace = "ns-adelete-nodes"
            self.filter = None
            self.ids = ["n1"]

    class DummyDeleteEdgesSpec:
        def __init__(self) -> None:
            self.namespace = "ns-adelete-edges"
            self.filter = {"rel": "knows"}
            self.ids = None

    spec_nodes = DummyDeleteNodesSpec()
    spec_edges = DummyDeleteEdgesSpec()

    res_nodes = await client.adelete_nodes(spec_nodes)
    res_edges = await client.adelete_edges(spec_edges)

    assert res_nodes == "adelete-nodes-result"
    assert res_edges == "adelete-edges-result"

    assert captured_nodes["arg"] == ["n1"]
    fw_ctx_nodes = captured_nodes["framework_ctx"]
    assert fw_ctx_nodes.get("framework") == "crewai"
    assert fw_ctx_nodes.get("operation") == "delete_nodes"
    assert fw_ctx_nodes.get("namespace") == "ns-adelete-nodes"
    assert fw_ctx_nodes.get("framework_version") == "fw-adelete"

    assert captured_edges["arg"] == {"rel": "knows"}
    fw_ctx_edges = captured_edges["framework_ctx"]
    assert fw_ctx_edges.get("framework") == "crewai"
    assert fw_ctx_edges.get("operation") == "delete_edges"
    assert fw_ctx_edges.get("namespace") == "ns-adelete-edges"
    assert fw_ctx_edges.get("framework_version") == "fw-adelete"


# ---------------------------------------------------------------------------
# Capabilities / health passthrough (basic + framework_ctx)
# ---------------------------------------------------------------------------


def test_capabilities_and_health_basic(graph_adapter: Any) -> None:
    """
    Capabilities and health should be surfaced as mappings.

    The detailed structure is tested in framework-agnostic graph contract
    tests; here we only assert that the CrewAI adapter normalizes to
    mapping-like results.
    """
    client = _make_client(graph_adapter)

    caps = client.capabilities()
    assert isinstance(caps, Mapping)

    health = client.health()
    assert isinstance(health, Mapping)


@pytest.mark.asyncio
async def test_async_capabilities_and_health_basic(graph_adapter: Any) -> None:
    """
    Async capabilities/health should also return mappings compatible with
    the sync variants.
    """
    client = _make_client(graph_adapter)

    acaps = await client.acapabilities()
    assert isinstance(acaps, Mapping)

    ahealth = await client.ahealth()
    assert isinstance(ahealth, Mapping)


def test_health_passes_framework_ctx_to_translator(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    health() should call translator.health with a framework_ctx that includes:
    - framework="crewai"
    - operation="health"
    - framework_version
    - namespace (from default_namespace)
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def health(
            self,
            *,
            op_ctx: Any = None,
            framework_ctx: Mapping[str, Any] | None = None,
        ) -> Mapping[str, Any]:
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx or {})
            return {"status": "ok"}

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return DummyTranslator()

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    # Patch OperationContext/core_ctx_from_crewai so _build_ctx works.
    class DummyOperationContext:
        pass

    def fake_core_ctx_from_crewai(
        task: Any,  # noqa: ARG001
        *,
        framework_version: Any = None,  # noqa: ARG001
        **extra: Any,  # noqa: ARG001
    ) -> Any:
        return DummyOperationContext()

    monkeypatch.setattr(
        crewai_adapter_module,
        "OperationContext",
        DummyOperationContext,
    )
    monkeypatch.setattr(
        crewai_adapter_module,
        "core_ctx_from_crewai",
        fake_core_ctx_from_crewai,
    )
    monkeypatch.setattr(
        crewai_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        crewai_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )

    client = _make_client(
        graph_adapter,
        framework_version="fw-health",
        default_namespace="ns-health",
    )

    result = client.health(task=_make_dummy_task())
    assert result == {"status": "ok"}

    fw_ctx = captured["framework_ctx"]
    assert fw_ctx.get("framework") == "crewai"
    assert fw_ctx.get("operation") == "health"
    assert fw_ctx.get("framework_version") == "fw-health"
    assert fw_ctx.get("namespace") == "ns-health"
    # We built a context (DummyOperationContext), so op_ctx should not be None.
    assert isinstance(captured["op_ctx"], DummyOperationContext)


# ---------------------------------------------------------------------------
# Resource management (context managers)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_closes_underlying_graph_adapter() -> None:
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

    # Sync context manager
    with CorpusCrewAIGraphClient(graph_adapter=adapter) as client:
        assert client is not None

    assert adapter.closed is True

    # Async context manager
    adapter2 = ClosingGraphAdapter()
    client2 = CorpusCrewAIGraphClient(graph_adapter=adapter2)

    async with client2:
        assert client2 is not None

    assert adapter2.aclosed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

