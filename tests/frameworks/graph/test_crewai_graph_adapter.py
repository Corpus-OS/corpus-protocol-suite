# tests/frameworks/graph/test_crewai_graph_adapter.py

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Dict, List, Callable, Type

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
        crewai_adapter_module,
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
        crewai_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )


class DummyBulkSpec:
    """Simple stand-in for BulkVerticesSpec used in wiring tests."""

    def __init__(
        self,
        namespace: str,
        limit: int,
        cursor: Any,
        filter_: Any,
    ) -> None:
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

    monkeypatch.setattr(
        crewai_adapter_module,
        "OperationContext",
        DummyOperationContext,
    )

    def fake_core_ctx_from_crewai(
        task: Any,  # noqa: ARG001
        *,
        framework_version: Any = None,
        **extra: Any,
    ) -> Any:
        captured["framework_version"] = framework_version
        captured["extra"] = extra
        return DummyOperationContext(task=task, **extra)

    monkeypatch.setattr(
        crewai_adapter_module,
        "core_ctx_from_crewai",
        fake_core_ctx_from_crewai,
    )

    client = _make_client(
        graph_adapter,
        framework_version="crewai-test-version",
    )

    fake_task = object()
    extra_ctx = {
        "request_id": "req-crewai-xyz",
        "tenant": "tenant-1",
    }

    result = client.query(
        "MATCH (n) RETURN n LIMIT 1",
        task=fake_task,
        extra_context=extra_ctx,
    )
    assert result is not None

    assert captured.get("framework_version") == "crewai-test-version"
    assert captured.get("extra") == extra_ctx


# ---------------------------------------------------------------------------
# Context translation failure path
# ---------------------------------------------------------------------------


def test_build_ctx_failure_raises_bad_request_like_error_and_attaches_context(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    _build_ctx should wrap failures from core_ctx_from_crewai in an error that:

    - Has code ErrorCodes.BAD_OPERATION_CONTEXT, and
    - Includes a helpful message
    - Causes attach_context to be called with framework='crewai' and a
      context_translation operation tag.

    We do *not* depend on the concrete BadRequest type.
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    def fake_core_ctx_from_crewai(
        task: Any,  # noqa: ARG001
        *,
        framework_version: Any = None,  # noqa: ARG001
        **extra: Any,  # noqa: ARG001
    ) -> Any:
        raise RuntimeError("boom from ctx builder")

    monkeypatch.setattr(
        crewai_adapter_module,
        "attach_context",
        fake_attach_context,
    )
    monkeypatch.setattr(
        crewai_adapter_module,
        "core_ctx_from_crewai",
        fake_core_ctx_from_crewai,
    )

    client = _make_client(graph_adapter, framework_version="ctx-fw")

    with pytest.raises(Exception) as exc_info:  # noqa: BLE001
        client._build_ctx(  # noqa: SLF001
            task=object(),
            extra_context={"foo": "bar"},
        )

    err = exc_info.value
    # Error code should be well-typed, but we don't care about the class.
    assert getattr(err, "code", None) == ErrorCodes.BAD_OPERATION_CONTEXT
    assert "Failed to build OperationContext from CrewAI inputs" in str(err)

    assert captured_ctx.get("framework") == "crewai"
    assert captured_ctx.get("operation") == "context_translation"


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

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
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
    # Implementation-specific but should look like a graph operation name
    assert str(captured_context.get("operation", "")).startswith("graph_")


@pytest.mark.asyncio
async def test_error_context_includes_crewai_metadata_async(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    Same as the sync error-context test but for the async query path.
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
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
    assert str(captured_context.get("operation", "")).startswith("graph_")


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
        task=object(),
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
        task=object(),
        extra_context={"request_id": "req-async"},
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Streaming: invalid chunks exercise validate_graph_result_type
# ---------------------------------------------------------------------------


def test_stream_query_invalid_chunk_triggers_validation(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    If the translator yields invalid chunks, stream_query should pass them
    through validate_graph_result_type, and failures there should surface
    to the caller.
    """
    captured: Dict[str, Any] = {}

    class BadChunkTranslator:
        def query_stream(
            self,
            raw_query: Mapping[str, Any],  # noqa: ARG002
            *,
            op_ctx: Any = None,  # noqa: ARG002
            framework_ctx: Mapping[str, Any] | None = None,  # noqa: ARG002
        ):
            # Yield a blatantly invalid chunk
            yield "not-a-chunk"

    _patch_create_graph_translator(monkeypatch, BadChunkTranslator)

    def fake_validate_graph_result_type(result: Any, **kwargs: Any) -> Any:
        captured["result"] = result
        captured["kwargs"] = kwargs
        raise RuntimeError("forced validation failure for chunk")

    monkeypatch.setattr(
        crewai_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )

    client = _make_client(graph_adapter)

    iterator = client.stream_query("MATCH (n) RETURN n")
    with pytest.raises(RuntimeError, match="forced validation failure for chunk"):
        next(iterator)

    assert captured.get("result") == "not-a-chunk"
    assert "expected_type" in captured.get("kwargs", {})


@pytest.mark.asyncio
async def test_astream_query_invalid_chunk_triggers_validation_async(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    Async streaming path should also exercise validate_graph_result_type when
    chunks are invalid.
    """
    captured: Dict[str, Any] = {}

    class BadChunkTranslator:
        async def arun_query_stream(
            self,
            raw_query: Mapping[str, Any],  # noqa: ARG002
            *,
            op_ctx: Any = None,  # noqa: ARG002
            framework_ctx: Mapping[str, Any] | None = None,  # noqa: ARG002
        ):
            # Simple async generator
            async def gen() -> Any:
                yield "not-a-chunk-async"

            return gen()

    _patch_create_graph_translator(monkeypatch, BadChunkTranslator)

    def fake_validate_graph_result_type(result: Any, **kwargs: Any) -> Any:
        captured["result"] = result
        captured["kwargs"] = kwargs
        raise RuntimeError("forced validation failure for async chunk")

    monkeypatch.setattr(
        crewai_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )

    client = _make_client(graph_adapter)

    aiter = await client.astream_query("MATCH (n) RETURN n")
    with pytest.raises(RuntimeError, match="forced validation failure for async chunk"):
        async for _ in aiter:  # noqa: B007
            break

    assert captured.get("result") == "not-a-chunk-async"
    assert "expected_type" in captured.get("kwargs", {})


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

    _patch_create_graph_translator(monkeypatch, DummyTranslator)
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    client = _make_client(graph_adapter, framework_version="fw-bulk-1")

    spec = DummyBulkSpec(
        namespace="ns-bulk",
        limit=42,
        cursor="cursor-token",
        filter_={"foo": "bar"},
    )

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

    _patch_create_graph_translator(monkeypatch, DummyTranslator)
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    client = _make_client(graph_adapter, framework_version="fw-abulk-1")

    spec = DummyBulkSpec(
        namespace="ns-abulk",
        limit=7,
        cursor=None,
        filter_={"bar": 1},
    )

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

    _patch_create_graph_translator(monkeypatch, DummyTranslator)
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    def fake_validate_batch_operations(*_: Any, **__: Any) -> None:
        return None

    monkeypatch.setattr(
        crewai_adapter_module,
        "validate_batch_operations",
        fake_validate_batch_operations,
    )

    client = _make_client(graph_adapter, framework_version="fw-batch-1")

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

    _patch_create_graph_translator(monkeypatch, DummyTranslator)
    _patch_validate_graph_result_type_passthrough(monkeypatch)

    def fake_validate_batch_operations(*_: Any, **__: Any) -> None:
        return None

    monkeypatch.setattr(
        crewai_adapter_module,
        "validate_batch_operations",
        fake_validate_batch_operations,
    )

    client = _make_client(graph_adapter, framework_version="fw-abatch-1")

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
# Capabilities / health passthrough (basic)
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
