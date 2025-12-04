from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List

import inspect

import pytest

import corpus_sdk.graph.framework_adapters.langchain as langchain_adapter_module
from corpus_sdk.graph.framework_adapters.langchain import (
    CorpusLangChainGraphClient,
    CorpusGraphTool,
    LangChainGraphFrameworkTranslator,
    create_corpus_graph_tool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(adapter: Any, **kwargs: Any) -> CorpusLangChainGraphClient:
    """Construct a CorpusLangChainGraphClient instance from the generic adapter."""
    return CorpusLangChainGraphClient(adapter=adapter, **kwargs)


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

    # Patch OperationContext inside the module so our fake can return
    # a simple dummy instance that passes the isinstance() check.
    class DummyOperationContext:
        def __init__(self, **kwargs: Any) -> None:
            self.attrs = kwargs

    monkeypatch.setattr(
        langchain_adapter_module,
        "OperationContext",
        DummyOperationContext,
    )

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

    client = _make_client(
        adapter,
        framework_version="langchain-test-version",
    )

    lc_config = {
        "configurable": {
            "user_id": "user-123",
            "run_id": "run-xyz",
        },
        "tags": ["foo", "bar"],
    }
    extra_ctx = {
        "request_id": "req-xyz",
        "tenant": "tenant-1",
    }

    result = client.query(
        "MATCH (n) RETURN n LIMIT 1",
        config=lc_config,
        extra_context=extra_ctx,
    )
    assert result is not None

    assert captured.get("config") == lc_config
    assert captured.get("framework_version") == "langchain-test-version"
    # extra_context should be merged into **extra
    assert captured.get("extra") == extra_ctx


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

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(
        langchain_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingTranslator:
        def query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from langchain graph adapter")

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return FailingTranslator()

    monkeypatch.setattr(
        langchain_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(adapter)

    config = {"configurable": {"user_id": "u-sync"}}

    with pytest.raises(RuntimeError, match="test error from langchain graph adapter"):
        client.query("MATCH (n) RETURN n", config=config)

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

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(
        langchain_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingTranslator:
        async def arun_query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from langchain graph adapter")

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return FailingTranslator()

    monkeypatch.setattr(
        langchain_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(adapter)

    config = {"configurable": {"user_id": "u-async"}}

    with pytest.raises(RuntimeError, match="test error from langchain graph adapter"):
        await client.aquery("MATCH (n) RETURN n", config=config)

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

    # Non-streaming query
    result = client.query("MATCH (n) RETURN n LIMIT 1")
    assert result is not None

    # Streaming query
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
    - astream_query() returns an async-iterable object (after any awaiting).
    """
    client = _make_client(adapter)

    assert hasattr(client, "aquery")
    assert hasattr(client, "astream_query")

    # Sync vs async query type parity
    sync_result = client.query("MATCH (n) RETURN n LIMIT 1")
    assert sync_result is not None

    coro = client.aquery("MATCH (n) RETURN n LIMIT 1")
    assert inspect.isawaitable(coro)
    async_result = await coro
    assert async_result is not None

    assert type(sync_result) is type(  # noqa: E721
        async_result
    ), "query and aquery should return the same result type"

    # Async streaming contract: awaitable or async-iterable, but must end up
    # as an async-iterable object.
    aiter = client.astream_query("MATCH (n) RETURN n LIMIT 2")
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    # Stronger contract: object must be async-iterable
    assert hasattr(
        aiter,
        "__aiter__",
    ), "astream_query must return an async-iterable (or awaitable resolving to one)"

    seen_any = False
    async for _ in aiter:  # noqa: B007
        seen_any = True
        break

    assert isinstance(seen_any, bool)


@pytest.mark.asyncio
async def test_async_query_accepts_optional_params_and_config(
    adapter: Any,
) -> None:
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
# Bulk vertices / batch semantics (LangChain wiring)
# ---------------------------------------------------------------------------


def test_bulk_vertices_builds_raw_request_and_calls_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
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
        langchain_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        langchain_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )

    client = _make_client(adapter)

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
    assert fw_ctx.get("framework") == "langchain"
    assert fw_ctx.get("operation") == "bulk_vertices"
    assert fw_ctx.get("namespace") == "ns-bulk"
    assert captured["op_ctx"] is None


@pytest.mark.asyncio
async def test_abulk_vertices_builds_raw_request_and_calls_translator_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
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
        langchain_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        langchain_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )

    client = _make_client(adapter)

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
    assert fw_ctx.get("framework") == "langchain"
    assert fw_ctx.get("operation") == "bulk_vertices"
    assert fw_ctx.get("namespace") == "ns-abulk"
    assert captured["op_ctx"] is None


def test_batch_builds_raw_batch_ops_and_calls_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    batch() should:

    - Validate batch operations (we stub validation here),
    - Translate BatchOperation-like objects into raw_batch_ops mappings
      passed to translator.batch(),
    - Pass a framework_ctx containing framework='langchain' and operation='batch'.
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
        langchain_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        langchain_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )
    monkeypatch.setattr(
        langchain_adapter_module,
        "validate_batch_operations",
        fake_validate_batch_operations,
    )

    client = _make_client(adapter)

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
    assert fw_ctx.get("framework") == "langchain"
    assert fw_ctx.get("operation") == "batch"
    assert captured["op_ctx"] is None


@pytest.mark.asyncio
async def test_abatch_builds_raw_batch_ops_and_calls_translator_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    abatch() should mirror batch wiring but via translator.arun_batch.
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
        langchain_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        langchain_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )
    monkeypatch.setattr(
        langchain_adapter_module,
        "validate_batch_operations",
        fake_validate_batch_operations,
    )

    client = _make_client(adapter)

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
    assert fw_ctx.get("framework") == "langchain"
    assert fw_ctx.get("operation") == "batch"
    assert captured["op_ctx"] is None


# ---------------------------------------------------------------------------
# Capabilities / health passthrough (basic + sync/async parity)
# ---------------------------------------------------------------------------


def test_capabilities_and_health_basic(adapter: Any) -> None:
    """
    Capabilities and health should be surfaced as mappings.

    The detailed structure is tested in framework-agnostic graph contract
    tests; here we only assert that the LangChain adapter normalizes to
    mapping-like results.
    """
    client = _make_client(adapter)

    caps = client.capabilities()
    assert isinstance(caps, Mapping)

    health = client.health()
    assert isinstance(health, Mapping)


@pytest.mark.asyncio
async def test_async_capabilities_and_health_basic(adapter: Any) -> None:
    """
    Async capabilities/health should also return mappings compatible with
    the sync variants, and expose the same key sets for basic parity.
    """
    client = _make_client(adapter)

    # Sync values for comparison
    caps = client.capabilities()
    health = client.health()

    acaps = await client.acapabilities()
    assert isinstance(acaps, Mapping)

    ahealth = await client.ahealth()
    assert isinstance(ahealth, Mapping)

    # Parity: async should expose the same keys as sync
    assert set(acaps.keys()) == set(
        caps.keys(),
    ), "acapabilities should expose the same keys as capabilities"
    assert set(ahealth.keys()) == set(
        health.keys(),
    ), "ahealth should expose the same keys as health"


# ---------------------------------------------------------------------------
# Resource management (context managers)
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

    # Sync context manager
    with CorpusLangChainGraphClient(adapter=adapter) as client:
        assert client is not None

    assert adapter.closed is True

    # Async context manager
    adapter2 = ClosingGraphAdapter()
    client2 = CorpusLangChainGraphClient(adapter=adapter2)

    async with client2:
        assert client2 is not None

    assert adapter2.aclosed is True


# ---------------------------------------------------------------------------
# CorpusGraphTool behavior / LangChain integration
# ---------------------------------------------------------------------------


def test_corpus_graph_tool_parses_string_and_mapping_input(
    adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    CorpusGraphTool should:

    - Accept plain string input as a query.
    - Accept mapping input with 'query' and optional fields.
    - Delegate to underlying client's query() method.
    """
    captured: Dict[str, Any] = {}

    class DummyClient(CorpusLangChainGraphClient):
        def __init__(self) -> None:
            # Avoid needing a real graph adapter here
            pass

        def query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            captured["kwargs"] = kwargs
            return {"ok": True, "query": kwargs.get("query")}

    dummy_client = DummyClient()
    tool = CorpusGraphTool(graph_client=dummy_client)

    # Plain string input
    out1 = tool._run("MATCH (n) RETURN n")
    assert "MATCH (n) RETURN n" in out1

    assert captured["kwargs"]["query"] == "MATCH (n) RETURN n"
    assert captured["kwargs"]["params"] is None

    # Mapping input
    out2 = tool._run(
        {
            "query": "MATCH (m) RETURN m",
            "params": {"limit": 5},
            "dialect": "cypher",
            "namespace": "tool-ns",
            "timeout_ms": 1234,
        },
    )
    assert "MATCH (m) RETURN m" in out2
    assert captured["kwargs"]["query"] == "MATCH (m) RETURN m"
    assert captured["kwargs"]["params"] == {"limit": 5}
    assert captured["kwargs"]["dialect"] == "cypher"
    assert captured["kwargs"]["namespace"] == "tool-ns"
    assert captured["kwargs"]["timeout_ms"] == 1234


def test_corpus_graph_tool_rejects_invalid_input_type() -> None:
    """
    CorpusGraphTool._parse_input should raise BadRequest when given an
    unsupported input type (e.g. int).
    """

    class DummyClient(CorpusLangChainGraphClient):
        def __init__(self) -> None:
            pass

    tool = CorpusGraphTool(graph_client=DummyClient())

    with pytest.raises(Exception) as exc_info:
        tool._parse_input(123)  # type: ignore[arg-type]

    # We don't depend on the concrete BadRequest type here; just check message.
    msg = str(exc_info.value)
    assert "Unsupported tool input type" in msg


@pytest.mark.asyncio
async def test_corpus_graph_tool_async_delegates_to_aquery() -> None:
    """
    _arun() should delegate to the underlying client's aquery() method and
    return a serialized string.
    """
    captured: Dict[str, Any] = {}

    class DummyClient(CorpusLangChainGraphClient):
        def __init__(self) -> None:
            pass

        async def aquery(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            captured["kwargs"] = kwargs
            return {"async_ok": True, "query": kwargs.get("query")}

    tool = CorpusGraphTool(graph_client=DummyClient())

    out = await tool._arun("MATCH (a) RETURN a")
    assert "MATCH (a) RETURN a" in out
    assert captured["kwargs"]["query"] == "MATCH (a) RETURN a"


def test_create_corpus_graph_tool_wraps_client_and_adapter(
    adapter: Any,
) -> None:
    """
    create_corpus_graph_tool should:

    - Construct a CorpusLangChainGraphClient wired to the given adapter.
    - Wrap it in a CorpusGraphTool.
    - Propagate default namespace/dialect into the client.
    """
    tool = create_corpus_graph_tool(
        adapter=adapter,
        default_dialect="cypher",
        default_namespace="prod",
        default_timeout_ms=9999,
        framework_version="lc-fw-2.0",
        name="my_graph_tool",
        description="My graph tool for LangChain",
    )

    assert isinstance(tool, CorpusGraphTool)
    assert tool.name == "my_graph_tool"
    assert "graph" in tool.description.lower()

    client = tool._graph_client
    assert isinstance(client, CorpusLangChainGraphClient)
    # Inspect private attributes to ensure wiring is correct
    assert getattr(client, "_graph") is adapter
    assert getattr(client, "_default_dialect") == "cypher"
    assert getattr(client, "_default_namespace") == "prod"
    assert getattr(client, "_default_timeout_ms") == 9999
    assert getattr(client, "_framework_version") == "lc-fw-2.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
