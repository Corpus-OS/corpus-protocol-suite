# tests/frameworks/graph/test_llamaindex_graph_adapter.py

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List

import inspect

import pytest

import corpus_sdk.graph.framework_adapters.llamaindex as llamaindex_adapter_module
from corpus_sdk.graph.framework_adapters.llamaindex import (
    CorpusLlamaIndexGraphClient,
    CorpusGraphStore,
    LlamaIndexGraphFrameworkTranslator,
)


# Whether llama_index is available in this environment
HAS_LLAMAINDEX = getattr(llamaindex_adapter_module, "_LlamaIndexGraphStore", None) is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(graph_adapter: Any, **kwargs: Any) -> CorpusLlamaIndexGraphClient:
    """Construct a CorpusLlamaIndexGraphClient instance from the generic adapter."""
    return CorpusLlamaIndexGraphClient(graph_adapter=graph_adapter, **kwargs)


# ---------------------------------------------------------------------------
# Constructor / translator behavior
# ---------------------------------------------------------------------------


def test_default_translator_uses_llamaindex_framework_translator(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    By default, CorpusLlamaIndexGraphClient should:

    - Construct a LlamaIndexGraphFrameworkTranslator instance, and
    - Pass it into create_graph_translator with framework="llamaindex".
    """
    captured: Dict[str, Any] = {}

    def fake_create_graph_translator(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["args"] = args
        captured["kwargs"] = kwargs

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(
        llamaindex_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(graph_adapter)

    # Trigger lazy translator construction
    _ = client._translator  # noqa: SLF001

    kwargs = captured["kwargs"]
    assert kwargs.get("framework") == "llamaindex"
    translator = kwargs.get("translator")
    assert isinstance(translator, LlamaIndexGraphFrameworkTranslator)


def test_framework_translator_override_is_respected(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    If framework_translator is provided, CorpusLlamaIndexGraphClient should pass
    it through to create_graph_translator instead of constructing its own
    LlamaIndexGraphFrameworkTranslator.
    """
    captured: Dict[str, Any] = {}

    class CustomTranslator(LlamaIndexGraphFrameworkTranslator):
        pass

    custom = CustomTranslator()

    def fake_create_graph_translator(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["args"] = args
        captured["kwargs"] = kwargs

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(
        llamaindex_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(
        graph_adapter,
        framework_translator=custom,
        framework_version="li-fw-1.2.3",
    )

    _ = client._translator  # noqa: SLF001

    kwargs = captured["kwargs"]
    assert kwargs.get("framework") == "llamaindex"
    assert kwargs.get("translator") is custom


# ---------------------------------------------------------------------------
# Context translation / core_ctx_from_llamaindex mapping
# ---------------------------------------------------------------------------


def test_llamaindex_callback_manager_and_extra_context_passed_to_core_ctx(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    Verify that callback_manager and extra_context are passed through to
    core_ctx_from_llamaindex with the configured framework_version.
    """
    captured: Dict[str, Any] = {}

    # Patch OperationContext inside the module so our fake can return
    # a simple dummy instance that passes the isinstance() check.
    class DummyOperationContext:
        def __init__(self, **kwargs: Any) -> None:
            self.attrs = kwargs

    monkeypatch.setattr(
        llamaindex_adapter_module,
        "OperationContext",
        DummyOperationContext,
    )

    def fake_core_ctx_from_llamaindex(
        callback_manager: Any,
        *,
        framework_version: Any = None,
        **extra: Any,
    ) -> Any:
        captured["callback_manager"] = callback_manager
        captured["framework_version"] = framework_version
        captured["extra"] = extra
        return DummyOperationContext()

    monkeypatch.setattr(
        llamaindex_adapter_module,
        "core_ctx_from_llamaindex",
        fake_core_ctx_from_llamaindex,
    )

    client = _make_client(
        graph_adapter,
        framework_version="llamaindex-test-version",
    )

    cb_mgr = object()
    extra_ctx = {
        "request_id": "req-li-xyz",
        "tenant": "tenant-42",
    }

    result = client.query(
        "MATCH (n) RETURN n LIMIT 1",
        callback_manager=cb_mgr,
        extra_context=extra_ctx,
    )
    assert result is not None

    assert captured.get("callback_manager") is cb_mgr
    assert captured.get("framework_version") == "llamaindex-test-version"
    # extra_context should be merged into **extra
    assert captured.get("extra") == extra_ctx


# ---------------------------------------------------------------------------
# Error-context decorator behavior
# ---------------------------------------------------------------------------


def test_error_context_includes_llamaindex_metadata_sync(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    When an error occurs during a sync graph operation, error context should
    include LlamaIndex-specific metadata via attach_context().
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(
        llamaindex_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingTranslator:
        def query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from llamaindex graph adapter")

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return FailingTranslator()

    monkeypatch.setattr(
        llamaindex_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(graph_adapter)

    with pytest.raises(RuntimeError, match="test error from llamaindex graph adapter"):
        client.query("MATCH (n) RETURN n")

    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "llamaindex"
    # Implementation-specific but should look like a graph operation name
    assert str(captured_context.get("operation", "")).startswith("graph_")


@pytest.mark.asyncio
async def test_error_context_includes_llamaindex_metadata_async(
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
        llamaindex_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingTranslator:
        async def arun_query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from llamaindex graph adapter")

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return FailingTranslator()

    monkeypatch.setattr(
        llamaindex_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(graph_adapter)

    with pytest.raises(RuntimeError, match="test error from llamaindex graph adapter"):
        await client.aquery("MATCH (n) RETURN n")

    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "llamaindex"
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
    client = _make_client(graph_adapter, default_namespace="li-ns")

    # Non-streaming query
    result = client.query("MATCH (n) RETURN n LIMIT 1")
    assert result is not None

    # Streaming query
    chunks = list(client.stream_query("MATCH (n) RETURN n LIMIT 2"))
    assert isinstance(chunks, list)


def test_sync_query_accepts_optional_params_and_callback_manager(
    graph_adapter: Any,
) -> None:
    """
    query() should accept params, dialect, namespace, timeout_ms, and
    callback_manager/extra_context kwargs without raising.
    """
    client = _make_client(graph_adapter, default_dialect="cypher")

    cb_mgr = object()

    result = client.query(
        "MATCH (n) RETURN n LIMIT $limit",
        params={"limit": 5},
        dialect="cypher",
        namespace="ctx-ns",
        timeout_ms=5000,
        callback_manager=cb_mgr,
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
async def test_async_query_accepts_optional_params_and_callback_manager(
    graph_adapter: Any,
) -> None:
    """
    aquery() should accept the same optional params and context as query().
    """
    client = _make_client(graph_adapter, default_namespace="async-li-ns")

    cb_mgr = object()

    result = await client.aquery(
        "MATCH (n) RETURN n LIMIT $limit",
        params={"limit": 3},
        dialect="cypher",
        namespace="async-li-ns",
        timeout_ms=2500,
        callback_manager=cb_mgr,
        extra_context={"request_id": "req-async"},
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Bulk vertices / batch semantics (LlamaIndex wiring)
# ---------------------------------------------------------------------------


def test_bulk_vertices_builds_raw_request_and_calls_translator(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    bulk_vertices() should:

    - Build the correct raw_request mapping from the spec, and
    - Call the underlying translator.bulk_vertices with that mapping and
      appropriate framework_ctx (namespace + framework metadata).
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
        llamaindex_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        llamaindex_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )

    client = _make_client(graph_adapter)

    class DummyBulkSpec:
        def __init__(self) -> None:
            self.namespace = "ns-bulk-li"
            self.limit = 42
            self.cursor = "cursor-token"
            self.filter = {"foo": "bar"}

    spec = DummyBulkSpec()

    result = client.bulk_vertices(spec)
    assert result == "bulk-result"

    raw = captured["raw_request"]
    assert raw == {
        "namespace": "ns-bulk-li",
        "limit": 42,
        "cursor": "cursor-token",
        "filter": {"foo": "bar"},
    }

    fw_ctx = captured["framework_ctx"]
    assert fw_ctx.get("framework") == "llamaindex"
    assert fw_ctx.get("operation") == "bulk_vertices"
    assert fw_ctx.get("namespace") == "ns-bulk-li"
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
        llamaindex_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        llamaindex_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )

    client = _make_client(graph_adapter)

    class DummyBulkSpec:
        def __init__(self) -> None:
            self.namespace = "ns-abulk-li"
            self.limit = 7
            self.cursor = None
            self.filter = {"bar": 1}

    spec = DummyBulkSpec()

    result = await client.abulk_vertices(spec)
    assert result == "bulk-result-async"

    raw = captured["raw_request"]
    assert raw == {
        "namespace": "ns-abulk-li",
        "limit": 7,
        "cursor": None,
        "filter": {"bar": 1},
    }

    fw_ctx = captured["framework_ctx"]
    assert fw_ctx.get("framework") == "llamaindex"
    assert fw_ctx.get("operation") == "bulk_vertices"
    assert fw_ctx.get("namespace") == "ns-abulk-li"
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
        llamaindex_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        llamaindex_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )
    monkeypatch.setattr(
        llamaindex_adapter_module,
        "validate_batch_operations",
        fake_validate_batch_operations,
    )

    client = _make_client(graph_adapter)

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
    assert fw_ctx.get("framework") == "llamaindex"
    assert fw_ctx.get("operation") == "batch"
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
        llamaindex_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        llamaindex_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )
    monkeypatch.setattr(
        llamaindex_adapter_module,
        "validate_batch_operations",
        fake_validate_batch_operations,
    )

    client = _make_client(graph_adapter)

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
    assert fw_ctx.get("framework") == "llamaindex"
    assert fw_ctx.get("operation") == "batch"
    assert captured["op_ctx"] is None


# ---------------------------------------------------------------------------
# Capabilities / health passthrough (basic)
# ---------------------------------------------------------------------------


def test_capabilities_and_health_basic(graph_adapter: Any) -> None:
    """
    Capabilities and health should be surfaced as mappings.

    The detailed structure is tested in framework-agnostic graph contract
    tests; here we only assert that the LlamaIndex adapter normalizes to
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
    with CorpusLlamaIndexGraphClient(graph_adapter=adapter) as client:
        assert client is not None

    assert adapter.closed is True

    # Async context manager
    adapter2 = ClosingGraphAdapter()
    client2 = CorpusLlamaIndexGraphClient(graph_adapter=adapter2)

    async with client2:
        assert client2 is not None

    assert adapter2.aclosed is True


# ---------------------------------------------------------------------------
# CorpusGraphStore behavior (optional LlamaIndex integration)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    HAS_LLAMAINDEX,
    reason="llama_index is installed; stub CorpusGraphStore is not used",
)
def test_corpus_graph_store_stub_raises_import_error_when_llamaindex_missing() -> None:
    """
    When llama_index is not installed, CorpusGraphStore should be a stub that
    raises ImportError on construction.
    """
    with pytest.raises(ImportError):
        CorpusGraphStore(client=object())  # type: ignore[call-arg]


@pytest.mark.skipif(
    not HAS_LLAMAINDEX,
    reason="llama_index is not installed; real CorpusGraphStore unavailable",
)
def test_corpus_graph_store_delegates_query_to_underlying_client() -> None:
    """
    When llama_index is installed, CorpusGraphStore should delegate queries to
    the underlying CorpusLlamaIndexGraphClient with the configured namespace.
    """
    captured: Dict[str, Any] = {}

    class DummyClient(CorpusLlamaIndexGraphClient):
        def __init__(self) -> None:
            # Avoid needing a real graph adapter here
            pass

        def query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            captured["args"] = args
            captured["kwargs"] = kwargs

            class Result:
                def __init__(self) -> None:
                    self.rows = [["s", "r", "o"]]

            return Result()

    dummy_client = DummyClient()
    store = CorpusGraphStore(
        client=dummy_client,
        namespace="store-ns",
        get_query="MATCH (s)-[r]->(o) WHERE s.id = $subj RETURN s, r, o",
    )

    # Generic GraphStore.query
    _ = store.query("MATCH (n) RETURN n", {"k": "v"})
    assert captured["args"][0] == "MATCH (n) RETURN n"
    assert captured["kwargs"]["params"] == {"k": "v"}
    assert captured["kwargs"]["namespace"] == "store-ns"

    # get() should call underlying .query with get_query and subject param
    captured.clear()
    rows = store.get("subj-1")
    assert isinstance(rows, list)
    assert rows == [["s", "r", "o"]]
    assert captured["args"][0] == "MATCH (s)-[r]->(o) WHERE s.id = $subj RETURN s, r, o"
    assert captured["kwargs"]["params"] == {"subj": "subj-1"}
    assert captured["kwargs"]["namespace"] == "store-ns"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

