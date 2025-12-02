# tests/frameworks/graph/test_llamaindex_graph_adapter.py

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List

import inspect

import pytest

import corpus_sdk.graph.framework_adapters.llamaindex as llamaindex_adapter_module
from corpus_sdk.graph.framework_adapters.llamaindex import (
    CorpusGraphStore,
    CorpusLlamaIndexGraphClient,
    ErrorCodes,
    LlamaIndexGraphFrameworkTranslator,
)

# Try to detect whether llama_index is installed in this environment.
try:  # pragma: no cover - environment dependent
    from llama_index.core.graph_stores.types import GraphStore as _LI_GraphStore

    HAS_LLAMAINDEX = True
except Exception:  # pragma: no cover - environment dependent
    _LI_GraphStore = None  # type: ignore[assignment]
    HAS_LLAMAINDEX = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(graph_adapter: Any, **kwargs: Any) -> CorpusLlamaIndexGraphClient:
    """Construct a CorpusLlamaIndexGraphClient instance from the generic adapter."""
    return CorpusLlamaIndexGraphClient(graph_adapter=graph_adapter, **kwargs)


def _mock_translator_with_capture(
    captured: Dict[str, Any],
    method_name: str,
    return_value: Any,
) -> Any:
    """Helper to create a sync translator that captures call arguments."""

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
    """Helper to create an async translator that captures call arguments."""

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
        framework_version="ll-fw-1.2.3",
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

    # Patch OperationContext so our fake ctx passes isinstance() check.
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

    cb_manager = object()
    extra_ctx = {"request_id": "req-xyz", "tenant": "tenant-1"}

    result = client.query(
        "MATCH (n) RETURN n LIMIT 1",
        callback_manager=cb_manager,
        extra_context=extra_ctx,
    )
    assert result is not None

    assert captured.get("callback_manager") is cb_manager
    assert captured.get("framework_version") == "llamaindex-test-version"
    assert captured.get("extra") == extra_ctx


def test_build_ctx_failure_raises_badrequest_with_error_code_and_context(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    If core_ctx_from_llamaindex raises, _build_ctx should wrap it in a BadRequest-like
    error with ErrorCodes.BAD_OPERATION_CONTEXT and call attach_context.
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_core_ctx_from_llamaindex(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        raise RuntimeError("boom!")

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(
        llamaindex_adapter_module,
        "core_ctx_from_llamaindex",
        fake_core_ctx_from_llamaindex,
    )
    monkeypatch.setattr(
        llamaindex_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    client = _make_client(graph_adapter)

    with pytest.raises(Exception) as exc_info:
        client.query("MATCH (n) RETURN n", callback_manager=object())

    err = exc_info.value
    # We don't import BadRequest here; just inspect shape.
    assert type(err).__name__ == "BadRequest"
    assert getattr(err, "code", None) == ErrorCodes.BAD_OPERATION_CONTEXT
    assert captured_ctx.get("framework") == "llamaindex"
    assert captured_ctx.get("operation") == "context_translation"


# ---------------------------------------------------------------------------
# Error-context decorator behavior
# ---------------------------------------------------------------------------


def test_sync_errors_include_llamaindex_metadata_in_context(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    When an error occurs during a sync graph operation, error context should
    include LlamaIndex-specific metadata via attach_context().
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

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
        client.query("MATCH (n) RETURN n", callback_manager={"user_id": "u-sync"})

    assert captured_ctx
    assert captured_ctx.get("framework") == "llamaindex"
    assert str(captured_ctx.get("operation", "")).startswith("graph_")


@pytest.mark.asyncio
async def test_async_errors_include_llamaindex_metadata_in_context(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    Same as the sync error-context test but for the async query path.
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

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
        await client.aquery("MATCH (n) RETURN n", callback_manager={"user_id": "u-async"})

    assert captured_ctx
    assert captured_ctx.get("framework") == "llamaindex"
    assert str(captured_ctx.get("operation", "")).startswith("graph_")


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
    client = _make_client(graph_adapter, default_namespace="ll-ns")

    result = client.query("MATCH (n) RETURN n LIMIT 1")
    assert result is not None

    chunks = list(client.stream_query("MATCH (n) RETURN n LIMIT 2"))
    assert isinstance(chunks, list)


def test_sync_query_accepts_optional_params_and_context(graph_adapter: Any) -> None:
    """
    query() should accept params, dialect, namespace, timeout_ms, and
    callback_manager/extra_context kwargs without raising.
    """
    client = _make_client(graph_adapter, default_dialect="cypher")

    result = client.query(
        "MATCH (n) RETURN n LIMIT $limit",
        params={"limit": 5},
        dialect="cypher",
        namespace="ctx-ns",
        timeout_ms=5000,
        callback_manager={"user_id": "u-sync"},
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
        callback_manager={"user_id": "u-async"},
        extra_context={"request_id": "req-async"},
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Upsert / delete wiring (Langchain-style coverage for LlamaIndex)
# ---------------------------------------------------------------------------


def test_upsert_nodes_passes_raw_nodes_with_framework_context(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    upsert_nodes() should:

    - Validate via validate_upsert_nodes_spec (stubbed here),
    - Pass spec.nodes as the first positional arg to translator.upsert_nodes,
    - Use framework_ctx carrying framework='llamaindex', operation, and namespace.
    """
    captured: Dict[str, Any] = {}

    translator = _mock_translator_with_capture(
        captured,
        method_name="upsert_nodes",
        return_value="upsert-nodes-result",
    )

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return translator

    def fake_validate_upsert_nodes_spec(spec: Any, **_: Any) -> None:
        captured["validated_spec"] = spec

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    monkeypatch.setattr(
        llamaindex_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        llamaindex_adapter_module,
        "validate_upsert_nodes_spec",
        fake_validate_upsert_nodes_spec,
    )
    monkeypatch.setattr(
        llamaindex_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )

    client = _make_client(graph_adapter)

    class DummyUpsertNodesSpec:
        def __init__(self) -> None:
            self.namespace = "nodes-ns"
            self.nodes = [{"id": "n1"}, {"id": "n2"}]

    spec = DummyUpsertNodesSpec()

    result = client.upsert_nodes(spec)
    assert result == "upsert-nodes-result"

    # validate_* should see the same spec
    assert captured["validated_spec"] is spec

    # Translator should receive raw nodes list as first positional arg
    assert "args" in captured
    assert captured["args"][0] == spec.nodes

    fw_ctx = captured.get("framework_ctx", {})
    assert fw_ctx.get("framework") == "llamaindex"
    assert fw_ctx.get("operation") == "upsert_nodes"
    assert fw_ctx.get("namespace") == "nodes-ns"


@pytest.mark.asyncio
async def test_aupsert_nodes_passes_raw_nodes_with_framework_context_async(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    aupsert_nodes() should mirror upsert_nodes() wiring but via
    translator.arun_upsert_nodes.
    """
    captured: Dict[str, Any] = {}

    translator = _mock_async_translator_with_capture(
        captured,
        method_name="arun_upsert_nodes",
        return_value="aupsert-nodes-result",
    )

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return translator

    def fake_validate_upsert_nodes_spec(spec: Any, **_: Any) -> None:
        captured["validated_spec"] = spec

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    monkeypatch.setattr(
        llamaindex_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        llamaindex_adapter_module,
        "validate_upsert_nodes_spec",
        fake_validate_upsert_nodes_spec,
    )
    monkeypatch.setattr(
        llamaindex_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )

    client = _make_client(graph_adapter)

    class DummyUpsertNodesSpec:
        def __init__(self) -> None:
            self.namespace = "async-nodes-ns"
            self.nodes = [{"id": "an1"}, {"id": "an2"}]

    spec = DummyUpsertNodesSpec()

    result = await client.aupsert_nodes(spec)
    assert result == "aupsert-nodes-result"

    assert captured["validated_spec"] is spec
    assert "args" in captured
    assert captured["args"][0] == spec.nodes

    fw_ctx = captured.get("framework_ctx", {})
    assert fw_ctx.get("framework") == "llamaindex"
    assert fw_ctx.get("operation") == "upsert_nodes"
    assert fw_ctx.get("namespace") == "async-nodes-ns"


def test_upsert_edges_passes_edges_with_framework_context(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    upsert_edges() should:

    - Call _validate_upsert_edges_spec on the client,
    - Pass spec.edges as first positional arg to translator.upsert_edges,
    - Use framework_ctx with framework='llamaindex' and proper namespace.
    """
    captured: Dict[str, Any] = {}

    translator = _mock_translator_with_capture(
        captured,
        method_name="upsert_edges",
        return_value="upsert-edges-result",
    )

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return translator

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    # Stub the instance method _validate_upsert_edges_spec
    def fake_validate_edges(self: Any, spec: Any) -> None:  # noqa: ARG001
        captured["validated_edges_spec"] = spec

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
        CorpusLlamaIndexGraphClient,
        "_validate_upsert_edges_spec",
        fake_validate_edges,
    )

    client = _make_client(graph_adapter)

    class DummyUpsertEdgesSpec:
        def __init__(self) -> None:
            self.namespace = "edges-ns"
            self.edges = [{"id": "e1"}, {"id": "e2"}]

    spec = DummyUpsertEdgesSpec()

    result = client.upsert_edges(spec)
    assert result == "upsert-edges-result"
    assert captured["validated_edges_spec"] is spec

    assert "args" in captured
    assert captured["args"][0] == spec.edges

    fw_ctx = captured.get("framework_ctx", {})
    assert fw_ctx.get("framework") == "llamaindex"
    assert fw_ctx.get("operation") == "upsert_edges"
    assert fw_ctx.get("namespace") == "edges-ns"


@pytest.mark.asyncio
async def test_aupsert_edges_passes_edges_with_framework_context_async(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    aupsert_edges() should mirror upsert_edges() wiring but via
    translator.arun_upsert_edges.
    """
    captured: Dict[str, Any] = {}

    translator = _mock_async_translator_with_capture(
        captured,
        method_name="arun_upsert_edges",
        return_value="aupsert-edges-result",
    )

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return translator

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    def fake_validate_edges(self: Any, spec: Any) -> None:  # noqa: ARG001
        captured["validated_edges_spec"] = spec

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
        CorpusLlamaIndexGraphClient,
        "_validate_upsert_edges_spec",
        fake_validate_edges,
    )

    client = _make_client(graph_adapter)

    class DummyUpsertEdgesSpec:
        def __init__(self) -> None:
            self.namespace = "async-edges-ns"
            self.edges = [{"id": "ae1"}, {"id": "ae2"}]

    spec = DummyUpsertEdgesSpec()

    result = await client.aupsert_edges(spec)
    assert result == "aupsert-edges-result"
    assert captured["validated_edges_spec"] is spec

    assert "args" in captured
    assert captured["args"][0] == spec.edges

    fw_ctx = captured.get("framework_ctx", {})
    assert fw_ctx.get("framework") == "llamaindex"
    assert fw_ctx.get("operation") == "upsert_edges"
    assert fw_ctx.get("namespace") == "async-edges-ns"


def test_delete_nodes_passes_ids_with_framework_context(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    delete_nodes() should pass a list of IDs or filter into translator.delete_nodes
    and include proper framework_ctx metadata.
    """
    captured: Dict[str, Any] = {}

    translator = _mock_translator_with_capture(
        captured,
        method_name="delete_nodes",
        return_value="delete-nodes-result",
    )

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return translator

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

    class DummyDeleteNodesSpec:
        def __init__(self) -> None:
            self.namespace = "del-ns"
            self.ids = ["n1", "n2"]
            self.filter = None

    spec = DummyDeleteNodesSpec()

    result = client.delete_nodes(spec)
    assert result == "delete-nodes-result"

    assert "args" in captured
    assert captured["args"][0] == ["n1", "n2"]

    fw_ctx = captured.get("framework_ctx", {})
    assert fw_ctx.get("framework") == "llamaindex"
    assert fw_ctx.get("operation") == "delete_nodes"
    assert fw_ctx.get("namespace") == "del-ns"


@pytest.mark.asyncio
async def test_adelete_nodes_passes_ids_with_framework_context_async(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    adelete_nodes() should mirror delete_nodes wiring via arun_delete_nodes.
    """
    captured: Dict[str, Any] = {}

    translator = _mock_async_translator_with_capture(
        captured,
        method_name="arun_delete_nodes",
        return_value="adelete-nodes-result",
    )

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return translator

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

    class DummyDeleteNodesSpec:
        def __init__(self) -> None:
            self.namespace = "adel-ns"
            self.ids = ["an1", "an2"]
            self.filter = None

    spec = DummyDeleteNodesSpec()

    result = await client.adelete_nodes(spec)
    assert result == "adelete-nodes-result"

    assert "args" in captured
    assert captured["args"][0] == ["an1", "an2"]

    fw_ctx = captured.get("framework_ctx", {})
    assert fw_ctx.get("framework") == "llamaindex"
    assert fw_ctx.get("operation") == "delete_nodes"
    assert fw_ctx.get("namespace") == "adel-ns"


def test_delete_edges_passes_ids_with_framework_context(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    delete_edges() should pass ID list or filter to translator.delete_edges
    and include proper framework_ctx.
    """
    captured: Dict[str, Any] = {}

    translator = _mock_translator_with_capture(
        captured,
        method_name="delete_edges",
        return_value="delete-edges-result",
    )

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return translator

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

    class DummyDeleteEdgesSpec:
        def __init__(self) -> None:
            self.namespace = "del-edges-ns"
            self.ids = ["e1", "e2"]
            self.filter = None

    spec = DummyDeleteEdgesSpec()

    result = client.delete_edges(spec)
    assert result == "delete-edges-result"

    assert "args" in captured
    assert captured["args"][0] == ["e1", "e2"]

    fw_ctx = captured.get("framework_ctx", {})
    assert fw_ctx.get("framework") == "llamaindex"
    assert fw_ctx.get("operation") == "delete_edges"
    assert fw_ctx.get("namespace") == "del-edges-ns"


@pytest.mark.asyncio
async def test_adelete_edges_passes_ids_with_framework_context_async(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    adelete_edges() should mirror delete_edges wiring via arun_delete_edges.
    """
    captured: Dict[str, Any] = {}

    translator = _mock_async_translator_with_capture(
        captured,
        method_name="arun_delete_edges",
        return_value="adelete-edges-result",
    )

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return translator

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

    class DummyDeleteEdgesSpec:
        def __init__(self) -> None:
            self.namespace = "adel-edges-ns"
            self.ids = ["ae1", "ae2"]
            self.filter = None

    spec = DummyDeleteEdgesSpec()

    result = await client.adelete_edges(spec)
    assert result == "adelete-edges-result"

    assert "args" in captured
    assert captured["args"][0] == ["ae1", "ae2"]

    fw_ctx = captured.get("framework_ctx", {})
    assert fw_ctx.get("framework") == "llamaindex"
    assert fw_ctx.get("operation") == "delete_edges"
    assert fw_ctx.get("namespace") == "adel-edges-ns"


# ---------------------------------------------------------------------------
# Bulk vertices / batch semantics (wiring)
# ---------------------------------------------------------------------------


def test_bulk_vertices_builds_raw_request_and_calls_translator(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    bulk_vertices() should:

    - Build the correct raw_request mapping from the spec, and
    - Call translator.bulk_vertices with that mapping and framework_ctx.
    """
    captured: Dict[str, Any] = {}

    translator = _mock_translator_with_capture(
        captured,
        method_name="bulk_vertices",
        return_value="bulk-result",
    )

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return translator

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
            self.namespace = "ns-bulk"
            self.limit = 42
            self.cursor = "cursor-token"
            self.filter = {"foo": "bar"}

    spec = DummyBulkSpec()

    result = client.bulk_vertices(spec)
    assert result == "bulk-result"

    # First positional arg should be raw_request mapping
    assert "args" in captured
    raw = captured["args"][0]
    assert raw == {
        "namespace": "ns-bulk",
        "limit": 42,
        "cursor": "cursor-token",
        "filter": {"foo": "bar"},
    }

    fw_ctx = captured.get("framework_ctx", {})
    assert fw_ctx.get("framework") == "llamaindex"
    assert fw_ctx.get("operation") == "bulk_vertices"
    assert fw_ctx.get("namespace") == "ns-bulk"
    assert captured.get("op_ctx") is None


@pytest.mark.asyncio
async def test_abulk_vertices_builds_raw_request_and_calls_translator_async(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    abulk_vertices() should mirror bulk_vertices wiring but via
    translator.arun_bulk_vertices.
    """
    captured: Dict[str, Any] = {}

    translator = _mock_async_translator_with_capture(
        captured,
        method_name="arun_bulk_vertices",
        return_value="bulk-result-async",
    )

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return translator

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
            self.namespace = "ns-abulk"
            self.limit = 7
            self.cursor = None
            self.filter = {"bar": 1}

    spec = DummyBulkSpec()

    result = await client.abulk_vertices(spec)
    assert result == "bulk-result-async"

    assert "args" in captured
    raw = captured["args"][0]
    assert raw == {
        "namespace": "ns-abulk",
        "limit": 7,
        "cursor": None,
        "filter": {"bar": 1},
    }

    fw_ctx = captured.get("framework_ctx", {})
    assert fw_ctx.get("framework") == "llamaindex"
    assert fw_ctx.get("operation") == "bulk_vertices"
    assert fw_ctx.get("namespace") == "ns-abulk"
    assert captured.get("op_ctx") is None


def test_batch_builds_raw_batch_ops_and_calls_translator(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    batch() should:

    - Validate batch operations (stubbed here),
    - Translate BatchOperation-like objects into raw_batch_ops mappings,
    - Call translator.batch with those ops and framework_ctx.
    """
    captured: Dict[str, Any] = {}

    translator = _mock_translator_with_capture(
        captured,
        method_name="batch",
        return_value="batch-result",
    )

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return translator

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    def fake_validate_batch_operations(*_: Any, **__: Any) -> None:
        captured["validated_batch"] = True

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
    assert captured.get("validated_batch") is True

    assert "args" in captured
    raw_ops = captured["args"][0]
    assert raw_ops == [
        {"op": "upsert_nodes", "args": {"id": "1"}},
        {"op": "delete_nodes", "args": {"ids": ["1", "2"]}},
    ]

    fw_ctx = captured.get("framework_ctx", {})
    assert fw_ctx.get("framework") == "llamaindex"
    assert fw_ctx.get("operation") == "batch"
    assert captured.get("op_ctx") is None


@pytest.mark.asyncio
async def test_abatch_builds_raw_batch_ops_and_calls_translator_async(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    abatch() should mirror batch wiring via translator.arun_batch.
    """
    captured: Dict[str, Any] = {}

    translator = _mock_async_translator_with_capture(
        captured,
        method_name="arun_batch",
        return_value="batch-result-async",
    )

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return translator

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    def fake_validate_batch_operations(*_: Any, **__: Any) -> None:
        captured["validated_batch"] = True

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
        DummyBatchOp("upsert_edges", {"id": "e1"}),
        DummyBatchOp("delete_edges", {"ids": ["e1", "e2"]}),
    ]

    result = await client.abatch(ops)
    assert result == "batch-result-async"
    assert captured.get("validated_batch") is True

    assert "args" in captured
    raw_ops = captured["args"][0]
    assert raw_ops == [
        {"op": "upsert_edges", "args": {"id": "e1"}},
        {"op": "delete_edges", "args": {"ids": ["e1", "e2"]}},
    ]

    fw_ctx = captured.get("framework_ctx", {})
    assert fw_ctx.get("framework") == "llamaindex"
    assert fw_ctx.get("operation") == "batch"
    assert captured.get("op_ctx") is None


# ---------------------------------------------------------------------------
# Capabilities / health passthrough (basic + framework_ctx)
# ---------------------------------------------------------------------------


def test_capabilities_and_health_basic(graph_adapter: Any) -> None:
    """
    Capabilities and health should be surfaced as mappings.

    Detailed structure is covered in generic graph contract tests; here we only
    assert that the LlamaIndex adapter normalizes to mapping-like results.
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


def test_health_uses_llamaindex_framework_ctx(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    Health should call translator.health with a framework_ctx that includes
    framework='llamaindex' and operation='health', and framework_version if set.
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def health(self, *, op_ctx: Any = None, framework_ctx: Mapping[str, Any]) -> Any:
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx)
            return {"status": "ok"}

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

    client = _make_client(graph_adapter, framework_version="fw-123")

    health = client.health()
    assert isinstance(health, Mapping)

    fw_ctx = captured.get("framework_ctx", {})
    assert fw_ctx.get("framework") == "llamaindex"
    assert fw_ctx.get("operation") == "health"
    assert fw_ctx.get("framework_version") == "fw-123"


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
    assert captured["kwargs"]["param_map"] == {"k": "v"}
    # GraphStore.query internally calls client.query with namespace param
    # but we didn't capture that here since we stored only raw args/kwargs
    # passed into DummyClient.query via CorpusGraphStore.query/ get() below.

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
