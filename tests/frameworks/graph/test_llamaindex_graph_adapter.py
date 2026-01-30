# tests/frameworks/graph/test_llamaindex_graph_adapter.py

from __future__ import annotations

from collections.abc import AsyncIterator as ABCAsyncIterator
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence

import asyncio
import inspect

import pytest

import corpus_sdk.graph.framework_adapters.llamaindex as llamaindex_adapter_module
from corpus_sdk.graph.framework_adapters.llamaindex import (
    CorpusGraphStore,
    CorpusLlamaIndexGraphClient,
    ErrorCodes,
    LlamaIndexGraphFrameworkTranslator,
)
from corpus_sdk.graph.graph_base import NotSupported, BadRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(adapter: Any, **kwargs: Any) -> CorpusLlamaIndexGraphClient:
    """Construct a CorpusLlamaIndexGraphClient instance from the generic adapter."""
    return CorpusLlamaIndexGraphClient(adapter=adapter, **kwargs)


def _has_llamaindex() -> bool:
    """
    Return True if llama_index is installed and the adapter module was able to
    import the GraphStore base class.
    """
    return getattr(llamaindex_adapter_module, "_LlamaIndexGraphStore", None) is not None


class _DummyOperationContext:
    """
    Minimal OperationContext-like object for tests.

    This is intentionally very small: the adapter uses a conservative structural
    check (attrs + one identifier/serialization surface).
    """

    def __init__(self, *, attrs: Optional[Dict[str, Any]] = None, request_id: str = "req") -> None:
        self.attrs = attrs or {}
        self.request_id = request_id

    def to_dict(self) -> Dict[str, Any]:
        return {"request_id": self.request_id, "attrs": dict(self.attrs)}


class _DummyNoIdContext:
    """
    Context-like object that has attrs but lacks request_id/traceparent/to_dict,
    so it should fail the adapter's stricter structural check.
    """

    def __init__(self) -> None:
        self.attrs = {"k": "v"}


# ---------------------------------------------------------------------------
# Constructor / translator behavior
# ---------------------------------------------------------------------------


def test_default_translator_uses_llamaindex_framework_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
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

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", fake_create_graph_translator)

    client = _make_client(adapter)

    # Trigger lazy translator construction
    _ = client._translator  # noqa: SLF001

    kwargs = captured["kwargs"]
    assert kwargs.get("framework") == "llamaindex"
    translator = kwargs.get("translator")
    assert isinstance(translator, LlamaIndexGraphFrameworkTranslator)


def test_framework_translator_override_is_respected(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
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

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", fake_create_graph_translator)

    client = _make_client(adapter, framework_translator=custom, framework_version="ll-fw-1.2.3")

    _ = client._translator  # noqa: SLF001

    kwargs = captured["kwargs"]
    assert kwargs.get("framework") == "llamaindex"
    assert kwargs.get("translator") is custom


def test_translator_lazy_initialization_is_cached(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    cached_property should ensure create_graph_translator is called exactly once
    per client instance.
    """
    calls: Dict[str, int] = {"n": 0}

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        calls["n"] += 1

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", fake_create_graph_translator)

    client = _make_client(adapter)
    _ = client._translator  # noqa: SLF001
    _ = client._translator  # noqa: SLF001

    assert calls["n"] == 1


def test_constructor_accepts_adapter_only(adapter: Any) -> None:
    """Basic construction should work with adapter passed via 'adapter'."""
    client = CorpusLlamaIndexGraphClient(adapter=adapter)
    assert client is not None


def test_constructor_accepts_graph_adapter_only(adapter: Any) -> None:
    """Basic construction should work with adapter passed via 'graph_adapter' alias."""
    client = CorpusLlamaIndexGraphClient(graph_adapter=adapter)
    assert client is not None


def test_constructor_rejects_both_adapter_and_graph_adapter_when_different() -> None:
    """Passing both adapter and graph_adapter with different objects must raise TypeError."""
    with pytest.raises(TypeError):
        CorpusLlamaIndexGraphClient(adapter=object(), graph_adapter=object())


# ---------------------------------------------------------------------------
# _looks_like_operation_context strictness (direct helper coverage)
# ---------------------------------------------------------------------------


def test_looks_like_operation_context_accepts_attrs_plus_request_id() -> None:
    """attrs + request_id should satisfy the structural check."""
    assert llamaindex_adapter_module._looks_like_operation_context(_DummyOperationContext()) is True  # type: ignore[attr-defined]


def test_looks_like_operation_context_accepts_attrs_plus_to_dict() -> None:
    """attrs + to_dict should satisfy the structural check even without request_id attr."""
    class Ctx:
        def __init__(self) -> None:
            self.attrs = {"x": 1}

        def to_dict(self) -> Dict[str, Any]:
            return {"x": 1}

    assert llamaindex_adapter_module._looks_like_operation_context(Ctx()) is True  # type: ignore[attr-defined]


def test_looks_like_operation_context_rejects_attrs_only() -> None:
    """attrs alone should be rejected to avoid false positives."""
    assert llamaindex_adapter_module._looks_like_operation_context(_DummyNoIdContext()) is False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Context translation / core_ctx_from_llamaindex mapping
# ---------------------------------------------------------------------------


def test_build_ctx_none_inputs_returns_none(adapter: Any) -> None:
    """_build_ctx should return None when callback_manager and extra_context are empty."""
    client = _make_client(adapter)
    ctx = client._build_ctx(callback_manager=None, extra_context=None)  # noqa: SLF001
    assert ctx is None


def test_build_ctx_passes_callback_manager_and_extra_context_to_core_ctx(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Verify that callback_manager and extra_context are passed through to
    core_ctx_from_llamaindex with the configured framework_version.
    """
    captured: Dict[str, Any] = {}

    # Make OperationContext be our dummy so isinstance() check always succeeds if used.
    monkeypatch.setattr(llamaindex_adapter_module, "OperationContext", _DummyOperationContext)

    def fake_core_ctx_from_llamaindex(
        callback_manager: Any,
        *,
        framework_version: Any = None,
        **extra: Any,
    ) -> Any:
        captured["callback_manager"] = callback_manager
        captured["framework_version"] = framework_version
        captured["extra"] = extra
        return _DummyOperationContext()

    monkeypatch.setattr(llamaindex_adapter_module, "core_ctx_from_llamaindex", fake_core_ctx_from_llamaindex)

    # Translator stub: must have query() since we’ll call client.query().
    class DummyTranslator:
        def query(self, *_: Any, **__: Any) -> Any:
            # Return a value; tests patch validate_graph_result_type where needed elsewhere,
            # but here we allow the default validator to run in the repo environment.
            return llamaindex_adapter_module.QueryResult(rows=[])  # type: ignore[attr-defined]

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter, framework_version="llamaindex-test-version")

    cb_manager = object()
    extra_ctx = {"request_id": "req-xyz", "tenant": "tenant-1"}

    _ = client.query("MATCH (n) RETURN n LIMIT 1", callback_manager=cb_manager, extra_context=extra_ctx)

    assert captured.get("callback_manager") is cb_manager
    assert captured.get("framework_version") == "llamaindex-test-version"
    assert captured.get("extra") == extra_ctx


def test_build_ctx_failure_attaches_context_and_query_proceeds_without_ctx(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If core_ctx_from_llamaindex raises, _build_ctx should:
    - call attach_context with framework='llamaindex' and operation='context_translation'
    - return None (best-effort)
    - allow the graph call to proceed with op_ctx=None
    """
    captured_ctx: Dict[str, Any] = {}
    captured_op_ctx: Dict[str, Any] = {}

    def fake_core_ctx_from_llamaindex(*_: Any, **__: Any) -> Any:
        raise RuntimeError("boom!")

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        # We only capture context; the exception type/message is not asserted.
        assert isinstance(exc, BaseException)
        captured_ctx.update(ctx)

    class DummyTranslator:
        def query(self, raw: Any, *, op_ctx: Any = None, framework_ctx: Any = None, mmr_config: Any = None) -> Any:  # noqa: ARG002
            captured_op_ctx["op_ctx"] = op_ctx
            return llamaindex_adapter_module.QueryResult(rows=[])  # type: ignore[attr-defined]

    monkeypatch.setattr(llamaindex_adapter_module, "core_ctx_from_llamaindex", fake_core_ctx_from_llamaindex)
    monkeypatch.setattr(llamaindex_adapter_module, "attach_context", fake_attach_context)
    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)

    # Should NOT raise: proceeds without OperationContext.
    _ = client.query("MATCH (n) RETURN n", callback_manager=object(), extra_context={"x": 1})

    assert captured_ctx.get("framework") == "llamaindex"
    assert captured_ctx.get("operation") == "context_translation"
    assert captured_ctx.get("error_code") == ErrorCodes.BAD_OPERATION_CONTEXT
    assert "extra_context_keys" in captured_ctx

    assert captured_op_ctx.get("op_ctx") is None


def test_build_ctx_non_operation_context_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If core_ctx_from_llamaindex returns something that does not look like an OperationContext,
    _build_ctx should return None.
    """
    def fake_core_ctx_from_llamaindex(*_: Any, **__: Any) -> Any:
        return _DummyNoIdContext()

    monkeypatch.setattr(llamaindex_adapter_module, "core_ctx_from_llamaindex", fake_core_ctx_from_llamaindex)

    client = _make_client(adapter)
    ctx = client._build_ctx(callback_manager={"user": "x"}, extra_context={"k": "v"})  # noqa: SLF001
    assert ctx is None


def test_build_ctx_enriches_attrs_with_framework_and_version(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    When the translated context has attrs, the adapter should enrich attrs with:
    - framework='llamaindex'
    - framework_version when configured
    """
    monkeypatch.setattr(llamaindex_adapter_module, "OperationContext", _DummyOperationContext)

    returned = _DummyOperationContext(attrs={"existing": 1}, request_id="r1")

    def fake_core_ctx_from_llamaindex(*_: Any, **__: Any) -> Any:
        return returned

    monkeypatch.setattr(llamaindex_adapter_module, "core_ctx_from_llamaindex", fake_core_ctx_from_llamaindex)

    client = _make_client(adapter, framework_version="fw-123")

    ctx = client._build_ctx(callback_manager=object(), extra_context={"x": 1})  # noqa: SLF001
    assert ctx is returned
    assert isinstance(ctx.attrs, dict)
    assert ctx.attrs["framework"] == "llamaindex"
    assert ctx.attrs["framework_version"] == "fw-123"
    assert ctx.attrs["existing"] == 1


def test_build_ctx_does_not_clobber_existing_framework_fields(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If attrs already contains framework/framework_version, the adapter should not overwrite them.
    """
    monkeypatch.setattr(llamaindex_adapter_module, "OperationContext", _DummyOperationContext)

    returned = _DummyOperationContext(attrs={"framework": "custom", "framework_version": "custom-v"}, request_id="r2")

    def fake_core_ctx_from_llamaindex(*_: Any, **__: Any) -> Any:
        return returned

    monkeypatch.setattr(llamaindex_adapter_module, "core_ctx_from_llamaindex", fake_core_ctx_from_llamaindex)

    client = _make_client(adapter, framework_version="fw-should-not-overwrite")

    ctx = client._build_ctx(callback_manager=object(), extra_context=None)  # noqa: SLF001
    assert ctx is returned
    assert ctx.attrs["framework"] == "custom"
    assert ctx.attrs["framework_version"] == "custom-v"


# ---------------------------------------------------------------------------
# _build_raw_query and framework_ctx behavior
# ---------------------------------------------------------------------------


def test_build_raw_query_precedence_explicit_values_override_defaults(adapter: Any) -> None:
    """
    Explicit call values should override client defaults for dialect/namespace/timeout.
    """
    client = _make_client(adapter, default_dialect="d0", default_namespace="ns0", default_timeout_ms=123)

    raw = client._build_raw_query(  # noqa: SLF001
        "Q",
        params={"a": 1},
        dialect="d1",
        namespace="ns1",
        timeout_ms=999,
        stream=False,
    )
    assert raw["text"] == "Q"
    assert raw["dialect"] == "d1"
    assert raw["namespace"] == "ns1"
    assert raw["timeout_ms"] == 999
    assert raw["params"] == {"a": 1}
    assert raw["stream"] is False


def test_build_raw_query_uses_defaults_when_not_provided(adapter: Any) -> None:
    """When dialect/namespace/timeout are not provided, client defaults apply."""
    client = _make_client(adapter, default_dialect="d0", default_namespace="ns0", default_timeout_ms=123)

    raw = client._build_raw_query("Q", params=None, dialect=None, namespace=None, timeout_ms=None, stream=False)  # noqa: SLF001
    assert raw["dialect"] == "d0"
    assert raw["namespace"] == "ns0"
    assert raw["timeout_ms"] == 123


def test_build_raw_query_stream_flag_is_set(adapter: Any) -> None:
    """The stream flag must match the call's stream argument."""
    client = _make_client(adapter)
    raw = client._build_raw_query("Q", params=None, dialect=None, namespace=None, timeout_ms=None, stream=True)  # noqa: SLF001
    assert raw["stream"] is True


def test_build_raw_query_non_json_params_emit_debug_log(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Non-JSON-serializable params should not raise, but should emit a debug log.
    """
    client = _make_client(adapter)

    class NotJSON:
        pass

    debug_calls: List[str] = []

    def fake_debug(msg: str, *args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        debug_calls.append(msg)

    monkeypatch.setattr(llamaindex_adapter_module.logger, "debug", fake_debug)

    raw = client._build_raw_query("Q", params={"x": NotJSON()}, dialect=None, namespace=None, timeout_ms=None, stream=False)  # noqa: SLF001
    assert raw["params"].keys() == {"x"}
    assert debug_calls  # at least one debug log emitted


def test_framework_ctx_namespace_override(adapter: Any) -> None:
    """_framework_ctx should use an explicitly supplied namespace over the client default."""
    client = _make_client(adapter, default_namespace="ns-default")
    fw = client._framework_ctx(operation="op", namespace="ns-explicit")  # noqa: SLF001
    assert fw["framework"] == "llamaindex"
    assert fw["operation"] == "op"
    assert fw["namespace"] == "ns-explicit"


def test_framework_ctx_uses_default_namespace(adapter: Any) -> None:
    """_framework_ctx should use the client default namespace if no explicit namespace is supplied."""
    client = _make_client(adapter, default_namespace="ns-default")
    fw = client._framework_ctx(operation="op", namespace=None)  # noqa: SLF001
    assert fw["namespace"] == "ns-default"


def test_framework_ctx_omits_namespace_when_none(adapter: Any) -> None:
    """If neither explicit namespace nor client default are set, 'namespace' should be omitted."""
    client = _make_client(adapter, default_namespace=None)
    fw = client._framework_ctx(operation="op", namespace=None)  # noqa: SLF001
    assert "namespace" not in fw


# ---------------------------------------------------------------------------
# Capabilities kwargs forwarding behavior (forward compatibility)
# ---------------------------------------------------------------------------


def test_capabilities_forwards_kwargs_if_supported(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If translator.capabilities supports kwargs, the adapter should forward them.
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def capabilities(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return {"ok": True}

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)

    caps = client.capabilities(foo=1, bar="x")
    assert isinstance(caps, Mapping)
    assert captured == {"foo": 1, "bar": "x"}


def test_capabilities_ignores_kwargs_if_not_supported(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If translator.capabilities does NOT support kwargs, the adapter should ignore kwargs.
    """
    class DummyTranslator:
        # No kwargs accepted
        def capabilities(self) -> Any:
            return {"ok": True}

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)

    # Should not raise, even though kwargs are provided.
    caps = client.capabilities(foo=1)
    assert isinstance(caps, Mapping)


@pytest.mark.asyncio
async def test_acapabilities_forwards_kwargs_if_supported(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_capabilities(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return {"ok": True}

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    caps = await client.acapabilities(foo=2)
    assert isinstance(caps, Mapping)
    assert captured == {"foo": 2}


@pytest.mark.asyncio
async def test_acapabilities_ignores_kwargs_if_not_supported(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    class DummyTranslator:
        async def arun_capabilities(self) -> Any:
            return {"ok": True}

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    caps = await client.acapabilities(foo=2)
    assert isinstance(caps, Mapping)


# ---------------------------------------------------------------------------
# Sync wrappers must raise inside an active asyncio loop (deadlock guard)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_query_raises_in_running_event_loop(adapter: Any) -> None:
    client = _make_client(adapter)
    with pytest.raises(RuntimeError, match=ErrorCodes.SYNC_WRAPPER_CALLED_IN_EVENT_LOOP):
        client.query("MATCH (n) RETURN n LIMIT 1")


@pytest.mark.asyncio
async def test_sync_capabilities_raises_in_running_event_loop(adapter: Any) -> None:
    client = _make_client(adapter)
    with pytest.raises(RuntimeError, match=ErrorCodes.SYNC_WRAPPER_CALLED_IN_EVENT_LOOP):
        client.capabilities()


@pytest.mark.asyncio
async def test_sync_upsert_nodes_raises_in_running_event_loop(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    # Avoid validator complexity: patch validate_upsert_nodes_spec to no-op.
    monkeypatch.setattr(llamaindex_adapter_module, "validate_upsert_nodes_spec", lambda *_a, **_k: None)

    class Spec:
        nodes: List[Any] = []
        namespace: Optional[str] = None

    client = _make_client(adapter)
    with pytest.raises(RuntimeError, match=ErrorCodes.SYNC_WRAPPER_CALLED_IN_EVENT_LOOP):
        client.upsert_nodes(Spec())


# ---------------------------------------------------------------------------
# GraphStore integration (optional dependency) – must never fail the suite
# ---------------------------------------------------------------------------


def test_corpus_graph_store_import_behavior_and_basic_wiring() -> None:
    """
    This single test is written to pass in BOTH environments:

    - If llama_index is not installed:
        CorpusGraphStore is a stub that raises ImportError on construction.
    - If llama_index is installed:
        CorpusGraphStore should delegate to the underlying client with the configured namespace.
    """
    if not _has_llamaindex():
        with pytest.raises(ImportError):
            CorpusGraphStore(client=object())  # type: ignore[call-arg]
        return

    # Real GraphStore case: use a minimal dummy client with a query() method.
    captured: Dict[str, Any] = {}

    class DummyClient:
        def query(self, query: str, *, params: Any = None, namespace: Any = None, **_: Any) -> Any:
            captured["query"] = query
            captured["params"] = params
            captured["namespace"] = namespace

            class Result:
                rows = [["s", "r", "o"]]

            return Result()

        def get_schema(self) -> Any:
            return {"schema": "ok"}

    store = CorpusGraphStore(client=DummyClient(), namespace="ns", get_query="GETQ")  # type: ignore[call-arg]
    rows = store.get("subj1")
    assert rows == [["s", "r", "o"]]
    assert captured["query"] == "GETQ"
    assert captured["params"] == {"subj": "subj1"}
    assert captured["namespace"] == "ns"


# ---------------------------------------------------------------------------
# Query dialect fallback on NotSupported (sync + async)
# ---------------------------------------------------------------------------


def test_query_retries_without_dialect_on_NotSupported_when_dialect_explicit(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If translator rejects the dialect (NotSupported) AND the caller explicitly passed dialect,
    adapter should retry without dialect.
    """
    calls: List[Dict[str, Any]] = []

    class DummyTranslator:
        def query(self, raw: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None, mmr_config: Any = None) -> Any:  # noqa: ARG002
            calls.append(dict(raw))
            if "dialect" in raw:
                raise NotSupported("no dialect")
            return llamaindex_adapter_module.QueryResult(rows=[])  # type: ignore[attr-defined]

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    _ = client.query("MATCH (n) RETURN n", dialect="cypher")

    assert len(calls) == 2
    assert "dialect" in calls[0]
    assert "dialect" not in calls[1]


def test_query_does_not_retry_without_dialect_when_dialect_not_explicit(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If the dialect was NOT explicitly passed (dialect=None), the adapter preserves behavior
    and does not retry on NotSupported.
    """
    class DummyTranslator:
        def query(self, raw: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None, mmr_config: Any = None) -> Any:  # noqa: ARG002
            raise NotSupported("no dialect")

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter, default_dialect="cypher-default")

    # dialect=None (caller did not explicitly provide), so no retry.
    with pytest.raises(NotSupported):
        _ = client.query("MATCH (n) RETURN n", dialect=None)


@pytest.mark.asyncio
async def test_aquery_retries_without_dialect_on_NotSupported_when_dialect_explicit(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    calls: List[Dict[str, Any]] = []

    class DummyTranslator:
        async def arun_query(self, raw: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None, mmr_config: Any = None) -> Any:  # noqa: ARG002
            calls.append(dict(raw))
            if "dialect" in raw:
                raise NotSupported("no dialect")
            return llamaindex_adapter_module.QueryResult(rows=[])  # type: ignore[attr-defined]

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    _ = await client.aquery("MATCH (n) RETURN n", dialect="cypher")

    assert len(calls) == 2
    assert "dialect" in calls[0]
    assert "dialect" not in calls[1]


@pytest.mark.asyncio
async def test_aquery_does_not_retry_without_dialect_when_dialect_not_explicit(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    class DummyTranslator:
        async def arun_query(self, raw: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None, mmr_config: Any = None) -> Any:  # noqa: ARG002
            raise NotSupported("no dialect")

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter, default_dialect="cypher-default")
    with pytest.raises(NotSupported):
        await client.aquery("MATCH (n) RETURN n", dialect=None)


# ---------------------------------------------------------------------------
# Query wiring: namespace precedence and op_ctx pass-through
# ---------------------------------------------------------------------------


def test_query_passes_framework_ctx_namespace_when_explicit(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def query(self, raw: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None, mmr_config: Any = None) -> Any:  # noqa: ARG002
            captured["raw"] = dict(raw)
            captured["framework_ctx"] = dict(framework_ctx or {})
            return llamaindex_adapter_module.QueryResult(rows=[])  # type: ignore[attr-defined]

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter, default_namespace="ns-default")
    _ = client.query("MATCH (n) RETURN n", namespace="ns-explicit")

    assert captured["raw"]["namespace"] == "ns-explicit"
    assert captured["framework_ctx"]["namespace"] == "ns-explicit"
    assert captured["framework_ctx"]["framework"] == "llamaindex"
    assert captured["framework_ctx"]["operation"] == "query"


def test_query_passes_op_ctx_from_build_ctx_to_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    # Ensure the adapter accepts our dummy context.
    monkeypatch.setattr(llamaindex_adapter_module, "OperationContext", _DummyOperationContext)

    def fake_core_ctx_from_llamaindex(*_: Any, **__: Any) -> Any:
        return _DummyOperationContext(attrs={"k": "v"}, request_id="req-opctx")

    monkeypatch.setattr(llamaindex_adapter_module, "core_ctx_from_llamaindex", fake_core_ctx_from_llamaindex)

    class DummyTranslator:
        def query(self, raw: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None, mmr_config: Any = None) -> Any:  # noqa: ARG002
            captured["op_ctx"] = op_ctx
            return llamaindex_adapter_module.QueryResult(rows=[])  # type: ignore[attr-defined]

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter, framework_version="fw-x")
    _ = client.query("MATCH (n) RETURN n", callback_manager={"cb": 1}, extra_context={"x": 2})

    ctx = captured.get("op_ctx")
    assert ctx is not None
    assert getattr(ctx, "request_id", None) == "req-opctx"
    assert isinstance(getattr(ctx, "attrs", None), dict)
    assert ctx.attrs["framework"] == "llamaindex"
    assert ctx.attrs["framework_version"] == "fw-x"


# ---------------------------------------------------------------------------
# Streaming: invalid chunk validation + astream normalization
# ---------------------------------------------------------------------------


def test_stream_query_invalid_chunk_triggers_validation_and_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If the translator yields an invalid chunk type, validate_graph_result_type should raise,
    and attach_context should capture llamaindex framework context (via the decorator wrapper).
    """
    captured_ctx: Dict[str, Any] = {}

    class DummyTranslator:
        def query_stream(self, *_: Any, **__: Any):
            yield "not-a-chunk"

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured_ctx.update(ctx)

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        raise TypeError(f"Bad chunk type: {type(result).__name__}")

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(llamaindex_adapter_module, "attach_context", fake_attach_context)
    monkeypatch.setattr(llamaindex_adapter_module, "validate_graph_result_type", fake_validate_graph_result_type)

    client = _make_client(adapter)

    with pytest.raises(TypeError, match="Bad chunk type: str"):
        list(client.stream_query("MATCH (n) RETURN n"))

    # The decorator should have attached useful context.
    assert captured_ctx.get("framework") == "llamaindex"
    assert "operation" in captured_ctx


@pytest.mark.asyncio
async def test_astream_query_invalid_chunk_triggers_validation_and_context_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured_ctx: Dict[str, Any] = {}

    async def gen() -> ABCAsyncIterator[Any]:
        yield "not-a-chunk"

    class DummyTranslator:
        def arun_query_stream(self, *_: Any, **__: Any) -> Any:
            return gen()

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured_ctx.update(ctx)

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        raise TypeError(f"Bad chunk type: {type(result).__name__}")

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(llamaindex_adapter_module, "attach_context", fake_attach_context)
    monkeypatch.setattr(llamaindex_adapter_module, "validate_graph_result_type", fake_validate_graph_result_type)

    client = _make_client(adapter)

    with pytest.raises(TypeError, match="Bad chunk type: str"):
        aiter = client.astream_query("MATCH (n) RETURN n")
        # astream_query always returns an async iterator (itself), so no await needed here.
        async for _ in aiter:
            break

    assert captured_ctx.get("framework") == "llamaindex"
    assert "operation" in captured_ctx


@pytest.mark.asyncio
async def test_astream_query_accepts_direct_async_iterator_from_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    async def gen():
        yield llamaindex_adapter_module.QueryChunk(type="data", data={"x": 1})  # type: ignore[attr-defined]

    class DummyTranslator:
        def arun_query_stream(self, *_: Any, **__: Any) -> Any:
            return gen()

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    # Validate should accept the chunk; allow real validator.

    client = _make_client(adapter)

    seen = False
    async for _ch in client.astream_query("MATCH (n) RETURN n"):
        seen = True
        break

    assert seen is True


@pytest.mark.asyncio
async def test_astream_query_accepts_awaitable_resolving_to_async_iterator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    async def gen():
        yield llamaindex_adapter_module.QueryChunk(type="data", data={"x": 1})  # type: ignore[attr-defined]

    async def resolve():
        return gen()

    class DummyTranslator:
        def arun_query_stream(self, *_: Any, **__: Any) -> Any:
            return resolve()

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)

    seen = False
    async for _ch in client.astream_query("MATCH (n) RETURN n"):
        seen = True
        break

    assert seen is True


@pytest.mark.asyncio
async def test_astream_query_bad_shape_raises_type_error_with_error_code(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    class DummyTranslator:
        def arun_query_stream(self, *_: Any, **__: Any) -> Any:
            return ["not", "an", "async", "iterator"]

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)

    with pytest.raises(TypeError, match=ErrorCodes.BAD_ASYNC_ITERATOR_SHAPE):
        aiter = client.astream_query("MATCH (n) RETURN n")
        async for _ in aiter:
            break


def test_stream_query_builds_raw_query_with_stream_true_and_namespace_precedence(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def query_stream(self, raw: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None):  # noqa: ARG002
            captured["raw"] = dict(raw)
            captured["framework_ctx"] = dict(framework_ctx or {})
            # Yield no chunks (valid: empty stream)
            if False:
                yield None

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter, default_namespace="ns-default")
    list(client.stream_query("MATCH (n) RETURN n", namespace="ns-explicit"))

    assert captured["raw"]["stream"] is True
    assert captured["raw"]["namespace"] == "ns-explicit"
    assert captured["framework_ctx"]["namespace"] == "ns-explicit"


# ---------------------------------------------------------------------------
# Upsert: namespace precedence and edge validation (sync + async)
# ---------------------------------------------------------------------------


def test_upsert_nodes_uses_spec_namespace_for_framework_ctx(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    # Avoid depending on UpsertNodesSpec concrete validation details.
    monkeypatch.setattr(llamaindex_adapter_module, "validate_upsert_nodes_spec", lambda *_a, **_k: None)

    class DummyTranslator:
        def upsert_nodes(self, nodes: Any, *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:  # noqa: ARG002
            captured["framework_ctx"] = dict(framework_ctx or {})
            return llamaindex_adapter_module.UpsertResult(ok=True)  # type: ignore[attr-defined]

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    class Spec:
        nodes = [{"id": "n1"}]
        namespace = "ns-spec"

    client = _make_client(adapter, default_namespace="ns-default")
    _ = client.upsert_nodes(Spec())

    assert captured["framework_ctx"]["namespace"] == "ns-spec"
    assert captured["framework_ctx"]["operation"] == "upsert_nodes"


def test_upsert_edges_validates_edges_and_does_not_mutate_spec_edges(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class Edge:
        def __init__(self) -> None:
            self.id = "e1"
            self.src = "s"
            self.dst = "d"
            self.label = "L"
            self.properties = {"k": "v"}

    edges_source = (Edge(),)
    edges_copy_before = list(edges_source)

    class Spec:
        edges = edges_source
        namespace = "ns-spec"

    class DummyTranslator:
        def upsert_edges(self, edges: Any, *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:  # noqa: ARG002
            captured["edges"] = edges
            captured["framework_ctx"] = dict(framework_ctx or {})
            return llamaindex_adapter_module.UpsertResult(ok=True)  # type: ignore[attr-defined]

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    spec = Spec()
    _ = client.upsert_edges(spec)

    # Translator receives a materialized list.
    assert isinstance(captured["edges"], list)
    assert len(captured["edges"]) == 1
    # Spec.edges must remain the original iterable (no mutation).
    assert spec.edges is edges_source
    assert list(spec.edges) == edges_copy_before
    assert captured["framework_ctx"]["operation"] == "upsert_edges"


@pytest.mark.asyncio
async def test_aupsert_edges_validates_edges_and_does_not_mutate_spec_edges_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class Edge:
        def __init__(self) -> None:
            self.id = "e1"
            self.src = "s"
            self.dst = "d"
            self.label = "L"
            self.properties = {"k": "v"}

    edges_source = (Edge(),)
    edges_copy_before = list(edges_source)

    class Spec:
        edges = edges_source
        namespace = "ns-spec"

    class DummyTranslator:
        async def arun_upsert_edges(self, edges: Any, *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:  # noqa: ARG002
            captured["edges"] = edges
            captured["framework_ctx"] = dict(framework_ctx or {})
            return llamaindex_adapter_module.UpsertResult(ok=True)  # type: ignore[attr-defined]

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    spec = Spec()
    _ = await client.aupsert_edges(spec)

    assert isinstance(captured["edges"], list)
    assert spec.edges is edges_source
    assert list(spec.edges) == edges_copy_before
    assert captured["framework_ctx"]["operation"] == "upsert_edges"


def test_upsert_edges_invalid_edges_none_raises_badrequest(adapter: Any) -> None:
    class Spec:
        edges = None
        namespace = None

    client = _make_client(adapter)
    with pytest.raises(BadRequest) as exc_info:
        client.upsert_edges(Spec())  # type: ignore[arg-type]
    assert getattr(exc_info.value, "code", None) == ErrorCodes.BAD_ADAPTER_RESULT


def test_upsert_edges_invalid_missing_required_field_raises_badrequest(adapter: Any) -> None:
    class Edge:
        id = "e1"
        src = "s"
        dst = "d"
        label = None  # missing

    class Spec:
        edges = [Edge()]
        namespace = None

    client = _make_client(adapter)
    with pytest.raises(BadRequest):
        client.upsert_edges(Spec())  # type: ignore[arg-type]


def test_upsert_edges_invalid_properties_not_json_serializable_raises_badrequest(adapter: Any) -> None:
    class NotJSON:
        pass

    class Edge:
        id = "e1"
        src = "s"
        dst = "d"
        label = "L"
        properties = {"x": NotJSON()}

    class Spec:
        edges = [Edge()]
        namespace = None

    client = _make_client(adapter)
    with pytest.raises(BadRequest):
        client.upsert_edges(Spec())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Delete selection behavior + delete_edges regression coverage
# ---------------------------------------------------------------------------


def test_delete_nodes_requires_filter_or_ids_sync(adapter: Any) -> None:
    class Spec:
        filter = None
        ids = []
        namespace = None

    client = _make_client(adapter)
    with pytest.raises(BadRequest) as exc_info:
        client.delete_nodes(Spec())  # type: ignore[arg-type]
    assert getattr(exc_info.value, "code", None) == ErrorCodes.BAD_ADAPTER_RESULT
    assert ErrorCodes.INVALID_DELETE_SPEC in str(exc_info.value)


def test_delete_edges_uses_filter_precedence_over_ids_and_passes_filter_to_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def delete_edges(self, raw: Any, *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:  # noqa: ARG002
            captured["raw"] = raw
            return llamaindex_adapter_module.DeleteResult(ok=True)  # type: ignore[attr-defined]

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    class Spec:
        filter = {"where": "x"}
        ids = ["e1"]
        namespace = None

    client = _make_client(adapter)
    _ = client.delete_edges(Spec())  # type: ignore[arg-type]
    assert captured["raw"] == {"where": "x"}


def test_delete_edges_uses_ids_when_filter_is_none(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def delete_edges(self, raw: Any, *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:  # noqa: ARG002
            captured["raw"] = raw
            return llamaindex_adapter_module.DeleteResult(ok=True)  # type: ignore[attr-defined]

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    class Spec:
        filter = None
        ids = ["e1", "e2"]
        namespace = None

    client = _make_client(adapter)
    _ = client.delete_edges(Spec())  # type: ignore[arg-type]
    assert captured["raw"] == ["e1", "e2"]


# ---------------------------------------------------------------------------
# Bulk vertices + traversal wiring (sync + async)
# ---------------------------------------------------------------------------


def test_bulk_vertices_builds_raw_request_and_calls_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def bulk_vertices(self, raw: Any, *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:  # noqa: ARG002
            captured["raw"] = dict(raw)
            captured["framework_ctx"] = dict(framework_ctx or {})
            return llamaindex_adapter_module.BulkVerticesResult(items=[], cursor=None)  # type: ignore[attr-defined]

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    class Spec:
        namespace = "ns-bulk"
        limit = 7
        cursor = "c1"
        filter = {"x": 1}

    client = _make_client(adapter)
    _ = client.bulk_vertices(Spec())  # type: ignore[arg-type]

    assert captured["raw"] == {"namespace": "ns-bulk", "limit": 7, "cursor": "c1", "filter": {"x": 1}}
    assert captured["framework_ctx"]["operation"] == "bulk_vertices"
    assert captured["framework_ctx"]["namespace"] == "ns-bulk"


@pytest.mark.asyncio
async def test_abulk_vertices_builds_raw_request_and_calls_translator_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_bulk_vertices(self, raw: Any, *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:  # noqa: ARG002
            captured["raw"] = dict(raw)
            captured["framework_ctx"] = dict(framework_ctx or {})
            return llamaindex_adapter_module.BulkVerticesResult(items=[], cursor=None)  # type: ignore[attr-defined]

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    class Spec:
        namespace = "ns-bulk"
        limit = 7
        cursor = None
        filter = None

    client = _make_client(adapter)
    _ = await client.abulk_vertices(Spec())  # type: ignore[arg-type]

    assert captured["raw"] == {"namespace": "ns-bulk", "limit": 7, "cursor": None, "filter": None}
    assert captured["framework_ctx"]["operation"] == "bulk_vertices"
    assert captured["framework_ctx"]["namespace"] == "ns-bulk"


def test_traversal_builds_raw_request_and_validates_result(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    traversal() should build the traversal request mapping and pass it to translator.traversal,
    then validate the returned TraversalResult.
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def traversal(self, raw: Any, *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:  # noqa: ARG002
            captured["raw"] = dict(raw)
            captured["framework_ctx"] = dict(framework_ctx or {})
            return "TRAVERSAL_RESULT"

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        # For this test, treat the translator output as already valid.
        return result

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(llamaindex_adapter_module, "validate_graph_result_type", fake_validate_graph_result_type)

    class Spec:
        start_nodes = ["n1"]
        max_depth = 2
        direction = "out"
        relationship_types = ["R"]
        node_filters = None
        relationship_filters = None
        return_properties = None
        namespace = "ns-trav"

    client = _make_client(adapter)
    out = client.traversal(Spec())  # type: ignore[arg-type]
    assert out == "TRAVERSAL_RESULT"
    assert captured["raw"]["start_nodes"] == ["n1"]
    assert captured["raw"]["namespace"] == "ns-trav"
    assert captured["framework_ctx"]["operation"] == "traversal"
    assert captured["framework_ctx"]["namespace"] == "ns-trav"


@pytest.mark.asyncio
async def test_atraversal_builds_raw_request_and_validates_result_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_traversal(self, raw: Any, *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:  # noqa: ARG002
            captured["raw"] = dict(raw)
            captured["framework_ctx"] = dict(framework_ctx or {})
            return "TRAVERSAL_RESULT"

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(llamaindex_adapter_module, "validate_graph_result_type", fake_validate_graph_result_type)

    class Spec:
        start_nodes = ["n1", "n2"]
        max_depth = 3
        direction = "in"
        relationship_types = None
        node_filters = {"k": "v"}
        relationship_filters = None
        return_properties = ["id"]
        namespace = "ns-trav"

    client = _make_client(adapter)
    out = await client.atraversal(Spec())  # type: ignore[arg-type]
    assert out == "TRAVERSAL_RESULT"
    assert captured["raw"]["start_nodes"] == ["n1", "n2"]
    assert captured["raw"]["max_depth"] == 3
    assert captured["framework_ctx"]["namespace"] == "ns-trav"


# ---------------------------------------------------------------------------
# Batch + Transaction wiring (sync + async where present)
# ---------------------------------------------------------------------------


def test_batch_builds_raw_ops_calls_translator_and_validates(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def batch(self, raw_ops: Any, *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:  # noqa: ARG002
            captured["raw_ops"] = raw_ops
            captured["framework_ctx"] = dict(framework_ctx or {})
            return "BATCH_RESULT"

    def fake_validate_batch_operations(ops: Any, *, operation: str, error_code: str) -> None:
        captured["validated"] = True
        captured["operation"] = operation
        captured["error_code"] = error_code

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(llamaindex_adapter_module, "validate_batch_operations", fake_validate_batch_operations)
    monkeypatch.setattr(llamaindex_adapter_module, "validate_graph_result_type", fake_validate_graph_result_type)

    class Op:
        def __init__(self, op: str, args: Dict[str, Any]) -> None:
            self.op = op
            self.args = args

    ops = [Op("x", {"a": 1}), Op("y", {"b": 2})]

    client = _make_client(adapter)
    out = client.batch(ops)  # type: ignore[arg-type]
    assert out == "BATCH_RESULT"

    assert captured["validated"] is True
    assert captured["operation"] == "batch"
    assert captured["error_code"] == ErrorCodes.INVALID_BATCH_OPS
    assert captured["raw_ops"] == [{"op": "x", "args": {"a": 1}}, {"op": "y", "args": {"b": 2}}]
    assert captured["framework_ctx"]["operation"] == "batch"


@pytest.mark.asyncio
async def test_abatch_builds_raw_ops_calls_translator_and_validates_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_batch(self, raw_ops: Any, *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:  # noqa: ARG002
            captured["raw_ops"] = raw_ops
            captured["framework_ctx"] = dict(framework_ctx or {})
            return "BATCH_RESULT"

    def fake_validate_batch_operations(ops: Any, *, operation: str, error_code: str) -> None:
        captured["validated"] = True
        captured["operation"] = operation
        captured["error_code"] = error_code

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(llamaindex_adapter_module, "validate_batch_operations", fake_validate_batch_operations)
    monkeypatch.setattr(llamaindex_adapter_module, "validate_graph_result_type", fake_validate_graph_result_type)

    class Op:
        def __init__(self, op: str, args: Dict[str, Any]) -> None:
            self.op = op
            self.args = args

    ops = [Op("x", {"a": 1})]

    client = _make_client(adapter)
    out = await client.abatch(ops)  # type: ignore[arg-type]
    assert out == "BATCH_RESULT"

    assert captured["validated"] is True
    assert captured["operation"] == "abatch"
    assert captured["error_code"] == ErrorCodes.INVALID_BATCH_OPS
    assert captured["raw_ops"] == [{"op": "x", "args": {"a": 1}}]
    assert captured["framework_ctx"]["operation"] == "batch"


def test_transaction_builds_raw_ops_calls_translator_and_validates(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def transaction(self, raw_ops: Any, *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:  # noqa: ARG002
            captured["raw_ops"] = raw_ops
            captured["framework_ctx"] = dict(framework_ctx or {})
            return "TX_RESULT"

    def fake_validate_batch_operations(ops: Any, *, operation: str, error_code: str) -> None:
        captured["validated"] = True
        captured["operation"] = operation
        captured["error_code"] = error_code

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    monkeypatch.setattr(llamaindex_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(llamaindex_adapter_module, "validate_batch_operations", fake_validate_batch_operations)
    monkeypatch.setattr(llamaindex_adapter_module, "validate_graph_result_type", fake_validate_graph_result_type)

    class Op:
        def __init__(self, op: str, args: Dict[str, Any]) -> None:
            self.op = op
            self.args = args

    ops = [Op("op1", {"x": 1}), Op("op2", {"y": 2})]

    client = _make_client(adapter)
    out = client.transaction(ops)  # type: ignore[arg-type]
    assert out == "TX_RESULT"

    assert captured["validated"] is True
    assert captured["operation"] == "transaction"
    assert captured["error_code"] == ErrorCodes.INVALID_BATCH_OPS
    assert captured["raw_ops"] == [{"op": "op1", "args": {"x": 1}}, {"op": "op2", "args": {"y": 2}}]
    assert captured["framework_ctx"]["operation"] == "transaction"


# ---------------------------------------------------------------------------
# Test count sanity check (for local development)
# ---------------------------------------------------------------------------

# This file is intentionally designed to be in the 50–55 test range when collected.
# If you add/remove tests, you may want to adjust the suite to preserve parity.

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
