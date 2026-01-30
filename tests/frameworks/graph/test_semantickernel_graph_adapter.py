# tests/frameworks/graph/test_semantickernel_graph_adapter.py

from __future__ import annotations

import asyncio
import builtins
import importlib
import logging
from collections.abc import Mapping
from typing import Any, Dict, Iterator, List, Optional, Tuple

import inspect

import pytest

import corpus_sdk.graph.framework_adapters.semantic_kernel as sk_adapter_module
from corpus_sdk.graph.framework_adapters.semantic_kernel import (
    CorpusSemanticKernelGraphClient,
    CorpusSemanticKernelPlugin,
    ErrorCodes,
    SemanticKernelGraphFrameworkTranslator,
)
from corpus_sdk.graph.graph_base import BadRequest, NotSupported


# ---------------------------------------------------------------------------
# HARD FAIL / HARD PASS ONLY
#
# Semantic Kernel must be installed in the environment for this suite.
# We intentionally do NOT import semantic_kernel here (per repo convention),
# because the adapter module already uses a soft-import pattern. Instead, we
# assert that the adapter successfully detected SK at import time.
# ---------------------------------------------------------------------------

if getattr(sk_adapter_module, "_semantic_kernel", None) is None:
    raise RuntimeError(
        "semantic_kernel was not detected by the Semantic Kernel adapter module. "
        "Install semantic-kernel (and ensure it imports) to run this test suite."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(adapter: Any, **kwargs: Any) -> CorpusSemanticKernelGraphClient:
    """Construct a CorpusSemanticKernelGraphClient instance from the generic adapter."""
    return CorpusSemanticKernelGraphClient(adapter=adapter, **kwargs)


def _mock_translator_with_capture(
    captured: Dict[str, Any],
    method_name: str,
    return_value: Any,
) -> Any:
    """Create a translator-like object whose named method captures args/kwargs."""

    class MockTranslator:
        def __getattr__(self, name: str) -> Any:
            if name == method_name:

                def method(*args: Any, **kwargs: Any) -> Any:
                    captured.setdefault("calls", []).append((name, args, kwargs))
                    captured["last_method"] = name
                    captured["last_args"] = args
                    captured["last_kwargs"] = kwargs
                    return return_value

                return method
            raise AttributeError(name)

    return MockTranslator()


def _mock_async_translator_with_capture(
    captured: Dict[str, Any],
    method_name: str,
    return_value: Any,
) -> Any:
    """Create an async translator-like object whose named method captures args/kwargs."""

    class MockTranslator:
        def __getattr__(self, name: str) -> Any:
            if name == method_name:

                async def method(*args: Any, **kwargs: Any) -> Any:
                    captured.setdefault("calls", []).append((name, args, kwargs))
                    captured["last_method"] = name
                    captured["last_args"] = args
                    captured["last_kwargs"] = kwargs
                    return return_value

                return method
            raise AttributeError(name)

    return MockTranslator()


def _edge_obj(
    *,
    edge_id: Optional[str] = "e1",
    src: Optional[str] = "n1",
    dst: Optional[str] = "n2",
    label: Optional[str] = "REL",
    properties: Any = None,
) -> Any:
    """Create a simple edge-like object with the attributes used by the adapter."""

    class Edge:
        pass

    e = Edge()
    e.id = edge_id
    e.src = src
    e.dst = dst
    e.label = label
    e.properties = properties
    return e


def _make_async_gen(items: List[Any]) -> Any:
    """Return an async generator yielding the provided items."""

    async def gen():
        for it in items:
            yield it

    return gen()


# ---------------------------------------------------------------------------
# 1) Construction & translator wiring (5 tests)
# ---------------------------------------------------------------------------


def test_default_translator_uses_semantickernel_framework_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    By default, CorpusSemanticKernelGraphClient should:
    - Construct a SemanticKernelGraphFrameworkTranslator instance, and
    - Pass it into create_graph_translator with framework="semantic_kernel".
    """
    captured: Dict[str, Any] = {}

    def fake_create_graph_translator(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["kwargs"] = kwargs

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", fake_create_graph_translator)

    client = _make_client(adapter)

    # Trigger lazy translator construction
    _ = client._translator  # noqa: SLF001

    kwargs = captured["kwargs"]
    assert kwargs.get("framework") == "semantic_kernel"
    translator = kwargs.get("translator")
    assert isinstance(translator, SemanticKernelGraphFrameworkTranslator)


def test_framework_translator_override_is_respected(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """If framework_translator is provided, it should be passed through unchanged."""
    captured: Dict[str, Any] = {}

    class CustomTranslator(SemanticKernelGraphFrameworkTranslator):
        pass

    custom = CustomTranslator()

    def fake_create_graph_translator(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["kwargs"] = kwargs

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", fake_create_graph_translator)

    client = _make_client(adapter, framework_translator=custom, framework_version="sk-fw-1.2.3")
    _ = client._translator  # noqa: SLF001

    kwargs = captured["kwargs"]
    assert kwargs.get("framework") == "semantic_kernel"
    assert kwargs.get("translator") is custom


def test_translator_is_cached_property_constructed_once(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """_translator should be cached: repeated access should not rebuild it."""
    calls: Dict[str, int] = {"count": 0}

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        calls["count"] += 1

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", fake_create_graph_translator)

    client = _make_client(adapter)

    _ = client._translator  # noqa: SLF001
    _ = client._translator  # noqa: SLF001

    assert calls["count"] == 1


def test_constructor_rejects_both_adapter_and_graph_adapter(adapter: Any) -> None:
    """Providing both adapter and graph_adapter should raise TypeError."""
    with pytest.raises(TypeError):
        CorpusSemanticKernelGraphClient(adapter=adapter, graph_adapter=adapter)


def test_constructor_requires_adapter() -> None:
    """Providing neither adapter nor graph_adapter should raise TypeError."""
    with pytest.raises(TypeError):
        CorpusSemanticKernelGraphClient()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 2) OperationContext translation behavior (6 tests)
# ---------------------------------------------------------------------------


def test_build_ctx_none_when_all_inputs_none(adapter: Any) -> None:
    """_build_ctx should return None when context/settings/extra_context are all empty."""
    client = _make_client(adapter)
    ctx = client._build_ctx(context=None, settings=None, extra_context=None)  # noqa: SLF001
    assert ctx is None


def test_context_translation_passes_context_settings_extra_and_framework_version(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """Verify context/settings/extra_context are passed through to core_ctx_from_semantic_kernel."""
    captured: Dict[str, Any] = {}

    class DummyOperationContext:
        def __init__(self) -> None:
            self.attrs: Dict[str, Any] = {}
            self.request_id = "req-1"

    # Ensure isinstance-path acceptance by making OperationContext a concrete class.
    monkeypatch.setattr(sk_adapter_module, "OperationContext", DummyOperationContext)

    def fake_core_ctx_from_semantic_kernel(
        context: Any,
        *,
        settings: Any = None,
        framework_version: Any = None,
        **extra: Any,
    ) -> Any:
        captured["context"] = context
        captured["settings"] = settings
        captured["framework_version"] = framework_version
        captured["extra"] = dict(extra)
        return DummyOperationContext()

    monkeypatch.setattr(sk_adapter_module, "core_ctx_from_semantic_kernel", fake_core_ctx_from_semantic_kernel)

    # Use a minimal translator so query proceeds.
    class DummyTranslator:
        def query(self, *_: Any, **__: Any) -> Any:
            return sk_adapter_module.QueryResult(records=[], summary={})  # type: ignore[attr-defined]

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter, framework_version="semantic-kernel-test-version")

    ctx_obj = object()
    settings = {"temperature": 0.3}
    extra_ctx = {"request_id": "req-xyz", "tenant": "tenant-1"}

    _ = client.query("MATCH (n) RETURN n LIMIT 1", context=ctx_obj, settings=settings, extra_context=extra_ctx)

    assert captured["context"] is ctx_obj
    assert captured["settings"] == settings
    assert captured["framework_version"] == "semantic-kernel-test-version"
    assert captured["extra"] == extra_ctx


def test_context_translation_enriches_attrs_when_mutable(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """When ctx.attrs is a dict, framework metadata should be added best-effort."""
    class DummyOperationContext:
        def __init__(self) -> None:
            self.attrs: Dict[str, Any] = {}
            self.request_id = "req-2"

    monkeypatch.setattr(sk_adapter_module, "OperationContext", DummyOperationContext)

    def fake_core_ctx_from_semantic_kernel(*_: Any, **__: Any) -> Any:
        return DummyOperationContext()

    monkeypatch.setattr(sk_adapter_module, "core_ctx_from_semantic_kernel", fake_core_ctx_from_semantic_kernel)

    client = _make_client(adapter, framework_version="v9.9.9")
    ctx = client._build_ctx(context={"x": 1}, settings={"y": 2}, extra_context={"z": 3})  # noqa: SLF001

    assert ctx is not None
    attrs = getattr(ctx, "attrs", {})
    assert attrs.get("framework") == "semantic_kernel"
    assert attrs.get("framework_version") == "v9.9.9"


def test_context_translation_failure_attaches_context_and_proceeds_with_op_ctx_none(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    FIXED ISSUE #1:
    If core_ctx_from_semantic_kernel raises, the adapter should:
      - call attach_context with BAD_OPERATION_CONTEXT, and
      - proceed with op_ctx=None (best-effort translation), not raise solely due to ctx.
    """
    attached: Dict[str, Any] = {}
    captured_call: Dict[str, Any] = {}

    def fake_core_ctx_from_semantic_kernel(*_: Any, **__: Any) -> Any:
        raise RuntimeError("boom")

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        attached.update(ctx)

    class DummyTranslator:
        def query(self, raw_query: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None, mmr_config: Any = None) -> Any:  # noqa: ARG002,E501
            captured_call["op_ctx"] = op_ctx
            captured_call["raw_query"] = dict(raw_query)
            captured_call["framework_ctx"] = dict(framework_ctx or {})
            return sk_adapter_module.QueryResult(records=[], summary={})  # type: ignore[attr-defined]

    monkeypatch.setattr(sk_adapter_module, "core_ctx_from_semantic_kernel", fake_core_ctx_from_semantic_kernel)
    monkeypatch.setattr(sk_adapter_module, "attach_context", fake_attach_context)
    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)

    result = client.query("MATCH (n) RETURN n", context={"user_id": "u1"}, settings={"temp": 0.2})
    assert result is not None

    assert attached
    assert attached.get("framework") == "semantic_kernel"
    assert attached.get("error_code") == ErrorCodes.BAD_OPERATION_CONTEXT
    assert captured_call.get("op_ctx") is None


def test_context_translation_returns_none_if_not_operation_context_like(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """If translation returns a non-context-like object, _build_ctx should return None."""
    def fake_core_ctx_from_semantic_kernel(*_: Any, **__: Any) -> Any:
        # Missing attrs and identifiers => should be rejected.
        return object()

    monkeypatch.setattr(sk_adapter_module, "core_ctx_from_semantic_kernel", fake_core_ctx_from_semantic_kernel)

    client = _make_client(adapter)
    ctx = client._build_ctx(context={"a": 1}, settings={"b": 2}, extra_context={"c": 3})  # noqa: SLF001
    assert ctx is None


def test_operation_context_structural_heuristic_requires_minimum_set() -> None:
    """Structural heuristic should require attrs + at least one identifier/serialization surface."""
    # attrs only => False
    class A:
        attrs = {}

    # attrs + request_id => True
    class B:
        attrs = {}
        request_id = "r"

    assert sk_adapter_module._looks_like_operation_context(A()) is False  # noqa: SLF001
    assert sk_adapter_module._looks_like_operation_context(B()) is True  # noqa: SLF001


# ---------------------------------------------------------------------------
# 3) Event-loop guard (4 tests - parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method_name, call_args",
    [
        ("query", ("MATCH (n) RETURN n",)),
        ("stream_query", ("MATCH (n) RETURN n",)),
        ("bulk_vertices", (object(),)),
        ("batch", ([],)),
    ],
)
async def test_sync_methods_raise_if_called_inside_running_event_loop(
    adapter: Any,
    method_name: str,
    call_args: Tuple[Any, ...],
) -> None:
    """Calling sync wrappers in a running event loop should raise to prevent deadlocks."""
    client = _make_client(adapter)

    method = getattr(client, method_name)
    with pytest.raises(RuntimeError) as exc_info:
        # stream_query returns a generator, so we need to consume it to trigger the check
        if method_name == "stream_query":
            list(method(*call_args))
        else:
            method(*call_args)

    assert ErrorCodes.SYNC_WRAPPER_CALLED_IN_EVENT_LOOP in str(exc_info.value)


# ---------------------------------------------------------------------------
# 4) Capabilities kwargs forwarding (3 tests)
# ---------------------------------------------------------------------------


def test_capabilities_forwards_kwargs_when_supported(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """If translator.capabilities supports kwargs, adapter should forward them."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def capabilities(self, **kwargs: Any) -> Any:
            captured["kwargs"] = dict(kwargs)
            return {"cap": True}

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(sk_adapter_module, "graph_capabilities_to_dict", lambda caps: dict(caps))

    client = _make_client(adapter)

    caps = client.capabilities(foo=1, bar="x")
    assert isinstance(caps, Mapping)
    assert captured["kwargs"] == {"foo": 1, "bar": "x"}


def test_capabilities_ignores_kwargs_when_not_supported(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If translator.capabilities does not accept kwargs, adapter should ignore them safely."""
    caplog.set_level(logging.DEBUG)

    class DummyTranslator:
        def capabilities(self) -> Any:
            return {"cap": True}

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(sk_adapter_module, "graph_capabilities_to_dict", lambda caps: dict(caps))

    client = _make_client(adapter)

    caps = client.capabilities(foo=1)
    assert isinstance(caps, Mapping)
    # Debug log is best-effort; do not make the test brittle on exact message.
    assert any("does not accept kwargs" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_acapabilities_forwards_kwargs_or_ignores_safely(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Async capabilities should:
      - forward kwargs when supported, and
      - ignore kwargs safely when not supported.
    """
    monkeypatch.setattr(sk_adapter_module, "graph_capabilities_to_dict", lambda caps: dict(caps))

    # Supported case
    captured_supported: Dict[str, Any] = {}

    class SupportedTranslator:
        async def arun_capabilities(self, **kwargs: Any) -> Any:
            captured_supported["kwargs"] = dict(kwargs)
            return {"cap": True}

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: SupportedTranslator())
    client1 = _make_client(adapter)
    caps1 = await client1.acapabilities(alpha=123)
    assert isinstance(caps1, Mapping)
    assert captured_supported["kwargs"] == {"alpha": 123}

    # Not supported case (TypeError fallback)
    class NotSupportedKwTranslator:
        async def arun_capabilities(self) -> Any:
            return {"cap": True}

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: NotSupportedKwTranslator())
    client2 = _make_client(adapter)
    caps2 = await client2.acapabilities(beta="x")
    assert isinstance(caps2, Mapping)


# ---------------------------------------------------------------------------
# 5) Query validation + raw mapping + params JSON diagnostics (4 tests)
# ---------------------------------------------------------------------------


def test_validate_graph_query_called_with_constant_sync_and_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """validate_graph_query should be called with ErrorCodes.INVALID_QUERY for both paths."""
    captured: Dict[str, Any] = {"sync": None, "async": None}

    def fake_validate_graph_query(query: str, *, operation: str, error_code: str) -> None:
        if operation == "query":
            captured["sync"] = error_code
        if operation == "aquery":
            captured["async"] = error_code

    class DummyTranslator:
        def query(self, *_a: Any, **_k: Any) -> Any:
            return {"ok": True}

        async def arun_query(self, *_a: Any, **_k: Any) -> Any:
            return {"ok": True}

    monkeypatch.setattr(sk_adapter_module, "validate_graph_query", fake_validate_graph_query)
    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(sk_adapter_module, "validate_graph_result_type", lambda v, **_k: v)

    client = _make_client(adapter)

    _ = client.query("MATCH (n) RETURN n")
    asyncio.run(client.aquery("MATCH (n) RETURN n"))

    assert captured["sync"] == ErrorCodes.INVALID_QUERY
    assert captured["async"] == ErrorCodes.INVALID_QUERY


def test_build_raw_query_precedence_and_stream_flags(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """Raw query should include defaults, allow overrides, and set stream flag correctly."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def query(self, raw_query: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None, mmr_config: Any = None) -> Any:  # noqa: ARG002,E501
            captured["query_raw"] = dict(raw_query)
            return {"ok": True}

        def query_stream(self, raw_query: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None) -> Iterator[Any]:  # noqa: ARG002,E501
            captured["stream_raw"] = dict(raw_query)
            yield {"chunk": 1}

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(sk_adapter_module, "validate_graph_result_type", lambda v, **_k: v)

    client = _make_client(adapter, default_dialect="cypher", default_namespace="ns-default", default_timeout_ms=111)

    _ = client.query("Q", params={"x": 1})
    list(client.stream_query("Q2", namespace="ns-explicit", timeout_ms=222))

    assert captured["query_raw"]["dialect"] == "cypher"
    assert captured["query_raw"]["namespace"] == "ns-default"
    assert captured["query_raw"]["timeout_ms"] == 111
    assert captured["query_raw"]["stream"] is False

    assert "dialect" in captured["stream_raw"]  # inherited default dialect
    assert captured["stream_raw"]["namespace"] == "ns-explicit"
    assert captured["stream_raw"]["timeout_ms"] == 222
    assert captured["stream_raw"]["stream"] is True


def test_params_non_serializable_logs_debug_not_raise(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Non-JSON-serializable params should log debug only and not raise."""
    caplog.set_level(logging.DEBUG)

    class NonJSON:
        pass

    class DummyTranslator:
        def query(self, *_a: Any, **_k: Any) -> Any:
            return {"ok": True}

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(sk_adapter_module, "validate_graph_result_type", lambda v, **_k: v)

    client = _make_client(adapter)

    _ = client.query("Q", params={"x": NonJSON()})
    assert any("not JSON-serializable" in rec.message for rec in caplog.records)


def test_query_passes_framework_ctx_and_op_ctx_none_when_no_context_inputs(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """Framework context should include operation and framework; op_ctx should be None if no inputs."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def query(self, raw_query: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None, mmr_config: Any = None) -> Any:  # noqa: ARG002,E501
            captured["op_ctx"] = op_ctx
            captured["framework_ctx"] = dict(framework_ctx or {})
            captured["raw_query"] = dict(raw_query)
            return {"ok": True}

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(sk_adapter_module, "validate_graph_result_type", lambda v, **_k: v)

    client = _make_client(adapter, default_namespace="ns-0")
    _ = client.query("Q", namespace="ns-1")

    assert captured["op_ctx"] is None
    assert captured["framework_ctx"]["framework"] == "semantic_kernel"
    assert captured["framework_ctx"]["operation"] == "query"
    assert captured["framework_ctx"]["namespace"] == "ns-1"


# ---------------------------------------------------------------------------
# 6) Dialect fallback on NotSupported (4 tests)
# ---------------------------------------------------------------------------


def test_query_retries_without_dialect_on_NotSupported_when_dialect_provided(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """If dialect explicitly provided and NotSupported raised, adapter should retry without dialect."""
    calls: List[Dict[str, Any]] = []

    class DummyTranslator:
        def query(self, raw_query: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None, mmr_config: Any = None) -> Any:  # noqa: ARG002,E501
            calls.append(dict(raw_query))
            if len(calls) == 1:
                raise NotSupported("dialect not supported")
            return {"ok": True}

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(sk_adapter_module, "validate_graph_result_type", lambda v, **_k: v)

    client = _make_client(adapter)

    _ = client.query("Q", dialect="cypher")

    assert len(calls) == 2
    assert "dialect" in calls[0]
    assert "dialect" not in calls[1]


def test_query_does_not_retry_on_NotSupported_when_no_dialect_provided(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """If dialect not explicitly provided, NotSupported should propagate."""
    class DummyTranslator:
        def query(self, *_a: Any, **_k: Any) -> Any:
            raise NotSupported("no support")

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)

    with pytest.raises(NotSupported):
        client.query("Q")


@pytest.mark.asyncio
async def test_aquery_retries_without_dialect_on_NotSupported_when_dialect_provided(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """Async variant should retry without dialect when explicitly provided."""
    calls: List[Dict[str, Any]] = []

    class DummyTranslator:
        async def arun_query(self, raw_query: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None, mmr_config: Any = None) -> Any:  # noqa: ARG002,E501
            calls.append(dict(raw_query))
            if len(calls) == 1:
                raise NotSupported("dialect not supported")
            return {"ok": True}

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(sk_adapter_module, "validate_graph_result_type", lambda v, **_k: v)

    client = _make_client(adapter)

    _ = await client.aquery("Q", dialect="cypher")

    assert len(calls) == 2
    assert "dialect" in calls[0]
    assert "dialect" not in calls[1]


@pytest.mark.asyncio
async def test_aquery_does_not_retry_on_NotSupported_when_no_dialect_provided(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """Async NotSupported should propagate when dialect not explicitly provided."""
    class DummyTranslator:
        async def arun_query(self, *_a: Any, **_k: Any) -> Any:
            raise NotSupported("no support")

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)

    with pytest.raises(NotSupported):
        await client.aquery("Q")


# ---------------------------------------------------------------------------
# 7) Streaming normalization + validation + error context (7 tests)
# ---------------------------------------------------------------------------


def test_stream_query_validates_each_chunk_and_invalid_attaches_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """stream_query should validate chunks and attach framework context when validation fails."""
    invalid_chunk = object()
    attached: Dict[str, Any] = {}

    class DummyTranslator:
        def query_stream(self, *_a: Any, **_k: Any) -> Iterator[Any]:
            yield invalid_chunk

    def fake_validate_graph_result_type(value: Any, **_k: Any) -> Any:
        raise ValueError("invalid chunk")

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        attached.update(ctx)

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(sk_adapter_module, "validate_graph_result_type", fake_validate_graph_result_type)
    monkeypatch.setattr(sk_adapter_module, "attach_context", fake_attach_context)

    client = _make_client(adapter)

    with pytest.raises(ValueError, match="invalid chunk"):
        list(client.stream_query("Q"))

    assert attached.get("framework") == "semantic_kernel"


@pytest.mark.asyncio
async def test_astream_query_accepts_direct_async_iterator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """If translator returns an AsyncIterator directly, adapter should consume it."""
    class DummyTranslator:
        def arun_query_stream(self, *_a: Any, **_k: Any) -> Any:
            return _make_async_gen([{"chunk": 1}])

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(sk_adapter_module, "validate_graph_result_type", lambda v, **_k: v)

    client = _make_client(adapter)

    aiter = client.astream_query("Q")
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    out = []
    async for c in aiter:
        out.append(c)

    assert out == [{"chunk": 1}]


@pytest.mark.asyncio
async def test_astream_query_accepts_awaitable_resolving_to_async_iterator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """If translator returns awaitable->AsyncIterator, adapter should await then consume."""
    class DummyTranslator:
        async def arun_query_stream(self, *_a: Any, **_k: Any) -> Any:
            return _make_async_gen([{"chunk": 1}, {"chunk": 2}])

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(sk_adapter_module, "validate_graph_result_type", lambda v, **_k: v)

    client = _make_client(adapter)

    aiter = client.astream_query("Q")
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    out = []
    async for c in aiter:
        out.append(c)

    assert out == [{"chunk": 1}, {"chunk": 2}]


@pytest.mark.asyncio
async def test_astream_query_invalid_shape_raises_typeerror_with_code(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """Invalid stream shape should raise TypeError containing BAD_ASYNC_ITERATOR_SHAPE."""
    class DummyTranslator:
        def arun_query_stream(self, *_a: Any, **_k: Any) -> Any:
            return {"not": "an iterator"}

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)

    with pytest.raises(TypeError) as exc_info:
        aiter = client.astream_query("Q")
        if inspect.isawaitable(aiter):
            aiter = await aiter  # type: ignore[assignment]
        async for _ in aiter:  # noqa: B007
            pass

    assert ErrorCodes.BAD_ASYNC_ITERATOR_SHAPE in str(exc_info.value)


@pytest.mark.asyncio
async def test_astream_query_invalid_chunk_attaches_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    FIXED ISSUE #2:
    validate_graph_result_type is synchronous; the test must not run event-loop
    bridging hacks (run_until_complete). It should use a sync validator fake.
    """
    invalid_chunk = object()
    attached: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_query_stream(self, *_a: Any, **_k: Any) -> Any:
            return _make_async_gen([invalid_chunk])

    def fake_validate_graph_result_type(value: Any, **_k: Any) -> Any:
        raise ValueError("invalid async chunk")

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        attached.update(ctx)

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(sk_adapter_module, "validate_graph_result_type", fake_validate_graph_result_type)
    monkeypatch.setattr(sk_adapter_module, "attach_context", fake_attach_context)

    client = _make_client(adapter)

    aiter = client.astream_query("Q")
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    with pytest.raises(ValueError, match="invalid async chunk"):
        async for _ in aiter:  # noqa: B007
            pass

    assert attached.get("framework") == "semantic_kernel"


@pytest.mark.asyncio
async def test_astream_query_cancelled_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """CancelledError should not be swallowed by adapter streaming loop."""
    async def gen():
        raise asyncio.CancelledError()

        yield  # pragma: no cover

    class DummyTranslator:
        async def arun_query_stream(self, *_a: Any, **_k: Any) -> Any:
            return gen()

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)

    aiter = client.astream_query("Q")
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    with pytest.raises(asyncio.CancelledError):
        async for _ in aiter:  # noqa: B007
            pass


def test_framework_ctx_operation_for_streaming_sync_and_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """framework_ctx.operation should be 'stream_query' for both stream_query and astream_query."""
    captured_sync: Dict[str, Any] = {}
    captured_async: Dict[str, Any] = {}

    class DummyTranslatorSync:
        def query_stream(self, raw_query: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None) -> Iterator[Any]:  # noqa: ARG002,E501
            captured_sync["framework_ctx"] = dict(framework_ctx or {})
            yield {"chunk": 1}

    class DummyTranslatorAsync:
        async def arun_query_stream(self, raw_query: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:  # noqa: ARG002,E501
            captured_async["framework_ctx"] = dict(framework_ctx or {})
            return _make_async_gen([{"chunk": 1}])

    # Sync
    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslatorSync())
    monkeypatch.setattr(sk_adapter_module, "validate_graph_result_type", lambda v, **_k: v)
    client1 = _make_client(adapter)
    list(client1.stream_query("Q"))
    assert captured_sync["framework_ctx"]["operation"] == "stream_query"

    # Async (run in a nested loop using asyncio.run to keep this test sync)
    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslatorAsync())
    client2 = _make_client(adapter)

    async def drive():
        aiter = client2.astream_query("Q")
        if inspect.isawaitable(aiter):
            aiter2 = await aiter
        else:
            aiter2 = aiter
        async for _ in aiter2:  # noqa: B007
            break

    asyncio.run(drive())
    assert captured_async["framework_ctx"]["operation"] == "stream_query"


# ---------------------------------------------------------------------------
# 8) Delete selector semantics (6 tests: 4 param + 2 param)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind, use_filter",
    [
        ("nodes", True),
        ("nodes", False),
        ("edges", True),
        ("edges", False),
    ],
)
def test_delete_filter_or_ids_paths_call_translator_with_expected_raw(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
    kind: str,
    use_filter: bool,
) -> None:
    """delete_nodes/delete_edges should pass filter or ids appropriately to the translator."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def delete_nodes(self, raw: Any, *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:  # noqa: ARG002,E501
            captured["raw"] = raw
            return {"ok": True}

        def delete_edges(self, raw: Any, *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:  # noqa: ARG002,E501
            captured["raw"] = raw
            return {"ok": True}

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(sk_adapter_module, "validate_graph_result_type", lambda v, **_k: v)

    client = _make_client(adapter)

    if kind == "nodes":
        class Spec:
            namespace = None
            filter = {"x": 1} if use_filter else None
            ids = None if use_filter else ["1", "2"]

        _ = client.delete_nodes(Spec())
    else:
        class Spec:
            namespace = None
            filter = {"y": 2} if use_filter else None
            ids = None if use_filter else ["e1"]

        _ = client.delete_edges(Spec())

    if use_filter:
        assert isinstance(captured["raw"], Mapping)
    else:
        assert isinstance(captured["raw"], list)
        assert captured["raw"]


@pytest.mark.parametrize("kind", ["nodes", "edges"])
def test_delete_missing_filter_and_ids_raises_and_async_too(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
    kind: str,
) -> None:
    """
    Missing filter and ids should raise BadRequest (sync),
    and also raise for the async delete path when driven explicitly.
    """
    class DummyTranslator:
        async def arun_delete_nodes(self, *_a: Any, **_k: Any) -> Any:
            return {"ok": True}

        async def arun_delete_edges(self, *_a: Any, **_k: Any) -> Any:
            return {"ok": True}

        def delete_nodes(self, *_a: Any, **_k: Any) -> Any:
            return {"ok": True}

        def delete_edges(self, *_a: Any, **_k: Any) -> Any:
            return {"ok": True}

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)

    if kind == "nodes":
        class Spec:
            namespace = None
            filter = None
            ids = None

        with pytest.raises(BadRequest) as exc_info:
            client.delete_nodes(Spec())
        assert getattr(exc_info.value, "code", None) == ErrorCodes.BAD_ADAPTER_RESULT
        assert ErrorCodes.INVALID_DELETE_SPEC in str(exc_info.value)

        async def drive():
            with pytest.raises(BadRequest):
                await client.adelete_nodes(Spec())

        asyncio.run(drive())
    else:
        class Spec:
            namespace = None
            filter = None
            ids = []

        with pytest.raises(BadRequest) as exc_info:
            client.delete_edges(Spec())
        assert getattr(exc_info.value, "code", None) == ErrorCodes.BAD_ADAPTER_RESULT
        assert ErrorCodes.INVALID_DELETE_SPEC in str(exc_info.value)

        async def drive():
            with pytest.raises(BadRequest):
                await client.adelete_edges(Spec())

        asyncio.run(drive())


# ---------------------------------------------------------------------------
# 9) Upsert edges validation (6 tests)
# ---------------------------------------------------------------------------


def test_upsert_edges_validates_and_is_side_effect_free_and_async_too(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    upsert_edges should:
    - materialize edges iterable to a list,
    - pass the list to the translator,
    - not require mutation of spec.edges,
    - and async path should behave equivalently.
    """
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def upsert_edges(self, edges: Any, *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:  # noqa: ARG002,E501
            captured["sync_edges"] = edges
            return {"ok": True}

        async def arun_upsert_edges(self, edges: Any, *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:  # noqa: ARG002,E501
            captured["async_edges"] = edges
            return {"ok": True}

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(sk_adapter_module, "validate_graph_result_type", lambda v, **_k: v)

    client = _make_client(adapter)

    def edge_iter():
        yield _edge_obj(edge_id="e1")
        yield _edge_obj(edge_id="e2")

    class Spec:
        namespace = None
        edges = edge_iter()

    _ = client.upsert_edges(Spec())
    assert isinstance(captured["sync_edges"], list)
    assert [e.id for e in captured["sync_edges"]] == ["e1", "e2"]

    async def drive():
        class Spec2:
            namespace = None
            edges = edge_iter()

        _ = await client.aupsert_edges(Spec2())
        assert isinstance(captured["async_edges"], list)
        assert [e.id for e in captured["async_edges"]] == ["e1", "e2"]

    asyncio.run(drive())


def test_upsert_edges_rejects_none_edges(adapter: Any) -> None:
    client = _make_client(adapter)

    class Spec:
        namespace = None
        edges = None

    with pytest.raises(BadRequest) as exc_info:
        client._validate_upsert_edges_spec(Spec())  # noqa: SLF001
    assert getattr(exc_info.value, "code", None) == ErrorCodes.BAD_ADAPTER_RESULT


def test_upsert_edges_rejects_empty_edges(adapter: Any) -> None:
    client = _make_client(adapter)

    class Spec:
        namespace = None
        edges: List[Any] = []

    with pytest.raises(BadRequest) as exc_info:
        client._validate_upsert_edges_spec(Spec())  # noqa: SLF001
    assert getattr(exc_info.value, "code", None) == ErrorCodes.BAD_ADAPTER_RESULT


@pytest.mark.parametrize(
    "edge_kwargs",
    [
        {"edge_id": None},
        {"src": None},
    ],
)
def test_upsert_edges_requires_required_fields(adapter: Any, edge_kwargs: Dict[str, Any]) -> None:
    client = _make_client(adapter)

    class Spec:
        namespace = None
        edges = [_edge_obj(**edge_kwargs)]

    with pytest.raises(BadRequest) as exc_info:
        client._validate_upsert_edges_spec(Spec())  # noqa: SLF001
    assert getattr(exc_info.value, "code", None) == ErrorCodes.BAD_ADAPTER_RESULT


def test_upsert_edges_rejects_non_json_properties(adapter: Any) -> None:
    client = _make_client(adapter)

    class NonJSON:
        pass

    class Spec:
        namespace = None
        edges = [_edge_obj(properties=NonJSON())]

    with pytest.raises(BadRequest) as exc_info:
        client._validate_upsert_edges_spec(Spec())  # noqa: SLF001
    assert getattr(exc_info.value, "code", None) == ErrorCodes.BAD_ADAPTER_RESULT


# ---------------------------------------------------------------------------
# 10) Single-source payload drift prevention (2 tests)
# ---------------------------------------------------------------------------


def test_bulk_vertices_sync_async_build_identical_raw_request(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """bulk_vertices and abulk_vertices should send identical raw_request shapes for same spec."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def bulk_vertices(self, raw_request: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:  # noqa: ARG002,E501
            captured["sync"] = dict(raw_request)
            return {"ok": True}

        async def arun_bulk_vertices(self, raw_request: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:  # noqa: ARG002,E501
            captured["async"] = dict(raw_request)
            return {"ok": True}

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(sk_adapter_module, "validate_graph_result_type", lambda v, **_k: v)

    class Spec:
        namespace = "ns"
        limit = 10
        cursor = "c"
        filter = {"x": 1}

    client = _make_client(adapter)

    _ = client.bulk_vertices(Spec())
    asyncio.run(client.abulk_vertices(Spec()))

    assert captured["sync"] == captured["async"]


def test_traversal_sync_async_build_identical_raw_request(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """traversal and atraversal should send identical raw_request shapes for same spec."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def traversal(self, raw_request: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:  # noqa: ARG002,E501
            captured["sync"] = dict(raw_request)
            return {"ok": True}

        async def arun_traversal(self, raw_request: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:  # noqa: ARG002,E501
            captured["async"] = dict(raw_request)
            return {"ok": True}

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(sk_adapter_module, "validate_graph_result_type", lambda v, **_k: v)

    class Spec:
        start_nodes = ["a", "b"]
        max_depth = 2
        direction = "out"
        relationship_types = ["REL"]
        node_filters = None
        relationship_filters = None
        return_properties = None
        namespace = "ns-trav"

    client = _make_client(adapter)

    _ = client.traversal(Spec())
    asyncio.run(client.atraversal(Spec()))

    assert captured["sync"] == captured["async"]


# ---------------------------------------------------------------------------
# 11) Close semantics alignment (3 tests)
# ---------------------------------------------------------------------------


def test_aclose_sets_closed_flag_when_async_close_succeeds(adapter: Any) -> None:
    """After successful aclose(), the client should be considered closed from sync perspective."""
    class ClosingAdapter:
        def __init__(self) -> None:
            self.closed = False
            self.aclosed = False

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.aclosed = True

        def capabilities(self) -> Dict[str, Any]:
            return {}

        def health(self) -> Dict[str, Any]:
            return {}

    a = ClosingAdapter()
    client = _make_client(a)

    asyncio.run(client.aclose())
    assert a.aclosed is True
    assert client._closed is True  # noqa: SLF001


def test_aclose_falls_back_to_close_when_no_aclose(adapter: Any) -> None:
    """If adapter lacks aclose(), client.aclose() should fall back to client.close()."""
    class ClosingAdapter:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

        def capabilities(self) -> Dict[str, Any]:
            return {}

        def health(self) -> Dict[str, Any]:
            return {}

    a = ClosingAdapter()
    client = _make_client(a)

    asyncio.run(client.aclose())
    assert a.closed is True


def test_close_is_idempotent(adapter: Any) -> None:
    """Calling close() multiple times should not raise and should only close once."""
    class ClosingAdapter:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

        def capabilities(self) -> Dict[str, Any]:
            return {}

        def health(self) -> Dict[str, Any]:
            return {}

    a = ClosingAdapter()
    client = _make_client(a)

    client.close()
    client.close()
    assert a.close_calls == 1


# ---------------------------------------------------------------------------
# 12) Semantic Kernel optional integration wrapper (4 tests)
# ---------------------------------------------------------------------------


def test_plugin_is_available_and_constructible_when_semantic_kernel_installed(adapter: Any) -> None:
    """CorpusSemanticKernelPlugin should be the real integration wrapper (not the ImportError stub)."""
    client = _make_client(adapter)
    plugin = CorpusSemanticKernelPlugin(client=client, namespace="ns-plugin")
    assert plugin is not None
    assert hasattr(plugin, "query")
    assert hasattr(plugin, "astream_query")


def test_plugin_namespace_precedence_and_forwarding_sync(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Namespace precedence for plugin:
      explicit namespace arg > plugin namespace > client default namespace.
    """
    captured: List[Dict[str, Any]] = []

    class DummyTranslator:
        def query(self, raw_query: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None, mmr_config: Any = None) -> Any:  # noqa: ARG002,E501
            captured.append(dict(raw_query))
            return {"ok": True}

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(sk_adapter_module, "validate_graph_result_type", lambda v, **_k: v)

    client = _make_client(adapter, default_namespace="ns-client")
    plugin = CorpusSemanticKernelPlugin(client=client, namespace="ns-plugin")

    _ = plugin.query("Q1")  # plugin namespace
    _ = plugin.query("Q2", namespace="ns-explicit")  # explicit namespace

    assert captured[0].get("namespace") == "ns-plugin"
    assert captured[1].get("namespace") == "ns-explicit"


@pytest.mark.asyncio
async def test_plugin_forwarding_async_query_and_stream(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """Plugin should forward async query and async streaming calls to the underlying client."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        async def arun_query(self, raw_query: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None, mmr_config: Any = None) -> Any:  # noqa: ARG002,E501
            captured["aquery_raw"] = dict(raw_query)
            return {"ok": True}

        async def arun_query_stream(self, raw_query: Mapping[str, Any], *, op_ctx: Any = None, framework_ctx: Any = None) -> Any:  # noqa: ARG002,E501
            captured["astream_raw"] = dict(raw_query)
            return _make_async_gen([{"chunk": 1}])

    monkeypatch.setattr(sk_adapter_module, "create_graph_translator", lambda *_a, **_k: DummyTranslator())
    monkeypatch.setattr(sk_adapter_module, "validate_graph_result_type", lambda v, **_k: v)

    client = _make_client(adapter)
    plugin = CorpusSemanticKernelPlugin(client=client, namespace="ns-plugin")

    _ = await plugin.aquery("Q")
    aiter = plugin.astream_query("Q2")
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]
    async for _ in aiter:  # noqa: B007
        break

    assert captured["aquery_raw"]["namespace"] == "ns-plugin"
    assert captured["astream_raw"]["namespace"] == "ns-plugin"


def test_plugin_stub_raises_importerror_when_semantic_kernel_missing_simulated(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Simulate a missing semantic_kernel install and ensure the plugin becomes a stub.

    We do this without importing semantic_kernel in the test:
    - monkeypatch builtins.__import__ to raise ImportError for 'semantic_kernel'
    - reload the adapter module
    - construct CorpusSemanticKernelPlugin => should raise ImportError
    - restore and reload module back to normal for other tests
    """
    original_import = builtins.__import__

    def blocked_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:  # noqa: A002,E501
        if name == "semantic_kernel":
            raise ImportError("simulated missing semantic_kernel")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    # Reload module under "missing SK" conditions.
    reloaded = importlib.reload(sk_adapter_module)

    try:
        assert getattr(reloaded, "_semantic_kernel", None) is None
        with pytest.raises(ImportError):
            _ = reloaded.CorpusSemanticKernelPlugin(client=_make_client(adapter))
    finally:
        # Restore import and reload back to a "healthy" state for the rest of the suite.
        monkeypatch.setattr(builtins, "__import__", original_import)
        importlib.reload(sk_adapter_module)


# ---------------------------------------------------------------------------
# NOTE:
# This file intentionally contains exactly 54 pytest test cases:
# - Parametrization expands some tests into multiple cases.
# - No tests are skipped by design: missing SK is a hard failure.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
