# tests/frameworks/graph/test_semantickernel_graph_adapter.py

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List

import inspect

import pytest

import corpus_sdk.graph.framework_adapters.semantic_kernel as sk_adapter_module
from corpus_sdk.graph.framework_adapters.semantic_kernel import (
    CorpusSemanticKernelGraphClient,
    ErrorCodes,
    SemanticKernelGraphFrameworkTranslator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(
    adapter: Any,
    **kwargs: Any,
) -> CorpusSemanticKernelGraphClient:
    """Construct a CorpusSemanticKernelGraphClient instance from the generic adapter."""
    return CorpusSemanticKernelGraphClient(adapter=adapter, **kwargs)


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
        captured["args"] = args
        captured["kwargs"] = kwargs

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(
        sk_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

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
    """
    If framework_translator is provided, CorpusSemanticKernelGraphClient should pass
    it through to create_graph_translator instead of constructing its own
    SemanticKernelGraphFrameworkTranslator.
    """
    captured: Dict[str, Any] = {}

    class CustomTranslator(SemanticKernelGraphFrameworkTranslator):
        pass

    custom = CustomTranslator()

    def fake_create_graph_translator(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["args"] = args
        captured["kwargs"] = kwargs

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(
        sk_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(
        adapter,
        framework_translator=custom,
        framework_version="sk-fw-1.2.3",
    )

    _ = client._translator  # noqa: SLF001

    kwargs = captured["kwargs"]
    assert kwargs.get("framework") == "semantic_kernel"
    assert kwargs.get("translator") is custom


# ---------------------------------------------------------------------------
# Context translation / core_ctx_from_semantic_kernel mapping
# ---------------------------------------------------------------------------


def test_semantickernel_context_and_extra_context_passed_to_core_ctx(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Verify that context/settings and extra_context are passed through to
    core_ctx_from_semantic_kernel with the configured framework_version.
    """
    captured: Dict[str, Any] = {}

    # Patch OperationContext so our fake ctx passes isinstance() check.
    class DummyOperationContext:
        def __init__(self, **kwargs: Any) -> None:
            self.attrs = kwargs

    monkeypatch.setattr(
        sk_adapter_module,
        "OperationContext",
        DummyOperationContext,
    )

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
        captured["extra"] = extra
        return DummyOperationContext()

    monkeypatch.setattr(
        sk_adapter_module,
        "core_ctx_from_semantic_kernel",
        fake_core_ctx_from_semantic_kernel,
    )

    class DummyTranslator:
        def query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return {"ok": True}

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return DummyTranslator()

    monkeypatch.setattr(
        sk_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(
        adapter,
        framework_version="semantic-kernel-test-version",
    )

    ctx = object()
    settings = {"temperature": 0.3}
    extra_ctx = {"request_id": "req-xyz", "tenant": "tenant-1"}

    result = client.query(
        "MATCH (n) RETURN n LIMIT 1",
        context=ctx,
        settings=settings,
        extra_context=extra_ctx,
    )
    assert result is not None

    assert captured.get("context") is ctx
    assert captured.get("settings") == settings
    assert captured.get("framework_version") == "semantic-kernel-test-version"
    assert captured.get("extra") == extra_ctx


def test_build_ctx_failure_raises_badrequest_with_error_code_and_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If core_ctx_from_semantic_kernel raises, _build_ctx should wrap it in a
    BadRequest-like error with ErrorCodes.BAD_OPERATION_CONTEXT and call
    attach_context at least once.
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_core_ctx_from_semantic_kernel(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        raise RuntimeError("boom!")

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    class DummyTranslator:
        def query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return {"ok": False}

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return DummyTranslator()

    monkeypatch.setattr(
        sk_adapter_module,
        "core_ctx_from_semantic_kernel",
        fake_core_ctx_from_semantic_kernel,
    )
    monkeypatch.setattr(
        sk_adapter_module,
        "attach_context",
        fake_attach_context,
    )
    monkeypatch.setattr(
        sk_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(adapter)

    with pytest.raises(Exception) as exc_info:
        client.query("MATCH (n) RETURN n", context=object())

    err = exc_info.value
    # Semantic assertion instead of relying on concrete type
    assert getattr(err, "code", None) == ErrorCodes.BAD_OPERATION_CONTEXT
    msg = str(err).lower()
    # Message should at least indicate it's about context / operation
    assert "operation" in msg or "context" in msg

    # And we should have called attach_context at least once
    assert captured_ctx
    assert captured_ctx.get("framework") == "semantic_kernel"


# ---------------------------------------------------------------------------
# Error-context decorator behavior
# ---------------------------------------------------------------------------


def test_sync_errors_include_semantickernel_metadata_in_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    When an error occurs during a sync graph operation, error context should
    include Semantic Kernel–specific metadata via attach_context().
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(
        sk_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingTranslator:
        def query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from semantic_kernel graph adapter")

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return FailingTranslator()

    monkeypatch.setattr(
        sk_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(adapter)

    with pytest.raises(RuntimeError, match="test error from semantic_kernel graph adapter"):
        client.query("MATCH (n) RETURN n", context={"user_id": "u-sync"})

    assert captured_ctx
    assert captured_ctx.get("framework") == "semantic_kernel"
    assert str(captured_ctx.get("operation", "")).startswith("graph_")


@pytest.mark.asyncio
async def test_async_errors_include_semantickernel_metadata_in_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Same as the sync error-context test but for the async query path.
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(
        sk_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingTranslator:
        async def arun_query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from semantic_kernel graph adapter")

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return FailingTranslator()

    monkeypatch.setattr(
        sk_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(adapter)

    with pytest.raises(RuntimeError, match="test error from semantic_kernel graph adapter"):
        await client.aquery("MATCH (n) RETURN n", context={"user_id": "u-async"})

    assert captured_ctx
    assert captured_ctx.get("framework") == "semantic_kernel"
    assert str(captured_ctx.get("operation", "")).startswith("graph_")


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
    client = _make_client(adapter, default_namespace="sk-ns")

    # Non-streaming query
    result = client.query("MATCH (n) RETURN n LIMIT 1")
    assert result is not None

    # Streaming query
    chunks = list(client.stream_query("MATCH (n) RETURN n LIMIT 2"))
    assert isinstance(chunks, list)


def test_sync_query_accepts_optional_params_and_context(adapter: Any) -> None:
    """
    query() should accept params, dialect, namespace, timeout_ms, and
    context/settings/extra_context kwargs without raising.
    """
    client = _make_client(adapter, default_dialect="cypher")

    result = client.query(
        "MATCH (n) RETURN n LIMIT $limit",
        params={"limit": 5},
        dialect="cypher",
        namespace="ctx-ns",
        timeout_ms=5000,
        context={"user_id": "u-sync"},
        settings={"temperature": 0.2},
        extra_context={"request_id": "req-sync"},
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Async semantics (basic smoke tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_query_and_stream_basic(adapter: Any) -> None:
    """
    Async aquery / astream_query should exist and produce results compatible
    with the sync API (non-None result / async-iterable of chunks).
    """
    client = _make_client(adapter)

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
    adapter: Any,
) -> None:
    """
    aquery() should accept the same optional params and context as query().
    """
    client = _make_client(adapter, default_namespace="async-sk-ns")

    result = await client.aquery(
        "MATCH (n) RETURN n LIMIT $limit",
        params={"limit": 3},
        dialect="cypher",
        namespace="async-sk-ns",
        timeout_ms=2500,
        context={"user_id": "u-async"},
        settings={"temperature": 0.5},
        extra_context={"request_id": "req-async"},
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Streaming validation tests
# ---------------------------------------------------------------------------


def test_stream_query_invalid_chunk_triggers_validation_and_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    stream_query() should pass each chunk through validate_graph_result_type
    and attach Semantic Kernel error context when validation fails.
    """
    captured: Dict[str, Any] = {}
    invalid_chunk = object()

    class DummyTranslator:
        def query_stream(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any = None,
            framework_ctx: Mapping[str, Any] | None = None,
        ):
            captured["raw_query"] = dict(raw_query)
            captured["framework_ctx"] = dict(framework_ctx or {})
            captured["op_ctx"] = op_ctx
            # Emit a single invalid chunk
            yield invalid_chunk

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return DummyTranslator()

    def fake_validate_graph_result_type(value: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["validated_chunk"] = value
        raise ValueError("invalid chunk from translator")

    error_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        error_ctx.update(ctx)

    monkeypatch.setattr(
        sk_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        sk_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )
    monkeypatch.setattr(
        sk_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    client = _make_client(adapter)

    stream = client.stream_query("MATCH (n) RETURN n")

    with pytest.raises(ValueError, match="invalid chunk from translator"):
        for _ in stream:
            pass

    # Ensure the invalid chunk was given to the validator
    assert captured.get("validated_chunk") is invalid_chunk

    # And that framework-specific error context was attached
    assert error_ctx
    assert error_ctx.get("framework") == "semantic_kernel"
    assert str(error_ctx.get("operation", "")).startswith("graph_")


@pytest.mark.asyncio
async def test_astream_query_invalid_chunk_triggers_validation_and_context_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    astream_query() should also validate chunks and attach Semantic Kernel
    error context when validation fails.
    """
    captured: Dict[str, Any] = {}
    invalid_chunk = object()

    class DummyTranslator:
        async def arun_query_stream(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any = None,
            framework_ctx: Mapping[str, Any] | None = None,
        ):
            captured["raw_query"] = dict(raw_query)
            captured["framework_ctx"] = dict(framework_ctx or {})
            captured["op_ctx"] = op_ctx

            async def gen():
                yield invalid_chunk

            # Return an async generator
            return gen()

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return DummyTranslator()

    async def fake_validate_graph_result_type(value: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        # Even though the real helper is sync, this keeps the test robust
        captured["validated_chunk"] = value
        raise ValueError("invalid async chunk from translator")

    error_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        error_ctx.update(ctx)

    monkeypatch.setattr(
        sk_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    # Wrap the async fake in a sync shim because the real function is sync.
    def sync_wrapper(value: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        # drive the async validator in a minimal way
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            fake_validate_graph_result_type(value, **kwargs),
        )

    monkeypatch.setattr(
        sk_adapter_module,
        "validate_graph_result_type",
        sync_wrapper,
    )
    monkeypatch.setattr(
        sk_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    client = _make_client(adapter)

    aiter = client.astream_query("MATCH (n) RETURN n")
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    with pytest.raises(ValueError, match="invalid async chunk from translator"):
        async for _ in aiter:  # noqa: B007
            pass

    assert captured.get("validated_chunk") is invalid_chunk
    assert error_ctx
    assert error_ctx.get("framework") == "semantic_kernel"
    assert str(error_ctx.get("operation", "")).startswith("graph_")


# ---------------------------------------------------------------------------
# Bulk vertices / batch semantics (wiring)
# ---------------------------------------------------------------------------


def test_bulk_vertices_builds_raw_request_and_calls_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
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
        sk_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        sk_adapter_module,
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

    assert "args" in captured
    raw = captured["args"][0]
    assert raw == {
        "namespace": "ns-bulk",
        "limit": 42,
        "cursor": "cursor-token",
        "filter": {"foo": "bar"},
    }

    fw_ctx = captured.get("framework_ctx", {})
    assert fw_ctx.get("framework") == "semantic_kernel"
    assert fw_ctx.get("operation") == "bulk_vertices"
    assert fw_ctx.get("namespace") == "ns-bulk"
    assert captured.get("op_ctx") is None


@pytest.mark.asyncio
async def test_abulk_vertices_builds_raw_request_and_calls_translator_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
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
        sk_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        sk_adapter_module,
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

    assert "args" in captured
    raw = captured["args"][0]
    assert raw == {
        "namespace": "ns-abulk",
        "limit": 7,
        "cursor": None,
        "filter": {"bar": 1},
    }

    fw_ctx = captured.get("framework_ctx", {})
    assert fw_ctx.get("framework") == "semantic_kernel"
    assert fw_ctx.get("operation") == "bulk_vertices"
    assert fw_ctx.get("namespace") == "ns-abulk"
    assert captured.get("op_ctx") is None


def test_batch_builds_raw_batch_ops_and_calls_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
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
        sk_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        sk_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )
    monkeypatch.setattr(
        sk_adapter_module,
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
    assert captured.get("validated_batch") is True

    assert "args" in captured
    raw_ops = captured["args"][0]
    assert raw_ops == [
        {"op": "upsert_nodes", "args": {"id": "1"}},
        {"op": "delete_nodes", "args": {"ids": ["1", "2"]}},
    ]

    fw_ctx = captured.get("framework_ctx", {})
    assert fw_ctx.get("framework") == "semantic_kernel"
    assert fw_ctx.get("operation") == "batch"
    assert captured.get("op_ctx") is None


@pytest.mark.asyncio
async def test_abatch_builds_raw_batch_ops_and_calls_translator_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
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
        sk_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        sk_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )
    monkeypatch.setattr(
        sk_adapter_module,
        "validate_batch_operations",
        fake_validate_batch_operations,
    )

    client = _make_client(adapter)

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
    assert fw_ctx.get("framework") == "semantic_kernel"
    assert fw_ctx.get("operation") == "batch"
    assert captured.get("op_ctx") is None


# ---------------------------------------------------------------------------
# Capabilities / health passthrough (basic)
# ---------------------------------------------------------------------------


def test_capabilities_and_health_basic(adapter: Any) -> None:
    """
    Capabilities and health should be surfaced as mappings.

    The detailed structure is tested in framework-agnostic graph contract
    tests; here we only assert that the Semantic Kernel adapter normalizes to
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
    the sync variants.
    """
    client = _make_client(adapter)

    acaps = await client.acapabilities()
    assert isinstance(acaps, Mapping)

    ahealth = await client.ahealth()
    assert isinstance(ahealth, Mapping)


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
    with CorpusSemanticKernelGraphClient(adapter=adapter) as client:
        assert client is not None

    assert adapter.closed is True

    # Async context manager
    adapter2 = ClosingGraphAdapter()
    client2 = CorpusSemanticKernelGraphClient(adapter=adapter2)

    async with client2:
        assert client2 is not None

    assert adapter2.aclosed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

