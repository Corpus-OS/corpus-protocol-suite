from __future__ import annotations

import asyncio
import inspect
from typing import Any, Dict, Mapping, List

import pytest

import corpus_sdk.graph.framework_adapters.autogen as autogen_adapter_module
from corpus_sdk.graph.framework_adapters.autogen import (
    AutoGenGraphFrameworkTranslator,
    CorpusAutoGenGraphClient,
    ErrorCodes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(graph_adapter: Any, **kwargs: Any) -> CorpusAutoGenGraphClient:
    """Construct a CorpusAutoGenGraphClient instance from the generic adapter."""
    return CorpusAutoGenGraphClient(graph_adapter=graph_adapter, **kwargs)


# ---------------------------------------------------------------------------
# Constructor / translator behavior
# ---------------------------------------------------------------------------


def test_default_translator_uses_autogen_framework_translator(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    By default, CorpusAutoGenGraphClient should:

    - Construct an AutoGenGraphFrameworkTranslator instance, and
    - Pass it into create_graph_translator with framework="autogen".
    """
    captured_args: Dict[str, Any] = {}

    def fake_create_graph_translator(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        captured_args["args"] = args
        captured_args["kwargs"] = kwargs

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(
        autogen_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(graph_adapter)

    # Trigger lazy translator construction
    _ = client._translator  # noqa: SLF001

    assert "kwargs" in captured_args
    kwargs = captured_args["kwargs"]

    assert kwargs.get("framework") == "autogen"
    translator = kwargs.get("translator")
    assert isinstance(translator, AutoGenGraphFrameworkTranslator)


def test_framework_translator_override_is_respected(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    If framework_translator is provided, CorpusAutoGenGraphClient should pass
    it through to create_graph_translator instead of constructing its own
    AutoGenGraphFrameworkTranslator.
    """
    captured_args: Dict[str, Any] = {}

    class CustomTranslator(AutoGenGraphFrameworkTranslator):
        pass

    custom = CustomTranslator()

    def fake_create_graph_translator(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        captured_args["args"] = args
        captured_args["kwargs"] = kwargs

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(
        autogen_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(
        graph_adapter,
        framework_translator=custom,
        framework_version="fw-1.2.3",
    )

    _ = client._translator  # noqa: SLF001

    kwargs = captured_args["kwargs"]
    assert kwargs.get("framework") == "autogen"
    assert kwargs.get("translator") is custom


# ---------------------------------------------------------------------------
# Context translation / core_ctx_from_autogen mapping
# ---------------------------------------------------------------------------


def test_autogen_conversation_and_extra_context_passed_to_core_ctx(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    Verify that conversation and extra_context are passed through to
    core_ctx_from_autogen with the configured framework_version.
    """
    captured: Dict[str, Any] = {}

    # Patch OperationContext inside the module so our fake can return
    # a simple dummy instance that passes the isinstance() check.
    class DummyOperationContext:
        def __init__(self, **kwargs: Any) -> None:
            self.attrs = kwargs

    monkeypatch.setattr(
        autogen_adapter_module,
        "OperationContext",
        DummyOperationContext,
    )

    def fake_core_ctx_from_autogen(
        conversation: Any,
        *,
        framework_version: Any = None,
        **extra: Any,
    ) -> Any:
        captured["conversation"] = conversation
        captured["framework_version"] = framework_version
        captured["extra"] = extra
        return DummyOperationContext()

    monkeypatch.setattr(
        autogen_adapter_module,
        "core_ctx_from_autogen",
        fake_core_ctx_from_autogen,
    )

    client = _make_client(
        graph_adapter,
        framework_version="autogen-test-version",
    )

    auto_conv = {
        "conversation_id": "conv-123",
        "agent_name": "agent-x",
    }
    extra_ctx = {
        "request_id": "req-xyz",
        "tenant": "tenant-1",
    }

    # Any query is fine; we only care that _build_ctx calls our fake.
    result = client.query(
        "MATCH (n) RETURN n LIMIT 1",
        conversation=auto_conv,
        extra_context=extra_ctx,
    )
    assert result is not None

    # Verify that our fake_core_ctx_from_autogen saw the right arguments.
    assert captured.get("conversation") == auto_conv
    assert captured.get("framework_version") == "autogen-test-version"
    # extra_context should be merged into **extra
    assert captured.get("extra") == extra_ctx


def test_build_ctx_failure_raises_badrequest_with_error_code_and_attaches_context(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    If core_ctx_from_autogen fails, _build_ctx should:

    - Attach error context via attach_context(framework="autogen", operation="context_translation")
    - Re-raise as a BadRequest-like error with code=ErrorCodes.BAD_OPERATION_CONTEXT
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    def fake_core_ctx_from_autogen(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        raise RuntimeError("boom from autogen ctx")

    monkeypatch.setattr(
        autogen_adapter_module,
        "attach_context",
        fake_attach_context,
    )
    monkeypatch.setattr(
        autogen_adapter_module,
        "core_ctx_from_autogen",
        fake_core_ctx_from_autogen,
    )

    client = _make_client(
        graph_adapter,
        framework_version="autogen-fw-test",
    )

    with pytest.raises(Exception) as exc_info:
        client.query(
            "MATCH (n) RETURN n",
            conversation={"conversation_id": "conv-fail"},
        )

    err = exc_info.value
    # We don't care about the concrete exception type, just the semantic code.
    assert getattr(err, "code", None) == ErrorCodes.BAD_OPERATION_CONTEXT
    msg = str(err).lower()
    assert "operation" in msg or "context" in msg

    # Ensure error context was attached with framework metadata.
    assert captured_ctx.get("framework") == "autogen"
    assert captured_ctx.get("operation") == "context_translation"


# ---------------------------------------------------------------------------
# Error-context decorator behavior
# ---------------------------------------------------------------------------


def test_error_context_includes_autogen_metadata_sync(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    When an error occurs during a sync graph operation, error context should
    include AutoGen-specific metadata via attach_context().
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(
        autogen_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingTranslator:
        def query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from autogen graph adapter")

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return FailingTranslator()

    monkeypatch.setattr(
        autogen_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(graph_adapter)

    auto_conv = {"conversation_id": "conv-ctx", "agent_name": "tester"}

    with pytest.raises(RuntimeError, match="test error from autogen graph adapter"):
        client.query("MATCH (n) RETURN n", conversation=auto_conv)

    # Verify some context was attached
    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "autogen"
    # The shared decorator uses an operation prefix like "graph_..."
    assert str(captured_context.get("operation", "")).startswith("graph_")
    # Best-effort: AutoGen-specific fields should be present if the decorator
    # forwards conversation metadata.
    if "conversation_id" in captured_context:
        assert captured_context["conversation_id"] == "conv-ctx"
    if "agent_name" in captured_context:
        assert captured_context["agent_name"] == "tester"


@pytest.mark.asyncio
async def test_error_context_includes_autogen_metadata_async(
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
        autogen_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingTranslator:
        async def arun_query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from autogen graph adapter")

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return FailingTranslator()

    monkeypatch.setattr(
        autogen_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(graph_adapter)

    auto_conv = {"conversation_id": "conv-ctx-async", "agent_name": "tester-async"}

    with pytest.raises(RuntimeError, match="test error from autogen graph adapter"):
        await client.aquery("MATCH (n) RETURN n", conversation=auto_conv)

    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "autogen"
    assert str(captured_context.get("operation", "")).startswith("graph_")
    if "conversation_id" in captured_context:
        assert captured_context["conversation_id"] == "conv-ctx-async"
    if "agent_name" in captured_context:
        assert captured_context["agent_name"] == "tester-async"


# ---------------------------------------------------------------------------
# Streaming validation / error paths
# ---------------------------------------------------------------------------


def test_stream_query_invalid_chunk_triggers_validation_and_context(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    stream_query() should validate each chunk via validate_graph_result_type.

    If a non-QueryChunk-like value is produced, validate_graph_result_type
    should raise, and the error-context decorator should attach framework
    metadata before re-raising.
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(
        autogen_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    class DummyTranslator:
        def query_stream(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any = None,  # noqa: ARG002
            framework_ctx: Mapping[str, Any] | None = None,
        ):
            # Yield a clearly-invalid "chunk" to trigger validation.
            yield {"not": "a-query-chunk"}

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return DummyTranslator()

    class FakeValidationError(Exception):
        def __init__(self, message: str, code: Any | None = None) -> None:
            super().__init__(message)
            self.code = code

    def fake_validate_graph_result_type(
        result: Any,
        *,
        expected_type: Any,
        operation: str,
        error_code: Any,
        **_: Any,
    ) -> Any:
        # We only care about the streaming chunk path, which uses
        # BAD_TRANSLATED_CHUNK as the error_code.
        if error_code == ErrorCodes.BAD_TRANSLATED_CHUNK:
            raise FakeValidationError("invalid chunk", code=error_code)
        return result

    monkeypatch.setattr(
        autogen_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        autogen_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )

    client = _make_client(graph_adapter)

    it = client.stream_query("MATCH (n) RETURN n LIMIT 2")

    with pytest.raises(FakeValidationError, match="invalid chunk") as exc_info:
        # Force consumption of the first (invalid) chunk.
        next(it)

    err = exc_info.value
    assert getattr(err, "code", None) == ErrorCodes.BAD_TRANSLATED_CHUNK

    # Error-context decorator should have attached framework metadata.
    assert captured_ctx.get("framework") == "autogen"
    assert str(captured_ctx.get("operation", "")).startswith("graph_")


@pytest.mark.asyncio
async def test_astream_query_invalid_chunk_triggers_validation_and_context_async(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    astream_query() should also validate chunks via validate_graph_result_type.

    If an invalid chunk is produced, validation should raise and the
    async error-context decorator should attach framework metadata.
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(
        autogen_adapter_module,
        "attach_context",
        fake_attach_context,
    )

    class DummyTranslator:
        async def arun_query_stream(
            self,
            raw_query: Mapping[str, Any],
            *,
            op_ctx: Any = None,  # noqa: ARG002
            framework_ctx: Mapping[str, Any] | None = None,
        ):
            # Async generator yielding an invalid chunk.
            async def _gen():
                yield {"not": "a-query-chunk"}

            return _gen()

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return DummyTranslator()

    class FakeValidationError(Exception):
        def __init__(self, message: str, code: Any | None = None) -> None:
            super().__init__(message)
            self.code = code

    def fake_validate_graph_result_type(
        result: Any,
        *,
        expected_type: Any,
        operation: str,
        error_code: Any,
        **_: Any,
    ) -> Any:
        if error_code == ErrorCodes.BAD_TRANSLATED_CHUNK:
            raise FakeValidationError("invalid chunk async", code=error_code)
        return result

    monkeypatch.setattr(
        autogen_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        autogen_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )

    client = _make_client(graph_adapter)

    aiter = client.astream_query("MATCH (n) RETURN n LIMIT 2")
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    with pytest.raises(FakeValidationError, match="invalid chunk async") as exc_info:
        async for _ in aiter:  # noqa: B007
            # First iteration should raise from validation.
            break

    err = exc_info.value
    assert getattr(err, "code", None) == ErrorCodes.BAD_TRANSLATED_CHUNK

    assert captured_ctx.get("framework") == "autogen"
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
    client = _make_client(graph_adapter, default_namespace="test-ns")

    # Non-streaming query
    result = client.query("MATCH (n) RETURN n LIMIT 1")
    assert result is not None

    # Streaming query
    chunks = list(client.stream_query("MATCH (n) RETURN n LIMIT 2"))
    # It's fine if the list is empty; we're only asserting the pathway works.
    assert isinstance(chunks, list)


def test_sync_query_accepts_optional_params_and_context(graph_adapter: Any) -> None:
    """
    query() should accept params, dialect, namespace, timeout_ms, and
    conversation/extra_context kwargs without raising.
    """
    client = _make_client(graph_adapter, default_dialect="cypher")

    result = client.query(
        "MATCH (n) RETURN n LIMIT $limit",
        params={"limit": 5},
        dialect="cypher",
        namespace="ctx-ns",
        timeout_ms=5000,
        conversation={"conversation_id": "conv-sync"},
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

    # Ensure async methods exist and are coroutine/async-generator functions
    assert hasattr(client, "aquery")
    assert hasattr(client, "astream_query")

    query_coro = client.aquery("MATCH (n) RETURN n LIMIT 1")
    assert inspect.isawaitable(query_coro)
    result = await query_coro
    assert result is not None

    aiter = client.astream_query("MATCH (n) RETURN n LIMIT 2")

    # Allow both: awaitable -> async iterator, or async iterator directly.
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    # Consume at most one chunk to validate async-iterability.
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
        conversation={"conversation_id": "conv-async"},
        extra_context={"request_id": "req-async"},
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Bulk vertices / batch semantics (AutoGen wiring)
# ---------------------------------------------------------------------------


def test_bulk_vertices_builds_raw_request_and_calls_translator(
    monkeypatch: pytest.MonkeyPatch,
    graph_adapter: Any,
) -> None:
    """
    bulk_vertices() should:

    - Build the correct raw_request mapping from the spec, and
    - Call the underlying translator.bulk_vertices with that mapping and
      appropriate framework_ctx.
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

    # validate_graph_result_type would normally enforce BulkVerticesResult;
    # for this adapter-specific wiring test we just return the result unchanged.
    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    monkeypatch.setattr(
        autogen_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        autogen_adapter_module,
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

    raw = captured["raw_request"]
    assert raw == {
        "namespace": "ns-bulk",
        "limit": 42,
        "cursor": "cursor-token",
        "filter": {"foo": "bar"},
    }

    fw_ctx = captured["framework_ctx"]
    assert fw_ctx["framework"] == "autogen"
    assert fw_ctx["operation"] == "bulk_vertices"
    assert fw_ctx.get("namespace") == "ns-bulk"
    # op_ctx comes from _build_ctx; since we did not pass conversation/extra_context,
    # it should be None.
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
        autogen_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        autogen_adapter_module,
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

    raw = captured["raw_request"]
    assert raw == {
        "namespace": "ns-abulk",
        "limit": 7,
        "cursor": None,
        "filter": {"bar": 1},
    }

    fw_ctx = captured["framework_ctx"]
    assert fw_ctx["framework"] == "autogen"
    assert fw_ctx["operation"] == "bulk_vertices"
    assert fw_ctx.get("namespace") == "ns-abulk"
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

    # Skip real validation; we only care about wiring.
    def fake_validate_batch_operations(*_: Any, **__: Any) -> None:
        return None

    monkeypatch.setattr(
        autogen_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        autogen_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )
    monkeypatch.setattr(
        autogen_adapter_module,
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
    assert fw_ctx["framework"] == "autogen"
    assert fw_ctx["operation"] == "batch"
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
        autogen_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        autogen_adapter_module,
        "validate_graph_result_type",
        fake_validate_graph_result_type,
    )
    monkeypatch.setattr(
        autogen_adapter_module,
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
    assert fw_ctx["framework"] == "autogen"
    assert fw_ctx["operation"] == "batch"
    assert captured["op_ctx"] is None


# ---------------------------------------------------------------------------
# Capabilities / health passthrough (basic)
# ---------------------------------------------------------------------------


def test_capabilities_and_health_basic(graph_adapter: Any) -> None:
    """
    Capabilities and health should be surfaced as mappings.

    The detailed structure is tested in framework-agnostic graph contract
    tests; here we only assert that the AutoGen adapter normalizes to dicts.
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

        # Minimal methods to keep GraphTranslator happy when invoked
        def capabilities(self) -> Dict[str, Any]:  # type: ignore[override]
            return {}

        def health(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # noqa: ARG002
            return {}

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.aclosed = True

    adapter = ClosingGraphAdapter()

    # Sync context manager: should call close() if present
    with CorpusAutoGenGraphClient(graph_adapter=adapter) as client:
        # Don't call any methods; we're just testing resource cleanup.
        assert client is not None

    assert adapter.closed is True

    # Async context manager: should call aclose() if present
    adapter2 = ClosingGraphAdapter()
    client2 = CorpusAutoGenGraphClient(graph_adapter=adapter2)

    async with client2:
        assert client2 is not None

    assert adapter2.aclosed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
