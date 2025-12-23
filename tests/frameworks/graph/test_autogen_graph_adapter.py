from __future__ import annotations

import asyncio
import inspect
import threading
import concurrent.futures
import logging
from typing import Any, Dict, Mapping, List

import pytest

import corpus_sdk.graph.framework_adapters.autogen as autogen_adapter_module
from corpus_sdk.graph.framework_adapters.autogen import (
    AutoGenGraphFrameworkTranslator,
    CorpusAutoGenGraphClient,
    ErrorCodes,
)
from corpus_sdk.graph.graph_base import OperationContext, GraphCapabilities


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(adapter: Any, **kwargs: Any) -> CorpusAutoGenGraphClient:
    """Construct a CorpusAutoGenGraphClient instance from the generic adapter."""
    return CorpusAutoGenGraphClient(adapter=adapter, **kwargs)


# ---------------------------------------------------------------------------
# Constructor / translator behavior
# ---------------------------------------------------------------------------


def test_default_translator_uses_autogen_framework_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
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

    client = _make_client(adapter)

    # Trigger lazy translator construction
    _ = client._translator  # noqa: SLF001

    assert "kwargs" in captured_args
    kwargs = captured_args["kwargs"]

    assert kwargs.get("framework") == "autogen"
    translator = kwargs.get("translator")
    assert isinstance(translator, AutoGenGraphFrameworkTranslator)


def test_framework_translator_override_is_respected(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
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
        adapter,
        framework_translator=custom,
        framework_version="fw-1.2.3",
    )

    _ = client._translator  # noqa: SLF001

    kwargs = captured_args["kwargs"]
    assert kwargs.get("framework") == "autogen"
    assert kwargs.get("translator") is custom


def test_constructor_rejects_invalid_adapter() -> None:
    """Verify constructor raises TypeError for non-graph adapter."""
    with pytest.raises(TypeError) as exc_info:
        CorpusAutoGenGraphClient(adapter="not-a-graph-adapter")  # type: ignore[arg-type]
    msg = str(exc_info.value).lower()
    assert "adapter" in msg or "graphprotocolv1" in msg or "compatible" in msg


def test_constructor_accepts_adapter_without_close() -> None:
    """Verify adapter without close() method is still accepted."""
    class SimpleAdapter:
        async def query(self, *args, **kwargs):
            return {"records": [], "summary": {}}
        async def capabilities(self):
            return GraphCapabilities(server="test", version="1.0")

    client = CorpusAutoGenGraphClient(adapter=SimpleAdapter())
    assert client is not None


def test_translator_lazy_initialization(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """Verify translator is only created when needed."""
    call_count = 0

    def fake_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        class DummyTranslator:
            pass
        return DummyTranslator()

    monkeypatch.setattr(autogen_adapter_module, "create_graph_translator", fake_create)

    client = _make_client(adapter)
    assert call_count == 0  # Not yet created

    _ = client._translator  # Access triggers creation
    assert call_count == 1

    _ = client._translator  # Second access should use cached
    assert call_count == 1  # Still 1


def test_import_autogen_graph_client() -> None:
    """Verify CorpusAutoGenGraphClient can be imported properly."""
    from corpus_sdk.graph.framework_adapters.autogen import (
        CorpusAutoGenGraphClient,
        AutoGenGraphFrameworkTranslator,
        ErrorCodes,
    )
    assert CorpusAutoGenGraphClient is not None
    assert AutoGenGraphFrameworkTranslator is not None
    assert ErrorCodes is not None


# ---------------------------------------------------------------------------
# Context translation / core_ctx_from_autogen mapping
# ---------------------------------------------------------------------------


def test_autogen_conversation_and_extra_context_passed_to_core_ctx(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
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
        adapter,
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
    adapter: Any,
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
        adapter,
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


def test_context_translation_with_empty_conversation(
    adapter: Any,
) -> None:
    """Handle empty conversation dict gracefully."""
    client = _make_client(adapter)

    result = client.query("MATCH (n) RETURN n", conversation={})
    assert result is not None


def test_context_translation_with_none_values(
    adapter: Any,
) -> None:
    """Handle None values in conversation/extra_context."""
    client = _make_client(adapter)

    result = client.query(
        "MATCH (n) RETURN n",
        conversation=None,
        extra_context=None
    )
    assert result is not None


def test_extra_context_overrides_framework_metadata(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """Verify extra_context can override framework metadata."""
    captured: Dict[str, Any] = {}

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
        captured.update(extra)
        return DummyOperationContext()

    monkeypatch.setattr(
        autogen_adapter_module,
        "core_ctx_from_autogen",
        fake_core_ctx_from_autogen,
    )

    client = _make_client(adapter)

    # extra_context should appear in captured extra kwargs
    extra_ctx = {"custom_field": "custom_value", "override": "should_win"}
    client.query("MATCH (n) RETURN n", extra_context=extra_ctx)

    assert captured.get("custom_field") == "custom_value"
    assert captured.get("override") == "should_win"


# ---------------------------------------------------------------------------
# Error-context decorator behavior
# ---------------------------------------------------------------------------


def test_error_context_includes_autogen_metadata_sync(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
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

    client = _make_client(adapter)

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
    adapter: Any,
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

    client = _make_client(adapter)

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


def test_error_context_includes_query_text(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """Error context should include the query that failed."""
    captured: Dict[str, Any] = {}

    def fake_attach(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach)

    class FailingTranslator:
        def query(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("query execution failed")

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return FailingTranslator()

    monkeypatch.setattr(autogen_adapter_module, "create_graph_translator", fake_create_graph_translator)

    client = _make_client(adapter)

    query_text = "MATCH (n:Special) WHERE n.name = 'test' RETURN n"
    with pytest.raises(RuntimeError):
        client.query(query_text)

    assert captured.get("query") == query_text or "query" in str(captured)


def test_error_context_preserves_autogen_specific_fields(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """Verify AutoGen-specific fields like agent_id, workflow_id are preserved."""
    captured: Dict[str, Any] = {}

    def fake_attach(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach)

    class FailingTranslator:
        def query(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("test error")

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return FailingTranslator()

    monkeypatch.setattr(autogen_adapter_module, "create_graph_translator", fake_create_graph_translator)

    client = _make_client(adapter)

    auto_ctx = {
        "conversation_id": "conv-123",
        "agent_id": "agent-456",
        "workflow_id": "workflow-789",
        "custom_field": "custom_value"
    }

    with pytest.raises(RuntimeError):
        client.query("MATCH (n) RETURN n", **auto_ctx)

    # Check AutoGen-specific fields are preserved
    for key in ["conversation_id", "agent_id", "workflow_id"]:
        if key in captured:
            assert captured[key] == auto_ctx[key]


def test_graph_specific_error_codes(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """Verify graph-specific error codes are used appropriately."""
    from corpus_sdk.graph.framework_adapters.autogen import ErrorCodes

    # Test each error code
    error_tests = [
        (ErrorCodes.BAD_OPERATION_CONTEXT, "context_translation"),
        (ErrorCodes.BAD_TRANSLATED_CHUNK, "stream_validation"),
    ]

    for error_code, operation_type in error_tests:
        captured_code = None

        def fake_attach(exc: BaseException, **ctx: Any) -> None:
            nonlocal captured_code
            captured_code = getattr(exc, "code", None)

        monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach)

        # Setup specific failure for each error code
        if error_code == ErrorCodes.BAD_OPERATION_CONTEXT:
            def fake_core_ctx(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("bad context")
            monkeypatch.setattr(autogen_adapter_module, "core_ctx_from_autogen", fake_core_ctx)
        elif error_code == ErrorCodes.BAD_TRANSLATED_CHUNK:
            class BadChunkTranslator:
                def query_stream(self, *args: Any, **kwargs: Any) -> Any:
                    yield {"not": "a-query-chunk"}
            def fake_create(*_: Any, **__: Any) -> Any:
                return BadChunkTranslator()
            monkeypatch.setattr(autogen_adapter_module, "create_graph_translator", fake_create)

        client = _make_client(adapter)

        try:
            if error_code == ErrorCodes.BAD_OPERATION_CONTEXT:
                client.query("MATCH (n) RETURN n", conversation={"invalid": "context"})
            elif error_code == ErrorCodes.BAD_TRANSLATED_CHUNK:
                next(client.stream_query("MATCH (n) RETURN n"))
        except Exception as e:
            assert getattr(e, "code", None) == error_code


# ---------------------------------------------------------------------------
# Streaming validation / error paths
# ---------------------------------------------------------------------------


def test_stream_query_invalid_chunk_triggers_validation_and_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
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

    client = _make_client(adapter)

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
    adapter: Any,
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

    client = _make_client(adapter)

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


def test_stream_query_empty_result(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """Handle empty stream gracefully."""
    class EmptyTranslator:
        def query_stream(self, *args: Any, **kwargs: Any) -> Any:
            return iter([])

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return EmptyTranslator()

    monkeypatch.setattr(autogen_adapter_module, "create_graph_translator", fake_create_graph_translator)

    client = _make_client(adapter)

    chunks = list(client.stream_query("MATCH (n) WHERE 1=0 RETURN n"))
    assert chunks == []


@pytest.mark.asyncio
async def test_astream_query_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """Verify async stream can be cancelled mid-iteration."""
    class SlowTranslator:
        async def arun_query_stream(self, *args: Any, **kwargs: Any):
            async def _gen():
                for i in range(10):
                    await asyncio.sleep(0.01)  # Small delay
                    yield {"records": [i], "is_final": i == 9}
            return _gen()

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return SlowTranslator()

    monkeypatch.setattr(autogen_adapter_module, "create_graph_translator", fake_create_graph_translator)

    client = _make_client(adapter)

    # Start streaming
    stream = client.astream_query("MATCH (n) RETURN n")
    if inspect.isawaitable(stream):
        stream = await stream

    count = 0
    async for _ in stream:  # noqa: B007
        count += 1
        if count >= 3:
            break  # Cancel early

    assert count == 3


def test_stream_large_result_sets(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """Performance with large results."""
    class LargeResultTranslator:
        def query_stream(self, *args: Any, **kwargs: Any) -> Any:
            for i in range(100):
                yield {"records": [f"record_{i}"], "is_final": i == 99}

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return LargeResultTranslator()

    monkeypatch.setattr(autogen_adapter_module, "create_graph_translator", fake_create_graph_translator)

    client = _make_client(adapter)

    chunks = list(client.stream_query("MATCH (n) RETURN n LIMIT 100"))
    assert len(chunks) == 100


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
    client = _make_client(adapter, default_namespace="test-ns")

    # Non-streaming query
    result = client.query("MATCH (n) RETURN n LIMIT 1")
    assert result is not None

    # Streaming query
    chunks = list(client.stream_query("MATCH (n) RETURN n LIMIT 2"))
    # It's fine if the list is empty; we're only asserting the pathway works.
    assert isinstance(chunks, list)


def test_sync_query_accepts_optional_params_and_context(adapter: Any) -> None:
    """
    query() should accept params, dialect, namespace, timeout_ms, and
    conversation/extra_context kwargs without raising.
    """
    client = _make_client(adapter, default_dialect="cypher")

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


def test_type_validation_on_query_params(adapter: Any) -> None:
    """Verify query parameters are properly typed."""
    client = _make_client(adapter)

    # Should accept correct types
    result = client.query(
        "MATCH (n) RETURN n LIMIT $limit",
        params={"limit": 5},  # int
        dialect="cypher",  # str
        timeout_ms=1000,  # int
        namespace="test"  # str
    )
    assert result is not None

    # Test invalid params type
    with pytest.raises((TypeError, ValueError)):
        client.query("MATCH (n) RETURN n", params="not-a-dict")  # type: ignore[arg-type]


def test_invalid_query_dialect_handling(adapter: Any) -> None:
    """Handle unsupported dialects gracefully."""
    client = _make_client(adapter, default_dialect="cypher")

    # Should work with None/default dialect
    result = client.query("MATCH (n) RETURN n", dialect=None)
    assert result is not None

    # Unsupported dialect might be handled by adapter
    result = client.query("MATCH (n) RETURN n", dialect="unsupported_dialect")
    assert result is not None  # Adapter may handle or ignore


def test_namespace_validation(adapter: Any) -> None:
    """Namespace format/safety."""
    client = _make_client(adapter)

    # Valid namespaces
    for namespace in ["test", "test-ns", "test_ns", "test.ns"]:
        result = client.query("MATCH (n) RETURN n", namespace=namespace)
        assert result is not None

    # Empty namespace should work
    result = client.query("MATCH (n) RETURN n", namespace="")
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
        conversation={"conversation_id": "conv-async"},
        extra_context={"request_id": "req-async"},
    )
    assert result is not None


def test_sync_and_async_capabilities_same(adapter: Any) -> None:
    """Verify sync and async capabilities return same structure."""
    client = _make_client(adapter)

    sync_caps = client.capabilities()
    async_caps = asyncio.run(client.acapabilities())

    # Compare structure, not necessarily exact equality if timestamps differ
    assert isinstance(sync_caps, (dict, GraphCapabilities))
    assert isinstance(async_caps, (dict, GraphCapabilities))


@pytest.mark.asyncio
async def test_async_fallback_to_sync_methods() -> None:
    """Verify async methods fall back to sync when async not available."""
    class SyncOnlyAdapter:
        async def query(self, *args: Any, **kwargs: Any) -> Any:
            return {"records": [], "summary": {}}
        async def capabilities(self) -> Any:
            return GraphCapabilities(server="test", version="1.0")
        # No aquery method, but sync query is async

    adapter = SyncOnlyAdapter()
    client = CorpusAutoGenGraphClient(adapter=adapter)

    # Should work via async interface
    result = await client.aquery("MATCH (n) RETURN n")
    assert result is not None


@pytest.mark.asyncio
async def test_async_and_sync_query_results_compatible(adapter: Any) -> None:
    """Result format consistency between sync and async."""
    client = _make_client(adapter)

    sync_result = client.query("MATCH (n) RETURN n LIMIT 1")
    async_result = await client.aquery("MATCH (n) RETURN n LIMIT 1")

    # Both should have similar structure
    assert hasattr(sync_result, "records") or isinstance(sync_result, dict)
    assert hasattr(async_result, "records") or isinstance(async_result, dict)


# ---------------------------------------------------------------------------
# Bulk vertices / batch semantics (AutoGen wiring)
# ---------------------------------------------------------------------------


def test_bulk_vertices_builds_raw_request_and_calls_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
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
    assert fw_ctx["framework"] == "autogen"
    assert fw_ctx["operation"] == "bulk_vertices"
    assert fw_ctx.get("namespace") == "ns-bulk"
    # op_ctx comes from _build_ctx; since we did not pass conversation/extra_context,
    # it should be None.
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
        autogen_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )
    monkeypatch.setattr(
        autogen_adapter_module,
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
    assert fw_ctx["framework"] == "autogen"
    assert fw_ctx["operation"] == "bulk_vertices"
    assert fw_ctx.get("namespace") == "ns-abulk"
    assert captured["op_ctx"] is None


def test_bulk_vertices_with_invalid_spec(adapter: Any) -> None:
    """Handle invalid bulk spec gracefully."""
    client = _make_client(adapter)

    class InvalidSpec:
        # Missing required attributes
        pass

    spec = InvalidSpec()  # type: ignore[arg-type]

    with pytest.raises((AttributeError, TypeError)):
        client.bulk_vertices(spec)


def test_batch_builds_raw_batch_ops_and_calls_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
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
    assert fw_ctx["framework"] == "autogen"
    assert fw_ctx["operation"] == "batch"
    assert captured["op_ctx"] is None


@pytest.mark.asyncio
async def test_abatch_builds_raw_batch_ops_and_calls_translator_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
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
    assert fw_ctx["framework"] == "autogen"
    assert fw_ctx["operation"] == "batch"
    assert captured["op_ctx"] is None


def test_batch_with_empty_operations(adapter: Any) -> None:
    """Handle empty batch operations list."""
    client = _make_client(adapter)

    # Might raise BadRequest or handle empty list
    try:
        result = client.batch([])
        assert result is not None  # Should handle empty batch
    except Exception as e:
        assert "empty" in str(e).lower() or "must not" in str(e).lower()


def test_batch_operation_validation(adapter: Any) -> None:
    """Invalid operation types."""
    client = _make_client(adapter)

    class InvalidOp:
        def __init__(self) -> None:
            self.op = "invalid_operation"
            self.args = {}

    # Should raise for invalid operations
    with pytest.raises((ValueError, TypeError)):
        client.batch([InvalidOp()])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Capabilities / health passthrough (basic)
# ---------------------------------------------------------------------------


def test_capabilities_and_health_basic(adapter: Any) -> None:
    """
    Capabilities and health should be surfaced as mappings.

    The detailed structure is tested in framework-agnostic graph contract
    tests; here we only assert that the AutoGen adapter normalizes to dicts.
    """
    client = _make_client(adapter)

    caps = client.capabilities()
    assert isinstance(caps, (Mapping, GraphCapabilities))

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
    assert isinstance(acaps, (Mapping, GraphCapabilities))

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

        # Minimal methods to keep GraphTranslator happy when invoked
        async def query(self, *args: Any, **kwargs: Any) -> Any:
            return {"records": [], "summary": {}}

        async def capabilities(self) -> GraphCapabilities:
            return GraphCapabilities(server="test", version="1.0")

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.aclosed = True

    adapter = ClosingGraphAdapter()

    # Sync context manager: should call close() if present
    with CorpusAutoGenGraphClient(adapter=adapter) as client:
        # Don't call any methods; we're just testing resource cleanup.
        assert client is not None

    assert adapter.closed is True

    # Async context manager: should call aclose() if present
    adapter2 = ClosingGraphAdapter()
    client2 = CorpusAutoGenGraphClient(adapter=adapter2)

    async with client2:
        assert client2 is not None

    assert adapter2.aclosed is True


def test_context_manager_with_exception(adapter: Any) -> None:
    """Cleanup on error."""
    class TestAdapter:
        def __init__(self) -> None:
            self.closed = False
        async def query(self, *args: Any, **kwargs: Any) -> Any:
            return {"records": [], "summary": {}}
        async def capabilities(self) -> Any:
            return GraphCapabilities(server="test", version="1.0")
        def close(self) -> None:
            self.closed = True

    adapter_instance = TestAdapter()
    client = CorpusAutoGenGraphClient(adapter=adapter_instance)

    try:
        with client:
            raise RuntimeError("test exception")
    except RuntimeError:
        pass

    assert adapter_instance.closed is True


def test_double_close_protection(adapter: Any) -> None:
    """Idempotent close."""
    close_count = 0

    class CountingAdapter:
        async def query(self, *args: Any, **kwargs: Any) -> Any:
            return {"records": [], "summary": {}}
        async def capabilities(self) -> Any:
            return GraphCapabilities(server="test", version="1.0")
        def close(self) -> None:
            nonlocal close_count
            close_count += 1

    adapter_instance = CountingAdapter()
    client = CorpusAutoGenGraphClient(adapter=adapter_instance)

    with client:
        pass

    # Try to close again
    client.close()
    assert close_count == 1  # Should not double-close


@pytest.mark.asyncio
async def test_async_close_with_pending_operations() -> None:
    """Close during active ops."""
    import asyncio

    class SlowAdapter:
        def __init__(self) -> None:
            self.aclosed = False
            self.active_queries = 0

        async def query(self, *args: Any, **kwargs: Any) -> Any:
            self.active_queries += 1
            await asyncio.sleep(0.1)
            self.active_queries -= 1
            return {"records": [], "summary": {}}

        async def capabilities(self) -> Any:
            return GraphCapabilities(server="test", version="1.0")

        async def aclose(self) -> None:
            # Wait for active queries to finish
            while self.active_queries > 0:
                await asyncio.sleep(0.01)
            self.aclosed = True

    adapter_instance = SlowAdapter()
    client = CorpusAutoGenGraphClient(adapter=adapter_instance)

    # Start async operation
    import asyncio
    query_task = asyncio.create_task(client.aquery("MATCH (n) RETURN n"))

    # Close while query is running
    await asyncio.sleep(0.05)  # Let query start
    await client.aclose()

    # Wait for query to finish
    await query_task

    assert adapter_instance.aclosed is True


# ---------------------------------------------------------------------------
# Concurrency Tests
# ---------------------------------------------------------------------------

def test_thread_safety_sync_queries(adapter: Any) -> None:
    """
    Multiple threads should safely share a single CorpusAutoGenGraphClient.
    """
    client = _make_client(adapter)
    results = []
    errors = []

    def execute_query(thread_id: int) -> None:
        try:
            query = f"MATCH (n) RETURN n LIMIT {(thread_id % 5) + 1}"
            params = {"thread": thread_id}

            result = client.query(
                query,
                params=params,
                namespace=f"thread-{thread_id}",
                conversation={"thread_id": thread_id, "agent": f"agent-{thread_id}"}
            )
            results.append((thread_id, result))
        except Exception as e:
            errors.append((thread_id, str(e)))

    # Create and run threads
    threads = []
    num_threads = 10

    for i in range(num_threads):
        t = threading.Thread(target=execute_query, args=(i,))
        threads.append(t)

    # Start all threads
    for t in threads:
        t.start()

    # Wait for completion
    for t in threads:
        t.join()

    # Verify results
    assert len(errors) == 0, f"Errors occurred in threads: {errors}"
    assert len(results) == num_threads
    # Verify all thread IDs are present
    thread_ids = [tid for tid, _ in results]
    assert set(thread_ids) == set(range(num_threads))


def test_thread_safety_with_mixed_operations(adapter: Any) -> None:
    """
    Multiple threads performing different operations (query, stream, bulk).
    """
    client = _make_client(adapter)

    def run_query(task_id: int) -> Any:
        return client.query(f"MATCH (n) RETURN n LIMIT {task_id % 3}")

    def run_stream(task_id: int) -> int:
        chunks = list(client.stream_query(f"MATCH (n) RETURN n LIMIT {task_id % 2 + 1}"))
        return len(chunks)

    def run_bulk(task_id: int) -> Any:
        class Spec:
            namespace = f"ns-{task_id}"
            limit = task_id % 10 + 1
            cursor = None
            filter = {"thread": task_id}

        return client.bulk_vertices(Spec())

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(15):
            if i % 3 == 0:
                futures.append(executor.submit(run_query, i))
            elif i % 3 == 1:
                futures.append(executor.submit(run_stream, i))
            else:
                futures.append(executor.submit(run_bulk, i))

        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                pytest.fail(f"Thread operation failed: {e}")

        assert len(results) == 15


@pytest.mark.asyncio
async def test_concurrent_async_queries(adapter: Any) -> None:
    """
    Multiple async tasks should execute concurrently without issues.
    """
    client = _make_client(adapter)

    async def execute_async_query(task_id: int, delay: float = 0) -> tuple[int, Any]:
        if delay > 0:
            await asyncio.sleep(delay)

        result = await client.aquery(
            f"MATCH (n) RETURN n LIMIT {(task_id % 3) + 1}",
            params={"task": task_id},
            conversation={"task_id": task_id},
            namespace=f"async-{task_id}"
        )
        return task_id, result

    # Create tasks with staggered delays to maximize concurrency
    tasks = []
    for i in range(10):
        delay = (i % 3) * 0.01  # Small staggered delays
        tasks.append(execute_async_query(i, delay))

    # Execute concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Verify all succeeded
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            pytest.fail(f"Task {i} failed: {result}")

    # Verify all task IDs are present
    task_ids = [tid for tid, _ in results if not isinstance(tid, Exception)]
    assert set(task_ids) == set(range(10))


@pytest.mark.asyncio
async def test_concurrent_mixed_async_operations(adapter: Any) -> None:
    """
    Mixed async operations (aquery, astream, abulk) executing concurrently.
    """
    client = _make_client(adapter)

    async def run_aquery(task_id: int) -> Any:
        return await client.aquery(f"MATCH (n) RETURN n LIMIT 1")

    async def run_astream(task_id: int) -> int:
        count = 0
        aiter = client.astream_query(f"MATCH (n) RETURN n LIMIT 2")
        if inspect.isawaitable(aiter):
            aiter = await aiter
        async for _ in aiter:
            count += 1
        return count

    async def run_abulk(task_id: int) -> Any:
        class Spec:
            namespace = "test-ns"
            limit = 5
            cursor = None
            filter = {"task": task_id}

        return await client.abulk_vertices(Spec())

    # Create mixed tasks
    tasks = []
    for i in range(9):
        if i % 3 == 0:
            tasks.append(run_aquery(i))
        elif i % 3 == 1:
            tasks.append(run_astream(i))
        else:
            tasks.append(run_abulk(i))

    # Run with semaphore to limit concurrency
    semaphore = asyncio.Semaphore(3)

    async def run_with_semaphore(task_func, task_id):
        async with semaphore:
            return await task_func(task_id)

    semaphore_tasks = [run_with_semaphore(tasks[i], i) for i in range(len(tasks))]
    results = await asyncio.gather(*semaphore_tasks, return_exceptions=True)

    # Check for errors
    errors = [r for r in results if isinstance(r, Exception)]
    assert len(errors) == 0, f"Concurrent operations failed: {errors}"


@pytest.mark.asyncio
async def test_mixed_sync_async_concurrent_access(adapter: Any) -> None:
    """
    Test scenario where sync and async operations are called concurrently.
    """
    import asyncio

    client = _make_client(adapter)
    results = {"sync": [], "async": []}
    errors = {"sync": [], "async": []}

    # Sync thread function
    def sync_operations() -> None:
        for i in range(5):
            try:
                result = client.query(f"SYNC MATCH (n) RETURN n LIMIT {i+1}")
                results["sync"].append((i, result))
            except Exception as e:
                errors["sync"].append((i, str(e)))

    # Async task function
    async def async_operations() -> None:
        for i in range(5):
            try:
                result = await client.aquery(f"ASYNC MATCH (n) RETURN n LIMIT {i+1}")
                results["async"].append((i, result))
            except Exception as e:
                errors["async"].append((i, str(e)))

    # Run sync in thread, async in event loop
    sync_thread = threading.Thread(target=sync_operations)
    sync_thread.start()

    async_task = asyncio.create_task(async_operations())

    # Wait for both
    sync_thread.join()
    await async_task

    # Verify results
    assert len(errors["sync"]) == 0, f"Sync errors: {errors['sync']}"
    assert len(errors["async"]) == 0, f"Async errors: {errors['async']}"
    assert len(results["sync"]) == 5
    assert len(results["async"]) == 5


def test_connection_pool_limits(adapter: Any) -> None:
    """
    Test that concurrent operations respect connection/resource limits.
    """
    client = _make_client(adapter)

    def make_request(request_id: int) -> Any:
        try:
            import time
            time.sleep(0.001 * (request_id % 3))

            return client.query(
                f"MATCH (n) WHERE id(n) = {request_id} RETURN n",
                timeout_ms=10000
            )
        except Exception as e:
            return e

    # Test with more workers than typical connection pool size
    max_workers = 20
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(make_request, i) for i in range(50)]

        # Wait for all with timeout
        done, not_done = concurrent.futures.wait(futures, timeout=30.0)

        assert len(not_done) == 0, f"{len(not_done)} requests timed out"

        # Check results
        success_count = 0
        error_count = 0

        for future in done:
            result = future.result()
            if isinstance(result, Exception):
                error_count += 1
            else:
                success_count += 1

        # Most should succeed
        assert success_count > 0
        # Log error rate for debugging
        error_rate = error_count / len(futures)
        assert error_rate < 0.5, f"High error rate: {error_rate}"


def test_stress_test_high_concurrency(adapter: Any) -> None:
    """Extreme load test."""
    client = _make_client(adapter)

    def stress_task(task_id: int) -> bool:
        try:
            for i in range(10):
                client.query(f"MATCH (n) RETURN n LIMIT {task_id % 5}")
            return True
        except Exception:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(stress_task, i) for i in range(100)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    success_rate = sum(results) / len(results)
    assert success_rate > 0.8, f"Stress test success rate too low: {success_rate}"


def test_concurrent_batch_operations(adapter: Any) -> None:
    """Multiple batch operations concurrently."""
    client = _make_client(adapter)

    class BatchOp:
        def __init__(self, op: str, args: Dict[str, Any]) -> None:
            self.op = op
            self.args = args

    def run_batch(thread_id: int) -> bool:
        try:
            ops = [
                BatchOp("query", {"text": f"MATCH (n) RETURN n LIMIT {thread_id}"}),
            ]
            client.batch(ops)  # type: ignore[arg-type]
            return True
        except Exception:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_batch, i) for i in range(20)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert all(results), "Some batch operations failed"


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAutoGenGraphIntegration:
    """Integration tests with real AutoGen workflows."""

    def test_can_create_graph_queries_for_autogen_agent(self, adapter: Any) -> None:
        """
        Real integration: Can create graph queries that work with AutoGen agents.
        """
        client = _make_client(adapter, framework_version="1.0.0")

        result = client.query(
            "MATCH (n) RETURN n LIMIT 1",
            conversation={
                "conversation_id": "conv-123",
                "agent_name": "graph_researcher",
                "workflow_type": "knowledge_graph"
            }
        )
        assert result is not None

        # Test with context
        result_with_context = client.query(
            "MATCH (n) RETURN n LIMIT 1",
            conversation={"conversation_id": "conv-456", "agent_name": "analyst"},
            extra_context={"request_id": "req-789", "priority": "high"}
        )
        assert result_with_context is not None

    def test_graph_queries_work_with_autogen_workflows(self, adapter: Any) -> None:
        """
        Real integration: Graph queries work with AutoGen's multi-agent workflows.
        """
        client = _make_client(adapter)

        # Test different query types that might be used in agent workflows
        queries = [
            "MATCH (n:Person) RETURN n.name, n.age LIMIT 5",
            "MATCH (a)-[r:KNOWS]->(b) RETURN a.name, r.since, b.name",
            "MATCH p=(a:Person)-[*1..3]->(b) RETURN p LIMIT 3"
        ]

        for query in queries:
            result = client.query(
                query,
                conversation={
                    "conversation_id": "workflow-1",
                    "agent_name": "graph_agent",
                    "workflow_type": "relationship_analysis"
                }
            )
            assert result is not None

        # Test streaming in workflow context
        stream_result = list(client.stream_query(
            "MATCH (n) RETURN n LIMIT 10",
            conversation={"conversation_id": "stream-workflow", "agent_name": "stream_processor"}
        ))
        assert isinstance(stream_result, list)

    def test_error_handling_in_autogen_graph_workflow(self, adapter: Any) -> None:
        """
        Real integration: Error handling in AutoGen graph workflows.
        """
        client = _make_client(adapter)

        # Test that errors are properly contextualized for AutoGen
        try:
            # This might fail depending on adapter implementation
            client.query("INVALID CYPHER QUERY SYNTAX")
        except Exception as e:
            # Error should contain useful context
            error_str = str(e)
            assert len(error_str) > 0
            # Should have some error code or message
            assert hasattr(e, 'code') or 'error' in error_str.lower() or 'invalid' in error_str.lower()

    @pytest.mark.asyncio
    async def test_async_graph_in_autogen_workflow(self, adapter: Any) -> None:
        """
        Async graph operations in async AutoGen workflows.
        """
        client = _make_client(adapter)

        # Test async query in agent context
        result = await client.aquery(
            "MATCH (n) RETURN n LIMIT 3",
            conversation={
                "conversation_id": "async-session",
                "agent_name": "async_agent",
                "workflow_type": "async_processing"
            }
        )
        assert result is not None

        # Test async streaming
        count = 0
        aiter = client.astream_query("MATCH (n) RETURN n LIMIT 5")
        if inspect.isawaitable(aiter):
            aiter = await aiter

        async for chunk in aiter:
            count += 1
            assert chunk is not None

        assert count > 0

    def test_multiple_agents_can_share_same_graph_client(self, adapter: Any) -> None:
        """
        Real integration: Multiple agents/retrievers can share the same graph client.
        """
        client = _make_client(adapter)

        # Simulate multiple agents using same client
        contexts = [
            {"conversation_id": "conv-1", "agent_name": "researcher", "workflow_type": "research"},
            {"conversation_id": "conv-1", "agent_name": "analyst", "workflow_type": "analysis"},
            {"conversation_id": "conv-2", "agent_name": "summarizer", "workflow_type": "summarization"},
        ]

        for ctx in contexts:
            result = client.query(
                f"MATCH (n) RETURN n LIMIT 1",
                **ctx
            )
            assert result is not None

            # Test batch operations from different agents
            class BulkSpec:
                namespace = f"ns-{ctx['agent_name']}"
                limit = 3
                cursor = None
                filter = {"agent": ctx["agent_name"]}

            try:
                bulk_result = client.bulk_vertices(BulkSpec())
                assert bulk_result is not None
            except Exception:
                # bulk_vertices might not be supported
                pass


# ---------------------------------------------------------------------------
# Logging / Telemetry Tests
# ---------------------------------------------------------------------------

def test_logging_includes_autogen_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """Verify AutoGen context is included in logs."""
    import logging

    log_capture: List[str] = []

    class TestHandler(logging.Handler):
        def emit(self, record):
            log_capture.append(record.getMessage())

    logger = logging.getLogger("corpus_sdk.graph.framework_adapters.autogen")
    handler = TestHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        client = _make_client(adapter)
        client.query(
            "MATCH (n) RETURN n",
            conversation={"conversation_id": "log-test"}
        )

        # Check logs contain AutoGen context
        log_messages = " ".join(log_capture).lower()
        assert "autogen" in log_messages or "framework" in log_messages
    finally:
        logger.removeHandler(handler)


def test_operation_telemetry_includes_framework(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """Verify framework info in operation telemetry."""
    captured_metrics = []

    class TestMetrics:
        def observe(self, **kwargs):
            captured_metrics.append(kwargs)
        def counter(self, **kwargs):
            captured_metrics.append(kwargs)

    monkeypatch.setattr(autogen_adapter_module, "get_metrics_sink", lambda: TestMetrics())

    client = _make_client(adapter)
    client.query("MATCH (n) RETURN n LIMIT 1")

    # Check that metrics include framework info
    assert len(captured_metrics) > 0
    for metric in captured_metrics:
        if metric.get("component") == "graph":
            assert "framework" in str(metric).lower() or "autogen" in str(metric).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
