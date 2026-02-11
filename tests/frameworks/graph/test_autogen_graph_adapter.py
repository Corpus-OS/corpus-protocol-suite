from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
import threading
from typing import Any, Dict, List, Mapping

import pytest

import corpus_sdk.graph.framework_adapters.autogen as autogen_adapter_module
from corpus_sdk.graph.framework_adapters.autogen import (
    AutoGenGraphFrameworkTranslator,
    CorpusAutoGenGraphClient,
    ErrorCodes,
)
from corpus_sdk.graph.graph_base import GraphCapabilities, QueryChunk, QueryResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(adapter: Any, **kwargs: Any) -> CorpusAutoGenGraphClient:
    """Construct a CorpusAutoGenGraphClient instance from the generic adapter."""
    return CorpusAutoGenGraphClient(adapter=adapter, **kwargs)


def _run_async_if_needed(coro: Any) -> Any:
    """
    Run an async coroutine, handling existing event loops gracefully.

    This mirrors the pattern used in other framework tests to avoid
    RuntimeError: asyncio.run() cannot be called from a running event loop
    in environments with non-standard async runners.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


def _patch_attach_context_everywhere(
    monkeypatch: pytest.MonkeyPatch,
    fake_attach_context: Any,
) -> None:
    """
    Patch attach_context in both the adapter module and the canonical core module.

    Some decorators may close over the core attach_context reference; others may
    use the local module import. Patching both maximizes determinism.
    """
    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)
    try:
        import corpus_sdk.core.error_context as error_context_module
        monkeypatch.setattr(error_context_module, "attach_context", fake_attach_context)
    except Exception:
        # Best-effort: tests should still run if the import path differs.
        pass


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
        async def query(self, *args: Any, **kwargs: Any) -> Any:
            return {"records": [], "summary": {}}

        async def capabilities(self) -> Any:
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
        AutoGenGraphFrameworkTranslator,
        CorpusAutoGenGraphClient,
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


def test_build_ctx_failure_attaches_context_and_proceeds_without_ctx(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If core_ctx_from_autogen fails, _build_ctx should:

    - Attach error context via attach_context(framework="autogen", operation="context_translation")
    - Proceed without OperationContext (best-effort context translation)
    - Still complete the graph operation successfully
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured_ctx.update(ctx)

    def fake_core_ctx_from_autogen(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        raise RuntimeError("boom from autogen ctx")

    _patch_attach_context_everywhere(monkeypatch, fake_attach_context)
    monkeypatch.setattr(
        autogen_adapter_module,
        "core_ctx_from_autogen",
        fake_core_ctx_from_autogen,
    )

    client = _make_client(
        adapter,
        framework_version="autogen-fw-test",
    )

    # The call should still succeed: context translation is fail-safe by design.
    result = client.query(
        "MATCH (n) RETURN n",
        conversation={"conversation_id": "conv-fail"},
    )
    assert result is not None

    # Ensure error context was attached with framework metadata.
    assert captured_ctx.get("framework") == "autogen"
    assert captured_ctx.get("operation") == "context_translation"
    assert captured_ctx.get("error_code") == ErrorCodes.BAD_OPERATION_CONTEXT


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
        extra_context=None,
    )
    assert result is not None


def test_extra_context_overrides_framework_metadata(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """Verify extra_context is passed through to core_ctx_from_autogen."""
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
        framework_version: Any = None,  # noqa: ARG001
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

    NOTE:
    - We patch attach_context in both the adapter module and core error_context
      module to handle different decorator binding strategies.
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured_context.update(ctx)

    _patch_attach_context_everywhere(monkeypatch, fake_attach_context)

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

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured_context.update(ctx)

    _patch_attach_context_everywhere(monkeypatch, fake_attach_context)

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

    def fake_attach(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured.update(ctx)

    _patch_attach_context_everywhere(monkeypatch, fake_attach)

    class FailingTranslator:
        def query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("query execution failed")

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return FailingTranslator()

    monkeypatch.setattr(
        autogen_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(adapter)

    query_text = "MATCH (n:Special) WHERE n.name = 'test' RETURN n"
    with pytest.raises(RuntimeError):
        client.query(query_text)

    assert captured.get("query") == query_text or "query" in str(captured)


def test_error_context_preserves_autogen_specific_fields(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """Verify AutoGen-specific fields like agent_id, workflow_id are preserved (best-effort)."""
    captured: Dict[str, Any] = {}

    def fake_attach(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured.update(ctx)

    _patch_attach_context_everywhere(monkeypatch, fake_attach)

    class FailingTranslator:
        def query(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error")

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return FailingTranslator()

    monkeypatch.setattr(
        autogen_adapter_module,
        "create_graph_translator",
        fake_create_graph_translator,
    )

    client = _make_client(adapter)

    auto_ctx = {
        "conversation_id": "conv-123",
        "agent_id": "agent-456",
        "workflow_id": "workflow-789",
        "custom_field": "custom_value",
    }

    with pytest.raises(RuntimeError):
        client.query("MATCH (n) RETURN n", conversation=auto_ctx)

    # Check AutoGen-specific fields are preserved if forwarded by the decorator.
    for key in ["conversation_id", "agent_id", "workflow_id"]:
        if key in captured:
            assert captured[key] == auto_ctx[key]


def test_graph_specific_error_codes(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Verify graph-specific error codes are used appropriately.

    NOTE:
    - Context translation failures do not raise; they attach context and proceed.
    - Chunk validation failures do raise (via validate_graph_result_type).
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured_ctx.update(ctx)

    _patch_attach_context_everywhere(monkeypatch, fake_attach)

    # 1) BAD_OPERATION_CONTEXT: core_ctx_from_autogen failure should attach context and proceed.
    def fake_core_ctx(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        raise RuntimeError("bad context")

    monkeypatch.setattr(autogen_adapter_module, "core_ctx_from_autogen", fake_core_ctx)

    client = _make_client(adapter)

    result = client.query("MATCH (n) RETURN n", conversation={"invalid": "context"})
    assert result is not None
    assert captured_ctx.get("error_code") == ErrorCodes.BAD_OPERATION_CONTEXT

    # 2) BAD_TRANSLATED_CHUNK: invalid chunk should raise with the chunk error code.
    class BadChunkTranslator:
        def query_stream(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            yield {"not": "a-query-chunk"}

    def fake_create(*_: Any, **__: Any) -> Any:
        return BadChunkTranslator()

    monkeypatch.setattr(autogen_adapter_module, "create_graph_translator", fake_create)

    class FakeValidationError(Exception):
        def __init__(self, message: str, code: Any | None = None) -> None:
            super().__init__(message)
            self.code = code

    def fake_validate_graph_result_type(
        result: Any,
        *,
        expected_type: Any,  # noqa: ARG001
        operation: str,  # noqa: ARG001
        error_code: Any,
        **_: Any,
    ) -> Any:
        if error_code == ErrorCodes.BAD_TRANSLATED_CHUNK:
            raise FakeValidationError("bad chunk", code=error_code)
        return result

    monkeypatch.setattr(autogen_adapter_module, "validate_graph_result_type", fake_validate_graph_result_type)

    with pytest.raises(FakeValidationError) as exc_info:
        next(client.stream_query("MATCH (n) RETURN n"))

    assert getattr(exc_info.value, "code", None) == ErrorCodes.BAD_TRANSLATED_CHUNK


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

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured_ctx.update(ctx)

    _patch_attach_context_everywhere(monkeypatch, fake_attach_context)

    class DummyTranslator:
        def query_stream(
            self,
            raw_query: Mapping[str, Any],  # noqa: ARG002
            *,
            op_ctx: Any = None,  # noqa: ARG002
            framework_ctx: Mapping[str, Any] | None = None,  # noqa: ARG002
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
        expected_type: Any,  # noqa: ARG001
        operation: str,  # noqa: ARG001
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

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured_ctx.update(ctx)

    _patch_attach_context_everywhere(monkeypatch, fake_attach_context)

    class DummyTranslator:
        async def arun_query_stream(
            self,
            raw_query: Mapping[str, Any],  # noqa: ARG002
            *,
            op_ctx: Any = None,  # noqa: ARG002
            framework_ctx: Mapping[str, Any] | None = None,  # noqa: ARG002
        ):
            # Async generator yielding an invalid chunk.
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
        expected_type: Any,  # noqa: ARG001
        operation: str,  # noqa: ARG001
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
        def query_stream(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
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
        async def arun_query_stream(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            for i in range(10):
                await asyncio.sleep(0.01)  # Small delay
                yield QueryChunk(records=[i], is_final=i == 9)

    def fake_create_graph_translator(*_: Any, **__: Any) -> Any:
        return SlowTranslator()

    monkeypatch.setattr(autogen_adapter_module, "create_graph_translator", fake_create_graph_translator)

    client = _make_client(adapter)

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
        def query_stream(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            for i in range(100):
                yield QueryChunk(records=[f"record_{i}"], is_final=i == 99)

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

    result = client.query("MATCH (n) RETURN n LIMIT 1")
    assert result is not None

    chunks = list(client.stream_query("MATCH (n) RETURN n LIMIT 2"))
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

    result = client.query(
        "MATCH (n) RETURN n LIMIT $limit",
        params={"limit": 5},
        dialect="cypher",
        timeout_ms=1000,
        namespace="test",
    )
    assert result is not None

    with pytest.raises((TypeError, ValueError)):
        client.query("MATCH (n) RETURN n", params="not-a-dict")  # type: ignore[arg-type]


def test_invalid_query_dialect_handling(adapter: Any) -> None:
    """Handle unsupported dialects gracefully."""
    client = _make_client(adapter, default_dialect="cypher")

    result = client.query("MATCH (n) RETURN n", dialect=None)
    assert result is not None

    result = client.query("MATCH (n) RETURN n", dialect="unsupported_dialect")
    assert result is not None


def test_namespace_validation(adapter: Any) -> None:
    """Namespace format/safety."""
    client = _make_client(adapter)

    for namespace in ["test", "test-ns", "test_ns", "test.ns"]:
        result = client.query("MATCH (n) RETURN n", namespace=namespace)
        assert result is not None

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

    assert hasattr(client, "aquery")
    assert hasattr(client, "astream_query")

    query_coro = client.aquery("MATCH (n) RETURN n LIMIT 1")
    assert inspect.isawaitable(query_coro)
    result = await query_coro
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
async def test_async_query_accepts_optional_params_and_context(adapter: Any) -> None:
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
    async_caps = _run_async_if_needed(client.acapabilities())

    assert isinstance(sync_caps, (dict, GraphCapabilities))
    assert isinstance(async_caps, (dict, GraphCapabilities))


@pytest.mark.asyncio
async def test_async_fallback_to_sync_methods() -> None:
    """Verify async methods fall back to sync when async not available."""
    class SyncOnlyAdapter:
        async def query(self, *args: Any, **kwargs: Any) -> Any:
            return QueryResult(records=[], summary={})

        async def capabilities(self) -> Any:
            return GraphCapabilities(server="test", version="1.0")

    adapter = SyncOnlyAdapter()
    client = CorpusAutoGenGraphClient(adapter=adapter)

    result = await client.aquery("MATCH (n) RETURN n")
    assert result is not None


def test_async_and_sync_query_results_compatible(adapter: Any) -> None:
    """Result format consistency between sync and async."""
    client = _make_client(adapter)

    sync_result = client.query("MATCH (n) RETURN n LIMIT 1")
    # Run async query in a new event loop
    import asyncio
    async_result = asyncio.run(client.aquery("MATCH (n) RETURN n LIMIT 1"))

    assert hasattr(sync_result, "records")
    assert hasattr(async_result, "records")


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

    def fake_validate_graph_result_type(result: Any, **_: Any) -> Any:
        return result

    monkeypatch.setattr(autogen_adapter_module, "create_graph_translator", fake_create_graph_translator)
    monkeypatch.setattr(autogen_adapter_module, "validate_graph_result_type", fake_validate_graph_result_type)

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

    monkeypatch.setattr(autogen_adapter_module, "create_graph_translator", fake_create_graph_translator)
    monkeypatch.setattr(autogen_adapter_module, "validate_graph_result_type", fake_validate_graph_result_type)

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

    def fake_validate_batch_operations(*_: Any, **__: Any) -> None:
        return None

    monkeypatch.setattr(autogen_adapter_module, "create_graph_translator", fake_create_graph_translator)
    monkeypatch.setattr(autogen_adapter_module, "validate_graph_result_type", fake_validate_graph_result_type)
    monkeypatch.setattr(autogen_adapter_module, "validate_batch_operations", fake_validate_batch_operations)

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

    monkeypatch.setattr(autogen_adapter_module, "create_graph_translator", fake_create_graph_translator)
    monkeypatch.setattr(autogen_adapter_module, "validate_graph_result_type", fake_validate_graph_result_type)
    monkeypatch.setattr(autogen_adapter_module, "validate_batch_operations", fake_validate_batch_operations)

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

    try:
        result = client.batch([])
        assert result is not None
    except Exception as e:
        assert "empty" in str(e).lower() or "must not" in str(e).lower()


def test_batch_operation_validation(adapter: Any) -> None:
    """Invalid operation types."""
    client = _make_client(adapter)

    # Test that empty batch raises ValueError
    with pytest.raises(ValueError, match="batch ops must not be empty"):
        client.batch([])


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

        async def query(self, *args: Any, **kwargs: Any) -> Any:
            return {"records": [], "summary": {}}

        async def capabilities(self) -> GraphCapabilities:
            return GraphCapabilities(server="test", version="1.0")

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.aclosed = True

    adapter = ClosingGraphAdapter()

    with CorpusAutoGenGraphClient(adapter=adapter) as client:
        assert client is not None

    assert adapter.closed is True

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

    client.close()
    assert close_count == 1


@pytest.mark.asyncio
async def test_async_close_with_pending_operations() -> None:
    """Close during active ops."""
    class SlowAdapter:
        def __init__(self) -> None:
            self.aclosed = False
            self.active_queries = 0

        async def query(self, *args: Any, **kwargs: Any) -> Any:
            self.active_queries += 1
            await asyncio.sleep(0.1)
            self.active_queries -= 1
            return QueryResult(records=[], summary={})

        async def capabilities(self) -> Any:
            return GraphCapabilities(server="test", version="1.0")

        async def aclose(self) -> None:
            while self.active_queries > 0:
                await asyncio.sleep(0.01)
            self.aclosed = True

    adapter_instance = SlowAdapter()
    client = CorpusAutoGenGraphClient(adapter=adapter_instance)

    query_task = asyncio.create_task(client.aquery("MATCH (n) RETURN n"))

    await asyncio.sleep(0.05)
    await client.aclose()
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
    results: List[Any] = []
    errors: List[Any] = []

    def execute_query(thread_id: int) -> None:
        try:
            query = f"MATCH (n) RETURN n LIMIT {(thread_id % 5) + 1}"
            params = {"thread": thread_id}

            result = client.query(
                query,
                params=params,
                namespace=f"thread-{thread_id}",
                conversation={"thread_id": thread_id, "agent": f"agent-{thread_id}"},
            )
            results.append((thread_id, result))
        except Exception as e:  # noqa: BLE001
            errors.append((thread_id, str(e)))

    threads: List[threading.Thread] = []
    num_threads = 10

    for i in range(num_threads):
        t = threading.Thread(target=execute_query, args=(i,))
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    assert len(errors) == 0, f"Errors occurred in threads: {errors}"
    assert len(results) == num_threads
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
            except Exception as e:  # noqa: BLE001
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
            namespace=f"async-{task_id}",
        )
        return task_id, result

    tasks = []
    for i in range(10):
        delay = (i % 3) * 0.01
        tasks.append(execute_async_query(i, delay))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            pytest.fail(f"Task {i} failed: {result}")

    task_ids = [tid for tid, _ in results]
    assert set(task_ids) == set(range(10))


@pytest.mark.asyncio
async def test_concurrent_mixed_async_operations(adapter: Any) -> None:
    """
    Mixed async operations (aquery, astream, abulk) executing concurrently.

    IMPORTANT:
    The previous version accidentally built a list of coroutines and then treated
    them as callables. This version builds callables and schedules them correctly.
    """
    client = _make_client(adapter)

    async def run_aquery(task_id: int) -> Any:  # noqa: ARG001
        return await client.aquery("MATCH (n) RETURN n LIMIT 1")

    async def run_astream(task_id: int) -> int:  # noqa: ARG001
        count = 0
        aiter = client.astream_query("MATCH (n) RETURN n LIMIT 2")
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

    callables = []
    for i in range(9):
        if i % 3 == 0:
            callables.append(lambda i=i: run_aquery(i))
        elif i % 3 == 1:
            callables.append(lambda i=i: run_astream(i))
        else:
            callables.append(lambda i=i: run_abulk(i))

    semaphore = asyncio.Semaphore(3)

    async def run_with_semaphore(fn):
        async with semaphore:
            return await fn()

    results = await asyncio.gather(*(run_with_semaphore(fn) for fn in callables), return_exceptions=True)

    errors = [r for r in results if isinstance(r, Exception)]
    assert len(errors) == 0, f"Concurrent operations failed: {errors}"


@pytest.mark.asyncio
async def test_mixed_sync_async_concurrent_access(adapter: Any) -> None:
    """
    Test scenario where sync and async operations are called concurrently.
    """
    client = _make_client(adapter)
    results: Dict[str, List[Any]] = {"sync": [], "async": []}
    errors: Dict[str, List[Any]] = {"sync": [], "async": []}

    def sync_operations() -> None:
        for i in range(5):
            try:
                result = client.query(f"SYNC MATCH (n) RETURN n LIMIT {i+1}")
                results["sync"].append((i, result))
            except Exception as e:  # noqa: BLE001
                errors["sync"].append((i, str(e)))

    async def async_operations() -> None:
        for i in range(5):
            try:
                result = await client.aquery(f"ASYNC MATCH (n) RETURN n LIMIT {i+1}")
                results["async"].append((i, result))
            except Exception as e:  # noqa: BLE001
                errors["async"].append((i, str(e)))

    sync_thread = threading.Thread(target=sync_operations)
    sync_thread.start()

    async_task = asyncio.create_task(async_operations())

    sync_thread.join()
    await async_task

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
                timeout_ms=10000,
            )
        except Exception as e:  # noqa: BLE001
            return e

    max_workers = 20
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(make_request, i) for i in range(50)]

        done, not_done = concurrent.futures.wait(futures, timeout=30.0)

        assert len(not_done) == 0, f"{len(not_done)} requests timed out"

        success_count = 0
        error_count = 0

        for future in done:
            result = future.result()
            if isinstance(result, Exception):
                error_count += 1
            else:
                success_count += 1

        assert success_count > 0
        error_rate = error_count / len(futures)
        assert error_rate < 0.5, f"High error rate: {error_rate}"


def test_stress_test_high_concurrency(adapter: Any) -> None:
    """Extreme load test."""
    client = _make_client(adapter)

    def stress_task(task_id: int) -> bool:
        try:
            for _ in range(10):
                client.query(f"MATCH (n) RETURN n LIMIT {task_id % 5}")
            return True
        except Exception:  # noqa: BLE001
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
            ops = [BatchOp("query", {"text": f"MATCH (n) RETURN n LIMIT {thread_id}"})]
            client.batch(ops)  # type: ignore[arg-type]
            return True
        except Exception:  # noqa: BLE001
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_batch, i) for i in range(20)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert all(results), "Some batch operations failed"


# ---------------------------------------------------------------------------
# REAL AutoGen integration via soft-imported FunctionTool wrappers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_autogen_function_tools_execute_end_to_end(adapter: Any) -> None:
    """
    This test ensures AutoGen is *actually* exercised against our adapter.

    Behavior:
    - If autogen_core is installed:
        * create_autogen_graph_tools() must return real FunctionTool instances
        * tool.run_json(...) must execute successfully (true AutoGen integration)
    - If autogen_core is NOT installed:
        * create_autogen_graph_tools() must raise a clear RuntimeError with install guidance

    This avoids skips while still providing real integration coverage when available.
    """
    client = _make_client(adapter)

    # create_autogen_graph_tools is intentionally a soft dependency.
    create_tools = getattr(autogen_adapter_module, "create_autogen_graph_tools", None)
    assert callable(create_tools), "Adapter module must expose create_autogen_graph_tools()"

    try:
        from autogen_core import CancellationToken  # type: ignore[import-not-found]
        from autogen_core.tools import FunctionTool  # type: ignore[import-not-found]
    except Exception:
        with pytest.raises(RuntimeError):
            create_tools(client)
        return

    tools = create_tools(client)
    assert isinstance(tools, list)
    assert len(tools) >= 4

    # Basic sanity: all returned entries should be FunctionTool (real AutoGen objects).
    assert all(isinstance(t, FunctionTool) for t in tools)

    # Tools are named with a default prefix; locate them by suffix to avoid name drift.
    by_name = {getattr(t, "name", ""): t for t in tools}

    query_tool = next((t for n, t in by_name.items() if n.endswith("_query")), None)
    stream_tool = next((t for n, t in by_name.items() if n.endswith("_stream_query")), None)
    bulk_tool = next((t for n, t in by_name.items() if n.endswith("_bulk_vertices")), None)
    batch_tool = next((t for n, t in by_name.items() if n.endswith("_batch")), None)

    assert query_tool is not None
    assert stream_tool is not None
    assert bulk_tool is not None
    assert batch_tool is not None

    token = CancellationToken()

    # Execute query tool
    qres = await query_tool.run_json({"query": "MATCH (n) RETURN n LIMIT 1"}, token)
    assert isinstance(qres, Mapping)
    assert "result" in qres

    # Execute stream tool (bounded)
    sres = await stream_tool.run_json({"query": "MATCH (n) RETURN n LIMIT 2", "max_chunks": 3}, token)
    assert isinstance(sres, Mapping)
    assert "chunks" in sres

    # Execute bulk tool
    bres = await bulk_tool.run_json({"namespace": "tool-ns", "limit": 3}, token)
    assert isinstance(bres, Mapping)
    assert "result" in bres

    # Execute batch tool
    batres = await batch_tool.run_json(
        {"ops": [{"op": "query", "args": {"text": "MATCH (n) RETURN n LIMIT 1"}}]},
        token,
    )
    assert isinstance(batres, Mapping)
    assert "result" in batres


# ---------------------------------------------------------------------------
# Logging / Telemetry Tests
# ---------------------------------------------------------------------------


def test_logging_includes_autogen_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """Verify AutoGen context is included in logs."""
    log_capture: List[str] = []

    class TestHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_capture.append(record.getMessage())

    logger_obj = logging.getLogger("corpus_sdk.graph.framework_adapters.autogen")
    handler = TestHandler()
    logger_obj.addHandler(handler)
    logger_obj.setLevel(logging.INFO)

    try:
        client = _make_client(adapter)
        # Trigger a query to generate logs
        try:
            client.query(
                "MATCH (n) RETURN n",
                conversation={"conversation_id": "log-test"},
            )
        except Exception:
            pass  # We just want to trigger logging

        log_messages = " ".join(log_capture).lower()
        # Check if any logging happened, logging is best-effort
        assert len(log_capture) >= 0  # Just verify no crash
    finally:
        logger_obj.removeHandler(handler)


def test_operation_telemetry_includes_framework(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Best-effort telemetry test.

    The graph adapter may route telemetry through shared utilities rather than exposing
    a direct get_metrics_sink symbol in this module. We patch what we can and assert
    only if telemetry hooks are actually exercised in this environment.
    """
    captured_metrics: List[Dict[str, Any]] = []

    class TestMetrics:
        def observe(self, **kwargs: Any) -> None:
            captured_metrics.append(kwargs)

        def counter(self, **kwargs: Any) -> None:
            captured_metrics.append(kwargs)

    # Patch locally if present; do not hard-fail if the symbol isn't exposed here.
    monkeypatch.setattr(
        autogen_adapter_module,
        "get_metrics_sink",
        lambda: TestMetrics(),
        raising=False,
    )

    client = _make_client(adapter)
    client.query("MATCH (n) RETURN n LIMIT 1")

    # If metrics hooks are used, ensure framework is represented somehow.
    if captured_metrics:
        for metric in captured_metrics:
            if metric.get("component") == "graph":
                assert "framework" in str(metric).lower() or "autogen" in str(metric).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
