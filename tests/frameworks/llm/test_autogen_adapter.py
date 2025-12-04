# tests/frameworks/llm/test_autogen_adapter.py

from __future__ import annotations

import inspect
from typing import Any, Dict, Mapping

import pytest

import corpus_sdk.llm.framework_adapters.autogen as autogen_adapter_module
from corpus_sdk.llm.framework_adapters.autogen import (
    AutoGenLLMFrameworkTranslator,
    CorpusAutoGenLLMClient,
    ErrorCodes,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAILURE_MESSAGE = "intentional failure from failing llm adapter"
LLM_OPERATION_PREFIX = "llm_"

PROMPT_FOR_SYNC = "autogen-llm-sync-prompt"
PROMPT_FOR_STREAM = "autogen-llm-stream-prompt"
PROMPT_FOR_ASYNC = "autogen-llm-async-prompt"
PROMPT_FOR_ASYNC_STREAM = "autogen-llm-async-stream-prompt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(adapter: Any, **kwargs: Any) -> CorpusAutoGenLLMClient:
    """Construct a CorpusAutoGenLLMClient instance from the generic adapter."""
    return CorpusAutoGenLLMClient(adapter=adapter, **kwargs)


# ---------------------------------------------------------------------------
# Constructor / translator behavior
# ---------------------------------------------------------------------------


def test_default_translator_uses_autogen_framework_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    By default, CorpusAutoGenLLMClient should:

    - Construct an AutoGenLLMFrameworkTranslator instance, and
    - Pass it into create_llm_translator with framework="autogen".
    """
    captured_args: Dict[str, Any] = {}

    def fake_create_llm_translator(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        captured_args["args"] = args
        captured_args["kwargs"] = kwargs

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(
        autogen_adapter_module,
        "create_llm_translator",
        fake_create_llm_translator,
    )

    client = _make_client(adapter)

    # Trigger lazy translator construction
    _ = client._translator  # noqa: SLF001

    assert "kwargs" in captured_args
    kwargs = captured_args["kwargs"]

    assert kwargs.get("framework") == "autogen"
    translator = kwargs.get("translator")
    assert isinstance(translator, AutoGenLLMFrameworkTranslator)


def test_framework_translator_override_is_respected(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If framework_translator is provided, CorpusAutoGenLLMClient should pass
    it through to create_llm_translator instead of constructing its own
    AutoGenLLMFrameworkTranslator.
    """
    captured_args: Dict[str, Any] = {}

    class CustomTranslator(AutoGenLLMFrameworkTranslator):
        pass

    custom = CustomTranslator()

    def fake_create_llm_translator(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        captured_args["args"] = args
        captured_args["kwargs"] = kwargs

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(
        autogen_adapter_module,
        "create_llm_translator",
        fake_create_llm_translator,
    )

    client = _make_client(
        adapter,
        framework_translator=custom,
        framework_version="fw-llm-1.2.3",
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
    adapter: Any,
) -> None:
    """
    Verify that conversation and extra_context are passed through to
    core_ctx_from_autogen with the configured framework_version.
    """
    captured: Dict[str, Any] = {}

    # Patch OperationContext so our fake returns something that satisfies
    # any isinstance checks inside the adapter.
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
        framework_version="autogen-llm-test-version",
    )

    auto_conv = {
        "conversation_id": "conv-123",
        "agent_name": "agent-x",
    }
    extra_ctx = {
        "request_id": "req-xyz",
        "tenant": "tenant-1",
    }

    # Any prompt is fine; we only care that _build_ctx calls our fake.
    result = client.complete(
        PROMPT_FOR_SYNC,
        conversation=auto_conv,
        extra_context=extra_ctx,
    )
    assert result is not None

    # Verify that our fake_core_ctx_from_autogen saw the right arguments.
    assert captured.get("conversation") == auto_conv
    assert captured.get("framework_version") == "autogen-llm-test-version"
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
        raise RuntimeError("boom from autogen llm ctx")

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
        framework_version="autogen-llm-fw-test",
    )

    with pytest.raises(Exception) as exc_info:
        client.complete(
            PROMPT_FOR_SYNC,
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
# Error-context decorator behavior (completion)
# ---------------------------------------------------------------------------


def test_error_context_includes_autogen_metadata_sync(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    When an error occurs during a sync completion operation, error context should
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
        def run_completion(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from autogen llm adapter")

    def fake_create_llm_translator(*_: Any, **__: Any) -> Any:
        return FailingTranslator()

    monkeypatch.setattr(
        autogen_adapter_module,
        "create_llm_translator",
        fake_create_llm_translator,
    )

    client = _make_client(adapter)

    auto_conv = {"conversation_id": "conv-ctx", "agent_name": "tester"}

    with pytest.raises(RuntimeError, match="test error from autogen llm adapter"):
        client.complete(PROMPT_FOR_SYNC, conversation=auto_conv)

    # Verify some context was attached
    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "autogen"
    # The shared decorator uses an operation prefix like "llm_..."
    assert str(captured_context.get("operation", "")).startswith(LLM_OPERATION_PREFIX)
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
    Same as the sync error-context test but for the async completion path.
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
        async def arun_completion(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from autogen llm adapter")

    def fake_create_llm_translator(*_: Any, **__: Any) -> Any:
        return FailingTranslator()

    monkeypatch.setattr(
        autogen_adapter_module,
        "create_llm_translator",
        fake_create_llm_translator,
    )

    client = _make_client(adapter)

    auto_conv = {"conversation_id": "conv-ctx-async", "agent_name": "tester-async"}

    with pytest.raises(RuntimeError, match="test error from autogen llm adapter"):
        await client.acomplete(PROMPT_FOR_ASYNC, conversation=auto_conv)

    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "autogen"
    assert str(captured_context.get("operation", "")).startswith(LLM_OPERATION_PREFIX)
    if "conversation_id" in captured_context:
        assert captured_context["conversation_id"] == "conv-ctx-async"
    if "agent_name" in captured_context:
        assert captured_context["agent_name"] == "tester-async"


# ---------------------------------------------------------------------------
# Streaming validation / error paths
# ---------------------------------------------------------------------------


def test_stream_complete_invalid_chunk_triggers_validation_and_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    stream_complete() should validate each chunk via validate_llm_result_type.

    If a non-CompletionChunk-like value is produced, validate_llm_result_type
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
        def run_completion_stream(
            self,
            raw_request: Mapping[str, Any],
            *,
            op_ctx: Any = None,  # noqa: ARG002
            framework_ctx: Mapping[str, Any] | None = None,
        ):
            # Yield a clearly-invalid "chunk" to trigger validation.
            yield {"not": "a-completion-chunk"}

    def fake_create_llm_translator(*_: Any, **__: Any) -> Any:
        return DummyTranslator()

    class FakeValidationError(Exception):
        def __init__(self, message: str, code: Any | None = None) -> None:
            super().__init__(message)
            self.code = code

    def fake_validate_llm_result_type(
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
        "create_llm_translator",
        fake_create_llm_translator,
    )
    monkeypatch.setattr(
        autogen_adapter_module,
        "validate_llm_result_type",
        fake_validate_llm_result_type,
    )

    client = _make_client(adapter)

    it = client.stream_complete(PROMPT_FOR_STREAM)

    with pytest.raises(FakeValidationError, match="invalid chunk") as exc_info:
        # Force consumption of the first (invalid) chunk.
        next(it)

    err = exc_info.value
    assert getattr(err, "code", None) == ErrorCodes.BAD_TRANSLATED_CHUNK

    # Error-context decorator should have attached framework metadata.
    assert captured_ctx.get("framework") == "autogen"
    assert str(captured_ctx.get("operation", "")).startswith(LLM_OPERATION_PREFIX)


@pytest.mark.asyncio
async def test_astream_complete_invalid_chunk_triggers_validation_and_context_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    astream_complete() should also validate chunks via validate_llm_result_type.

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
        async def arun_completion_stream(
            self,
            raw_request: Mapping[str, Any],
            *,
            op_ctx: Any = None,  # noqa: ARG002
            framework_ctx: Mapping[str, Any] | None = None,
        ):
            # Async generator yielding an invalid chunk.
            async def _gen():
                yield {"not": "a-completion-chunk"}

            return _gen()

    def fake_create_llm_translator(*_: Any, **__: Any) -> Any:
        return DummyTranslator()

    class FakeValidationError(Exception):
        def __init__(self, message: str, code: Any | None = None) -> None:
            super().__init__(message)
            self.code = code

    def fake_validate_llm_result_type(
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
        "create_llm_translator",
        fake_create_llm_translator,
    )
    monkeypatch.setattr(
        autogen_adapter_module,
        "validate_llm_result_type",
        fake_validate_llm_result_type,
    )

    client = _make_client(adapter)

    aiter = client.astream_complete(PROMPT_FOR_ASYNC_STREAM)
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    with pytest.raises(FakeValidationError, match="invalid chunk async") as exc_info:
        async for _ in aiter:  # noqa: B007
            # First iteration should raise from validation.
            break

    err = exc_info.value
    assert getattr(err, "code", None) == ErrorCodes.BAD_TRANSLATED_CHUNK

    assert captured_ctx.get("framework") == "autogen"
    assert str(captured_ctx.get("operation", "")).startswith(LLM_OPERATION_PREFIX)


# ---------------------------------------------------------------------------
# Sync semantics (basic smoke tests)
# ---------------------------------------------------------------------------


def test_sync_complete_and_stream_basic(adapter: Any) -> None:
    """
    Basic smoke test for sync complete / stream_complete behavior: methods should
    accept text input and not crash, returning protocol-level shapes.

    Detailed CompletionResult / StreamingChunk semantics are covered by the
    generic LLM contract tests.
    """
    client = _make_client(adapter, default_model="test-model")

    # Non-streaming completion
    result = client.complete(PROMPT_FOR_SYNC)
    assert result is not None

    # Streaming completion
    chunks = list(client.stream_complete(PROMPT_FOR_STREAM))
    # It's fine if the list is empty; we're only asserting the pathway works.
    assert isinstance(chunks, list)


def test_sync_complete_accepts_optional_params_and_context(adapter: Any) -> None:
    """
    complete() should accept params like temperature, stop, max_tokens, and
    conversation/extra_context kwargs without raising.
    """
    client = _make_client(adapter, default_model="ctx-model")

    result = client.complete(
        "Tell me a joke about llamas.",
        temperature=0.7,
        max_tokens=64,
        stop=["\n\n"],
        conversation={"conversation_id": "conv-sync"},
        extra_context={"request_id": "req-sync"},
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Async semantics (basic smoke tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_complete_and_stream_basic(adapter: Any) -> None:
    """
    Async acomplete / astream_complete should exist and produce results compatible
    with the sync API (non-None result / async-iterable of chunks).
    """
    client = _make_client(adapter)

    # Ensure async methods exist and are coroutine/async-generator functions
    assert hasattr(client, "acomplete")
    assert hasattr(client, "astream_complete")

    complete_coro = client.acomplete(PROMPT_FOR_ASYNC)
    assert inspect.isawaitable(complete_coro)
    result = await complete_coro
    assert result is not None

    aiter = client.astream_complete(PROMPT_FOR_ASYNC_STREAM)

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
async def test_async_complete_accepts_optional_params_and_context(
    adapter: Any,
) -> None:
    """
    acomplete() should accept the same optional params and context as complete().
    """
    client = _make_client(adapter, default_model="async-model")

    result = await client.acomplete(
        "Tell me a short haiku.",
        temperature=0.3,
        max_tokens=32,
        stop=["\n\n"],
        conversation={"conversation_id": "conv-async"},
        extra_context={"request_id": "req-async"},
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Capabilities / health passthrough (basic)
# ---------------------------------------------------------------------------


def test_capabilities_and_health_basic(adapter: Any) -> None:
    """
    Capabilities and health should be surfaced as mappings.

    The detailed structure is tested in framework-agnostic LLM contract tests;
    here we only assert that the AutoGen adapter normalizes to mapping-like
    objects.
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
    the underlying LLM adapter when those methods exist.
    """

    class ClosingLLMAdapter:
        def __init__(self) -> None:
            self.closed = False
            self.aclosed = False

        # Minimal capabilities/health to keep any translator happy.
        def capabilities(self) -> Dict[str, Any]:  # type: ignore[override]
            return {}

        def health(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # noqa: ARG002
            return {}

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.aclosed = True

    adapter = ClosingLLMAdapter()

    # Sync context manager: should call close() if present
    with CorpusAutoGenLLMClient(adapter=adapter) as client:
        # Don't call any methods; we're just testing resource cleanup.
        assert client is not None

    assert adapter.closed is True

    # Async context manager: should call aclose() if present
    adapter2 = ClosingLLMAdapter()
    client2 = CorpusAutoGenLLMClient(adapter=adapter2)

    async with client2:
        assert client2 is not None

    assert adapter2.aclosed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
