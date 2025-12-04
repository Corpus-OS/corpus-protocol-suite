# tests/frameworks/llm/test_crewai_llm_adapter.py

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Dict, Type

import pytest

import corpus_sdk.llm.framework_adapters.crewai as crewai_llm_module
from corpus_sdk.llm.framework_adapters.crewai import (
    CorpusCrewAILLMClient,
    CrewAILLMFrameworkTranslator,
    ErrorCodes,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYNC_PROMPT = "crewai-sync-prompt"
STREAM_PROMPT = "crewai-stream-prompt"
ASYNC_PROMPT = "crewai-async-prompt"
ASYNC_STREAM_PROMPT = "crewai-async-stream-prompt"

FAILURE_MESSAGE_SYNC = "test error from crewai llm adapter"
FAILURE_MESSAGE_ASYNC = "test async error from crewai llm adapter"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(adapter: Any, **kwargs: Any) -> CorpusCrewAILLMClient:
    """Construct a CorpusCrewAILLMClient instance from the generic adapter."""
    return CorpusCrewAILLMClient(adapter=adapter, **kwargs)


def _patch_create_llm_translator(
    monkeypatch: pytest.MonkeyPatch,
    translator_cls: Type[Any],
) -> None:
    """
    Patch create_llm_translator to always return an instance of translator_cls.

    translator_cls is expected to be a class; instances are created with no args.
    """

    def fake_create_llm_translator(*_: Any, **__: Any) -> Any:
        return translator_cls()

    monkeypatch.setattr(
        crewai_llm_module,
        "create_llm_translator",
        fake_create_llm_translator,
    )


def _patch_validate_llm_result_type_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch validate_llm_result_type to simply return the result unchanged."""

    def fake_validate_llm_result_type(result: Any, **_: Any) -> Any:
        return result

    monkeypatch.setattr(
        crewai_llm_module,
        "validate_llm_result_type",
        fake_validate_llm_result_type,
    )


# ---------------------------------------------------------------------------
# Constructor / adapter-surface behavior
# ---------------------------------------------------------------------------


def test_constructor_rejects_adapter_without_llm_surface() -> None:
    """
    CorpusCrewAILLMClient should enforce that the underlying adapter exposes
    a basic LLMProtocolV1-compatible interface; if not, __init__ should raise.
    """

    class BadAdapter:
        # Deliberately missing LLM methods like complete/acomplete
        def __init__(self) -> None:
            pass

    with pytest.raises(TypeError) as exc_info:
        CorpusCrewAILLMClient(adapter=BadAdapter())

    msg = str(exc_info.value)
    # Keep this message loose enough that implementations can tweak wording.
    assert "LLMProtocolV1" in msg or "LLM-protocol" in msg or "LLM" in msg


def test_constructor_accepts_minimal_llm_adapter() -> None:
    """
    A minimal adapter implementing the core sync complete() surface should
    be accepted and produce a usable client.
    """

    class MinimalLLMAdapter:
        def complete(self, prompt: str, **_: Any) -> Dict[str, Any]:
            return {"completion": f"echo:{prompt}"}

        def capabilities(self) -> Dict[str, Any]:
            return {}

        def health(self) -> Dict[str, Any]:
            return {}

    client = CorpusCrewAILLMClient(adapter=MinimalLLMAdapter())
    assert client is not None

    result = client.complete(SYNC_PROMPT)
    assert isinstance(result, Mapping)


# ---------------------------------------------------------------------------
# Translator behavior
# ---------------------------------------------------------------------------


def test_default_translator_uses_crewai_framework_translator(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    By default, CorpusCrewAILLMClient should:

    - Construct a CrewAILLMFrameworkTranslator instance, and
    - Pass it into create_llm_translator with framework="crewai".
    """
    captured: Dict[str, Any] = {}

    def fake_create_llm_translator(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["args"] = args
        captured["kwargs"] = kwargs

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(
        crewai_llm_module,
        "create_llm_translator",
        fake_create_llm_translator,
    )

    client = _make_client(adapter)

    # Trigger lazy translator construction
    _ = client._translator  # noqa: SLF001

    assert "kwargs" in captured
    kwargs = captured["kwargs"]

    assert kwargs.get("framework") == "crewai"
    translator = kwargs.get("translator")
    assert isinstance(translator, CrewAILLMFrameworkTranslator)


def test_framework_translator_override_is_respected(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If framework_translator is provided, CorpusCrewAILLMClient should pass
    it through to create_llm_translator instead of constructing its own
    CrewAILLMFrameworkTranslator.
    """
    captured: Dict[str, Any] = {}

    class CustomTranslator(CrewAILLMFrameworkTranslator):
        pass

    custom = CustomTranslator()

    def fake_create_llm_translator(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["args"] = args
        captured["kwargs"] = kwargs

        class DummyTranslator:
            pass

        return DummyTranslator()

    monkeypatch.setattr(
        crewai_llm_module,
        "create_llm_translator",
        fake_create_llm_translator,
    )

    client = _make_client(
        adapter,
        framework_translator=custom,
        framework_version="crewai-llm-fw-1.2.3",
    )

    _ = client._translator  # noqa: SLF001

    kwargs = captured["kwargs"]
    assert kwargs.get("framework") == "crewai"
    assert kwargs.get("translator") is custom


# ---------------------------------------------------------------------------
# Context translation / core_ctx_from_crewai mapping
# ---------------------------------------------------------------------------


def test_crewai_task_and_extra_context_passed_to_core_ctx(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Verify that task and extra_context are passed through to core_ctx_from_crewai
    with the configured framework_version.
    """
    captured: Dict[str, Any] = {}

    # Patch OperationContext so our fake ctx passes isinstance() checks.
    class DummyOperationContext:
        def __init__(self, **kwargs: Any) -> None:
            self.attrs = kwargs

    monkeypatch.setattr(
        crewai_llm_module,
        "OperationContext",
        DummyOperationContext,
    )

    def fake_core_ctx_from_crewai(
        task: Any,  # noqa: ARG001
        *,
        framework_version: Any = None,
        **extra: Any,
    ) -> Any:
        captured["framework_version"] = framework_version
        captured["extra"] = extra
        return DummyOperationContext(task=task, **extra)

    monkeypatch.setattr(
        crewai_llm_module,
        "core_ctx_from_crewai",
        fake_core_ctx_from_crewai,
    )

    client = _make_client(adapter, framework_version="crewai-llm-test-fw")

    fake_task = object()
    extra_ctx = {
        "request_id": "req-crewai-llm-xyz",
        "tenant": "tenant-llm-1",
    }

    result = client.complete(
        SYNC_PROMPT,
        task=fake_task,
        extra_context=extra_ctx,
    )
    assert result is not None

    assert captured.get("framework_version") == "crewai-llm-test-fw"
    assert captured.get("extra") == extra_ctx


def test_build_ctx_failure_raises_bad_request_like_error_and_attaches_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    _build_ctx should wrap failures from core_ctx_from_crewai in an error that:

    - Has code ErrorCodes.BAD_OPERATION_CONTEXT, and
    - Includes a helpful message, and
    - Causes attach_context to be called with framework='crewai' and a
      context_translation operation tag.
    """
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    def fake_core_ctx_from_crewai(
        task: Any,  # noqa: ARG001
        *,
        framework_version: Any = None,  # noqa: ARG001
        **extra: Any,  # noqa: ARG001
    ) -> Any:
        raise RuntimeError("boom from llm ctx builder")

    monkeypatch.setattr(
        crewai_llm_module,
        "attach_context",
        fake_attach_context,
    )
    monkeypatch.setattr(
        crewai_llm_module,
        "core_ctx_from_crewai",
        fake_core_ctx_from_crewai,
    )

    client = _make_client(adapter, framework_version="crewai-llm-fw")

    with pytest.raises(Exception) as exc_info:  # noqa: BLE001
        client._build_ctx(  # noqa: SLF001
            task=object(),
            extra_context={"foo": "bar"},
        )

    err = exc_info.value
    assert getattr(err, "code", None) == ErrorCodes.BAD_OPERATION_CONTEXT
    assert "OperationContext" in str(err) or "context" in str(err)

    assert captured_ctx.get("framework") == "crewai"
    assert captured_ctx.get("operation") == "context_translation"


# ---------------------------------------------------------------------------
# Error-context decorator behavior
# ---------------------------------------------------------------------------


def test_error_context_includes_crewai_metadata_sync(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    When an error occurs during a sync LLM operation, error context should
    include CrewAI-specific metadata via attach_context().
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(
        crewai_llm_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingTranslator:
        def complete(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError(FAILURE_MESSAGE_SYNC)

    _patch_create_llm_translator(monkeypatch, FailingTranslator)

    client = _make_client(adapter)

    with pytest.raises(RuntimeError, match=FAILURE_MESSAGE_SYNC):
        client.complete(SYNC_PROMPT)

    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "crewai"
    # LLM operations should be tagged in a consistent way (e.g., "llm_complete")
    assert isinstance(captured_context.get("operation"), str)
    assert "llm" in captured_context["operation"] or "complete" in captured_context[
        "operation"
    ]


@pytest.mark.asyncio
async def test_error_context_includes_crewai_metadata_async(
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
        crewai_llm_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingTranslator:
        async def acomplete(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError(FAILURE_MESSAGE_ASYNC)

    _patch_create_llm_translator(monkeypatch, FailingTranslator)

    client = _make_client(adapter)

    with pytest.raises(RuntimeError, match=FAILURE_MESSAGE_ASYNC):
        await client.acomplete(ASYNC_PROMPT)

    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "crewai"
    assert isinstance(captured_context.get("operation"), str)
    assert "llm" in captured_context["operation"] or "complete" in captured_context[
        "operation"
    ]


# ---------------------------------------------------------------------------
# Sync semantics (basic smoke + type stability)
# ---------------------------------------------------------------------------


def test_sync_complete_type_stable_across_calls(adapter: Any) -> None:
    """
    For simple prompts, the client.complete method should return the same
    *type* across calls. This mirrors the graph type-stability tests.
    """
    client = _make_client(adapter)

    result1 = client.complete(SYNC_PROMPT)
    result2 = client.complete(SYNC_PROMPT + " again")

    assert result1 is not None
    assert result2 is not None
    assert type(result1) is type(result2)


def test_sync_complete_and_stream_basic(adapter: Any) -> None:
    """
    Basic smoke test for sync complete / stream behavior: methods should
    accept text input and not crash, returning protocol-level shapes.
    """
    client = _make_client(adapter)

    # Non-streaming completion
    result = client.complete(SYNC_PROMPT)
    assert result is not None

    # Streaming completion
    if hasattr(client, "stream"):
        iterator = client.stream(STREAM_PROMPT)
        # It's fine if the stream is empty; just assert it is iterable.
        seen_any = False
        for _ in iterator:  # noqa: B007
            seen_any = True
            break
        assert isinstance(seen_any, bool)


def test_sync_complete_accepts_optional_params_and_context(adapter: Any) -> None:
    """
    complete() should accept params, temperature, namespace, timeout_ms, and
    task/extra_context kwargs without raising.
    """
    client = _make_client(adapter)

    result = client.complete(
        SYNC_PROMPT,
        params={"foo": "bar"},
        temperature=0.3,
        namespace="crewai-llm-ns",
        timeout_ms=2000,
        task=object(),
        extra_context={"request_id": "req-llm-sync"},
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Async semantics (basic)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_complete_and_stream_basic(adapter: Any) -> None:
    """
    Async acomplete / astream should exist and produce results compatible
    with the sync API (non-None result / async-iterable of chunks).
    """
    client = _make_client(adapter)

    assert hasattr(client, "acomplete")
    assert hasattr(client, "astream")

    coro = client.acomplete(ASYNC_PROMPT)
    assert inspect.isawaitable(coro)
    result = await coro
    assert result is not None

    aiter = client.astream(ASYNC_STREAM_PROMPT)
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

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
    client = _make_client(adapter)

    result = await client.acomplete(
        ASYNC_PROMPT,
        params={"foo": "baz"},
        temperature=0.1,
        namespace="crewai-llm-async-ns",
        timeout_ms=3000,
        task=object(),
        extra_context={"request_id": "req-llm-async"},
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Streaming: invalid chunks exercise validate_llm_result_type
# ---------------------------------------------------------------------------


def test_stream_invalid_chunk_triggers_validation(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If the translator yields invalid chunks, stream() should pass them through
    validate_llm_result_type, and failures there should surface to the caller.
    """
    captured: Dict[str, Any] = {}

    class BadChunkTranslator:
        def stream(
            self,
            raw_request: Mapping[str, Any],  # noqa: ARG002
            *,
            op_ctx: Any = None,  # noqa: ARG002
            framework_ctx: Mapping[str, Any] | None = None,  # noqa: ARG002
        ):
            # Yield a blatantly invalid chunk
            yield "not-a-llm-chunk"

    _patch_create_llm_translator(monkeypatch, BadChunkTranslator)

    def fake_validate_llm_result_type(result: Any, **kwargs: Any) -> Any:
        captured["result"] = result
        captured["kwargs"] = kwargs
        raise RuntimeError("forced validation failure for llm chunk")

    monkeypatch.setattr(
        crewai_llm_module,
        "validate_llm_result_type",
        fake_validate_llm_result_type,
    )

    client = _make_client(adapter)

    if not hasattr(client, "stream"):
        pytest.skip("Client does not expose stream() surface")

    iterator = client.stream(STREAM_PROMPT)

    with pytest.raises(RuntimeError, match="forced validation failure for llm chunk"):
        next(iterator)

    assert captured.get("result") == "not-a-llm-chunk"
    assert "expected_type" in captured.get("kwargs", {})


@pytest.mark.asyncio
async def test_astream_invalid_chunk_triggers_validation_async(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Async streaming path should also exercise validate_llm_result_type when
    chunks are invalid.
    """
    captured: Dict[str, Any] = {}

    class BadChunkTranslator:
        async def astream(
            self,
            raw_request: Mapping[str, Any],  # noqa: ARG002
            *,
            op_ctx: Any = None,  # noqa: ARG002
            framework_ctx: Mapping[str, Any] | None = None,  # noqa: ARG002
        ):
            # Simple async generator
            async def gen() -> Any:
                yield "not-a-llm-chunk-async"

            return gen()

    _patch_create_llm_translator(monkeypatch, BadChunkTranslator)

    def fake_validate_llm_result_type(result: Any, **kwargs: Any) -> Any:
        captured["result"] = result
        captured["kwargs"] = kwargs
        raise RuntimeError("forced validation failure for async llm chunk")

    monkeypatch.setattr(
        crewai_llm_module,
        "validate_llm_result_type",
        fake_validate_llm_result_type,
    )

    client = _make_client(adapter)

    aiter = await client.astream(ASYNC_STREAM_PROMPT)
    with pytest.raises(RuntimeError, match="forced validation failure for async llm chunk"):
        async for _ in aiter:  # noqa: B007
            break

    assert captured.get("result") == "not-a-llm-chunk-async"
    assert "expected_type" in captured.get("kwargs", {})


# ---------------------------------------------------------------------------
# Capabilities / health passthrough (basic)
# ---------------------------------------------------------------------------


def test_capabilities_and_health_basic(adapter: Any) -> None:
    """
    Capabilities and health should be surfaced as mappings.

    The detailed structure is tested in framework-agnostic LLM contract
    tests; here we only assert that the CrewAI adapter normalizes to
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
    the underlying adapter when those methods exist.
    """

    class ClosingLLMAdapter:
        def __init__(self) -> None:
            self.closed = False
            self.aclosed = False

        def complete(self, prompt: str, **_: Any) -> Dict[str, Any]:
            return {"completion": prompt}

        def capabilities(self) -> Dict[str, Any]:
            return {}

        def health(self) -> Dict[str, Any]:
            return {}

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.aclosed = True

    adapter = ClosingLLMAdapter()

    # Sync context manager
    with CorpusCrewAILLMClient(adapter=adapter) as client:
        assert client is not None

    assert adapter.closed is True

    # Async context manager
    adapter2 = ClosingLLMAdapter()
    client2 = CorpusCrewAILLMClient(adapter=adapter2)

    async with client2:
        assert client2 is not None

    assert adapter2.aclosed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
