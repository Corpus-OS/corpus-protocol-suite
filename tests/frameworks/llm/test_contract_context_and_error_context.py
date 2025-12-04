# tests/frameworks/llm/test_contract_context_and_error_context.py

from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable

import pytest

from tests.frameworks.registries.llm_registry import (
    LLMFrameworkDescriptor,
    iter_llm_framework_descriptors,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAILURE_MESSAGE = "intentional failure from failing llm adapter"

RICH_CONTEXT = {
    "request_id": "req-llm-123",
    "user_id": "user-llm-abc",
    "tags": ["test", "llm"],
    "nested": {"key": "value", "depth": 2},
}

PROMPT_TEXT = "ctx-prompt"
STREAM_PROMPT_TEXT = "ctx-stream-prompt"
ASYNC_PROMPT_TEXT = "ctx-async-prompt"
ASYNC_STREAM_PROMPT_TEXT = "ctx-async-stream-prompt"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=list(iter_llm_framework_descriptors()),
    name="framework_descriptor",
)
def framework_descriptor_fixture(
    request: pytest.FixtureRequest,
) -> LLMFrameworkDescriptor:
    """
    Parameterized over all registered LLM framework descriptors.

    Frameworks that are not actually available in the environment (e.g. the
    underlying LangChain / LlamaIndex / Semantic Kernel libraries are missing)
    are skipped via descriptor.is_available().
    """
    descriptor: LLMFrameworkDescriptor = request.param
    if not descriptor.is_available():
        pytest.skip(f"Framework '{descriptor.name}' not available in this environment")
    return descriptor


@pytest.fixture
def llm_client_instance(
    framework_descriptor: LLMFrameworkDescriptor,
    adapter: Any,
) -> Any:
    """
    Construct a concrete LLM client instance for the given descriptor.

    Mirrors the construction pattern used in the other contract tests: each
    framework adapter is expected to take a `llm_adapter` kwarg that wraps a
    Corpus LLMProtocolV1 implementation.
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    client_cls = getattr(module, framework_descriptor.adapter_class)

    init_kwargs: dict[str, Any] = {"llm_adapter": adapter}

    # Additional framework-specific kwargs (e.g., model name) can be added
    # here if needed, but all current adapters provide sensible defaults.
    instance = client_cls(**init_kwargs)
    return instance


@pytest.fixture
def failing_llm_adapter() -> Any:
    """
    A minimal LLM adapter whose core methods always fail.

    Used only for error-context tests to ensure the decorators invoke
    attach_context() and propagate the exception.

    It implements the minimum LLMProtocolV1 surface expected by the adapters.
    """

    class FailingLLMAdapter:
        def complete(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        def stream(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        def count_tokens(self, *args: Any, **kwargs: Any) -> int:
            raise RuntimeError(FAILURE_MESSAGE)

        def health(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError(FAILURE_MESSAGE)

        def capabilities(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError(FAILURE_MESSAGE)

    return FailingLLMAdapter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_method(instance: Any, name: str | None) -> Callable[..., Any]:
    """
    Helper to fetch a method from the instance and assert it is callable.

    If name is None, this fails fast with a clear assertion message.
    """
    assert name, "Expected a non-empty method name"
    attr = getattr(instance, name, None)
    assert callable(attr), f"{instance!r} missing expected callable method {name!r}"
    return attr


def _maybe_call_with_context(
    descriptor: LLMFrameworkDescriptor,
    fn: Callable[..., Any],
    prompt: str,
    context: Any,
    *,
    extra_kwargs: dict[str, Any] | None = None,
) -> Any:
    """
    Call an LLM function, respecting descriptor.context_kwarg if present.

    This helper allows injecting either a valid Mapping context or an
    intentionally invalid context for robustness tests, while also passing
    additional kwargs (e.g. streaming flags).
    """
    kwargs: dict[str, Any] = dict(extra_kwargs or {})
    if descriptor.context_kwarg:
        kwargs[descriptor.context_kwarg] = context
    return fn(prompt, **kwargs)


def _build_error_wrapped_client_instance(
    framework_descriptor: LLMFrameworkDescriptor,
    failing_llm_adapter: Any,
) -> Any:
    """
    Construct an LLM client instance wired to a failing LLM adapter.

    Used only for error-context tests (we expect calls to raise).
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    client_cls = getattr(module, framework_descriptor.adapter_class)

    init_kwargs: dict[str, Any] = {"llm_adapter": failing_llm_adapter}
    return client_cls(**init_kwargs)


def _patch_attach_context(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
) -> list[tuple[BaseException, dict[str, Any]]]:
    """
    Patch the module-local attach_context used by decorators and capture calls.
    """
    calls: list[tuple[BaseException, dict[str, Any]]] = []

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        calls.append((exc, ctx))

    monkeypatch.setattr(module, "attach_context", fake_attach_context)
    return calls


# ---------------------------------------------------------------------------
# Context contract tests
# ---------------------------------------------------------------------------


def test_rich_mapping_context_is_accepted_and_does_not_break_completions(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    If a framework declares a context_kwarg, it should:

    - accept a rich Mapping (with extra / nested keys),
    - not raise TypeError / ValueError, and
    - still return a valid completion result.

    Frameworks without a declared context_kwarg are skipped here.
    """
    if not framework_descriptor.context_kwarg:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare a context_kwarg",
        )

    if not framework_descriptor.completion_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare a sync completion_method",
        )

    completion_method = _get_method(
        llm_client_instance,
        framework_descriptor.completion_method,
    )

    rich_context = {
        **RICH_CONTEXT,
        "tags": [*RICH_CONTEXT["tags"], framework_descriptor.name],
    }

    result = _maybe_call_with_context(
        framework_descriptor,
        completion_method,
        PROMPT_TEXT,
        context=rich_context,
    )
    # We don't assert detailed shape (framework-specific), just that it didn't crash.
    assert result is not None


def test_invalid_context_type_is_tolerated_and_does_not_crash(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    Passing an invalid context type (non-Mapping) should not crash the adapter.

    The framework adapters are expected to either:
    - log a warning and ignore the context, or
    - gracefully treat it as "no context".

    In all cases, completions should still return results.
    """
    if not framework_descriptor.context_kwarg:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare a context_kwarg",
        )

    if not framework_descriptor.completion_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare a sync completion_method",
        )

    completion_method = _get_method(
        llm_client_instance,
        framework_descriptor.completion_method,
    )

    invalid_contexts = ["not-a-mapping", 12345]

    for invalid_ctx in invalid_contexts:
        result = _maybe_call_with_context(
            framework_descriptor,
            completion_method,
            PROMPT_TEXT,
            context=invalid_ctx,
        )
        assert result is not None


def test_context_is_optional_and_omitting_it_still_works(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    Even when a framework supports a context kwarg, it must still work
    when no context is provided.

    For frameworks without a context_kwarg, this simply exercises the
    basic completion surface.
    """
    if not framework_descriptor.completion_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare a sync completion_method",
        )

    completion_method = _get_method(
        llm_client_instance,
        framework_descriptor.completion_method,
    )

    # No context kwarg passed at all.
    result = completion_method(PROMPT_TEXT)
    assert result is not None


# ---------------------------------------------------------------------------
# Error-context decorator contract tests
# ---------------------------------------------------------------------------


def test_error_context_is_attached_on_sync_completion_failure(
    framework_descriptor: LLMFrameworkDescriptor,
    failing_llm_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the underlying LLM adapter raises during a sync completion operation,
    the framework adapter's error-context wrapper should:

    - call attach_context() with the exception and useful metadata, and
    - re-raise the original exception (or a wrapped one).
    """
    if not framework_descriptor.completion_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare a sync completion_method",
        )

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _build_error_wrapped_client_instance(
        framework_descriptor,
        failing_llm_adapter,
    )

    completion_method = _get_method(
        instance,
        framework_descriptor.completion_method,
    )

    with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
        if framework_descriptor.context_kwarg:
            completion_method(
                "err-sync-complete",
                **{framework_descriptor.context_kwarg: {}},
            )
        else:
            completion_method("err-sync-complete")

    assert calls, "attach_context was not called on sync completion failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    # For current adapters, framework matches descriptor.name (e.g. "langchain").
    assert ctx["framework"] == framework_descriptor.name


def test_error_context_is_attached_on_sync_stream_failure_when_method_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    failing_llm_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When streaming is supported via an explicit streaming_method, sync stream
    failures should go through the error-context decorator and call
    attach_context().
    """
    if not framework_descriptor.streaming_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare a sync streaming_method",
        )

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _build_error_wrapped_client_instance(
        framework_descriptor,
        failing_llm_adapter,
    )

    stream_method = _get_method(instance, framework_descriptor.streaming_method)

    with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
        if framework_descriptor.context_kwarg:
            iterator = stream_method(
                STREAM_PROMPT_TEXT,
                **{framework_descriptor.context_kwarg: {}},
            )
        else:
            iterator = stream_method(STREAM_PROMPT_TEXT)

        # Consume to ensure the failure actually surfaces even if lazy.
        for _ in iterator:  # noqa: B007
            pass

    assert calls, "attach_context was not called on sync stream failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name


def test_error_context_is_attached_on_sync_stream_failure_when_kwarg_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    failing_llm_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When streaming is enabled via a streaming_kwarg on the completion method
    (e.g. stream=True), failures should also trigger error-context attachment.
    """
    if not framework_descriptor.streaming_kwarg:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare a streaming_kwarg",
        )

    if not framework_descriptor.completion_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare a sync completion_method",
        )

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _build_error_wrapped_client_instance(
        framework_descriptor,
        failing_llm_adapter,
    )

    completion_method = _get_method(
        instance,
        framework_descriptor.completion_method,
    )

    with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
        kwargs: dict[str, Any] = {framework_descriptor.streaming_kwarg: True}
        if framework_descriptor.context_kwarg:
            kwargs[framework_descriptor.context_kwarg] = {}

        iterator = completion_method(STREAM_PROMPT_TEXT, **kwargs)

        # Treat result as an iterator/iterable; if the framework chooses to
        # eagerly raise on call, the with-block still catches it.
        if hasattr(iterator, "__iter__"):
            for _ in iterator:  # noqa: B007
                pass

    assert calls, "attach_context was not called on streaming-kwarg failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name


@pytest.mark.asyncio
async def test_error_context_is_attached_on_async_completion_failure_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
    failing_llm_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async completion is supported, async failures should go through the
    error-context decorator and call attach_context().
    """
    if not framework_descriptor.async_completion_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async completion",
        )

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _build_error_wrapped_client_instance(
        framework_descriptor,
        failing_llm_adapter,
    )

    acompletion_method = _get_method(
        instance,
        framework_descriptor.async_completion_method,
    )

    with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
        if framework_descriptor.context_kwarg:
            coro = acompletion_method(
                ASYNC_PROMPT_TEXT,
                **{framework_descriptor.context_kwarg: {}},
            )
        else:
            coro = acompletion_method(ASYNC_PROMPT_TEXT)

        assert inspect.isawaitable(coro), "Async completion must return an awaitable"
        await coro  # noqa: PT018

    assert calls, "attach_context was not called on async completion failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name


@pytest.mark.asyncio
async def test_error_context_is_attached_on_async_stream_failure_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
    failing_llm_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async streaming is supported via async_streaming_method, failures
    should go through the error-context decorator and call attach_context().
    """
    if not framework_descriptor.async_streaming_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async streaming",
        )

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _build_error_wrapped_client_instance(
        framework_descriptor,
        failing_llm_adapter,
    )

    astream_method = _get_method(
        instance,
        framework_descriptor.async_streaming_method,
    )

    with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
        if framework_descriptor.context_kwarg:
            aiter = astream_method(
                ASYNC_STREAM_PROMPT_TEXT,
                **{framework_descriptor.context_kwarg: {}},
            )
        else:
            aiter = astream_method(ASYNC_STREAM_PROMPT_TEXT)

        # aiter may be an async iterator or an awaitable that resolves to one.
        if inspect.isawaitable(aiter):
            aiter = await aiter  # type: ignore[assignment]

        async for _ in aiter:  # noqa: B007
            pass

    assert calls, "attach_context was not called on async stream failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
