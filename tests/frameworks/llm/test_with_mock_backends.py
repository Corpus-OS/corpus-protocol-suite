# tests/frameworks/llm/test_with_mock_backends.py

from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Type

import pytest

from tests.frameworks.registries.llm_registry import (
    LLMFrameworkDescriptor,
    iter_llm_framework_descriptors,
)

LLM_OPERATION_PREFIX = "llm_"
FAILURE_MESSAGE_SYNC = "intentional llm backend failure (sync)"
FAILURE_MESSAGE_ASYNC = "intentional llm backend failure (async)"


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


# ---------------------------------------------------------------------------
# "Evil" LLM backends (LLMProtocolV1-ish)
# ---------------------------------------------------------------------------


class InvalidResultLLMBackend:
    """
    Backend that returns blatantly invalid results for LLM operations.

    - complete/acomplete: return a non-text, non-ChatResult scalar
    - stream/astream: return a non-iterable/non-async-iterable
    - count_tokens: return a non-integer

    Framework adapters should surface coercion / validation errors rather than
    silently treating these as valid LLM results.
    """

    # Core completion surfaces expected by the translator
    def complete(self, *args: Any, **kwargs: Any) -> Any:
        return 123456  # clearly not a ChatResult / text-like

    async def acomplete(self, *args: Any, **kwargs: Any) -> Any:
        return 123456

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        # Return something non-iterable
        return 3.14159

    async def astream(self, *args: Any, **kwargs: Any) -> Any:
        # Awaitable resolving to something non-async-iterable
        return 3.14159

    def count_tokens(self, *args: Any, **kwargs: Any) -> Any:
        return "not-an-int"


class EmptyResultLLMBackend:
    """
    Backend that always returns obviously empty / None-like results.

    Used to verify that adapters do not silently treat `None` completion as
    fully valid, particularly for the primary completion surface.

    Note: streaming returning no chunks may be acceptable in some cases, so we
    focus assertions on the main completion path.
    """

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        return None

    async def acomplete(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        # Empty iterator
        return iter(())

    async def astream(self, *args: Any, **kwargs: Any) -> Any:
        async def _aiter():
            if False:  # pragma: no cover - structure only
                yield None

        return _aiter()

    def count_tokens(self, *args: Any, **kwargs: Any) -> Any:
        return 0


class RaisingLLMBackend:
    """
    Backend that always raises.

    Used to validate that error-context decorators still attach context when
    failures originate in the LLM backend rather than higher-level code.
    """

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    async def acomplete(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_ASYNC)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    async def astream(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_ASYNC)

    def count_tokens(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_method(instance: Any, name: str | None) -> Callable[..., Any]:
    """Helper to fetch a method from the instance and assert it is callable."""
    assert name is not None, "Method name must not be None"
    attr = getattr(instance, name, None)
    assert callable(attr), f"{instance!r} missing expected callable method {name!r}"
    return attr


def _make_llm_with_evil_backend(
    framework_descriptor: LLMFrameworkDescriptor,
    backend_cls: Type[Any],
) -> Any:
    """
    Instantiate the framework LLM adapter/client with an 'evil' backend.

    This bypasses the normal LLM adapter fixture wiring and lets us simulate
    misbehaving backends in a controlled way.
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    llm_cls = getattr(module, framework_descriptor.adapter_class)

    backend = backend_cls()

    init_kwargs: dict[str, Any] = {}

    # Let the descriptor drive how the backend is injected; fall back to
    # a conventional 'llm_adapter' kwarg if not provided.
    adapter_param_name = getattr(
        framework_descriptor,
        "adapter_param_name",
        "llm_adapter",
    )
    init_kwargs[adapter_param_name] = backend

    # Many adapters accept a model name; default to something explicit if
    # the constructor requires it but the descriptor doesn't override it.
    if "model" in getattr(llm_cls, "__init__", lambda *a, **k: None).__code__.co_varnames:  # type: ignore[union-attr]
        init_kwargs.setdefault("model", "mock-backend-model")

    instance = llm_cls(**init_kwargs)
    return instance


def _call_invoke(
    descriptor: LLMFrameworkDescriptor,
    instance: Any,
    prompt: str,
) -> Any:
    """
    Call the primary sync invoke/complete surface for the LLM.

    This abstracts over frameworks that use different method names and may
    accept an extra context/config kwarg.
    """
    invoke_fn = _get_method(instance, descriptor.invoke_method)
    if descriptor.context_kwarg:
        return invoke_fn(prompt, **{descriptor.context_kwarg: {}})
    return invoke_fn(prompt)


def _call_stream(
    descriptor: LLMFrameworkDescriptor,
    instance: Any,
    prompt: str,
) -> Any:
    """
    Call the sync streaming surface (if declared).
    """
    assert descriptor.stream_method is not None
    stream_fn = _get_method(instance, descriptor.stream_method)
    if descriptor.context_kwarg:
        return stream_fn(prompt, **{descriptor.context_kwarg: {}})
    return stream_fn(prompt)


# ---------------------------------------------------------------------------
# Invalid result behavior
# ---------------------------------------------------------------------------


def test_invalid_backend_result_causes_errors_for_sync_invoke(
    framework_descriptor: LLMFrameworkDescriptor,
) -> None:
    """
    If the backend returns a clearly invalid result type for the primary
    completion surface, the framework adapter should surface an error rather
    than silently treating it as a valid LLM result.
    """
    if not framework_descriptor.invoke_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare a sync invoke method",
        )

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        InvalidResultLLMBackend,
    )

    with pytest.raises(Exception):  # noqa: BLE001
        _call_invoke(framework_descriptor, instance, "invalid-llm-sync-invoke")


def test_invalid_backend_result_causes_errors_for_sync_stream_when_declared(
    framework_descriptor: LLMFrameworkDescriptor,
) -> None:
    """
    Same as the invoke test, but for the sync streaming surface when declared.
    """
    if not framework_descriptor.stream_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare sync streaming",
        )

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        InvalidResultLLMBackend,
    )

    with pytest.raises(Exception):  # noqa: BLE001
        iterator = _call_stream(
            framework_descriptor,
            instance,
            "invalid-llm-sync-stream",
        )

        # Force iteration to trigger type/shape errors
        for _ in iterator:  # noqa: B007
            pass


@pytest.mark.asyncio
async def test_async_invalid_backend_result_causes_errors_for_invoke_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
) -> None:
    """
    When async is supported, invalid backend results for the async invoke
    surface should also surface as errors, not valid-looking LLM results.
    """
    if not framework_descriptor.async_invoke_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async invoke",
        )

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        InvalidResultLLMBackend,
    )

    ainvoke_fn = _get_method(
        instance,
        framework_descriptor.async_invoke_method,
    )

    with pytest.raises(Exception):  # noqa: BLE001
        if framework_descriptor.context_kwarg:
            coro = ainvoke_fn(
                "invalid-llm-async-invoke",
                **{framework_descriptor.context_kwarg: {}},
            )
        else:
            coro = ainvoke_fn("invalid-llm-async-invoke")

        assert inspect.isawaitable(coro), "Async invoke method must return an awaitable"
        await coro  # noqa: PT018


@pytest.mark.asyncio
async def test_async_invalid_backend_result_causes_errors_for_stream_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
) -> None:
    """
    When async streaming is supported, invalid backend results for
    the async streaming surface should also surface as errors.
    """
    if not framework_descriptor.async_stream_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async streaming",
        )

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        InvalidResultLLMBackend,
    )

    astream_fn = _get_method(
        instance,
        framework_descriptor.async_stream_method,
    )

    with pytest.raises(Exception):  # noqa: BLE001
        if framework_descriptor.context_kwarg:
            aiter = astream_fn(
                "invalid-llm-async-stream",
                **{framework_descriptor.context_kwarg: {}},
            )
        else:
            aiter = astream_fn("invalid-llm-async-stream")

        # Allow awaitable -> async iterator or async iterator directly
        if inspect.isawaitable(aiter):
            aiter = await aiter  # type: ignore[assignment]

        async for _ in aiter:  # noqa: B007
            pass


# ---------------------------------------------------------------------------
# Empty completion behavior
# ---------------------------------------------------------------------------


def test_empty_backend_completion_is_not_silently_treated_as_valid(
    framework_descriptor: LLMFrameworkDescriptor,
) -> None:
    """
    When the backend returns None for the primary completion surface, the
    adapter should not silently treat it as a fully valid completion.

    Acceptable behaviors:
    - Raise an Exception (preferred), or
    - Return a result that is obviously not text-like / ChatResult.

    This test simply asserts that we *don't* get a None completion passed
    straight through to the caller.
    """
    if not framework_descriptor.invoke_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare a sync invoke method",
        )

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        EmptyResultLLMBackend,
    )

    try:
        result = _call_invoke(framework_descriptor, instance, "empty-llm-completion")
    except Exception:  # noqa: BLE001
        # Raising is acceptable and expected for many implementations.
        return

    # If it did not raise, the result must at least not be None; adapters
    # should coerce or wrap it into something text-like.
    assert result is not None, (
        "EmptyResultLLMBackend produced a None completion; adapters should "
        "treat None completions as errors or coerce them into a text-like result."
    )


# ---------------------------------------------------------------------------
# Error-context when backend raises
# ---------------------------------------------------------------------------


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


def test_backend_exception_is_wrapped_with_error_context_on_invoke(
    framework_descriptor: LLMFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the backend raises during a sync invoke/complete operation, the
    framework adapter's error-context decorator should:

    - call attach_context() with the exception and useful metadata, and
    - re-raise the original exception (or a wrapped one).
    """
    if not framework_descriptor.invoke_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare a sync invoke method",
        )

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        RaisingLLMBackend,
    )

    with pytest.raises(RuntimeError, match="backend failure"):
        _call_invoke(framework_descriptor, instance, "err-llm-sync-invoke")

    assert calls, "attach_context was not called for backend invoke failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name
    assert str(ctx["operation"]).startswith(LLM_OPERATION_PREFIX)


@pytest.mark.asyncio
async def test_async_backend_exception_is_wrapped_with_error_context_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async is supported, backend exceptions in async invoke should also go
    through the error-context decorators and call attach_context().
    """
    if not framework_descriptor.async_invoke_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async invoke",
        )

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        RaisingLLMBackend,
    )

    ainvoke_fn = _get_method(
        instance,
        framework_descriptor.async_invoke_method,
    )

    with pytest.raises(RuntimeError, match="backend failure"):
        if framework_descriptor.context_kwarg:
            coro = ainvoke_fn(
                "err-llm-async-invoke",
                **{framework_descriptor.context_kwarg: {}},
            )
        else:
            coro = ainvoke_fn("err-llm-async-invoke")

        assert inspect.isawaitable(coro), "Async invoke method must return an awaitable"
        await coro  # noqa: PT018

    assert calls, "attach_context was not called for async backend failures"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name
    assert str(ctx["operation"]).startswith(LLM_OPERATION_PREFIX)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
