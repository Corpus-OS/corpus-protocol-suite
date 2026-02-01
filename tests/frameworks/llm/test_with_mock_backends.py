# tests/frameworks/llm/test_with_mock_backends.py

from __future__ import annotations

import importlib
import inspect
from collections.abc import AsyncIterable, Iterable, Mapping
from typing import Any, Callable, Optional, Type

import pytest

from tests.frameworks.registries.llm_registry import (
    LLMFrameworkDescriptor,
    iter_llm_framework_descriptors,
)

LLM_OPERATION_PREFIX = "llm_"
FAILURE_MESSAGE_SYNC = "intentional llm backend failure (sync)"
FAILURE_MESSAGE_ASYNC = "intentional llm backend failure (async)"

# Rich mapping context used across all calls in this file.
#
# Why this exists:
# - Other conformance suites enforce that adapters tolerate "rich mapping context"
#   being passed either as a nested context object (e.g. config={...}) or as regular kwargs.
# - This file must not silently "pass" while only testing the narrow empty-context path;
#   therefore we prefer exercising a realistic, structured payload.
#
# NOTE: Adapters are expected to accept and ignore unknown context keys.
RICH_CONTEXT: dict[str, Any] = {
    "request_id": "req-123",
    "user_id": "user-abc",
    "tags": ["test"],
    "nested": {"depth": 2, "key": "value"},
}

# Performance guardrails:
# - Streaming tests intentionally consume only a small number of chunks to avoid
#   hanging if an adapter returns an unbounded iterator/async iterator.
MAX_STREAM_CHUNKS_TO_CONSUME = 10


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

    IMPORTANT POLICY (no pytest.skip):
    - We do not skip unavailable frameworks.
    - Tests must pass by asserting correct "unavailable" signaling when a framework
      is not installed, and must fully run when it is available.

    Rationale:
    - Skipping can hide registry/environment drift and yields false confidence.
    - This policy aligns with the embedding conformance suite while preserving
      LLM-specific nuances (optional streaming/token counting surfaces).
    """
    descriptor: LLMFrameworkDescriptor = request.param
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


class IterationRaisingLLMBackend:
    """
    Backend whose streaming surfaces raise *during iteration* rather than at call-time.

    This models real-world streaming failure modes where:
    - a stream starts successfully (yields at least one chunk/token),
    - then fails mid-stream during consumption.

    These failures must still flow through the framework adapter's error-context
    decorators and attach context.
    """

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        # Not used in iteration-time streaming tests; keep deterministic.
        return "ok"

    async def acomplete(self, *args: Any, **kwargs: Any) -> Any:
        return "ok"

    def stream(self, *args: Any, **kwargs: Any) -> Iterable[Any]:
        # Yield one chunk/token to prove the stream started, then fail deterministically.
        yield "chunk-1"
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    async def astream(self, *args: Any, **kwargs: Any) -> AsyncIterable[Any]:
        # Yield one chunk/token to prove the stream started, then fail deterministically.
        yield "chunk-1"
        raise RuntimeError(FAILURE_MESSAGE_ASYNC)

    def count_tokens(self, *args: Any, **kwargs: Any) -> Any:
        return 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_unavailable_contract(descriptor: LLMFrameworkDescriptor) -> None:
    """
    Validate that an unavailable framework descriptor is behaving as expected.

    The suite policy is "no skip": when unavailable, tests pass by asserting
    correct unavailability signaling (import failure and/or falsey availability flags).
    """
    assert descriptor.is_available() is False

    # If availability_attr is set, adapter module should generally import and expose the flag.
    # If import fails or flag is missing/False, that is an acceptable "unavailable" signal.
    availability_attr = getattr(descriptor, "availability_attr", None)
    adapter_module = getattr(descriptor, "adapter_module", None)
    if availability_attr and adapter_module:
        try:
            module = importlib.import_module(adapter_module)
        except Exception:
            return
        flag = getattr(module, availability_attr, None)
        assert flag is None or bool(flag) is False


def _get_method(instance: Any, name: str | None) -> Callable[..., Any]:
    """Helper to fetch a method from the instance and assert it is callable."""
    assert name is not None, "Method name must not be None"
    attr = getattr(instance, name, None)
    assert callable(attr), f"{instance!r} missing expected callable method {name!r}"
    return attr


def _context_payload(descriptor: LLMFrameworkDescriptor) -> dict[str, Any]:
    """
    Build a minimal context payload for the framework.

    Why this exists:
    - Some frameworks accept context via a single structured kwarg (e.g. config=...).
    - Others accept context via **kwargs directly.
    - We provide a consistent payload (RICH_CONTEXT) and let _call_with_context
      route it to the correct call signature.
    """
    ctx = dict(RICH_CONTEXT)

    # Best-effort traceability: include framework name in tags.
    # This is non-fatal if tags cannot be coerced to a list.
    try:
        tags = list(ctx.get("tags", []))
        tags.append(descriptor.name)
        ctx["tags"] = tags
    except Exception:
        pass

    return ctx


def _call_with_context(
    descriptor: LLMFrameworkDescriptor,
    fn: Callable[..., Any],
    *args: Any,
    context: Mapping[str, Any],
) -> Any:
    """
    Call an LLM adapter method with context in a robust, framework-agnostic way.

    Primary strategy:
      - If descriptor.context_kwarg is set, pass {context_kwarg: context}.

    Compatibility fallback:
      - If that raises TypeError due to an unexpected keyword argument, retry by
        spreading the mapping into kwargs (useful for **kwargs style surfaces).

    This mirrors the resilience patterns used in other conformance suites.
    """
    context_kwarg = getattr(descriptor, "context_kwarg", None)
    if not context_kwarg:
        return fn(*args)

    try:
        return fn(*args, **{context_kwarg: dict(context)})
    except TypeError as e:
        msg = str(e)
        unexpected_kw = f"unexpected keyword argument '{context_kwarg}'" in msg or (
            "unexpected keyword" in msg and context_kwarg in msg
        )
        if unexpected_kw:
            # Spread as kwargs for BaseEmbedding-style / **kwargs LLM surfaces.
            return fn(*args, **dict(context))
        raise


def _consume_sync_stream_best_effort(iterator: Any) -> None:
    """
    Consume up to MAX_STREAM_CHUNKS_TO_CONSUME items from a sync iterator.

    This prevents runaway/never-ending iterators from hanging the suite while
    still forcing iteration-time errors to surface.
    """
    for i, _ in enumerate(iterator):  # noqa: B007
        if i + 1 >= MAX_STREAM_CHUNKS_TO_CONSUME:
            break


async def _consume_async_stream_best_effort(aiter: Any) -> None:
    """
    Consume up to MAX_STREAM_CHUNKS_TO_CONSUME items from an async iterator.

    This prevents runaway/never-ending async iterators from hanging the suite while
    still forcing iteration-time errors to surface.
    """
    n = 0
    async for _ in aiter:  # noqa: B007
        n += 1
        if n >= MAX_STREAM_CHUNKS_TO_CONSUME:
            break


def _make_llm_with_evil_backend(
    framework_descriptor: LLMFrameworkDescriptor,
    backend_cls: Type[Any],
) -> Any:
    """
    Instantiate the framework LLM adapter/client with an 'evil' backend.

    This bypasses the normal LLM adapter fixture wiring and lets us simulate
    misbehaving backends in a controlled way.
    """
    # Defensive import hardening: surface syntax errors with actionable diagnostics.
    try:
        module = importlib.import_module(framework_descriptor.adapter_module)
    except SyntaxError as e:
        pytest.fail(
            f"Adapter module failed to import for {framework_descriptor.name!r}: "
            f"SyntaxError at line {e.lineno}: {e.msg}\n"
            f"Text: {e.text!r}",
            pytrace=True,
        )

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
    #
    # NOTE: We avoid heavy reflection; checking co_varnames is fast and sufficient here.
    try:
        init_vars = getattr(llm_cls, "__init__", lambda *a, **k: None).__code__.co_varnames  # type: ignore[union-attr]
        if "model" in init_vars:
            init_kwargs.setdefault("model", "mock-backend-model")
    except Exception:
        # If introspection fails, we don't guess; adapters should either provide defaults
        # or the registry should capture the requirement.
        pass

    instance = llm_cls(**init_kwargs)
    return instance


def _require_core_surfaces_declared(descriptor: LLMFrameworkDescriptor) -> None:
    """
    Enforce minimal registry alignment for the LLM suite.

    Why this exists:
    - We want to avoid false confidence where tests silently skip because a method
      name wasn't declared in the descriptor.
    - Unlike the graph suite, LLM adapters may legitimately omit streaming and/or
      token counting surfaces. We enforce a smaller "core" contract here.

    Core contract:
    - invoke_method must be declared (sync completion surface)
    - async surfaces must be internally consistent with supports_async, when present
    """
    assert descriptor.invoke_method, f"{descriptor.name}: invoke_method must be declared"

    supports_async = getattr(descriptor, "supports_async", None)
    if supports_async is False:
        assert getattr(descriptor, "async_invoke_method", None) is None
        assert getattr(descriptor, "async_stream_method", None) is None
    elif supports_async is True:
        # If a framework advertises async, it must declare async invoke.
        assert getattr(descriptor, "async_invoke_method", None), (
            f"{descriptor.name}: supports_async is True but async_invoke_method is not declared"
        )


def _call_invoke(
    descriptor: LLMFrameworkDescriptor,
    instance: Any,
    prompt: str,
) -> Any:
    """
    Call the primary sync invoke/complete surface for the LLM.

    This abstracts over frameworks that use different method names and may
    accept context via a single kwarg or via **kwargs.
    """
    invoke_fn = _get_method(instance, descriptor.invoke_method)
    ctx = _context_payload(descriptor)
    return _call_with_context(descriptor, invoke_fn, prompt, context=ctx)


def _call_stream(
    descriptor: LLMFrameworkDescriptor,
    instance: Any,
    prompt: str,
) -> Any:
    """
    Call the sync streaming surface (if declared).

    NOTE: Streaming is optional for some LLM frameworks; tests guard on the
    descriptor's declared stream_method to avoid forcing optional surfaces.
    """
    assert descriptor.stream_method is not None
    stream_fn = _get_method(instance, descriptor.stream_method)
    ctx = _context_payload(descriptor)
    return _call_with_context(descriptor, stream_fn, prompt, context=ctx)


def _call_count_tokens(
    descriptor: LLMFrameworkDescriptor,
    instance: Any,
    prompt: str,
) -> Any:
    """
    Call the token counting surface (if declared).

    LLM nuance:
    - Not all frameworks expose token counting in their adapter interface.
    - When declared, it must return an int (>= 0) or raise.
    """
    count_tokens_method = getattr(descriptor, "count_tokens_method", None)
    assert count_tokens_method is not None
    fn = _get_method(instance, count_tokens_method)
    ctx = _context_payload(descriptor)
    return _call_with_context(descriptor, fn, prompt, context=ctx)


def _patch_attach_context(
    adapter_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[BaseException, dict[str, Any]]],
) -> None:
    """
    Patch attach_context in both:
      1) the adapter module (module-local reference used by decorators), and
      2) the shared corpus_sdk.core.error_context module.

    This ensures we observe context attachment even if an adapter references either symbol.
    """

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        calls.append((exc, dict(ctx)))

    # Patch adapter module local symbol (most common pattern).
    if hasattr(adapter_module, "attach_context"):
        monkeypatch.setattr(adapter_module, "attach_context", fake_attach_context)

    # Patch shared canonical location for safety.
    try:
        core_mod = importlib.import_module("corpus_sdk.core.error_context")
        if hasattr(core_mod, "attach_context"):
            monkeypatch.setattr(core_mod, "attach_context", fake_attach_context)
    except Exception:
        # Minimal environments may not import core module; module-local patch is still valuable.
        pass


def _assert_optional_surface_consistency(
    descriptor: LLMFrameworkDescriptor,
    attr_flag_name: str,
    method_attr_name: str,
) -> None:
    """
    Best-effort descriptor consistency check for optional surfaces.

    If a descriptor provides a boolean flag (e.g. supports_streaming=True) and
    the corresponding method name is missing, that indicates registry drift.

    This helper is intentionally defensive:
    - If the flag isn't present on the descriptor, we don't assume anything.
    """
    supports_flag = getattr(descriptor, attr_flag_name, None)
    method_name = getattr(descriptor, method_attr_name, None)
    if supports_flag is True:
        assert method_name, f"{descriptor.name}: {attr_flag_name} is True but {method_attr_name} is not declared"
    if supports_flag is False:
        assert method_name is None, f"{descriptor.name}: {attr_flag_name} is False but {method_attr_name} is declared"


def _is_text_like_or_chat_resultish(value: Any) -> bool:
    """
    Best-effort check for a "valid-looking" LLM completion result.

    LLM nuance:
    - Some frameworks return plain strings.
    - Others return ChatResult-like objects with content/text/message fields.

    We keep this intentionally permissive and non-prescriptive:
    - It is used only as a negative test to avoid accepting obviously empty results.
    """
    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    # Try common content-bearing attributes; treat non-empty strings as "text-like".
    for attr in ("content", "text", "message", "output", "generation"):
        if hasattr(value, attr):
            try:
                inner = getattr(value, attr)
                if isinstance(inner, str) and inner.strip():
                    return True
            except Exception:
                # If an attribute access fails, we do not treat it as valid.
                return False

    # Unknown object type: we conservatively treat it as "possibly valid".
    # This avoids overfitting to framework-specific result classes.
    return True


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

    IMPORTANT POLICY (no pytest.skip):
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    _require_core_surfaces_declared(framework_descriptor)

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

    LLM nuance:
    - Some adapters may not expose streaming; this test only runs when declared.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    _require_core_surfaces_declared(framework_descriptor)
    _assert_optional_surface_consistency(framework_descriptor, "supports_streaming", "stream_method")

    if not framework_descriptor.stream_method:
        # Streaming is optional; if not declared, we return after consistency checks above.
        return

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
        _consume_sync_stream_best_effort(iterator)


@pytest.mark.asyncio
async def test_async_invalid_backend_result_causes_errors_for_invoke_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
) -> None:
    """
    When async is supported, invalid backend results for the async invoke
    surface should also surface as errors, not valid-looking LLM results.

    IMPORTANT POLICY (no pytest.skip):
    - If framework is unavailable, validate the unavailable contract and return.
    - If async isn't supported, validate descriptor consistency and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    _require_core_surfaces_declared(framework_descriptor)

    supports_async = getattr(framework_descriptor, "supports_async", False)
    if not supports_async:
        assert framework_descriptor.async_invoke_method is None
        assert framework_descriptor.async_stream_method is None
        return

    assert framework_descriptor.async_invoke_method is not None

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        InvalidResultLLMBackend,
    )

    ainvoke_fn = _get_method(
        instance,
        framework_descriptor.async_invoke_method,
    )

    with pytest.raises(Exception):  # noqa: BLE001
        coro = _call_with_context(
            framework_descriptor,
            ainvoke_fn,
            "invalid-llm-async-invoke",
            context=_context_payload(framework_descriptor),
        )
        assert inspect.isawaitable(coro), "Async invoke method must return an awaitable"
        await coro  # noqa: PT018


@pytest.mark.asyncio
async def test_async_invalid_backend_result_causes_errors_for_stream_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
) -> None:
    """
    When async streaming is supported, invalid backend results for
    the async streaming surface should also surface as errors.

    LLM nuance:
    - Async streaming surfaces may return an async iterator directly OR an awaitable
      that resolves to an async iterator. We support both shapes.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    _require_core_surfaces_declared(framework_descriptor)
    _assert_optional_surface_consistency(framework_descriptor, "supports_async_streaming", "async_stream_method")

    supports_async = getattr(framework_descriptor, "supports_async", False)
    if not supports_async:
        assert framework_descriptor.async_stream_method is None
        return

    if not framework_descriptor.async_stream_method:
        # Async streaming is optional; if not declared, return after consistency checks above.
        return

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        InvalidResultLLMBackend,
    )

    astream_fn = _get_method(
        instance,
        framework_descriptor.async_stream_method,
    )

    with pytest.raises(Exception):  # noqa: BLE001
        aiter = _call_with_context(
            framework_descriptor,
            astream_fn,
            "invalid-llm-async-stream",
            context=_context_payload(framework_descriptor),
        )

        # Allow awaitable -> async iterator or async iterator directly
        if inspect.isawaitable(aiter):
            aiter = await aiter  # type: ignore[assignment]

        await _consume_async_stream_best_effort(aiter)


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
    - Return a result that is obviously not empty (text-like / ChatResult-like).

    This test asserts that we *don't* get an "empty" completion passed straight
    through to the caller.

    IMPORTANT POLICY (no pytest.skip):
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    _require_core_surfaces_declared(framework_descriptor)

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        EmptyResultLLMBackend,
    )

    try:
        result = _call_invoke(framework_descriptor, instance, "empty-llm-completion")
    except Exception:  # noqa: BLE001
        # Raising is acceptable and expected for many implementations.
        return

    # If it did not raise, the result must at least look non-empty.
    # This prevents false passes where adapters propagate None/empty text.
    assert _is_text_like_or_chat_resultish(result), (
        "EmptyResultLLMBackend produced an empty completion; adapters should "
        "treat empty completions as errors or coerce them into a meaningful text-like result."
    )


# ---------------------------------------------------------------------------
# Streaming error-context behavior (call-time + iteration-time)
# ---------------------------------------------------------------------------


def test_backend_exception_is_wrapped_with_error_context_on_stream_calltime_when_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Validate error-context decoration for sync streaming failures that occur at call-time.

    This aligns with the graph conformance suite's coverage of streaming failure modes.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    _require_core_surfaces_declared(framework_descriptor)
    _assert_optional_surface_consistency(framework_descriptor, "supports_streaming", "stream_method")

    if not framework_descriptor.stream_method:
        return

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls: list[tuple[BaseException, dict[str, Any]]] = []
    _patch_attach_context(module, monkeypatch, calls)

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        RaisingLLMBackend,
    )

    with pytest.raises(RuntimeError, match="backend failure"):
        _call_stream(framework_descriptor, instance, "err-llm-sync-stream-calltime")

    assert calls, "attach_context was not called for backend stream call-time failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert ctx.get("framework") == framework_descriptor.name
    assert str(ctx.get("operation", "")).startswith(LLM_OPERATION_PREFIX)


def test_backend_exception_is_wrapped_with_error_context_on_stream_iteration_when_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Validate error-context decoration for sync streaming failures that occur during iteration.

    This models real streaming failure modes where a stream begins successfully,
    then fails while being consumed.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    _require_core_surfaces_declared(framework_descriptor)
    _assert_optional_surface_consistency(framework_descriptor, "supports_streaming", "stream_method")

    if not framework_descriptor.stream_method:
        return

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls: list[tuple[BaseException, dict[str, Any]]] = []
    _patch_attach_context(module, monkeypatch, calls)

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        IterationRaisingLLMBackend,
    )

    with pytest.raises(RuntimeError, match="backend failure"):
        iterator = _call_stream(framework_descriptor, instance, "err-llm-sync-stream-iteration")

        # Force iteration to trigger iteration-time exceptions while avoiding hangs.
        assert isinstance(iterator, Iterable) or True  # best-effort; iterability validated by consumption
        _consume_sync_stream_best_effort(iterator)

    assert calls, "attach_context was not called for backend stream iteration failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert ctx.get("framework") == framework_descriptor.name
    assert str(ctx.get("operation", "")).startswith(LLM_OPERATION_PREFIX)


@pytest.mark.asyncio
async def test_async_backend_exception_is_wrapped_with_error_context_on_stream_calltime_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async streaming is supported, backend exceptions at call-time must
    go through the error-context decorators and call attach_context().

    LLM nuance:
    - The async streaming surface may return an async iterator directly or an
      awaitable that resolves to an async iterator.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    _require_core_surfaces_declared(framework_descriptor)
    _assert_optional_surface_consistency(framework_descriptor, "supports_async_streaming", "async_stream_method")

    supports_async = getattr(framework_descriptor, "supports_async", False)
    if not supports_async:
        assert framework_descriptor.async_stream_method is None
        return

    if not framework_descriptor.async_stream_method:
        return

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls: list[tuple[BaseException, dict[str, Any]]] = []
    _patch_attach_context(module, monkeypatch, calls)

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        RaisingLLMBackend,
    )

    astream_fn = _get_method(instance, framework_descriptor.async_stream_method)

    with pytest.raises(RuntimeError, match="backend failure"):
        aiter = _call_with_context(
            framework_descriptor,
            astream_fn,
            "err-llm-async-stream-calltime",
            context=_context_payload(framework_descriptor),
        )
        if inspect.isawaitable(aiter):
            aiter = await aiter  # type: ignore[assignment]
        await _consume_async_stream_best_effort(aiter)

    assert calls, "attach_context was not called for async backend stream call-time failures"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert ctx.get("framework") == framework_descriptor.name
    assert str(ctx.get("operation", "")).startswith(LLM_OPERATION_PREFIX)


@pytest.mark.asyncio
async def test_async_backend_exception_is_wrapped_with_error_context_on_stream_iteration_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async streaming is supported, backend exceptions during async iteration
    must also go through the error-context decorators and call attach_context().

    This models real streaming failure modes where an async stream begins and then
    fails mid-consumption.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    _require_core_surfaces_declared(framework_descriptor)
    _assert_optional_surface_consistency(framework_descriptor, "supports_async_streaming", "async_stream_method")

    supports_async = getattr(framework_descriptor, "supports_async", False)
    if not supports_async:
        assert framework_descriptor.async_stream_method is None
        return

    if not framework_descriptor.async_stream_method:
        return

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls: list[tuple[BaseException, dict[str, Any]]] = []
    _patch_attach_context(module, monkeypatch, calls)

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        IterationRaisingLLMBackend,
    )

    astream_fn = _get_method(instance, framework_descriptor.async_stream_method)

    with pytest.raises(RuntimeError, match="backend failure"):
        aiter = _call_with_context(
            framework_descriptor,
            astream_fn,
            "err-llm-async-stream-iteration",
            context=_context_payload(framework_descriptor),
        )
        if inspect.isawaitable(aiter):
            aiter = await aiter  # type: ignore[assignment]

        # Force async iteration to trigger iteration-time exceptions while avoiding hangs.
        await _consume_async_stream_best_effort(aiter)

    assert calls, "attach_context was not called for async backend stream iteration failures"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert ctx.get("framework") == framework_descriptor.name
    assert str(ctx.get("operation", "")).startswith(LLM_OPERATION_PREFIX)


# ---------------------------------------------------------------------------
# Token counting behavior (optional LLM surface)
# ---------------------------------------------------------------------------


def test_invalid_backend_result_causes_errors_for_count_tokens_when_declared(
    framework_descriptor: LLMFrameworkDescriptor,
) -> None:
    """
    If the backend returns a clearly invalid result type for token counting,
    the framework adapter should surface an error rather than silently treating
    it as a valid token count.

    Acceptable behaviors:
    - Raise an Exception (preferred), or
    - Return an int (>= 0).

    LLM nuance:
    - Not all frameworks expose token counting; this test runs only when declared.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    _require_core_surfaces_declared(framework_descriptor)
    _assert_optional_surface_consistency(framework_descriptor, "supports_token_counting", "count_tokens_method")

    count_tokens_method = getattr(framework_descriptor, "count_tokens_method", None)
    if not count_tokens_method:
        return

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        InvalidResultLLMBackend,
    )

    try:
        result = _call_count_tokens(framework_descriptor, instance, "count-tokens-invalid")
    except Exception:  # noqa: BLE001
        return

    assert isinstance(result, int) and result >= 0, (
        "InvalidResultLLMBackend produced a non-integer token count; adapters should "
        "raise or coerce token counts into non-negative integers."
    )


def test_backend_exception_is_wrapped_with_error_context_on_count_tokens_when_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the backend raises during token counting, the framework adapter's
    error-context decorator should call attach_context().

    LLM nuance:
    - Not all frameworks expose token counting; this test runs only when declared.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    _require_core_surfaces_declared(framework_descriptor)
    _assert_optional_surface_consistency(framework_descriptor, "supports_token_counting", "count_tokens_method")

    count_tokens_method = getattr(framework_descriptor, "count_tokens_method", None)
    if not count_tokens_method:
        return

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls: list[tuple[BaseException, dict[str, Any]]] = []
    _patch_attach_context(module, monkeypatch, calls)

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        RaisingLLMBackend,
    )

    with pytest.raises(RuntimeError, match="backend failure"):
        _call_count_tokens(framework_descriptor, instance, "count-tokens-raise")

    assert calls, "attach_context was not called for backend token counting failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert ctx.get("framework") == framework_descriptor.name
    assert str(ctx.get("operation", "")).startswith(LLM_OPERATION_PREFIX)


# ---------------------------------------------------------------------------
# Error-context when backend raises (invoke)
# ---------------------------------------------------------------------------


def test_backend_exception_is_wrapped_with_error_context_on_invoke(
    framework_descriptor: LLMFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the backend raises during a sync invoke/complete operation, the
    framework adapter's error-context decorator should:

    - call attach_context() with the exception and useful metadata, and
    - re-raise the original exception (or a wrapped one).

    IMPORTANT POLICY (no pytest.skip):
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    _require_core_surfaces_declared(framework_descriptor)

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls: list[tuple[BaseException, dict[str, Any]]] = []
    _patch_attach_context(module, monkeypatch, calls)

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

    IMPORTANT POLICY (no pytest.skip):
    - If framework is unavailable, validate the unavailable contract and return.
    - If async isn't supported, validate descriptor consistency and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    _require_core_surfaces_declared(framework_descriptor)

    supports_async = getattr(framework_descriptor, "supports_async", False)
    if not supports_async:
        assert framework_descriptor.async_invoke_method is None
        assert framework_descriptor.async_stream_method is None
        return

    assert framework_descriptor.async_invoke_method is not None

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls: list[tuple[BaseException, dict[str, Any]]] = []
    _patch_attach_context(module, monkeypatch, calls)

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        RaisingLLMBackend,
    )

    ainvoke_fn = _get_method(
        instance,
        framework_descriptor.async_invoke_method,
    )

    with pytest.raises(RuntimeError, match="backend failure"):
        coro = _call_with_context(
            framework_descriptor,
            ainvoke_fn,
            "err-llm-async-invoke",
            context=_context_payload(framework_descriptor),
        )
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
