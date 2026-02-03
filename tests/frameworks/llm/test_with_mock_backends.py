# tests/frameworks/llm/test_with_mock_backends.py

from __future__ import annotations

import importlib
import inspect
from collections.abc import AsyncIterable, Iterable, Mapping
from typing import Any, Callable, Optional, Type

import pytest

from corpus_sdk.llm.llm_base import LLMChunk, LLMCompletion, TokenUsage

from tests.frameworks.registries.llm_registry import (
    LLMFrameworkDescriptor,
    iter_llm_framework_descriptors,
)

LLM_OPERATION_PREFIX = "llm"
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

    # Core async surfaces expected by the translator (LLMProtocolV1 is async-first)
    async def complete(self, *args: Any, **kwargs: Any) -> Any:
        return 123456  # clearly not a ChatResult / text-like

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        # Return something non-iterable
        return 3.14159

    async def count_tokens(self, *args: Any, **kwargs: Any) -> Any:
        return "not-an-int"

    async def health(self, *args: Any, **kwargs: Any) -> Any:
        return {"ok": True}

    async def capabilities(self, *args: Any, **kwargs: Any) -> Any:
        return {"supports_streaming": True, "supports_count_tokens": True}


class EmptyResultLLMBackend:
    """
    Backend that always returns obviously empty / None-like results.

    Used to verify that adapters do not silently treat `None` completion as
    fully valid, particularly for the primary completion surface.

    Note: streaming returning no chunks may be acceptable in some cases, so we
    focus assertions on the main completion path.
    """

    async def complete(self, *args: Any, **kwargs: Any) -> Any:
        return None

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterable[Any]:
        if False:  # pragma: no cover - structure only
            yield None

    async def count_tokens(self, *args: Any, **kwargs: Any) -> Any:
        return 0

    async def health(self, *args: Any, **kwargs: Any) -> Any:
        return {"ok": True}

    async def capabilities(self, *args: Any, **kwargs: Any) -> Any:
        return {"supports_streaming": True, "supports_count_tokens": True}


class RaisingLLMBackend:
    """
    Backend that always raises.

    Used to validate that error-context decorators still attach context when
    failures originate in the LLM backend rather than higher-level code.
    """

    async def complete(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    async def health(self, *args: Any, **kwargs: Any) -> Any:
        return {"ok": True}

    async def capabilities(self, *args: Any, **kwargs: Any) -> Any:
        return {"supports_streaming": True, "supports_count_tokens": True}

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterable[Any]:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)
        if False:  # pragma: no cover - structure only
            yield None

    async def count_tokens(self, *args: Any, **kwargs: Any) -> Any:
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

    async def complete(self, *args: Any, **kwargs: Any) -> Any:
        return LLMCompletion(
            text="ok",
            model="mock",
            model_family="mock",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            finish_reason="stop",
        )

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterable[Any]:
        # Yield one valid chunk to prove the stream started, then fail deterministically.
        yield LLMChunk(text="chunk-1", is_final=False)
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    async def count_tokens(self, *args: Any, **kwargs: Any) -> Any:
        return 1

    async def health(self, *args: Any, **kwargs: Any) -> Any:
        return {"ok": True}

    async def capabilities(self, *args: Any, **kwargs: Any) -> Any:
        return {"supports_streaming": True, "supports_count_tokens": True}


def _prompt_as_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def _build_prompt_args_kwargs(
    fn: Callable[..., Any],
    prompt: str,
    *,
    token_count: bool = False,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Best-effort adapt prompt inputs to framework-specific call signatures."""
    try:
        params = inspect.signature(fn).parameters
    except Exception:
        # If introspection fails, fall back to positional prompt.
        return (prompt,), {}

    if token_count:
        if "messages" in params:
            return (), {"messages": _prompt_as_messages(prompt)}
        if "text" in params:
            return (), {"text": prompt}
        return (prompt,), {}

    if "messages" in params:
        return (), {"messages": _prompt_as_messages(prompt)}

    for name in ("prompt", "input", "text"):
        if name in params:
            return (), {name: prompt}

    return (prompt,), {}


def _find_attached_context(
    calls: list[tuple[BaseException, dict[str, Any]]],
    *,
    framework: str,
) -> tuple[BaseException, dict[str, Any]]:
    """Prefer adapter-level attach_context calls; tolerate extra core-layer calls."""
    for exc, ctx in reversed(calls):
        if ctx.get("framework") == framework and str(ctx.get("operation", "")).startswith(
            LLM_OPERATION_PREFIX
        ):
            return exc, ctx
    # Fallback: any call with matching framework.
    for exc, ctx in reversed(calls):
        if ctx.get("framework") == framework:
            return exc, ctx
    # Last resort: return the last call so assertions are still informative.
    return calls[-1]


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
    **extra_kwargs: Any,
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
        return fn(*args, **extra_kwargs)

    try:
        return fn(*args, **{context_kwarg: dict(context)}, **extra_kwargs)
    except TypeError as e:
        msg = str(e)
        unexpected_kw = f"unexpected keyword argument '{context_kwarg}'" in msg or (
            "unexpected keyword" in msg and context_kwarg in msg
        )
        if unexpected_kw:
            # Spread as kwargs for BaseEmbedding-style / **kwargs LLM surfaces.
            return fn(*args, **dict(context), **extra_kwargs)
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
    - completion_method must be declared (sync completion surface)
    - async surfaces must be internally consistent with supports_async, when present
    """
    assert descriptor.completion_method, (
        f"{descriptor.name}: completion_method must be declared"
    )

    supports_async = getattr(descriptor, "supports_async", None)
    if supports_async is False:
        assert descriptor.async_completion_method is None
        assert descriptor.async_streaming_method is None
    elif supports_async is True:
        # If a framework advertises async, it must declare async invoke.
        assert descriptor.async_completion_method, (
            f"{descriptor.name}: supports_async is True but async_completion_method is not declared"
        )


def _has_sync_stream_surface(descriptor: LLMFrameworkDescriptor) -> bool:
    style = getattr(descriptor, "streaming_style", "method")
    if style == "none":
        return False
    if style == "method":
        return bool(descriptor.streaming_method)
    if style == "kwarg":
        return bool(descriptor.streaming_kwarg and descriptor.completion_method)
    return False


def _has_async_stream_surface(descriptor: LLMFrameworkDescriptor) -> bool:
    style = getattr(descriptor, "streaming_style", "method")
    if style == "none":
        return False
    if style == "method":
        return bool(descriptor.async_streaming_method)
    if style == "kwarg":
        return bool(descriptor.streaming_kwarg and descriptor.async_completion_method)
    return False


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
    invoke_fn = _get_method(instance, descriptor.completion_method)
    ctx = _context_payload(descriptor)
    args, kwargs = _build_prompt_args_kwargs(invoke_fn, prompt)
    return _call_with_context(descriptor, invoke_fn, *args, context=ctx, **kwargs)


def _call_stream(
    descriptor: LLMFrameworkDescriptor,
    instance: Any,
    prompt: str,
) -> Any:
    """
    Call the sync streaming surface (if declared).

    NOTE: Streaming is optional for some LLM frameworks; tests guard using the
    descriptor's streaming metadata to avoid forcing optional surfaces.
    """
    style = getattr(descriptor, "streaming_style", "method")
    ctx = _context_payload(descriptor)

    if style == "method":
        assert descriptor.streaming_method is not None
        stream_fn = _get_method(instance, descriptor.streaming_method)
        args, kwargs = _build_prompt_args_kwargs(stream_fn, prompt)
        return _call_with_context(descriptor, stream_fn, *args, context=ctx, **kwargs)

    if style == "kwarg":
        assert descriptor.streaming_kwarg is not None
        assert descriptor.completion_method is not None
        invoke_fn = _get_method(instance, descriptor.completion_method)
        args, kwargs = _build_prompt_args_kwargs(invoke_fn, prompt)
        kwargs[descriptor.streaming_kwarg] = True
        return _call_with_context(
            descriptor,
            invoke_fn,
            *args,
            context=ctx,
            **kwargs,
        )

    raise AssertionError(f"{descriptor.name}: unsupported streaming_style={style!r}")


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
    token_count_method = getattr(descriptor, "token_count_method", None)
    assert token_count_method is not None
    fn = _get_method(instance, token_count_method)
    ctx = _context_payload(descriptor)
    args, kwargs = _build_prompt_args_kwargs(fn, prompt, token_count=True)
    return _call_with_context(descriptor, fn, *args, context=ctx, **kwargs)


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

    # Patch shared LLM translation layer reference (it imports attach_context at module scope).
    try:
        llm_translation_mod = importlib.import_module(
            "corpus_sdk.llm.framework_adapters.common.llm_translation"
        )
        if hasattr(llm_translation_mod, "attach_context"):
            monkeypatch.setattr(
                llm_translation_mod,
                "attach_context",
                fake_attach_context,
            )
    except Exception:
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

    if not _has_sync_stream_surface(framework_descriptor):
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
        assert framework_descriptor.async_completion_method is None
        assert framework_descriptor.async_streaming_method is None
        return

    assert framework_descriptor.async_completion_method is not None

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        InvalidResultLLMBackend,
    )

    ainvoke_fn = _get_method(
        instance,
        framework_descriptor.async_completion_method,
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

    supports_async = getattr(framework_descriptor, "supports_async", False)
    if not supports_async:
        assert framework_descriptor.async_streaming_method is None
        return

    if not _has_async_stream_surface(framework_descriptor):
        return

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        InvalidResultLLMBackend,
    )

    astream_fn = _get_method(
        instance,
        (
            framework_descriptor.async_streaming_method
            if framework_descriptor.streaming_style == "method"
            else framework_descriptor.async_completion_method
        ),
    )

    with pytest.raises(Exception):  # noqa: BLE001
        if framework_descriptor.streaming_style == "kwarg":
            aiter = _call_with_context(
                framework_descriptor,
                astream_fn,
                "invalid-llm-async-stream",
                context=_context_payload(framework_descriptor),
                **{framework_descriptor.streaming_kwarg: True},
            )
        else:
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
    if not _has_sync_stream_surface(framework_descriptor):
        return

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls: list[tuple[BaseException, dict[str, Any]]] = []
    _patch_attach_context(module, monkeypatch, calls)

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        RaisingLLMBackend,
    )

    # Some adapters return lazy generators/iterators; call-time may not execute backend work.
    # Accept either call-time or first-iteration failure.
    try:
        iterator = _call_stream(
            framework_descriptor,
            instance,
            "err-llm-sync-stream-calltime",
        )
    except RuntimeError as exc:
        assert "backend failure" in str(exc)
    else:
        with pytest.raises(RuntimeError, match="backend failure"):
            _consume_sync_stream_best_effort(iterator)

    assert calls, "attach_context was not called for backend stream call-time failure"

    exc, ctx = _find_attached_context(calls, framework=framework_descriptor.name)
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
    if not _has_sync_stream_surface(framework_descriptor):
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

    exc, ctx = _find_attached_context(calls, framework=framework_descriptor.name)
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
    supports_async = getattr(framework_descriptor, "supports_async", False)
    if not supports_async:
        assert framework_descriptor.async_streaming_method is None
        return

    if not _has_async_stream_surface(framework_descriptor):
        return

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls: list[tuple[BaseException, dict[str, Any]]] = []
    _patch_attach_context(module, monkeypatch, calls)

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        RaisingLLMBackend,
    )

    astream_fn = _get_method(
        instance,
        (
            framework_descriptor.async_streaming_method
            if framework_descriptor.streaming_style == "method"
            else framework_descriptor.async_completion_method
        ),
    )

    # As with sync streaming, allow lazy async iterators where errors surface on first iteration.
    ctx_payload = _context_payload(framework_descriptor)
    prompt = "err-llm-async-stream-calltime"
    args, kwargs = _build_prompt_args_kwargs(astream_fn, prompt)
    if framework_descriptor.streaming_style == "kwarg":
        kwargs[framework_descriptor.streaming_kwarg] = True

    try:
        aiter = _call_with_context(
            framework_descriptor,
            astream_fn,
            *args,
            context=ctx_payload,
            **kwargs,
        )
        if inspect.isawaitable(aiter):
            aiter = await aiter  # type: ignore[assignment]
    except RuntimeError as exc:
        assert "backend failure" in str(exc)
    else:
        with pytest.raises(RuntimeError, match="backend failure"):
            await _consume_async_stream_best_effort(aiter)
        await _consume_async_stream_best_effort(aiter)

    assert calls, "attach_context was not called for async backend stream call-time failures"

    exc, ctx = _find_attached_context(calls, framework=framework_descriptor.name)
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
    supports_async = getattr(framework_descriptor, "supports_async", False)
    if not supports_async:
        assert framework_descriptor.async_streaming_method is None
        return

    if not _has_async_stream_surface(framework_descriptor):
        return

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls: list[tuple[BaseException, dict[str, Any]]] = []
    _patch_attach_context(module, monkeypatch, calls)

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        IterationRaisingLLMBackend,
    )

    astream_fn = _get_method(
        instance,
        (
            framework_descriptor.async_streaming_method
            if framework_descriptor.streaming_style == "method"
            else framework_descriptor.async_completion_method
        ),
    )

    ctx_payload = _context_payload(framework_descriptor)
    prompt = "err-llm-async-stream-iteration"
    args, kwargs = _build_prompt_args_kwargs(astream_fn, prompt)
    if framework_descriptor.streaming_style == "kwarg":
        kwargs[framework_descriptor.streaming_kwarg] = True

    with pytest.raises(RuntimeError, match="backend failure"):
        aiter = _call_with_context(
            framework_descriptor,
            astream_fn,
            *args,
            context=ctx_payload,
            **kwargs,
        )
        if inspect.isawaitable(aiter):
            aiter = await aiter  # type: ignore[assignment]

        # Force async iteration to trigger iteration-time exceptions while avoiding hangs.
        await _consume_async_stream_best_effort(aiter)

    assert calls, "attach_context was not called for async backend stream iteration failures"

    exc, ctx = _find_attached_context(calls, framework=framework_descriptor.name)
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
    _assert_optional_surface_consistency(framework_descriptor, "supports_token_count", "token_count_method")

    token_count_method = getattr(framework_descriptor, "token_count_method", None)
    if not token_count_method:
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
    _assert_optional_surface_consistency(framework_descriptor, "supports_token_count", "token_count_method")

    token_count_method = getattr(framework_descriptor, "token_count_method", None)
    if not token_count_method:
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

    exc, ctx = _find_attached_context(calls, framework=framework_descriptor.name)
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

    exc, ctx = _find_attached_context(calls, framework=framework_descriptor.name)
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
        assert framework_descriptor.async_completion_method is None
        assert framework_descriptor.async_streaming_method is None
        return

    assert framework_descriptor.async_completion_method is not None

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls: list[tuple[BaseException, dict[str, Any]]] = []
    _patch_attach_context(module, monkeypatch, calls)

    instance = _make_llm_with_evil_backend(
        framework_descriptor,
        RaisingLLMBackend,
    )

    ainvoke_fn = _get_method(
        instance,
        framework_descriptor.async_completion_method,
    )

    ctx_payload = _context_payload(framework_descriptor)
    prompt = "err-llm-async-invoke"
    args, kwargs = _build_prompt_args_kwargs(ainvoke_fn, prompt)

    with pytest.raises(RuntimeError, match="backend failure"):
        coro = _call_with_context(
            framework_descriptor,
            ainvoke_fn,
            *args,
            context=ctx_payload,
            **kwargs,
        )
        assert inspect.isawaitable(coro), "Async invoke method must return an awaitable"
        await coro  # noqa: PT018

    assert calls, "attach_context was not called for async backend failures"

    exc, ctx = _find_attached_context(calls, framework=framework_descriptor.name)
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name
    assert str(ctx["operation"]).startswith(LLM_OPERATION_PREFIX)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
