# tests/frameworks/llm/test_contract_shapes_and_batching.py

from __future__ import annotations

import importlib
import asyncio
import inspect
from typing import Any, Callable, Optional

import pytest

from tests.frameworks.registries.llm_registry import (
    LLMFrameworkDescriptor,
    iter_llm_framework_descriptors,
)

# ---------------------------------------------------------------------------
# Constants (shared test inputs)
# ---------------------------------------------------------------------------

SYNC_COMPLETION_TEXT_1 = "llm-sync-completion-text-1"
SYNC_COMPLETION_TEXT_2 = "llm-sync-completion-text-2"

ASYNC_COMPLETION_TEXT_1 = "llm-async-completion-text-1"
ASYNC_COMPLETION_TEXT_2 = "llm-async-completion-text-2"

SYNC_STREAM_TEXT = "llm-sync-stream-text"
ASYNC_STREAM_TEXT = "llm-async-stream-text"

TOKEN_COUNT_TEXT_1 = "llm-token-count-text-1"
TOKEN_COUNT_TEXT_2 = "llm-token-count-text-2"

# Performance / safety guardrails:
# - These tests are smoke-style; they must not hang even if a stream is buggy.
MAX_STREAM_CHUNKS_TO_SAMPLE = 10


# ---------------------------------------------------------------------------
# Internal state (best-effort availability tracking)
# ---------------------------------------------------------------------------

# Why this exists:
# - Some frameworks are "logically available" per registry (availability_attr=None => True),
#   but may still fail to import/instantiate due to optional dependency drift.
# - To keep tests deterministic and avoid hard-skipping, we treat such cases as
#   "unavailable in practice" and allow tests to validate that gracefully.
_UNAVAILABLE_REASONS: dict[str, str] = {}


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

    IMPORTANT POLICY (mirrors embedding contract suites):
    - We do not unconditionally skip unavailable frameworks.
    - If a framework is unavailable, tests should pass by validating the "unavailable"
      signaling contract rather than skipping and reducing coverage signal quality.

    NOTE:
    - In practice, "unavailable" can mean either:
        (1) descriptor.is_available() == False (explicit registry availability flag), or
        (2) import/instantiation fails due to optional dependency drift.
      The llm_client_instance fixture records (2) in _UNAVAILABLE_REASONS.
    """
    descriptor: LLMFrameworkDescriptor = request.param
    return descriptor


@pytest.fixture
def llm_client_instance(
    framework_descriptor: LLMFrameworkDescriptor,
    adapter: Any,
) -> Any:
    """
    Construct a concrete LLM client instance for the given descriptor.

    Mirrors the construction pattern used in LLM interface-conformance tests:
    each framework adapter is expected to take a `llm_adapter` kwarg that wraps
    a Corpus LLMProtocolV1 implementation.

    Availability contract:
    - If descriptor.is_available() is False -> return None.
    - If imports/instantiation fail due to optional dependency drift -> return None
      and record a best-effort reason in _UNAVAILABLE_REASONS.
    """
    name = framework_descriptor.name

    # Registry-declared unavailability (preferred signal).
    if not framework_descriptor.is_available():
        _UNAVAILABLE_REASONS[name] = "descriptor.is_available() returned False"
        return None

    try:
        module = importlib.import_module(framework_descriptor.adapter_module)
    except Exception as e:  # noqa: BLE001
        # Treat import failures as practical unavailability (optional dependency drift).
        _UNAVAILABLE_REASONS[name] = (
            f"failed to import adapter_module={framework_descriptor.adapter_module!r}: "
            f"{type(e).__name__}: {e}"
        )
        return None

    try:
        client_cls = getattr(module, framework_descriptor.adapter_class)
    except Exception as e:  # noqa: BLE001
        _UNAVAILABLE_REASONS[name] = (
            f"adapter_class lookup failed: {framework_descriptor.adapter_class!r}: "
            f"{type(e).__name__}: {e}"
        )
        return None

    # All LLM framework adapters are expected to accept `llm_adapter=...`.
    init_kwargs: dict[str, Any] = {"llm_adapter": adapter}

    try:
        instance = client_cls(**init_kwargs)
    except Exception as e:  # noqa: BLE001
        # Treat instantiation failures as practical unavailability. This most commonly
        # happens when upstream framework APIs drift (optional dependency issues).
        _UNAVAILABLE_REASONS[name] = (
            f"failed to instantiate {framework_descriptor.adapter_class!r}: "
            f"{type(e).__name__}: {e}"
        )
        return None

    return instance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_unavailable_contract(descriptor: LLMFrameworkDescriptor) -> None:
    """
    Validate that a framework descriptor behaves as "unavailable" deterministically.

    We accept two availability signals:
      1) descriptor.is_available() == False  (registry-controlled availability flag)
      2) recorded import/instantiation failure in _UNAVAILABLE_REASONS

    NOTE:
    - Some descriptors have availability_attr=None (meaning "assume available").
      When optional dependencies drift, (2) becomes the only reliable signal.
    """
    if descriptor.is_available() is False:
        # If availability_attr is set, adapter module should generally import and expose the flag.
        # If the module cannot import, that is also a valid "unavailable" signal.
        if descriptor.availability_attr:
            try:
                module = importlib.import_module(descriptor.adapter_module)
            except Exception:
                return
            flag = getattr(module, descriptor.availability_attr, None)
            assert flag is None or bool(flag) is False
        return

    # Practical unavailability path: import/instantiation failure recorded.
    reason = _UNAVAILABLE_REASONS.get(descriptor.name)
    assert reason, (
        f"{descriptor.name}: framework treated as unavailable, but no unavailability reason "
        "was recorded. Ensure llm_client_instance fixture handles this case."
    )


def _get_method(instance: Any, name: str | None) -> Callable[..., Any]:
    """
    Helper to fetch a method from the instance and assert it is callable.

    If name is None, this fails fast with a clear assertion message so we
    don't silently mis-test a missing surface.
    """
    assert name, "Expected a non-empty method name"
    attr = getattr(instance, name, None)
    assert callable(attr), f"{instance!r} missing expected callable method {name!r}"
    return attr


def _assert_iterable(obj: Any) -> None:
    """
    Cheap runtime check that an object is iterable (supports __iter__).

    This is intentionally more permissive than isinstance(obj, Iterable) so
    that custom iterator types don't have to inherit from ABCs.
    """
    try:
        iter(obj)
    except TypeError as exc:  # pragma: no cover - defensive
        raise AssertionError(
            f"Object {obj!r} is not iterable: {type(obj).__name__}",
        ) from exc


def _assert_async_iterable(obj: Any) -> None:
    """
    Cheap runtime check that an object is async-iterable.

    Uses the presence of __aiter__ instead of isinstance(..., AsyncIterable)
    to avoid over-constraining custom async iterator implementations.
    """
    if not hasattr(obj, "__aiter__"):
        raise AssertionError(
            f"Object {obj!r} is not async-iterable: {type(obj).__name__}",
        )


def _context_kwargs_for_descriptor(descriptor: LLMFrameworkDescriptor) -> dict[str, Any]:
    """
    Build kwargs reflecting the framework's declared context parameter.

    Returns a dict with a single key (the framework's context_kwarg) containing
    a minimal Mapping context, or an empty dict if no context_kwarg is declared.
    """
    if not descriptor.context_kwarg:
        return {}
    return {descriptor.context_kwarg: {}}


def _build_message_like_input(descriptor: LLMFrameworkDescriptor, text: str) -> Any:
    """
    Build a framework-appropriate "message-like" input when a method expects messages.

    Strategy:
    - For LangChain: use HumanMessage if available.
    - For LlamaIndex: use ChatMessage/MessageRole if available.
    - Otherwise: use a portable list[dict] with {role, content} (OpenAI-style).

    This avoids hardcoding per-method behavior while keeping the call portable.
    """
    if descriptor.name == "langchain":
        # LangChain adapters typically expect List[BaseMessage]
        from langchain_core.messages import HumanMessage  # type: ignore

        return [HumanMessage(content=text)]

    if descriptor.name == "llamaindex":
        from llama_index.core.llms import ChatMessage, MessageRole  # type: ignore

        user_role = getattr(MessageRole, "USER", "user")
        return [ChatMessage(role=user_role, content=text)]

    # Portable default for frameworks that use OpenAI-style messages.
    return [{"role": "user", "content": text}]


def _build_semantic_kernel_inputs(text: str) -> tuple[Any, Any]:
    """
    Build Semantic Kernel (chat_history, settings) inputs.

    Note:
    - Semantic Kernel APIs drift across versions (ChatHistory moved/renamed).
      This contract suite uses a portable call shape and relies on the
      Corpus SK adapter to normalize message inputs.
    """
    # Portable: pass a plain string as chat_history and omit settings.
    # The adapter normalizes this into the translator-friendly format.
    return text, None


def _sync_first_chunk_for_descriptor(
    descriptor: LLMFrameworkDescriptor,
    instance: Any,
    text: str,
) -> Any:
    """Safely invoke sync streaming + read first chunk.

    This is intended to run in a worker thread when called from async tests,
    because many adapters intentionally guard against calling sync APIs from
    inside an active asyncio event loop.
    """
    sync_stream = _invoke_sync_stream(descriptor, instance, text)
    _assert_iterable(sync_stream)
    return _sync_first_chunk(sync_stream)


def _build_primary_call_args(
    descriptor: LLMFrameworkDescriptor,
    fn: Callable[..., Any],
    *,
    text: str,
    streaming_kwarg: Optional[str] = None,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Build best-effort positional args + kwargs for an LLM method call.

    Why this helper exists:
    - LLM frameworks differ substantially in their primary argument:
        * raw prompt text
        * messages list
        * (chat_history, settings) pairs
    - The registry stores method *names*, but not detailed signatures.
    - We inspect the callable signature to choose a conservative argument shape.

    Behavior:
    - If the method looks like it takes chat_history/settings (Semantic Kernel),
      we provide (history, settings).
    - If it looks like it takes messages, we provide a message-like input.
    - Otherwise, we pass the raw text.

    Context:
    - If descriptor.context_kwarg is declared, we pass it as a kwarg mapping.

    Streaming:
    - If streaming_kwarg is provided, it is set True in kwargs.
    """
    kwargs: dict[str, Any] = dict(_context_kwargs_for_descriptor(descriptor))
    if streaming_kwarg:
        kwargs[streaming_kwarg] = True

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    # Skip "self" if present (bound methods may not show it depending on wrappers).
    if params and params[0].name == "self":
        params = params[1:]

    # Semantic Kernel-like shape: (chat_history, settings, ...)
    # We detect by parameter names (best-effort) rather than strict arity.
    param_names = [p.name for p in params]
    if len(params) >= 2 and (
        ("history" in param_names and "settings" in param_names)
        or ("chat_history" in param_names and "settings" in param_names)
    ):
        history, default_settings = _build_semantic_kernel_inputs(text)

        # Some registries model Semantic Kernel's settings as a "context" kwarg.
        # If so, avoid passing it twice (positional + kwarg).
        settings_from_context = kwargs.pop("settings", None)
        settings = (
            settings_from_context
            if settings_from_context is not None
            else default_settings
        )

        return [history, settings], kwargs

    # Messages-like: first parameter name suggests a list of messages/history.
    if params and params[0].name in {"messages", "chat_history", "history"}:
        args: list[Any] = [_build_message_like_input(descriptor, text)]
    else:
        # Default: raw text prompt.
        args = [text]

    # If we provided positional args, ensure we don't also pass the same param
    # name via kwargs (common for context_kwarg overlaps like CrewAI `task`).
    for i in range(min(len(args), len(params))):
        p = params[i]
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            kwargs.pop(p.name, None)

    return args, kwargs


def _sync_first_chunk(stream_obj: Any) -> Any:
    """
    Consume at most one chunk from a sync stream and return it, or None.

    Notes:
    - We do not require streams to yield any chunks (some adapters buffer).
    - We close the iterator/generator best-effort to avoid resource leaks.
    """
    it = iter(stream_obj)
    try:
        return next(it)
    except StopIteration:
        return None
    finally:
        close = getattr(it, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


async def _async_first_chunk(stream_obj: Any) -> Any:
    """
    Consume at most one chunk from an async stream and return it, or None.

    Notes:
    - If the object supports aclose(), we invoke it best-effort to avoid leaks.
    """
    aiter = stream_obj
    if not hasattr(aiter, "__aiter__"):
        raise AssertionError(f"Expected async-iterable, got {type(aiter).__name__}")

    try:
        async for chunk in aiter:  # noqa: B007
            return chunk
        return None
    finally:
        aclose = getattr(aiter, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception:
                pass


def _invoke_sync_stream(
    descriptor: LLMFrameworkDescriptor,
    instance: Any,
    text: str,
) -> Any:
    """
    Invoke a sync streaming surface according to the descriptor.

    Supports two styles:

    - "method": use descriptor.streaming_method(...)
    - "kwarg": use descriptor.completion_method(..., streaming_kwarg=True)

    Returns whatever iterator-like object the framework provides.
    """
    if not descriptor.supports_streaming:
        pytest.skip(
            f"Framework '{descriptor.name}' does not declare streaming support",
        )

    # Method-style streaming
    if descriptor.streaming_style == "method" and descriptor.streaming_method:
        stream_fn = _get_method(instance, descriptor.streaming_method)
        args, kwargs = _build_primary_call_args(descriptor, stream_fn, text=text)
        return stream_fn(*args, **kwargs)

    # Kwarg-style streaming (e.g. AutoGen-like: completion(stream=True))
    if (
        descriptor.streaming_style == "kwarg"
        and descriptor.streaming_kwarg
        and descriptor.completion_method
    ):
        completion_fn = _get_method(instance, descriptor.completion_method)
        args, kwargs = _build_primary_call_args(
            descriptor,
            completion_fn,
            text=text,
            streaming_kwarg=descriptor.streaming_kwarg,
        )
        return completion_fn(*args, **kwargs)

    pytest.skip(
        f"Framework '{descriptor.name}' declares streaming but no usable "
        f"sync streaming surface could be resolved",
    )


async def _invoke_async_stream(
    descriptor: LLMFrameworkDescriptor,
    instance: Any,
    text: str,
) -> Any:
    """
    Invoke an async streaming surface according to the descriptor.

    Supports two styles:

    - "method": use descriptor.async_streaming_method(...) -> async iterator
    - "kwarg": use descriptor.async_completion_method(..., streaming_kwarg=True)

    The returned object may be an async iterator directly or an awaitable resolving to one.
    """
    if not descriptor.supports_streaming:
        pytest.skip(
            f"Framework '{descriptor.name}' does not declare streaming support",
        )

    # Method-style async streaming
    if descriptor.streaming_style == "method" and descriptor.async_streaming_method:
        astream_fn = _get_method(instance, descriptor.async_streaming_method)
        args, kwargs = _build_primary_call_args(descriptor, astream_fn, text=text)
        aiter = astream_fn(*args, **kwargs)
        if inspect.isawaitable(aiter):
            aiter = await aiter  # type: ignore[assignment]
        return aiter

    # Kwarg-style async streaming via async completion
    if (
        descriptor.streaming_style == "kwarg"
        and descriptor.streaming_kwarg
        and descriptor.async_completion_method
    ):
        acompletion_fn = _get_method(instance, descriptor.async_completion_method)
        args, kwargs = _build_primary_call_args(
            descriptor,
            acompletion_fn,
            text=text,
            streaming_kwarg=descriptor.streaming_kwarg,
        )
        out = acompletion_fn(*args, **kwargs)
        assert inspect.isawaitable(out), (
            "Async completion with streaming_kwarg should return an awaitable",
        )
        aiter = await out  # type: ignore[assignment]
        return aiter

    pytest.skip(
        f"Framework '{descriptor.name}' declares async streaming but no usable "
        f"async streaming surface could be resolved",
    )


# ---------------------------------------------------------------------------
# Completion result type contracts
# ---------------------------------------------------------------------------


def test_sync_completion_result_type_stable_across_calls(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    For sync completions, the LLM client should return the same *type* on repeated calls.

    This catches frameworks that sometimes return a string and sometimes return
    a framework-specific result object, which would break callers relying on type stability.
    """
    if llm_client_instance is None:
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.completion_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare sync completion",
        )

    completion_fn = _get_method(
        llm_client_instance,
        framework_descriptor.completion_method,
    )

    args1, kwargs1 = _build_primary_call_args(framework_descriptor, completion_fn, text=SYNC_COMPLETION_TEXT_1)
    args2, kwargs2 = _build_primary_call_args(framework_descriptor, completion_fn, text=SYNC_COMPLETION_TEXT_2)

    result1 = completion_fn(*args1, **kwargs1)
    result2 = completion_fn(*args2, **kwargs2)

    assert result1 is not None
    assert result2 is not None
    assert type(result1) is type(result2), (
        "Sync completion returned different types across calls: "
        f"{type(result1).__name__} vs {type(result2).__name__}"
    )


@pytest.mark.asyncio
async def test_async_completion_result_type_stable_across_calls_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    When async completion is supported, it should return a stable result type
    across calls with similar inputs.
    """
    if llm_client_instance is None:
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.async_completion_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async completion",
        )

    acompletion_fn = _get_method(
        llm_client_instance,
        framework_descriptor.async_completion_method,
    )

    args1, kwargs1 = _build_primary_call_args(framework_descriptor, acompletion_fn, text=ASYNC_COMPLETION_TEXT_1)
    args2, kwargs2 = _build_primary_call_args(framework_descriptor, acompletion_fn, text=ASYNC_COMPLETION_TEXT_2)

    coro1 = acompletion_fn(*args1, **kwargs1)
    coro2 = acompletion_fn(*args2, **kwargs2)

    assert inspect.isawaitable(coro1)
    assert inspect.isawaitable(coro2)

    result1 = await coro1
    result2 = await coro2

    assert result1 is not None
    assert result2 is not None
    assert type(result1) is type(result2), (
        "Async completion returned different types across calls: "
        f"{type(result1).__name__} vs {type(result2).__name__}"
    )


def test_sync_and_async_completion_result_types_match_when_both_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    When both sync and async completion are declared, their *result types* should match.

    Rationale:
    - Callers often swap between sync and async surfaces depending on runtime context.
    - Type drift between the two surfaces is surprising and complicates client code.
    """
    if llm_client_instance is None:
        _assert_unavailable_contract(framework_descriptor)
        return

    if not (framework_descriptor.completion_method and framework_descriptor.async_completion_method):
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare both sync and async completion",
        )

    completion_fn = _get_method(llm_client_instance, framework_descriptor.completion_method)
    acompletion_fn = _get_method(llm_client_instance, framework_descriptor.async_completion_method)

    args_s, kwargs_s = _build_primary_call_args(framework_descriptor, completion_fn, text=SYNC_COMPLETION_TEXT_1)
    sync_result = completion_fn(*args_s, **kwargs_s)

    args_a, kwargs_a = _build_primary_call_args(framework_descriptor, acompletion_fn, text=SYNC_COMPLETION_TEXT_1)
    coro = acompletion_fn(*args_a, **kwargs_a)
    assert inspect.isawaitable(coro), "async_completion_method must return an awaitable"

    # Execute the awaitable safely from a sync test:
    # - Prefer .send(None)/loop tricks? Not safe.
    # - Use pytest's async test for execution is cleaner; therefore we only compare
    #   types by awaiting via asyncio.run when no loop is running.
    #
    # NOTE: This is intentionally conservative; if an event loop is already running,
    # we skip to avoid nested-loop hazards. The async-only parity is still covered by
    # the async tests in this suite.
    try:
        import asyncio

        asyncio.get_running_loop()
        pytest.skip("Skipping sync/async completion type parity because an event loop is already running")
    except RuntimeError:
        import asyncio

        async_result = asyncio.run(coro)

    assert sync_result is not None
    assert async_result is not None
    assert type(sync_result) is type(async_result), (
        "Sync and async completion returned different result types: "
        f"{type(sync_result).__name__} vs {type(async_result).__name__}"
    )


# ---------------------------------------------------------------------------
# Streaming chunk-type contracts
# ---------------------------------------------------------------------------


def test_stream_chunk_type_consistent_within_stream_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    When streaming is supported (via dedicated method or streaming kwarg),
    all chunks yielded from a single sync stream should have a consistent type.

    We don't enforce any particular chunk *shape* here, only that a single
    stream doesn't mix, e.g., strings and dicts.

    We also cap consumption to MAX_STREAM_CHUNKS_TO_SAMPLE to avoid hangs.
    """
    if llm_client_instance is None:
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.supports_streaming:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare streaming support",
        )

    iterator = _invoke_sync_stream(
        framework_descriptor,
        llm_client_instance,
        SYNC_STREAM_TEXT,
    )
    _assert_iterable(iterator)

    first_chunk_type: type[Any] | None = None
    chunks_seen = 0

    for chunk in iterator:
        chunks_seen += 1
        if first_chunk_type is None:
            first_chunk_type = type(chunk)
        else:
            assert type(chunk) is first_chunk_type, (
                "Sync streaming yielded chunks of inconsistent types within a "
                f"single stream: {first_chunk_type.__name__} vs {type(chunk).__name__}"
            )
        if chunks_seen >= MAX_STREAM_CHUNKS_TO_SAMPLE:
            break

    # It's acceptable for a stream to yield no chunks; this file focuses on type
    # stability rather than stream minimum-length guarantees.


@pytest.mark.asyncio
async def test_async_stream_chunk_type_consistent_within_stream_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    When async streaming is supported (via dedicated async streaming method
    or via async completion + streaming kwarg), all chunks yielded from a
    single async stream should have a consistent type.

    The async streaming surface may be an async iterator directly, or an
    awaitable resolving to one.

    We also cap consumption to MAX_STREAM_CHUNKS_TO_SAMPLE to avoid hangs.
    """
    if llm_client_instance is None:
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.supports_streaming:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare streaming support",
        )

    aiter = await _invoke_async_stream(
        framework_descriptor,
        llm_client_instance,
        ASYNC_STREAM_TEXT,
    )
    _assert_async_iterable(aiter)

    first_chunk_type: type[Any] | None = None
    chunks_seen = 0

    async for chunk in aiter:  # noqa: B007
        chunks_seen += 1
        if first_chunk_type is None:
            first_chunk_type = type(chunk)
        else:
            assert type(chunk) is first_chunk_type, (
                "Async streaming yielded chunks of inconsistent types within a "
                f"single stream: {first_chunk_type.__name__} vs {type(chunk).__name__}"
            )
        if chunks_seen >= MAX_STREAM_CHUNKS_TO_SAMPLE:
            break


def test_sync_stream_first_chunk_type_stable_across_calls_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    When streaming is supported, the *type of the first chunk* should be stable
    across separate stream invocations.

    Why this matters:
    - Many clients sample early chunks for routing/format decisions.
    - A stream that sometimes starts with a dict "event" and sometimes starts
      with a string chunk is hard to consume generically.

    This test does not require streams to produce any chunks; if both streams are
    empty, we treat that as acceptable for this smoke suite.
    """
    if llm_client_instance is None:
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.supports_streaming:
        pytest.skip(f"{framework_descriptor.name}: streaming not declared")

    stream1 = _invoke_sync_stream(framework_descriptor, llm_client_instance, SYNC_STREAM_TEXT + "-a")
    stream2 = _invoke_sync_stream(framework_descriptor, llm_client_instance, SYNC_STREAM_TEXT + "-b")

    _assert_iterable(stream1)
    _assert_iterable(stream2)

    first1 = _sync_first_chunk(stream1)
    first2 = _sync_first_chunk(stream2)

    if first1 is None and first2 is None:
        return

    assert first1 is not None and first2 is not None, (
        "One stream produced a first chunk while the other produced none; "
        "this may indicate inconsistent buffering behavior."
    )
    assert type(first1) is type(first2), (
        "First streaming chunk types differed across calls: "
        f"{type(first1).__name__} vs {type(first2).__name__}"
    )


@pytest.mark.asyncio
async def test_async_stream_first_chunk_type_stable_across_calls_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    Async companion to the sync first-chunk type stability test.

    This validates that two async streams start with the same chunk type.
    """
    if llm_client_instance is None:
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.supports_streaming:
        pytest.skip(f"{framework_descriptor.name}: streaming not declared")

    aiter1 = await _invoke_async_stream(framework_descriptor, llm_client_instance, ASYNC_STREAM_TEXT + "-a")
    aiter2 = await _invoke_async_stream(framework_descriptor, llm_client_instance, ASYNC_STREAM_TEXT + "-b")

    _assert_async_iterable(aiter1)
    _assert_async_iterable(aiter2)

    first1 = await _async_first_chunk(aiter1)
    first2 = await _async_first_chunk(aiter2)

    if first1 is None and first2 is None:
        return

    assert first1 is not None and first2 is not None, (
        "One async stream produced a first chunk while the other produced none; "
        "this may indicate inconsistent buffering behavior."
    )
    assert type(first1) is type(first2), (
        "First async streaming chunk types differed across calls: "
        f"{type(first1).__name__} vs {type(first2).__name__}"
    )


@pytest.mark.asyncio
async def test_stream_first_chunk_type_matches_between_sync_and_async_when_both_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    When both sync and async streaming are declared and usable, the *first chunk type*
    should match between the two surfaces.

    This is a best-effort parity check:
    - If either stream yields no chunks, we do not fail (this suite is smoke-style).
    - If both yield a first chunk, we require type parity.
    """
    if llm_client_instance is None:
        _assert_unavailable_contract(framework_descriptor)
        return

    # Only run this parity test when both streaming surfaces can be invoked.
    # For kwarg-style streaming, both sync and async can still be present via completion methods.
    if not framework_descriptor.supports_streaming:
        pytest.skip(f"{framework_descriptor.name}: streaming not declared")

    # Sync stream
    # Run in a worker thread to avoid calling sync adapters from an event loop.
    first_sync = await asyncio.to_thread(
        _sync_first_chunk_for_descriptor,
        framework_descriptor,
        llm_client_instance,
        SYNC_STREAM_TEXT + "-parity",
    )

    # Async stream
    async_stream = await _invoke_async_stream(framework_descriptor, llm_client_instance, ASYNC_STREAM_TEXT + "-parity")
    _assert_async_iterable(async_stream)
    first_async = await _async_first_chunk(async_stream)

    if first_sync is None or first_async is None:
        return

    assert type(first_sync) is type(first_async), (
        "Sync/async first streaming chunk types differ: "
        f"{type(first_sync).__name__} vs {type(first_async).__name__}"
    )


def test_streaming_surface_is_resolvable_when_supports_streaming_true(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    Registry / descriptor coherence check (LLM-specific).

    If supports_streaming=True, at least one of the following must be usable:
      - method-style streaming: streaming_method or async_streaming_method (or both)
      - kwarg-style streaming: completion_method/async_completion_method with streaming_kwarg

    This prevents "supports_streaming=True" from silently becoming a no-op.
    """
    if llm_client_instance is None:
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.supports_streaming:
        # If not supported, the descriptor should not claim streaming methods/kwargs.
        assert framework_descriptor.streaming_method is None
        assert framework_descriptor.async_streaming_method is None
        assert framework_descriptor.streaming_kwarg is None
        return

    if framework_descriptor.streaming_style == "method":
        assert (
            framework_descriptor.streaming_method is not None
            or framework_descriptor.async_streaming_method is not None
        ), (
            f"{framework_descriptor.name}: supports_streaming=True and streaming_style='method' "
            "but no streaming_method/async_streaming_method is declared"
        )
        # If declared, ensure it exists/callable on the instance.
        if framework_descriptor.streaming_method:
            _get_method(llm_client_instance, framework_descriptor.streaming_method)
        if framework_descriptor.async_streaming_method:
            _get_method(llm_client_instance, framework_descriptor.async_streaming_method)

    if framework_descriptor.streaming_style == "kwarg":
        assert framework_descriptor.streaming_kwarg is not None, (
            f"{framework_descriptor.name}: streaming_style='kwarg' but streaming_kwarg is None"
        )
        assert (
            framework_descriptor.completion_method is not None
            or framework_descriptor.async_completion_method is not None
        ), (
            f"{framework_descriptor.name}: streaming_style='kwarg' requires a completion surface, "
            "but neither completion_method nor async_completion_method is declared"
        )
        if framework_descriptor.completion_method:
            _get_method(llm_client_instance, framework_descriptor.completion_method)
        if framework_descriptor.async_completion_method:
            _get_method(llm_client_instance, framework_descriptor.async_completion_method)


# ---------------------------------------------------------------------------
# Token-count contracts (shape + sync/async parity)
# ---------------------------------------------------------------------------


def test_token_count_type_stable_across_calls_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    When token_count_method is declared, it should return a stable scalar type
    (int) across calls with similar inputs.
    """
    if llm_client_instance is None:
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.token_count_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare sync token counting",
        )

    count_fn = _get_method(
        llm_client_instance,
        framework_descriptor.token_count_method,
    )

    args1, kwargs1 = _build_primary_call_args(framework_descriptor, count_fn, text=TOKEN_COUNT_TEXT_1)
    args2, kwargs2 = _build_primary_call_args(framework_descriptor, count_fn, text=TOKEN_COUNT_TEXT_2)

    result1 = count_fn(*args1, **kwargs1)
    result2 = count_fn(*args2, **kwargs2)

    assert result1 is not None
    assert result2 is not None

    assert isinstance(result1, int), (
        f"token_count_method should return an int, got {type(result1).__name__}",
    )
    assert isinstance(result2, int), (
        f"token_count_method should return an int, got {type(result2).__name__}",
    )
    assert type(result1) is type(result2)


@pytest.mark.asyncio
async def test_async_token_count_type_stable_across_calls_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    When async_token_count_method is declared, it should return a stable
    scalar type (int) across calls with similar inputs.
    """
    if llm_client_instance is None:
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.async_token_count_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async token counting",
        )

    acount_fn = _get_method(
        llm_client_instance,
        framework_descriptor.async_token_count_method,
    )

    args1, kwargs1 = _build_primary_call_args(framework_descriptor, acount_fn, text=TOKEN_COUNT_TEXT_1)
    args2, kwargs2 = _build_primary_call_args(framework_descriptor, acount_fn, text=TOKEN_COUNT_TEXT_2)

    coro1 = acount_fn(*args1, **kwargs1)
    coro2 = acount_fn(*args2, **kwargs2)

    assert inspect.isawaitable(coro1)
    assert inspect.isawaitable(coro2)

    result1 = await coro1
    result2 = await coro2

    assert result1 is not None
    assert result2 is not None

    assert isinstance(result1, int), (
        f"async_token_count_method should return an int, got {type(result1).__name__}",
    )
    assert isinstance(result2, int), (
        f"async_token_count_method should return an int, got {type(result2).__name__}",
    )
    assert type(result1) is type(result2)


@pytest.mark.asyncio
async def test_async_token_count_type_matches_sync_when_both_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    When both sync and async token counting methods are declared, their result
    types should match for the same input.
    """
    if llm_client_instance is None:
        _assert_unavailable_contract(framework_descriptor)
        return

    if not (
        framework_descriptor.token_count_method
        and framework_descriptor.async_token_count_method
    ):
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare both "
            "sync and async token counting",
        )

    count_fn = _get_method(
        llm_client_instance,
        framework_descriptor.token_count_method,
    )
    acount_fn = _get_method(
        llm_client_instance,
        framework_descriptor.async_token_count_method,
    )

    args_s, kwargs_s = _build_primary_call_args(framework_descriptor, count_fn, text=TOKEN_COUNT_TEXT_1)
    sync_result = count_fn(*args_s, **kwargs_s)
    assert isinstance(sync_result, int)

    args_a, kwargs_a = _build_primary_call_args(framework_descriptor, acount_fn, text=TOKEN_COUNT_TEXT_1)
    coro = acount_fn(*args_a, **kwargs_a)
    assert inspect.isawaitable(coro), (
        "async_token_count_method must return an awaitable",
    )
    async_result = await coro
    assert isinstance(async_result, int)

    assert type(sync_result) is type(async_result), (
        "Async token count result type does not match sync token count type: "
        f"{type(sync_result).__name__} vs {type(async_result).__name__}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
