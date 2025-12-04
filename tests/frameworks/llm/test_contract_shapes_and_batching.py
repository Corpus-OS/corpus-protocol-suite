# tests/frameworks/llm/test_contract_shapes_and_batching.py

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
    underlying LangChain / LlamaIndex / Semantic Kernel / AutoGen libraries
    are missing) are skipped via descriptor.is_available().
    """
    descriptor: LLMFrameworkDescriptor = request.param
    if not descriptor.is_available():
        pytest.skip(
            f"Framework '{descriptor.name}' not available in this environment",
        )
    return descriptor


@pytest.fixture
def llm_client_instance(
    framework_descriptor: LLMFrameworkDescriptor,
    adapter: Any,
) -> Any:
    """
    Construct a concrete LLM client instance for the given descriptor.

    Mirrors the construction pattern used in the LLM interface-conformance
    tests: each framework adapter is expected to take a `llm_adapter` kwarg
    that wraps a Corpus LLMProtocolV1 implementation.
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    client_cls = getattr(module, framework_descriptor.adapter_class)

    # All LLM framework adapters are expected to accept `llm_adapter=...`.
    init_kwargs: dict[str, Any] = {"llm_adapter": adapter}

    instance = client_cls(**init_kwargs)
    return instance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _maybe_call_with_context(
    descriptor: LLMFrameworkDescriptor,
    fn: Callable[..., Any],
    first_arg: Any,
) -> Any:
    """
    Call an LLM client method, respecting descriptor.context_kwarg if present.

    This helper is used for simple scalar-text completions/streams where the
    adapter surfaces accept something like:

        fn(text, context=...)

    Frameworks that need richer inputs (messages, histories) should adapt their
    completion/streaming surfaces to accept simple text prompts as part of the
    contract enforced by these tests.
    """
    if descriptor.context_kwarg:
        return fn(first_arg, **{descriptor.context_kwarg: {}})
    return fn(first_arg)


def _invoke_sync_stream(
    descriptor: LLMFrameworkDescriptor,
    instance: Any,
    text: str,
) -> Any:
    """
    Invoke a sync streaming surface according to the descriptor.

    Supports two styles:

    - "method": use descriptor.streaming_method(text)
    - "kwarg": use descriptor.completion_method(text, streaming_kwarg=True)

    Returns whatever iterator-like object the framework provides.
    """
    if not descriptor.supports_streaming:
        pytest.skip(
            f"Framework '{descriptor.name}' does not declare streaming support",
        )

    # Method-style streaming (preferred when present)
    if descriptor.streaming_style == "method" and descriptor.streaming_method:
        stream_fn = _get_method(instance, descriptor.streaming_method)
        return _maybe_call_with_context(descriptor, stream_fn, text)

    # Kwarg-style streaming (e.g. AutoGen-like: completion(stream=True))
    if (
        descriptor.streaming_style == "kwarg"
        and descriptor.streaming_kwarg
        and descriptor.completion_method
    ):
        completion_fn = _get_method(instance, descriptor.completion_method)
        kwargs: dict[str, Any] = {descriptor.streaming_kwarg: True}
        if descriptor.context_kwarg:
            kwargs[descriptor.context_kwarg] = {}
        return completion_fn(text, **kwargs)

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

    - "method": use descriptor.async_streaming_method(text) -> async iterator
    - "kwarg": use descriptor.async_completion_method(text, streaming_kwarg=True)
    """
    if not descriptor.supports_streaming:
        pytest.skip(
            f"Framework '{descriptor.name}' does not declare streaming support",
        )

    # Method-style async streaming (preferred when present)
    if descriptor.streaming_style == "method" and descriptor.async_streaming_method:
        astream_fn = _get_method(instance, descriptor.async_streaming_method)
        aiter = _maybe_call_with_context(descriptor, astream_fn, text)

        # allow either async iterator directly or awaitable resolving to one
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
        kwargs: dict[str, Any] = {descriptor.streaming_kwarg: True}
        if descriptor.context_kwarg:
            kwargs[descriptor.context_kwarg] = {}
        aiter = acompletion_fn(text, **kwargs)
        assert inspect.isawaitable(aiter), (
            "Async completion with streaming_kwarg should return an awaitable",
        )
        aiter = await aiter  # type: ignore[assignment]
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
    For simple sync text completions, the LLM client should return the same
    *type* on repeated calls with similar inputs.

    This catches frameworks that sometimes return a string, sometimes a
    framework-specific result object, etc.
    """
    if not framework_descriptor.completion_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare sync completion",
        )

    completion_fn = _get_method(
        llm_client_instance,
        framework_descriptor.completion_method,
    )

    result1 = _maybe_call_with_context(
        framework_descriptor,
        completion_fn,
        SYNC_COMPLETION_TEXT_1,
    )
    result2 = _maybe_call_with_context(
        framework_descriptor,
        completion_fn,
        SYNC_COMPLETION_TEXT_2,
    )

    assert result1 is not None
    assert result2 is not None
    assert type(result1) is type(
        result2,
    ), (
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
    if not framework_descriptor.async_completion_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async completion",
        )

    acompletion_fn = _get_method(
        llm_client_instance,
        framework_descriptor.async_completion_method,
    )

    coro1 = _maybe_call_with_context(
        framework_descriptor,
        acompletion_fn,
        ASYNC_COMPLETION_TEXT_1,
    )
    coro2 = _maybe_call_with_context(
        framework_descriptor,
        acompletion_fn,
        ASYNC_COMPLETION_TEXT_2,
    )

    assert inspect.isawaitable(coro1)
    assert inspect.isawaitable(coro2)

    result1 = await coro1
    result2 = await coro2

    assert result1 is not None
    assert result2 is not None
    assert type(result1) is type(
        result2,
    ), (
        "Async completion returned different types across calls: "
        f"{type(result1).__name__} vs {type(result2).__name__}"
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
    all chunks yielded from a single sync stream should have a consistent
    type.

    We don't enforce any particular chunk *shape* here, only that a single
    stream doesn't mix, e.g., strings and dicts.
    """
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
    for chunk in iterator:
        if first_chunk_type is None:
            first_chunk_type = type(chunk)
        else:
            assert type(chunk) is first_chunk_type, (
                "Sync streaming yielded chunks of inconsistent types within a "
                f"single stream: {first_chunk_type.__name__} vs "
                f"{type(chunk).__name__}"
            )

    # It's acceptable for a stream to yield no chunks at all; the key
    # contract here is *type* consistency, not minimum length.


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
    """
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
    async for chunk in aiter:  # noqa: B007
        if first_chunk_type is None:
            first_chunk_type = type(chunk)
        else:
            assert type(chunk) is first_chunk_type, (
                "Async streaming yielded chunks of inconsistent types within a "
                f"single stream: {first_chunk_type.__name__} vs "
                f"{type(chunk).__name__}"
            )


# ---------------------------------------------------------------------------
# Token-count contracts (shape + sync/async parity)
# ---------------------------------------------------------------------------


def test_token_count_type_stable_across_calls_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    When token_count_method is declared, it should return a stable scalar type
    (typically int) across calls with similar inputs.
    """
    if not framework_descriptor.token_count_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare sync token counting",
        )

    count_fn = _get_method(
        llm_client_instance,
        framework_descriptor.token_count_method,
    )

    result1 = count_fn(TOKEN_COUNT_TEXT_1)
    result2 = count_fn(TOKEN_COUNT_TEXT_2)

    assert result1 is not None
    assert result2 is not None

    # We expect a scalar-ish count, usually an int; don't over-constrain.
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
    scalar type across calls with similar inputs.
    """
    if not framework_descriptor.async_token_count_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async token counting",
        )

    acount_fn = _get_method(
        llm_client_instance,
        framework_descriptor.async_token_count_method,
    )

    coro1 = acount_fn(TOKEN_COUNT_TEXT_1)
    coro2 = acount_fn(TOKEN_COUNT_TEXT_2)

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

    sync_result = count_fn(TOKEN_COUNT_TEXT_1)
    assert isinstance(sync_result, int)

    coro = acount_fn(TOKEN_COUNT_TEXT_1)
    assert inspect.isawaitable(coro), (
        "async_token_count_method must return an awaitable",
    )
    async_result = await coro
    assert isinstance(async_result, int)

    assert type(sync_result) is type(
        async_result,
    ), (
        "Async token count result type does not match sync token count type: "
        f"{type(sync_result).__name__} vs {type(async_result).__name__}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
