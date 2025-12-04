# tests/frameworks/vector/test_contract_shapes_and_batching.py

from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable

import pytest

from tests.frameworks.registries.vector_registry import (
    VectorFrameworkDescriptor,
    iter_vector_framework_descriptors,
)


# ---------------------------------------------------------------------------
# Constants (shared test inputs)
# ---------------------------------------------------------------------------

ADD_SINGLE_TEXT = ["vec-add-single"]
ADD_MULTI_TEXTS = [f"vec-add-{i}" for i in range(5)]

QUERY_TEXT_FOR_TYPE = "vec-query-type-test"
QUERY_TEXT_FOR_ASYNC = "vec-query-async-test"

STREAM_TEXT_FOR_TYPE = "vec-stream-type-test"
ASYNC_STREAM_TEXT_FOR_TYPE = "vec-stream-async-type-test"

MMR_QUERY_TEXT = "vec-mmr-query-test"
MMR_QUERY_TEXT_ALT = "vec-mmr-query-test-alt"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=list(iter_vector_framework_descriptors()),
    name="framework_descriptor",
)
def framework_descriptor_fixture(
    request: pytest.FixtureRequest,
) -> VectorFrameworkDescriptor:
    """
    Parameterized over all registered vector framework descriptors.

    Frameworks that are not actually available in the environment (e.g. the
    underlying LlamaIndex / Semantic Kernel libraries are missing) are skipped
    via descriptor.is_available().
    """
    descriptor: VectorFrameworkDescriptor = request.param
    if not descriptor.is_available():
        pytest.skip(
            f"Framework '{descriptor.name}' not available in this environment",
        )
    return descriptor


@pytest.fixture
def vector_client_instance(
    framework_descriptor: VectorFrameworkDescriptor,
    adapter: Any,
) -> Any:
    """
    Construct a concrete vector client/store instance for the given descriptor.

    Mirrors the construction pattern used in the other framework contract tests:
    each vector framework adapter is expected to take an `adapter` kwarg that
    wraps a Corpus vector protocol implementation provided by the top-level
    pytest plugin (see conftest.py).
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    client_cls = getattr(module, framework_descriptor.adapter_class)

    init_kwargs: dict[str, Any] = {"adapter": adapter}
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
    descriptor: VectorFrameworkDescriptor,
    fn: Callable[..., Any],
    *args: Any,
    **base_kwargs: Any,
) -> Any:
    """
    Call a vector client method, respecting descriptor.context_kwarg if present.

    This helper works for all single-argument vector surfaces we test here:
    - add(texts: list[str])
    - query(text: str)
    - stream_query(text: str)
    - mmr_query(text: str)
    and their async counterparts (where the first arg is the primary payload).
    """
    kwargs = dict(base_kwargs)
    if descriptor.context_kwarg:
        kwargs.setdefault(descriptor.context_kwarg, {})
    return fn(*args, **kwargs)


def _type_stability_assert(result1: Any, result2: Any, label: str) -> None:
    """
    Helper to assert type stability across two results, when both are non-None.
    """
    assert result1 is not None
    assert result2 is not None
    assert type(result1) is type(
        result2,
    ), (
        f"{label} returned different result types across calls: "
        f"{type(result1).__name__} vs {type(result2).__name__}"
    )


# ---------------------------------------------------------------------------
# Add / upsert contracts
# ---------------------------------------------------------------------------


def test_add_accepts_single_and_multiple_items(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    The add/upsert surface should accept both single-item and multi-item
    batches without raising.

    We intentionally do *not* over-specify the return shape, since vector
    frameworks vary widely here (None, list of IDs, status objects, etc.).
    """
    add_fn = _get_method(vector_client_instance, framework_descriptor.add_method)

    # Single-item batch
    _maybe_call_with_context(
        framework_descriptor,
        add_fn,
        ADD_SINGLE_TEXT,
    )

    # Multi-item batch
    _maybe_call_with_context(
        framework_descriptor,
        add_fn,
        ADD_MULTI_TEXTS,
    )


@pytest.mark.asyncio
async def test_async_add_does_not_crash_and_is_awaitable_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    When async add is declared, it should be callable, return an awaitable,
    and not raise for simple single/multi-item batches.

    We again do not assert specific return shapes: only awaitability +
    non-crashing behavior.
    """
    if not framework_descriptor.async_add_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async add",
        )

    aadd_fn = _get_method(
        vector_client_instance,
        framework_descriptor.async_add_method,
    )

    # Single-item batch
    coro1 = _maybe_call_with_context(
        framework_descriptor,
        aadd_fn,
        ADD_SINGLE_TEXT,
    )
    assert inspect.isawaitable(coro1), "Async add must return an awaitable"
    await coro1  # noqa: PT018

    # Multi-item batch
    coro2 = _maybe_call_with_context(
        framework_descriptor,
        aadd_fn,
        ADD_MULTI_TEXTS,
    )
    assert inspect.isawaitable(coro2), "Async add must return an awaitable"
    await coro2  # noqa: PT018


# ---------------------------------------------------------------------------
# Query / type stability + async parity
# ---------------------------------------------------------------------------


def test_query_result_type_stable_across_calls(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    For simple similarity queries, the vector client should return the same
    *type* on repeated calls with similar inputs.

    This catches frameworks that sometimes return, e.g., a list and sometimes
    a dict, which would break callers relying on type stability.
    """
    query_fn = _get_method(
        vector_client_instance,
        framework_descriptor.query_method,
    )

    result1 = _maybe_call_with_context(
        framework_descriptor,
        query_fn,
        QUERY_TEXT_FOR_TYPE,
    )
    result2 = _maybe_call_with_context(
        framework_descriptor,
        query_fn,
        QUERY_TEXT_FOR_TYPE + "-again",
    )

    _type_stability_assert(result1, result2, label="query()")


@pytest.mark.asyncio
async def test_async_query_type_matches_sync_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    When async query is declared, the async result type should match the sync
    result type for a similar input.
    """
    if not framework_descriptor.async_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async query",
        )

    query_fn = _get_method(
        vector_client_instance,
        framework_descriptor.query_method,
    )
    aquery_fn = _get_method(
        vector_client_instance,
        framework_descriptor.async_query_method,
    )

    sync_result = _maybe_call_with_context(
        framework_descriptor,
        query_fn,
        QUERY_TEXT_FOR_ASYNC,
    )

    coro = _maybe_call_with_context(
        framework_descriptor,
        aquery_fn,
        QUERY_TEXT_FOR_ASYNC,
    )
    assert inspect.isawaitable(coro), "Async query must return an awaitable"

    async_result = await coro
    _type_stability_assert(sync_result, async_result, label="async vs sync query")


# ---------------------------------------------------------------------------
# Streaming contracts (sync + async)
# ---------------------------------------------------------------------------


def test_stream_chunk_type_consistent_within_stream_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    When sync streaming is declared, all chunks yielded from a single stream
    should have a consistent type.

    We don't enforce any particular chunk *shape* here, only that a single
    stream doesn't mix, e.g., dicts and strings.
    """
    if not framework_descriptor.stream_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare sync streaming",
        )

    stream_fn = _get_method(
        vector_client_instance,
        framework_descriptor.stream_query_method,
    )

    iterator = _maybe_call_with_context(
        framework_descriptor,
        stream_fn,
        STREAM_TEXT_FOR_TYPE,
    )

    _assert_iterable(iterator)

    first_chunk_type: type[Any] | None = None
    for chunk in iterator:
        if first_chunk_type is None:
            first_chunk_type = type(chunk)
        else:
            assert type(chunk) is first_chunk_type, (
                "Streaming yielded chunks of inconsistent types within a single stream: "
                f"{first_chunk_type.__name__} vs {type(chunk).__name__}"
            )

    # It's acceptable for a stream to yield no chunks at all; the key
    # contract here is type consistency, not minimum length.


@pytest.mark.asyncio
async def test_async_stream_chunk_type_consistent_within_stream_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    When async streaming is declared, all chunks yielded from a single async
    stream should have a consistent type.

    The async streaming surface may be an async iterator directly, or an
    awaitable resolving to one.
    """
    if not framework_descriptor.async_stream_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async streaming",
        )

    astream_fn = _get_method(
        vector_client_instance,
        framework_descriptor.async_stream_query_method,
    )

    aiter = _maybe_call_with_context(
        framework_descriptor,
        astream_fn,
        ASYNC_STREAM_TEXT_FOR_TYPE,
    )

    # Allow both: awaitable -> async iterator, or async iterator directly.
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    _assert_async_iterable(aiter)

    first_chunk_type: type[Any] | None = None
    async for chunk in aiter:  # noqa: B007
        if first_chunk_type is None:
            first_chunk_type = type(chunk)
        else:
            assert type(chunk) is first_chunk_type, (
                "Async streaming yielded chunks of inconsistent types within a single stream: "
                f"{first_chunk_type.__name__} vs {type(chunk).__name__}"
            )


@pytest.mark.asyncio
async def test_async_stream_result_type_compatible_with_sync_when_both_declared(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    When both sync and async streaming are declared, the chunk type from async
    streaming should match the chunk type from sync streaming.

    We only compare the first yielded chunk from each to keep the test cheap.
    """
    if not framework_descriptor.stream_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare sync streaming",
        )

    if not framework_descriptor.async_stream_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async streaming",
        )

    stream_fn = _get_method(
        vector_client_instance,
        framework_descriptor.stream_query_method,
    )
    astream_fn = _get_method(
        vector_client_instance,
        framework_descriptor.async_stream_query_method,
    )

    # Sync stream: grab first chunk type (if any)
    sync_iter = _maybe_call_with_context(
        framework_descriptor,
        stream_fn,
        STREAM_TEXT_FOR_TYPE,
    )
    _assert_iterable(sync_iter)

    sync_first_type: type[Any] | None = None
    for chunk in sync_iter:
        sync_first_type = type(chunk)
        break

    # Async stream: same
    aiter = _maybe_call_with_context(
        framework_descriptor,
        astream_fn,
        ASYNC_STREAM_TEXT_FOR_TYPE,
    )

    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    _assert_async_iterable(aiter)

    async_first_type: type[Any] | None = None
    async for chunk in aiter:  # noqa: B007
        async_first_type = type(chunk)
        break

    # If either stream produced no chunks, we can't compare types meaningfully;
    # that's fine and not considered a failure.
    if sync_first_type is not None and async_first_type is not None:
        assert (
            sync_first_type is async_first_type
        ), (
            "Async streaming chunk type does not match sync streaming chunk type: "
            f"{sync_first_type.__name__} vs {async_first_type.__name__}"
        )


# ---------------------------------------------------------------------------
# MMR query contracts
# ---------------------------------------------------------------------------


def test_mmr_query_type_matches_normal_query_when_supported(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    When MMR is supported, the MMR query surface should return a result type
    compatible with the normal query surface for similar inputs.
    """
    if not framework_descriptor.supports_mmr or not framework_descriptor.mmr_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare MMR support",
        )

    query_fn = _get_method(
        vector_client_instance,
        framework_descriptor.query_method,
    )
    mmr_fn = _get_method(
        vector_client_instance,
        framework_descriptor.mmr_query_method,
    )

    query_result = _maybe_call_with_context(
        framework_descriptor,
        query_fn,
        MMR_QUERY_TEXT,
    )
    mmr_result = _maybe_call_with_context(
        framework_descriptor,
        mmr_fn,
        MMR_QUERY_TEXT,
    )

    _type_stability_assert(query_result, mmr_result, label="mmr vs query")


def test_mmr_query_result_type_stable_across_calls_when_supported(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    When MMR is supported, repeated calls with similar inputs should return a
    stable result type.
    """
    if not framework_descriptor.supports_mmr or not framework_descriptor.mmr_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare MMR support",
        )

    mmr_fn = _get_method(
        vector_client_instance,
        framework_descriptor.mmr_query_method,
    )

    result1 = _maybe_call_with_context(
        framework_descriptor,
        mmr_fn,
        MMR_QUERY_TEXT,
    )
    result2 = _maybe_call_with_context(
        framework_descriptor,
        mmr_fn,
        MMR_QUERY_TEXT_ALT,
    )

    _type_stability_assert(result1, result2, label="mmr_query()")


@pytest.mark.asyncio
async def test_async_mmr_type_matches_sync_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    When async MMR is declared, the async result type should match the sync
    result type for a similar input.
    """
    if not framework_descriptor.supports_mmr:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare MMR support",
        )

    if not framework_descriptor.mmr_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare sync MMR method",
        )

    if not framework_descriptor.async_mmr_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async MMR method",
        )

    mmr_fn = _get_method(
        vector_client_instance,
        framework_descriptor.mmr_query_method,
    )
    ammr_fn = _get_method(
        vector_client_instance,
        framework_descriptor.async_mmr_query_method,
    )

    sync_result = _maybe_call_with_context(
        framework_descriptor,
        mmr_fn,
        MMR_QUERY_TEXT,
    )

    coro = _maybe_call_with_context(
        framework_descriptor,
        ammr_fn,
        MMR_QUERY_TEXT,
    )
    assert inspect.isawaitable(coro), "Async MMR method must return an awaitable"

    async_result = await coro
    _type_stability_assert(sync_result, async_result, label="async vs sync MMR")


# ---------------------------------------------------------------------------
# Context kwarg minimal shape sanity
# ---------------------------------------------------------------------------


def test_context_kwarg_does_not_change_query_result_type_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    If a context_kwarg is declared, passing it should not change the *type*
    of the query result.

    This is a light sanity check; deeper context semantics are covered in
    dedicated context/error-context tests.
    """
    if not framework_descriptor.context_kwarg:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare a context_kwarg",
        )

    query_fn = _get_method(
        vector_client_instance,
        framework_descriptor.query_method,
    )

    # Without context
    result_no_ctx = query_fn(QUERY_TEXT_FOR_TYPE)

    # With context
    result_with_ctx = query_fn(
        QUERY_TEXT_FOR_TYPE,
        **{framework_descriptor.context_kwarg: {"test": "value"}},
    )

    _type_stability_assert(result_no_ctx, result_with_ctx, label="query with/without context")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
