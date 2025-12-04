# tests/frameworks/graph/test_contract_shapes_and_batching.py

from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable

import pytest

from corpus_sdk.graph.graph_base import BulkVerticesSpec, BatchOperation
from tests.frameworks.registries.graph_registry import (
    GraphFrameworkDescriptor,
    iter_graph_framework_descriptors,
)


# ---------------------------------------------------------------------------
# Constants (shared test inputs)
# ---------------------------------------------------------------------------

QUERY_TEXT_FOR_TYPE = "graph-shape-type-test"
STREAM_TEXT_FOR_TYPE = "graph-stream-type-test"
ASYNC_QUERY_TEXT_FOR_TYPE = "graph-shape-async-query-test"
ASYNC_STREAM_TEXT_FOR_TYPE = "graph-shape-async-stream-test"

BULK_NAMESPACE_DEFAULT: str | None = None
BULK_NAMESPACE_EXPLICIT = "tenant-A"
BULK_LIMIT_SMALL = 5
BULK_LIMIT_ZERO = 0

BATCH_QUERIES_DEFAULT = [
    "batch-op-alpha",
    "batch-op-beta",
    "batch-op-gamma",
]
BATCH_QUERIES_ALT = [
    "batch-op-delta",
    "batch-op-epsilon",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=list(iter_graph_framework_descriptors()),
    name="framework_descriptor",
)
def framework_descriptor_fixture(
    request: pytest.FixtureRequest,
) -> GraphFrameworkDescriptor:
    """
    Parameterized over all registered graph framework descriptors.

    Frameworks that are not actually available in the environment (e.g. the
    underlying LangChain / LlamaIndex / Semantic Kernel / etc. libraries are
    missing) are skipped via descriptor.is_available().
    """
    descriptor: GraphFrameworkDescriptor = request.param
    if not descriptor.is_available():
        pytest.skip(
            f"Framework '{descriptor.name}' not available in this environment",
        )
    return descriptor


@pytest.fixture
def graph_client_instance(
    framework_descriptor: GraphFrameworkDescriptor,
    adapter: Any,
) -> Any:
    """
    Construct a concrete graph client instance for the given descriptor.

    Mirrors the construction pattern used in the interface-conformance tests:
    each framework adapter is expected to take a `adapter` kwarg that
    wraps a Corpus GraphProtocolV1 implementation.
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
    descriptor: GraphFrameworkDescriptor,
    fn: Callable[..., Any],
    first_arg: Any,
) -> Any:
    """
    Call a graph client method, respecting descriptor.context_kwarg if present.

    This helper works for all single-arg graph surfaces we test here:
    - query(text)
    - stream_query(text)
    - aquery(text)
    - astream_query(text)
    """
    if descriptor.context_kwarg:
        return fn(first_arg, **{descriptor.context_kwarg: {}})
    return fn(first_arg)


def _build_bulk_spec(namespace: str | None, limit: int) -> BulkVerticesSpec:
    """
    Helper to construct a BulkVerticesSpec with explicit cursor/filter.

    Passing cursor=None and filter=None is safe even when the underlying
    dataclass fields are optional. This keeps the test agnostic to any
    defaulting behavior inside adapters.
    """
    return BulkVerticesSpec(
        namespace=namespace,
        limit=limit,
        cursor=None,
        filter=None,
    )


def _build_batch_ops(queries: list[str]) -> list[BatchOperation]:
    """
    Helper to construct a list of BatchOperation(query=...) entries from
    simple query strings.

    Centralizing this keeps batch test data consistent and easy to tweak.
    """
    return [
        BatchOperation(
            op="query",
            args={"query": q},
        )
        for q in queries
    ]


# ---------------------------------------------------------------------------
# Query / stream shape + type contracts
# ---------------------------------------------------------------------------


def test_query_result_type_stable_across_calls(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    For simple text queries, the graph client should return the same *type*
    on repeated calls with similar inputs.

    This catches frameworks that sometimes return a dict and sometimes a
    custom result object or list, which would break callers relying on type
    stability.
    """
    query_fn = _get_method(
        graph_client_instance,
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

    assert result1 is not None
    assert result2 is not None
    assert type(result1) is type(
        result2,
    ), (
        "Sync query returned different types across calls: "
        f"{type(result1).__name__} vs {type(result2).__name__}"
    )


def test_stream_chunk_type_consistent_within_stream_when_declared(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    When sync streaming is declared, all chunks yielded from a single stream
    should have a consistent type.

    We don't enforce any particular chunk *shape* here (that's adapter-level),
    only that a single stream doesn't mix, e.g., dicts and strings.
    """
    if not framework_descriptor.stream_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare sync streaming",
        )

    stream_fn = _get_method(
        graph_client_instance,
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
    # contract here is *type* consistency, not minimum length.


@pytest.mark.asyncio
async def test_async_stream_chunk_type_consistent_within_stream_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
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
        graph_client_instance,
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


# ---------------------------------------------------------------------------
# Bulk vertices: basic shape + edge cases + async/sync parity
# ---------------------------------------------------------------------------


def test_bulk_vertices_result_type_stable_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    When bulk_vertices is supported, repeated calls with similar specs should
    return the same *type*.

    We intentionally don't over-specify the shape of BulkVerticesResult; that
    is an adapter/protocol concern. Here we just enforce type stability.
    """
    if not framework_descriptor.supports_bulk_vertices:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare bulk_vertices support",
        )

    assert framework_descriptor.bulk_vertices_method is not None

    bulk_fn = _get_method(
        graph_client_instance,
        framework_descriptor.bulk_vertices_method,
    )

    spec1 = _build_bulk_spec(namespace=BULK_NAMESPACE_DEFAULT, limit=BULK_LIMIT_SMALL)
    spec2 = _build_bulk_spec(namespace=BULK_NAMESPACE_DEFAULT, limit=BULK_LIMIT_SMALL - 2)

    result1 = bulk_fn(spec1)
    result2 = bulk_fn(spec2)

    assert result1 is not None
    assert result2 is not None
    assert type(result1) is type(
        result2,
    ), (
        "bulk_vertices returned different result types across calls: "
        f"{type(result1).__name__} vs {type(result2).__name__}"
    )


def test_bulk_vertices_limit_zero_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    limit=0 is a valid edge case and should not cause errors.

    We don't assert specific result semantics here (empty vs non-empty),
    just that the call is accepted and returns a value.
    """
    if not framework_descriptor.supports_bulk_vertices:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare bulk_vertices support",
        )

    assert framework_descriptor.bulk_vertices_method is not None

    bulk_fn = _get_method(
        graph_client_instance,
        framework_descriptor.bulk_vertices_method,
    )

    spec = _build_bulk_spec(namespace=BULK_NAMESPACE_DEFAULT, limit=BULK_LIMIT_ZERO)
    result = bulk_fn(spec)
    assert result is not None


def test_bulk_vertices_with_explicit_namespace_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Using an explicit namespace should be supported wherever bulk_vertices
    is declared. This is important for multi-tenant / multi-dataset setups.
    """
    if not framework_descriptor.supports_bulk_vertices:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare bulk_vertices support",
        )

    assert framework_descriptor.bulk_vertices_method is not None

    bulk_fn = _get_method(
        graph_client_instance,
        framework_descriptor.bulk_vertices_method,
    )

    spec = _build_bulk_spec(namespace=BULK_NAMESPACE_EXPLICIT, limit=BULK_LIMIT_SMALL)
    result = bulk_fn(spec)
    assert result is not None


@pytest.mark.asyncio
async def test_async_bulk_vertices_type_matches_sync_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    When async bulk_vertices is supported, the async result type should match
    the sync result type for the same spec.

    This ensures callers can switch between sync/async without having to
    special-case result handling.
    """
    if not framework_descriptor.supports_bulk_vertices:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare bulk_vertices support",
        )

    if not framework_descriptor.async_bulk_vertices_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async bulk_vertices",
        )

    assert framework_descriptor.bulk_vertices_method is not None

    bulk_fn = _get_method(
        graph_client_instance,
        framework_descriptor.bulk_vertices_method,
    )
    abulk_fn = _get_method(
        graph_client_instance,
        framework_descriptor.async_bulk_vertices_method,
    )

    spec = _build_bulk_spec(namespace=BULK_NAMESPACE_DEFAULT, limit=BULK_LIMIT_SMALL)

    sync_result = bulk_fn(spec)
    assert sync_result is not None

    coro = abulk_fn(spec)
    assert inspect.isawaitable(coro), (
        "Async bulk_vertices method must return an awaitable",
    )

    async_result = await coro
    assert async_result is not None

    assert type(async_result) is type(
        sync_result,
    ), (
        "Async bulk_vertices result type does not match sync result type: "
        f"{type(sync_result).__name__} vs {type(async_result).__name__}"
    )


# ---------------------------------------------------------------------------
# Batch: length + type contracts + edge cases
# ---------------------------------------------------------------------------


def test_batch_result_length_matches_ops_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    When batch operations are supported, the BatchResult (when it is
    sequence-like) should have length equal to the number of BatchOperation
    items passed in.

    We only enforce this when the result exposes __len__, so adapters are
    free to return non-sequence result types if desired.
    """
    if not framework_descriptor.supports_batch:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare batch support",
        )

    assert framework_descriptor.batch_method is not None

    batch_fn = _get_method(
        graph_client_instance,
        framework_descriptor.batch_method,
    )

    ops = _build_batch_ops(BATCH_QUERIES_DEFAULT)

    result = batch_fn(ops)
    assert result is not None

    if hasattr(result, "__len__"):
        assert len(result) == len(
            ops,
        ), "BatchResult length does not match number of operations"


def test_empty_batch_handling_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Batch methods should gracefully handle an empty list of operations.

    We don't require any particular result shape here, but if the result is
    sequence-like, we expect it to have length 0.
    """
    if not framework_descriptor.supports_batch:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare batch support",
        )

    assert framework_descriptor.batch_method is not None

    batch_fn = _get_method(
        graph_client_instance,
        framework_descriptor.batch_method,
    )

    ops: list[BatchOperation] = []

    result = batch_fn(ops)
    assert result is not None

    if hasattr(result, "__len__"):
        assert len(result) == 0, "BatchResult for empty ops list should be length 0"


def test_batch_result_type_stable_across_calls_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    When batch operations are supported, repeated calls should return the same
    result *type* for similar inputs.
    """
    if not framework_descriptor.supports_batch:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare batch support",
        )

    assert framework_descriptor.batch_method is not None

    batch_fn = _get_method(
        graph_client_instance,
        framework_descriptor.batch_method,
    )

    ops1 = _build_batch_ops([BATCH_QUERIES_DEFAULT[0]])
    ops2 = _build_batch_ops([BATCH_QUERIES_ALT[0]])

    result1 = batch_fn(ops1)
    result2 = batch_fn(ops2)

    assert result1 is not None
    assert result2 is not None
    assert type(result1) is type(
        result2,
    ), (
        "batch() returned different result types across calls: "
        f"{type(result1).__name__} vs {type(result2).__name__}"
    )


@pytest.mark.asyncio
async def test_async_batch_type_matches_sync_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    When async batch is supported, the async result type should match the
    sync result type for the same operations list.
    """
    if not framework_descriptor.supports_batch:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare batch support",
        )

    if not framework_descriptor.async_batch_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async batch",
        )

    assert framework_descriptor.batch_method is not None

    batch_fn = _get_method(
        graph_client_instance,
        framework_descriptor.batch_method,
    )
    abatch_fn = _get_method(
        graph_client_instance,
        framework_descriptor.async_batch_method,
    )

    ops = _build_batch_ops(BATCH_QUERIES_DEFAULT[:2])

    sync_result = batch_fn(ops)
    assert sync_result is not None

    coro = abatch_fn(ops)
    assert inspect.isawaitable(coro), "Async batch method must return an awaitable"

    async_result = await coro
    assert async_result is not None

    assert type(async_result) is type(
        sync_result,
    ), (
        "Async batch result type does not match sync batch result type: "
        f"{type(sync_result).__name__} vs {type(async_result).__name__}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
