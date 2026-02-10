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

# Rich mapping context used across all calls in this file.
#
# Why this exists:
# - Other conformance suites enforce that adapters tolerate "rich mapping context"
#   being passed as regular kwargs (e.g. request_id=..., tags=[...]).
# - This file must not silently "pass" while only testing the narrow context_kwarg
#   path; therefore we always splat these kwargs for every call.
#
# NOTE: Adapters are expected to accept and ignore unknown context keys.
RICH_CONTEXT: dict[str, Any] = {
    "request_id": "req-123",
    "user_id": "user-abc",
    "tags": ["test"],
    "nested": {"depth": 2, "key": "value"},
}

# Performance guardrails:
# - These tests are intentionally lightweight (smoke-style).
# - We cap stream consumption so a buggy stream cannot hang the suite.
MAX_STREAM_CHUNKS_TO_SAMPLE = 10

# Stronger stream validation:
# - To avoid "false coverage" where a stream yields 0 chunks and type checks do nothing,
#   we require at least this many chunks for the within-stream type-consistency tests.
MIN_STREAM_CHUNKS_REQUIRED_FOR_TYPE_TEST = 2


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
    missing) were historically skipped via descriptor.is_available().

    UPDATED STRICT POLICY (ACTIVE TESTING):
    - This suite is used to validate conformance for certification-like workflows.
    - Therefore, registered frameworks must be available in the test environment.
    - If a framework is registered but not available, we fail loudly to prevent
      false confidence and to force the registry/environment to be aligned.
    """
    descriptor: GraphFrameworkDescriptor = request.param
    assert descriptor.is_available(), (
        f"Framework '{descriptor.name}' is registered but not available in this environment. "
        "This suite is configured for active testing: frameworks must be present and testable."
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
    # Defensive import hardening: syntax errors must fail with a clear message.
    # This keeps conformance failures actionable (line number + offending text).
    try:
        module = importlib.import_module(framework_descriptor.adapter_module)
    except SyntaxError as e:
        pytest.fail(
            f"Adapter module failed to import for {framework_descriptor.name!r}: "
            f"SyntaxError at line {e.lineno}: {e.msg}\n"
            f"Text: {e.text!r}",
            pytrace=True,
        )

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


def _context_kwargs_for_descriptor(descriptor: GraphFrameworkDescriptor) -> dict[str, Any]:
    """
    Build kwargs that nest RICH_CONTEXT under the framework-specific context parameter.

    Each framework adapter expects context to be passed via its specific context_kwarg
    (e.g., 'conversation' for AutoGen, 'task' for CrewAI, 'config' for LangChain).
    We nest RICH_CONTEXT under that parameter to avoid TypeError from unexpected kwargs.

    This prevents "false passes" where tests only exercise the happy path.
    """
    kw: dict[str, Any] = {}

    if descriptor.context_kwarg:
        # Nest RICH_CONTEXT under the framework-specific context parameter
        ctx = dict(RICH_CONTEXT)
        
        # Best-effort traceability: include framework name in tags.
        # This is non-fatal if tags cannot be coerced to a list.
        try:
            tags = list(ctx.get("tags", []))
            tags.append(descriptor.name)
            ctx["tags"] = tags
        except Exception:
            pass
        
        kw[descriptor.context_kwarg] = ctx

    return kw


def _maybe_call_with_context(
    descriptor: GraphFrameworkDescriptor,
    fn: Callable[..., Any],
    first_arg: Any,
) -> Any:
    """
    Call a graph client method, applying rich mapping context kwargs and
    descriptor.context_kwarg (if declared).

    This helper works for all single-arg graph surfaces we test here:
    - query(text)
    - stream_query(text)
    - aquery(text)
    - astream_query(text)
    """
    return fn(first_arg, **_context_kwargs_for_descriptor(descriptor))


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
            args={"text": q},
        )
        for q in queries
    ]


def _require_all_surfaces_declared(descriptor: GraphFrameworkDescriptor) -> None:
    """
    Enforce ACTIVE TESTING POLICY: all method surfaces must be enabled/declared.

    Why this exists:
    - The user requirement is to "test everything" and ensure "all methods are set to true".
    - If the registry marks a surface unsupported/absent, we fail here rather than silently
      skipping or returning early.

    This test only checks registry metadata. Method existence/callability is validated
    further below via _get_method(...) in the individual tests.
    """
    # Required query surface must always be present
    assert descriptor.query_method, f"{descriptor.name}: query_method must be declared"

    # Streaming must be declared and supported for active testing
    assert descriptor.stream_query_method, f"{descriptor.name}: stream_query_method must be declared for active testing"

    # Bulk and batch must be declared and supported for active testing
    assert descriptor.supports_bulk_vertices is True, (
        f"{descriptor.name}: supports_bulk_vertices must be True for active testing"
    )
    assert descriptor.bulk_vertices_method, f"{descriptor.name}: bulk_vertices_method must be declared for active testing"

    assert descriptor.supports_batch is True, (
        f"{descriptor.name}: supports_batch must be True for active testing"
    )
    assert descriptor.batch_method, f"{descriptor.name}: batch_method must be declared for active testing"

    # Async must be declared and all async methods must be present
    assert descriptor.supports_async is True, f"{descriptor.name}: supports_async must be True for active testing"
    assert descriptor.async_query_method, f"{descriptor.name}: async_query_method must be declared for active testing"
    assert descriptor.async_stream_query_method, (
        f"{descriptor.name}: async_stream_query_method must be declared for active testing"
    )
    assert descriptor.async_bulk_vertices_method, (
        f"{descriptor.name}: async_bulk_vertices_method must be declared for active testing"
    )
    assert descriptor.async_batch_method, f"{descriptor.name}: async_batch_method must be declared for active testing"


# ---------------------------------------------------------------------------
# Registry enforcement (ensures "all methods true" before exercising shapes)
# ---------------------------------------------------------------------------


def test_registry_declares_all_surfaces_enabled_for_active_testing(
    framework_descriptor: GraphFrameworkDescriptor,
) -> None:
    """
    Ensure the registry is configured for active testing of *all* surfaces.

    This test is intentionally strict and fails when a framework descriptor marks
    a surface unsupported/absent. The goal is to force registry alignment with the
    conformance policy rather than silently reducing coverage.
    """
    _require_all_surfaces_declared(framework_descriptor)


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
    _require_all_surfaces_declared(framework_descriptor)

    query_fn = _get_method(graph_client_instance, framework_descriptor.query_method)

    result1 = _maybe_call_with_context(framework_descriptor, query_fn, QUERY_TEXT_FOR_TYPE)
    result2 = _maybe_call_with_context(framework_descriptor, query_fn, QUERY_TEXT_FOR_TYPE + "-again")

    assert result1 is not None
    assert result2 is not None
    assert type(result1) is type(result2), (
        "Sync query returned different types across calls: "
        f"{type(result1).__name__} vs {type(result2).__name__}"
    )


def test_stream_chunk_type_consistent_within_stream(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    All chunks yielded from a single sync stream should have a consistent type.

    We don't enforce any particular chunk *shape* here (that's adapter-level),
    only that a single stream doesn't mix, e.g., dicts and strings.

    STRICTNESS NOTE:
    - To avoid false coverage, we require the stream to yield at least
      MIN_STREAM_CHUNKS_REQUIRED_FOR_TYPE_TEST chunks for this test.
    """
    _require_all_surfaces_declared(framework_descriptor)

    stream_fn = _get_method(graph_client_instance, framework_descriptor.stream_query_method)

    iterator = _maybe_call_with_context(framework_descriptor, stream_fn, STREAM_TEXT_FOR_TYPE)
    _assert_iterable(iterator)

    first_chunk_type: type[Any] | None = None
    chunks_seen = 0

    for chunk in iterator:
        chunks_seen += 1
        if first_chunk_type is None:
            first_chunk_type = type(chunk)
        else:
            assert type(chunk) is first_chunk_type, (
                "Streaming yielded chunks of inconsistent types within a single stream: "
                f"{first_chunk_type.__name__} vs {type(chunk).__name__}"
            )

        # Performance guardrail: don't consume unbounded streams.
        if chunks_seen >= MAX_STREAM_CHUNKS_TO_SAMPLE:
            break

    assert chunks_seen >= MIN_STREAM_CHUNKS_REQUIRED_FOR_TYPE_TEST, (
        "Stream did not yield enough chunks to validate type consistency. "
        f"Expected >= {MIN_STREAM_CHUNKS_REQUIRED_FOR_TYPE_TEST}, saw {chunks_seen}."
    )


@pytest.mark.asyncio
async def test_async_stream_chunk_type_consistent_within_stream(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    All chunks yielded from a single async stream should have a consistent type.

    The async streaming surface may be an async iterator directly, or an
    awaitable resolving to one.

    STRICTNESS NOTE:
    - To avoid false coverage, we require the stream to yield at least
      MIN_STREAM_CHUNKS_REQUIRED_FOR_TYPE_TEST chunks for this test.
    """
    _require_all_surfaces_declared(framework_descriptor)

    astream_fn = _get_method(graph_client_instance, framework_descriptor.async_stream_query_method)

    aiter = _maybe_call_with_context(framework_descriptor, astream_fn, ASYNC_STREAM_TEXT_FOR_TYPE)

    # Allow both: awaitable -> async iterator, or async iterator directly.
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    _assert_async_iterable(aiter)

    first_chunk_type: type[Any] | None = None
    chunks_seen = 0

    async for chunk in aiter:  # noqa: B007
        chunks_seen += 1
        if first_chunk_type is None:
            first_chunk_type = type(chunk)
        else:
            assert type(chunk) is first_chunk_type, (
                "Async streaming yielded chunks of inconsistent types within a single stream: "
                f"{first_chunk_type.__name__} vs {type(chunk).__name__}"
            )

        # Performance guardrail: don't consume unbounded async streams.
        if chunks_seen >= MAX_STREAM_CHUNKS_TO_SAMPLE:
            break

    assert chunks_seen >= MIN_STREAM_CHUNKS_REQUIRED_FOR_TYPE_TEST, (
        "Async stream did not yield enough chunks to validate type consistency. "
        f"Expected >= {MIN_STREAM_CHUNKS_REQUIRED_FOR_TYPE_TEST}, saw {chunks_seen}."
    )


# ---------------------------------------------------------------------------
# Bulk vertices: basic shape + edge cases + async/sync parity
# ---------------------------------------------------------------------------


def test_bulk_vertices_result_type_stable_across_calls(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Repeated bulk_vertices calls with similar specs should return the same *type*.

    We intentionally don't over-specify the shape of BulkVerticesResult; that
    is an adapter/protocol concern. Here we just enforce type stability.
    """
    _require_all_surfaces_declared(framework_descriptor)

    bulk_fn = _get_method(graph_client_instance, framework_descriptor.bulk_vertices_method)

    spec1 = _build_bulk_spec(namespace=BULK_NAMESPACE_DEFAULT, limit=BULK_LIMIT_SMALL)
    spec2 = _build_bulk_spec(namespace=BULK_NAMESPACE_DEFAULT, limit=BULK_LIMIT_SMALL - 2)

    result1 = bulk_fn(spec1, **_context_kwargs_for_descriptor(framework_descriptor))
    result2 = bulk_fn(spec2, **_context_kwargs_for_descriptor(framework_descriptor))

    assert result1 is not None
    assert result2 is not None
    assert type(result1) is type(result2), (
        "bulk_vertices returned different result types across calls: "
        f"{type(result1).__name__} vs {type(result2).__name__}"
    )


def test_bulk_vertices_limit_zero_is_rejected(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    limit=0 is rejected as invalid input per API contract.

    The protocol requires limit to be positive for bulk operations.
    """
    _require_all_surfaces_declared(framework_descriptor)

    from corpus_sdk.graph.graph_base import BadRequest
    
    bulk_fn = _get_method(graph_client_instance, framework_descriptor.bulk_vertices_method)
    spec = _build_bulk_spec(namespace=BULK_NAMESPACE_DEFAULT, limit=BULK_LIMIT_ZERO)

    with pytest.raises(BadRequest, match="limit must be positive"):
        bulk_fn(spec, **_context_kwargs_for_descriptor(framework_descriptor))


def test_bulk_vertices_with_explicit_namespace_is_accepted(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Using an explicit namespace should be supported wherever bulk_vertices
    is declared. This is important for multi-tenant / multi-dataset setups.
    """
    _require_all_surfaces_declared(framework_descriptor)

    bulk_fn = _get_method(graph_client_instance, framework_descriptor.bulk_vertices_method)
    spec = _build_bulk_spec(namespace=BULK_NAMESPACE_EXPLICIT, limit=BULK_LIMIT_SMALL)

    result = bulk_fn(spec, **_context_kwargs_for_descriptor(framework_descriptor))
    assert result is not None


@pytest.mark.asyncio
async def test_async_bulk_vertices_type_stable_across_calls(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Async bulk_vertices should return consistent types across calls.

    We test async consistency without calling sync methods from async context
    (which would trigger event loop errors in framework adapters).
    """
    _require_all_surfaces_declared(framework_descriptor)

    abulk_fn = _get_method(graph_client_instance, framework_descriptor.async_bulk_vertices_method)

    spec1 = _build_bulk_spec(namespace=BULK_NAMESPACE_DEFAULT, limit=BULK_LIMIT_SMALL)
    spec2 = _build_bulk_spec(namespace=BULK_NAMESPACE_DEFAULT, limit=BULK_LIMIT_SMALL - 2)

    coro1 = abulk_fn(spec1, **_context_kwargs_for_descriptor(framework_descriptor))
    assert inspect.isawaitable(coro1), "Async bulk_vertices method must return an awaitable"
    async_result1 = await coro1
    assert async_result1 is not None

    coro2 = abulk_fn(spec2, **_context_kwargs_for_descriptor(framework_descriptor))
    async_result2 = await coro2
    assert async_result2 is not None

    assert type(async_result1) is type(async_result2), (
        "Async bulk_vertices returned different result types across calls: "
        f"{type(async_result1).__name__} vs {type(async_result2).__name__}"
    )


# ---------------------------------------------------------------------------
# Batch: length + type contracts + edge cases
# ---------------------------------------------------------------------------


def test_batch_result_length_matches_ops_when_sized(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    BatchResult.results list length should match the number of operations.

    Each BatchOperation should produce exactly one result entry in the
    BatchResult.results list, maintaining 1:1 correspondence.
    """
    _require_all_surfaces_declared(framework_descriptor)

    batch_fn = _get_method(graph_client_instance, framework_descriptor.batch_method)
    ops = _build_batch_ops(BATCH_QUERIES_DEFAULT)

    result = batch_fn(ops, **_context_kwargs_for_descriptor(framework_descriptor))
    assert result is not None
    
    # BatchResult has a .results attribute that should be a list
    assert hasattr(result, "results"), "BatchResult must have a 'results' attribute"
    assert isinstance(result.results, list), "BatchResult.results must be a list"
    assert len(result.results) == len(ops), (
        f"BatchResult.results length ({len(result.results)}) does not match "
        f"number of operations ({len(ops)})"
    )


def test_empty_batch_is_rejected(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Empty batch operations are rejected per API validation.

    The protocol requires at least one operation in a batch.
    """
    _require_all_surfaces_declared(framework_descriptor)

    batch_fn = _get_method(graph_client_instance, framework_descriptor.batch_method)
    ops: list[BatchOperation] = []

    with pytest.raises(ValueError, match="batch ops must not be empty"):
        batch_fn(ops, **_context_kwargs_for_descriptor(framework_descriptor))


def test_batch_result_type_stable_across_calls(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Repeated batch() calls should return the same result *type* for similar inputs.
    """
    _require_all_surfaces_declared(framework_descriptor)

    batch_fn = _get_method(graph_client_instance, framework_descriptor.batch_method)

    ops1 = _build_batch_ops([BATCH_QUERIES_DEFAULT[0]])
    ops2 = _build_batch_ops([BATCH_QUERIES_ALT[0]])

    result1 = batch_fn(ops1, **_context_kwargs_for_descriptor(framework_descriptor))
    result2 = batch_fn(ops2, **_context_kwargs_for_descriptor(framework_descriptor))

    assert result1 is not None
    assert result2 is not None
    assert type(result1) is type(result2), (
        "batch() returned different result types across calls: "
        f"{type(result1).__name__} vs {type(result2).__name__}"
    )


@pytest.mark.asyncio
async def test_async_batch_type_stable_across_calls(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Async batch should return consistent types across calls.

    We test async consistency without calling sync methods from async context
    (which would trigger event loop errors in framework adapters).
    """
    _require_all_surfaces_declared(framework_descriptor)

    abatch_fn = _get_method(graph_client_instance, framework_descriptor.async_batch_method)

    ops1 = _build_batch_ops(BATCH_QUERIES_DEFAULT[:2])
    ops2 = _build_batch_ops(BATCH_QUERIES_ALT[:2])

    coro1 = abatch_fn(ops1, **_context_kwargs_for_descriptor(framework_descriptor))
    assert inspect.isawaitable(coro1), "Async batch method must return an awaitable"
    async_result1 = await coro1
    assert async_result1 is not None

    coro2 = abatch_fn(ops2, **_context_kwargs_for_descriptor(framework_descriptor))
    async_result2 = await coro2
    assert async_result2 is not None

    assert type(async_result1) is type(async_result2), (
        "Async batch returned different result types across calls: "
        f"{type(async_result1).__name__} vs {type(async_result2).__name__}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
