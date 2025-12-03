# tests/frameworks/graph/test_contract_interface_conformance.py

from __future__ import annotations

import asyncio
import importlib
import inspect
from collections.abc import Mapping
from typing import Any, Callable

import pytest

from tests.frameworks.registries.graph_registry import (
    GraphFrameworkDescriptor,
    iter_graph_framework_descriptors,
)

# ---------------------------------------------------------------------------
# Constants (shared test inputs)
# ---------------------------------------------------------------------------

SYNC_QUERY_TEXT = "graph-sync-query"
SYNC_STREAM_TEXT = "graph-sync-stream"
ASYNC_QUERY_TEXT = "graph-async-query"
ASYNC_STREAM_TEXT = "graph-async-stream"
CONTEXT_QUERY_TEXT = "graph-context-query"


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
    underlying LangChain / LlamaIndex / Semantic Kernel libraries are missing)
    are skipped via descriptor.is_available().
    """
    descriptor: GraphFrameworkDescriptor = request.param
    if not descriptor.is_available():
        pytest.skip(f"Framework '{descriptor.name}' not available in this environment")
    return descriptor


@pytest.fixture
def graph_client_instance(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_adapter: Any,
) -> Any:
    """
    Construct a concrete graph client instance for the given descriptor.

    This uses the registry metadata to import the client class and instantiate
    it with the *generic* Corpus graph adapter provided by the top-level pytest
    plugin (see conftest.py).

    The client class is expected to wrap a GraphProtocolV1 implementation.
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    client_cls = getattr(module, framework_descriptor.adapter_class)

    # All graph framework adapters take a graph_adapter implementing the
    # GraphProtocolV1 surface. The global `graph_adapter` fixture is pluggable.
    init_kwargs: dict[str, Any] = {"graph_adapter": graph_adapter}

    # Additional framework-specific kwargs can be added here if needed.

    instance = client_cls(**init_kwargs)
    return instance


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


def _run_async_if_needed(coro: Any) -> Any:
    """
    Run an async coroutine, handling existing event loops gracefully.

    Used for optional async surfaces (e.g. acapabilities/ahealth) in tests
    that are not themselves marked async.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # Fall back to the current event loop if one is already running.
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Core interface / surface contract tests
# ---------------------------------------------------------------------------


def test_can_instantiate_graph_client(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Each registered framework descriptor should be instantiable with the
    pluggable Corpus graph adapter and any inferred kwargs.

    Sanity-check that the instance exposes the methods the descriptor claims.
    """
    # Required sync query method
    _get_method(graph_client_instance, framework_descriptor.query_method)

    # Optional sync streaming
    if framework_descriptor.stream_query_method:
        _get_method(graph_client_instance, framework_descriptor.stream_query_method)

    # Optional bulk / batch methods (when support flags are declared)
    if framework_descriptor.supports_bulk_vertices:
        assert (
            framework_descriptor.bulk_vertices_method is not None
        ), f"{framework_descriptor.name}: supports_bulk_vertices=True but bulk_vertices_method is None"
        _get_method(graph_client_instance, framework_descriptor.bulk_vertices_method)

    if framework_descriptor.supports_batch:
        assert (
            framework_descriptor.batch_method is not None
        ), f"{framework_descriptor.name}: supports_batch=True but batch_method is None"
        _get_method(graph_client_instance, framework_descriptor.batch_method)

    # Async surfaces (if any async declared)
    if framework_descriptor.supports_async:
        # Registry policy: if supports_async=True then async_query_method and
        # async_stream_query_method must both be non-None.
        assert (
            framework_descriptor.async_query_method is not None
        ), f"{framework_descriptor.name}: supports_async=True but async_query_method is None"
        assert (
            framework_descriptor.async_stream_query_method is not None
        ), f"{framework_descriptor.name}: supports_async=True but async_stream_query_method is None"

        _get_method(graph_client_instance, framework_descriptor.async_query_method)
        _get_method(graph_client_instance, framework_descriptor.async_stream_query_method)

        # Optional async bulk/batch surfaces
        if framework_descriptor.async_bulk_vertices_method:
            _get_method(
                graph_client_instance,
                framework_descriptor.async_bulk_vertices_method,
            )

        if framework_descriptor.async_batch_method:
            _get_method(graph_client_instance, framework_descriptor.async_batch_method)


def test_async_methods_exist_when_supports_async_true(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Ensure that when supports_async=True, async query & stream methods exist.

    This mirrors the stricter policy enforced by the registry tests:
    if async support is declared, both async query and async stream surfaces
    must be present and callable.
    """
    if not framework_descriptor.supports_async:
        pytest.skip("Framework does not declare async support")

    # Registry promises these are non-None when supports_async is True
    assert framework_descriptor.async_query_method is not None
    assert framework_descriptor.async_stream_query_method is not None

    aquery = getattr(
        graph_client_instance,
        framework_descriptor.async_query_method,
        None,
    )
    astream = getattr(
        graph_client_instance,
        framework_descriptor.async_stream_query_method,
        None,
    )

    assert callable(aquery), "Async query method is not callable"
    assert callable(astream), "Async stream method is not callable"


def test_sync_query_interface_conformance(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Validate that the sync query method accepts a simple text input and
    returns some non-None result.

    Detailed QueryResult shape is covered by separate shape/batching tests.
    """
    query_fn = _get_method(graph_client_instance, framework_descriptor.query_method)

    if framework_descriptor.context_kwarg:
        result = query_fn(SYNC_QUERY_TEXT, **{framework_descriptor.context_kwarg: {}})
    else:
        result = query_fn(SYNC_QUERY_TEXT)

    assert result is not None


def test_sync_streaming_interface_when_declared(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Validate that the sync streaming method (when declared) accepts a text
    input and returns an iterable of chunks.

    We don't assert detailed chunk shape here; that's covered elsewhere.
    """
    if not framework_descriptor.stream_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare sync streaming",
        )

    stream_fn = _get_method(
        graph_client_instance,
        framework_descriptor.stream_query_method,
    )

    if framework_descriptor.context_kwarg:
        iterator = stream_fn(
            SYNC_STREAM_TEXT,
            **{framework_descriptor.context_kwarg: {}},
        )
    else:
        iterator = stream_fn(SYNC_STREAM_TEXT)

    # At minimum, the returned object must be iterable.
    seen_any = False
    for _ in iterator:  # noqa: B007
        seen_any = True
        break

    # It's fine if no chunks are produced; the contract is about iterability.
    assert iterator is not None
    assert isinstance(seen_any, bool)


@pytest.mark.asyncio
async def test_async_query_interface_conformance_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Validate that the async query method (when declared) accepts text input
    and returns a result compatible with the sync API (non-None).
    """
    if not framework_descriptor.async_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async query",
        )

    aquery_fn = _get_method(
        graph_client_instance,
        framework_descriptor.async_query_method,
    )

    if framework_descriptor.context_kwarg:
        coro = aquery_fn(
            ASYNC_QUERY_TEXT,
            **{framework_descriptor.context_kwarg: {}},
        )
    else:
        coro = aquery_fn(ASYNC_QUERY_TEXT)

    assert inspect.isawaitable(coro), "Async query method must return an awaitable"

    result = await coro
    assert result is not None


@pytest.mark.asyncio
async def test_async_streaming_interface_conformance_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Validate that the async streaming method (when declared) accepts text input
    and produces an async-iterable of chunks.

    The returned object may be an async iterator directly, or an awaitable
    that resolves to one (mirroring the error-context tests).
    """
    if not framework_descriptor.async_stream_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async streaming",
        )

    astream_fn = _get_method(
        graph_client_instance,
        framework_descriptor.async_stream_query_method,
    )

    if framework_descriptor.context_kwarg:
        aiter = astream_fn(
            ASYNC_STREAM_TEXT,
            **{framework_descriptor.context_kwarg: {}},
        )
    else:
        aiter = astream_fn(ASYNC_STREAM_TEXT)

    # Allow both: awaitable -> async iterator, or async iterator directly.
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    # Consume at most one chunk to validate async-iterability.
    seen_any = False
    async for _ in aiter:  # noqa: B007
        seen_any = True
        break

    assert isinstance(seen_any, bool)


def test_context_kwarg_is_accepted_when_declared(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    If a context_kwarg is declared in the descriptor, the corresponding
    query method should accept that kwarg without raising TypeError.
    """
    if not framework_descriptor.context_kwarg:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare a context_kwarg",
        )

    ctx_kw = framework_descriptor.context_kwarg

    query_fn = _get_method(graph_client_instance, framework_descriptor.query_method)

    # Should not raise TypeError
    result = query_fn(CONTEXT_QUERY_TEXT, **{ctx_kw: {"test": "value"}})
    assert result is not None


def test_method_signatures_consistent_between_sync_and_async(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Verify that sync and async methods have consistent signatures
    (same parameters except maybe the return annotation), where both
    variants are declared.

    This covers query, stream, bulk_vertices, and batch surfaces.
    """

    def _compare_signatures(sync_name: str | None, async_name: str | None) -> None:
        if not sync_name or not async_name:
            return

        sync_fn = _get_method(graph_client_instance, sync_name)
        async_fn = _get_method(graph_client_instance, async_name)

        sync_sig = inspect.signature(sync_fn)
        async_sig = inspect.signature(async_fn)

        # Skip "self" for bound methods
        sync_params = list(sync_sig.parameters.keys())[1:]
        async_params = list(async_sig.parameters.keys())[1:]

        assert (
            sync_params == async_params
        ), f"Signature mismatch between {sync_name!r} and {async_name!r}"

    # Query
    _compare_signatures(
        framework_descriptor.query_method,
        framework_descriptor.async_query_method,
    )

    # Streaming
    _compare_signatures(
        framework_descriptor.stream_query_method,
        framework_descriptor.async_stream_query_method,
    )

    # Bulk vertices
    _compare_signatures(
        framework_descriptor.bulk_vertices_method,
        framework_descriptor.async_bulk_vertices_method,
    )

    # Batch
    _compare_signatures(
        framework_descriptor.batch_method,
        framework_descriptor.async_batch_method,
    )


# ---------------------------------------------------------------------------
# Capabilities / health passthrough contract
# ---------------------------------------------------------------------------


def test_capabilities_contract_if_declared(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    If a framework declares has_capabilities=True, it should expose a
    capabilities() method returning a mapping. Async variants (when present)
    should behave similarly.
    """
    if not framework_descriptor.has_capabilities:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not expose capabilities",
        )

    # Sync capabilities
    capabilities = getattr(graph_client_instance, "capabilities", None)
    assert callable(capabilities), "capabilities() method is missing"

    caps_result = capabilities()
    assert isinstance(
        caps_result,
        Mapping,
    ), "capabilities() should return a mapping"

    # Async capabilities (best-effort)
    async_caps = getattr(graph_client_instance, "acapabilities", None)
    if async_caps is not None and callable(async_caps):
        acaps_result = _run_async_if_needed(async_caps())
        assert isinstance(
            acaps_result,
            Mapping,
        ), "acapabilities() should return a mapping"


def test_health_contract_if_declared(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    If a framework declares has_health=True, it should expose a health()
    method returning a mapping. Async variants (when present) should behave
    similarly.
    """
    if not framework_descriptor.has_health:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not expose health",
        )

    # Sync health
    health = getattr(graph_client_instance, "health", None)
    assert callable(health), "health() method is missing"

    health_result = health()
    assert isinstance(
        health_result,
        Mapping,
    ), "health() should return a mapping"

    # Async health (best-effort)
    async_health = getattr(graph_client_instance, "ahealth", None)
    if async_health is not None and callable(async_health):
        ahealth_result = _run_async_if_needed(async_health())
        assert isinstance(
            ahealth_result,
            Mapping,
        ), "ahealth() should return a mapping"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

