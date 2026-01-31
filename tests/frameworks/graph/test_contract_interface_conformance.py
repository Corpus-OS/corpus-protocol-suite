# tests/frameworks/graph/test_contract_interface_conformance.py

from __future__ import annotations

import importlib
import inspect
from collections.abc import AsyncIterable, Iterable, Mapping
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

# A "rich mapping context" that will be splatted as kwargs (mirrors other suites)
RICH_CONTEXT: dict[str, Any] = {
    "request_id": "req-123",
    "user_id": "user-abc",
    "tags": ["test"],
    "nested": {"depth": 2, "key": "value"},
}


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

    IMPORTANT POLICY ALIGNMENT (NO SKIPS):
    - We do not skip unavailable frameworks.
    - Tests must pass by asserting correct "unavailable" signaling when a framework
      is not installed, and must fully run when it is available.
    """
    descriptor: GraphFrameworkDescriptor = request.param
    return descriptor


@pytest.fixture
def graph_client_instance(
    framework_descriptor: GraphFrameworkDescriptor,
    adapter: Any,
) -> Any:
    """
    Construct a concrete graph client instance for the given descriptor.

    Availability contract:
    - If a framework is unavailable, this fixture returns None and tests must
      treat that as a validated pass condition (not a skip).
    - If a framework is available but instantiation fails, tests should fail
      (real regression or adapter import issue).
    """
    if not framework_descriptor.is_available():
        return None

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

    # NOTE:
    # This file historically used `adapter=...` directly. In the newer graph
    # context/error tests, the injection kwarg is registry-driven.
    # We keep this file stable and focused on interface conformance.
    init_kwargs: dict[str, Any] = {"adapter": adapter}
    return client_cls(**init_kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_unavailable_contract(descriptor: GraphFrameworkDescriptor) -> None:
    """
    Validate that an unavailable framework descriptor is behaving as expected.

    The test suite policy is "no skip": when unavailable, tests must pass by
    asserting correct unavailability signaling.
    """
    assert descriptor.is_available() is False

    # If availability_attr is set, adapter module should generally import and expose the flag.
    # If the module cannot import, that is also a valid "unavailable" signal.
    if descriptor.availability_attr:
        try:
            module = importlib.import_module(descriptor.adapter_module)
        except Exception:
            return
        flag = getattr(module, descriptor.availability_attr, None)
        # Either missing (treated as unavailable) or False.
        assert flag is None or bool(flag) is False


def _get_method(instance: Any, name: str | None) -> Callable[..., Any]:
    assert name, "Expected a non-empty method name"
    attr = getattr(instance, name, None)
    assert callable(attr), f"{instance!r} missing expected callable method {name!r}"
    return attr


def _get_unbound_method(owner: type, name: str) -> Callable[..., Any]:
    attr = getattr(owner, name, None)
    assert callable(attr), f"{owner!r} missing expected callable method {name!r}"
    return attr


def _context_kwargs_for_descriptor(framework_descriptor: GraphFrameworkDescriptor) -> dict[str, Any]:
    """
    Build kwargs reflecting the framework's declared context parameter.

    Returns a dict with a single key (the framework's context_kwarg) containing
    the rich context mapping, or an empty dict if no context_kwarg is declared.
    """
    kw: dict[str, Any] = {}

    if framework_descriptor.context_kwarg:
        # Build a rich context with test data plus framework tag
        ctx = dict(RICH_CONTEXT)
        try:
            tags = list(ctx.get("tags", []))
            tags.append(framework_descriptor.name)
            ctx["tags"] = tags
        except Exception:
            pass
        # Pass the entire context under the framework-specific kwarg
        kw[framework_descriptor.context_kwarg] = ctx

    return kw


def _call_with_minimal_args(
    fn: Callable[..., Any],
    *,
    kind: str,
    text: str,
    framework_descriptor: GraphFrameworkDescriptor,
) -> Any:
    from corpus_sdk.graph.graph_base import BulkVerticesSpec, BatchOperation

    kw = _context_kwargs_for_descriptor(framework_descriptor)

    if kind in {"query", "stream", "async_query", "async_stream"}:
        return fn(text, **kw)

    if kind == "bulk":
        # bulk_vertices expects a BulkVerticesSpec, not a list
        spec = BulkVerticesSpec(namespace="test", limit=10)
        return fn(spec, **kw)

    if kind == "batch":
        # batch expects a list of BatchOperation
        ops = [BatchOperation(op="test", args={})]
        return fn(ops, **kw)

    if kind in {"capabilities", "health"}:
        return fn(**kw)

    raise AssertionError(f"Unknown call kind: {kind!r}")


def _params_list(sig: inspect.Signature) -> list[inspect.Parameter]:
    params = list(sig.parameters.values())
    if params and params[0].name == "self":
        return params[1:]
    return params


# ---------------------------------------------------------------------------
# Core interface / surface contract tests
# ---------------------------------------------------------------------------


def test_can_instantiate_graph_client(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert graph_client_instance is not None

    _get_method(graph_client_instance, framework_descriptor.query_method)

    if framework_descriptor.stream_query_method:
        _get_method(graph_client_instance, framework_descriptor.stream_query_method)

    if framework_descriptor.supports_bulk_vertices:
        assert framework_descriptor.bulk_vertices_method is not None, (
            f"{framework_descriptor.name}: supports_bulk_vertices=True "
            f"but bulk_vertices_method is None"
        )
        _get_method(graph_client_instance, framework_descriptor.bulk_vertices_method)

    if framework_descriptor.supports_batch:
        assert framework_descriptor.batch_method is not None, (
            f"{framework_descriptor.name}: supports_batch=True but batch_method is None"
        )
        _get_method(graph_client_instance, framework_descriptor.batch_method)

    if framework_descriptor.supports_async:
        assert framework_descriptor.async_query_method is not None, (
            f"{framework_descriptor.name}: supports_async=True but async_query_method is None"
        )
        assert framework_descriptor.async_stream_query_method is not None, (
            f"{framework_descriptor.name}: supports_async=True but async_stream_query_method is None"
        )

        _get_method(graph_client_instance, framework_descriptor.async_query_method)
        _get_method(graph_client_instance, framework_descriptor.async_stream_query_method)

        if framework_descriptor.async_bulk_vertices_method:
            _get_method(graph_client_instance, framework_descriptor.async_bulk_vertices_method)

        if framework_descriptor.async_batch_method:
            _get_method(graph_client_instance, framework_descriptor.async_batch_method)


def test_sync_query_interface_conformance(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert graph_client_instance is not None

    query_fn = _get_method(graph_client_instance, framework_descriptor.query_method)
    result = _call_with_minimal_args(
        query_fn,
        kind="query",
        text=SYNC_QUERY_TEXT,
        framework_descriptor=framework_descriptor,
    )
    assert result is not None


def test_sync_streaming_interface_when_declared(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert graph_client_instance is not None

    if not framework_descriptor.stream_query_method:
        assert framework_descriptor.stream_query_method is None
        return

    stream_fn = _get_method(graph_client_instance, framework_descriptor.stream_query_method)

    iterator = _call_with_minimal_args(
        stream_fn,
        kind="stream",
        text=SYNC_STREAM_TEXT,
        framework_descriptor=framework_descriptor,
    )

    assert iterator is not None
    assert isinstance(iterator, Iterable), "Sync stream must return an iterable"

    for _ in iterator:  # noqa: B007
        break


@pytest.mark.asyncio
async def test_async_query_interface_conformance_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert graph_client_instance is not None

    if not framework_descriptor.async_query_method:
        assert framework_descriptor.async_query_method is None
        return

    aquery_fn = _get_method(graph_client_instance, framework_descriptor.async_query_method)

    coro = _call_with_minimal_args(
        aquery_fn,
        kind="async_query",
        text=ASYNC_QUERY_TEXT,
        framework_descriptor=framework_descriptor,
    )
    assert inspect.isawaitable(coro), "Async query method must return an awaitable"

    result = await coro
    assert result is not None


@pytest.mark.asyncio
async def test_async_streaming_interface_conformance_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert graph_client_instance is not None

    if not framework_descriptor.async_stream_query_method:
        assert framework_descriptor.async_stream_query_method is None
        return

    astream_fn = _get_method(graph_client_instance, framework_descriptor.async_stream_query_method)

    aiter = _call_with_minimal_args(
        astream_fn,
        kind="async_stream",
        text=ASYNC_STREAM_TEXT,
        framework_descriptor=framework_descriptor,
    )

    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    assert isinstance(aiter, AsyncIterable), "Async stream must yield an async-iterable"

    async for _ in aiter:  # noqa: B007
        break


def test_context_kwarg_is_accepted_when_declared_on_primary_query(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert graph_client_instance is not None

    if not framework_descriptor.context_kwarg:
        assert framework_descriptor.context_kwarg is None
        return

    query_fn = _get_method(graph_client_instance, framework_descriptor.query_method)
    result = query_fn(
        CONTEXT_QUERY_TEXT,
        **_context_kwargs_for_descriptor(framework_descriptor),
    )
    assert result is not None


def test_bulk_and_batch_methods_are_callable_when_declared(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert graph_client_instance is not None

    if framework_descriptor.supports_bulk_vertices and framework_descriptor.bulk_vertices_method:
        bulk_fn = _get_method(graph_client_instance, framework_descriptor.bulk_vertices_method)
        _call_with_minimal_args(
            bulk_fn,
            kind="bulk",
            text="",
            framework_descriptor=framework_descriptor,
        )

    if framework_descriptor.supports_batch and framework_descriptor.batch_method:
        batch_fn = _get_method(graph_client_instance, framework_descriptor.batch_method)
        _call_with_minimal_args(
            batch_fn,
            kind="batch",
            text="",
            framework_descriptor=framework_descriptor,
        )


@pytest.mark.asyncio
async def test_async_bulk_and_batch_methods_are_awaitable_when_declared(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert graph_client_instance is not None

    if framework_descriptor.async_bulk_vertices_method:
        abulk_fn = _get_method(graph_client_instance, framework_descriptor.async_bulk_vertices_method)
        coro = _call_with_minimal_args(
            abulk_fn,
            kind="bulk",
            text="",
            framework_descriptor=framework_descriptor,
        )
        assert inspect.isawaitable(coro)
        await coro

    if framework_descriptor.async_batch_method:
        abatch_fn = _get_method(graph_client_instance, framework_descriptor.async_batch_method)
        coro = _call_with_minimal_args(
            abatch_fn,
            kind="batch",
            text="",
            framework_descriptor=framework_descriptor,
        )
        assert inspect.isawaitable(coro)
        await coro


def test_method_signatures_consistent_between_sync_and_async(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert graph_client_instance is not None

    owner = type(graph_client_instance)

    def _compare_signatures(sync_name: str | None, async_name: str | None) -> None:
        if not sync_name or not async_name:
            return

        sync_unbound = _get_unbound_method(owner, sync_name)
        async_unbound = _get_unbound_method(owner, async_name)

        sync_sig = inspect.signature(sync_unbound)
        async_sig = inspect.signature(async_unbound)

        sync_params = _params_list(sync_sig)
        async_params = _params_list(async_sig)

        sync_view = [(p.name, p.kind) for p in sync_params]
        async_view = [(p.name, p.kind) for p in async_params]

        assert sync_view == async_view, (
            f"Signature mismatch between {sync_name!r} and {async_name!r}: "
            f"{sync_view} != {async_view}"
        )

    _compare_signatures(framework_descriptor.query_method, framework_descriptor.async_query_method)
    _compare_signatures(framework_descriptor.stream_query_method, framework_descriptor.async_stream_query_method)
    _compare_signatures(framework_descriptor.bulk_vertices_method, framework_descriptor.async_bulk_vertices_method)
    _compare_signatures(framework_descriptor.batch_method, framework_descriptor.async_batch_method)


# ---------------------------------------------------------------------------
# Capabilities / health passthrough contract (NO SKIPS)
# ---------------------------------------------------------------------------


def test_capabilities_contract_matches_registry_flag(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    NO SKIPS:
      - If framework is unavailable -> validate unavailable contract and pass.
      - If has_capabilities=True -> capabilities() must exist and return Mapping.
      - If has_capabilities=False -> capabilities() must NOT exist (or must not be callable).
        (If it exists, registry is wrong; force a failure so it gets fixed.)
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert graph_client_instance is not None

    capabilities = getattr(graph_client_instance, "capabilities", None)

    if framework_descriptor.has_capabilities:
        assert callable(capabilities), "Registry says has_capabilities=True but capabilities() is missing"
        caps_result = _call_with_minimal_args(
            capabilities,
            kind="capabilities",
            text="",
            framework_descriptor=framework_descriptor,
        )
        assert isinstance(caps_result, Mapping), "capabilities() should return a Mapping"

        # Async variant is optional, but if present it must be callable.
        async_caps = getattr(graph_client_instance, "acapabilities", None)
        if async_caps is not None:
            assert callable(async_caps), "acapabilities exists but is not callable"
    else:
        assert not callable(capabilities), (
            "Registry says has_capabilities=False but capabilities() exists/callable; "
            "either remove the method or flip has_capabilities=True in the registry"
        )
        async_caps = getattr(graph_client_instance, "acapabilities", None)
        assert not callable(async_caps), (
            "Registry says has_capabilities=False but acapabilities() exists/callable; "
            "either remove it or flip has_capabilities=True"
        )


@pytest.mark.asyncio
async def test_async_capabilities_returns_mapping_if_present(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    NO SKIPS:
      - If framework is unavailable -> validate unavailable contract and pass.
      - If acapabilities() exists, it must return Mapping.
      - If it does not exist, test passes (async variant is optional).
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert graph_client_instance is not None

    async_caps = getattr(graph_client_instance, "acapabilities", None)
    if not callable(async_caps):
        return

    coro = _call_with_minimal_args(
        async_caps,
        kind="capabilities",
        text="",
        framework_descriptor=framework_descriptor,
    )
    assert inspect.isawaitable(coro)
    result = await coro
    assert isinstance(result, Mapping), "acapabilities() should return a Mapping"


def test_health_contract_matches_registry_flag(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    NO SKIPS:
      - If framework is unavailable -> validate unavailable contract and pass.
      - If has_health=True -> health() must exist and return Mapping.
      - If has_health=False -> health() must NOT exist (or must not be callable).
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert graph_client_instance is not None

    health = getattr(graph_client_instance, "health", None)

    if framework_descriptor.has_health:
        assert callable(health), "Registry says has_health=True but health() is missing"
        health_result = _call_with_minimal_args(
            health,
            kind="health",
            text="",
            framework_descriptor=framework_descriptor,
        )
        assert isinstance(health_result, Mapping), "health() should return a Mapping"

        async_health = getattr(graph_client_instance, "ahealth", None)
        if async_health is not None:
            assert callable(async_health), "ahealth exists but is not callable"
    else:
        assert not callable(health), (
            "Registry says has_health=False but health() exists/callable; "
            "either remove the method or flip has_health=True in the registry"
        )
        async_health = getattr(graph_client_instance, "ahealth", None)
        assert not callable(async_health), (
            "Registry says has_health=False but ahealth() exists/callable; "
            "either remove it or flip has_health=True"
        )


@pytest.mark.asyncio
async def test_async_health_returns_mapping_if_present(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    NO SKIPS:
      - If framework is unavailable -> validate unavailable contract and pass.
      - If ahealth() exists, it must return Mapping.
      - If it does not exist, test passes (async variant is optional).
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert graph_client_instance is not None

    async_health = getattr(graph_client_instance, "ahealth", None)
    if not callable(async_health):
        return

    coro = _call_with_minimal_args(
        async_health,
        kind="health",
        text="",
        framework_descriptor=framework_descriptor,
    )
    assert inspect.isawaitable(coro)
    result = await coro
    assert isinstance(result, Mapping), "ahealth() should return a Mapping"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
