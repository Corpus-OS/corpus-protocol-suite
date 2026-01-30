# tests/frameworks/graph/test_graph_registry_self_check.py

from __future__ import annotations

import dataclasses
import importlib
import inspect
import re
from typing import Any, Callable, Optional

import pytest

from corpus_sdk.graph.graph_base import (
    BatchOperation,
    GraphTraversalSpec,
)
from tests.frameworks.registries.graph_registry import (
    GRAPH_FRAMEWORKS,
    GraphFrameworkDescriptor,
    iter_graph_framework_descriptors,
    register_graph_framework_descriptor,
    has_graph_framework,
    get_graph_framework_descriptor,
    get_graph_framework_descriptor_safe,
    iter_available_graph_framework_descriptors,
)


def _normalize_version_range(s: Optional[str]) -> Optional[str]:
    """
    Normalize version range formatting so tests accept either:
      - "<=2.5.0" or "<= 2.5.0"
      - ">=1.2.0" or ">= 1.2.0"

    We intentionally treat version_range() as informational. The conformance
    suite cares about correctness of bounds, not whitespace.
    """
    if s is None:
        return None
    # Remove any whitespace after >= or <= operators.
    s = re.sub(r"(<=|>=)\s+", r"\1", s)
    # Normalize comma spacing to exactly ", " (tolerate ",", ",  ", " , ").
    s = re.sub(r"\s*,\s*", ", ", s)
    return s


def _get_callable(instance: Any, name: Optional[str]) -> Callable[..., Any]:
    """
    Fetch a callable method from an instance, failing with a clear message.

    We keep this strict because the point of this file is to ensure the registry
    accurately describes real framework adapter surfaces.
    """
    assert name, "Expected a non-empty method name"
    attr = getattr(instance, name, None)
    assert callable(attr), f"{instance!r} missing expected callable method {name!r}"
    return attr


@pytest.fixture
def all_descriptors():
    """Fixture providing all registered graph framework descriptors."""
    return list(iter_graph_framework_descriptors())


@pytest.fixture(
    params=list(iter_graph_framework_descriptors()),
    name="framework_descriptor",
)
def framework_descriptor_fixture(request: pytest.FixtureRequest) -> GraphFrameworkDescriptor:
    """
    Parameterized fixture over all registered graph framework descriptors.

    Frameworks that are not installed/available are skipped (best-effort),
    consistent with the graph contract suite's current policy.
    """
    desc: GraphFrameworkDescriptor = request.param
    if not desc.is_available():
        pytest.skip(f"Framework '{desc.name}' not available in this environment")
    return desc


@pytest.fixture
def graph_client_instance(framework_descriptor: GraphFrameworkDescriptor, adapter: Any) -> Any:
    """
    Construct a concrete framework graph client instance using the injected Corpus adapter.

    IMPORTANT:
    - This fixture is the mechanism that ensures CORPUS_ADAPTER injection is actually
      exercised for the registry self-check tests.
    - Running this test file with:
        CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter
      ensures the framework adapter is wired to your mock backend.

    Injection compatibility:
    - Different framework adapters use different constructor kwarg names for the injected
      Corpus graph adapter (e.g. "adapter" vs "corpus_adapter").
    - The registry descriptor provides `adapter_init_kwarg` so tests do not hardcode this.
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    client_cls = getattr(module, framework_descriptor.adapter_class)

    init_kw = getattr(framework_descriptor, "adapter_init_kwarg", None) or "adapter"
    init_kwargs = {init_kw: adapter}
    return client_cls(**init_kwargs)


def test_graph_registry_keys_match_descriptor_name(all_descriptors) -> None:
    """
    Sanity check: registry keys should always match descriptor.name.

    This keeps lookups and reporting consistent and prevents copy/paste errors
    when adding new graph frameworks.
    """
    for descriptor in all_descriptors:
        assert descriptor.name in GRAPH_FRAMEWORKS
        assert GRAPH_FRAMEWORKS[descriptor.name] is descriptor


def test_graph_registry_descriptors_validate_cleanly(all_descriptors) -> None:
    """
    Run the descriptor-level validation hook to catch obvious inconsistencies
    (e.g. async stream defined without async query).
    """
    for descriptor in all_descriptors:
        # validate() may emit warnings but should not raise
        descriptor.validate()


def test_descriptor_is_available_does_not_raise(all_descriptors) -> None:
    """
    Ensure is_available() doesn't crash for any registered descriptor.

    This is a smoke test that verifies the availability check doesn't raise
    unexpected exceptions (ImportError, AttributeError) when called.
    """
    for descriptor in all_descriptors:
        result = descriptor.is_available()
        assert isinstance(result, bool)


def test_version_range_formatting() -> None:
    """
    Test version_range() returns expected format for various version combinations.

    IMPORTANT:
    - This test accepts either "<=2.5.0" or "<= 2.5.0" (space tolerance),
      because formatting is informational.
    """
    base_kwargs = dict(
        adapter_module="test.module",
        adapter_class="TestGraphClient",
        query_method="query",
        stream_query_method="stream_query",
        bulk_vertices_method="bulk_vertices",
        batch_method="batch",
    )

    # Test no versions
    desc1 = GraphFrameworkDescriptor(name="test1", **base_kwargs)
    assert desc1.version_range() is None

    # Test minimum version only
    desc2 = GraphFrameworkDescriptor(name="test2", minimum_framework_version="1.0.0", **base_kwargs)
    assert _normalize_version_range(desc2.version_range()) == ">=1.0.0"

    # Test maximum version only
    desc3 = GraphFrameworkDescriptor(name="test3", tested_up_to_version="2.5.0", **base_kwargs)
    assert _normalize_version_range(desc3.version_range()) == "<=2.5.0"

    # Test both versions
    desc4 = GraphFrameworkDescriptor(
        name="test4",
        minimum_framework_version="1.2.0",
        tested_up_to_version="3.0.0",
        **base_kwargs,
    )
    assert _normalize_version_range(desc4.version_range()) == ">=1.2.0, <=3.0.0"


def test_async_method_consistency(all_descriptors) -> None:
    """
    Check that async graph query/stream support is properly declared.

    Policy: if any async support is declared, both async query and async streaming
    should be present for API consistency.
    """
    for descriptor in all_descriptors:
        if descriptor.supports_async:
            assert descriptor.async_query_method is not None, (
                f"{descriptor.name}: has async support but async_query_method is None"
            )
            assert descriptor.async_stream_query_method is not None, (
                f"{descriptor.name}: has async support but async_stream_query_method is None"
            )


def test_streaming_support_property(all_descriptors) -> None:
    """
    Ensures supports_streaming matches the existence of any streaming method name.
    """
    for descriptor in all_descriptors:
        has_streaming = (
            descriptor.stream_query_method is not None
            or descriptor.async_stream_query_method is not None
        )
        assert descriptor.supports_streaming == has_streaming, (
            f"{descriptor.name}: supports_streaming property mismatch"
        )


def test_supports_async_property(all_descriptors) -> None:
    """
    Ensures supports_async matches whether ANY async method is declared.
    """
    for descriptor in all_descriptors:
        has_async = bool(
            descriptor.async_query_method
            or descriptor.async_stream_query_method
            or descriptor.async_bulk_vertices_method
            or descriptor.async_batch_method
            or getattr(descriptor, "async_capabilities_method", None)
            or getattr(descriptor, "async_health_method", None)
            or getattr(descriptor, "async_schema_method", None)
            or getattr(descriptor, "async_transaction_method", None)
            or getattr(descriptor, "async_traversal_method", None)
        )
        assert descriptor.supports_async == has_async, (
            f"{descriptor.name}: supports_async property mismatch"
        )


def test_bulk_vertices_and_batch_properties(all_descriptors) -> None:
    """
    Ensures supports_bulk_vertices and supports_batch reflect whether those methods exist.
    """
    for descriptor in all_descriptors:
        has_bulk = (
            descriptor.bulk_vertices_method is not None
            or descriptor.async_bulk_vertices_method is not None
        )
        assert descriptor.supports_bulk_vertices == has_bulk, (
            f"{descriptor.name}: supports_bulk_vertices property mismatch"
        )

        has_batch = (
            descriptor.batch_method is not None
            or descriptor.async_batch_method is not None
        )
        assert descriptor.supports_batch == has_batch, (
            f"{descriptor.name}: supports_batch property mismatch"
        )


def test_registry_declared_capabilities_and_health_are_callable_and_return_values(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    HARD CHECK: if registry declares capability/health method names, they must:
      - exist on the framework adapter instance,
      - be callable,
      - return a non-None value.

    This directly validates wiring against the injected Corpus adapter.
    """
    # Capabilities (sync)
    if getattr(framework_descriptor, "capabilities_method", None):
        caps_fn = _get_callable(graph_client_instance, framework_descriptor.capabilities_method)
        caps = caps_fn()
        assert caps is not None, f"{framework_descriptor.name}: capabilities() returned None"

    # Health (sync)
    if getattr(framework_descriptor, "health_method", None):
        health_fn = _get_callable(graph_client_instance, framework_descriptor.health_method)
        health = health_fn()
        assert health is not None, f"{framework_descriptor.name}: health() returned None"


def test_registry_declared_schema_transaction_traversal_are_callable_and_return_values(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    HARD CHECK: if registry declares schema/transaction/traversal method names, they must:
      - exist on the framework adapter instance,
      - be callable,
      - return non-None results (and minimally sane shapes).

    These tests are intentionally lightweight on shape because deeper semantics
    are validated in dedicated contract tests. The point here is "wired + callable".
    """
    # Schema (sync)
    if getattr(framework_descriptor, "schema_method", None):
        schema_fn = _get_callable(graph_client_instance, framework_descriptor.schema_method)
        schema = schema_fn()
        assert schema is not None, f"{framework_descriptor.name}: schema() returned None"
        # Best-effort: schema objects typically have nodes/edges; keep this tolerant.
        assert hasattr(schema, "nodes") or isinstance(schema, dict), (
            f"{framework_descriptor.name}: schema result shape unexpected: {type(schema).__name__}"
        )

    # Transaction (sync)
    if getattr(framework_descriptor, "transaction_method", None):
        tx_fn = _get_callable(graph_client_instance, framework_descriptor.transaction_method)
        tx_res = tx_fn([BatchOperation(op="query", args={"text": "RETURN 1"})])
        assert tx_res is not None, f"{framework_descriptor.name}: transaction() returned None"
        # Best-effort: BatchResult typically has results list
        assert hasattr(tx_res, "results") or isinstance(tx_res, dict), (
            f"{framework_descriptor.name}: transaction result shape unexpected: {type(tx_res).__name__}"
        )

    # Traversal (sync)
    if getattr(framework_descriptor, "traversal_method", None):
        trav_fn = _get_callable(graph_client_instance, framework_descriptor.traversal_method)
        trav_res = trav_fn(GraphTraversalSpec(start_nodes=["v:start:1"], max_depth=1))
        assert trav_res is not None, f"{framework_descriptor.name}: traversal() returned None"
        assert hasattr(trav_res, "nodes") or isinstance(trav_res, dict), (
            f"{framework_descriptor.name}: traversal result shape unexpected: {type(trav_res).__name__}"
        )


@pytest.mark.asyncio
async def test_registry_declared_async_methods_are_awaitable_and_return_values(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    HARD CHECK (async): if registry declares async method names, they must:
      - exist on the framework adapter instance,
      - be callable,
      - return an awaitable,
      - resolve to non-None values.

    This mirrors the sync checks but ensures the async surface is real.
    """
    # Async capabilities
    if getattr(framework_descriptor, "async_capabilities_method", None):
        acaps_fn = _get_callable(graph_client_instance, framework_descriptor.async_capabilities_method)
        out = acaps_fn()
        assert inspect.isawaitable(out), (
            f"{framework_descriptor.name}: {framework_descriptor.async_capabilities_method} must return awaitable"
        )
        caps = await out
        assert caps is not None, f"{framework_descriptor.name}: async capabilities returned None"

    # Async health
    if getattr(framework_descriptor, "async_health_method", None):
        ahealth_fn = _get_callable(graph_client_instance, framework_descriptor.async_health_method)
        out = ahealth_fn()
        assert inspect.isawaitable(out), (
            f"{framework_descriptor.name}: {framework_descriptor.async_health_method} must return awaitable"
        )
        health = await out
        assert health is not None, f"{framework_descriptor.name}: async health returned None"

    # Async schema
    if getattr(framework_descriptor, "async_schema_method", None):
        aschema_fn = _get_callable(graph_client_instance, framework_descriptor.async_schema_method)
        out = aschema_fn()
        assert inspect.isawaitable(out), (
            f"{framework_descriptor.name}: {framework_descriptor.async_schema_method} must return awaitable"
        )
        schema = await out
        assert schema is not None, f"{framework_descriptor.name}: async schema returned None"

    # Async transaction
    if getattr(framework_descriptor, "async_transaction_method", None):
        atx_fn = _get_callable(graph_client_instance, framework_descriptor.async_transaction_method)
        out = atx_fn([BatchOperation(op="query", args={"text": "RETURN 1"})])
        assert inspect.isawaitable(out), (
            f"{framework_descriptor.name}: {framework_descriptor.async_transaction_method} must return awaitable"
        )
        tx_res = await out
        assert tx_res is not None, f"{framework_descriptor.name}: async transaction returned None"

    # Async traversal
    if getattr(framework_descriptor, "async_traversal_method", None):
        atrav_fn = _get_callable(graph_client_instance, framework_descriptor.async_traversal_method)
        out = atrav_fn(GraphTraversalSpec(start_nodes=["v:start:1"], max_depth=1))
        assert inspect.isawaitable(out), (
            f"{framework_descriptor.name}: {framework_descriptor.async_traversal_method} must return awaitable"
        )
        trav_res = await out
        assert trav_res is not None, f"{framework_descriptor.name}: async traversal returned None"


def test_register_graph_framework_descriptor() -> None:
    """
    Test dynamic registration functionality for graph frameworks.

    This tests the ability to add new framework descriptors at runtime,
    which is useful for testing experimental or third-party graph adapters.
    """
    original_registry = dict(GRAPH_FRAMEWORKS)
    try:
        base_kwargs = dict(
            adapter_module="test.module",
            adapter_class="TestGraphClient",
            query_method="query",
            stream_query_method="stream_query",
            bulk_vertices_method="bulk_vertices",
            batch_method="batch",
        )

        # Create a test descriptor
        test_desc = GraphFrameworkDescriptor(
            name="test_framework",
            async_query_method="aquery",
            async_stream_query_method="astream_query",
            **base_kwargs,
        )

        # Should not exist initially
        assert not has_graph_framework("test_framework")
        assert get_graph_framework_descriptor_safe("test_framework") is None

        # Test registration without overwrite
        register_graph_framework_descriptor(test_desc)

        # Should now exist
        assert has_graph_framework("test_framework")
        assert get_graph_framework_descriptor_safe("test_framework") is test_desc
        assert get_graph_framework_descriptor("test_framework") is test_desc

        # Test registration with existing name fails without overwrite
        duplicate_desc = GraphFrameworkDescriptor(
            name="test_framework",
            adapter_module="other.module",
            adapter_class="OtherClient",
            query_method="other_query",
            stream_query_method="other_stream_query",
            bulk_vertices_method="other_bulk_vertices",
            batch_method="other_batch",
        )

        with pytest.raises(KeyError, match="already registered"):
            register_graph_framework_descriptor(duplicate_desc, overwrite=False)

        # Capture overwrite warning explicitly (so it does not show up as a test-suite warning summary).
        with pytest.warns(RuntimeWarning, match="being overwritten"):
            register_graph_framework_descriptor(duplicate_desc, overwrite=True)

        assert get_graph_framework_descriptor("test_framework") is duplicate_desc
    finally:
        GRAPH_FRAMEWORKS.clear()
        GRAPH_FRAMEWORKS.update(original_registry)


def test_get_descriptor_variants() -> None:
    """
    Test both get_descriptor functions behave as expected.

    Verifies that the safe version returns None for unknown frameworks
    while the regular version raises KeyError.
    """
    existing_name = list(GRAPH_FRAMEWORKS.keys())[0]

    # Test existing framework
    assert get_graph_framework_descriptor(existing_name) is not None
    assert get_graph_framework_descriptor_safe(existing_name) is not None

    # Test non-existent framework
    non_existent = "non_existent_graph_framework_xyz123"

    with pytest.raises(KeyError, match=non_existent):
        get_graph_framework_descriptor(non_existent)

    assert get_graph_framework_descriptor_safe(non_existent) is None


def test_descriptor_immutability() -> None:
    """
    Test that graph descriptors are immutable (frozen dataclass).
    """
    descriptor = GraphFrameworkDescriptor(
        name="test",
        adapter_module="test.module",
        adapter_class="TestGraphClient",
        query_method="query",
        stream_query_method="stream_query",
        bulk_vertices_method="bulk_vertices",
        batch_method="batch",
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.name = "modified"

    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.query_method = "modified_query"


def test_iterator_functions() -> None:
    """
    Test that iterator functions return expected results.
    """
    # iter_graph_framework_descriptors
    all_descs = list(iter_graph_framework_descriptors())
    assert len(all_descs) == len(GRAPH_FRAMEWORKS)

    for desc in all_descs:
        assert desc.name in GRAPH_FRAMEWORKS

    # iter_available_graph_framework_descriptors
    available_descs = list(iter_available_graph_framework_descriptors())

    assert len(available_descs) <= len(all_descs)

    for desc in available_descs:
        assert desc.is_available()


def test_descriptor_validation_edge_cases() -> None:
    """
    Test descriptor validation with edge cases.
    """
    base_kwargs = dict(
        adapter_module="test.module",
        adapter_class="TestGraphClient",
        query_method="query",
        stream_query_method="stream_query",
        bulk_vertices_method="bulk_vertices",
        batch_method="batch",
    )

    # Missing required methods
    with pytest.raises(ValueError, match="query_method and stream_query_method must both be set"):
        GraphFrameworkDescriptor(
            name="bad1",
            adapter_module="test.module",
            adapter_class="TestGraphClient",
            query_method="",
            stream_query_method="stream_query",
            bulk_vertices_method="bulk_vertices",
            batch_method="batch",
        )

    with pytest.raises(ValueError, match="query_method and stream_query_method must both be set"):
        GraphFrameworkDescriptor(
            name="bad2",
            adapter_module="test.module",
            adapter_class="TestGraphClient",
            query_method="query",
            stream_query_method="",
            bulk_vertices_method="bulk_vertices",
            batch_method="batch",
        )

    # Dotted adapter_class (should warn but not fail)
    with pytest.warns(RuntimeWarning, match="adapter_class should be a class name only"):
        GraphFrameworkDescriptor(
            name="warn1",
            adapter_module="test.module",
            adapter_class="some.module.ClassName",
            query_method="query",
            stream_query_method="stream_query",
            bulk_vertices_method="bulk_vertices",
            batch_method="batch",
        )

    # Async stream without async query (should warn but not fail)
    with pytest.warns(
        RuntimeWarning,
        match="async_stream_query_method is set but async_query_method is None",
    ):
        GraphFrameworkDescriptor(
            name="warn2",
            adapter_module="test.module",
            adapter_class="TestGraphClient",
            query_method="query",
            stream_query_method="stream_query",
            bulk_vertices_method="bulk_vertices",
            batch_method="batch",
            async_stream_query_method="astream_query",
            # async_query_method=None (implicit)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
