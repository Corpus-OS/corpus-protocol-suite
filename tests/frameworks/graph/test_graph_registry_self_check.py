# tests/frameworks/graph/test_graph_registry_self_check.py

import dataclasses
import pytest

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


@pytest.fixture
def all_descriptors():
    """Fixture providing all registered graph framework descriptors."""
    return list(iter_graph_framework_descriptors())


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
    desc1 = GraphFrameworkDescriptor(
        name="test1",
        **base_kwargs,
    )
    assert desc1.version_range() is None

    # Test minimum version only
    desc2 = GraphFrameworkDescriptor(
        name="test2",
        minimum_framework_version="1.0.0",
        **base_kwargs,
    )
    assert desc2.version_range() == ">=1.0.0"

    # Test maximum version only
    desc3 = GraphFrameworkDescriptor(
        name="test3",
        tested_up_to_version="2.5.0",
        **base_kwargs,
    )
    # Accept either "<=2.5.0" or "<= 2.5.0" (tolerate formatting differences).
    assert desc3.version_range() in {"<=2.5.0", "<= 2.5.0"}

    # Test both versions
    desc4 = GraphFrameworkDescriptor(
        name="test4",
        minimum_framework_version="1.2.0",
        tested_up_to_version="3.0.0",
        **base_kwargs,
    )
    # Be tolerant about spaces after "<=" as well.
    assert desc4.version_range() in {">=1.2.0, <=3.0.0", ">=1.2.0, <= 3.0.0"}


def test_async_method_consistency(all_descriptors) -> None:
    """
    Check that async graph query/stream support is properly declared.

    Policy: if any async query/stream method is declared, both should be
    present for API consistency.
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
    Test the supports_streaming property logic.

    Ensures the property correctly reflects whether ANY streaming method
    is declared (sync or async).
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
    Test the supports_async property logic.

    Ensures the property correctly reflects whether ANY async method
    is declared (query or stream).
    """
    for descriptor in all_descriptors:
        has_async = (
            descriptor.async_query_method is not None
            or descriptor.async_stream_query_method is not None
        )
        assert descriptor.supports_async == has_async, (
            f"{descriptor.name}: supports_async property mismatch"
        )


def test_bulk_vertices_and_batch_properties(all_descriptors) -> None:
    """
    Test supports_bulk_vertices and supports_batch properties.

    Ensures the flags correctly reflect whether bulk/batch methods
    are actually declared on the descriptor.
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

        # Test overwrite works (and captures the expected warning to avoid noisy output)
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
    with pytest.raises(
        ValueError, match="query_method and stream_query_method must both be set"
    ):
        GraphFrameworkDescriptor(
            name="bad1",
            adapter_module="test.module",
            adapter_class="TestGraphClient",
            query_method="",
            stream_query_method="stream_query",
            bulk_vertices_method="bulk_vertices",
            batch_method="batch",
        )

    with pytest.raises(
        ValueError, match="query_method and stream_query_method must both be set"
    ):
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
    with pytest.warns(
        RuntimeWarning, match="adapter_class should be a class name only"
    ):
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
