# tests/frameworks/embedding/test_embedding_registry_self_check.py

import dataclasses
import pytest

from tests.frameworks.registries.embedding_registry import (
    EMBEDDING_FRAMEWORKS,
    EmbeddingFrameworkDescriptor,
    iter_embedding_framework_descriptors,
    register_framework_descriptor,
    has_framework,
    get_embedding_framework_descriptor,
    get_embedding_framework_descriptor_safe,
    iter_available_framework_descriptors,
)


@pytest.fixture
def all_descriptors():
    """Fixture providing all registered descriptors."""
    return list(iter_embedding_framework_descriptors())


def test_embedding_registry_keys_match_descriptor_name(all_descriptors) -> None:
    """
    Sanity check: registry keys should always match descriptor.name.

    This keeps lookups and reporting consistent and prevents copy/paste errors
    when adding new frameworks.
    """
    for descriptor in all_descriptors:
        assert descriptor.name in EMBEDDING_FRAMEWORKS
        assert EMBEDDING_FRAMEWORKS[descriptor.name] is descriptor


def test_embedding_registry_descriptors_validate_cleanly(all_descriptors) -> None:
    """
    Run the descriptor-level validation hook to catch obvious inconsistencies
    (e.g. async query defined without async batch).
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
        # Should not raise ImportError or AttributeError
        result = descriptor.is_available()
        assert isinstance(result, bool)


def test_version_range_formatting() -> None:
    """
    Test version_range() returns expected format for various version combinations.
    """
    # Test no versions
    desc1 = EmbeddingFrameworkDescriptor(
        name="test1",
        adapter_module="test.module",
        adapter_class="TestAdapter",
        batch_method="embed",
        query_method="query",
    )
    assert desc1.version_range() is None

    # Test minimum version only
    desc2 = EmbeddingFrameworkDescriptor(
        name="test2",
        adapter_module="test.module",
        adapter_class="TestAdapter",
        batch_method="embed",
        query_method="query",
        minimum_framework_version="1.0.0",
    )
    assert desc2.version_range() == ">=1.0.0"

    # Test maximum version only
    desc3 = EmbeddingFrameworkDescriptor(
        name="test3",
        adapter_module="test.module",
        adapter_class="TestAdapter",
        batch_method="embed",
        query_method="query",
        tested_up_to_version="2.5.0",
    )
    assert desc3.version_range() == "<=2.5.0"

    # Test both versions
    desc4 = EmbeddingFrameworkDescriptor(
        name="test4",
        adapter_module="test.module",
        adapter_class="TestAdapter",
        batch_method="embed",
        query_method="query",
        minimum_framework_version="1.2.0",
        tested_up_to_version="3.0.0",
    )
    assert desc4.version_range() == ">=1.2.0, <=3.0.0"


def test_async_method_consistency(all_descriptors) -> None:
    """
    Check that async support is properly declared.

    This is a stricter version of the warning check in validate().
    Our policy: if any async method is declared, both should be present
    for API consistency.
    """
    for descriptor in all_descriptors:
        if descriptor.supports_async:
            # Framework policy: async should be all-or-nothing
            assert descriptor.async_batch_method is not None, (
                f"{descriptor.name}: has async support but async_batch_method is None"
            )
            assert descriptor.async_query_method is not None, (
                f"{descriptor.name}: has async support but async_query_method is None"
            )


def test_register_framework_descriptor() -> None:
    """
    Test dynamic registration functionality.

    This tests the ability to add new framework descriptors at runtime,
    which is useful for testing experimental or third-party adapters.
    """
    # Snapshot registry for test isolation
    original_registry = dict(EMBEDDING_FRAMEWORKS)
    try:
        # Create a test descriptor
        test_desc = EmbeddingFrameworkDescriptor(
            name="test_framework",
            adapter_module="test.module",
            adapter_class="TestAdapter",
            batch_method="embed",
            query_method="query",
            async_batch_method="aembed",
            async_query_method="aquery",
        )

        # Should not exist initially
        assert not has_framework("test_framework")
        assert get_embedding_framework_descriptor_safe("test_framework") is None

        # Test registration without overwrite
        register_framework_descriptor(test_desc)

        # Should now exist
        assert has_framework("test_framework")
        assert get_embedding_framework_descriptor_safe("test_framework") is test_desc
        assert get_embedding_framework_descriptor("test_framework") is test_desc

        # Test registration with existing name fails without overwrite
        duplicate_desc = EmbeddingFrameworkDescriptor(
            name="test_framework",
            adapter_module="other.module",
            adapter_class="OtherAdapter",
            batch_method="other_embed",
            query_method="other_query",
        )

        with pytest.raises(KeyError, match="already registered"):
            register_framework_descriptor(duplicate_desc, overwrite=False)

        # Test overwrite works
        register_framework_descriptor(duplicate_desc, overwrite=True)
        assert get_embedding_framework_descriptor("test_framework") is duplicate_desc
    finally:
        # Restore registry to original state
        EMBEDDING_FRAMEWORKS.clear()
        EMBEDDING_FRAMEWORKS.update(original_registry)


def test_supports_async_property(all_descriptors) -> None:
    """
    Test the supports_async property logic.

    Ensures the property correctly reflects whether ANY async method
    is declared, not necessarily all of them.
    """
    for descriptor in all_descriptors:
        has_async = (
            descriptor.async_batch_method is not None
            or descriptor.async_query_method is not None
        )
        assert descriptor.supports_async == has_async, (
            f"{descriptor.name}: supports_async property mismatch"
        )


def test_get_descriptor_variants() -> None:
    """
    Test both get_descriptor functions behave as expected.

    Verifies that the safe version returns None for unknown frameworks
    while the regular version raises KeyError.
    """
    # Get a known framework name
    existing_name = list(EMBEDDING_FRAMEWORKS.keys())[0]

    # Test existing framework
    assert get_embedding_framework_descriptor(existing_name) is not None
    assert get_embedding_framework_descriptor_safe(existing_name) is not None

    # Test non-existent framework
    non_existent = "non_existent_framework_xyz123"

    # Regular version should raise
    with pytest.raises(KeyError, match=non_existent):
        get_embedding_framework_descriptor(non_existent)

    # Safe version should return None
    assert get_embedding_framework_descriptor_safe(non_existent) is None


def test_descriptor_immutability() -> None:
    """
    Test that descriptors are immutable (frozen dataclass).
    """
    descriptor = EmbeddingFrameworkDescriptor(
        name="test",
        adapter_module="test.module",
        adapter_class="TestAdapter",
        batch_method="embed",
        query_method="query",
    )

    # Should not be able to modify attributes
    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.name = "modified"

    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.batch_method = "modified_embed"


def test_iterator_functions() -> None:
    """
    Test that iterator functions return expected results.
    """
    # Test iter_embedding_framework_descriptors
    all_descs = list(iter_embedding_framework_descriptors())
    assert len(all_descs) == len(EMBEDDING_FRAMEWORKS)

    # All descriptors should be in the registry
    for desc in all_descs:
        assert desc.name in EMBEDDING_FRAMEWORKS

    # Test iter_available_framework_descriptors
    available_descs = list(iter_available_framework_descriptors())

    # Available descriptors should be a subset
    assert len(available_descs) <= len(all_descs)

    # Each available descriptor should pass is_available()
    for desc in available_descs:
        assert desc.is_available()


def test_descriptor_validation_edge_cases() -> None:
    """
    Test descriptor validation with edge cases.
    """
    # Test missing required methods
    with pytest.raises(
        ValueError, match="batch_method and query_method must both be set"
    ):
        EmbeddingFrameworkDescriptor(
            name="bad1",
            adapter_module="test.module",
            adapter_class="TestAdapter",
            batch_method="",
            query_method="query",
        )

    with pytest.raises(
        ValueError, match="batch_method and query_method must both be set"
    ):
        EmbeddingFrameworkDescriptor(
            name="bad2",
            adapter_module="test.module",
            adapter_class="TestAdapter",
            batch_method="embed",
            query_method="",
        )

    # Test with dotted adapter_class (should warn but not fail)
    with pytest.warns(
        RuntimeWarning, match="adapter_class should be a class name only"
    ):
        EmbeddingFrameworkDescriptor(
            name="warn1",
            adapter_module="test.module",
            adapter_class="some.module.ClassName",
            batch_method="embed",
            query_method="query",
        )

    # Test async query without batch (should warn but not fail)
    with pytest.warns(
        RuntimeWarning,
        match="async_query_method is set but async_batch_method is None",
    ):
        EmbeddingFrameworkDescriptor(
            name="warn2",
            adapter_module="test.module",
            adapter_class="TestAdapter",
            batch_method="embed",
            query_method="query",
            async_query_method="aquery",
            # async_batch_method=None (implicit)
        )


if __name__ == "__main__":
    # Allow running as standalone script
    pytest.main([__file__, "-v"])
