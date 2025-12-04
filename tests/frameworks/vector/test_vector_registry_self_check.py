# tests/frameworks/vector/test_vector_registry_self_check.py

import dataclasses
import types
import pytest

from tests.frameworks.registries.vector_registry import (
    VECTOR_FRAMEWORKS,
    VectorFrameworkDescriptor,
    iter_vector_framework_descriptors,
    iter_available_vector_framework_descriptors,
    register_vector_framework_descriptor,
    unregister_vector_framework_descriptor,
    has_vector_framework,
    get_vector_framework_descriptor,
    get_vector_framework_descriptor_safe,
)

# We intentionally reach into the module internals here for cache behavior tests.
from tests.frameworks.registries.vector_registry import (  # type: ignore
    _AVAILABILITY_CACHE,
    Version,
)


@pytest.fixture
def all_descriptors():
    """Fixture providing all registered vector framework descriptors."""
    return list(iter_vector_framework_descriptors())


def test_vector_registry_keys_match_descriptor_name(all_descriptors) -> None:
    """
    Sanity check: registry keys should always match descriptor.name.

    This keeps lookups and reporting consistent and prevents copy/paste errors
    when adding new vector frameworks.
    """
    for descriptor in all_descriptors:
        assert descriptor.name in VECTOR_FRAMEWORKS
        assert VECTOR_FRAMEWORKS[descriptor.name] is descriptor


def test_registered_descriptors_validate_cleanly(all_descriptors) -> None:
    """
    Smoke test: all descriptors currently registered in VECTOR_FRAMEWORKS
    should validate without raising.

    This is about the *current registry contents* being internally consistent,
    not about edge-case behavior of VectorFrameworkDescriptor.validate itself.
    """
    for descriptor in all_descriptors:
        # validate() may emit warnings but should not raise
        descriptor.validate()


@pytest.mark.parametrize(
    "name, minimum, tested_up_to, expected",
    [
        ("test1", None, None, None),
        ("test2", "1.0.0", None, ">=1.0.0"),
        ("test3", None, "2.5.0", "<=2.5.0"),
        ("test4", "1.2.0", "3.0.0", ">=1.2.0, <= 3.0.0"),
    ],
)
def test_version_range_formatting(name, minimum, tested_up_to, expected) -> None:
    """
    Test version_range() returns expected format for various version combinations.

    Uses parametrization for concise coverage of the common cases.
    """
    desc = VectorFrameworkDescriptor(
        name=name,
        adapter_module="test.module",
        adapter_class="TestVectorStore",
        add_method="add",
        query_method="query",
        minimum_framework_version=minimum,
        tested_up_to_version=tested_up_to,
    )
    assert desc.version_range() == expected


def test_descriptor_is_available_does_not_raise(all_descriptors) -> None:
    """
    Ensure is_available() doesn't crash for any registered descriptor.

    This is a smoke test that verifies the availability check doesn't raise
    unexpected exceptions (ImportError, AttributeError) when called.
    """
    for descriptor in all_descriptors:
        result = descriptor.is_available()
        assert isinstance(result, bool)


def test_async_method_consistency(all_descriptors) -> None:
    """
    Check that async core vector support is properly declared.

    Policy: if any async surface exists, we expect at least async_add_method
    and async_query_method to be present for API consistency.
    """
    for descriptor in all_descriptors:
        if descriptor.supports_async:
            assert descriptor.async_add_method is not None, (
                f"{descriptor.name}: has async support but async_add_method is None"
            )
            assert descriptor.async_query_method is not None, (
                f"{descriptor.name}: has async support but async_query_method is None"
            )


def test_streaming_support_property(all_descriptors) -> None:
    """
    Test the supports_streaming property/flag logic.

    Ensures the flag correctly reflects whether ANY streaming method
    is declared (sync or async) for the registered descriptors.
    """
    for descriptor in all_descriptors:
        has_streaming = (
            descriptor.stream_query_method is not None
            or descriptor.async_stream_query_method is not None
        )
        assert descriptor.supports_streaming == has_streaming, (
            f"{descriptor.name}: supports_streaming property mismatch"
        )


def test_mmr_support_property(all_descriptors) -> None:
    """
    Test the supports_mmr property/flag logic.

    Ensures the flag correctly reflects whether ANY MMR method
    is declared (sync or async) for the registered descriptors.
    """
    for descriptor in all_descriptors:
        has_mmr = (
            descriptor.mmr_query_method is not None
            or descriptor.async_mmr_query_method is not None
        )
        assert descriptor.supports_mmr == has_mmr, (
            f"{descriptor.name}: supports_mmr property mismatch"
        )


def test_supports_async_property(all_descriptors) -> None:
    """
    Test the supports_async property logic.

    Ensures the property correctly reflects whether ANY async method
    is declared.
    """
    for descriptor in all_descriptors:
        has_async = any(
            [
                descriptor.async_add_method is not None,
                descriptor.async_delete_method is not None,
                descriptor.async_query_method is not None,
                descriptor.async_stream_query_method is not None,
                descriptor.async_mmr_query_method is not None,
            ]
        )
        assert descriptor.supports_async == has_async, (
            f"{descriptor.name}: supports_async property mismatch"
        )


def test_register_and_unregister_vector_framework_descriptor() -> None:
    """
    Test dynamic registration and unregistration functionality for vector frameworks.

    This covers add, overwrite, and removal, as well as availability cache reset.
    """
    original_registry = dict(VECTOR_FRAMEWORKS)
    original_cache = dict(_AVAILABILITY_CACHE)
    try:
        # Create a test descriptor with full async surfaces for consistency.
        test_desc = VectorFrameworkDescriptor(
            name="test_framework",
            adapter_module="test.module",
            adapter_class="TestVectorStore",
            add_method="add",
            async_add_method="aadd",
            delete_method="delete",
            async_delete_method="adelete",
            query_method="query",
            async_query_method="aquery",
            stream_query_method="stream_query",
            async_stream_query_method="astream_query",
            mmr_query_method="mmr_query",
            async_mmr_query_method="ammr_query",
        )

        # Should not exist initially
        assert not has_vector_framework("test_framework")
        assert get_vector_framework_descriptor_safe("test_framework") is None

        # Register without overwrite
        register_vector_framework_descriptor(test_desc)
        assert has_vector_framework("test_framework")
        assert get_vector_framework_descriptor_safe("test_framework") is test_desc
        assert get_vector_framework_descriptor("test_framework") is test_desc

        # Register duplicate without overwrite should fail
        duplicate_desc = VectorFrameworkDescriptor(
            name="test_framework",
            adapter_module="other.module",
            adapter_class="OtherVectorStore",
            add_method="add2",
            query_method="query2",
        )
        with pytest.raises(KeyError, match="already registered"):
            register_vector_framework_descriptor(duplicate_desc, overwrite=False)

        # Overwrite should succeed
        register_vector_framework_descriptor(duplicate_desc, overwrite=True)
        assert get_vector_framework_descriptor("test_framework") is duplicate_desc

        # Unregister should remove the descriptor and clear availability cache
        unregister_vector_framework_descriptor("test_framework")
        assert not has_vector_framework("test_framework")
        assert get_vector_framework_descriptor_safe("test_framework") is None

        # Unregistering missing with ignore_missing=False should raise
        with pytest.raises(KeyError, match="not registered"):
            unregister_vector_framework_descriptor("test_framework", ignore_missing=False)

    finally:
        VECTOR_FRAMEWORKS.clear()
        VECTOR_FRAMEWORKS.update(original_registry)
        _AVAILABILITY_CACHE.clear()
        _AVAILABILITY_CACHE.update(original_cache)


def test_iter_available_vector_framework_descriptors_empty(monkeypatch) -> None:
    """
    Test iter_available_vector_framework_descriptors behavior when no frameworks
    are available (all report is_available() == False).
    """
    from tests.frameworks.registries import vector_registry as reg_module

    original_registry = dict(VECTOR_FRAMEWORKS)
    original_cache = dict(_AVAILABILITY_CACHE)
    try:
        VECTOR_FRAMEWORKS.clear()
        _AVAILABILITY_CACHE.clear()

        # Register a descriptor that depends on an availability_attr, and mock
        # importlib.import_module to make it appear unavailable.
        test_desc = VectorFrameworkDescriptor(
            name="unavailable_framework",
            adapter_module="fake.module",
            adapter_class="FakeVectorStore",
            add_method="add",
            query_method="query",
            availability_attr="AVAILABLE",
        )
        register_vector_framework_descriptor(test_desc)

        fake_mod = types.SimpleNamespace(AVAILABLE=False)

        def fake_import(name):
            assert name == "fake.module"
            return fake_mod

        monkeypatch.setattr(reg_module.importlib, "import_module", fake_import)

        available = list(iter_available_vector_framework_descriptors())
        assert available == []
    finally:
        VECTOR_FRAMEWORKS.clear()
        VECTOR_FRAMEWORKS.update(original_registry)
        _AVAILABILITY_CACHE.clear()
        _AVAILABILITY_CACHE.update(original_cache)


def test_get_descriptor_variants() -> None:
    """
    Test both get_descriptor functions behave as expected.

    Verifies that the safe version returns None for unknown frameworks
    while the regular version raises KeyError.
    """
    existing_name = list(VECTOR_FRAMEWORKS.keys())[0]

    # Existing
    assert get_vector_framework_descriptor(existing_name) is not None
    assert get_vector_framework_descriptor_safe(existing_name) is not None

    # Non-existent
    non_existent = "non_existent_vector_framework_xyz123"

    with pytest.raises(KeyError, match=non_existent):
        get_vector_framework_descriptor(non_existent)

    assert get_vector_framework_descriptor_safe(non_existent) is None


def test_descriptor_immutability() -> None:
    """
    Test that vector descriptors are immutable (frozen dataclass).
    """
    descriptor = VectorFrameworkDescriptor(
        name="test",
        adapter_module="test.module",
        adapter_class="TestVectorStore",
        add_method="add",
        query_method="query",
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.name = "modified"

    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.add_method = "modified_add"


def test_iterator_functions() -> None:
    """
    Test that iterator functions return expected results.
    """
    # iter_vector_framework_descriptors
    all_descs = list(iter_vector_framework_descriptors())
    assert len(all_descs) == len(VECTOR_FRAMEWORKS)

    for desc in all_descs:
        assert desc.name in VECTOR_FRAMEWORKS

    # iter_available_vector_framework_descriptors
    available_descs = list(iter_available_vector_framework_descriptors())
    assert len(available_descs) <= len(all_descs)

    for desc in available_descs:
        assert desc.is_available()


def test_vector_descriptor_validate_edge_cases() -> None:
    """
    Unit tests for VectorFrameworkDescriptor.validate edge cases.

    This focuses on descriptor *behavior* (warnings/errors for bad inputs),
    independent of the concrete registry contents.
    """
    # Missing required methods
    with pytest.raises(
        ValueError, match="add_method and query_method must both be set"
    ):
        VectorFrameworkDescriptor(
            name="bad1",
            adapter_module="test.module",
            adapter_class="TestVectorStore",
            add_method="",
            query_method="query",
        )

    with pytest.raises(
        ValueError, match="add_method and query_method must both be set"
    ):
        VectorFrameworkDescriptor(
            name="bad2",
            adapter_module="test.module",
            adapter_class="TestVectorStore",
            add_method="add",
            query_method="",
        )

    # Dotted adapter_class (should warn but not fail)
    with pytest.warns(
        RuntimeWarning, match="adapter_class should be a class name only"
    ):
        VectorFrameworkDescriptor(
            name="warn1",
            adapter_module="test.module",
            adapter_class="some.module.ClassName",
            add_method="add",
            query_method="query",
        )

    # Async streaming without sync streaming counterpart (should warn)
    with pytest.warns(
        RuntimeWarning,
        match="async_stream_query_method is set but stream_query_method is None",
    ):
        VectorFrameworkDescriptor(
            name="warn2",
            adapter_module="test.module",
            adapter_class="TestVectorStore",
            add_method="add",
            query_method="query",
            async_stream_query_method="astream_query",
        )

    # supports_streaming flag without methods (should warn)
    with pytest.warns(
        RuntimeWarning,
        match="supports_streaming is True but neither stream_query_method nor async_stream_query_method is set",
    ):
        VectorFrameworkDescriptor(
            name="warn3",
            adapter_module="test.module",
            adapter_class="TestVectorStore",
            add_method="add",
            query_method="query",
            supports_streaming=True,
        )

    # supports_mmr flag without methods (should warn)
    with pytest.warns(
        RuntimeWarning,
        match="supports_mmr is True but neither mmr_query_method nor async_mmr_query_method is set",
    ):
        VectorFrameworkDescriptor(
            name="warn4",
            adapter_module="test.module",
            adapter_class="TestVectorStore",
            add_method="add",
            query_method="query",
            supports_mmr=True,
        )

    # Version ordering error only if packaging.Version is available
    if Version is not None:
        with pytest.raises(
            ValueError,
            match="minimum_framework_version '2.0.0' is greater than tested_up_to_version '1.0.0'",
        ):
            VectorFrameworkDescriptor(
                name="bad_version_range",
                adapter_module="test.module",
                adapter_class="TestVectorStore",
                add_method="add",
                query_method="query",
                minimum_framework_version="2.0.0",
                tested_up_to_version="1.0.0",
            )
