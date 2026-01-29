# tests/frameworks/embedding/test_embedding_registry_self_check.py

import dataclasses
import re
from typing import Any, Optional

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


def _canonicalize_version_range(value: Optional[str]) -> Optional[str]:
    """
    Canonicalize version_range() formatting for stable comparisons.

    Why this exists
    --------------
    Some implementations may include minor whitespace variations (e.g. "<= 2.5.0" vs "<=2.5.0")
    while preserving identical meaning. This helper removes comparator-adjacent whitespace
    so the test remains focused on semantic format correctness rather than spacing trivia.

    Notes
    -----
    - We keep the comma+space convention stable ("..., ...") for readability while
      allowing comparator whitespace normalization.
    - None is preserved as None.
    """
    if value is None:
        return None

    # Normalize any whitespace immediately after comparators (<=, >=).
    s = re.sub(r"(<=|>=)\s+", r"\1", value)

    # Normalize comma spacing to a single ", " for readability and stability.
    s = re.sub(r",\s*", ", ", s)

    return s


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
    Run the descriptor-level validation hook to catch obvious inconsistencies.

    NOTE:
    - Descriptors self-validate in __post_init__ (frozen dataclass).
    - This test is an explicit smoke pass to ensure shipped registry entries
      remain coherent as the registry contract evolves.
    """
    for descriptor in all_descriptors:
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


def test_get_installed_framework_version_does_not_raise(all_descriptors) -> None:
    """
    Ensure get_installed_framework_version() never raises.

    This probe is best-effort (TEST-ONLY): it should return None when unknown or
    when the underlying framework is not installed.
    """
    for descriptor in all_descriptors:
        v = descriptor.get_installed_framework_version()
        assert v is None or (isinstance(v, str) and v.strip()), (
            f"{descriptor.name}: expected None or non-empty version string, got {v!r}"
        )


def test_sample_context_is_dict_when_provided(all_descriptors) -> None:
    """
    Registry quality check: if sample_context is provided, it should be a dict.

    sample_context is used by contract tests to exercise context translation
    deterministically across frameworks.
    """
    for descriptor in all_descriptors:
        if descriptor.sample_context is not None:
            assert isinstance(descriptor.sample_context, dict), (
                f"{descriptor.name}: sample_context must be a dict when provided, "
                f"got {type(descriptor.sample_context).__name__}"
            )


def test_version_range_formatting() -> None:
    """
    Test version_range() returns expected format for various version combinations.

    This test enforces the intended human-readable formatting while allowing minor
    comparator-adjacent whitespace differences (e.g. "<= 2.5.0" vs "<=2.5.0") via
    canonicalization, so we remain focused on correctness and consistency.
    """
    # Test no versions
    desc1 = EmbeddingFrameworkDescriptor(
        name="test1",
        adapter_module="test.module",
        adapter_class="TestAdapter",
        batch_method="embed",
        query_method="query",
    )
    assert _canonicalize_version_range(desc1.version_range()) is None

    # Test minimum version only
    desc2 = EmbeddingFrameworkDescriptor(
        name="test2",
        adapter_module="test.module",
        adapter_class="TestAdapter",
        batch_method="embed",
        query_method="query",
        minimum_framework_version="1.0.0",
    )
    assert _canonicalize_version_range(desc2.version_range()) == _canonicalize_version_range(">=1.0.0")

    # Test maximum version only
    desc3 = EmbeddingFrameworkDescriptor(
        name="test3",
        adapter_module="test.module",
        adapter_class="TestAdapter",
        batch_method="embed",
        query_method="query",
        tested_up_to_version="2.5.0",
    )
    assert _canonicalize_version_range(desc3.version_range()) == _canonicalize_version_range("<=2.5.0")

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
    assert _canonicalize_version_range(desc4.version_range()) == _canonicalize_version_range(">=1.2.0, <=3.0.0")


def test_async_method_consistency(all_descriptors) -> None:
    """
    Check that async support is properly declared.

    Policy:
    - supports_async reflects whether ANY async method field is present.
    - validation requires async to be all-or-nothing: if supports_async is True,
      both async_batch_method and async_query_method must be present.
    """
    for descriptor in all_descriptors:
        if descriptor.supports_async:
            assert descriptor.async_batch_method is not None, (
                f"{descriptor.name}: supports_async=True but async_batch_method is None"
            )
            assert descriptor.async_query_method is not None, (
                f"{descriptor.name}: supports_async=True but async_query_method is None"
            )


def test_register_framework_descriptor() -> None:
    """
    Test dynamic registration functionality.

    This tests the ability to add new framework descriptors at runtime,
    which is useful for testing experimental or third-party adapters.
    """
    original_registry = dict(EMBEDDING_FRAMEWORKS)
    try:
        test_desc = EmbeddingFrameworkDescriptor(
            name="test_framework",
            adapter_module="test.module",
            adapter_class="TestAdapter",
            batch_method="embed",
            query_method="query",
            async_batch_method="aembed",
            async_query_method="aquery",
        )

        assert not has_framework("test_framework")
        assert get_embedding_framework_descriptor_safe("test_framework") is None

        register_framework_descriptor(test_desc)

        assert has_framework("test_framework")
        assert get_embedding_framework_descriptor_safe("test_framework") is test_desc
        assert get_embedding_framework_descriptor("test_framework") is test_desc

        duplicate_desc = EmbeddingFrameworkDescriptor(
            name="test_framework",
            adapter_module="other.module",
            adapter_class="OtherAdapter",
            batch_method="other_embed",
            query_method="other_query",
        )

        with pytest.raises(KeyError, match="already registered"):
            register_framework_descriptor(duplicate_desc, overwrite=False)

        register_framework_descriptor(duplicate_desc, overwrite=True)
        assert get_embedding_framework_descriptor("test_framework") is duplicate_desc

        # Smoke-check cache reset behavior via public probes (never raise).
        assert isinstance(duplicate_desc.is_available(), bool)
        v = duplicate_desc.get_installed_framework_version()
        assert v is None or (isinstance(v, str) and v.strip())
    finally:
        EMBEDDING_FRAMEWORKS.clear()
        EMBEDDING_FRAMEWORKS.update(original_registry)


def test_supports_async_property(all_descriptors) -> None:
    """
    Test the supports_async property logic.

    supports_async reflects whether ANY async method is declared.
    Note: validation enforces that if supports_async is True, both async methods exist.
    """
    for descriptor in all_descriptors:
        has_async = (
            descriptor.async_batch_method is not None
            or descriptor.async_query_method is not None
        )
        assert descriptor.supports_async == has_async, (
            f"{descriptor.name}: supports_async property mismatch"
        )

        # Policy lock: if supports_async is True, both async methods must be present.
        if descriptor.supports_async:
            assert descriptor.async_batch_method is not None
            assert descriptor.async_query_method is not None


def test_get_descriptor_variants() -> None:
    """
    Test both get_descriptor functions behave as expected.

    Verifies that the safe version returns None for unknown frameworks
    while the regular version raises KeyError.
    """
    existing_name = list(EMBEDDING_FRAMEWORKS.keys())[0]

    assert get_embedding_framework_descriptor(existing_name) is not None
    assert get_embedding_framework_descriptor_safe(existing_name) is not None

    non_existent = "non_existent_framework_xyz123"

    with pytest.raises(KeyError, match=non_existent):
        get_embedding_framework_descriptor(non_existent)

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

    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.name = "modified"

    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.batch_method = "modified_embed"


def test_iterator_functions() -> None:
    """
    Test that iterator functions return expected results.
    """
    all_descs = list(iter_embedding_framework_descriptors())
    assert len(all_descs) == len(EMBEDDING_FRAMEWORKS)

    for desc in all_descs:
        assert desc.name in EMBEDDING_FRAMEWORKS

    available_descs = list(iter_available_framework_descriptors())
    assert len(available_descs) <= len(all_descs)

    for desc in available_descs:
        assert desc.is_available()


def test_descriptor_validation_edge_cases() -> None:
    """
    Test descriptor validation with edge cases.
    """
    # Missing required methods => ValueError
    with pytest.raises(ValueError, match="batch_method and query_method must both be set"):
        EmbeddingFrameworkDescriptor(
            name="bad1",
            adapter_module="test.module",
            adapter_class="TestAdapter",
            batch_method="",
            query_method="query",
        )

    with pytest.raises(ValueError, match="batch_method and query_method must both be set"):
        EmbeddingFrameworkDescriptor(
            name="bad2",
            adapter_module="test.module",
            adapter_class="TestAdapter",
            batch_method="embed",
            query_method="",
        )

    # Dotted adapter_class => warn (non-fatal)
    with pytest.warns(RuntimeWarning, match="adapter_class should be a class name only"):
        EmbeddingFrameworkDescriptor(
            name="warn1",
            adapter_module="test.module",
            adapter_class="some.module.ClassName",
            batch_method="embed",
            query_method="query",
        )

    # Partial async declaration => invalid under current registry policy (fatal)
    with pytest.raises(ValueError, match="requires both async_batch_method and async_query_method"):
        EmbeddingFrameworkDescriptor(
            name="bad_async_partial",
            adapter_module="test.module",
            adapter_class="TestAdapter",
            batch_method="embed",
            query_method="query",
            async_query_method="aquery",
            # async_batch_method intentionally omitted
        )


def test_descriptor_validation_new_field_edge_cases() -> None:
    """
    Validate new descriptor fields and their registry-level constraints/warnings.
    """
    # requires_embedding_dimension=True without embedding_dimension_kwarg => warn (tests may not know kwarg name)
    with pytest.warns(RuntimeWarning, match="embedding_dimension_kwarg"):
        EmbeddingFrameworkDescriptor(
            name="dimwarn",
            adapter_module="test.module",
            adapter_class="TestAdapter",
            batch_method="embed",
            query_method="query",
            requires_embedding_dimension=True,
            # embedding_dimension_kwarg intentionally omitted to trigger warning
        )

    # Aliases must be a dict if provided => ValueError
    with pytest.raises(ValueError, match="aliases must be a dict"):
        EmbeddingFrameworkDescriptor(
            name="bad_aliases_type",
            adapter_module="test.module",
            adapter_class="TestAdapter",
            batch_method="embed",
            query_method="query",
            aliases=["not", "a", "dict"],  # type: ignore[arg-type]
        )

    # Aliases keys/values must be non-empty strings => ValueError
    with pytest.raises(ValueError, match="aliases keys and values must be non-empty strings"):
        EmbeddingFrameworkDescriptor(
            name="bad_aliases_empty",
            adapter_module="test.module",
            adapter_class="TestAdapter",
            batch_method="embed",
            query_method="query",
            aliases={"": "embed_documents"},
        )

    with pytest.raises(ValueError, match="aliases keys and values must be non-empty strings"):
        EmbeddingFrameworkDescriptor(
            name="bad_aliases_empty_value",
            adapter_module="test.module",
            adapter_class="TestAdapter",
            batch_method="embed",
            query_method="query",
            aliases={"embed_documents": ""},
        )

    # Self-mapping alias => warn (non-fatal)
    with pytest.warns(RuntimeWarning, match="self-mapping"):
        EmbeddingFrameworkDescriptor(
            name="warn_alias_self",
            adapter_module="test.module",
            adapter_class="TestAdapter",
            batch_method="embed",
            query_method="query",
            aliases={"embed_documents": "embed_documents"},
        )

    # has_capabilities=True but method names not fully set => warn (non-fatal)
    with pytest.warns(RuntimeWarning, match="has_capabilities=True but capabilities_method"):
        EmbeddingFrameworkDescriptor(
            name="warn_caps_methods",
            adapter_module="test.module",
            adapter_class="TestAdapter",
            batch_method="embed",
            query_method="query",
            has_capabilities=True,
            # capabilities_method / async_capabilities_method intentionally omitted
        )

    # has_health=True but method names not fully set => warn (non-fatal)
    with pytest.warns(RuntimeWarning, match="has_health=True but health_method"):
        EmbeddingFrameworkDescriptor(
            name="warn_health_methods",
            adapter_module="test.module",
            adapter_class="TestAdapter",
            batch_method="embed",
            query_method="query",
            has_health=True,
            # health_method / async_health_method intentionally omitted
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
