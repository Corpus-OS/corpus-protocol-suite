# tests/frameworks/llm/test_llm_registry_self_check.py

import dataclasses
import importlib
import re
from typing import Optional
import warnings

# Suppress expected warnings before importing registry.
#
# IMPORTANT:
# Some frameworks are intentionally async-first (e.g. semantic_kernel) and may
# not provide sync completion methods. The registry's validation emits a soft
# warning for that case. We treat this as expected in the conformance suite.
warnings.filterwarnings(
    "ignore",
    message=r"semantic_kernel.*async_completion_method is set but completion_method is None",
    category=RuntimeWarning,
)

import pytest

from tests.frameworks.registries.llm_registry import (
    LLM_FRAMEWORKS,
    LLMFrameworkDescriptor,
    iter_llm_framework_descriptors,
    register_llm_framework_descriptor,
    has_llm_framework,
    get_llm_framework_descriptor,
    get_llm_framework_descriptor_safe,
    iter_available_llm_framework_descriptors,
)

# Filter expected warnings for async-only frameworks and controlled overwrite warnings.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:semantic_kernel.*async_completion_method is set but completion_method is None:RuntimeWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:Framework.*is being overwritten in the LLM registry:RuntimeWarning"
    ),
]


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


@pytest.fixture
def all_descriptors():
    """Fixture providing all registered LLM framework descriptors."""
    return list(iter_llm_framework_descriptors())


def test_llm_registry_keys_match_descriptor_name(all_descriptors) -> None:
    """
    Sanity check: registry keys should always match descriptor.name.

    This keeps lookups and reporting consistent and prevents copy/paste errors
    when adding new LLM frameworks.
    """
    for descriptor in all_descriptors:
        assert descriptor.name in LLM_FRAMEWORKS
        assert LLM_FRAMEWORKS[descriptor.name] is descriptor


def test_llm_registry_descriptors_validate_cleanly(all_descriptors) -> None:
    """
    Run the descriptor-level validation hook to catch obvious inconsistencies
    (e.g. async streaming defined without async completion).

    NOTE:
        validate() may emit warnings for async-only frameworks; those are filtered
        above and should not fail this test.
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
        adapter_class="TestLLMClient",
        completion_method="create",
        streaming_style="none",  # Avoid warning about missing streaming methods
    )

    # No versions
    desc1 = LLMFrameworkDescriptor(
        name="test1",
        **base_kwargs,
    )
    assert desc1.version_range() is None

    # Minimum version only
    desc2 = LLMFrameworkDescriptor(
        name="test2",
        minimum_framework_version="1.0.0",
        **base_kwargs,
    )
    assert _normalize_version_range(desc2.version_range()) == ">=1.0.0"

    # Maximum version only
    desc3 = LLMFrameworkDescriptor(
        name="test3",
        tested_up_to_version="2.5.0",
        **base_kwargs,
    )
    assert _normalize_version_range(desc3.version_range()) == "<=2.5.0"

    # Both versions
    desc4 = LLMFrameworkDescriptor(
        name="test4",
        minimum_framework_version="1.2.0",
        tested_up_to_version="3.0.0",
        **base_kwargs,
    )
    assert _normalize_version_range(desc4.version_range()) == ">=1.2.0, <=3.0.0"


def test_async_method_consistency(all_descriptors) -> None:
    """
    Check that async support is properly declared.

    Policy: if a framework declares async support, it must at least expose an
    async completion method (async streaming is optional).

    NOTE:
        This matches the registry contract and the self-check expectations:
        supports_async is strictly about async completion/stream surfaces.
    """
    for descriptor in all_descriptors:
        if descriptor.supports_async:
            assert descriptor.async_completion_method is not None, (
                f"{descriptor.name}: supports_async is True but "
                f"async_completion_method is None"
            )


def test_supports_streaming_property(all_descriptors) -> None:
    """
    Test the supports_streaming property logic.

    Ensures the property correctly reflects whether ANY streaming capability
    is declared (sync method, async method, or streaming_kwarg).
    """
    for descriptor in all_descriptors:
        has_streaming = (
            descriptor.streaming_method is not None
            or descriptor.async_streaming_method is not None
            or descriptor.streaming_kwarg is not None
        )
        assert descriptor.supports_streaming == has_streaming, (
            f"{descriptor.name}: supports_streaming property mismatch"
        )


def test_supports_async_property(all_descriptors) -> None:
    """
    Test the supports_async property logic.

    Ensures the property correctly reflects whether ANY async method
    is declared (completion or streaming).
    """
    for descriptor in all_descriptors:
        has_async = (
            descriptor.async_completion_method is not None
            or descriptor.async_streaming_method is not None
        )
        assert descriptor.supports_async == has_async, (
            f"{descriptor.name}: supports_async property mismatch"
        )


def test_supports_token_count_property(all_descriptors) -> None:
    """
    Test the supports_token_count property logic.

    Ensures the property correctly reflects whether a token-counting
    method (sync or async) is declared.
    """
    for descriptor in all_descriptors:
        has_token_count = (
            descriptor.token_count_method is not None
            or descriptor.async_token_count_method is not None
        )
        assert descriptor.supports_token_count == has_token_count, (
            f"{descriptor.name}: supports_token_count property mismatch"
        )


def test_streaming_flag_method_consistency(all_descriptors) -> None:
    """
    Test that supports_streaming aligns with actual streaming capabilities.
    """
    for descriptor in all_descriptors:
        if descriptor.supports_streaming:
            assert (
                descriptor.streaming_method is not None
                or descriptor.async_streaming_method is not None
                or descriptor.streaming_kwarg is not None
            ), (
                f"{descriptor.name}: supports_streaming is True but no streaming "
                f"method or streaming_kwarg is defined"
            )
        else:
            # If flag is False, there should be no streaming affordances.
            assert (
                descriptor.streaming_method is None
                and descriptor.async_streaming_method is None
                and descriptor.streaming_kwarg is None
            ), (
                f"{descriptor.name}: supports_streaming is False but streaming "
                f"methods/kwarg are present"
            )


def test_streaming_style_consistency(all_descriptors) -> None:
    """
    Test that streaming_style is consistent with the underlying methods/kwarg.

    Policy:
    - 'method': at least one of streaming_method or async_streaming_method present
    - 'kwarg': streaming_kwarg present
    - 'none' : no streaming capabilities
    """
    for descriptor in all_descriptors:
        style = descriptor.streaming_style

        if style == "method":
            assert (
                descriptor.streaming_method is not None
                or descriptor.async_streaming_method is not None
            ), (
                f"{descriptor.name}: streaming_style='method' but no streaming "
                f"methods are defined"
            )
        elif style == "kwarg":
            assert descriptor.streaming_kwarg is not None, (
                f"{descriptor.name}: streaming_style='kwarg' but streaming_kwarg "
                f"is None"
            )
        elif style == "none":
            assert (
                descriptor.streaming_method is None
                and descriptor.async_streaming_method is None
                and descriptor.streaming_kwarg is None
            ), (
                f"{descriptor.name}: streaming_style='none' but streaming "
                f"capabilities are present"
            )
        else:
            pytest.fail(
                f"{descriptor.name}: unexpected streaming_style value {style!r}"
            )


def test_registry_declared_surfaces_exist_when_available(all_descriptors) -> None:
    """
    Ensure that registry-declared adapter surfaces exist on the adapter class.

    This is a TEST-ONLY drift detector that catches:
        - adapter_module refactors (module path changes)
        - adapter_class renames
        - method renames (completion/stream/token_count)

    IMPORTANT:
        This test is intentionally best-effort and environment-safe:
        - If a framework is not available (descriptor.is_available() is False),
          we do not require importing or introspecting its adapter class.
        - If importing the adapter module fails despite is_available() being True,
          we treat it as a hard failure because it indicates registry drift or
          a broken adapter module.
    """
    for descriptor in all_descriptors:
        # If the underlying framework is not installed/available, we do not enforce
        # class/method presence here. Other conformance layers may handle skipping.
        if not descriptor.is_available():
            continue

        try:
            module = importlib.import_module(descriptor.adapter_module)
        except ImportError as e:
            pytest.fail(
                f"{descriptor.name}: adapter_module {descriptor.adapter_module!r} "
                f"could not be imported even though is_available() is True: {e}"
            )

        cls = getattr(module, descriptor.adapter_class, None)
        assert cls is not None, (
            f"{descriptor.name}: adapter_class {descriptor.adapter_class!r} "
            f"not found in module {descriptor.adapter_module!r}"
        )

        # Helper to assert a named attribute exists and is callable.
        def _assert_callable(method_name: Optional[str], *, label: str) -> None:
            if not method_name:
                return
            attr = getattr(cls, method_name, None)
            assert attr is not None, (
                f"{descriptor.name}: declared {label} {method_name!r} is missing on "
                f"class {descriptor.adapter_class!r}"
            )
            assert callable(attr), (
                f"{descriptor.name}: declared {label} {method_name!r} exists but is not callable "
                f"on class {descriptor.adapter_class!r}"
            )

        _assert_callable(descriptor.completion_method, label="completion_method")
        _assert_callable(descriptor.async_completion_method, label="async_completion_method")

        # Streaming surfaces (method style) are validated by declared method presence.
        _assert_callable(descriptor.streaming_method, label="streaming_method")
        _assert_callable(descriptor.async_streaming_method, label="async_streaming_method")

        # Token counting surfaces are validated by declared method presence.
        _assert_callable(descriptor.token_count_method, label="token_count_method")
        _assert_callable(descriptor.async_token_count_method, label="async_token_count_method")

        # Streaming kwarg style cannot be validated fully without signature inspection.
        # Here we simply ensure the completion methods exist when kwarg streaming is selected.
        if descriptor.streaming_style == "kwarg":
            assert descriptor.streaming_kwarg is not None, (
                f"{descriptor.name}: streaming_style='kwarg' requires streaming_kwarg"
            )
            assert (
                descriptor.completion_method is not None
                or descriptor.async_completion_method is not None
            ), (
                f"{descriptor.name}: streaming_style='kwarg' requires a completion method"
            )


def test_register_llm_framework_descriptor() -> None:
    """
    Test dynamic registration functionality for LLM frameworks.

    This tests the ability to add new framework descriptors at runtime,
    which is useful for testing experimental or third-party LLM adapters.
    """
    original_registry = dict(LLM_FRAMEWORKS)
    try:
        base_kwargs = dict(
            adapter_module="test.module",
            adapter_class="TestLLMClient",
            completion_method="create",
            async_completion_method="acreate",
            streaming_method="create_stream",
            async_streaming_method="acreate_stream",
            token_count_method="count_tokens",
            async_token_count_method="acount_tokens",
        )

        # Create a test descriptor
        test_desc = LLMFrameworkDescriptor(
            name="test_framework",
            streaming_style="method",  # Explicitly set since we have streaming methods
            **base_kwargs,
        )

        # Should not exist initially
        assert not has_llm_framework("test_framework")
        assert get_llm_framework_descriptor_safe("test_framework") is None

        # Registration without overwrite
        register_llm_framework_descriptor(test_desc)

        # Should now exist
        assert has_llm_framework("test_framework")
        assert get_llm_framework_descriptor_safe("test_framework") is test_desc
        assert get_llm_framework_descriptor("test_framework") is test_desc

        # Register duplicate without overwrite should fail
        duplicate_desc = LLMFrameworkDescriptor(
            name="test_framework",
            adapter_module="other.module",
            adapter_class="OtherLLMClient",
            completion_method="other_create",
            streaming_style="none",  # No streaming for this one
        )

        with pytest.raises(KeyError, match="already registered"):
            register_llm_framework_descriptor(duplicate_desc, overwrite=False)

        # Overwrite should succeed
        register_llm_framework_descriptor(duplicate_desc, overwrite=True)
        assert get_llm_framework_descriptor("test_framework") is duplicate_desc
    finally:
        LLM_FRAMEWORKS.clear()
        LLM_FRAMEWORKS.update(original_registry)


def test_get_descriptor_variants() -> None:
    """
    Test both get_descriptor functions behave as expected.

    Verifies that the safe version returns None for unknown frameworks
    while the regular version raises KeyError.
    """
    existing_name = list(LLM_FRAMEWORKS.keys())[0]

    # Existing framework
    assert get_llm_framework_descriptor(existing_name) is not None
    assert get_llm_framework_descriptor_safe(existing_name) is not None

    # Non-existent framework
    non_existent = "non_existent_llm_framework_xyz123"

    with pytest.raises(KeyError, match=re.escape(non_existent)):
        get_llm_framework_descriptor(non_existent)

    assert get_llm_framework_descriptor_safe(non_existent) is None


def test_descriptor_immutability() -> None:
    """
    Test that LLM framework descriptors are immutable (frozen dataclass).
    """
    descriptor = LLMFrameworkDescriptor(
        name="test",
        adapter_module="test.module",
        adapter_class="TestLLMClient",
        completion_method="create",
        streaming_style="none",  # Avoid warning
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.name = "modified"

    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.completion_method = "other_create"


def test_iterator_functions() -> None:
    """
    Test that iterator functions return expected results.
    """
    # iter_llm_framework_descriptors
    all_descs = list(iter_llm_framework_descriptors())
    assert len(all_descs) == len(LLM_FRAMEWORKS)

    for desc in all_descs:
        assert desc.name in LLM_FRAMEWORKS

    # iter_available_llm_framework_descriptors
    available_descs = list(iter_available_llm_framework_descriptors())

    assert len(available_descs) <= len(all_descs)

    for desc in available_descs:
        assert desc.is_available()


def test_descriptor_validation_edge_cases() -> None:
    """
    Test descriptor validation with edge cases.
    """
    base_kwargs = dict(
        adapter_module="test.module",
        adapter_class="TestLLMClient",
    )

    # Missing required completion methods
    with pytest.raises(
        ValueError,
        match=r"completion_method or async_completion_method must be set",
    ):
        LLMFrameworkDescriptor(
            name="bad1",
            completion_method=None,
            async_completion_method=None,
            **base_kwargs,
        )

    # Dotted adapter_class (should warn but not fail)
    with pytest.warns(
        RuntimeWarning,
        match=r"adapter_class should be a class name only",
    ):
        LLMFrameworkDescriptor(
            name="warn1",
            adapter_module="test.module",
            adapter_class="some.module.ClientClass",
            completion_method="create",
            streaming_style="none",  # Avoid extra warning
        )

    # Async streaming without async completion (should warn but not fail)
    with pytest.warns(
        RuntimeWarning,
        match=r"async_streaming_method is set but async_completion_method is None",
    ):
        LLMFrameworkDescriptor(
            name="warn2",
            adapter_module="test.module",
            adapter_class="TestLLMClient",
            completion_method="create",
            async_streaming_method="astream",
            streaming_style="method",  # Explicitly set for clarity
            # async_completion_method=None (implicit)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
