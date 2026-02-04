# tests/frameworks/vector/test_vector_registry_self_check.py

import dataclasses
import types
from typing import Optional

import pytest

from tests.frameworks.registries.vector_registry import (
    VECTOR_FRAMEWORKS,
    VectorFrameworkDescriptor,
    get_vector_framework_descriptor,
    get_vector_framework_descriptor_safe,
    has_vector_framework,
    iter_available_vector_framework_descriptors,
    iter_vector_framework_descriptors,
    register_vector_framework_descriptor,
    unregister_vector_framework_descriptor,
)

# We intentionally reach into the module internals here for cache behavior tests.
from tests.frameworks.registries.vector_registry import (  # type: ignore
    Version,
    _AVAILABILITY_CACHE,
    _availability_cache_key,
)


@pytest.fixture
def all_descriptors():
    """Fixture providing all registered vector framework descriptors."""
    return list(iter_vector_framework_descriptors())


# =============================================================================
# Core registry invariants
# =============================================================================


def test_vector_registry_keys_match_descriptor_name(all_descriptors) -> None:
    """
    Sanity check: registry keys should always match descriptor.name.

    This keeps lookups and reporting consistent and prevents copy/paste errors
    when adding new vector frameworks.
    """
    for descriptor in all_descriptors:
        assert descriptor.name in VECTOR_FRAMEWORKS
        assert VECTOR_FRAMEWORKS[descriptor.name] is descriptor


def test_iter_vector_framework_descriptors_returns_all() -> None:
    """
    iter_vector_framework_descriptors should return all registered descriptors.

    We validate both length and membership identity (not just equality) to ensure
    the iterator yields the concrete objects stored in the registry.
    """
    all_descs = list(iter_vector_framework_descriptors())
    assert len(all_descs) == len(VECTOR_FRAMEWORKS)

    for desc in all_descs:
        assert VECTOR_FRAMEWORKS[desc.name] is desc


def test_get_descriptor_variants() -> None:
    """
    Test both get_descriptor functions behave as expected.

    Verifies that the safe version returns None for unknown frameworks
    while the regular version raises KeyError.
    """
    existing_name = list(VECTOR_FRAMEWORKS.keys())[0]

    # Existing
    assert get_vector_framework_descriptor(existing_name) is VECTOR_FRAMEWORKS[existing_name]
    assert get_vector_framework_descriptor_safe(existing_name) is VECTOR_FRAMEWORKS[existing_name]

    # Non-existent
    non_existent = "non_existent_vector_framework_xyz123"

    with pytest.raises(KeyError, match=non_existent):
        get_vector_framework_descriptor(non_existent)

    assert get_vector_framework_descriptor_safe(non_existent) is None


def test_has_vector_framework_basic() -> None:
    """
    Sanity check for has_vector_framework.

    Ensures it returns True for all registered keys and False for an unknown key.
    """
    for key in VECTOR_FRAMEWORKS.keys():
        assert has_vector_framework(key)

    assert not has_vector_framework("definitely_not_a_framework_xyz123")


# =============================================================================
# Descriptor validation + immutability
# =============================================================================


def test_registered_descriptors_validate_cleanly(all_descriptors) -> None:
    """
    Smoke test: all descriptors currently registered in VECTOR_FRAMEWORKS
    should validate without raising.

    validate() may emit warnings, but the registered descriptors should be
    internally consistent as stored.
    """
    for descriptor in all_descriptors:
        descriptor.validate()


def test_descriptor_immutability_frozen_dataclass() -> None:
    """
    VectorFrameworkDescriptor is a frozen dataclass.

    This guarantees conformance tests can't accidentally mutate registry metadata
    in-place, which would produce order-dependent flakiness.
    """
    descriptor = VectorFrameworkDescriptor(
        name="test",
        adapter_module="test.module",
        adapter_class="TestVectorClient",
        adapter_init_kwarg="adapter",
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.name = "modified"

    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.adapter_init_kwarg = "modified_adapter_kwarg"


@pytest.mark.parametrize("bad_kwarg", ["", "   ", "\n"])
def test_validate_requires_nonempty_adapter_init_kwarg(bad_kwarg: str) -> None:
    """
    adapter_init_kwarg must always be a non-empty string.

    Tests rely on this field to inject the underlying Corpus vector adapter
    without hardcoding a constructor signature.
    """
    with pytest.raises(ValueError, match="adapter_init_kwarg must be a non-empty string"):
        VectorFrameworkDescriptor(
            name="bad_adapter_kwarg",
            adapter_module="test.module",
            adapter_class="TestVectorClient",
            adapter_init_kwarg=bad_kwarg,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "capabilities_method",
        "query_method",
        "upsert_method",
        "delete_method",
        "health_method",
        "create_namespace_method",
        "delete_namespace_method",
    ],
)
def test_validate_requires_core_protocol_method_names(field_name: str) -> None:
    """
    The wrapper API surface must provide non-empty method names for required protocol
    operations so conformance tests can dynamically call into those methods.
    """
    kwargs = dict(
        name="bad_methods",
        adapter_module="test.module",
        adapter_class="TestVectorClient",
        adapter_init_kwarg="adapter",
    )
    kwargs[field_name] = ""  # type: ignore[index]
    with pytest.raises(ValueError, match=f"{field_name} must be a non-empty string"):
        VectorFrameworkDescriptor(**kwargs)  # type: ignore[arg-type]


def test_validate_batch_query_expectation_requires_method_name() -> None:
    """
    If has_batch_query=True, batch_query_method must be a non-empty string.

    This is an API-shape expectation (not a runtime capability guarantee).
    """
    with pytest.raises(ValueError, match="has_batch_query=True requires batch_query_method to be set"):
        VectorFrameworkDescriptor(
            name="bad_batch_query",
            adapter_module="test.module",
            adapter_class="TestVectorClient",
            adapter_init_kwarg="adapter",
            has_batch_query=True,
            batch_query_method=None,
        )


# =============================================================================
# Soft warning checks (API-shape expectations)
# =============================================================================


def test_validate_warns_on_dotted_adapter_class() -> None:
    """
    adapter_class should be a bare class name (not dotted).

    We only warn (not fail) to avoid breaking tests if someone passes a dotted path,
    but the warning helps catch mistakes early.
    """
    with pytest.warns(RuntimeWarning, match="adapter_class should be a class name only"):
        VectorFrameworkDescriptor(
            name="warn_dotted_class",
            adapter_module="test.module",
            adapter_class="some.module.TestVectorClient",
            adapter_init_kwarg="adapter",
        )


@pytest.mark.parametrize(
    "kwarg_overrides, warning_match",
    [
        (
            {"supports_docstore_injection": True, "docstore_init_kwarg": None},
            "supports_docstore_injection=True but docstore_init_kwarg is None",
        ),
        (
            {"supports_config_injection": True, "config_init_kwarg": None},
            "supports_config_injection=True but config_init_kwarg is None",
        ),
        (
            {"supports_mode_switch": True, "mode_init_kwarg": None},
            "supports_mode_switch=True but mode_init_kwarg is None",
        ),
        (
            {"supports_auto_normalize_toggle": True, "supports_config_injection": False},
            "supports_auto_normalize_toggle=True but supports_config_injection=False",
        ),
    ],
)
def test_validate_warns_on_constructor_knob_expectation_mismatches(
    kwarg_overrides: dict, warning_match: str
) -> None:
    """
    Constructor knob expectations are intentionally warnings, not hard errors.

    These warnings protect the conformance suite from silently losing its ability
    to configure wrapper clients, while still allowing incremental adapter work.
    """
    base = dict(
        name="warn_knobs",
        adapter_module="test.module",
        adapter_class="TestVectorClient",
        adapter_init_kwarg="adapter",
    )
    base.update(kwarg_overrides)

    with pytest.warns(RuntimeWarning, match=warning_match):
        VectorFrameworkDescriptor(**base)  # type: ignore[arg-type]


# =============================================================================
# Versioning behavior
# =============================================================================


@pytest.mark.parametrize(
    "name, minimum, tested_up_to, expected",
    [
        ("test1", None, None, None),
        ("test2", "1.0.0", None, ">=1.0.0"),
        ("test3", None, "2.5.0", "<=2.5.0"),
        ("test4", "1.2.0", "3.0.0", ">=1.2.0, <= 3.0.0"),
    ],
)
def test_version_range_formatting(name: str, minimum: Optional[str], tested_up_to: Optional[str], expected: Optional[str]) -> None:
    """
    version_range() returns the expected human-readable string.

    This is informational today, but is useful for reporting and future conditional
    skipping or expectation adjustment.
    """
    desc = VectorFrameworkDescriptor(
        name=name,
        adapter_module="test.module",
        adapter_class="TestVectorClient",
        adapter_init_kwarg="adapter",
        minimum_framework_version=minimum,
        tested_up_to_version=tested_up_to,
    )
    # Accept either spacing style ("<=2.5.0" vs "<= 2.5.0") for consistency
    # across registries and to avoid brittle formatting-only failures.
    actual = desc.version_range()
    if expected is None:
        assert actual is None
    else:
        assert actual is not None
        assert actual.replace(" ", "") == expected.replace(" ", "")


def test_version_ordering_validation_if_packaging_available() -> None:
    """
    Version ordering is validated only when packaging.Version is available.

    If available, minimum > tested_up_to should raise ValueError.
    """
    if Version is None:
        pytest.skip("packaging is not installed; Version ordering validation is skipped")

    with pytest.raises(ValueError, match="minimum_framework_version.*is greater than tested_up_to_version"):
        VectorFrameworkDescriptor(
            name="bad_version_range",
            adapter_module="test.module",
            adapter_class="TestVectorClient",
            adapter_init_kwarg="adapter",
            minimum_framework_version="2.0.0",
            tested_up_to_version="1.0.0",
        )


# =============================================================================
# Availability checks + caching correctness
# =============================================================================


def test_descriptor_is_available_returns_bool_for_all_registered(all_descriptors) -> None:
    """
    Ensure is_available() doesn't crash for any registered descriptor.

    This is a smoke test that verifies the availability check doesn't raise
    unexpected exceptions when called.
    """
    for descriptor in all_descriptors:
        result = descriptor.is_available()
        assert isinstance(result, bool)


def test_is_available_no_availability_attr_defaults_true_and_does_not_import(monkeypatch) -> None:
    """
    If availability_attr is not set, availability is assumed True and no import occurs.

    This ensures "always-on" adapters don't accidentally incur import-time side effects.
    """
    desc = VectorFrameworkDescriptor(
        name="no_attr",
        adapter_module="some.module",
        adapter_class="TestVectorClient",
        adapter_init_kwarg="adapter",
        availability_attr=None,
    )

    def boom_import(_: str):
        raise AssertionError("import_module should not be called when availability_attr is None")

    from tests.frameworks.registries import vector_registry as reg_module

    monkeypatch.setattr(reg_module.importlib, "import_module", boom_import)
    assert desc.is_available() is True


def test_is_available_import_error_returns_false_when_availability_attr_set(monkeypatch) -> None:
    """
    If availability_attr is set but importing the module fails, the framework is unavailable.
    """
    desc = VectorFrameworkDescriptor(
        name="import_error",
        adapter_module="fake.import_error",
        adapter_class="TestVectorClient",
        adapter_init_kwarg="adapter",
        availability_attr="AVAILABLE",
    )

    from tests.frameworks.registries import vector_registry as reg_module

    def fake_import(_: str):
        raise ImportError("nope")

    monkeypatch.setattr(reg_module.importlib, "import_module", fake_import)
    assert desc.is_available() is False


def test_is_available_missing_attr_warns_and_returns_false(monkeypatch) -> None:
    """
    If the module imports but the availability_attr is missing, we warn and treat unavailable.

    This prevents silent false positives when adapter modules forget to define the flag.
    """
    desc = VectorFrameworkDescriptor(
        name="missing_attr",
        adapter_module="fake.missing_attr",
        adapter_class="TestVectorClient",
        adapter_init_kwarg="adapter",
        availability_attr="AVAILABLE",
    )

    fake_mod = types.SimpleNamespace()  # no AVAILABLE attr present

    from tests.frameworks.registries import vector_registry as reg_module

    def fake_import(_: str):
        return fake_mod

    monkeypatch.setattr(reg_module.importlib, "import_module", fake_import)

    with pytest.warns(RuntimeWarning, match="availability_attr.*not found"):
        assert desc.is_available() is False


def test_availability_cache_keyed_by_adapter_module_and_attr_not_name(monkeypatch) -> None:
    """
    Availability caching must not be keyed only by name.

    Two descriptors with the same name but different adapter_module / availability_attr
    must not share cached availability results.
    """
    from tests.frameworks.registries import vector_registry as reg_module

    # Ensure a clean cache for this test.
    _AVAILABILITY_CACHE.clear()

    mod_a = types.SimpleNamespace(AVAILABLE=True)
    mod_b = types.SimpleNamespace(AVAILABLE=False)

    def fake_import(name: str):
        if name == "fake.a":
            return mod_a
        if name == "fake.b":
            return mod_b
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(reg_module.importlib, "import_module", fake_import)

    desc_a = VectorFrameworkDescriptor(
        name="same_name",
        adapter_module="fake.a",
        adapter_class="A",
        adapter_init_kwarg="adapter",
        availability_attr="AVAILABLE",
    )
    desc_b = VectorFrameworkDescriptor(
        name="same_name",
        adapter_module="fake.b",
        adapter_class="B",
        adapter_init_kwarg="adapter",
        availability_attr="AVAILABLE",
    )

    assert desc_a.is_available() is True
    assert desc_b.is_available() is False

    # Confirm distinct cache entries exist.
    assert _availability_cache_key("fake.a", "AVAILABLE") in _AVAILABILITY_CACHE
    assert _availability_cache_key("fake.b", "AVAILABLE") in _AVAILABILITY_CACHE


def test_register_overwrite_and_unregister_invalidate_availability_cache(monkeypatch) -> None:
    """
    Dynamic overwrite must not leak stale cached availability results.

    This test validates:
      - register caches availability
      - overwrite replaces descriptor and future availability reflects the new identity
      - unregister clears the cache entry for the removed descriptor
    """
    from tests.frameworks.registries import vector_registry as reg_module

    original_registry = dict(VECTOR_FRAMEWORKS)
    original_cache = dict(_AVAILABILITY_CACHE)

    try:
        VECTOR_FRAMEWORKS.clear()
        _AVAILABILITY_CACHE.clear()

        mod_true = types.SimpleNamespace(AVAILABLE=True)
        mod_false = types.SimpleNamespace(AVAILABLE=False)

        def fake_import(name: str):
            if name == "fake.true":
                return mod_true
            if name == "fake.false":
                return mod_false
            raise AssertionError(f"unexpected import: {name}")

        monkeypatch.setattr(reg_module.importlib, "import_module", fake_import)

        # Register A -> available True (cached).
        desc_a = VectorFrameworkDescriptor(
            name="test_framework",
            adapter_module="fake.true",
            adapter_class="ClientA",
            adapter_init_kwarg="adapter",
            availability_attr="AVAILABLE",
        )
        register_vector_framework_descriptor(desc_a)
        assert VECTOR_FRAMEWORKS["test_framework"] is desc_a
        assert desc_a.is_available() is True
        assert _availability_cache_key("fake.true", "AVAILABLE") in _AVAILABILITY_CACHE

        # Overwrite with B -> available False (must not reuse A's cache).
        desc_b = VectorFrameworkDescriptor(
            name="test_framework",
            adapter_module="fake.false",
            adapter_class="ClientB",
            adapter_init_kwarg="adapter",
            availability_attr="AVAILABLE",
        )
        with pytest.warns(RuntimeWarning, match="being overwritten"):
            register_vector_framework_descriptor(desc_b, overwrite=True)
        assert VECTOR_FRAMEWORKS["test_framework"] is desc_b
        assert desc_b.is_available() is False
        assert _availability_cache_key("fake.false", "AVAILABLE") in _AVAILABILITY_CACHE

        # Unregister -> descriptor removed + cache key for the removed descriptor cleared.
        unregister_vector_framework_descriptor("test_framework")
        assert "test_framework" not in VECTOR_FRAMEWORKS
        assert _availability_cache_key("fake.false", "AVAILABLE") not in _AVAILABILITY_CACHE

    finally:
        VECTOR_FRAMEWORKS.clear()
        VECTOR_FRAMEWORKS.update(original_registry)
        _AVAILABILITY_CACHE.clear()
        _AVAILABILITY_CACHE.update(original_cache)


def test_iter_available_vector_framework_descriptors_filters_unavailable(monkeypatch) -> None:
    """
    iter_available_vector_framework_descriptors should yield only descriptors whose
    is_available() returns True.

    We patch importlib.import_module to control availability_attr evaluation.
    """
    from tests.frameworks.registries import vector_registry as reg_module

    original_registry = dict(VECTOR_FRAMEWORKS)
    original_cache = dict(_AVAILABILITY_CACHE)

    try:
        VECTOR_FRAMEWORKS.clear()
        _AVAILABILITY_CACHE.clear()

        desc_true = VectorFrameworkDescriptor(
            name="available_framework",
            adapter_module="fake.true",
            adapter_class="ClientTrue",
            adapter_init_kwarg="adapter",
            availability_attr="AVAILABLE",
        )
        desc_false = VectorFrameworkDescriptor(
            name="unavailable_framework",
            adapter_module="fake.false",
            adapter_class="ClientFalse",
            adapter_init_kwarg="adapter",
            availability_attr="AVAILABLE",
        )
        register_vector_framework_descriptor(desc_true)
        register_vector_framework_descriptor(desc_false)

        mod_true = types.SimpleNamespace(AVAILABLE=True)
        mod_false = types.SimpleNamespace(AVAILABLE=False)

        def fake_import(name: str):
            if name == "fake.true":
                return mod_true
            if name == "fake.false":
                return mod_false
            raise AssertionError(f"unexpected import: {name}")

        monkeypatch.setattr(reg_module.importlib, "import_module", fake_import)

        available = list(iter_available_vector_framework_descriptors())
        assert available == [desc_true]

    finally:
        VECTOR_FRAMEWORKS.clear()
        VECTOR_FRAMEWORKS.update(original_registry)
        _AVAILABILITY_CACHE.clear()
        _AVAILABILITY_CACHE.update(original_cache)
