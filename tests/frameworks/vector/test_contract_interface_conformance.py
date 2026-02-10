# tests/frameworks/vector/test_contract_interface_conformance.py

from __future__ import annotations

import importlib
import inspect
from dataclasses import replace
from typing import Any, Callable, Optional

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
    _AVAILABILITY_CACHE,
    Version,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_method(instance: Any, name: str | None) -> Callable[..., Any]:
    """
    Fetch a method from the instance and assert it is callable.

    If name is None, fail fast with a clear assertion message.
    """
    assert name, "Expected a non-empty method name"
    attr = getattr(instance, name, None)
    assert callable(attr), f"{instance!r} missing expected callable method {name!r}"
    return attr


def _inject_framework_context_kwarg(
    descriptor: VectorFrameworkDescriptor,
    kwargs: dict[str, Any],
    context: Any,
) -> dict[str, Any]:
    """
    Inject framework-specific context into kwargs *only* when descriptor.context_kwarg is set.

    This helper is intentionally conservative so tests don't leak unexpected kwargs
    into adapters that do not declare a context surface.
    """
    if descriptor.context_kwarg:
        kwargs.setdefault(descriptor.context_kwarg, context)
    return kwargs


def _signature_accepts_kwarg(fn: Callable[..., Any], kw: str) -> bool:
    """
    Return True if the callable signature can accept `kw`, either via an explicit
    parameter name or a **kwargs catch-all.

    This avoids fragile behavioral calls that require constructing full protocol specs.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        # Some callables may not expose a Python signature (builtins/extension).
        # In that case we treat acceptability as unknown and do not fail based
        # purely on missing introspection.
        return True

    params = sig.parameters.values()
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return True
    return kw in sig.parameters


def _import_client_class(desc: VectorFrameworkDescriptor) -> type:
    """
    Import the adapter module and resolve the adapter class with clear errors.

    This makes failures attributable: either registry metadata is wrong,
    or the adapter module/class is missing or broken.
    """
    try:
        module = importlib.import_module(desc.adapter_module)
    except Exception as e:  # noqa: BLE001
        raise AssertionError(
            f"{desc.name}: failed to import adapter_module {desc.adapter_module!r}: {e}"
        ) from e

    try:
        client_cls = getattr(module, desc.adapter_class)
    except Exception as e:  # noqa: BLE001
        raise AssertionError(
            f"{desc.name}: adapter_class {desc.adapter_class!r} not found in module "
            f"{desc.adapter_module!r}: {e}"
        ) from e

    if not isinstance(client_cls, type):
        raise AssertionError(
            f"{desc.name}: resolved adapter_class {desc.adapter_class!r} is not a class"
        )
    return client_cls


def _maybe_construct_client(
    desc: VectorFrameworkDescriptor,
    adapter: Any,
) -> Optional[Any]:
    """
    Construct a concrete vector client/store instance for the given descriptor.

    No skips:
    - If the framework is unavailable per descriptor.is_available(), we return None.
    - If it is available, we require that import + construction succeeds.

    IMPORTANT:
    We must respect descriptor.adapter_init_kwarg (tests must never hardcode "adapter").
    """
    if not desc.is_available():
        # Unavailable frameworks should not be construct-required by contract tests.
        # Returning None lets tests assert availability behavior without skipping.
        return None

    client_cls = _import_client_class(desc)

    init_kwargs: dict[str, Any] = {desc.adapter_init_kwarg: adapter}
    try:
        return client_cls(**init_kwargs)
    except TypeError as e:
        # The most common "contract drift" failure: constructor kwarg mismatch.
        raise AssertionError(
            f"{desc.name}: failed to construct client with adapter_init_kwarg="
            f"{desc.adapter_init_kwarg!r}: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=list(iter_vector_framework_descriptors()), name="framework_descriptor")
def framework_descriptor_fixture(request: pytest.FixtureRequest) -> VectorFrameworkDescriptor:
    """
    Parameterized over all registered vector framework descriptors.

    IMPORTANT: We do not skip unavailable frameworks. Instead, tests assert that:
    - unavailable descriptors are excluded from iter_available_*; and
    - available descriptors can be imported/constructed and expose required surfaces.
    """
    descriptor: VectorFrameworkDescriptor = request.param
    return descriptor


@pytest.fixture
def vector_client_instance(
    framework_descriptor: VectorFrameworkDescriptor,
    adapter: Any,
) -> Optional[Any]:
    """
    Construct a vector client instance when the descriptor reports availability.

    Returns None when the framework is unavailable in the environment.
    """
    return _maybe_construct_client(framework_descriptor, adapter)


# ---------------------------------------------------------------------------
# Core registry integrity tests (no framework imports required)
# ---------------------------------------------------------------------------


def test_registry_keys_match_descriptor_name() -> None:
    """
    Sanity check: registry keys should always match descriptor.name.
    """
    for key, desc in VECTOR_FRAMEWORKS.items():
        assert key == desc.name, f"Registry key {key!r} does not match descriptor.name {desc.name!r}"


def test_get_descriptor_variants_behave_consistently() -> None:
    """
    get_vector_framework_descriptor should raise on unknown keys while the safe
    variant should return None.
    """
    existing_name = next(iter(VECTOR_FRAMEWORKS.keys()))
    assert get_vector_framework_descriptor(existing_name) is VECTOR_FRAMEWORKS[existing_name]
    assert get_vector_framework_descriptor_safe(existing_name) is VECTOR_FRAMEWORKS[existing_name]

    missing = "missing_vector_framework_xyz123"
    with pytest.raises(KeyError):
        _ = get_vector_framework_descriptor(missing)
    assert get_vector_framework_descriptor_safe(missing) is None


def test_has_vector_framework_matches_registry() -> None:
    """
    has_vector_framework must reflect membership in VECTOR_FRAMEWORKS.
    """
    existing_name = next(iter(VECTOR_FRAMEWORKS.keys()))
    assert has_vector_framework(existing_name) is True
    assert has_vector_framework("missing_vector_framework_xyz123") is False


def test_iter_vector_framework_descriptors_covers_registry_values() -> None:
    """
    iter_vector_framework_descriptors must iterate exactly the registered values.
    """
    values = list(iter_vector_framework_descriptors())
    assert len(values) == len(VECTOR_FRAMEWORKS)
    for d in values:
        assert VECTOR_FRAMEWORKS[d.name] is d


def test_descriptor_validate_is_idempotent_for_registered_descriptors() -> None:
    """
    validate() may emit warnings, but should not raise for registered descriptors.
    """
    for desc in iter_vector_framework_descriptors():
        desc.validate()


def test_required_descriptor_fields_are_non_empty_strings() -> None:
    """
    Descriptor required method-name fields must be non-empty strings.
    """
    required = (
        "capabilities_method",
        "query_method",
        "upsert_method",
        "delete_method",
        "create_namespace_method",
        "delete_namespace_method",
        "health_method",
        "adapter_init_kwarg",
        "adapter_module",
        "adapter_class",
        "name",
    )
    for desc in iter_vector_framework_descriptors():
        for field_name in required:
            v = getattr(desc, field_name)
            assert isinstance(v, str) and v.strip(), f"{desc.name}: {field_name} must be a non-empty string"


def test_batch_query_descriptor_coherence() -> None:
    """
    If has_batch_query=True, batch_query_method must be a non-empty string.
    """
    for desc in iter_vector_framework_descriptors():
        if desc.has_batch_query:
            assert desc.batch_query_method is not None and desc.batch_query_method.strip(), (
                f"{desc.name}: has_batch_query=True requires batch_query_method"
            )


def test_constructor_knob_expectations_are_coherent() -> None:
    """
    Registry expectations about constructor knobs should be coherent:
    - If tests expect injection, the corresponding kwarg name should be set.
    - If supports_auto_normalize_toggle=True, supports_config_injection should generally be True.
    """
    for desc in iter_vector_framework_descriptors():
        if desc.supports_docstore_injection:
            assert desc.docstore_init_kwarg, (
                f"{desc.name}: supports_docstore_injection=True requires docstore_init_kwarg"
            )

        if desc.supports_config_injection:
            assert desc.config_init_kwarg, (
                f"{desc.name}: supports_config_injection=True requires config_init_kwarg"
            )

        if desc.supports_mode_switch:
            assert desc.mode_init_kwarg, (
                f"{desc.name}: supports_mode_switch=True requires mode_init_kwarg"
            )

        if desc.supports_auto_normalize_toggle:
            assert desc.supports_config_injection, (
                f"{desc.name}: supports_auto_normalize_toggle=True should imply supports_config_injection=True"
            )


def test_version_range_formatting_smoke() -> None:
    """
    version_range() should produce stable human-readable strings.
    """
    d0 = VectorFrameworkDescriptor(name="t0", adapter_module="m", adapter_class="C")
    assert d0.version_range() is None

    d1 = replace(d0, name="t1", minimum_framework_version="1.0.0")
    assert d1.version_range() == ">=1.0.0"

    d2 = replace(d0, name="t2", tested_up_to_version="2.0.0")
    assert d2.version_range() == "<=2.0.0"

    d3 = replace(d0, name="t3", minimum_framework_version="1.0.0", tested_up_to_version="2.0.0")
    assert d3.version_range() == ">=1.0.0, <= 2.0.0"


def test_dotted_adapter_class_emits_warning_but_does_not_raise() -> None:
    """
    adapter_class should be a bare class name. A dotted path should warn but not fail.
    """
    with pytest.warns(RuntimeWarning, match="adapter_class should be a class name only"):
        _ = VectorFrameworkDescriptor(
            name="warn_dotted",
            adapter_module="m",
            adapter_class="some.module.ClassName",
        )


def test_version_ordering_validation_behaves_as_expected() -> None:
    """
    When both bounds are present:
    - If packaging.Version is available, an inverted range should raise ValueError.
    - If packaging is not available, it should warn (best-effort validation).
    """
    if Version is None:
        with pytest.warns(RuntimeWarning, match="cannot validate version range ordering"):
            _ = VectorFrameworkDescriptor(
                name="no_packaging",
                adapter_module="m",
                adapter_class="C",
                minimum_framework_version="2.0.0",
                tested_up_to_version="1.0.0",
            )
    else:
        with pytest.raises(ValueError, match="minimum_framework_version.*greater than tested_up_to_version"):
            _ = VectorFrameworkDescriptor(
                name="bad_order",
                adapter_module="m",
                adapter_class="C",
                minimum_framework_version="2.0.0",
                tested_up_to_version="1.0.0",
            )


# ---------------------------------------------------------------------------
# Availability behavior tests (cache + iter_available) - no skips
# ---------------------------------------------------------------------------


def test_iter_available_only_includes_descriptors_reporting_available() -> None:
    """
    iter_available_vector_framework_descriptors must be consistent with is_available().

    This ensures availability classification is deterministic and testable.
    """
    available = list(iter_available_vector_framework_descriptors())
    for desc in available:
        assert desc.is_available() is True, f"{desc.name}: iter_available included an unavailable descriptor"

    # Conversely: anything unavailable must not appear in the available iterator.
    available_names = {d.name for d in available}
    for desc in iter_vector_framework_descriptors():
        if not desc.is_available():
            assert desc.name not in available_names, (
                f"{desc.name}: descriptor reports unavailable but appears in iter_available_*"
            )


def test_is_available_caches_result_by_adapter_module_and_attr(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    is_available() should cache results by adapter_module + availability_attr.

    We patch importlib.import_module to count calls and ensure the second call is cached.
    """
    from tests.frameworks.registries import vector_registry as reg_module

    _AVAILABILITY_CACHE.clear()

    calls: list[str] = []

    def fake_import(name: str) -> Any:
        calls.append(name)
        # Provide an object that has an availability flag we can read.
        return type("M", (), {"FLAG": True})()

    monkeypatch.setattr(reg_module.importlib, "import_module", fake_import)

    desc = VectorFrameworkDescriptor(
        name="cache_test",
        adapter_module="fake.module",
        adapter_class="C",
        availability_attr="FLAG",
    )

    assert desc.is_available() is True
    assert desc.is_available() is True  # second call should hit cache
    assert calls.count("fake.module") == 1, "import_module should be called once due to caching"


def test_register_and_unregister_reset_availability_cache() -> None:
    """
    Register/unregister should clear the availability cache for the affected descriptor key.
    """
    original_registry = dict(VECTOR_FRAMEWORKS)
    original_cache = dict(_AVAILABILITY_CACHE)

    try:
        VECTOR_FRAMEWORKS.clear()
        _AVAILABILITY_CACHE.clear()

        desc = VectorFrameworkDescriptor(
            name="dyn",
            adapter_module="dyn.module",
            adapter_class="DynClient",
            availability_attr="FLAG",
        )

        # Seed cache for the descriptor key in the same way is_available would.
        _AVAILABILITY_CACHE["dyn.module:FLAG"] = True

        register_vector_framework_descriptor(desc)
        assert "dyn.module:FLAG" not in _AVAILABILITY_CACHE, "register should clear availability cache for descriptor"

        # Re-seed and ensure unregister also clears.
        _AVAILABILITY_CACHE["dyn.module:FLAG"] = True
        unregister_vector_framework_descriptor("dyn")
        assert "dyn.module:FLAG" not in _AVAILABILITY_CACHE, "unregister should clear availability cache for descriptor"

    finally:
        VECTOR_FRAMEWORKS.clear()
        VECTOR_FRAMEWORKS.update(original_registry)
        _AVAILABILITY_CACHE.clear()
        _AVAILABILITY_CACHE.update(original_cache)


# ---------------------------------------------------------------------------
# Framework adapter surface tests (bidirectional: registry <-> adapter)
# ---------------------------------------------------------------------------


def test_can_import_adapter_module_and_resolve_class_when_available(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Optional[Any],
) -> None:
    """
    Bidirectional flow:
    - If descriptor reports available, we require adapter module import + class resolution + construction.
    - If descriptor reports unavailable, we require that this test does not force construction.
    """
    if framework_descriptor.is_available():
        assert vector_client_instance is not None, (
            f"{framework_descriptor.name}: descriptor reports available but client could not be constructed"
        )
    else:
        assert vector_client_instance is None, (
            f"{framework_descriptor.name}: descriptor reports unavailable but client was constructed"
        )


def test_client_exposes_required_vector_protocol_v1_surface_when_available(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Optional[Any],
) -> None:
    """
    When available, the adapter must expose the required wrapper surface declared in the registry.
    """
    if vector_client_instance is None:
        # Unavailable frameworks are validated via iter_available/is_available tests.
        return

    _get_method(vector_client_instance, framework_descriptor.capabilities_method)
    _get_method(vector_client_instance, framework_descriptor.query_method)
    _get_method(vector_client_instance, framework_descriptor.upsert_method)
    _get_method(vector_client_instance, framework_descriptor.delete_method)
    _get_method(vector_client_instance, framework_descriptor.create_namespace_method)
    _get_method(vector_client_instance, framework_descriptor.delete_namespace_method)
    _get_method(vector_client_instance, framework_descriptor.health_method)


def test_batch_query_method_callable_presence_when_declared_and_available(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Optional[Any],
) -> None:
    """
    If has_batch_query=True and the framework is available, the adapter surface must include a
    callable batch_query method.

    This is a pure surface test: runtime behavior may still raise NotSupported depending
    on capabilities.supports_batch_queries, but the wrapper-level surface must exist.
    """
    if vector_client_instance is None:
        return

    if framework_descriptor.has_batch_query:
        _get_method(vector_client_instance, framework_descriptor.batch_query_method)


def test_context_kwarg_signature_acceptance_when_declared_and_available(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Optional[Any],
) -> None:
    """
    If a framework declares context_kwarg and is available, the major surfaces should accept it
    (either explicitly or via **kwargs).

    We validate via signature introspection to avoid requiring protocol-spec objects.
    """
    if vector_client_instance is None:
        return

    if not framework_descriptor.context_kwarg:
        return

    ctx_kw = framework_descriptor.context_kwarg

    for method_name in (
        framework_descriptor.query_method,
        framework_descriptor.upsert_method,
        framework_descriptor.delete_method,
        framework_descriptor.health_method,
        framework_descriptor.capabilities_method,
    ):
        fn = _get_method(vector_client_instance, method_name)
        assert _signature_accepts_kwarg(fn, ctx_kw), (
            f"{framework_descriptor.name}: method {method_name!r} does not appear to accept "
            f"declared context_kwarg {ctx_kw!r} (no parameter and no **kwargs)"
        )


# ---------------------------------------------------------------------------
# Pure unit tests for helper behavior and adapter_init_kwarg handling
# ---------------------------------------------------------------------------


def test_context_injection_does_not_occur_when_context_kwarg_is_none() -> None:
    """
    Ensure our helper does not leak unexpected kwargs when descriptor.context_kwarg is None.
    """
    desc = VectorFrameworkDescriptor(
        name="no_context",
        adapter_module="test.module",
        adapter_class="TestClient",
        context_kwarg=None,
    )

    kwargs: dict[str, Any] = {"existing": 1}
    out = _inject_framework_context_kwarg(desc, kwargs, context={"k": "v"})

    assert out is kwargs
    assert out == {"existing": 1}


def test_context_injection_occurs_when_context_kwarg_is_set() -> None:
    """
    Ensure our helper injects the declared context kwarg when descriptor.context_kwarg is set.
    """
    desc = VectorFrameworkDescriptor(
        name="with_context",
        adapter_module="test.module",
        adapter_class="TestClient",
        context_kwarg="ctx",
    )

    kwargs: dict[str, Any] = {"existing": 1}
    out = _inject_framework_context_kwarg(desc, kwargs, context={"k": "v"})

    assert out is kwargs  # helper mutates in-place by design
    assert out["existing"] == 1
    assert out["ctx"] == {"k": "v"}


def test_adapter_init_kwarg_is_respected_with_nonstandard_kwarg() -> None:
    """
    Ensure adapter_init_kwarg is honored by construction logic.

    This is a pure unit test: we do not import any real framework adapter. Instead,
    we synthesize a client class that only accepts a nonstandard kwarg, and ensure
    our construction approach uses descriptor.adapter_init_kwarg.
    """

    class SyntheticClient:
        def __init__(self, *, injected_adapter: Any) -> None:
            self.injected_adapter = injected_adapter

    adapter = object()

    desc = VectorFrameworkDescriptor(
        name="synthetic",
        adapter_module="synthetic.module",
        adapter_class="SyntheticClient",
        adapter_init_kwarg="injected_adapter",
    )

    init_kwargs: dict[str, Any] = {desc.adapter_init_kwarg: adapter}
    instance = SyntheticClient(**init_kwargs)
    assert instance.injected_adapter is adapter


def test_register_overwrite_emits_warning_and_updates_registry() -> None:
    """
    Overwriting an existing registry entry should emit a warning and update the mapping.
    """
    original_registry = dict(VECTOR_FRAMEWORKS)
    original_cache = dict(_AVAILABILITY_CACHE)

    try:
        VECTOR_FRAMEWORKS.clear()
        _AVAILABILITY_CACHE.clear()

        d1 = VectorFrameworkDescriptor(name="x", adapter_module="m1", adapter_class="C1")
        d2 = VectorFrameworkDescriptor(name="x", adapter_module="m2", adapter_class="C2")

        register_vector_framework_descriptor(d1, overwrite=False)
        assert VECTOR_FRAMEWORKS["x"] is d1

        with pytest.warns(RuntimeWarning, match="being overwritten"):
            register_vector_framework_descriptor(d2, overwrite=True)

        assert VECTOR_FRAMEWORKS["x"] is d2

    finally:
        VECTOR_FRAMEWORKS.clear()
        VECTOR_FRAMEWORKS.update(original_registry)
        _AVAILABILITY_CACHE.clear()
        _AVAILABILITY_CACHE.update(original_cache)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
