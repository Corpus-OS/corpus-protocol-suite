# tests/frameworks/registries/vector_registry.py
"""
Registry of vector framework adapters used by the conformance test suite.

This module is TEST-ONLY. It provides lightweight metadata describing how to:
- Import each vector framework adapter
- Construct its client (including injected underlying Corpus vector adapter)
- Call its async vector protocol methods (capabilities, query, batch_query, upsert, delete, namespace ops, health)
- Pass framework-specific context (if applicable)
- Know which wrapper-level surfaces should exist so conformance tests can be strict
  without hardcoding framework specifics.

Contract tests in tests/frameworks/vector/ use this registry to stay
framework-agnostic. Adding a new vector framework typically means:

1. Implement the adapter under corpus_sdk.vector.framework_adapters.*.
2. Add a new VectorFrameworkDescriptor entry here (or register dynamically).
3. Run the vector contract tests.

Version fields
--------------
`minimum_framework_version` and `tested_up_to_version` are currently informational,
but we validate that the range is coherent when possible. In the future, tests may
use them to conditionally skip or adjust expectations based on the installed
framework version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import importlib
import warnings

try:  # Optional, used only for version ordering validation
    from packaging.version import Version
except Exception:  # pragma: no cover - packaging may not be installed
    Version = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Availability cache
# ---------------------------------------------------------------------------
# IMPORTANT:
# Cache key must be stable across descriptor overwrites. Caching by name can
# produce stale results when tests overwrite registry entries under the same name.
# We therefore cache on a composite key derived from adapter_module + availability_attr.
_AVAILABILITY_CACHE: Dict[str, bool] = {}


def _availability_cache_key(adapter_module: str, availability_attr: Optional[str]) -> str:
    return f"{adapter_module}:{availability_attr or ''}"


@dataclass(frozen=True)
class VectorFrameworkDescriptor:
    """
    Description of a vector framework adapter (TEST-ONLY).

    Identity / import
    -----------------
    name:
        Short, stable identifier (e.g. "autogen", "langchain").
    adapter_module:
        Dotted import path for the adapter module.
    adapter_class:
        Name of the adapter class within adapter_module (bare class name).

    Constructor wiring
    ------------------
    adapter_init_kwarg:
        Name of the constructor kwarg used to inject the underlying Corpus vector adapter.
        Tests MUST use this field rather than hardcoding `adapter=...`.

    docstore_init_kwarg:
        Optional constructor kwarg used to inject a docstore (if wrapper supports injection).
    config_init_kwarg:
        Optional constructor kwarg used to inject VectorAdapterConfig (if wrapper supports it).
    mode_init_kwarg:
        Optional constructor kwarg used to set adapter mode ("thin"/"standalone") if wrapper supports it.

    Async protocol surface (VectorProtocolV1)
    -----------------------------------------
    capabilities_method, query_method, batch_query_method, upsert_method, delete_method,
    create_namespace_method, delete_namespace_method, health_method:
        Method names on the wrapper/adapter instance.

    Context injection
    -----------------
    context_kwarg:
        Name of the kwargs parameter used for framework-specific context, if any.

    Wrapper-level surface expectations (what tests should assert exists)
    -------------------------------------------------------------------
    has_batch_query:
        True if the adapter surface includes batch_query method (even if it may raise NotSupported at runtime
        depending on capabilities.supports_batch_queries).

    supports_wire_handler:
        True if conformance tests should validate WireVectorHandler envelope behavior against this adapter.

    supports_docstore_injection / supports_config_injection / supports_mode_switch / supports_auto_normalize_toggle:
        Whether the wrapper constructor is expected to accept those knobs (the kwarg names are provided above).
        These are *test expectations* about the wrapper API shape, not backend capability flags.
        Backend capability flags live in VectorCapabilities returned by capabilities().

    Availability + versions
    -----------------------
    availability_attr:
        Optional module-level boolean that indicates whether the underlying framework is installed.
    minimum_framework_version / tested_up_to_version:
        Informational bounds with best-effort ordering validation.
    """

    name: str
    adapter_module: str
    adapter_class: str

    # Constructor injection kwarg for the underlying Corpus vector adapter.
    adapter_init_kwarg: str = "adapter"

    # Optional constructor kwargs for additional knobs.
    docstore_init_kwarg: Optional[str] = None
    config_init_kwarg: Optional[str] = None
    mode_init_kwarg: Optional[str] = None

    # VectorProtocolV1 surface (async-first).
    capabilities_method: str = "capabilities"
    query_method: str = "query"
    batch_query_method: Optional[str] = "batch_query"
    upsert_method: str = "upsert"
    delete_method: str = "delete"
    create_namespace_method: str = "create_namespace"
    delete_namespace_method: str = "delete_namespace"
    health_method: str = "health"

    # Framework-specific context kwarg (optional).
    context_kwarg: Optional[str] = None

    # Wrapper-level expectations for tests.
    has_batch_query: bool = True
    supports_wire_handler: bool = True

    supports_docstore_injection: bool = False
    supports_config_injection: bool = False
    supports_mode_switch: bool = False
    supports_auto_normalize_toggle: bool = False

    availability_attr: Optional[str] = None
    minimum_framework_version: Optional[str] = None
    tested_up_to_version: Optional[str] = None

    def __post_init__(self) -> None:
        self.validate()

    def is_available(self) -> bool:
        """
        Check if the underlying framework appears available for testing.

        If availability_attr is set, this checks that boolean on the adapter module.
        Otherwise assumes the framework is available.

        Results are cached using adapter_module + availability_attr to avoid stale results
        when descriptors are overwritten in tests.
        """
        cache_key = _availability_cache_key(self.adapter_module, self.availability_attr)
        if cache_key in _AVAILABILITY_CACHE:
            return _AVAILABILITY_CACHE[cache_key]

        if not self.availability_attr:
            _AVAILABILITY_CACHE[cache_key] = True
            return True

        try:
            module = importlib.import_module(self.adapter_module)
        except ImportError:
            _AVAILABILITY_CACHE[cache_key] = False
            return False

        attr_value = getattr(module, self.availability_attr, None)
        if attr_value is None:
            warnings.warn(
                f"{self.name}: availability_attr {self.availability_attr!r} not found on "
                f"module {self.adapter_module!r}; treating framework as unavailable",
                RuntimeWarning,
                stacklevel=2,
            )
            available = False
        else:
            available = bool(attr_value)

        _AVAILABILITY_CACHE[cache_key] = available
        return available

    def version_range(self) -> Optional[str]:
        """
        Return a human-readable version range string, if any.
        """
        if not self.minimum_framework_version and not self.tested_up_to_version:
            return None
        if self.minimum_framework_version and self.tested_up_to_version:
            return f">={self.minimum_framework_version}, <= {self.tested_up_to_version}"
        if self.minimum_framework_version:
            return f">={self.minimum_framework_version}"
        return f"<= {self.tested_up_to_version}"

    def validate(self) -> None:
        """
        Perform basic consistency checks on this descriptor.

        Raises
        ------
        ValueError on missing required fields or obviously inconsistent version bounds.
        """
        # Required: adapter init kwarg must be non-empty.
        if not isinstance(self.adapter_init_kwarg, str) or not self.adapter_init_kwarg.strip():
            raise ValueError(f"{self.name}: adapter_init_kwarg must be a non-empty string")

        # Required: core operations must exist (async protocol).
        for field_name in ("capabilities_method", "query_method", "upsert_method", "delete_method", "health_method"):
            v = getattr(self, field_name)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"{self.name}: {field_name} must be a non-empty string")

        for field_name in ("create_namespace_method", "delete_namespace_method"):
            v = getattr(self, field_name)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"{self.name}: {field_name} must be a non-empty string")

        # Batch query expectations.
        if self.has_batch_query and not (self.batch_query_method and self.batch_query_method.strip()):
            raise ValueError(f"{self.name}: has_batch_query=True requires batch_query_method to be set")

        # adapter_class should be a bare class name, not a dotted path.
        if "." in self.adapter_class:
            warnings.warn(
                f"{self.name}: adapter_class should be a class name only, "
                f"not a dotted path ({self.adapter_class!r})",
                RuntimeWarning,
                stacklevel=2,
            )

        # Constructor knobs expectations: if tests expect injection, require kwarg names.
        if self.supports_docstore_injection and not self.docstore_init_kwarg:
            warnings.warn(
                f"{self.name}: supports_docstore_injection=True but docstore_init_kwarg is None; "
                "tests may not know how to inject a docstore",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.supports_config_injection and not self.config_init_kwarg:
            warnings.warn(
                f"{self.name}: supports_config_injection=True but config_init_kwarg is None; "
                "tests may not know how to inject VectorAdapterConfig",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.supports_mode_switch and not self.mode_init_kwarg:
            warnings.warn(
                f"{self.name}: supports_mode_switch=True but mode_init_kwarg is None; "
                "tests may not know how to set thin/standalone mode",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.supports_auto_normalize_toggle and not self.supports_config_injection:
            # In your base, auto_normalize is driven by VectorAdapterConfig, so this is a reasonable sanity check.
            warnings.warn(
                f"{self.name}: supports_auto_normalize_toggle=True but supports_config_injection=False; "
                "auto_normalize is typically toggled via VectorAdapterConfig",
                RuntimeWarning,
                stacklevel=2,
            )

        # Version ordering validation (best-effort).
        if self.minimum_framework_version and self.tested_up_to_version:
            if Version is None:
                warnings.warn(
                    f"{self.name}: cannot validate version range ordering because "
                    "'packaging' is not installed "
                    f"(min={self.minimum_framework_version!r}, "
                    f"max={self.tested_up_to_version!r})",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                try:
                    min_v = Version(self.minimum_framework_version)
                    max_v = Version(self.tested_up_to_version)
                except Exception:
                    warnings.warn(
                        f"{self.name}: could not parse version range "
                        f"(min={self.minimum_framework_version!r}, "
                        f"max={self.tested_up_to_version!r}) for ordering validation",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    if min_v > max_v:
                        raise ValueError(
                            f"{self.name}: minimum_framework_version "
                            f"{self.minimum_framework_version!r} "
                            f"is greater than tested_up_to_version "
                            f"{self.tested_up_to_version!r}",
                        )


# ---------------------------------------------------------------------------
# Known vector framework adapters
# ---------------------------------------------------------------------------
# NOTE:
# These entries are *wrapper surface* descriptors for conformance tests.
# Backend capability truth comes from adapter.capabilities() at runtime.
VECTOR_FRAMEWORKS: Dict[str, VectorFrameworkDescriptor] = {
    # ------------------------------------------------------------------ #
    # AutoGen
    # ------------------------------------------------------------------ #
    "autogen": VectorFrameworkDescriptor(
        name="autogen",
        adapter_module="corpus_sdk.vector.framework_adapters.autogen",
        adapter_class="CorpusAutoGenVectorClient",
        adapter_init_kwarg="adapter",
        # VectorProtocolV1 surface
        capabilities_method="capabilities",
        query_method="query",
        batch_query_method="batch_query",
        upsert_method="upsert",
        delete_method="delete",
        create_namespace_method="create_namespace",
        delete_namespace_method="delete_namespace",
        health_method="health",
        # Framework context
        context_kwarg="conversation",
        # Expectations
        has_batch_query=True,
        supports_wire_handler=True,
        # If your wrapper supports passing config/mode/docstore, flip these to True and set kwargs.
        supports_docstore_injection=False,
        supports_config_injection=False,
        supports_mode_switch=False,
        supports_auto_normalize_toggle=False,
        availability_attr=None,
    ),

    # ------------------------------------------------------------------ #
    # CrewAI
    # ------------------------------------------------------------------ #
    "crewai": VectorFrameworkDescriptor(
        name="crewai",
        adapter_module="corpus_sdk.vector.framework_adapters.crewai",
        adapter_class="CorpusCrewAIVectorClient",
        adapter_init_kwarg="adapter",
        capabilities_method="capabilities",
        query_method="query",
        batch_query_method="batch_query",
        upsert_method="upsert",
        delete_method="delete",
        create_namespace_method="create_namespace",
        delete_namespace_method="delete_namespace",
        health_method="health",
        context_kwarg="task",
        has_batch_query=True,
        supports_wire_handler=True,
        supports_docstore_injection=False,
        supports_config_injection=False,
        supports_mode_switch=False,
        supports_auto_normalize_toggle=False,
        availability_attr=None,
    ),

    # ------------------------------------------------------------------ #
    # LangChain
    # ------------------------------------------------------------------ #
    "langchain": VectorFrameworkDescriptor(
        name="langchain",
        adapter_module="corpus_sdk.vector.framework_adapters.langchain",
        adapter_class="CorpusLangChainVectorClient",
        adapter_init_kwarg="adapter",
        capabilities_method="capabilities",
        query_method="query",
        batch_query_method="batch_query",
        upsert_method="upsert",
        delete_method="delete",
        create_namespace_method="create_namespace",
        delete_namespace_method="delete_namespace",
        health_method="health",
        context_kwarg="config",
        has_batch_query=True,
        supports_wire_handler=True,
        # Reuse the common availability flag pattern if your adapter module defines it.
        availability_attr="LANGCHAIN_AVAILABLE",
    ),

    # ------------------------------------------------------------------ #
    # LlamaIndex
    # ------------------------------------------------------------------ #
    "llamaindex": VectorFrameworkDescriptor(
        name="llamaindex",
        adapter_module="corpus_sdk.vector.framework_adapters.llamaindex",
        adapter_class="CorpusLlamaIndexVectorClient",
        adapter_init_kwarg="adapter",
        capabilities_method="capabilities",
        query_method="query",
        batch_query_method="batch_query",
        upsert_method="upsert",
        delete_method="delete",
        create_namespace_method="create_namespace",
        delete_namespace_method="delete_namespace",
        health_method="health",
        context_kwarg="callback_manager",
        has_batch_query=True,
        supports_wire_handler=True,
        availability_attr="LLAMAINDEX_AVAILABLE",
    ),

    # ------------------------------------------------------------------ #
    # Semantic Kernel
    # ------------------------------------------------------------------ #
    "semantic_kernel": VectorFrameworkDescriptor(
        name="semantic_kernel",
        adapter_module="corpus_sdk.vector.framework_adapters.semantic_kernel",
        adapter_class="CorpusSemanticKernelVectorClient",
        adapter_init_kwarg="adapter",
        capabilities_method="capabilities",
        query_method="query",
        batch_query_method="batch_query",
        upsert_method="upsert",
        delete_method="delete",
        create_namespace_method="create_namespace",
        delete_namespace_method="delete_namespace",
        health_method="health",
        context_kwarg="context",
        has_batch_query=True,
        supports_wire_handler=True,
        availability_attr="SEMANTIC_KERNEL_AVAILABLE",
    ),
}


def get_vector_framework_descriptor(name: str) -> VectorFrameworkDescriptor:
    """
    Look up a vector framework descriptor by name.

    Raises KeyError if the framework is not registered.
    """
    return VECTOR_FRAMEWORKS[name]


def get_vector_framework_descriptor_safe(name: str) -> Optional[VectorFrameworkDescriptor]:
    """
    Safe lookup for a vector framework descriptor.

    Returns None instead of raising KeyError when the framework is unknown.
    """
    return VECTOR_FRAMEWORKS.get(name)


def has_vector_framework(name: str) -> bool:
    """Return True if a framework with the given name is registered."""
    return name in VECTOR_FRAMEWORKS


# Backwards-compatible alias, if anything still uses the old name.
def has_framework(name: str) -> bool:
    return has_vector_framework(name)


def iter_vector_framework_descriptors() -> Iterable[VectorFrameworkDescriptor]:
    """Iterate over all registered vector framework descriptors."""
    return VECTOR_FRAMEWORKS.values()


def iter_available_vector_framework_descriptors() -> Iterable[VectorFrameworkDescriptor]:
    """
    Iterate over descriptors for frameworks that appear available.

    This is useful for tests that should only run when the underlying
    framework (LangChain, LlamaIndex, Semantic Kernel, etc.) is installed.
    """
    return (desc for desc in VECTOR_FRAMEWORKS.values() if desc.is_available())


def register_vector_framework_descriptor(
    descriptor: VectorFrameworkDescriptor,
    overwrite: bool = False,
) -> None:
    """
    Register a new vector framework descriptor dynamically (TEST-ONLY).

    Parameters
    ----------
    descriptor:
        Descriptor to register. Its `name` is used as the registry key.
    overwrite:
        If False (default), attempting to overwrite an existing entry raises KeyError.
        If True, replaces an existing entry (with a warning).
    """
    if descriptor.name in VECTOR_FRAMEWORKS and not overwrite:
        raise KeyError(f"Framework {descriptor.name!r} is already registered")

    if descriptor.name in VECTOR_FRAMEWORKS and overwrite:
        warnings.warn(
            f"Framework {descriptor.name!r} is being overwritten in the vector registry",
            RuntimeWarning,
            stacklevel=2,
        )

    VECTOR_FRAMEWORKS[descriptor.name] = descriptor

    # Reset availability cache for this descriptor so future checks re-evaluate.
    _AVAILABILITY_CACHE.pop(_availability_cache_key(descriptor.adapter_module, descriptor.availability_attr), None)


def unregister_vector_framework_descriptor(
    name: str,
    ignore_missing: bool = True,
) -> None:
    """
    Unregister a vector framework descriptor dynamically (TEST-ONLY).

    Parameters
    ----------
    name:
        Name of the framework to unregister.
    ignore_missing:
        If False, raise KeyError when not registered. If True, ignore missing entries.
    """
    if name in VECTOR_FRAMEWORKS:
        desc = VECTOR_FRAMEWORKS[name]
        del VECTOR_FRAMEWORKS[name]
        _AVAILABILITY_CACHE.pop(_availability_cache_key(desc.adapter_module, desc.availability_attr), None)
    elif not ignore_missing:
        raise KeyError(f"Framework {name!r} is not registered")


__all__ = [
    "VectorFrameworkDescriptor",
    "VECTOR_FRAMEWORKS",
    "get_vector_framework_descriptor",
    "get_vector_framework_descriptor_safe",
    "has_vector_framework",
    "has_framework",
    "iter_vector_framework_descriptors",
    "iter_available_vector_framework_descriptors",
    "register_vector_framework_descriptor",
    "unregister_vector_framework_descriptor",
]
