# tests/frameworks/registries/vector_registry.py
"""
Registry of vector framework adapters used by the conformance test suite.

This module is TEST-ONLY. It provides lightweight metadata describing how to:
- Import each vector framework adapter
- Construct its client/store
- Call its sync / async add, delete, query, streaming, and MMR methods
- Pass framework-specific context
- Know which extra semantics (health, capabilities, streaming, MMR) to expect

Contract tests in tests/frameworks/vector/ use this registry to stay
framework-agnostic. Adding a new vector framework typically means:

1. Implement the adapter under corpus_sdk.vector.framework_adapters.*.
2. Add a new VectorFrameworkDescriptor entry here (or register dynamically).
3. Run the vector contract tests.

Version fields
--------------
`minimum_framework_version` and `tested_up_to_version` are currently informational.
In the future, tests may use them to conditionally skip or adjust expectations
based on the installed framework version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import importlib
import warnings


@dataclass(frozen=True)
class VectorFrameworkDescriptor:
    """
    Description of a vector framework adapter.

    Fields
    ------
    name:
        Short, stable identifier (e.g. "llamaindex", "semantic_kernel").
    adapter_module:
        Dotted import path for the adapter module.
    adapter_class:
        Name of the adapter class within adapter_module.

    add_method:
        Name of the *sync* add/upsert method (vector insert), e.g. "add" or
        "add_texts". Must accept batched inputs.
    async_add_method:
        Name of the *async* add/upsert method, or None if not supported.

    delete_method:
        Name of the *sync* delete method, or None if not supported.
        The exact semantics (by ID, by ref_doc_id, by filter) are framework-
        specific; tests use this via framework-agnostic helpers.
    async_delete_method:
        Name of the *async* delete method, or None.

    query_method:
        Name of the *sync* similarity query method (non-streaming).
    async_query_method:
        Name of the *async* similarity query method, or None.

    stream_query_method:
        Name of the *sync* streaming query method, or None.
    async_stream_query_method:
        Name of the *async* streaming query method, or None.

    mmr_query_method:
        Name of the *sync* Maximal Marginal Relevance (MMR) query method, or None.
    async_mmr_query_method:
        Name of the *async* MMR query method, or None.

    context_kwarg:
        Name of the kwargs parameter used for framework-specific context
        (e.g. "callback_manager", "sk_context", "config").

    has_capabilities:
        True if the adapter exposes a capabilities()/acapabilities() surface.

    has_health:
        True if the adapter exposes a health()/ahealth() surface.

    supports_streaming:
        True if the adapter is expected to support streaming queries.

    supports_mmr:
        True if the adapter is expected to support MMR queries.

    availability_attr:
        Optional module-level boolean that indicates whether the underlying
        framework is actually installed, e.g. "LANGCHAIN_VECTOR_AVAILABLE".
        Tests can skip or adjust expectations when this is False.

    minimum_framework_version:
        Optional minimum framework version (string) this adapter/registry entry
        has been validated against. Informational for now.

    tested_up_to_version:
        Optional maximum framework version (string) this adapter/registry entry
        is known to work with. Informational for now.
    """

    name: str
    adapter_module: str
    adapter_class: str

    add_method: str
    async_add_method: Optional[str] = None

    delete_method: Optional[str] = None
    async_delete_method: Optional[str] = None

    query_method: str
    async_query_method: Optional[str] = None

    stream_query_method: Optional[str] = None
    async_stream_query_method: Optional[str] = None

    mmr_query_method: Optional[str] = None
    async_mmr_query_method: Optional[str] = None

    context_kwarg: Optional[str] = None

    has_capabilities: bool = False
    has_health: bool = False

    supports_streaming: bool = False
    supports_mmr: bool = False

    availability_attr: Optional[str] = None
    minimum_framework_version: Optional[str] = None
    tested_up_to_version: Optional[str] = None

    def __post_init__(self) -> None:
        """
        Run basic consistency checks after dataclass initialization.

        This will raise early for obviously invalid descriptors (e.g. missing
        core method names) and emit non-fatal warnings for softer issues.
        """
        # Since the dataclass is frozen, we must not mutate; validation is read-only.
        self.validate()

    @property
    def supports_async(self) -> bool:
        """True if any async method is declared."""
        return bool(
            self.async_add_method
            or self.async_delete_method
            or self.async_query_method
            or self.async_stream_query_method
            or self.async_mmr_query_method
        )

    def is_available(self) -> bool:
        """
        Check if the underlying framework appears available for testing.

        If availability_attr is set, this checks that boolean on the adapter
        module. Otherwise assumes the framework is available.
        """
        if not self.availability_attr:
            return True

        try:
            module = importlib.import_module(self.adapter_module)
        except ImportError:
            return False

        return bool(getattr(module, self.availability_attr, False))

    def version_range(self) -> Optional[str]:
        """
        Return a human-readable version range string, if any.

        Example: ">=0.1.0, <=0.3.5" or None if no constraints are set.
        """
        if not self.minimum_framework_version and not self.tested_up_to_version:
            return None

        if self.minimum_framework_version and self.tested_up_to_version:
            return f">={self.minimum_framework_version}, <={self.tested_up_to_version}"
        if self.minimum_framework_version:
            return f">={self.minimum_framework_version}"
        return f"<={self.tested_up_to_version}"

    def validate(self) -> None:
        """
        Perform basic consistency checks on this descriptor.

        Raises
        ------
        ValueError
            If required fields like add_method/query_method are missing.
        """
        # Required core methods: add_method and query_method.
        if not self.add_method or not self.query_method:
            raise ValueError(
                f"{self.name}: add_method and query_method must both be set",
            )

        # Async consistency warnings (soft).

        if self.async_add_method and not self.add_method:
            warnings.warn(
                f"{self.name}: async_add_method is set but add_method is None "
                f"(async should usually have a sync counterpart)",
                RuntimeWarning,
                stacklevel=2,
            )

        if self.async_delete_method and not self.delete_method:
            warnings.warn(
                f"{self.name}: async_delete_method is set but delete_method is None "
                f"(async should usually have a sync counterpart)",
                RuntimeWarning,
                stacklevel=2,
            )

        if self.async_query_method and not self.query_method:
            warnings.warn(
                f"{self.name}: async_query_method is set but query_method is None "
                f"(async should usually have a sync counterpart)",
                RuntimeWarning,
                stacklevel=2,
            )

        if self.async_stream_query_method and not self.stream_query_method:
            warnings.warn(
                f"{self.name}: async_stream_query_method is set but "
                f"stream_query_method is None (async streaming should have a "
                f"sync counterpart)",
                RuntimeWarning,
                stacklevel=2,
            )

        if self.async_mmr_query_method and not self.mmr_query_method:
            warnings.warn(
                f"{self.name}: async_mmr_query_method is set but "
                f"mmr_query_method is None (async should usually have a "
                f"sync counterpart)",
                RuntimeWarning,
                stacklevel=2,
            )

        # Feature flags vs method names (soft warnings).

        if self.supports_streaming and not (
            self.stream_query_method or self.async_stream_query_method
        ):
            warnings.warn(
                f"{self.name}: supports_streaming is True but neither "
                f"stream_query_method nor async_stream_query_method is set",
                RuntimeWarning,
                stacklevel=2,
            )

        if self.supports_mmr and not (
            self.mmr_query_method or self.async_mmr_query_method
        ):
            warnings.warn(
                f"{self.name}: supports_mmr is True but neither mmr_query_method "
                f"nor async_mmr_query_method is set",
                RuntimeWarning,
                stacklevel=2,
            )

        # adapter_class should be a bare class name, not a dotted path
        if "." in self.adapter_class:
            warnings.warn(
                f"{self.name}: adapter_class should be a class name only, "
                f"not a dotted path ({self.adapter_class!r})",
                RuntimeWarning,
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# Known vector framework adapters
# ---------------------------------------------------------------------------

VECTOR_FRAMEWORKS: Dict[str, VectorFrameworkDescriptor] = {
    # ------------------------------------------------------------------ #
    # LlamaIndex
    # ------------------------------------------------------------------ #
    "llamaindex": VectorFrameworkDescriptor(
        name="llamaindex",
        adapter_module="corpus_sdk.vector.framework_adapters.llamaindex",
        adapter_class="CorpusLlamaIndexVectorStore",
        # Node-based add / delete; batched by design.
        add_method="add",
        async_add_method="aadd",
        delete_method="delete",           # delete(ref_doc_id=...)
        async_delete_method="adelete",
        # Core similarity query surface.
        query_method="query",
        async_query_method="aquery",
        # Streaming query.
        stream_query_method="query_stream",
        async_stream_query_method=None,   # No explicit async streaming variant today.
        # MMR surface.
        mmr_query_method="query_mmr",
        async_mmr_query_method="aquery_mmr",
        # LlamaIndex-specific context path.
        context_kwarg="callback_manager",
        has_capabilities=False,           # Capabilities are internal via VectorTranslator.
        has_health=False,
        supports_streaming=True,
        supports_mmr=True,
        availability_attr=None,
        minimum_framework_version=None,
        tested_up_to_version=None,
    ),

    # ------------------------------------------------------------------ #
    # Semantic Kernel
    # ------------------------------------------------------------------ #
    "semantic_kernel": VectorFrameworkDescriptor(
        name="semantic_kernel",
        adapter_module="corpus_sdk.vector.framework_adapters.semantic_kernel",
        adapter_class="CorpusSemanticKernelVectorStore",
        # Text-based add; accepts batched texts+metadata.
        add_method="add_texts",
        async_add_method="aadd_texts",
        delete_method="delete",
        async_delete_method="adelete",
        # AI-friendly similarity search surface.
        query_method="similarity_search",
        async_query_method="asimilarity_search",
        # Streaming similarity search.
        stream_query_method="similarity_search_stream",
        async_stream_query_method=None,  # Streaming currently sync-only on the store.
        # MMR search surface.
        mmr_query_method="max_marginal_relevance_search",
        async_mmr_query_method="amax_marginal_relevance_search",
        # Semantic Kernel–specific context.
        context_kwarg="sk_context",
        has_capabilities=False,          # Store uses get_capabilities/aget_capabilities,
                                         # not capabilities()/acapabilities() by contract.
        has_health=False,
        supports_streaming=True,
        supports_mmr=True,
        availability_attr=None,
        minimum_framework_version=None,
        tested_up_to_version=None,
    ),
}


def get_vector_framework_descriptor(name: str) -> VectorFrameworkDescriptor:
    """
    Look up a vector framework descriptor by name.

    Raises
    ------
    KeyError if the framework is not registered.
    """
    return VECTOR_FRAMEWORKS[name]


def get_vector_framework_descriptor_safe(
    name: str,
) -> Optional[VectorFrameworkDescriptor]:
    """
    Safe lookup for a vector framework descriptor.

    Returns None instead of raising KeyError when the framework is unknown.
    """
    return VECTOR_FRAMEWORKS.get(name)


def has_vector_framework(name: str) -> bool:
    """
    Return True if a framework with the given name is registered.
    """
    return name in VECTOR_FRAMEWORKS


# Backwards-compatible alias, if anything still uses the generic name.
def has_framework(name: str) -> bool:
    return has_vector_framework(name)


def iter_vector_framework_descriptors() -> Iterable[VectorFrameworkDescriptor]:
    """
    Iterate over all registered vector framework descriptors.
    """
    return VECTOR_FRAMEWORKS.values()


def iter_available_vector_framework_descriptors() -> Iterable[VectorFrameworkDescriptor]:
    """
    Iterate over descriptors for frameworks that appear available.

    This is useful for tests that should only run when the underlying
    framework (LlamaIndex, Semantic Kernel, etc.) is installed.
    """
    return (desc for desc in VECTOR_FRAMEWORKS.values() if desc.is_available())


def register_vector_framework_descriptor(
    descriptor: VectorFrameworkDescriptor,
    overwrite: bool = False,
) -> None:
    """
    Register a new vector framework descriptor dynamically (TEST-ONLY).

    This is primarily intended for test scenarios where you want to plug in
    an experimental or third-party vector framework adapter.

    Parameters
    ----------
    descriptor:
        Descriptor to register. Its `name` is used as the registry key.
    overwrite:
        If False (default), attempting to overwrite an existing entry will
        raise KeyError. If True, an existing entry with the same name is
        replaced.
    """
    if descriptor.name in VECTOR_FRAMEWORKS and not overwrite:
        raise KeyError(f"Framework {descriptor.name!r} is already registered")

    VECTOR_FRAMEWORKS[descriptor.name] = descriptor


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
]
