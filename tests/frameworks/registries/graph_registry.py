# tests/frameworks/registries/graph_registry.py
"""
Registry of graph framework adapters used by the conformance test suite.

This module is TEST-ONLY. It provides lightweight metadata describing how to:
- Import each graph framework adapter
- Construct its client
- Call its sync / async query & streaming methods
- Pass framework-specific context
- Know which extra semantics (health, capabilities, bulk, batch) to expect

Contract tests in tests/frameworks/graph/ use this registry to stay
framework-agnostic. Adding a new graph framework typically means:

1. Implement the adapter under corpus_sdk.graph.framework_adapters.*.
2. Add a new GraphFrameworkDescriptor entry here (or register dynamically).
3. Run the graph contract tests.

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

# Simple in-memory cache to avoid repeatedly importing modules for availability checks.
#
# IMPORTANT: cache key must be stable across descriptor overwrites. Caching only by
# descriptor.name can lead to stale results when a test overwrites the registry entry.
# We therefore cache on a composite key derived from adapter_module + availability_attr.
_AVAILABILITY_CACHE: Dict[str, bool] = {}


def _availability_cache_key(adapter_module: str, availability_attr: Optional[str]) -> str:
    """Build a stable cache key for availability checks (do not use descriptor.name)."""
    return f"{adapter_module}:{availability_attr or ''}"


@dataclass(frozen=True)
class GraphFrameworkDescriptor:
    """
    Description of a graph framework adapter (TEST-ONLY).

    Fields
    ------
    name:
        Short, stable identifier (e.g. "autogen", "langchain").
    adapter_module:
        Dotted import path for the adapter module.
    adapter_class:
        Name of the adapter class within adapter_module (bare class name).

    adapter_init_kwarg:
        Name of the constructor kwarg used to inject the underlying Corpus graph adapter.
        This keeps the test suite framework-agnostic while still enforcing that the
        framework adapter is actually wired to the injected backend.

        Examples:
        - "adapter"        (common in some framework adapters)
        - "corpus_adapter" (common in other framework adapters)

        IMPORTANT:
        - Tests should use this field rather than hardcoding `adapter=...`.
        - If a framework adapter changes its constructor signature, update this value.

    query_method:
        Name of the *sync* query method (non-streaming).
    async_query_method:
        Name of the *async* query method, or None if not supported.

    stream_query_method:
        Name of the *sync* streaming query method, or None.
    async_stream_query_method:
        Name of the *async* streaming query method, or None.

    bulk_vertices_method:
        Name of the *sync* bulk_vertices method, or None.
    async_bulk_vertices_method:
        Name of the *async* bulk_vertices method, or None.

    batch_method:
        Name of the *sync* batch method, or None.
    async_batch_method:
        Name of the *async* batch method, or None.

    # ------------------------------------------------------------------
    # Extended graph protocol surfaces (optional, but testable)
    # ------------------------------------------------------------------
    # These method-name fields allow the conformance suite to:
    #   1) instantiate the framework adapter with an injected Corpus graph adapter, and
    #   2) call the framework adapter methods to verify wiring, passthrough, and contracts.
    #
    # IMPORTANT:
    # - These are *names only* (strings). Tests use them to get callables from instances.
    # - If a method is not supported by a framework adapter, leave it as None.
    #
    # capabilities_method / async_capabilities_method:
    #   Framework adapter capability discovery passthrough.
    #
    # health_method / async_health_method:
    #   Framework adapter health passthrough.
    #
    # schema_method / async_schema_method:
    #   Schema discovery (Graph Protocol optional surface).
    #
    # transaction_method / async_transaction_method:
    #   Transaction operation support (Graph Protocol optional surface).
    #
    # traversal_method / async_traversal_method:
    #   Traversal operation support (Graph Protocol optional surface).
    capabilities_method:
        Name of the *sync* capabilities method, or None.
    async_capabilities_method:
        Name of the *async* capabilities method, or None.

    health_method:
        Name of the *sync* health method, or None.
    async_health_method:
        Name of the *async* health method, or None.

    schema_method:
        Name of the *sync* get_schema method, or None.
    async_schema_method:
        Name of the *async* get_schema method, or None.

    transaction_method:
        Name of the *sync* transaction method, or None.
    async_transaction_method:
        Name of the *async* transaction method, or None.

    traversal_method:
        Name of the *sync* traversal method, or None.
    async_traversal_method:
        Name of the *async* traversal method, or None.

    context_kwarg:
        Name of the kwargs parameter used for framework-specific context
        (e.g. "conversation", "task", "config", "callback_manager", "context").

    has_capabilities:
        True if the adapter exposes a capabilities()/acapabilities() surface.

    has_health:
        True if the adapter exposes a health()/ahealth() surface.

    supports_streaming:
        True if the adapter is expected to support streaming queries.

    supports_bulk_vertices:
        True if the adapter is expected to support bulk_vertices.

    supports_batch:
        True if the adapter is expected to support batch operations.

    availability_attr:
        Optional module-level boolean that indicates whether the underlying
        framework is actually installed, e.g. "LANGCHAIN_TOOL_AVAILABLE".
        Tests can skip or adjust expectations when this is False.

    minimum_framework_version:
        Optional minimum framework version (string) this adapter/registry entry
        has been validated against.

    tested_up_to_version:
        Optional maximum framework version (string) this adapter/registry entry
        is known to work with.
    """

    name: str
    adapter_module: str
    adapter_class: str

    # Constructor injection kwarg for the underlying Corpus adapter.
    adapter_init_kwarg: str = "adapter"

    query_method: str = ""
    async_query_method: Optional[str] = None

    stream_query_method: Optional[str] = None
    async_stream_query_method: Optional[str] = None

    bulk_vertices_method: Optional[str] = None
    async_bulk_vertices_method: Optional[str] = None

    batch_method: Optional[str] = None
    async_batch_method: Optional[str] = None

    # Extended graph protocol surfaces
    capabilities_method: Optional[str] = None
    async_capabilities_method: Optional[str] = None

    health_method: Optional[str] = None
    async_health_method: Optional[str] = None

    schema_method: Optional[str] = None
    async_schema_method: Optional[str] = None

    transaction_method: Optional[str] = None
    async_transaction_method: Optional[str] = None

    traversal_method: Optional[str] = None
    async_traversal_method: Optional[str] = None

    context_kwarg: Optional[str] = None

    has_capabilities: bool = False
    has_health: bool = False

    supports_streaming: bool = False
    supports_bulk_vertices: bool = False
    supports_batch: bool = False

    availability_attr: Optional[str] = None
    minimum_framework_version: Optional[str] = None
    tested_up_to_version: Optional[str] = None

    def __post_init__(self) -> None:
        """
        Run basic consistency checks after dataclass initialization.

        This will raise early for obviously invalid descriptors (e.g. missing
        method names) and emit non-fatal warnings for softer issues.
        """
        self.validate()

    @property
    def supports_async(self) -> bool:
        """True if any async method is declared."""
        return bool(
            self.async_query_method
            or self.async_stream_query_method
            or self.async_bulk_vertices_method
            or self.async_batch_method
            or self.async_capabilities_method
            or self.async_health_method
            or self.async_schema_method
            or self.async_transaction_method
            or self.async_traversal_method
        )

    def is_available(self) -> bool:
        """
        Check if the underlying framework appears available for testing.

        If availability_attr is set, this checks that boolean on the adapter
        module. Otherwise assumes the framework is available.

        Results are cached per module+attribute to avoid repeated imports in
        large test suites while staying stable across descriptor overwrites.
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

        Example: ">=0.1.0, <=0.3.5" or None if no constraints are set.
        """
        if not self.minimum_framework_version and not self.tested_up_to_version:
            return None

        # NOTE:
        # Keep formatting stable and predictable; tests may normalize whitespace,
        # but this function should remain consistent.
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
            If required fields like query_method/stream_query_method are missing,
            or when version bounds are obviously inconsistent.
        """
        # Required methods: query_method and stream_query_method
        # (All current graph adapters support both)
        if not self.query_method or not self.stream_query_method:
            raise ValueError(
                f"{self.name}: query_method and stream_query_method must both be set",
            )

        # Constructor kwarg must be a non-empty string when provided.
        if not isinstance(self.adapter_init_kwarg, str) or not self.adapter_init_kwarg.strip():
            raise ValueError(f"{self.name}: adapter_init_kwarg must be a non-empty string")

        # Async consistency warnings (soft)
        if self.async_stream_query_method and not self.stream_query_method:
            warnings.warn(
                f"{self.name}: async_stream_query_method is set but "
                f"stream_query_method is None (async should have a sync counterpart)",
                RuntimeWarning,
                stacklevel=2,
            )

        if self.async_bulk_vertices_method and not self.bulk_vertices_method:
            warnings.warn(
                f"{self.name}: async_bulk_vertices_method is set but "
                f"bulk_vertices_method is None (async should have a sync counterpart)",
                RuntimeWarning,
                stacklevel=2,
            )

        if self.async_batch_method and not self.batch_method:
            warnings.warn(
                f"{self.name}: async_batch_method is set but "
                f"batch_method is None (async should have a sync counterpart)",
                RuntimeWarning,
                stacklevel=2,
            )

        # IMPORTANT: Async stream without async query is an edge-case inconsistency that
        # the graph registry self-check tests explicitly validate.
        if self.async_stream_query_method and not self.async_query_method:
            warnings.warn(
                f"{self.name}: async_stream_query_method is set but async_query_method is None",
                RuntimeWarning,
                stacklevel=2,
            )

        # Streaming flags vs method names (soft warnings)
        if self.supports_streaming and not self.stream_query_method:
            warnings.warn(
                f"{self.name}: supports_streaming is True but "
                f"stream_query_method is not set",
                RuntimeWarning,
                stacklevel=2,
            )

        if self.supports_bulk_vertices and not self.bulk_vertices_method:
            warnings.warn(
                f"{self.name}: supports_bulk_vertices is True but "
                f"bulk_vertices_method is not set",
                RuntimeWarning,
                stacklevel=2,
            )

        if self.supports_batch and not self.batch_method:
            warnings.warn(
                f"{self.name}: supports_batch is True but "
                f"batch_method is not set",
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

        # Version ordering validation (best-effort)
        if self.minimum_framework_version and self.tested_up_to_version:
            if Version is None:
                # packaging not installed; we can't validate ordering robustly
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

        # If only one bound is set, there's nothing to order-check; they remain
        # informational for tests.


# ---------------------------------------------------------------------------
# Known graph framework adapters
# ---------------------------------------------------------------------------

GRAPH_FRAMEWORKS: Dict[str, GraphFrameworkDescriptor] = {
    # ------------------------------------------------------------------ #
    # AutoGen
    # ------------------------------------------------------------------ #
    "autogen": GraphFrameworkDescriptor(
        name="autogen",
        adapter_module="corpus_sdk.graph.framework_adapters.autogen",
        adapter_class="CorpusAutoGenGraphClient",
        adapter_init_kwarg="adapter",
        query_method="query",
        async_query_method="aquery",
        stream_query_method="stream_query",
        async_stream_query_method="astream_query",
        bulk_vertices_method="bulk_vertices",
        async_bulk_vertices_method="abulk_vertices",
        batch_method="batch",
        async_batch_method="abatch",
        # Extended surfaces (declared so tests can verify wiring)
        capabilities_method="capabilities",
        async_capabilities_method="acapabilities",
        health_method="health",
        async_health_method="ahealth",
        schema_method="get_schema",
        async_schema_method="aget_schema",
        transaction_method="transaction",
        async_transaction_method="atransaction",
        traversal_method="traversal",
        async_traversal_method="atraversal",
        context_kwarg="conversation",
        has_capabilities=True,
        has_health=True,
        supports_streaming=True,
        supports_bulk_vertices=True,
        supports_batch=True,
        availability_attr=None,
    ),
    # ------------------------------------------------------------------ #
    # CrewAI
    # ------------------------------------------------------------------ #
    "crewai": GraphFrameworkDescriptor(
        name="crewai",
        adapter_module="corpus_sdk.graph.framework_adapters.crewai",
        adapter_class="CorpusCrewAIGraphClient",
        # CrewAI framework adapters use standard adapter injection
        adapter_init_kwarg="adapter",
        query_method="query",
        async_query_method="aquery",
        stream_query_method="stream_query",
        async_stream_query_method="astream_query",
        bulk_vertices_method="bulk_vertices",
        async_bulk_vertices_method="abulk_vertices",
        batch_method="batch",
        async_batch_method="abatch",
        # Extended surfaces (declared so tests can verify wiring)
        capabilities_method="capabilities",
        async_capabilities_method="acapabilities",
        health_method="health",
        async_health_method="ahealth",
        schema_method="get_schema",
        async_schema_method="aget_schema",
        transaction_method="transaction",
        async_transaction_method="atransaction",
        traversal_method="traversal",
        async_traversal_method="atraversal",
        context_kwarg="task",
        has_capabilities=True,
        has_health=True,
        supports_streaming=True,
        supports_bulk_vertices=True,
        supports_batch=True,
        availability_attr=None,
    ),
    # ------------------------------------------------------------------ #
    # LangChain
    # ------------------------------------------------------------------ #
    "langchain": GraphFrameworkDescriptor(
        name="langchain",
        adapter_module="corpus_sdk.graph.framework_adapters.langchain",
        adapter_class="CorpusLangChainGraphClient",
        adapter_init_kwarg="adapter",
        query_method="query",
        async_query_method="aquery",
        stream_query_method="stream_query",
        async_stream_query_method="astream_query",
        bulk_vertices_method="bulk_vertices",
        async_bulk_vertices_method="abulk_vertices",
        batch_method="batch",
        async_batch_method="abatch",
        # Extended surfaces (declared so tests can verify wiring)
        capabilities_method="capabilities",
        async_capabilities_method="acapabilities",
        health_method="health",
        async_health_method="ahealth",
        schema_method="get_schema",
        async_schema_method="aget_schema",
        transaction_method="transaction",
        async_transaction_method="atransaction",
        traversal_method="traversal",
        async_traversal_method="atraversal",
        context_kwarg="config",
        has_capabilities=True,
        has_health=True,
        supports_streaming=True,
        supports_bulk_vertices=True,
        supports_batch=True,
        availability_attr=None,
    ),
    # ------------------------------------------------------------------ #
    # LlamaIndex
    # ------------------------------------------------------------------ #
    "llamaindex": GraphFrameworkDescriptor(
        name="llamaindex",
        adapter_module="corpus_sdk.graph.framework_adapters.llamaindex",
        adapter_class="CorpusLlamaIndexGraphClient",
        # LlamaIndex framework adapters use standard adapter injection
        adapter_init_kwarg="adapter",
        query_method="query",
        async_query_method="aquery",
        stream_query_method="stream_query",
        async_stream_query_method="astream_query",
        bulk_vertices_method="bulk_vertices",
        async_bulk_vertices_method="abulk_vertices",
        batch_method="batch",
        async_batch_method="abatch",
        # Extended surfaces (declared so tests can verify wiring)
        capabilities_method="capabilities",
        async_capabilities_method="acapabilities",
        health_method="health",
        async_health_method="ahealth",
        schema_method="get_schema",
        async_schema_method="aget_schema",
        transaction_method="transaction",
        async_transaction_method="atransaction",
        traversal_method="traversal",
        async_traversal_method="atraversal",
        context_kwarg="callback_manager",
        has_capabilities=True,
        has_health=True,
        supports_streaming=True,
        supports_bulk_vertices=True,
        supports_batch=True,
        availability_attr=None,
    ),
    # ------------------------------------------------------------------ #
    # Semantic Kernel
    # ------------------------------------------------------------------ #
    "semantic_kernel": GraphFrameworkDescriptor(
        name="semantic_kernel",
        adapter_module="corpus_sdk.graph.framework_adapters.semantic_kernel",
        adapter_class="CorpusSemanticKernelGraphClient",
        # Semantic Kernel framework adapters use standard adapter injection
        adapter_init_kwarg="adapter",
        query_method="query",
        async_query_method="aquery",
        stream_query_method="stream_query",
        async_stream_query_method="astream_query",
        bulk_vertices_method="bulk_vertices",
        async_bulk_vertices_method="abulk_vertices",
        batch_method="batch",
        async_batch_method="abatch",
        # Extended surfaces (declared so tests can verify wiring)
        capabilities_method="capabilities",
        async_capabilities_method="acapabilities",
        health_method="health",
        async_health_method="ahealth",
        schema_method="get_schema",
        async_schema_method="aget_schema",
        transaction_method="transaction",
        async_transaction_method="atransaction",
        traversal_method="traversal",
        async_traversal_method="atraversal",
        context_kwarg="context",
        has_capabilities=True,
        has_health=True,
        supports_streaming=True,
        supports_bulk_vertices=True,
        supports_batch=True,
        availability_attr=None,
    ),
}


def get_graph_framework_descriptor(name: str) -> GraphFrameworkDescriptor:
    """
    Look up a graph framework descriptor by name.

    Raises
    ------
    KeyError if the framework is not registered.
    """
    return GRAPH_FRAMEWORKS[name]


def get_graph_framework_descriptor_safe(
    name: str,
) -> Optional[GraphFrameworkDescriptor]:
    """
    Safe lookup for a graph framework descriptor.

    Returns None instead of raising KeyError when the framework is unknown.
    """
    return GRAPH_FRAMEWORKS.get(name)


def has_graph_framework(name: str) -> bool:
    """
    Return True if a framework with the given name is registered.
    """
    return name in GRAPH_FRAMEWORKS


# Backwards-compatible alias, if anything still uses the old name.
def has_framework(name: str) -> bool:
    return has_graph_framework(name)


def iter_graph_framework_descriptors() -> Iterable[GraphFrameworkDescriptor]:
    """
    Iterate over all registered graph framework descriptors.
    """
    return GRAPH_FRAMEWORKS.values()


def iter_available_graph_framework_descriptors() -> Iterable[GraphFrameworkDescriptor]:
    """
    Iterate over descriptors for frameworks that appear available.

    This is useful for tests that should only run when the underlying
    framework (LangChain, LlamaIndex, Semantic Kernel, etc.) is installed.
    """
    return (desc for desc in GRAPH_FRAMEWORKS.values() if desc.is_available())


def register_graph_framework_descriptor(
    descriptor: GraphFrameworkDescriptor,
    overwrite: bool = False,
) -> None:
    """
    Register a new graph framework descriptor dynamically (TEST-ONLY).

    This is primarily intended for test scenarios where you want to plug in
    an experimental or third-party graph framework adapter.

    Parameters
    ----------
    descriptor:
        Descriptor to register. Its `name` is used as the registry key.
    overwrite:
        If False (default), attempting to overwrite an existing entry will
        raise KeyError. If True, an existing entry with the same name is
        replaced (with a warning).
    """
    if descriptor.name in GRAPH_FRAMEWORKS and not overwrite:
        raise KeyError(f"Framework {descriptor.name!r} is already registered")

    if descriptor.name in GRAPH_FRAMEWORKS and overwrite:
        warnings.warn(
            f"Framework {descriptor.name!r} is being overwritten in the graph registry",
            RuntimeWarning,
            stacklevel=2,
        )

    GRAPH_FRAMEWORKS[descriptor.name] = descriptor
    # Reset availability cache for this descriptor so future checks re-evaluate.
    _AVAILABILITY_CACHE.pop(_availability_cache_key(descriptor.adapter_module, descriptor.availability_attr), None)


def unregister_graph_framework_descriptor(
    name: str,
    ignore_missing: bool = True,
) -> None:
    """
    Unregister a graph framework descriptor dynamically (TEST-ONLY).

    Useful for tests that temporarily override or replace registry entries.

    Parameters
    ----------
    name:
        Name of the framework to unregister.
    ignore_missing:
        If False, raise KeyError when the framework is not registered.
        If True (default), missing entries are ignored.
    """
    if name in GRAPH_FRAMEWORKS:
        desc = GRAPH_FRAMEWORKS[name]
        del GRAPH_FRAMEWORKS[name]
        _AVAILABILITY_CACHE.pop(_availability_cache_key(desc.adapter_module, desc.availability_attr), None)
    elif not ignore_missing:
        raise KeyError(f"Framework {name!r} is not registered")


__all__ = [
    "GraphFrameworkDescriptor",
    "GRAPH_FRAMEWORKS",
    "get_graph_framework_descriptor",
    "get_graph_framework_descriptor_safe",
    "has_graph_framework",
    "has_framework",
    "iter_graph_framework_descriptors",
    "iter_available_graph_framework_descriptors",
    "register_graph_framework_descriptor",
    "unregister_graph_framework_descriptor",
]
