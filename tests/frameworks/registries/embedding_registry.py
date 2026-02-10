# tests/frameworks/registries/embedding_registry.py
"""
Registry of embedding framework adapters used by the conformance test suite.

This module is TEST-ONLY. It provides lightweight metadata describing how to:
- Import each embedding framework adapter
- Call its sync / async batch & query methods
- Pass framework-specific context
- Know which extra semantics (health, capabilities, dimension, versions) to expect

Contract tests in tests/frameworks/embedding/ use this registry to stay
framework-agnostic. Adding a new embedding framework typically means:

1. Implement the adapter under corpus_sdk.embedding.framework_adapters.*.
2. Add a new EmbeddingFrameworkDescriptor entry here (or register dynamically).
3. Run the embedding contract tests.

Version fields
--------------
`minimum_framework_version` and `tested_up_to_version` are currently informational,
but we validate that the range is coherent when possible. In the future, tests may
use them to conditionally skip or adjust expectations based on the installed
framework version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional
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

# Best-effort cache for framework version probes (avoid repeated imports).
_VERSION_CACHE: Dict[str, Optional[str]] = {}


def _availability_cache_key(adapter_module: str, availability_attr: Optional[str]) -> str:
    """
    Build a stable cache key for availability checks.

    We intentionally do NOT use descriptor.name, because tests may dynamically overwrite
    entries with the same name but different module / attribute.
    """
    return f"{adapter_module}:{availability_attr or ''}"


def _version_cache_key(adapter_module: str) -> str:
    """Cache key for best-effort framework version probes."""
    return f"{adapter_module}:framework_version"


@dataclass(frozen=True)
class EmbeddingFrameworkDescriptor:
    """
    Description of an embedding framework adapter (TEST-ONLY).

    Fields
    ------
    name:
        Short, stable identifier (e.g. "autogen", "langchain").
    adapter_module:
        Dotted import path for the adapter module.
    adapter_class:
        Name of the adapter class within adapter_module (bare class name).

    batch_method:
        Name of the *sync* batch embedding method.
        Returns a 2D list of floats.
    query_method:
        Name of the *sync* single-text embedding method.
        Returns a 1D list of floats.

    async_batch_method:
        Name of the *async* batch embedding method, or None if not supported.
    async_query_method:
        Name of the *async* single-text embedding method, or None.
        If any async method is provided, tests generally expect both batch
        and query async methods to be present.

    context_kwarg:
        Name of the kwargs parameter used for framework-specific context
        (e.g. "autogen_context", "config", "llamaindex_context", "sk_context").
        This is metadata for tests; it is not enforced at runtime here.

    requires_embedding_dimension:
        True if the adapter requires a known embedding dimension up-front
        (either via adapter.get_embedding_dimension() or an explicit override).

    embedding_dimension_kwarg:
        If requires_embedding_dimension=True, this indicates the constructor kwarg
        used to pass an explicit embedding dimension override (e.g. "embedding_dimension").
        This allows tests to satisfy the contract without guessing kwarg names.

    has_capabilities:
        True if tests expect the adapter to expose a capabilities()/acapabilities()
        surface (it may still raise NotImplementedError at runtime).

    capabilities_method / async_capabilities_method:
        Optional explicit method names for capabilities surfaces when present.
        If provided, tests can call these methods directly without guessing.

    has_health:
        True if tests expect the adapter to expose a health()/ahealth() surface
        (it may still raise NotImplementedError at runtime).

    health_method / async_health_method:
        Optional explicit method names for health surfaces when present.
        If provided, tests can call these methods directly without guessing.

    primary_surface:
        Optional marker describing the adapter's primary interface shape.
        Examples:
          - "embedder": standard embed_documents/embed_query-style surface
          - "baseembedding": LlamaIndex BaseEmbedding internal surfaces (_get_*)
          - "embedding_generator": Semantic Kernel embedding generator surfaces
        This is a hint for tests; it is not enforced at runtime here.

    aliases:
        Optional mapping of alternate method names to primary ones.
        Useful when frameworks expose both "native" names and adapter-provided aliases.
        Example for Semantic Kernel:
          {"embed_documents": "generate_embeddings", "embed_query": "generate_embedding"}

    sample_context:
        Optional minimal context payload that tests can pass for this framework to
        exercise context translation in a stable way.

    availability_attr:
        Optional module-level boolean that indicates whether the underlying
        framework is actually installed, e.g. "LANGCHAIN_AVAILABLE".
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

    batch_method: str
    query_method: str

    async_batch_method: Optional[str] = None
    async_query_method: Optional[str] = None

    context_kwarg: Optional[str] = None

    requires_embedding_dimension: bool = False
    embedding_dimension_kwarg: Optional[str] = None

    has_capabilities: bool = False
    capabilities_method: Optional[str] = None
    async_capabilities_method: Optional[str] = None

    has_health: bool = False
    health_method: Optional[str] = None
    async_health_method: Optional[str] = None

    availability_attr: Optional[str] = None
    minimum_framework_version: Optional[str] = None
    tested_up_to_version: Optional[str] = None

    primary_surface: Optional[str] = None
    aliases: Optional[Dict[str, str]] = None
    sample_context: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """
        Run basic consistency checks after dataclass initialization.

        This will raise early for obviously invalid descriptors (e.g. missing
        method names) and emit non-fatal warnings for softer issues.
        """
        # Since the dataclass is frozen, we must not mutate; validation is read-only.
        self.validate()

    @property
    def supports_async(self) -> bool:
        """True if any async embedding method is declared."""
        return bool(self.async_batch_method or self.async_query_method)

    def is_available(self) -> bool:
        """
        Check if the underlying framework appears available for testing.

        If availability_attr is set, this checks that boolean on the adapter
        module. Otherwise assumes the framework is available.

        Results are cached using a composite key derived from adapter_module and
        availability_attr to avoid stale cache results when descriptors are overwritten.
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

    def get_installed_framework_version(self) -> Optional[str]:
        """
        Best-effort runtime framework version probe (TEST-ONLY).

        Strategy:
        1) Import adapter module and look for module-level _FRAMEWORK_VERSION
           (this is the preferred pattern across Corpus framework adapters).
        2) If missing, attempt to infer from common framework packages using
           adapter_module name heuristics.
        3) Never raise; returns None when unknown/unavailable.

        Results are cached per adapter_module to avoid repeated imports in large
        test suites.
        """
        cache_key = _version_cache_key(self.adapter_module)
        if cache_key in _VERSION_CACHE:
            return _VERSION_CACHE[cache_key]

        version: Optional[str] = None

        try:
            module = importlib.import_module(self.adapter_module)
            v = getattr(module, "_FRAMEWORK_VERSION", None)
            if isinstance(v, str) and v.strip():
                version = v.strip()
        except Exception:
            version = None

        if version is None:
            # Heuristic fallback based on framework name (best-effort only).
            try:
                if self.name == "langchain":
                    import langchain_core as _lc  # type: ignore

                    version = getattr(_lc, "__version__", None)
                elif self.name == "llamaindex":
                    import llama_index as _li  # type: ignore

                    version = getattr(_li, "__version__", None)
                elif self.name == "semantic_kernel":
                    import semantic_kernel as _sk  # type: ignore

                    version = getattr(_sk, "__version__", None)
                elif self.name == "autogen":
                    # Microsoft AutoGen versioning varies by package; best-effort only.
                    import autogen_core as _ac  # type: ignore

                    version = getattr(_ac, "__version__", None)
                elif self.name == "crewai":
                    import crewai as _cw  # type: ignore

                    version = getattr(_cw, "__version__", None)
            except Exception:
                version = None

        # Normalize empty/whitespace to None
        if isinstance(version, str):
            version = version.strip() or None

        _VERSION_CACHE[cache_key] = version
        return version

    def version_range(self) -> Optional[str]:
        """
        Return a human-readable version range string, if any.

        Example: ">=0.1.0, <=0.3.5" or None if no constraints are set.
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
        ValueError
            If required fields like batch_method/query_method are missing,
            or when version bounds are obviously inconsistent.
        """
        # Method name checks
        if not self.batch_method or not self.query_method:
            raise ValueError(
                f"{self.name}: batch_method and query_method must both be set",
            )

        # Async consistency:
        # If you declare ANY async method, tests expect BOTH async batch + async query
        # to be present. This keeps tests and updated adapters aligned.
        if self.supports_async:
            if not self.async_batch_method or not self.async_query_method:
                raise ValueError(
                    f"{self.name}: supports_async=True requires both async_batch_method "
                    f"and async_query_method to be set "
                    f"(got async_batch_method={self.async_batch_method!r}, "
                    f"async_query_method={self.async_query_method!r})",
                )

        # adapter_class should be a bare class name, not a dotted path
        if "." in self.adapter_class:
            warnings.warn(
                f"{self.name}: adapter_class should be a class name only, "
                f"not a dotted path ({self.adapter_class!r})",
                RuntimeWarning,
                stacklevel=2,
            )

        # Dimension requirement coherence:
        # If requires_embedding_dimension=True, we strongly recommend embedding_dimension_kwarg.
        if self.requires_embedding_dimension and not self.embedding_dimension_kwarg:
            warnings.warn(
                f"{self.name}: requires_embedding_dimension=True but embedding_dimension_kwarg is None; "
                "tests may not know how to satisfy the constructor requirement",
                RuntimeWarning,
                stacklevel=2,
            )

        # Capabilities/health surface coherence:
        if self.has_capabilities:
            if not (self.capabilities_method and self.async_capabilities_method):
                # Not fatal: older descriptors may only use booleans; keep compatibility.
                warnings.warn(
                    f"{self.name}: has_capabilities=True but capabilities_method/async_capabilities_method "
                    "are not fully set; tests may need to guess method names",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if self.has_health:
            if not (self.health_method and self.async_health_method):
                warnings.warn(
                    f"{self.name}: has_health=True but health_method/async_health_method "
                    "are not fully set; tests may need to guess method names",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Aliases sanity: avoid self-referential loops and empty strings.
        if self.aliases is not None:
            if not isinstance(self.aliases, dict):
                raise ValueError(f"{self.name}: aliases must be a dict if provided")
            for k, v in self.aliases.items():
                if not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip():
                    raise ValueError(f"{self.name}: aliases keys and values must be non-empty strings")
                if k == v:
                    warnings.warn(
                        f"{self.name}: aliases contains a self-mapping {k!r} -> {v!r}",
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

        # Future: additional structural validation could go here (e.g. verifying
        # that adapter_module/adapter_class exist in a strict mode).


# ---------------------------------------------------------------------------
# Known embedding framework adapters
# ---------------------------------------------------------------------------

EMBEDDING_FRAMEWORKS: Dict[str, EmbeddingFrameworkDescriptor] = {
    # ------------------------------------------------------------------ #
    # AutoGen
    # ------------------------------------------------------------------ #
    "autogen": EmbeddingFrameworkDescriptor(
        name="autogen",
        adapter_module="corpus_sdk.embedding.framework_adapters.autogen",
        adapter_class="CorpusAutoGenEmbeddings",
        batch_method="embed_documents",
        query_method="embed_query",
        async_batch_method="aembed_documents",
        async_query_method="aembed_query",
        context_kwarg="autogen_context",
        requires_embedding_dimension=False,
        has_capabilities=True,  # capabilities / acapabilities (best-effort)
        capabilities_method="capabilities",
        async_capabilities_method="acapabilities",
        has_health=True,        # health / ahealth (best-effort)
        health_method="health",
        async_health_method="ahealth",
        availability_attr=None,
        primary_surface="embedder",
        aliases=None,
        sample_context={
            "agent_name": "test_agent",
            "conversation_id": "conv_test_1",
            "workflow_type": "conformance",
            "request_id": "req_test_1",
        },
    ),

    # ------------------------------------------------------------------ #
    # CrewAI
    # ------------------------------------------------------------------ #
    "crewai": EmbeddingFrameworkDescriptor(
        name="crewai",
        adapter_module="corpus_sdk.embedding.framework_adapters.crewai",
        adapter_class="CorpusCrewAIEmbeddings",
        batch_method="embed_documents",
        query_method="embed_query",
        async_batch_method="aembed_documents",
        async_query_method="aembed_query",
        context_kwarg="crewai_context",
        requires_embedding_dimension=False,
        has_capabilities=True,  # may raise NotImplementedError depending on adapter
        capabilities_method="capabilities",
        async_capabilities_method="acapabilities",
        has_health=True,        # may raise NotImplementedError depending on adapter
        health_method="health",
        async_health_method="ahealth",
        availability_attr=None,
        primary_surface="embedder",
        aliases=None,
        sample_context={
            "agent_role": "test_agent",
            "task_id": "task_test_1",
            "workflow": "conformance",
            "crew_id": "crew_test_1",
        },
    ),

    # ------------------------------------------------------------------ #
    # LangChain
    # ------------------------------------------------------------------ #
    "langchain": EmbeddingFrameworkDescriptor(
        name="langchain",
        adapter_module="corpus_sdk.embedding.framework_adapters.langchain",
        adapter_class="CorpusLangChainEmbeddings",
        batch_method="embed_documents",
        query_method="embed_query",
        async_batch_method="aembed_documents",
        async_query_method="aembed_query",
        context_kwarg="config",
        requires_embedding_dimension=False,
        has_capabilities=False,
        has_health=False,
        availability_attr="LANGCHAIN_AVAILABLE",
        primary_surface="embedder",
        aliases=None,
        sample_context={
            "run_id": "run_test_1",
            "run_name": "conformance",
            "tags": ["conformance", "embedding"],
            "metadata": {"suite": "embedding_conformance"},
            "configurable": {"tenant": "test"},
        },
    ),

    # ------------------------------------------------------------------ #
    # LlamaIndex
    # ------------------------------------------------------------------ #
    # Note: LlamaIndex uses internal methods on BaseEmbedding for query/text
    # embedding. We treat these as the primary API surfaces for conformance.
    "llamaindex": EmbeddingFrameworkDescriptor(
        name="llamaindex",
        adapter_module="corpus_sdk.embedding.framework_adapters.llamaindex",
        adapter_class="CorpusLlamaIndexEmbeddings",
        batch_method="_get_text_embeddings",
        query_method="_get_query_embedding",
        async_batch_method="_aget_text_embeddings",
        async_query_method="_aget_query_embedding",
        context_kwarg="llamaindex_context",
        requires_embedding_dimension=True,  # enforced by __init__
        embedding_dimension_kwarg="embedding_dimension",
        has_capabilities=False,
        has_health=False,
        availability_attr="LLAMAINDEX_AVAILABLE",
        primary_surface="baseembedding",
        aliases=None,
        sample_context={
            "node_ids": ["node_test_1", "node_test_2"],
            "index_id": "index_test_1",
            "trace_id": "trace_test_1",
            "workflow": "conformance",
        },
    ),

    # ------------------------------------------------------------------ #
    # Semantic Kernel
    # ------------------------------------------------------------------ #
    # Primary API is generate_embedding(s), with async variants. There are
    # additional aliases (embed_documents/embed_query) that are covered by
    # Semantic Kernel–specific tests.
    "semantic_kernel": EmbeddingFrameworkDescriptor(
        name="semantic_kernel",
        adapter_module="corpus_sdk.embedding.framework_adapters.semantic_kernel",
        adapter_class="CorpusSemanticKernelEmbeddings",
        batch_method="generate_embeddings",
        query_method="generate_embedding",
        async_batch_method="generate_embeddings_async",
        async_query_method="generate_embedding_async",
        context_kwarg="sk_context",
        requires_embedding_dimension=True,  # enforced by __init__
        embedding_dimension_kwarg="embedding_dimension",
        has_capabilities=False,
        has_health=False,
        availability_attr="SEMANTIC_KERNEL_AVAILABLE",
        primary_surface="embedding_generator",
        aliases={
            "embed_documents": "generate_embeddings",
            "embed_query": "generate_embedding",
            "aembed_documents": "generate_embeddings_async",
            "aembed_query": "generate_embedding_async",
        },
        sample_context={
            "plugin_name": "test_plugin",
            "function_name": "test_function",
            "kernel_id": "kernel_test_1",
            "memory_type": "conformance",
            "request_id": "req_test_1",
        },
    ),
}


def get_embedding_framework_descriptor(name: str) -> EmbeddingFrameworkDescriptor:
    """
    Look up an embedding framework descriptor by name.

    Raises
    ------
    KeyError if the framework is not registered.
    """
    return EMBEDDING_FRAMEWORKS[name]


def get_embedding_framework_descriptor_safe(
    name: str,
) -> Optional[EmbeddingFrameworkDescriptor]:
    """
    Safe lookup for an embedding framework descriptor.

    Returns None instead of raising KeyError when the framework is unknown.
    """
    return EMBEDDING_FRAMEWORKS.get(name)


def has_framework(name: str) -> bool:
    """
    Return True if a framework with the given name is registered.
    """
    return name in EMBEDDING_FRAMEWORKS


def iter_embedding_framework_descriptors() -> Iterable[EmbeddingFrameworkDescriptor]:
    """
    Iterate over all registered embedding framework descriptors.
    """
    return EMBEDDING_FRAMEWORKS.values()


def iter_available_framework_descriptors() -> Iterable[EmbeddingFrameworkDescriptor]:
    """
    Iterate over descriptors for frameworks that appear available.

    This is useful for tests that should only run when the underlying
    framework (LangChain, LlamaIndex, Semantic Kernel, etc.) is installed.
    """
    return (desc for desc in EMBEDDING_FRAMEWORKS.values() if desc.is_available())


def register_framework_descriptor(
    descriptor: EmbeddingFrameworkDescriptor,
    overwrite: bool = False,
) -> None:
    """
    Register a new framework descriptor dynamically (TEST-ONLY).

    This is primarily intended for test scenarios where you want to plug in
    an experimental or third-party embedding framework adapter.

    Parameters
    ----------
    descriptor:
        Descriptor to register. Its `name` is used as the registry key.
    overwrite:
        If False (default), attempting to overwrite an existing entry will
        raise KeyError. If True, an existing entry with the same name is
        replaced (with a warning).
    """
    if descriptor.name in EMBEDDING_FRAMEWORKS and not overwrite:
        raise KeyError(f"Framework {descriptor.name!r} is already registered")

    if descriptor.name in EMBEDDING_FRAMEWORKS and overwrite:
        warnings.warn(
            f"Framework {descriptor.name!r} is being overwritten in the registry",
            RuntimeWarning,
            stacklevel=2,
        )

    EMBEDDING_FRAMEWORKS[descriptor.name] = descriptor

    # Reset availability cache for this descriptor so future checks re-evaluate.
    _AVAILABILITY_CACHE.pop(_availability_cache_key(descriptor.adapter_module, descriptor.availability_attr), None)
    _VERSION_CACHE.pop(_version_cache_key(descriptor.adapter_module), None)


def unregister_framework_descriptor(
    name: str,
    ignore_missing: bool = True,
) -> None:
    """
    Unregister a framework descriptor dynamically (TEST-ONLY).

    Useful for tests that temporarily override or replace registry entries.

    Parameters
    ----------
    name:
        Name of the framework to unregister.
    ignore_missing:
        If False, raise KeyError when the framework is not registered.
        If True (default), missing entries are ignored.
    """
    if name in EMBEDDING_FRAMEWORKS:
        desc = EMBEDDING_FRAMEWORKS[name]
        del EMBEDDING_FRAMEWORKS[name]
        _AVAILABILITY_CACHE.pop(_availability_cache_key(desc.adapter_module, desc.availability_attr), None)
        _VERSION_CACHE.pop(_version_cache_key(desc.adapter_module), None)
    elif not ignore_missing:
        raise KeyError(f"Framework {name!r} is not registered")


__all__ = [
    "EmbeddingFrameworkDescriptor",
    "EMBEDDING_FRAMEWORKS",
    "get_embedding_framework_descriptor",
    "get_embedding_framework_descriptor_safe",
    "has_framework",
    "iter_embedding_framework_descriptors",
    "iter_available_framework_descriptors",
    "register_framework_descriptor",
    "unregister_framework_descriptor",
]
