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
class EmbeddingFrameworkDescriptor:
    """
    Description of an embedding framework adapter.

    Fields
    ------
    name:
        Short, stable identifier (e.g. "autogen", "langchain").
    adapter_module:
        Dotted import path for the adapter module.
    adapter_class:
        Name of the adapter class within adapter_module.

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

    context_kwarg:
        Name of the kwargs parameter used for framework-specific context
        (e.g. "autogen_context", "config", "llamaindex_context", "sk_context").

    requires_embedding_dimension:
        True if the adapter requires a known embedding dimension up-front
        (either via adapter.get_embedding_dimension() or an explicit override).

    has_capabilities:
        True if the adapter exposes a capabilities()/acapabilities() surface.

    has_health:
        True if the adapter exposes a health()/ahealth() surface.

    availability_attr:
        Optional module-level boolean that indicates whether the underlying
        framework is actually installed, e.g. "LANGCHAIN_AVAILABLE".
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

    batch_method: str
    query_method: str

    async_batch_method: Optional[str] = None
    async_query_method: Optional[str] = None

    context_kwarg: Optional[str] = None

    requires_embedding_dimension: bool = False
    has_capabilities: bool = False
    has_health: bool = False

    availability_attr: Optional[str] = None
    minimum_framework_version: Optional[str] = None
    tested_up_to_version: Optional[str] = None

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
            If required fields like batch_method/query_method are missing.
        """
        # Method name checks
        if not self.batch_method or not self.query_method:
            raise ValueError(
                f"{self.name}: batch_method and query_method must both be set",
            )

        # Async consistency warning (soft)
        if self.async_query_method and not self.async_batch_method:
            warnings.warn(
                f"{self.name}: async_query_method is set but async_batch_method is None",
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

        # Future: when minimum_framework_version / tested_up_to_version are set
        # we could enforce ordering here (e.g. minimum <= tested_up_to_version).
        # For now they remain informational only.


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
        has_health=True,        # health / ahealth (best-effort)
        availability_attr=None,
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
        has_health=True,        # may raise NotImplementedError depending on adapter
        availability_attr=None,
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
        has_capabilities=False,
        has_health=False,
        availability_attr="LLAMAINDEX_AVAILABLE",
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
        has_capabilities=False,
        has_health=False,
        availability_attr="SEMANTIC_KERNEL_AVAILABLE",
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
        replaced.
    """
    if descriptor.name in EMBEDDING_FRAMEWORKS and not overwrite:
        raise KeyError(f"Framework {descriptor.name!r} is already registered")

    EMBEDDING_FRAMEWORKS[descriptor.name] = descriptor


__all__ = [
    "EmbeddingFrameworkDescriptor",
    "EMBEDDING_FRAMEWORKS",
    "get_embedding_framework_descriptor",
    "get_embedding_framework_descriptor_safe",
    "has_framework",
    "iter_embedding_framework_descriptors",
    "iter_available_framework_descriptors",
    "register_framework_descriptor",
]

