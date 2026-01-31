# tests/frameworks/registries/llm_registry.py
"""
Registry of LLM framework adapters used by the conformance test suite.

This module is TEST-ONLY. It provides lightweight metadata describing how to:
- Import each LLM framework adapter
- Construct its client
- Call its sync / async completion & streaming methods
- Pass framework-specific context
- Know which extra semantics (health, capabilities, token counting) to expect

Contract tests in tests/frameworks/llm/ use this registry to stay
framework-agnostic. Adding a new LLM framework typically means:

1. Implement the adapter under corpus_sdk.llm.framework_adapters.*.
2. Add a new LLMFrameworkDescriptor entry here (or register dynamically).
3. Run the LLM contract tests.

Version fields
--------------
`minimum_framework_version` and `tested_up_to_version` are currently informational,
but we validate that the range is coherent when possible. In the future, tests may
use them to conditionally skip or adjust expectations based on the installed
framework version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional
import importlib
import warnings

try:  # Optional, used only for version ordering validation
    from packaging.version import Version
except Exception:  # pragma: no cover - packaging may not be installed
    Version = None  # type: ignore[assignment]

# Simple in-memory cache to avoid repeatedly importing modules for availability checks.
_AVAILABILITY_CACHE: Dict[str, bool] = {}


@dataclass(frozen=True)
class LLMFrameworkDescriptor:
    """
    Description of an LLM framework adapter (TEST-ONLY).

    Fields
    ------
    name:
        Short, stable identifier (e.g. "autogen", "langchain").
    adapter_module:
        Dotted import path for the adapter module.
    adapter_class:
        Name of the adapter class within adapter_module (bare class name).

    completion_method:
        Name of the *sync* completion method, or None if not supported.
    async_completion_method:
        Name of the *async* completion method, or None if not supported.

    streaming_method:
        Name of the *sync* streaming method, or None.
        This is for frameworks that expose a separate streaming surface
        (e.g. LangChain `_stream`, CrewAI `stream`).
    async_streaming_method:
        Name of the *async* streaming method, or None.

    token_count_method:
        Name of the *sync* token-counting method, or None.
    async_token_count_method:
        Name of the *async* token-counting method, or None.

    streaming_kwarg:
        Optional name of a boolean kwarg on the completion method that
        enables streaming (e.g. `stream=True` for AutoGen/OpenAI-style APIs).

    streaming_style:
        How streaming is accessed:
        - "method": use streaming_method / async_streaming_method
        - "kwarg": use completion_method / async_completion_method with
                   `streaming_kwarg=True`
        - "none": framework does not provide streaming

    context_kwarg:
        Name of the kwargs parameter used for framework-specific context
        (e.g. "conversation", "task", "config", "callback_manager", "settings").

    has_capabilities:
        True if the adapter exposes a capabilities surface.

    capabilities_method / async_capabilities_method:
        Method names for capabilities on the adapter.

        NOTE:
            These are optional for backwards compatibility; when omitted and
            has_capabilities=True, conformance tests may still assume a
            conventional async method name ("capabilities"). Explicitly
            listing method names here keeps the registry unambiguous.

    has_health:
        True if the adapter exposes a health surface.

    health_method / async_health_method:
        Method names for health on the adapter.

        NOTE:
            As with capabilities, these are optional for backwards compatibility.
            Explicitly listing method names is preferred for clarity.

    supports_streaming:
        True if the adapter is expected to support streaming responses.

    supports_token_count:
        True if the adapter is expected to support token counting.

    availability_attr:
        Optional module-level boolean that indicates whether the underlying
        framework is actually installed, e.g. "LANGCHAIN_LLM_AVAILABLE".
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

    completion_method: Optional[str] = None
    async_completion_method: Optional[str] = None

    streaming_method: Optional[str] = None
    async_streaming_method: Optional[str] = None

    token_count_method: Optional[str] = None
    async_token_count_method: Optional[str] = None

    streaming_kwarg: Optional[str] = None
    streaming_style: Literal["method", "kwarg", "none"] = "method"

    context_kwarg: Optional[str] = None

    # Extra surfaces (capabilities/health) are intentionally represented
    # explicitly so the conformance suite can decide what to test per framework.
    has_capabilities: bool = False
    has_health: bool = False

    # Explicit method names for capabilities/health (prefer async names).
    # These are optional to preserve compatibility with existing test logic,
    # but descriptors SHOULD set them when has_capabilities/has_health is True.
    capabilities_method: Optional[str] = None
    async_capabilities_method: Optional[str] = None
    health_method: Optional[str] = None
    async_health_method: Optional[str] = None

    supports_streaming: bool = False
    supports_token_count: bool = False

    availability_attr: Optional[str] = None
    minimum_framework_version: Optional[str] = None
    tested_up_to_version: Optional[str] = None

    def __post_init__(self) -> None:
        """
        Run basic consistency checks after dataclass initialization.

        This will raise early for obviously invalid descriptors (e.g. missing
        completion methods) and emit non-fatal warnings for softer issues.
        """
        self.validate()

    @property
    def supports_async(self) -> bool:
        """
        True if any async method is declared.

        This includes async completion/streaming/token-count surfaces as well
        as async capability/health methods when explicitly declared.
        """
        return bool(
            self.async_completion_method
            or self.async_streaming_method
            or self.async_token_count_method
            or self.async_capabilities_method
            or self.async_health_method
        )

    def is_available(self) -> bool:
        """
        Check if the underlying framework appears available for testing.

        If availability_attr is set, this checks that boolean on the adapter
        module. Otherwise assumes the framework is available.

        Results are cached per-descriptor name to avoid repeated imports in
        large test suites.
        """
        cache_key = self.name
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
            If required fields like completion_method/async_completion_method
            are missing, or when version bounds are obviously inconsistent.
        """
        # At least one completion entrypoint must exist.
        if not (self.completion_method or self.async_completion_method):
            raise ValueError(
                f"{self.name}: at least one of completion_method or "
                f"async_completion_method must be set",
            )

        # Async completion without sync counterpart (soft warning).
        if self.async_completion_method and not self.completion_method:
            warnings.warn(
                f"{self.name}: async_completion_method is set but "
                f"completion_method is None (async should usually "
                f"have a sync counterpart)",
                RuntimeWarning,
                stacklevel=2,
            )

        # Async streaming should have an async completion counterpart.
        if self.async_streaming_method and not self.async_completion_method:
            warnings.warn(
                f"{self.name}: async_streaming_method is set but "
                f"async_completion_method is None (async streaming "
                f"should have an async completion counterpart)",
                RuntimeWarning,
                stacklevel=2,
            )

        # Sync streaming without async counterpart (soft warning).
        if self.streaming_method and not self.async_streaming_method:
            warnings.warn(
                f"{self.name}: streaming_method is set but "
                f"async_streaming_method is None (consider adding async streaming "
                f"for parity)",
                RuntimeWarning,
                stacklevel=2,
            )

        # Async token-count without sync counterpart (soft warning).
        if self.async_token_count_method and not self.token_count_method:
            warnings.warn(
                f"{self.name}: async_token_count_method is set but "
                f"token_count_method is None (async should have a sync counterpart)",
                RuntimeWarning,
                stacklevel=2,
            )

        # Capability/health method presence checks (kept conservative).
        # If a descriptor claims the surface exists, it SHOULD specify the async method
        # name explicitly to keep the registry unambiguous.
        if self.has_capabilities and not (self.capabilities_method or self.async_capabilities_method):
            warnings.warn(
                f"{self.name}: has_capabilities is True but no "
                f"capabilities_method/async_capabilities_method is set "
                f"(tests may assume a conventional async method name)",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.has_health and not (self.health_method or self.async_health_method):
            warnings.warn(
                f"{self.name}: has_health is True but no "
                f"health_method/async_health_method is set "
                f"(tests may assume a conventional async method name)",
                RuntimeWarning,
                stacklevel=2,
            )

        # Streaming flags vs methods/kwarg (soft warnings).
        if self.supports_streaming and not (
            self.streaming_method
            or self.async_streaming_method
            or self.streaming_kwarg
        ):
            warnings.warn(
                f"{self.name}: supports_streaming is True but no "
                f"streaming_method/async_streaming_method/streaming_kwarg is set",
                RuntimeWarning,
                stacklevel=2,
            )

        if self.supports_token_count and not (
            self.token_count_method or self.async_token_count_method
        ):
            warnings.warn(
                f"{self.name}: supports_token_count is True but "
                f"no token_count_method/async_token_count_method is set",
                RuntimeWarning,
                stacklevel=2,
            )

        # streaming_style sanity checks
        if self.streaming_style == "method":
            if not (self.streaming_method or self.async_streaming_method):
                warnings.warn(
                    f"{self.name}: streaming_style='method' but "
                    f"no streaming_method/async_streaming_method is set",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if self.streaming_style == "kwarg":
            if not self.streaming_kwarg:
                raise ValueError(
                    f"{self.name}: streaming_style='kwarg' requires "
                    f"streaming_kwarg to be set",
                )
            if self.streaming_method or self.async_streaming_method:
                warnings.warn(
                    f"{self.name}: streaming_style='kwarg' but streaming_method "
                    f"or async_streaming_method is also set (tests may prefer "
                    f"one style; this is potentially ambiguous)",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if self.streaming_style == "none" and self.supports_streaming:
            warnings.warn(
                f"{self.name}: streaming_style='none' but supports_streaming=True",
                RuntimeWarning,
                stacklevel=2,
            )

        # adapter_class should be a bare class name, not a dotted path.
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
# Known LLM framework adapters
# ---------------------------------------------------------------------------

LLM_FRAMEWORKS: Dict[str, LLMFrameworkDescriptor] = {
    # ------------------------------------------------------------------ #
    # AutoGen
    # ------------------------------------------------------------------ #
    "autogen": LLMFrameworkDescriptor(
        name="autogen",
        adapter_module="corpus_sdk.llm.framework_adapters.autogen",
        adapter_class="CorpusAutoGenChatClient",
        completion_method="create",
        async_completion_method="acreate",
        streaming_method=None,
        async_streaming_method=None,
        # AutoGen uses stream=True on completion methods
        streaming_kwarg="stream",
        streaming_style="kwarg",
        token_count_method=None,
        async_token_count_method=None,
        context_kwarg="conversation",
        # Conformance expectation: the adapter layer exposes these surfaces even when
        # the underlying framework does not have a first-class analogue.
        has_capabilities=True,
        has_health=True,
        # Prefer async method names because the conformance suite is async-first.
        async_capabilities_method="capabilities",
        async_health_method="health",
        supports_streaming=True,
        supports_token_count=False,
        availability_attr=None,
        minimum_framework_version=None,
        tested_up_to_version=None,
    ),

    # ------------------------------------------------------------------ #
    # CrewAI
    # ------------------------------------------------------------------ #
    "crewai": LLMFrameworkDescriptor(
        name="crewai",
        adapter_module="corpus_sdk.llm.framework_adapters.crewai",
        adapter_class="CorpusCrewAILLM",
        completion_method="complete",
        async_completion_method="acomplete",
        streaming_method="stream",
        async_streaming_method="astream",
        streaming_kwarg=None,
        streaming_style="method",
        token_count_method=None,
        async_token_count_method=None,
        context_kwarg="task",
        has_capabilities=True,
        has_health=True,
        async_capabilities_method="capabilities",
        async_health_method="health",
        supports_streaming=True,
        supports_token_count=False,
        availability_attr=None,
        minimum_framework_version=None,
        tested_up_to_version=None,
    ),

    # ------------------------------------------------------------------ #
    # LangChain
    # ------------------------------------------------------------------ #
    "langchain": LLMFrameworkDescriptor(
        name="langchain",
        adapter_module="corpus_sdk.llm.framework_adapters.langchain",
        adapter_class="CorpusLangChainLLM",
        # LangChain generation is via internal _generate/_agenerate
        completion_method="_generate",
        async_completion_method="_agenerate",
        streaming_method="_stream",
        async_streaming_method="_astream",
        streaming_kwarg=None,
        streaming_style="method",
        token_count_method="get_num_tokens_from_messages",
        async_token_count_method=None,
        context_kwarg="config",
        has_capabilities=True,
        has_health=True,
        async_capabilities_method="capabilities",
        async_health_method="health",
        supports_streaming=True,
        supports_token_count=True,
        availability_attr="LANGCHAIN_LLM_AVAILABLE",
        minimum_framework_version=None,
        tested_up_to_version=None,
    ),

    # ------------------------------------------------------------------ #
    # LlamaIndex
    # ------------------------------------------------------------------ #
    "llamaindex": LLMFrameworkDescriptor(
        name="llamaindex",
        adapter_module="corpus_sdk.llm.framework_adapters.llamaindex",
        adapter_class="CorpusLlamaIndexLLM",
        completion_method="chat",
        async_completion_method="achat",
        streaming_method="stream_chat",
        async_streaming_method="astream_chat",
        streaming_kwarg=None,
        streaming_style="method",
        token_count_method="count_tokens",
        async_token_count_method=None,
        context_kwarg="callback_manager",
        has_capabilities=True,
        has_health=True,
        async_capabilities_method="capabilities",
        async_health_method="health",
        supports_streaming=True,
        supports_token_count=True,
        availability_attr=None,
        minimum_framework_version=None,
        tested_up_to_version=None,
    ),

    # ------------------------------------------------------------------ #
    # Semantic Kernel
    # ------------------------------------------------------------------ #
    "semantic_kernel": LLMFrameworkDescriptor(
        name="semantic_kernel",
        adapter_module="corpus_sdk.llm.framework_adapters.semantic_kernel",
        adapter_class="CorpusSemanticKernelChatCompletion",
        completion_method=None,
        async_completion_method="get_chat_message_content",
        streaming_method=None,
        async_streaming_method="get_streaming_chat_message_content",
        streaming_kwarg=None,
        streaming_style="method",
        token_count_method="count_tokens",
        async_token_count_method=None,
        context_kwarg="settings",
        has_capabilities=True,
        has_health=True,
        async_capabilities_method="capabilities",
        async_health_method="health",
        supports_streaming=True,
        supports_token_count=True,
        availability_attr=None,
        minimum_framework_version=None,
        tested_up_to_version=None,
    ),
}


def get_llm_framework_descriptor(name: str) -> LLMFrameworkDescriptor:
    """
    Look up an LLM framework descriptor by name.

    Raises
    ------
    KeyError if the framework is not registered.
    """
    return LLM_FRAMEWORKS[name]


def get_llm_framework_descriptor_safe(name: str) -> Optional[LLMFrameworkDescriptor]:
    """
    Safe lookup for an LLM framework descriptor.

    Returns None instead of raising KeyError when the framework is unknown.
    """
    return LLM_FRAMEWORKS.get(name)


def has_llm_framework(name: str) -> bool:
    """
    Return True if a framework with the given name is registered.
    """
    return name in LLM_FRAMEWORKS


# Backwards-compatible alias, if anything still uses the old name.
def has_framework(name: str) -> bool:
    return has_llm_framework(name)


def iter_llm_framework_descriptors() -> Iterable[LLMFrameworkDescriptor]:
    """
    Iterate over all registered LLM framework descriptors.
    """
    return LLM_FRAMEWORKS.values()


def iter_available_llm_framework_descriptors() -> Iterable[LLMFrameworkDescriptor]:
    """
    Iterate over descriptors for frameworks that appear available.

    This is useful for tests that should only run when the underlying
    framework (LangChain, LlamaIndex, Semantic Kernel, etc.) is installed.
    """
    return (desc for desc in LLM_FRAMEWORKS.values() if desc.is_available())


def register_llm_framework_descriptor(
    descriptor: LLMFrameworkDescriptor,
    overwrite: bool = False,
) -> None:
    """
    Register a new LLM framework descriptor dynamically (TEST-ONLY).

    This is primarily intended for test scenarios where you want to plug in
    an experimental or third-party LLM framework adapter.

    Parameters
    ----------
    descriptor:
        Descriptor to register. Its `name` is used as the registry key.
    overwrite:
        If False (default), attempting to overwrite an existing entry will
        raise KeyError. If True, an existing entry with the same name is
        replaced (with a warning).
    """
    if descriptor.name in LLM_FRAMEWORKS and not overwrite:
        raise KeyError(f"Framework {descriptor.name!r} is already registered")

    if descriptor.name in LLM_FRAMEWORKS and overwrite:
        warnings.warn(
            f"Framework {descriptor.name!r} is being overwritten in the LLM registry",
            RuntimeWarning,
            stacklevel=2,
        )

    LLM_FRAMEWORKS[descriptor.name] = descriptor
    # Reset availability cache for this descriptor so future checks re-evaluate.
    _AVAILABILITY_CACHE.pop(descriptor.name, None)


def unregister_llm_framework_descriptor(
    name: str,
    ignore_missing: bool = True,
) -> None:
    """
    Unregister an LLM framework descriptor dynamically (TEST-ONLY).

    Useful for tests that temporarily override or replace registry entries.

    Parameters
    ----------
    name:
        Name of the framework to unregister.
    ignore_missing:
        If False, raise KeyError when the framework is not registered.
        If True (default), missing entries are ignored.
    """
    if name in LLM_FRAMEWORKS:
        del LLM_FRAMEWORKS[name]
        _AVAILABILITY_CACHE.pop(name, None)
    elif not ignore_missing:
        raise KeyError(f"Framework {name!r} is not registered")


__all__ = [
    "LLMFrameworkDescriptor",
    "LLM_FRAMEWORKS",
    "get_llm_framework_descriptor",
    "get_llm_framework_descriptor_safe",
    "has_llm_framework",
    "has_framework",
    "iter_llm_framework_descriptors",
    "iter_available_llm_framework_descriptors",
    "register_llm_framework_descriptor",
    "unregister_llm_framework_descriptor",
]
