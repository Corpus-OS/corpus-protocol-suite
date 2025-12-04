# tests/frameworks/registries/llm_registry.py
"""
Registry of LLM framework adapters used by the conformance test suite.

This module is TEST-ONLY. It provides lightweight metadata describing how to:
- Import each LLM framework adapter
- Construct its client
- Call its sync / async completion & streaming methods
- Pass framework-specific context or config
- Know which extra semantics (token counting, health, capabilities) to expect

Contract tests in tests/frameworks/llm/ use this registry to stay
framework-agnostic. Adding a new LLM framework typically means:

1. Implement the adapter under corpus_sdk.llm.framework_adapters.*.
2. Add a new LLMFrameworkDescriptor entry here (or register dynamically).
3. Run the LLM contract tests.

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
class LLMFrameworkDescriptor:
    """
    Description of an LLM framework adapter.

    Fields
    ------
    name:
        Short, stable identifier (e.g. "autogen", "langchain").
    adapter_module:
        Dotted import path for the adapter module.
    adapter_class:
        Name of the adapter class within adapter_module.

    completion_method:
        Name of the *sync* completion / generation method.
        (For frameworks that only expose async APIs, leave this None.)
    async_completion_method:
        Name of the *async* completion / generation method, or None.

    streaming_method:
        Name of the *sync* streaming method, or None.
        For frameworks that re-use completion_method with a streaming kwarg,
        leave this as None and set streaming_kwarg instead.
    async_streaming_method:
        Name of the *async* streaming method, or None.

    streaming_kwarg:
        Name of the kwarg that toggles streaming on completion_method /
        async_completion_method (e.g. "stream" for AutoGen), or None.

    token_count_method:
        Name of the token-counting method (sync), or None if not exposed.

    context_kwarg:
        Name of the kwargs parameter used for framework-specific context /
        configuration (e.g. "conversation", "task", "config", "callback_manager",
        "ctx"), or None if not applicable or positional.

    has_capabilities:
        True if the adapter exposes a capabilities()/acapabilities() surface.

    has_health:
        True if the adapter exposes a health()/ahealth() surface.

    supports_streaming:
        True if the adapter is expected to support streaming (sync or async).

    supports_token_counting:
        True if the adapter exposes a public token-counting helper.

    availability_attr:
        Optional module-level boolean that indicates whether the underlying
        framework is actually installed, e.g. "LANGCHAIN_LLM_AVAILABLE".
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

    completion_method: Optional[str]
    async_completion_method: Optional[str] = None

    streaming_method: Optional[str] = None
    async_streaming_method: Optional[str] = None

    streaming_kwarg: Optional[str] = None

    token_count_method: Optional[str] = None
    context_kwarg: Optional[str] = None

    has_capabilities: bool = False
    has_health: bool = False

    supports_streaming: bool = False
    supports_token_counting: bool = False

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
        return bool(self.async_completion_method or self.async_streaming_method)

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
            If required fields like completion_method are missing.
        """
        # Required completion method (unless this is explicitly async-only)
        if not self.completion_method and not self.async_completion_method:
            raise ValueError(
                f"{self.name}: at least one of completion_method or "
                f"async_completion_method must be set",
            )

        # Async streaming without any async completion or streaming kwarg
        if self.async_streaming_method and not (
            self.async_completion_method or self.streaming_kwarg
        ):
            warnings.warn(
                f"{self.name}: async_streaming_method is set but no "
                f"async_completion_method or streaming_kwarg is declared",
                RuntimeWarning,
                stacklevel=2,
            )

        # Streaming flags vs method names (soft warnings)
        if self.supports_streaming and not (
            self.streaming_method
            or self.async_streaming_method
            or self.streaming_kwarg
        ):
            warnings.warn(
                f"{self.name}: supports_streaming is True but neither "
                f"streaming_method, async_streaming_method nor streaming_kwarg "
                f"is set",
                RuntimeWarning,
                stacklevel=2,
            )

        if self.supports_token_counting and not self.token_count_method:
            warnings.warn(
                f"{self.name}: supports_token_counting is True but "
                f"token_count_method is not set",
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
# Known LLM framework adapters
# ---------------------------------------------------------------------------

LLM_FRAMEWORKS: Dict[str, LLMFrameworkDescriptor] = {
    # ------------------------------------------------------------------ #
    # AutoGen LLM
    # ------------------------------------------------------------------ #
    "autogen": LLMFrameworkDescriptor(
        name="autogen",
        adapter_module="corpus_sdk.llm.framework_adapters.autogen",
        adapter_class="CorpusAutoGenChatClient",
        # OpenAI-style create/acreate with stream=True for streaming.
        completion_method="create",
        async_completion_method="acreate",
        streaming_method=None,
        async_streaming_method=None,
        streaming_kwarg="stream",
        token_count_method=None,  # token counting is via underlying adapter, not exposed here
        context_kwarg="conversation",  # primary framework-specific context kwarg
        has_capabilities=False,
        has_health=False,
        supports_streaming=True,
        supports_token_counting=False,
        availability_attr=None,
    ),

    # ------------------------------------------------------------------ #
    # CrewAI LLM
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
        token_count_method=None,
        # Primary OperationContext override kwarg; other CrewAI-specific
        # context (agent_role, crew_id, etc.) is passed via kwargs as well.
        context_kwarg="ctx",
        has_capabilities=False,
        has_health=False,
        supports_streaming=True,
        supports_token_counting=False,
        availability_attr=None,
    ),

    # ------------------------------------------------------------------ #
    # LangChain LLM
    # ------------------------------------------------------------------ #
    "langchain": LLMFrameworkDescriptor(
        name="langchain",
        adapter_module="corpus_sdk.llm.framework_adapters.langchain",
        adapter_class="CorpusLangChainLLM",
        # LangChain calls these internally; tests may call them directly.
        completion_method="_generate",
        async_completion_method="_agenerate",
        streaming_method="_stream",
        async_streaming_method="_astream",
        streaming_kwarg=None,
        token_count_method="get_num_tokens_from_messages",
        context_kwarg="config",
        has_capabilities=False,
        has_health=False,
        supports_streaming=True,
        supports_token_counting=True,
        availability_attr="LANGCHAIN_LLM_AVAILABLE"
        if hasattr(importlib, "import_module")
        else None,  # can be wired up in the adapter module, if present
    ),

    # ------------------------------------------------------------------ #
    # LlamaIndex LLM
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
        token_count_method="count_tokens",
        context_kwarg="callback_manager",
        has_capabilities=False,
        has_health=False,
        supports_streaming=True,
        supports_token_counting=True,
        availability_attr=None,
    ),

    # ------------------------------------------------------------------ #
    # Semantic Kernel LLM
    # ------------------------------------------------------------------ #
    "semantic_kernel": LLMFrameworkDescriptor(
        name="semantic_kernel",
        adapter_module="corpus_sdk.llm.framework_adapters.semantic_kernel",
        adapter_class="CorpusSemanticKernelChatCompletion",
        # Semantic Kernel adapter is async-only for completions.
        completion_method=None,
        async_completion_method="get_chat_message_content",
        streaming_method=None,
        async_streaming_method="get_streaming_chat_message_content",
        streaming_kwarg=None,
        token_count_method="count_tokens",
        context_kwarg=None,  # context is derived from PromptExecutionSettings
        has_capabilities=False,
        has_health=False,
        supports_streaming=True,
        supports_token_counting=True,
        availability_attr=None,
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


def get_llm_framework_descriptor_safe(
    name: str,
) -> Optional[LLMFrameworkDescriptor]:
    """
    Safe lookup for an LLM framework descriptor.

    Returns None instead of raising KeyError when the framework is unknown.
    """
    return LLM_FRAMEWORKS.get(name)


def has_llm_framework(name: str) -> bool:
    """
    Return True if an LLM framework with the given name is registered.
    """
    return name in LLM_FRAMEWORKS


# Backwards-compatible alias, if anything still uses the old generic name.
def has_framework(name: str) -> bool:
    return has_llm_framework(name)


def iter_llm_framework_descriptors() -> Iterable[LLMFrameworkDescriptor]:
    """
    Iterate over all registered LLM framework descriptors.
    """
    return LLM_FRAMEWORKS.values()


def iter_available_llm_framework_descriptors() -> Iterable[LLMFrameworkDescriptor]:
    """
    Iterate over descriptors for LLM frameworks that appear available.

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
        replaced.
    """
    if descriptor.name in LLM_FRAMEWORKS and not overwrite:
        raise KeyError(f"Framework {descriptor.name!r} is already registered")

    LLM_FRAMEWORKS[descriptor.name] = descriptor


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
]

