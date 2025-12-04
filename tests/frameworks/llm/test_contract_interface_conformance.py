# tests/frameworks/llm/test_contract_interface_conformance.py

from __future__ import annotations

import asyncio
import importlib
import inspect
from collections.abc import Mapping
from typing import Any, Callable

import pytest

from tests.frameworks.registries.llm_registry import (
    LLMFrameworkDescriptor,
    iter_llm_framework_descriptors,
)

# ---------------------------------------------------------------------------
# Constants (shared test inputs)
# ---------------------------------------------------------------------------

SYNC_COMPLETION_TEXT = "llm-sync-completion"
SYNC_STREAM_TEXT = "llm-sync-stream"
ASYNC_COMPLETION_TEXT = "llm-async-completion"
ASYNC_STREAM_TEXT = "llm-async-stream"
CONTEXT_COMPLETION_TEXT = "llm-context-completion"
TOKEN_COUNT_TEXT = "llm-token-count"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=list(iter_llm_framework_descriptors()),
    name="framework_descriptor",
)
def framework_descriptor_fixture(
    request: pytest.FixtureRequest,
) -> LLMFrameworkDescriptor:
    """
    Parameterized over all registered LLM framework descriptors.

    Frameworks that are not actually available in the environment (e.g. the
    underlying LangChain / LlamaIndex / Semantic Kernel libraries are missing)
    are skipped via descriptor.is_available().
    """
    descriptor: LLMFrameworkDescriptor = request.param
    if not descriptor.is_available():
        pytest.skip(f"Framework '{descriptor.name}' not available in this environment")
    return descriptor


@pytest.fixture
def llm_client_instance(
    framework_descriptor: LLMFrameworkDescriptor,
    adapter: Any,
) -> Any:
    """
    Construct a concrete LLM client/adapter instance for the given descriptor.

    This uses the registry metadata to import the client class and instantiate
    it with the *generic* Corpus LLM adapter provided by the top-level pytest
    plugin (see conftest.py).

    The client class is expected to wrap an LLMProtocolV1 implementation.
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    client_cls = getattr(module, framework_descriptor.adapter_class)

    # All LLM framework adapters take an LLMProtocolV1 implementation under a
    # consistent kwarg name (llm_adapter).
    init_kwargs: dict[str, Any] = {"llm_adapter": adapter}

    # Additional framework-specific kwargs can be added here if needed.

    instance = client_cls(**init_kwargs)
    return instance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_method(instance: Any, name: str | None) -> Callable[..., Any]:
    """
    Helper to fetch a method from the instance and assert it is callable.

    If name is None, this fails fast with a clear assertion message.
    """
    assert name, "Expected a non-empty method name"
    attr = getattr(instance, name, None)
    assert callable(attr), f"{instance!r} missing expected callable method {name!r}"
    return attr


def _run_async_if_needed(coro: Any) -> Any:
    """
    Run an async coroutine, handling existing event loops gracefully.

    Used for optional async surfaces (e.g. acapabilities/ahealth) in tests
    that are not themselves marked async.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # Fall back to the current event loop if one is already running.
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


def _build_sync_completion_args(
    descriptor: LLMFrameworkDescriptor,
    text: str,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Build positional args + kwargs for a sync completion call appropriate
    for the given framework.
    """
    kwargs: dict[str, Any] = {}
    if descriptor.context_kwarg:
        kwargs[descriptor.context_kwarg] = {}

    # Framework-specific message shapes
    if descriptor.name == "langchain":
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content=text)]
        return [messages], kwargs

    if descriptor.name == "llamaindex":
        from llama_index.core.llms import ChatMessage, MessageRole

        user_role = getattr(MessageRole, "USER", "user")
        messages = [ChatMessage(role=user_role, content=text)]
        return [messages], kwargs

    # Default: frameworks that accept raw string or general message-like
    return [text], kwargs


def _build_async_completion_args(
    descriptor: LLMFrameworkDescriptor,
    text: str,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Build positional args + kwargs for an async completion call appropriate
    for the given framework.
    """
    kwargs: dict[str, Any] = {}
    if descriptor.context_kwarg:
        kwargs[descriptor.context_kwarg] = {}

    if descriptor.name == "langchain":
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content=text)]
        return [messages], kwargs

    if descriptor.name == "llamaindex":
        from llama_index.core.llms import ChatMessage, MessageRole

        user_role = getattr(MessageRole, "USER", "user")
        messages = [ChatMessage(role=user_role, content=text)]
        return [messages], kwargs

    if descriptor.name == "semantic_kernel":
        from semantic_kernel.connectors.ai.chat_completion_client_base import (
            ChatHistory,
            PromptExecutionSettings,
        )

        history = ChatHistory()
        # Most SK versions expose add_user_message; fall back to generic add_message
        if hasattr(history, "add_user_message"):
            history.add_user_message(text)
        elif hasattr(history, "add_message"):
            history.add_message(text)
        else:
            # Last resort: hope ChatHistory is iterable over something reasonable
            pass

        settings = PromptExecutionSettings()
        return [history, settings], kwargs

    # Default: treat as raw text
    return [text], kwargs


def _build_sync_stream_args(
    descriptor: LLMFrameworkDescriptor,
    text: str,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Build args/kwargs for a sync streaming call appropriate for the framework.
    """
    return _build_sync_completion_args(descriptor, text)


def _build_async_stream_args(
    descriptor: LLMFrameworkDescriptor,
    text: str,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Build args/kwargs for an async streaming call appropriate for the framework.
    """
    return _build_async_completion_args(descriptor, text)


def _build_token_count_args(
    descriptor: LLMFrameworkDescriptor,
    text: str,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Build args/kwargs for token counting calls appropriate for the framework.
    """
    kwargs: dict[str, Any] = {}

    if descriptor.name == "langchain":
        # We assume token_count_method is get_num_tokens(text: str)
        return [text], kwargs

    if descriptor.name == "llamaindex":
        from llama_index.core.llms import ChatMessage, MessageRole

        user_role = getattr(MessageRole, "USER", "user")
        messages = [ChatMessage(role=user_role, content=text)]
        return [messages], kwargs

    if descriptor.name == "semantic_kernel":
        from semantic_kernel.connectors.ai.chat_completion_client_base import (
            ChatHistory,
            PromptExecutionSettings,
        )

        history = ChatHistory()
        if hasattr(history, "add_user_message"):
            history.add_user_message(text)
        elif hasattr(history, "add_message"):
            history.add_message(text)

        settings = PromptExecutionSettings()
        return [history, settings], kwargs

    # Default: assume simple text-based token counting
    return [text], kwargs


# ---------------------------------------------------------------------------
# Core interface / surface contract tests
# ---------------------------------------------------------------------------


def test_can_instantiate_llm_client(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    Each registered framework descriptor should be instantiable with the
    pluggable Corpus LLM adapter and any inferred kwargs.

    Sanity-check that the instance exposes the methods the descriptor claims.
    """
    # Sync completion (when declared)
    if framework_descriptor.completion_method:
        _get_method(llm_client_instance, framework_descriptor.completion_method)

    # Sync streaming (when declared)
    if framework_descriptor.streaming_method:
        _get_method(llm_client_instance, framework_descriptor.streaming_method)

    # Token counting (when declared)
    if framework_descriptor.token_count_method:
        _get_method(llm_client_instance, framework_descriptor.token_count_method)

    # Async surfaces (if any async declared)
    if framework_descriptor.supports_async:
        if framework_descriptor.async_completion_method:
            _get_method(
                llm_client_instance,
                framework_descriptor.async_completion_method,
            )

        if framework_descriptor.async_streaming_method:
            _get_method(
                llm_client_instance,
                framework_descriptor.async_streaming_method,
            )

        if framework_descriptor.async_token_count_method:
            _get_method(
                llm_client_instance,
                framework_descriptor.async_token_count_method,
            )


def test_async_methods_exist_when_supports_async_true(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    Ensure that when supports_async=True, async completion exists and is callable.

    Unlike graph/embedding, LLM frameworks may be async-only, so we only
    require an async completion surface here; async streaming is optional.
    """
    if not framework_descriptor.supports_async:
        pytest.skip("Framework does not declare async support")

    # Registry policy for LLMs: async_completion_method must be present.
    assert (
        framework_descriptor.async_completion_method is not None
    ), f"{framework_descriptor.name}: supports_async=True but async_completion_method is None"

    acomplete = getattr(
        llm_client_instance,
        framework_descriptor.async_completion_method,
        None,
    )
    assert callable(acomplete), "Async completion method is not callable"

    # If an async streaming method is declared, it must be callable as well.
    if framework_descriptor.async_streaming_method:
        astream = getattr(
            llm_client_instance,
            framework_descriptor.async_streaming_method,
            None,
        )
        assert callable(astream), "Async streaming method is not callable"


def test_sync_completion_interface_conformance_when_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    Validate that the sync completion method (when declared) accepts a simple
    text-style input (or framework-specific message type) and returns a
    non-None result.

    Detailed response shapes are covered by framework-specific tests.
    """
    if not framework_descriptor.completion_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare sync completion",
        )

    completion_fn = _get_method(
        llm_client_instance,
        framework_descriptor.completion_method,
    )

    args, kwargs = _build_sync_completion_args(
        framework_descriptor,
        SYNC_COMPLETION_TEXT,
    )

    result = completion_fn(*args, **kwargs)
    assert result is not None


def test_sync_streaming_interface_when_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    Validate that the sync streaming method (when declared) accepts input and
    returns an iterable of chunks.

    For frameworks that stream via kwarg on the completion method only
    (streaming_style == "kwarg"), this test is skipped; a separate test
    covers the kwarg-based streaming behavior.
    """
    if not framework_descriptor.streaming_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare sync streaming method",
        )

    if getattr(framework_descriptor, "streaming_style", "method") == "kwarg":
        pytest.skip(
            f"Framework '{framework_descriptor.name}' uses kwarg-based streaming, "
            "not a dedicated streaming method",
        )

    stream_fn = _get_method(
        llm_client_instance,
        framework_descriptor.streaming_method,
    )

    args, kwargs = _build_sync_stream_args(
        framework_descriptor,
        SYNC_STREAM_TEXT,
    )

    iterator = stream_fn(*args, **kwargs)

    seen_any = False
    for _ in iterator:  # noqa: B007
        seen_any = True
        break

    # The contract is about iterability; it's fine if no chunks are produced.
    assert iterator is not None
    assert isinstance(seen_any, bool)


def test_sync_streaming_via_kwarg_when_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    For frameworks that declare streaming via a kwarg on the completion method
    (streaming_style == "kwarg"), validate that setting the kwarg produces an
    iterable of chunks.
    """
    if getattr(framework_descriptor, "streaming_style", "method") != "kwarg":
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not use kwarg-based streaming",
        )

    assert (
        framework_descriptor.completion_method is not None
    ), f"{framework_descriptor.name}: streaming_style='kwarg' but completion_method is None"
    assert (
        framework_descriptor.streaming_kwarg is not None
    ), f"{framework_descriptor.name}: streaming_style='kwarg' but streaming_kwarg is None"

    completion_fn = _get_method(
        llm_client_instance,
        framework_descriptor.completion_method,
    )

    args, kwargs = _build_sync_completion_args(
        framework_descriptor,
        SYNC_STREAM_TEXT,
    )
    kwargs[framework_descriptor.streaming_kwarg] = True

    iterator = completion_fn(*args, **kwargs)

    seen_any = False
    for _ in iterator:  # noqa: B007
        seen_any = True
        break

    assert iterator is not None
    assert isinstance(seen_any, bool)


@pytest.mark.asyncio
async def test_async_completion_interface_conformance_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    Validate that the async completion method (when declared) accepts input
    and returns a result compatible with the sync API (non-None).
    """
    if not framework_descriptor.async_completion_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async completion",
        )

    acomplete_fn = _get_method(
        llm_client_instance,
        framework_descriptor.async_completion_method,
    )

    args, kwargs = _build_async_completion_args(
        framework_descriptor,
        ASYNC_COMPLETION_TEXT,
    )

    coro = acomplete_fn(*args, **kwargs)
    assert inspect.isawaitable(coro), "Async completion method must return an awaitable"

    result = await coro
    assert result is not None


@pytest.mark.asyncio
async def test_async_streaming_interface_conformance_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    Validate that the async streaming method (when declared) accepts input
    and produces an async-iterable of chunks.

    The returned object may be an async iterator directly or an awaitable
    resolving to one.
    """
    if not framework_descriptor.async_streaming_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async streaming",
        )

    astream_fn = _get_method(
        llm_client_instance,
        framework_descriptor.async_streaming_method,
    )

    args, kwargs = _build_async_stream_args(
        framework_descriptor,
        ASYNC_STREAM_TEXT,
    )

    aiter = astream_fn(*args, **kwargs)

    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    seen_any = False
    async for _ in aiter:  # noqa: B007
        seen_any = True
        break

    assert isinstance(seen_any, bool)


def test_context_kwarg_is_accepted_when_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    If a context_kwarg is declared in the descriptor, the corresponding
    completion method should accept that kwarg without raising TypeError.
    """
    if not framework_descriptor.context_kwarg:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare a context_kwarg",
        )

    if not framework_descriptor.completion_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare sync completion",
        )

    completion_fn = _get_method(
        llm_client_instance,
        framework_descriptor.completion_method,
    )

    args, kwargs = _build_sync_completion_args(
        framework_descriptor,
        CONTEXT_COMPLETION_TEXT,
    )
    kwargs[framework_descriptor.context_kwarg] = {"test": "value"}

    result = completion_fn(*args, **kwargs)
    assert result is not None


def test_method_signatures_consistent_between_sync_and_async(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    Verify that sync and async methods have consistent signatures
    (same parameters except maybe the return annotation), where both
    variants are declared.

    Covers completion and streaming surfaces.
    """

    def _compare_signatures(sync_name: str | None, async_name: str | None) -> None:
        if not sync_name or not async_name:
            return

        sync_fn = _get_method(llm_client_instance, sync_name)
        async_fn = _get_method(llm_client_instance, async_name)

        sync_sig = inspect.signature(sync_fn)
        async_sig = inspect.signature(async_fn)

        # Skip "self" for bound methods
        sync_params = list(sync_sig.parameters.keys())[1:]
        async_params = list(async_sig.parameters.keys())[1:]

        assert (
            sync_params == async_params
        ), f"Signature mismatch between {sync_name!r} and {async_name!r}"

    # Completion
    _compare_signatures(
        framework_descriptor.completion_method,
        framework_descriptor.async_completion_method,
    )

    # Streaming
    _compare_signatures(
        framework_descriptor.streaming_method,
        framework_descriptor.async_streaming_method,
    )


# ---------------------------------------------------------------------------
# Token counting contract
# ---------------------------------------------------------------------------


def test_token_count_interface_when_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    If a framework declares token_count_method, it should be callable and
    return an integer token count for a simple input.
    """
    if not framework_descriptor.token_count_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare token_count_method",
        )

    count_fn = _get_method(
        llm_client_instance,
        framework_descriptor.token_count_method,
    )

    args, kwargs = _build_token_count_args(
        framework_descriptor,
        TOKEN_COUNT_TEXT,
    )

    result = count_fn(*args, **kwargs)
    assert isinstance(result, int)
    assert result >= 0


@pytest.mark.asyncio
async def test_async_token_count_interface_when_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    If a framework declares async_token_count_method, it should be callable and
    return an integer token count for a simple input.
    """
    if not framework_descriptor.async_token_count_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async_token_count_method",
        )

    acount_fn = _get_method(
        llm_client_instance,
        framework_descriptor.async_token_count_method,
    )

    args, kwargs = _build_token_count_args(
        framework_descriptor,
        TOKEN_COUNT_TEXT,
    )

    coro = acount_fn(*args, **kwargs)
    assert inspect.isawaitable(coro)
    result = await coro
    assert isinstance(result, int)
    assert result >= 0


# ---------------------------------------------------------------------------
# Capabilities / health passthrough contract
# ---------------------------------------------------------------------------


def test_capabilities_contract_if_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    If a framework declares has_capabilities=True, it should expose a
    capabilities() method returning a mapping. Async variants (when present)
    should behave similarly.
    """
    if not framework_descriptor.has_capabilities:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not expose capabilities",
        )

    capabilities = getattr(llm_client_instance, "capabilities", None)
    assert callable(capabilities), "capabilities() method is missing"

    caps_result = capabilities()
    assert isinstance(
        caps_result,
        Mapping,
    ), "capabilities() should return a mapping"

    async_caps = getattr(llm_client_instance, "acapabilities", None)
    if async_caps is not None and callable(async_caps):
        acaps_result = _run_async_if_needed(async_caps())
        assert isinstance(
            acaps_result,
            Mapping,
        ), "acapabilities() should return a mapping"


def test_health_contract_if_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    If a framework declares has_health=True, it should expose a health()
    method returning a mapping. Async variants (when present) should behave
    similarly.
    """
    if not framework_descriptor.has_health:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not expose health",
        )

    health = getattr(llm_client_instance, "health", None)
    assert callable(health), "health() method is missing"

    health_result = health()
    assert isinstance(
        health_result,
        Mapping,
    ), "health() should return a mapping"

    async_health = getattr(llm_client_instance, "ahealth", None)
    if async_health is not None and callable(async_health):
        ahealth_result = _run_async_if_needed(async_health())
        assert isinstance(
            ahealth_result,
            Mapping,
        ), "ahealth() should return a mapping"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
