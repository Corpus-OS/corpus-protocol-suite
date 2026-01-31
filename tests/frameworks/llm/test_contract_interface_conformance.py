# tests/frameworks/llm/test_contract_interface_conformance.py

from __future__ import annotations

import asyncio
import importlib
import inspect
from dataclasses import dataclass
from collections.abc import AsyncIterable, Iterable, Mapping
from typing import Any, Callable, Optional

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

# Rich context payload (kept mapping-like for cross-framework tolerance)
RICH_CONTEXT: dict[str, Any] = {
    "request_id": "req-llm-123",
    "user_id": "user-llm-abc",
    "tags": ["test", "llm"],
    "nested": {"key": "value", "depth": 2},
}


# ---------------------------------------------------------------------------
# Build result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ClientBuildResult:
    """
    Container for client construction results.

    We use this to preserve a strict "no skips" policy while still cleanly
    expressing "unavailable" frameworks as a validated pass condition.
    """

    instance: Any
    error: Optional[BaseException] = None


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

    IMPORTANT POLICY ALIGNMENT (mirrors embedding/graph):
    - We do not skip unavailable frameworks.
    - Tests must pass by asserting correct "unavailable" signaling when a framework
      is not installed or cannot be constructed due to optional dependency issues.
    """
    descriptor: LLMFrameworkDescriptor = request.param
    return descriptor


@pytest.fixture
def llm_client_build(
    framework_descriptor: LLMFrameworkDescriptor,
    adapter: Any,
) -> _ClientBuildResult:
    """
    Construct a concrete LLM client/adapter instance for the given descriptor.

    Availability contract:
    - If the framework is unavailable (descriptor.is_available() is False),
      return instance=None and tests validate that contract.
    - If descriptor.is_available() is True but construction fails due to
      optional dependency / import mismatch issues, also return instance=None
      and tests validate the "unavailable due to dependency" contract.
    - If descriptor.is_available() is True and construction fails for other
      reasons, tests will fail (real adapter regression).
    """
    if not framework_descriptor.is_available():
        return _ClientBuildResult(instance=None, error=None)

    try:
        module = importlib.import_module(framework_descriptor.adapter_module)
    except Exception as e:
        # Import errors are treated as availability failures when the adapter module
        # cannot be loaded (common when optional framework deps are missing).
        return _ClientBuildResult(instance=None, error=e)

    try:
        client_cls = getattr(module, framework_descriptor.adapter_class)
    except Exception as e:
        # Missing class is a real mismatch between registry and adapter module.
        return _ClientBuildResult(instance=None, error=e)

    try:
        # All LLM framework adapters take an LLMProtocolV1 implementation under a
        # consistent kwarg name (llm_adapter).
        instance = client_cls(llm_adapter=adapter)
        return _ClientBuildResult(instance=instance, error=None)
    except BaseException as e:
        # Construction failures are interpreted by tests: either "dependency/unavailable"
        # or "real regression" depending on error type/message.
        return _ClientBuildResult(instance=None, error=e)


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


def _get_unbound_method(owner: type, name: str) -> Callable[..., Any]:
    """Fetch an unbound method from the class for signature comparison."""
    attr = getattr(owner, name, None)
    assert callable(attr), f"{owner!r} missing expected callable method {name!r}"
    return attr


def _dispose_awaitable(value: Any) -> None:
    """
    Dispose of an awaitable produced in a *sync* test so it doesn't emit
    "coroutine was never awaited" warnings.

    Notes:
    - Prefer coroutine.close() when available.
    - For Futures/Tasks, cancel() is the safest non-blocking disposal method.
    """
    if inspect.iscoroutine(value):
        value.close()
        return

    if isinstance(value, asyncio.Future):
        value.cancel()
        return

    close = getattr(value, "close", None)
    if callable(close):
        close()
        return

    cancel = getattr(value, "cancel", None)
    if callable(cancel):
        cancel()
        return


def _is_dependency_unavailable_error(err: Optional[BaseException]) -> bool:
    """
    Heuristic for identifying "framework is unavailable" failures.

    We treat these as validated pass conditions even if descriptor.is_available()
    returned True, because optional dependencies can be:
      - installed but incompatible (API drift),
      - partially installed,
      - importable but missing specific symbols.

    We DO NOT treat generic runtime failures as unavailable; those remain failures.
    """
    if err is None:
        return False

    # Straight import failures.
    if isinstance(err, (ImportError, ModuleNotFoundError)):
        return True

    # Common adapter "optional dependency" sentinel messages.
    msg = str(err).lower()

    # Example: semantic-kernel adapter uses a RuntimeError wrapper around import failures.
    if "not installed" in msg and "pip install" in msg:
        return True
    if "failed to import" in msg and "install" in msg:
        return True

    # Symbol drift often manifests as "cannot import name X".
    if "cannot import name" in msg:
        return True

    return False


def _assert_unavailable_contract(
    descriptor: LLMFrameworkDescriptor,
    build: _ClientBuildResult,
) -> None:
    """
    Validate the "unavailable framework" contract in a no-skip test suite.

    Acceptable "unavailable" signals:
      - descriptor.is_available() is False
      - OR construction/import failed with a dependency-unavailable error
    """
    if descriptor.is_available() is False:
        assert build.instance is None
        return

    # If registry says available but we couldn't construct, only allow if the error
    # is clearly a dependency/import availability issue.
    assert build.instance is None, "Expected no instance for unavailable contract validation"
    assert _is_dependency_unavailable_error(build.error), (
        f"{descriptor.name}: registry reports available, but client construction failed with "
        f"non-availability error {type(build.error).__name__}: {build.error}"
    )


def _context_kwargs_for_descriptor(descriptor: LLMFrameworkDescriptor) -> dict[str, Any]:
    """
    Build kwargs reflecting the framework's declared context parameter.

    Returns a dict with a single key (descriptor.context_kwarg) containing a rich
    mapping context, or an empty dict if no context_kwarg is declared.
    """
    kw: dict[str, Any] = {}
    if descriptor.context_kwarg:
        # Pass the entire context under the framework-specific kwarg.
        ctx = dict(RICH_CONTEXT)
        try:
            tags = list(ctx.get("tags", []))
            tags.append(descriptor.name)
            ctx["tags"] = tags
        except Exception:
            pass
        kw[descriptor.context_kwarg] = ctx
    return kw


def _build_messages_default(text: str) -> list[dict[str, str]]:
    """
    Default message shape for frameworks that accept OpenAI-style chat messages.
    """
    return [{"role": "user", "content": text}]


def _safe_import(module_name: str) -> Any:
    """
    Import a module defensively.

    If the import fails, raise ImportError so caller can treat it as an
    availability failure (instead of crashing tests).
    """
    return importlib.import_module(module_name)


def _build_sync_completion_call(
    descriptor: LLMFrameworkDescriptor,
    text: str,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Build positional args + kwargs for a sync completion call appropriate
    for the given framework.
    """
    kwargs = _context_kwargs_for_descriptor(descriptor)

    # LangChain: _generate(messages, **config)
    if descriptor.name == "langchain":
        try:
            messages_mod = _safe_import("langchain_core.messages")
            HumanMessage = getattr(messages_mod, "HumanMessage")
        except Exception as e:
            raise ImportError("langchain_core.messages import failed") from e
        messages = [HumanMessage(content=text)]
        return [messages], kwargs

    # LlamaIndex: chat(messages, **kwargs) style (varies by version)
    if descriptor.name == "llamaindex":
        try:
            llms_mod = _safe_import("llama_index.core.llms")
            ChatMessage = getattr(llms_mod, "ChatMessage")
            MessageRole = getattr(llms_mod, "MessageRole")
        except Exception as e:
            raise ImportError("llama_index.core.llms import failed") from e
        user_role = getattr(MessageRole, "USER", "user")
        messages = [ChatMessage(role=user_role, content=text)]
        return [messages], kwargs

    # AutoGen-style: create(messages, stream=..., conversation=...)
    if descriptor.name == "autogen":
        messages = _build_messages_default(text)
        return [messages], kwargs

    # CrewAI commonly accepts prompt-like strings (adapter-specific); keep minimal.
    if descriptor.name == "crewai":
        return [text], kwargs

    # Semantic Kernel is async-first in the registry (sync completion is None).
    # If a sync completion method is ever added, default to text-based call.
    return [text], kwargs


def _build_async_completion_call(
    descriptor: LLMFrameworkDescriptor,
    text: str,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Build positional args + kwargs for an async completion call appropriate
    for the given framework.
    """
    kwargs = _context_kwargs_for_descriptor(descriptor)

    if descriptor.name == "langchain":
        try:
            messages_mod = _safe_import("langchain_core.messages")
            HumanMessage = getattr(messages_mod, "HumanMessage")
        except Exception as e:
            raise ImportError("langchain_core.messages import failed") from e
        messages = [HumanMessage(content=text)]
        return [messages], kwargs

    if descriptor.name == "llamaindex":
        try:
            llms_mod = _safe_import("llama_index.core.llms")
            ChatMessage = getattr(llms_mod, "ChatMessage")
            MessageRole = getattr(llms_mod, "MessageRole")
        except Exception as e:
            raise ImportError("llama_index.core.llms import failed") from e
        user_role = getattr(MessageRole, "USER", "user")
        messages = [ChatMessage(role=user_role, content=text)]
        return [messages], kwargs

    if descriptor.name == "semantic_kernel":
        # Registry describes async-only completion via get_chat_message_content(chat_history, settings)
        try:
            base_mod = _safe_import("semantic_kernel.connectors.ai.chat_completion_client_base")
            ChatHistory = getattr(base_mod, "ChatHistory")
            PromptExecutionSettings = getattr(base_mod, "PromptExecutionSettings")
        except Exception as e:
            raise ImportError("semantic_kernel ChatHistory/settings import failed") from e

        history = ChatHistory()
        if hasattr(history, "add_user_message"):
            history.add_user_message(text)
        elif hasattr(history, "add_message"):
            history.add_message(text)

        settings = PromptExecutionSettings()
        return [history, settings], kwargs

    if descriptor.name == "autogen":
        messages = _build_messages_default(text)
        return [messages], kwargs

    if descriptor.name == "crewai":
        return [text], kwargs

    return [text], kwargs


def _build_sync_stream_call(
    descriptor: LLMFrameworkDescriptor,
    text: str,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Build args/kwargs for a sync streaming call appropriate for the framework.

    NOTE:
      - For streaming_style="method", this is passed to streaming_method.
      - For streaming_style="kwarg", this is passed to completion_method with streaming_kwarg=True.
    """
    # Most frameworks share the same input surface for completion and streaming.
    return _build_sync_completion_call(descriptor, text)


def _build_async_stream_call(
    descriptor: LLMFrameworkDescriptor,
    text: str,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Build args/kwargs for an async streaming call appropriate for the framework.
    """
    return _build_async_completion_call(descriptor, text)


def _build_token_count_call(
    descriptor: LLMFrameworkDescriptor,
    text: str,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Build args/kwargs for token counting calls appropriate for the framework.
    """
    kwargs = _context_kwargs_for_descriptor(descriptor)

    # LangChain registry uses get_num_tokens_from_messages(messages)
    if descriptor.name == "langchain":
        try:
            messages_mod = _safe_import("langchain_core.messages")
            HumanMessage = getattr(messages_mod, "HumanMessage")
        except Exception as e:
            raise ImportError("langchain_core.messages import failed") from e
        messages = [HumanMessage(content=text)]
        return [messages], kwargs

    # LlamaIndex token counting often accepts messages
    if descriptor.name == "llamaindex":
        try:
            llms_mod = _safe_import("llama_index.core.llms")
            ChatMessage = getattr(llms_mod, "ChatMessage")
            MessageRole = getattr(llms_mod, "MessageRole")
        except Exception as e:
            raise ImportError("llama_index.core.llms import failed") from e
        user_role = getattr(MessageRole, "USER", "user")
        messages = [ChatMessage(role=user_role, content=text)]
        return [messages], kwargs

    # Semantic Kernel: count_tokens(history, settings) style in some versions
    if descriptor.name == "semantic_kernel":
        try:
            base_mod = _safe_import("semantic_kernel.connectors.ai.chat_completion_client_base")
            ChatHistory = getattr(base_mod, "ChatHistory")
            PromptExecutionSettings = getattr(base_mod, "PromptExecutionSettings")
        except Exception as e:
            raise ImportError("semantic_kernel ChatHistory/settings import failed") from e

        history = ChatHistory()
        if hasattr(history, "add_user_message"):
            history.add_user_message(text)
        elif hasattr(history, "add_message"):
            history.add_message(text)

        settings = PromptExecutionSettings()
        return [history, settings], kwargs

    # Default: assume simple text-based token counting
    return [text], kwargs


def _assert_stream_like_sync(value: Any) -> None:
    """
    Validate that a sync streaming result looks iterable.

    We consume at most one element to trigger lazy errors without draining streams.
    """
    assert value is not None

    try:
        it = iter(value)
    except TypeError as e:
        raise AssertionError(f"Expected an iterable stream, got {type(value).__name__}") from e

    try:
        next(it)
    except StopIteration:
        # Empty streams are valid; many adapters may emit nothing for trivial prompts.
        pass


async def _assert_stream_like_async(value: Any) -> None:
    """
    Validate that an async streaming result looks async-iterable.

    The returned object may be:
      - an async iterator directly, OR
      - an awaitable resolving to an async iterator.
    """
    if inspect.isawaitable(value):
        value = await value  # noqa: PLW2901

    assert value is not None
    assert isinstance(value, AsyncIterable), f"Async stream must be async-iterable, got {type(value).__name__}"

    async for _ in value:  # noqa: B007
        # Consume at most one item.
        break


# ---------------------------------------------------------------------------
# Core interface / surface contract tests (NO SKIPS)
# ---------------------------------------------------------------------------


def test_can_instantiate_llm_client(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_build: _ClientBuildResult,
) -> None:
    """
    Instantiation conformance (NO SKIPS).

    - If unavailable, validate unavailable contract and return.
    - If available, construction must succeed and declared methods must exist.
    """
    if llm_client_build.instance is None:
        _assert_unavailable_contract(framework_descriptor, llm_client_build)
        return

    instance = llm_client_build.instance

    # Completion/streaming/token-count methods should exist exactly when declared.
    if framework_descriptor.completion_method:
        _get_method(instance, framework_descriptor.completion_method)
    if framework_descriptor.streaming_method:
        _get_method(instance, framework_descriptor.streaming_method)
    if framework_descriptor.token_count_method:
        _get_method(instance, framework_descriptor.token_count_method)

    if framework_descriptor.async_completion_method:
        _get_method(instance, framework_descriptor.async_completion_method)
    if framework_descriptor.async_streaming_method:
        _get_method(instance, framework_descriptor.async_streaming_method)
    if framework_descriptor.async_token_count_method:
        _get_method(instance, framework_descriptor.async_token_count_method)


def test_registry_flags_are_coherent_with_declared_methods_when_available(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_build: _ClientBuildResult,
) -> None:
    """
    Registry coherence conformance (NO SKIPS).

    Mirrors graph: flags must be coherent with the descriptor fields.
    """
    if llm_client_build.instance is None:
        _assert_unavailable_contract(framework_descriptor, llm_client_build)
        return

    # supports_streaming must reflect actual streaming affordances declared.
    has_streaming = bool(
        framework_descriptor.streaming_method
        or framework_descriptor.async_streaming_method
        or framework_descriptor.streaming_kwarg
    )
    assert framework_descriptor.supports_streaming == has_streaming, (
        f"{framework_descriptor.name}: supports_streaming flag mismatch "
        f"(supports_streaming={framework_descriptor.supports_streaming}, "
        f"declared={has_streaming})"
    )

    # supports_token_count must reflect token-count methods declared.
    has_token_count = bool(
        framework_descriptor.token_count_method or framework_descriptor.async_token_count_method
    )
    assert framework_descriptor.supports_token_count == has_token_count, (
        f"{framework_descriptor.name}: supports_token_count flag mismatch "
        f"(supports_token_count={framework_descriptor.supports_token_count}, "
        f"declared={has_token_count})"
    )

    # supports_async must reflect async methods declared (property exists on the descriptor).
    has_async = bool(framework_descriptor.async_completion_method or framework_descriptor.async_streaming_method)
    assert framework_descriptor.supports_async == has_async, (
        f"{framework_descriptor.name}: supports_async property mismatch "
        f"(supports_async={framework_descriptor.supports_async}, declared={has_async})"
    )

    # streaming_style must be coherent with declared streaming surface.
    if framework_descriptor.streaming_style == "method":
        assert framework_descriptor.streaming_method or framework_descriptor.async_streaming_method, (
            f"{framework_descriptor.name}: streaming_style='method' but no streaming_method/async_streaming_method"
        )
    elif framework_descriptor.streaming_style == "kwarg":
        assert framework_descriptor.streaming_kwarg, (
            f"{framework_descriptor.name}: streaming_style='kwarg' but streaming_kwarg is None"
        )
        assert framework_descriptor.completion_method or framework_descriptor.async_completion_method, (
            f"{framework_descriptor.name}: streaming_style='kwarg' but no completion method declared"
        )
    elif framework_descriptor.streaming_style == "none":
        assert not has_streaming, f"{framework_descriptor.name}: streaming_style='none' but streaming surface declared"
    else:
        raise AssertionError(f"{framework_descriptor.name}: unexpected streaming_style={framework_descriptor.streaming_style!r}")


def test_async_methods_exist_when_supports_async_true(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_build: _ClientBuildResult,
) -> None:
    """
    Async existence contract (NO SKIPS).

    LLM nuance:
    - Some frameworks are async-only (sync completion may be None).
    Policy:
    - If supports_async is True, async_completion_method MUST be declared and callable.
    """
    if llm_client_build.instance is None:
        _assert_unavailable_contract(framework_descriptor, llm_client_build)
        return

    instance = llm_client_build.instance

    if not framework_descriptor.supports_async:
        return

    assert framework_descriptor.async_completion_method is not None, (
        f"{framework_descriptor.name}: supports_async=True but async_completion_method is None"
    )
    _get_method(instance, framework_descriptor.async_completion_method)

    if framework_descriptor.async_streaming_method:
        _get_method(instance, framework_descriptor.async_streaming_method)


def test_sync_completion_interface_conformance_when_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_build: _ClientBuildResult,
) -> None:
    """
    Sync completion interface conformance (NO SKIPS).

    - If sync completion is not declared, test passes (async-only framework).
    - If declared, method must accept best-effort minimal inputs and return non-None.
    """
    if llm_client_build.instance is None:
        _assert_unavailable_contract(framework_descriptor, llm_client_build)
        return

    if not framework_descriptor.completion_method:
        return

    instance = llm_client_build.instance
    completion_fn = _get_method(instance, framework_descriptor.completion_method)

    args, kwargs = _build_sync_completion_call(framework_descriptor, SYNC_COMPLETION_TEXT)
    result = completion_fn(*args, **kwargs)
    assert result is not None


def test_sync_streaming_interface_when_method_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_build: _ClientBuildResult,
) -> None:
    """
    Sync streaming interface conformance for streaming_style='method' (NO SKIPS).

    - If supports_streaming=False or streaming_style != 'method', test passes.
    - If streaming_style='method', streaming_method must exist and return iterable.
    """
    if llm_client_build.instance is None:
        _assert_unavailable_contract(framework_descriptor, llm_client_build)
        return

    if not framework_descriptor.supports_streaming:
        return

    if framework_descriptor.streaming_style != "method":
        return

    # For method-based streaming, at least one sync streaming method should exist.
    if not framework_descriptor.streaming_method:
        # Some adapters may be async-only streaming; sync part is optional.
        return

    instance = llm_client_build.instance
    stream_fn = _get_method(instance, framework_descriptor.streaming_method)

    args, kwargs = _build_sync_stream_call(framework_descriptor, SYNC_STREAM_TEXT)
    iterator = stream_fn(*args, **kwargs)
    _assert_stream_like_sync(iterator)


def test_sync_streaming_via_kwarg_when_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_build: _ClientBuildResult,
) -> None:
    """
    Sync kwarg-based streaming conformance (NO SKIPS).

    For streaming_style='kwarg':
      - completion_method must exist
      - streaming_kwarg must exist
      - calling completion_method(..., streaming_kwarg=True) must return an iterable
    """
    if llm_client_build.instance is None:
        _assert_unavailable_contract(framework_descriptor, llm_client_build)
        return

    if not framework_descriptor.supports_streaming:
        return

    if framework_descriptor.streaming_style != "kwarg":
        return

    assert framework_descriptor.streaming_kwarg is not None
    assert framework_descriptor.completion_method is not None

    instance = llm_client_build.instance
    completion_fn = _get_method(instance, framework_descriptor.completion_method)

    args, kwargs = _build_sync_completion_call(framework_descriptor, SYNC_STREAM_TEXT)
    kwargs[framework_descriptor.streaming_kwarg] = True
    iterator = completion_fn(*args, **kwargs)
    _assert_stream_like_sync(iterator)


@pytest.mark.asyncio
async def test_async_completion_interface_conformance_when_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_build: _ClientBuildResult,
) -> None:
    """
    Async completion conformance (NO SKIPS).

    - If async completion is not declared, test passes.
    - If declared, method must return an awaitable and result must be non-None.
    """
    if llm_client_build.instance is None:
        _assert_unavailable_contract(framework_descriptor, llm_client_build)
        return

    if not framework_descriptor.async_completion_method:
        return

    instance = llm_client_build.instance
    acomplete_fn = _get_method(instance, framework_descriptor.async_completion_method)

    args, kwargs = _build_async_completion_call(framework_descriptor, ASYNC_COMPLETION_TEXT)
    out = acomplete_fn(*args, **kwargs)
    assert inspect.isawaitable(out), "Async completion method must return an awaitable"
    result = await out  # noqa: PT018
    assert result is not None


@pytest.mark.asyncio
async def test_async_streaming_interface_when_method_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_build: _ClientBuildResult,
) -> None:
    """
    Async streaming interface conformance for streaming_style='method' (NO SKIPS).

    - If async streaming method is declared, it must produce an async-iterable.
    - If not declared, test passes.
    """
    if llm_client_build.instance is None:
        _assert_unavailable_contract(framework_descriptor, llm_client_build)
        return

    if not framework_descriptor.async_streaming_method:
        return

    instance = llm_client_build.instance
    astream_fn = _get_method(instance, framework_descriptor.async_streaming_method)

    args, kwargs = _build_async_stream_call(framework_descriptor, ASYNC_STREAM_TEXT)
    out = astream_fn(*args, **kwargs)
    await _assert_stream_like_async(out)


@pytest.mark.asyncio
async def test_async_streaming_via_kwarg_when_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_build: _ClientBuildResult,
) -> None:
    """
    Async kwarg-based streaming conformance (NO SKIPS).

    This is an LLM-specific nuance:
    - Some frameworks expose streaming via stream=True on async completion methods.

    For streaming_style='kwarg':
      - async_completion_method must exist
      - streaming_kwarg must exist
      - calling async_completion_method(..., streaming_kwarg=True) must return:
            * an async iterator, OR
            * an awaitable resolving to an async iterator
    """
    if llm_client_build.instance is None:
        _assert_unavailable_contract(framework_descriptor, llm_client_build)
        return

    if not framework_descriptor.supports_streaming:
        return

    if framework_descriptor.streaming_style != "kwarg":
        return

    if not framework_descriptor.async_completion_method:
        # Some kwarg-stream frameworks may only support sync streaming; allow pass.
        return

    assert framework_descriptor.streaming_kwarg is not None

    instance = llm_client_build.instance
    acomplete_fn = _get_method(instance, framework_descriptor.async_completion_method)

    args, kwargs = _build_async_completion_call(framework_descriptor, ASYNC_STREAM_TEXT)
    kwargs[framework_descriptor.streaming_kwarg] = True

    out = acomplete_fn(*args, **kwargs)

    # Accept either:
    # - awaitable -> async iterator, OR
    # - async iterator directly (some frameworks return it without an await).
    if inspect.isawaitable(out):
        out = await out  # noqa: PLW2901

    assert isinstance(out, AsyncIterable), f"Async kwarg stream must yield AsyncIterable, got {type(out).__name__}"
    async for _ in out:  # noqa: B007
        break


def test_context_kwarg_is_accepted_when_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_build: _ClientBuildResult,
) -> None:
    """
    Context kwarg acceptance contract (NO SKIPS).

    If context_kwarg is declared:
      - Prefer testing sync completion if present.
      - Otherwise, validate async completion returns an awaitable (and dispose it)
        to avoid loop hazards in a sync test.
    """
    if llm_client_build.instance is None:
        _assert_unavailable_contract(framework_descriptor, llm_client_build)
        return

    if not framework_descriptor.context_kwarg:
        return

    instance = llm_client_build.instance
    ctx_kw = framework_descriptor.context_kwarg

    # Prefer sync completion where available.
    if framework_descriptor.completion_method:
        completion_fn = _get_method(instance, framework_descriptor.completion_method)
        args, kwargs = _build_sync_completion_call(framework_descriptor, CONTEXT_COMPLETION_TEXT)
        kwargs[ctx_kw] = {"test": "value", **dict(RICH_CONTEXT)}
        result = completion_fn(*args, **kwargs)
        assert result is not None
        return

    # Async-only frameworks: validate async completion accepts the context kwarg.
    if framework_descriptor.async_completion_method:
        acomplete_fn = _get_method(instance, framework_descriptor.async_completion_method)
        args, kwargs = _build_async_completion_call(framework_descriptor, CONTEXT_COMPLETION_TEXT)
        kwargs[ctx_kw] = {"test": "value", **dict(RICH_CONTEXT)}
        out = acomplete_fn(*args, **kwargs)
        assert inspect.isawaitable(out), "Async completion must return an awaitable"
        _dispose_awaitable(out)
        return

    # If neither sync nor async completion exists, registry is inconsistent.
    raise AssertionError(
        f"{framework_descriptor.name}: context_kwarg declared but no completion surface exists"
    )


def test_method_signatures_consistent_between_sync_and_async_when_comparable(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_build: _ClientBuildResult,
) -> None:
    """
    Signature consistency (best-effort, NO SKIPS).

    Mirrors graph approach but adds LLM-specific tolerance:
    - Many adapters wrap methods with decorators that introduce *args/**kwargs.
    - We only enforce exact signature equality when BOTH methods have
      concrete, comparable parameter lists (i.e., neither uses *args/**kwargs).

    This prevents false negatives while still catching accidental drift in
    "stable" surfaces.
    """
    if llm_client_build.instance is None:
        _assert_unavailable_contract(framework_descriptor, llm_client_build)
        return

    instance = llm_client_build.instance
    owner = type(instance)

    def _compare(sync_name: str | None, async_name: str | None) -> None:
        if not sync_name or not async_name:
            return

        sync_unbound = _get_unbound_method(owner, sync_name)
        async_unbound = _get_unbound_method(owner, async_name)

        sync_sig = inspect.signature(sync_unbound)
        async_sig = inspect.signature(async_unbound)

        sync_params = list(sync_sig.parameters.values())
        async_params = list(async_sig.parameters.values())

        # Drop self if present.
        if sync_params and sync_params[0].name == "self":
            sync_params = sync_params[1:]
        if async_params and async_params[0].name == "self":
            async_params = async_params[1:]

        # If either uses varargs/kwargs, signatures are not reliably comparable.
        if any(p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) for p in sync_params):
            return
        if any(p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) for p in async_params):
            return

        sync_view = [(p.name, p.kind) for p in sync_params]
        async_view = [(p.name, p.kind) for p in async_params]

        assert sync_view == async_view, (
            f"{framework_descriptor.name}: signature mismatch between {sync_name!r} and {async_name!r}: "
            f"{sync_view} != {async_view}"
        )

    _compare(framework_descriptor.completion_method, framework_descriptor.async_completion_method)
    _compare(framework_descriptor.streaming_method, framework_descriptor.async_streaming_method)
    _compare(framework_descriptor.token_count_method, framework_descriptor.async_token_count_method)


# ---------------------------------------------------------------------------
# Token counting contract (NO SKIPS)
# ---------------------------------------------------------------------------


def test_token_count_contract_matches_registry_flag(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_build: _ClientBuildResult,
) -> None:
    """
    NO SKIPS token-count contract (mirrors graph cap/health style):

      - If supports_token_count=True:
          * token_count_method must be declared and callable
          * calling it returns int >= 0
      - If supports_token_count=False:
          * token_count_method and async_token_count_method should be None
    """
    if llm_client_build.instance is None:
        _assert_unavailable_contract(framework_descriptor, llm_client_build)
        return

    instance = llm_client_build.instance

    if not framework_descriptor.supports_token_count:
        assert framework_descriptor.token_count_method is None
        assert framework_descriptor.async_token_count_method is None
        return

    assert framework_descriptor.token_count_method is not None, (
        f"{framework_descriptor.name}: supports_token_count=True but token_count_method is None"
    )

    count_fn = _get_method(instance, framework_descriptor.token_count_method)
    args, kwargs = _build_token_count_call(framework_descriptor, TOKEN_COUNT_TEXT)
    result = count_fn(*args, **kwargs)
    assert isinstance(result, int)
    assert result >= 0


@pytest.mark.asyncio
async def test_async_token_count_returns_int_when_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_build: _ClientBuildResult,
) -> None:
    """
    Async token-count conformance (NO SKIPS):

      - If async_token_count_method is declared:
          * it must return an awaitable
          * awaited result must be int >= 0
      - If not declared, test passes (async variant is optional).
    """
    if llm_client_build.instance is None:
        _assert_unavailable_contract(framework_descriptor, llm_client_build)
        return

    if not framework_descriptor.async_token_count_method:
        return

    instance = llm_client_build.instance
    acount_fn = _get_method(instance, framework_descriptor.async_token_count_method)

    args, kwargs = _build_token_count_call(framework_descriptor, TOKEN_COUNT_TEXT)
    out = acount_fn(*args, **kwargs)
    assert inspect.isawaitable(out)
    result = await out  # noqa: PT018
    assert isinstance(result, int)
    assert result >= 0


# ---------------------------------------------------------------------------
# Capabilities / health passthrough contract (NO SKIPS; mirrors graph)
# ---------------------------------------------------------------------------


def test_capabilities_contract_matches_registry_flag(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_build: _ClientBuildResult,
) -> None:
    """
    NO SKIPS:
      - If has_capabilities=True -> capabilities() must exist and return Mapping.
      - If has_capabilities=False -> capabilities() must NOT exist (or must not be callable).
    """
    if llm_client_build.instance is None:
        _assert_unavailable_contract(framework_descriptor, llm_client_build)
        return

    instance = llm_client_build.instance
    capabilities = getattr(instance, "capabilities", None)

    if framework_descriptor.has_capabilities:
        assert callable(capabilities), (
            f"{framework_descriptor.name}: has_capabilities=True but capabilities() is missing"
        )
        caps_result = capabilities()
        assert isinstance(caps_result, Mapping), "capabilities() should return a Mapping"

        async_caps = getattr(instance, "acapabilities", None)
        if async_caps is not None:
            assert callable(async_caps), "acapabilities exists but is not callable"
    else:
        assert not callable(capabilities), (
            f"{framework_descriptor.name}: has_capabilities=False but capabilities() exists/callable; "
            "either remove the method or flip has_capabilities=True in the registry"
        )
        async_caps = getattr(instance, "acapabilities", None)
        assert not callable(async_caps), (
            f"{framework_descriptor.name}: has_capabilities=False but acapabilities() exists/callable; "
            "either remove it or flip has_capabilities=True"
        )


@pytest.mark.asyncio
async def test_async_capabilities_returns_mapping_if_present(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_build: _ClientBuildResult,
) -> None:
    """
    NO SKIPS:
      - If acapabilities() exists, it must return Mapping.
      - If it does not exist, test passes (async variant is optional).
    """
    if llm_client_build.instance is None:
        _assert_unavailable_contract(framework_descriptor, llm_client_build)
        return

    instance = llm_client_build.instance
    async_caps = getattr(instance, "acapabilities", None)
    if not callable(async_caps):
        return

    out = async_caps()
    assert inspect.isawaitable(out)
    result = await out  # noqa: PT018
    assert isinstance(result, Mapping), "acapabilities() should return a Mapping"


def test_health_contract_matches_registry_flag(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_build: _ClientBuildResult,
) -> None:
    """
    NO SKIPS:
      - If has_health=True -> health() must exist and return Mapping.
      - If has_health=False -> health() must NOT exist (or must not be callable).
    """
    if llm_client_build.instance is None:
        _assert_unavailable_contract(framework_descriptor, llm_client_build)
        return

    instance = llm_client_build.instance
    health = getattr(instance, "health", None)

    if framework_descriptor.has_health:
        assert callable(health), f"{framework_descriptor.name}: has_health=True but health() is missing"
        health_result = health()
        assert isinstance(health_result, Mapping), "health() should return a Mapping"

        async_health = getattr(instance, "ahealth", None)
        if async_health is not None:
            assert callable(async_health), "ahealth exists but is not callable"
    else:
        assert not callable(health), (
            f"{framework_descriptor.name}: has_health=False but health() exists/callable; "
            "either remove the method or flip has_health=True in the registry"
        )
        async_health = getattr(instance, "ahealth", None)
        assert not callable(async_health), (
            f"{framework_descriptor.name}: has_health=False but ahealth() exists/callable; "
            "either remove it or flip has_health=True"
        )


@pytest.mark.asyncio
async def test_async_health_returns_mapping_if_present(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_build: _ClientBuildResult,
) -> None:
    """
    NO SKIPS:
      - If ahealth() exists, it must return Mapping.
      - If it does not exist, test passes (async variant is optional).
    """
    if llm_client_build.instance is None:
        _assert_unavailable_contract(framework_descriptor, llm_client_build)
        return

    instance = llm_client_build.instance
    async_health = getattr(instance, "ahealth", None)
    if not callable(async_health):
        return

    out = async_health()
    assert inspect.isawaitable(out)
    result = await out  # noqa: PT018
    assert isinstance(result, Mapping), "ahealth() should return a Mapping"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
