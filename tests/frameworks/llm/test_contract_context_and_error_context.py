# tests/frameworks/llm/test_contract_context_and_error_context.py

from __future__ import annotations

import asyncio
import importlib
import inspect
from collections.abc import Mapping as ABCMapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

import pytest

from tests.frameworks.registries.llm_registry import (
    LLMFrameworkDescriptor,
    iter_llm_framework_descriptors,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAILURE_MESSAGE = "intentional failure from failing llm adapter"
LLM_OPERATION_PREFIX = "llm_"

RICH_CONTEXT = {
    "request_id": "req-llm-123",
    "user_id": "user-llm-abc",
    "tags": ["test", "llm"],
    "nested": {"key": "value", "depth": 2},
}

PROMPT_TEXT = "ctx-prompt"
STREAM_PROMPT_TEXT = "ctx-stream-prompt"
ASYNC_PROMPT_TEXT = "ctx-async-prompt"
ASYNC_STREAM_PROMPT_TEXT = "ctx-async-stream-prompt"

# A portable fallback message list for adapters that prefer message-shaped inputs.
# NOTE:
# This test file intentionally focuses on context tolerance + error-context decoration,
# not deep LLM semantics (those live in dedicated contract tests elsewhere).
FALLBACK_MESSAGES = [{"role": "user", "content": PROMPT_TEXT}]
FALLBACK_STREAM_MESSAGES = [{"role": "user", "content": STREAM_PROMPT_TEXT}]

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

    IMPORTANT POLICY ALIGNMENT (mirrors graph + embedding):
    - We do not skip unavailable frameworks.
    - Tests must pass by asserting correct "unavailable" signaling when a framework
      is not installed, and must fully run when it is available.
    """
    descriptor: LLMFrameworkDescriptor = request.param
    return descriptor


@pytest.fixture
def llm_client_instance(
    framework_descriptor: LLMFrameworkDescriptor,
    adapter: Any,
) -> Any:
    """
    Construct a concrete LLM client instance for the given descriptor.

    Mirrors the construction pattern used in the other contract tests:
    each framework adapter is expected to take a kwarg that wraps a Corpus
    LLMProtocolV1-like implementation.

    Availability contract:
    - If a framework is unavailable, this fixture returns None and tests must
      treat that as a validated pass condition (not a skip).

    IMPORTANT:
    - LLM adapters historically accepted an injection kwarg named "llm_adapter".
      To avoid brittle coupling (and to preserve the "no skip" policy), this fixture
      also attempts a best-effort fallback to other common injection kwarg names
      if "llm_adapter" is rejected.
    """
    if not framework_descriptor.is_available():
        return None

    module = importlib.import_module(framework_descriptor.adapter_module)
    client_cls = getattr(module, framework_descriptor.adapter_class)

    # Preferred injection kwarg for LLM framework adapters.
    init_kwargs: dict[str, Any] = {"llm_adapter": adapter}

    try:
        return client_cls(**init_kwargs)
    except TypeError as e:
        # Best-effort compatibility: avoid hard failures if the adapter uses a different name.
        msg = str(e)
        unexpected_llm_adapter = "llm_adapter" in msg and "unexpected keyword" in msg
        if not unexpected_llm_adapter:
            raise

    # Fallback injection names used in other domains and older adapters.
    for alt_kw in ("corpus_adapter", "adapter"):
        try:
            return client_cls(**{alt_kw: adapter})
        except TypeError:
            continue

    # If we reach here, we were unable to construct the adapter instance.
    # This is a real failure when the framework is considered "available".
    raise TypeError(
        f"{framework_descriptor.name}: could not construct adapter {framework_descriptor.adapter_class!r} "
        f"from module {framework_descriptor.adapter_module!r} using known injection kwargs"
    )


@pytest.fixture
def failing_llm_adapter_calltime() -> Any:
    """
    A minimal LLM adapter whose core methods always fail at call-time.

    Used only for error-context tests to ensure wrappers invoke attach_context()
    and propagate the exception.

    Notes (mirrors graph call-time fixture intent):
    - Failures that occur during request construction, input parsing, auth checks,
      or immediate upstream invocation are modeled as call-time failures.
    - Streaming call-time failures are distinct from in-stream failures; both must be covered.
    """

    class FailingLLMAdapterCalltime:
        def complete(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        def stream(self, *args: Any, **kwargs: Any) -> Any:
            # Call-time failure for streaming surface.
            raise RuntimeError(FAILURE_MESSAGE)

        def count_tokens(self, *args: Any, **kwargs: Any) -> int:
            raise RuntimeError(FAILURE_MESSAGE)

        def health(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError(FAILURE_MESSAGE)

        def capabilities(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError(FAILURE_MESSAGE)

    return FailingLLMAdapterCalltime()


@pytest.fixture
def failing_llm_adapter_instream() -> Any:
    """
    A minimal LLM adapter whose streaming methods fail during iteration.

    Why this fixture exists (mirrors graph in-stream fixture intent):
    - Streaming failures occur in two real-world places:
        1) call-time (request construction / auth / parameter validation)
        2) in-stream (network drop / upstream crash / mid-stream tool failure)
    - Many Python streaming surfaces are lazy; errors manifest during iteration.

    This fixture ensures the decorator sees errors raised "in-stream", not only
    at call time.
    """

    class FailingLLMAdapterInStream:
        def complete(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        def stream(self, *args: Any, **kwargs: Any):
            # Raise during iteration (first iteration attempt).
            raise RuntimeError(FAILURE_MESSAGE)
            yield  # pragma: no cover  # keeps generator shape for type checkers

        def count_tokens(self, *args: Any, **kwargs: Any) -> int:
            raise RuntimeError(FAILURE_MESSAGE)

        def health(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError(FAILURE_MESSAGE)

        def capabilities(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError(FAILURE_MESSAGE)

    return FailingLLMAdapterInStream()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_unavailable_contract(descriptor: LLMFrameworkDescriptor) -> None:
    """
    Validate that an unavailable framework descriptor is behaving as expected.

    The test suite policy is "no skip": when unavailable, tests must pass by
    asserting correct unavailability signaling.
    """
    assert descriptor.is_available() is False

    # If availability_attr is set, adapter module should generally import and expose the flag.
    # If the module cannot import, that is also a valid "unavailable" signal.
    if descriptor.availability_attr:
        try:
            module = importlib.import_module(descriptor.adapter_module)
        except Exception:
            return
        flag = getattr(module, descriptor.availability_attr, None)
        # Either missing (treated as unavailable) or False.
        assert flag is None or bool(flag) is False


def _get_method(instance: Any, name: str | None) -> Callable[..., Any]:
    """
    Helper to fetch a method from the instance and assert it is callable.

    If name is None, this fails fast with a clear assertion message.
    """
    assert name, "Expected a non-empty method name"
    attr = getattr(instance, name, None)
    assert callable(attr), f"{instance!r} missing expected callable method {name!r}"
    return attr


def _reset_async_bridge_state_best_effort() -> None:
    """
    Best-effort reset of the AsyncBridge circuit breaker and related sticky state.

    Rationale:
    - Some framework adapters bridge async protocol calls from sync code via AsyncBridge.
    - AsyncBridge may include a circuit breaker that can trip after repeated failures.
    - In these tests, repeated failures are *expected* and should not poison subsequent
      test cases by forcing a circuit-open error instead of the original exception.

    Implementation strategy:
    - Perform optional import (test environment may omit corpus_sdk modules).
    - Reset any exposed breaker state using a tolerant attribute/method search.
    - Never raise from this helper (tests should remain authoritative and deterministic).
    """
    try:
        mod = importlib.import_module("corpus_sdk.core.async_bridge")
    except Exception:
        return

    bridge = getattr(mod, "AsyncBridge", None)
    if bridge is None:
        return

    # Common patterns: classmethod reset(), reset_circuit_breaker(), or breaker.reset().
    for meth_name in ("reset_circuit_breaker", "reset", "clear"):
        meth = getattr(bridge, meth_name, None)
        if callable(meth):
            try:
                meth()
                return
            except Exception:
                # Continue searching; do not fail tests from reset attempts.
                pass

    breaker = getattr(bridge, "_circuit_breaker", None)
    if breaker is not None:
        for breaker_meth_name in ("reset", "clear", "close"):
            breaker_meth = getattr(breaker, breaker_meth_name, None)
            if callable(breaker_meth):
                try:
                    breaker_meth()
                    return
                except Exception:
                    pass


def _run_awaitable_from_sync(value: Any) -> Any:
    """
    Execute an awaitable from synchronous test code and return its result.

    Why this exists (mirrors graph/embedding):
    - Some adapters expose async-only returns even when called from a sync surface,
      or return awaitables from alias/multiplexed methods.
    - Sync tests should remain robust without assuming an event loop is available.

    Event-loop safety:
    - If no loop is running in this thread, we use asyncio.run (fast path).
    - If a loop *is* running (unusual for sync tests), we execute in a worker
      thread and use asyncio.run there to avoid nested-loop hazards.
    """
    if not inspect.isawaitable(value):
        return value

    # Fast path: no running loop in this thread.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)  # type: ignore[arg-type]

    # Conservative fallback: run in a worker thread to avoid "loop already running".
    def _thread_runner() -> Any:
        return asyncio.run(value)  # type: ignore[arg-type]

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-conformance-await") as ex:
        fut = ex.submit(_thread_runner)
        return fut.result()


async def _consume_one_async(aiter: Any) -> Any:
    """
    Consume at most one item from an async iterator and return it.

    Notes:
    - If the stream yields nothing, this returns None without failing.
    - This is intentionally conservative to keep tests fast and avoid draining streams.
    """
    async for item in aiter:  # noqa: B007
        return item
    return None


def _assert_stream_like(value: Any) -> None:
    """
    Validate that a value looks like a stream surface.

    We accept either:
    - a sync iterable (preferred for sync stream methods),
    - an async iterable, or
    - an awaitable resolving to either kind.

    Additionally:
    - We attempt to consume at most one element to catch lazy validation errors that
      only occur once iteration begins (a common nuance in streaming).
    - StopIteration / empty streams are allowed and treated as success.
    """
    value = _run_awaitable_from_sync(value)

    # Async iterator case: consume one item in a controlled way.
    if hasattr(value, "__aiter__") and callable(getattr(value, "__aiter__", None)):
        _run_awaitable_from_sync(_consume_one_async(value))
        return

    # Sync iterator case.
    try:
        it = iter(value)
    except TypeError as e:
        raise AssertionError(f"Expected an iterable/async-iterable stream, got {type(value).__name__}") from e

    # Consume at most one item to catch lazy failures without draining the stream.
    try:
        next(it)
    except StopIteration:
        # Empty streams are valid; some adapters may yield nothing for trivial prompts.
        pass


async def _assert_stream_like_async(value: Any) -> None:
    """
    Async-native version of _assert_stream_like.

    Why this exists (mirrors graph):
    - In async tests, we should avoid spinning up worker threads or calling asyncio.run.
    - This helper handles awaitables + async iterables in the native event loop.

    Behavior:
    - If value is awaitable, await it once.
    - If the resulting value is an async iterable, consume at most one item.
    - If it is a sync iterable, consume at most one item.
    """
    if inspect.isawaitable(value):
        value = await value  # noqa: PLW2901

    if hasattr(value, "__aiter__") and callable(getattr(value, "__aiter__", None)):
        await _consume_one_async(value)
        return

    try:
        it = iter(value)
    except TypeError as e:
        raise AssertionError(f"Expected an iterable/async-iterable stream, got {type(value).__name__}") from e

    try:
        next(it)
    except StopIteration:
        pass


def _maybe_call_with_context(
    descriptor: LLMFrameworkDescriptor,
    fn: Callable[..., Any],
    prompt_or_messages: Any,
    context: Any,
    *,
    extra_kwargs: Optional[dict[str, Any]] = None,
) -> Any:
    """
    Call an LLM function, respecting descriptor.context_kwarg if present.

    Compatibility behavior (mirrors graph/embedding patterns):
      - If descriptor.context_kwarg is set, pass {context_kwarg: context}.
      - If that raises TypeError due to an unexpected keyword argument AND context is a Mapping,
        retry by spreading the mapping into kwargs (**context). This supports **kwargs-style surfaces.
      - If context is invalid (non-Mapping) and the call fails with TypeError/ValueError, retry
        without context to validate graceful tolerance (ignore invalid context).

    NOTE:
    - We keep call shapes intentionally minimal and portable.
    - Dedicated LLM contract tests should validate full message/tool semantics.
    """
    kwargs: dict[str, Any] = dict(extra_kwargs or {})

    if not descriptor.context_kwarg:
        return fn(prompt_or_messages, **kwargs)

    try:
        return fn(prompt_or_messages, **kwargs, **{descriptor.context_kwarg: context})
    except TypeError as e:
        msg = str(e)
        unexpected_kw = f"unexpected keyword argument '{descriptor.context_kwarg}'" in msg or (
            "unexpected keyword" in msg and descriptor.context_kwarg in msg
        )
        if unexpected_kw and isinstance(context, ABCMapping):
            return fn(prompt_or_messages, **kwargs, **dict(context))

        if not isinstance(context, ABCMapping):
            # Invalid-context tolerance: retry without context rather than crashing.
            return fn(prompt_or_messages, **kwargs)
        raise
    except ValueError:
        if not isinstance(context, ABCMapping):
            return fn(prompt_or_messages, **kwargs)
        raise


def _patch_attach_context(
    adapter_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[BaseException, dict[str, Any]]]:
    """
    Patch attach_context in both:
      1) the adapter module (module-local reference used by decorators), and
      2) the shared corpus_sdk.core.error_context module (defensive best-effort).

    This ensures we observe context attachment even if an adapter references either symbol.
    """
    calls: list[tuple[BaseException, dict[str, Any]]] = []

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        calls.append((exc, dict(ctx)))

    # Patch adapter module local symbol (most common pattern).
    if hasattr(adapter_module, "attach_context"):
        monkeypatch.setattr(adapter_module, "attach_context", fake_attach_context)

    # Patch shared canonical location for safety.
    try:
        core_mod = importlib.import_module("corpus_sdk.core.error_context")
        if hasattr(core_mod, "attach_context"):
            monkeypatch.setattr(core_mod, "attach_context", fake_attach_context)
    except Exception:
        # If this import fails in a minimal environment, we still keep the module-local patch.
        pass

    return calls


def _assert_error_context_minimum(
    descriptor: LLMFrameworkDescriptor,
    ctx: dict[str, Any],
) -> None:
    """
    Assert minimum error-context fields for conformance alignment.

    We intentionally enforce:
      - framework
      - operation
      - error_codes

    Operation name is expected to look like an LLM operation (e.g. starts with "llm_").
    Keep this tolerant: different frameworks may choose slightly different operation labels
    while still remaining useful for debugging.
    """
    assert "framework" in ctx
    assert "operation" in ctx
    assert "error_codes" in ctx

    assert isinstance(ctx["framework"], str) and ctx["framework"]
    op = ctx["operation"]
    assert isinstance(op, str) and op

    allowed_exact = {
        "capabilities",
        "health",
        "complete",
        "stream",
        "count_tokens",
        "llm_capabilities",
        "llm_health",
        "llm_complete",
        "llm_stream",
        "llm_count_tokens",
    }
    assert op.startswith(LLM_OPERATION_PREFIX) or op in allowed_exact, (
        f"{descriptor.name}: unexpected operation name {op!r}; "
        f"expected prefix {LLM_OPERATION_PREFIX!r} or one of {sorted(allowed_exact)}"
    )

    # error_codes should be a low-cardinality structure (list/tuple/set of strings) or a string.
    codes = ctx.get("error_codes")
    assert codes is not None
    if isinstance(codes, (list, tuple, set)):
        for c in codes:
            assert isinstance(c, str) and c
    elif isinstance(codes, str):
        assert codes
    else:
        raise AssertionError(f"{descriptor.name}: error_codes must be a string or sequence of strings, got {type(codes).__name__}")


def _iter_registry_methods(descriptor: LLMFrameworkDescriptor) -> list[tuple[str, str, bool]]:
    """
    Iterate over *all* registry-declared method names and categorize them.

    Returns a list of tuples:
      (method_name, kind, is_async)

    kind is one of:
      - complete, stream_method, stream_kwarg, token_count, capabilities, health

    Why this helper exists (mirrors graph):
    - The registry is the source of truth for method coverage.
    - This conformance file must ensure that every declared surface is tested.
    """
    methods: list[tuple[str, str, bool]] = []

    # Completion
    if descriptor.completion_method:
        methods.append((descriptor.completion_method, "complete", False))
    if descriptor.async_completion_method:
        methods.append((descriptor.async_completion_method, "complete", True))

    # Streaming (method style)
    if descriptor.streaming_method:
        methods.append((descriptor.streaming_method, "stream_method", False))
    if descriptor.async_streaming_method:
        methods.append((descriptor.async_streaming_method, "stream_method", True))

    # Streaming (kwarg style): use completion_method as the callable surface,
    # but categorize separately so tests can apply the streaming kwarg.
    if descriptor.streaming_kwarg and descriptor.completion_method:
        methods.append((descriptor.completion_method, "stream_kwarg", False))
    if descriptor.streaming_kwarg and descriptor.async_completion_method:
        methods.append((descriptor.async_completion_method, "stream_kwarg", True))

    # Token counting
    if descriptor.token_count_method:
        methods.append((descriptor.token_count_method, "token_count", False))
    if descriptor.async_token_count_method:
        methods.append((descriptor.async_token_count_method, "token_count", True))

    # Capabilities/health: registry provides boolean flags but not method names.
    # Conformance expectation:
    # - If has_capabilities is True, the adapter should expose at least one of:
    #       capabilities() or acapabilities()
    # - If has_health is True, the adapter should expose at least one of:
    #       health() or ahealth()
    #
    # We include the conventional names here so "all methods" tests cover them.
    if descriptor.has_capabilities:
        methods.append(("capabilities", "capabilities", False))
        methods.append(("acapabilities", "capabilities", True))
    if descriptor.has_health:
        methods.append(("health", "health", False))
        methods.append(("ahealth", "health", True))

    # Deduplicate while preserving order.
    deduped: list[tuple[str, str, bool]] = []
    seen: set[tuple[str, bool, str]] = set()
    for name, kind, is_async in methods:
        key = (name, is_async, kind)
        if key not in seen:
            deduped.append((name, kind, is_async))
            seen.add(key)

    return deduped


def _best_effort_call_args(kind: str) -> tuple[list[Any], dict[str, Any]]:
    """
    Provide best-effort positional arguments for a given method kind.

    IMPORTANT:
    - These are minimal placeholders intended to exercise adapter wiring.
    - Dedicated contract tests elsewhere should validate deep semantics.
    """
    if kind == "complete":
        return [PROMPT_TEXT], {}
    if kind == "stream_method":
        return [STREAM_PROMPT_TEXT], {}
    if kind == "stream_kwarg":
        # Uses completion method with streaming kwarg; args are still the prompt.
        return [STREAM_PROMPT_TEXT], {}
    if kind == "token_count":
        # Many token counters accept either prompt text or message lists; tests should tolerate both.
        return [PROMPT_TEXT], {}
    if kind in {"capabilities", "health"}:
        return [], {}
    raise AssertionError(f"Unknown method kind: {kind!r}")


def _call_declared_method_with_context_best_effort(
    descriptor: LLMFrameworkDescriptor,
    fn: Callable[..., Any],
    kind: str,
    context: Any,
) -> Any:
    """
    Best-effort invocation of any registry-declared method with context.

    This is intentionally conservative (mirrors graph patterns):
    - If descriptor.context_kwarg is set, we try to pass {context_kwarg: context}.
    - If that fails due to unexpected kwarg and context is a Mapping, we retry by expanding **context.
    - If context is invalid (non-Mapping) and raises TypeError/ValueError, we retry without context.

    Streaming nuance:
    - For streaming_kwarg kinds, we pass descriptor.streaming_kwarg=True.
    """
    args, kwargs = _best_effort_call_args(kind)

    # Streaming kwarg path is driven through completion method calls.
    if kind == "stream_kwarg":
        if descriptor.streaming_kwarg:
            kwargs = {**kwargs, descriptor.streaming_kwarg: True}

    # Token counting sometimes prefers message-shaped inputs; do a minimal retry if needed.
    if kind == "token_count":
        try:
            return _maybe_call_with_context(descriptor, fn, args[0], context=context, extra_kwargs=kwargs)
        except TypeError:
            # Retry with a portable message list. This preserves conformance intent
            # without forcing a single signature across frameworks.
            return _maybe_call_with_context(descriptor, fn, FALLBACK_MESSAGES, context=context, extra_kwargs=kwargs)

    # Default behavior for other kinds.
    if args:
        return _maybe_call_with_context(descriptor, fn, args[0], context=context, extra_kwargs=kwargs)
    # No-arg methods (capabilities/health) still accept context in some frameworks; pass only if supported.
    if not descriptor.context_kwarg:
        return fn(**kwargs)
    try:
        return fn(**kwargs, **{descriptor.context_kwarg: context})
    except TypeError as e:
        msg = str(e)
        unexpected_kw = f"unexpected keyword argument '{descriptor.context_kwarg}'" in msg or (
            "unexpected keyword" in msg and descriptor.context_kwarg in msg
        )
        if unexpected_kw and isinstance(context, ABCMapping):
            return fn(**kwargs, **dict(context))
        if not isinstance(context, ABCMapping):
            return fn(**kwargs)
        raise
    except ValueError:
        if not isinstance(context, ABCMapping):
            return fn(**kwargs)
        raise


def _build_error_wrapped_client_instance(
    framework_descriptor: LLMFrameworkDescriptor,
    failing_llm_adapter: Any,
) -> Any:
    """
    Construct an LLM client instance wired to a failing LLM adapter.

    Used only for error-context tests (we expect calls to raise).

    IMPORTANT:
    - We mirror llm_client_instance construction logic so error-context tests do not
      accidentally pass due to different constructor kwargs.
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    client_cls = getattr(module, framework_descriptor.adapter_class)

    try:
        return client_cls(llm_adapter=failing_llm_adapter)
    except TypeError as e:
        msg = str(e)
        unexpected_llm_adapter = "llm_adapter" in msg and "unexpected keyword" in msg
        if not unexpected_llm_adapter:
            raise

    for alt_kw in ("corpus_adapter", "adapter"):
        try:
            return client_cls(**{alt_kw: failing_llm_adapter})
        except TypeError:
            continue

    raise TypeError(
        f"{framework_descriptor.name}: could not construct adapter {framework_descriptor.adapter_class!r} "
        f"from module {framework_descriptor.adapter_module!r} using known injection kwargs"
    )


# ---------------------------------------------------------------------------
# Registry method coverage tests
# ---------------------------------------------------------------------------


def test_registry_declared_methods_exist_and_are_callable_when_available(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    Registry-method presence conformance.

    If a framework is available, every method name declared on the registry descriptor
    must exist on the instantiated client and be callable.

    This is a strict contract: if the registry declares a method, it must be present.

    Availability policy:
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert llm_client_instance is not None

    for method_name, kind, is_async in _iter_registry_methods(framework_descriptor):
        # Capabilities/health are represented as conventional names; validate presence only
        # when the registry claims they exist, and accept either sync or async variants.
        if kind == "capabilities":
            if hasattr(llm_client_instance, "capabilities") or hasattr(llm_client_instance, "acapabilities"):
                continue
            raise AssertionError(
                f"{framework_descriptor.name}: has_capabilities=True but neither 'capabilities' nor 'acapabilities' exists"
            )
        if kind == "health":
            if hasattr(llm_client_instance, "health") or hasattr(llm_client_instance, "ahealth"):
                continue
            raise AssertionError(
                f"{framework_descriptor.name}: has_health=True but neither 'health' nor 'ahealth' exists"
            )

        # Normal declared methods must exist by name.
        _get_method(llm_client_instance, method_name)


def test_registry_flags_are_coherent_with_declared_methods_when_available(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    Registry coherence conformance.

    The registry includes boolean flags describing which surfaces are expected:
    - supports_streaming / supports_token_count
    - has_capabilities / has_health

    This test ensures those flags are coherent with the method name fields.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert llm_client_instance is not None

    # Streaming coherence: if supports_streaming, we require *some* streaming mechanism.
    if framework_descriptor.supports_streaming:
        assert (
            framework_descriptor.streaming_method
            or framework_descriptor.async_streaming_method
            or framework_descriptor.streaming_kwarg
        ), f"{framework_descriptor.name}: supports_streaming=True but no streaming_method/async_streaming_method/streaming_kwarg"

    # Token counting coherence
    if framework_descriptor.supports_token_count:
        assert (
            framework_descriptor.token_count_method
            or framework_descriptor.async_token_count_method
        ), f"{framework_descriptor.name}: supports_token_count=True but no token_count_method/async_token_count_method"

    # Capabilities/health coherence: registry signals should imply conventional surface existence.
    if framework_descriptor.has_capabilities:
        assert hasattr(llm_client_instance, "capabilities") or hasattr(llm_client_instance, "acapabilities"), (
            f"{framework_descriptor.name}: has_capabilities=True but no capabilities/acapabilities method present"
        )

    if framework_descriptor.has_health:
        assert hasattr(llm_client_instance, "health") or hasattr(llm_client_instance, "ahealth"), (
            f"{framework_descriptor.name}: has_health=True but no health/ahealth method present"
        )


# ---------------------------------------------------------------------------
# Context contract tests
# ---------------------------------------------------------------------------


def test_rich_mapping_context_is_accepted_across_all_registry_declared_sync_methods(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    Ensure *every registry-declared sync method* is exercised under rich Mapping context.

    Why this exists (mirrors graph):
    - The LLM registry declares multiple method surfaces (completion/stream/token_count/etc.).
    - This file must ensure all those surfaces tolerate context consistently.

    Policy:
    - If framework is unavailable, validate the unavailable contract and return.
    - If framework does not declare a context_kwarg, validate that fact and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert llm_client_instance is not None

    if not framework_descriptor.context_kwarg:
        assert framework_descriptor.context_kwarg is None
        return

    rich_context = {**RICH_CONTEXT, "tags": [*RICH_CONTEXT["tags"], framework_descriptor.name]}

    for method_name, kind, is_async in _iter_registry_methods(framework_descriptor):
        if is_async:
            continue

        # Skip conventional cap/health names if they do not exist on the instance.
        if kind == "capabilities" and not hasattr(llm_client_instance, "capabilities"):
            continue
        if kind == "health" and not hasattr(llm_client_instance, "health"):
            continue

        fn = _get_method(llm_client_instance, method_name)

        out = _call_declared_method_with_context_best_effort(
            framework_descriptor,
            fn,
            kind,
            context=rich_context,
        )

        # Streaming nuance: validate stream-like shape without draining the stream.
        if kind in {"stream_method", "stream_kwarg"}:
            _assert_stream_like(out)
        else:
            out = _run_awaitable_from_sync(out)
            assert out is not None


def test_invalid_context_is_tolerated_across_all_registry_declared_sync_methods(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    Ensure *every registry-declared sync method* tolerates invalid (non-Mapping) context.

    Contract expectation (mirrors graph):
    - Adapters may ignore invalid context types rather than crashing.
    - If a method rejects invalid context with TypeError/ValueError, retry without context
      should still succeed.

    Policy:
    - If framework is unavailable, validate the unavailable contract and return.
    - If framework does not declare a context_kwarg, validate that fact and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert llm_client_instance is not None

    if not framework_descriptor.context_kwarg:
        assert framework_descriptor.context_kwarg is None
        return

    invalid_contexts = ["not-a-mapping", 12345]

    for method_name, kind, is_async in _iter_registry_methods(framework_descriptor):
        if is_async:
            continue

        if kind == "capabilities" and not hasattr(llm_client_instance, "capabilities"):
            continue
        if kind == "health" and not hasattr(llm_client_instance, "health"):
            continue

        fn = _get_method(llm_client_instance, method_name)

        for invalid_ctx in invalid_contexts:
            out = _call_declared_method_with_context_best_effort(
                framework_descriptor,
                fn,
                kind,
                context=invalid_ctx,
            )

            if kind in {"stream_method", "stream_kwarg"}:
                _assert_stream_like(out)
            else:
                out = _run_awaitable_from_sync(out)
                assert out is not None


def test_context_is_optional_across_all_registry_declared_sync_methods(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    Ensure *every registry-declared sync method* remains functional when context is omitted.

    Policy:
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert llm_client_instance is not None

    for method_name, kind, is_async in _iter_registry_methods(framework_descriptor):
        if is_async:
            continue

        if kind == "capabilities" and not hasattr(llm_client_instance, "capabilities"):
            continue
        if kind == "health" and not hasattr(llm_client_instance, "health"):
            continue

        fn = _get_method(llm_client_instance, method_name)
        args, kwargs = _best_effort_call_args(kind)

        # For stream_kwarg, apply streaming flag through the completion method.
        if kind == "stream_kwarg" and framework_descriptor.streaming_kwarg:
            kwargs = {**kwargs, framework_descriptor.streaming_kwarg: True}

        # Token count signature tolerance: retry message-list if needed.
        if kind == "token_count":
            try:
                out = fn(*args, **kwargs)
            except TypeError:
                out = fn(FALLBACK_MESSAGES, **kwargs)
        else:
            out = fn(*args, **kwargs)

        if kind in {"stream_method", "stream_kwarg"}:
            _assert_stream_like(out)
        else:
            out = _run_awaitable_from_sync(out)
            assert out is not None


@pytest.mark.asyncio
async def test_rich_mapping_context_is_accepted_across_all_registry_declared_async_methods(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    Ensure *every registry-declared async method* is exercised under rich Mapping context.

    Policy:
    - If framework is unavailable, validate the unavailable contract and return.
    - If no async methods are declared, validate that and return.
    - If framework does not declare a context_kwarg, validate that fact and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert llm_client_instance is not None

    if not framework_descriptor.supports_async:
        return

    if not framework_descriptor.context_kwarg:
        assert framework_descriptor.context_kwarg is None
        return

    rich_context = {**RICH_CONTEXT, "tags": [*RICH_CONTEXT["tags"], framework_descriptor.name]}

    for method_name, kind, is_async in _iter_registry_methods(framework_descriptor):
        if not is_async:
            continue

        # Skip conventional cap/health names if they do not exist on the instance.
        if kind == "capabilities" and not hasattr(llm_client_instance, "acapabilities"):
            continue
        if kind == "health" and not hasattr(llm_client_instance, "ahealth"):
            continue

        fn = _get_method(llm_client_instance, method_name)
        out = _call_declared_method_with_context_best_effort(
            framework_descriptor,
            fn,
            kind,
            context=rich_context,
        )

        if kind in {"stream_method", "stream_kwarg"}:
            await _assert_stream_like_async(out)
        else:
            assert inspect.isawaitable(out), "Async method must return an awaitable"
            res = await out  # noqa: PT018
            assert res is not None


@pytest.mark.asyncio
async def test_invalid_context_is_tolerated_across_all_registry_declared_async_methods(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    Ensure *every registry-declared async method* tolerates invalid (non-Mapping) context.

    Policy:
    - If framework is unavailable, validate the unavailable contract and return.
    - If no async methods are declared, validate that and return.
    - If framework does not declare a context_kwarg, validate that fact and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert llm_client_instance is not None

    if not framework_descriptor.supports_async:
        return

    if not framework_descriptor.context_kwarg:
        assert framework_descriptor.context_kwarg is None
        return

    invalid_contexts = ["not-a-mapping", 12345]

    for method_name, kind, is_async in _iter_registry_methods(framework_descriptor):
        if not is_async:
            continue

        if kind == "capabilities" and not hasattr(llm_client_instance, "acapabilities"):
            continue
        if kind == "health" and not hasattr(llm_client_instance, "ahealth"):
            continue

        fn = _get_method(llm_client_instance, method_name)

        for invalid_ctx in invalid_contexts:
            out = _call_declared_method_with_context_best_effort(
                framework_descriptor,
                fn,
                kind,
                context=invalid_ctx,
            )

            if kind in {"stream_method", "stream_kwarg"}:
                await _assert_stream_like_async(out)
            else:
                assert inspect.isawaitable(out), "Async method must return an awaitable"
                res = await out  # noqa: PT018
                assert res is not None


@pytest.mark.asyncio
async def test_context_is_optional_across_all_registry_declared_async_methods(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    Ensure *every registry-declared async method* remains functional when context is omitted.

    Policy:
    - If framework is unavailable, validate the unavailable contract and return.
    - If no async methods are declared, validate that and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert llm_client_instance is not None

    if not framework_descriptor.supports_async:
        return

    for method_name, kind, is_async in _iter_registry_methods(framework_descriptor):
        if not is_async:
            continue

        if kind == "capabilities" and not hasattr(llm_client_instance, "acapabilities"):
            continue
        if kind == "health" and not hasattr(llm_client_instance, "ahealth"):
            continue

        fn = _get_method(llm_client_instance, method_name)
        args, kwargs = _best_effort_call_args(kind)

        if kind == "stream_kwarg" and framework_descriptor.streaming_kwarg:
            kwargs = {**kwargs, framework_descriptor.streaming_kwarg: True}

        if kind == "token_count":
            try:
                out = fn(*args, **kwargs)
            except TypeError:
                out = fn(FALLBACK_MESSAGES, **kwargs)
        else:
            out = fn(*args, **kwargs)

        if kind in {"stream_method", "stream_kwarg"}:
            await _assert_stream_like_async(out)
        else:
            assert inspect.isawaitable(out), "Async method must return an awaitable"
            res = await out  # noqa: PT018
            assert res is not None


# ---------------------------------------------------------------------------
# Error-context decorator contract tests
# ---------------------------------------------------------------------------


def test_error_context_is_attached_on_sync_failure_for_all_registry_declared_methods(
    framework_descriptor: LLMFrameworkDescriptor,
    failing_llm_adapter_instream: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure *every registry-declared sync method* attaches error context on failure.

    Why this matters (mirrors graph):
    - The LLM registry declares multiple surfaces beyond completion.
    - All of them must be wrapped by error-context decorators to support debugging.

    Notes:
    - This test uses a failing adapter that raises for all operations.
    - Streaming failures are triggered by iterating, to match real-world failure modes.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    # Ensure prior expected failures do not trip sticky circuit breakers and poison this test.
    _reset_async_bridge_state_best_effort()

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(module, monkeypatch)

    instance = _build_error_wrapped_client_instance(framework_descriptor, failing_llm_adapter_instream)
    assert instance is not None

    for method_name, kind, is_async in _iter_registry_methods(framework_descriptor):
        if is_async:
            continue

        # Reset best-effort between operations to reduce the chance of sticky failures
        # masking underlying exceptions (e.g., circuit breaker behavior).
        _reset_async_bridge_state_best_effort()

        # Skip conventional cap/health names if they do not exist on the instance.
        if kind == "capabilities" and not hasattr(instance, "capabilities"):
            continue
        if kind == "health" and not hasattr(instance, "health"):
            continue

        fn = _get_method(instance, method_name)

        with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
            out = _call_declared_method_with_context_best_effort(
                framework_descriptor,
                fn,
                kind,
                context={},
            )

            if kind in {"stream_method", "stream_kwarg"}:
                # Trigger in-stream failure (decorator should see exceptions raised during iteration).
                for _ in out:  # noqa: B007
                    pass
            else:
                # If a sync method returns an awaitable (rare but possible), execute it safely
                # so that call-site failures still propagate correctly.
                _run_awaitable_from_sync(out)

        assert calls, f"attach_context was not called for sync method {method_name!r}"
        exc, ctx = calls[-1]
        assert isinstance(exc, RuntimeError)
        _assert_error_context_minimum(framework_descriptor, ctx)


def test_error_context_is_attached_on_sync_stream_calltime_failure_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
    failing_llm_adapter_calltime: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When streaming is supported, sync streaming failures that occur at call-time
    must go through the error-context decorator and call attach_context().

    This mirrors the graph call-time streaming failure test.

    Policy:
    - If framework is unavailable, validate the unavailable contract and return.
    - If streaming is not declared, validate that and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.supports_streaming:
        return

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(module, monkeypatch)

    instance = _build_error_wrapped_client_instance(framework_descriptor, failing_llm_adapter_calltime)
    assert instance is not None

    # Prefer explicit streaming_method if present; else test kwarg streaming via completion_method.
    if framework_descriptor.streaming_method:
        stream_fn = _get_method(instance, framework_descriptor.streaming_method)
        with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
            out = _maybe_call_with_context(
                framework_descriptor,
                stream_fn,
                STREAM_PROMPT_TEXT,
                context={},
            )
            # For call-time failures, error may be raised immediately.
            # If not, Python generators may raise on first iteration attempt.
            if hasattr(out, "__iter__"):
                next(iter(out))
    elif framework_descriptor.streaming_kwarg and framework_descriptor.completion_method:
        complete_fn = _get_method(instance, framework_descriptor.completion_method)
        with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
            out = _maybe_call_with_context(
                framework_descriptor,
                complete_fn,
                STREAM_PROMPT_TEXT,
                context={},
                extra_kwargs={framework_descriptor.streaming_kwarg: True},
            )
            if hasattr(out, "__iter__"):
                next(iter(out))
    else:
        # supports_streaming=True but no known mechanism is declared; registry coherence test covers this.
        return

    assert calls, "attach_context was not called on sync stream call-time failure"
    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    _assert_error_context_minimum(framework_descriptor, ctx)


@pytest.mark.asyncio
async def test_error_context_is_attached_on_async_failure_for_all_registry_declared_methods(
    framework_descriptor: LLMFrameworkDescriptor,
    failing_llm_adapter_instream: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure *every registry-declared async method* attaches error context on failure.

    Notes (mirrors graph):
    - Streaming failures are triggered by async iteration.
    - This test complements the sync "all methods" failure test above.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.supports_async:
        return

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(module, monkeypatch)

    instance = _build_error_wrapped_client_instance(framework_descriptor, failing_llm_adapter_instream)
    assert instance is not None

    for method_name, kind, is_async in _iter_registry_methods(framework_descriptor):
        if not is_async:
            continue

        # Skip conventional cap/health names if they do not exist on the instance.
        if kind == "capabilities" and not hasattr(instance, "acapabilities"):
            continue
        if kind == "health" and not hasattr(instance, "ahealth"):
            continue

        fn = _get_method(instance, method_name)

        with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
            out = _call_declared_method_with_context_best_effort(
                framework_descriptor,
                fn,
                kind,
                context={},
            )

            if kind in {"stream_method", "stream_kwarg"}:
                # Handle awaitable-to-aiter or direct async iterator.
                if inspect.isawaitable(out):
                    out = await out  # noqa: PLW2901
                async for _ in out:  # noqa: B007
                    pass
            else:
                assert inspect.isawaitable(out), f"Async method {method_name!r} must return an awaitable"
                await out  # noqa: PT018

        assert calls, f"attach_context was not called for async method {method_name!r}"
        exc, ctx = calls[-1]
        assert isinstance(exc, RuntimeError)
        _assert_error_context_minimum(framework_descriptor, ctx)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
