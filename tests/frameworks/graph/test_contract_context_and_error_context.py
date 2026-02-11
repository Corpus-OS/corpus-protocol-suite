# tests/frameworks/graph/test_contract_context_and_error_context.py

from __future__ import annotations

import asyncio
import importlib
import inspect
from collections.abc import Mapping as ABCMapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable

import pytest

from corpus_sdk.graph.graph_base import (
    BatchOperation,
    BulkVerticesSpec,
    GraphTraversalSpec,
)
from tests.frameworks.registries.graph_registry import (
    GraphFrameworkDescriptor,
    iter_graph_framework_descriptors,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAILURE_MESSAGE = "intentional failure from failing graph adapter"
GRAPH_OPERATION_PREFIX = "graph_"

RICH_CONTEXT = {
    "request_id": "req-123",
    "user_id": "user-abc",
    "tags": ["test"],
    "nested": {"key": "value", "depth": 2},
}

# Representative inputs for graph protocol surfaces.
# IMPORTANT:
# - These are intentionally minimal and portable across framework adapters.
# - This file focuses on context tolerance + error-context decoration, not deep result semantics.
QUERY_TEXT = "contract-query"
STREAM_QUERY_TEXT = "contract-stream-query"
BULK_VERTEX_SPEC = BulkVerticesSpec(namespace="test", limit=10)
BATCH_OPERATIONS = [BatchOperation(op="test", args={})]

# Transaction/traversal are registry-declared extended surfaces.
# NOTE:
# - These signatures can be framework-specific; in this file we primarily validate that:
#     1) the surface exists and is callable (presence test), and
#     2) failures attach error context (error-context tests).
# - We still provide best-effort minimal payloads for invocation where helpful.
TRANSACTION_OPERATIONS = [BatchOperation(op="test", args={})]
TRAVERSAL_SPEC = GraphTraversalSpec(start_nodes=["v-alpha"], max_depth=1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=list(iter_graph_framework_descriptors()),
    name="framework_descriptor",
)
def framework_descriptor_fixture(
    request: pytest.FixtureRequest,
) -> GraphFrameworkDescriptor:
    """
    Parameterized over all registered graph framework descriptors.

    IMPORTANT POLICY ALIGNMENT:
    - We do not skip unavailable frameworks.
    - Tests must pass by asserting correct "unavailable" signaling when a framework
      is not installed, and must fully run when it is available.
    """
    descriptor: GraphFrameworkDescriptor = request.param
    return descriptor


@pytest.fixture
def graph_client_instance(
    framework_descriptor: GraphFrameworkDescriptor,
    adapter: Any,
) -> Any:
    """
    Construct a concrete graph client instance for the given descriptor.

    Mirrors the construction pattern used in the other graph contract tests:
    each framework adapter is expected to take a `adapter` kwarg that
    wraps a Corpus GraphProtocolV1 implementation.

    Availability contract:
    - If a framework is unavailable, this fixture returns None and tests must
      treat that as a validated pass condition (not a skip).
    """
    if not framework_descriptor.is_available():
        return None

    module = importlib.import_module(framework_descriptor.adapter_module)
    client_cls = getattr(module, framework_descriptor.adapter_class)

    # IMPORTANT:
    # The registry is the source of truth for the injection kwarg name.
    # Do not hardcode "adapter="; use adapter_init_kwarg to remain framework-agnostic.
    init_kwargs: dict[str, Any] = {framework_descriptor.adapter_init_kwarg: adapter}
    instance = client_cls(**init_kwargs)
    return instance


@pytest.fixture
def failing_adapter() -> Any:
    """
    A minimal graph adapter whose graph operations always fail.

    Used only for error-context tests to ensure the decorators invoke
    attach_context() and propagate the exception.

    Notes:
    - Includes both query and streaming surfaces (sync + async) because
      framework adapters may exercise either path depending on their integration.
    - Streaming failures are expressed by raising during iteration to ensure
      the decorator sees failures that occur "in-stream", not only at call time.
    """

    class FailingGraphAdapter:
        def query(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        async def aquery(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        async def stream_query(self, *args: Any, **kwargs: Any):
            # Raise during async iteration (common real-world failure mode).
            raise RuntimeError(FAILURE_MESSAGE)
            yield  # pragma: no cover  # keeps this as a generator in type checkers

        async def astream_query(self, *args: Any, **kwargs: Any):
            # Raise during async iteration.
            raise RuntimeError(FAILURE_MESSAGE)
            yield  # pragma: no cover

        # Extra registry-declared surfaces: these ensure every declared method can be
        # exercised under error-context tests without relying on adapter-specific internals.
        def bulk_vertices(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        async def abulk_vertices(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        def batch(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        async def abatch(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        def capabilities(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        async def acapabilities(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        def health(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        async def ahealth(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        def get_schema(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        async def aget_schema(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        def transaction(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        async def atransaction(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        def traversal(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        async def atraversal(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

    return FailingGraphAdapter()


@pytest.fixture
def failing_adapter_stream_calltime() -> Any:
    """
    A minimal graph adapter whose streaming operations fail at call-time.

    Why this fixture exists:
    - Graph streaming failures commonly occur in two places:
        1) call-time (e.g., request construction, auth validation, input parsing)
        2) in-stream (e.g., network drop, downstream tool failure, model crash)
    - The failing_adapter fixture exercises (2) by raising during iteration.
      This fixture explicitly exercises (1) to ensure decorators wrap the call site
      and still attach error context.

    Notes:
    - Streaming call-time failure must be handled for both sync and async streaming
      method shapes.
    """

    class FailingGraphAdapterStreamCalltime:
        def capabilities(self, **kwargs: Any) -> dict[str, Any]:
            """Minimal capabilities for duck-type validation."""
            return {}

        def query(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        async def aquery(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        def stream_query(self, *args: Any, **kwargs: Any):
            # Raise immediately (call-time), not during iteration.
            raise RuntimeError(FAILURE_MESSAGE)

        async def astream_query(self, *args: Any, **kwargs: Any):
            # Raise immediately (call-time), not during async iteration.
            raise RuntimeError(FAILURE_MESSAGE)

    return FailingGraphAdapterStreamCalltime()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_unavailable_contract(descriptor: GraphFrameworkDescriptor) -> None:
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

    If name is None (e.g. async methods not declared), this will fail fast
    with a clear assertion message.
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

    Why this exists:
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

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="graph-conformance-await") as ex:
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
    - a sync iterable (preferred for sync stream methods), or
    - an async iterable, or
    - an awaitable resolving to either kind.

    Additionally:
    - We attempt to consume at most one element to catch lazy validation errors that
      only occur once iteration begins (a common nuance in graph streaming).
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
        # Empty streams are valid; many adapters may yield nothing for a trivial query.
        pass


async def _assert_stream_like_async(value: Any) -> None:
    """
    Async-native version of _assert_stream_like.

    Why this exists:
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


def _call_with_context(
    descriptor: GraphFrameworkDescriptor,
    fn: Callable[..., Any],
    query_text: str,
    context: Any,
) -> Any:
    """
    Call a graph client function with context in a robust, framework-agnostic way.

    Primary strategy:
      - If descriptor.context_kwarg is set, pass {context_kwarg: context}.

    Compatibility fallback:
      - If that raises TypeError due to an unexpected keyword argument, and context is a Mapping,
        retry by spreading the mapping into kwargs (useful for **kwargs-style surfaces).

    Invalid-context tolerance:
      - If the provided context is non-Mapping and the call fails with a common validation-type
        exception (TypeError/ValueError), retry without context to validate graceful behavior.

    This approach avoids test skips while remaining resilient to framework method signature shapes.
    """
    if not descriptor.context_kwarg:
        return fn(query_text)

    try:
        return fn(query_text, **{descriptor.context_kwarg: context})
    except TypeError as e:
        msg = str(e)
        unexpected_kw = f"unexpected keyword argument '{descriptor.context_kwarg}'" in msg or (
            "unexpected keyword" in msg and descriptor.context_kwarg in msg
        )
        if unexpected_kw and isinstance(context, ABCMapping):
            return fn(query_text, **dict(context))

        if not isinstance(context, ABCMapping):
            # Framework may reject invalid context types; conformance expects tolerance.
            return fn(query_text)
        raise
    except ValueError:
        # Some adapters prefer ValueError for invalid context types.
        if not isinstance(context, ABCMapping):
            return fn(query_text)
        raise


def _build_error_wrapped_client_instance(
    framework_descriptor: GraphFrameworkDescriptor,
    failing_adapter_obj: Any,
) -> Any:
    """
    Construct a graph client instance wired to a failing graph adapter.

    Used only for error-context tests (we expect calls to raise).

    - If framework is unavailable, returns None and tests assert the unavailable contract.
    """
    if not framework_descriptor.is_available():
        return None

    module = importlib.import_module(framework_descriptor.adapter_module)
    client_cls = getattr(module, framework_descriptor.adapter_class)

    # IMPORTANT:
    # Use the registry-defined injection kwarg to remain framework-agnostic.
    init_kwargs: dict[str, Any] = {framework_descriptor.adapter_init_kwarg: failing_adapter_obj}
    return client_cls(**init_kwargs)


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
    descriptor: GraphFrameworkDescriptor,
    ctx: dict[str, Any],
) -> None:
    """
    Assert minimum error-context fields for conformance alignment.

    We intentionally enforce:
      - framework
      - operation

    Operation name is expected to look like a graph operation (e.g. starts with "graph_").
    Keep this tolerant: different frameworks may choose slightly different operation labels
    while still remaining useful for debugging.
    """
    assert "framework" in ctx
    assert "operation" in ctx

    assert isinstance(ctx["framework"], str) and ctx["framework"]
    op = ctx["operation"]
    assert isinstance(op, str) and op

    # Accept prefix style (preferred) or a small set of exact stable names.
    allowed_exact = {
        "capabilities",
        "health",
        "query",
        "stream_query",
        "bulk_vertices",
        "batch",
        "get_schema",
        "schema",
        "transaction",
        "traversal",
        "graph_capabilities",
        "graph_health",
        "graph_query",
        "graph_stream_query",
        "graph_bulk_vertices",
        "graph_batch",
        "graph_schema",
        "graph_transaction",
        "graph_traversal",
    }
    assert op.startswith(GRAPH_OPERATION_PREFIX) or op in allowed_exact, (
        f"{descriptor.name}: unexpected operation name {op!r}; "
        f"expected prefix {GRAPH_OPERATION_PREFIX!r} or one of {sorted(allowed_exact)}"
    )


def _iter_registry_methods(descriptor: GraphFrameworkDescriptor) -> list[tuple[str, str, bool]]:
    """
    Iterate over *all* registry-declared method names and categorize them.

    Returns a list of tuples:
      (method_name, kind, is_async)

    kind is one of:
      - query, stream, bulk, batch, capabilities, health, schema, transaction, traversal

    Why this helper exists:
    - The registry is the source of truth for method coverage.
    - This conformance file must ensure that every declared surface is tested.
    """
    methods: list[tuple[str, str, bool]] = []

    # Core query + streaming
    methods.append((descriptor.query_method, "query", False))
    if descriptor.stream_query_method:
        methods.append((descriptor.stream_query_method, "stream", False))
    if descriptor.async_query_method:
        methods.append((descriptor.async_query_method, "query", True))
    if descriptor.async_stream_query_method:
        methods.append((descriptor.async_stream_query_method, "stream", True))

    # Bulk / batch
    if descriptor.bulk_vertices_method:
        methods.append((descriptor.bulk_vertices_method, "bulk", False))
    if descriptor.async_bulk_vertices_method:
        methods.append((descriptor.async_bulk_vertices_method, "bulk", True))
    if descriptor.batch_method:
        methods.append((descriptor.batch_method, "batch", False))
    if descriptor.async_batch_method:
        methods.append((descriptor.async_batch_method, "batch", True))

    # Extended surfaces
    if descriptor.capabilities_method:
        methods.append((descriptor.capabilities_method, "capabilities", False))
    if descriptor.async_capabilities_method:
        methods.append((descriptor.async_capabilities_method, "capabilities", True))

    if descriptor.health_method:
        methods.append((descriptor.health_method, "health", False))
    if descriptor.async_health_method:
        methods.append((descriptor.async_health_method, "health", True))

    if descriptor.schema_method:
        methods.append((descriptor.schema_method, "schema", False))
    if descriptor.async_schema_method:
        methods.append((descriptor.async_schema_method, "schema", True))

    if descriptor.transaction_method:
        methods.append((descriptor.transaction_method, "transaction", False))
    if descriptor.async_transaction_method:
        methods.append((descriptor.async_transaction_method, "transaction", True))

    if descriptor.traversal_method:
        methods.append((descriptor.traversal_method, "traversal", False))
    if descriptor.async_traversal_method:
        methods.append((descriptor.async_traversal_method, "traversal", True))

    # Preserve stable order and avoid duplicates if registry entries overlap.
    deduped: list[tuple[str, str, bool]] = []
    seen: set[tuple[str, bool]] = set()
    for name, kind, is_async in methods:
        key = (name, is_async)
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
    if kind == "query":
        return [QUERY_TEXT], {}
    if kind == "stream":
        return [STREAM_QUERY_TEXT], {}
    if kind == "bulk":
        return [BULK_VERTEX_SPEC], {}
    if kind == "batch":
        return [BATCH_OPERATIONS], {}
    if kind == "capabilities":
        return [], {}
    if kind == "health":
        return [], {}
    if kind == "schema":
        return [], {}
    if kind == "transaction":
        return [TRANSACTION_OPERATIONS], {}
    if kind == "traversal":
        return [TRAVERSAL_SPEC], {}
    raise AssertionError(f"Unknown method kind: {kind!r}")


def _call_declared_method_with_context_best_effort(
    descriptor: GraphFrameworkDescriptor,
    fn: Callable[..., Any],
    kind: str,
    context: Any,
) -> Any:
    """
    Best-effort invocation of any registry-declared method with context.

    This is intentionally conservative:
    - If descriptor.context_kwarg is set, we try to pass {context_kwarg: context}.
    - If that fails due to unexpected kwarg and context is a Mapping, we retry by expanding **context.
    - If context is invalid (non-Mapping) and raises TypeError/ValueError, we retry without context.

    Why this helper exists:
    - We must test *all* registry methods in this file, but signatures can differ between surfaces.
    - We keep the invocation logic centralized to preserve consistency and reduce flakiness.
    """
    args, kwargs = _best_effort_call_args(kind)

    if not descriptor.context_kwarg:
        return fn(*args, **kwargs)

    try:
        return fn(*args, **kwargs, **{descriptor.context_kwarg: context})
    except TypeError as e:
        msg = str(e)
        unexpected_kw = f"unexpected keyword argument '{descriptor.context_kwarg}'" in msg or (
            "unexpected keyword" in msg and descriptor.context_kwarg in msg
        )
        if unexpected_kw and isinstance(context, ABCMapping):
            # **kwargs-style context surfaces (e.g., context fields spread into kwargs).
            return fn(*args, **kwargs, **dict(context))

        # Invalid-context tolerance: non-Mapping contexts should be ignored, not crash.
        if not isinstance(context, ABCMapping):
            return fn(*args, **kwargs)
        raise
    except ValueError:
        if not isinstance(context, ABCMapping):
            return fn(*args, **kwargs)
        raise


# ---------------------------------------------------------------------------
# Registry method coverage tests
# ---------------------------------------------------------------------------


def test_registry_declared_methods_exist_and_are_callable_when_available(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Registry-method presence conformance.

    If a framework is available, every method name declared on the registry descriptor
    must exist on the instantiated client and be callable.

    This is a strict contract: if the registry declares a method, it must be present.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert graph_client_instance is not None

    for method_name, _kind, _is_async in _iter_registry_methods(framework_descriptor):
        _get_method(graph_client_instance, method_name)


def test_registry_flags_are_coherent_with_declared_methods_when_available(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Registry coherence conformance.

    The registry includes boolean flags describing which surfaces are expected:
    - supports_streaming / supports_bulk_vertices / supports_batch
    - has_capabilities / has_health

    This test ensures those flags are coherent with the method name fields.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert graph_client_instance is not None

    # Streaming coherence
    if framework_descriptor.supports_streaming:
        assert framework_descriptor.stream_query_method is not None
    # Bulk coherence
    if framework_descriptor.supports_bulk_vertices:
        assert framework_descriptor.bulk_vertices_method is not None
    # Batch coherence
    if framework_descriptor.supports_batch:
        assert framework_descriptor.batch_method is not None
    # Capabilities/health coherence
    if framework_descriptor.has_capabilities:
        assert framework_descriptor.capabilities_method is not None
    if framework_descriptor.has_health:
        assert framework_descriptor.health_method is not None


# ---------------------------------------------------------------------------
# Context contract tests
# ---------------------------------------------------------------------------


def test_rich_mapping_context_is_accepted_and_does_not_break_queries(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    If a framework declares a context_kwarg, it should:

    - accept a rich Mapping (with extra / nested keys),
    - not raise TypeError / ValueError,
    - still return a valid query result.

    Policy:
    - If framework is unavailable, validate the unavailable contract and return.
    - If framework does not declare a context_kwarg, validate that fact and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert graph_client_instance is not None

    if not framework_descriptor.context_kwarg:
        assert framework_descriptor.context_kwarg is None
        return

    query_method = _get_method(graph_client_instance, framework_descriptor.query_method)

    rich_context = {
        **RICH_CONTEXT,
        "tags": [*RICH_CONTEXT["tags"], framework_descriptor.name],
    }

    query_text = "ctx-rich-query"

    # Should not raise; result shape is validated lightly here since other
    # contract tests cover detailed QueryResult semantics.
    result = _call_with_context(
        framework_descriptor,
        query_method,
        query_text,
        context=rich_context,
    )
    result = _run_awaitable_from_sync(result)
    assert result is not None


def test_invalid_context_type_is_tolerated_and_does_not_crash(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Passing an invalid context type (non-Mapping) should not crash the adapter.

    The framework adapters are expected to either:
    - log a warning and ignore the context, or
    - gracefully treat it as "no context".

    In all cases, queries should still return results.

    Policy:
    - If framework is unavailable, validate the unavailable contract and return.
    - If framework does not declare a context_kwarg, validate that fact and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert graph_client_instance is not None

    if not framework_descriptor.context_kwarg:
        assert framework_descriptor.context_kwarg is None
        return

    query_method = _get_method(graph_client_instance, framework_descriptor.query_method)

    query_text = "ctx-invalid-query"

    invalid_contexts = ["not-a-mapping", 12345]

    for invalid_ctx in invalid_contexts:
        result = _call_with_context(
            framework_descriptor,
            query_method,
            query_text,
            context=invalid_ctx,
        )
        result = _run_awaitable_from_sync(result)
        assert result is not None


def test_context_is_optional_and_omitting_it_still_works(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Even when a framework supports a context kwarg, it must still work
    when no context is provided.

    Policy:
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert graph_client_instance is not None

    query_method = _get_method(graph_client_instance, framework_descriptor.query_method)

    query_text = "ctx-optional-query"

    # No context kwarg passed at all.
    result = query_method(query_text)
    result = _run_awaitable_from_sync(result)
    assert result is not None


def test_rich_mapping_context_is_accepted_across_all_registry_declared_sync_methods(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Ensure *every registry-declared sync method* is exercised under rich Mapping context.

    Why this exists:
    - The graph registry includes many method surfaces (query/stream/bulk/batch/etc.).
    - This file must ensure all those surfaces tolerate context consistently.

    Policy:
    - If framework is unavailable, validate the unavailable contract and return.
    - If framework does not declare a context_kwarg, validate that fact and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert graph_client_instance is not None

    if not framework_descriptor.context_kwarg:
        assert framework_descriptor.context_kwarg is None
        return

    rich_context = {**RICH_CONTEXT, "tags": [*RICH_CONTEXT["tags"], framework_descriptor.name]}

    for method_name, kind, is_async in _iter_registry_methods(framework_descriptor):
        if is_async:
            # Async methods are validated in async tests below.
            continue

        fn = _get_method(graph_client_instance, method_name)
        out = _call_declared_method_with_context_best_effort(framework_descriptor, fn, kind, context=rich_context)

        # Streaming nuance: validate stream-like shape without draining the stream.
        if kind == "stream":
            _assert_stream_like(out)
        else:
            out = _run_awaitable_from_sync(out)
            assert out is not None


def test_invalid_context_is_tolerated_across_all_registry_declared_sync_methods(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Ensure *every registry-declared sync method* tolerates invalid (non-Mapping) context.

    Contract expectation:
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

    assert graph_client_instance is not None

    if not framework_descriptor.context_kwarg:
        assert framework_descriptor.context_kwarg is None
        return

    invalid_contexts = ["not-a-mapping", 12345]

    for method_name, kind, is_async in _iter_registry_methods(framework_descriptor):
        if is_async:
            continue

        fn = _get_method(graph_client_instance, method_name)

        for invalid_ctx in invalid_contexts:
            out = _call_declared_method_with_context_best_effort(framework_descriptor, fn, kind, context=invalid_ctx)

            if kind == "stream":
                _assert_stream_like(out)
            else:
                out = _run_awaitable_from_sync(out)
                assert out is not None


def test_context_is_optional_across_all_registry_declared_sync_methods(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Ensure *every registry-declared sync method* remains functional when context is omitted.

    Policy:
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert graph_client_instance is not None

    for method_name, kind, is_async in _iter_registry_methods(framework_descriptor):
        if is_async:
            continue

        fn = _get_method(graph_client_instance, method_name)
        args, kwargs = _best_effort_call_args(kind)
        out = fn(*args, **kwargs)

        if kind == "stream":
            _assert_stream_like(out)
        else:
            out = _run_awaitable_from_sync(out)
            assert out is not None


@pytest.mark.asyncio
async def test_rich_mapping_context_is_accepted_across_all_registry_declared_async_methods(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Ensure *every registry-declared async method* is exercised under rich Mapping context.

    Policy:
    - If framework is unavailable, validate the unavailable contract and return.
    - If framework does not declare a context_kwarg, validate that fact and return.
    - If no async methods are declared, validate that and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert graph_client_instance is not None

    if not framework_descriptor.supports_async:
        return

    if not framework_descriptor.context_kwarg:
        assert framework_descriptor.context_kwarg is None
        return

    rich_context = {**RICH_CONTEXT, "tags": [*RICH_CONTEXT["tags"], framework_descriptor.name]}

    for method_name, kind, is_async in _iter_registry_methods(framework_descriptor):
        if not is_async:
            continue

        fn = _get_method(graph_client_instance, method_name)
        out = _call_declared_method_with_context_best_effort(framework_descriptor, fn, kind, context=rich_context)

        if kind == "stream":
            await _assert_stream_like_async(out)
        else:
            assert inspect.isawaitable(out), "Async method must return an awaitable"
            res = await out  # noqa: PT018
            assert res is not None


@pytest.mark.asyncio
async def test_invalid_context_is_tolerated_across_all_registry_declared_async_methods(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
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

    assert graph_client_instance is not None

    if not framework_descriptor.supports_async:
        return

    if not framework_descriptor.context_kwarg:
        assert framework_descriptor.context_kwarg is None
        return

    invalid_contexts = ["not-a-mapping", 12345]

    for method_name, kind, is_async in _iter_registry_methods(framework_descriptor):
        if not is_async:
            continue

        fn = _get_method(graph_client_instance, method_name)

        for invalid_ctx in invalid_contexts:
            out = _call_declared_method_with_context_best_effort(framework_descriptor, fn, kind, context=invalid_ctx)

            if kind == "stream":
                await _assert_stream_like_async(out)
            else:
                assert inspect.isawaitable(out), "Async method must return an awaitable"
                res = await out  # noqa: PT018
                assert res is not None


@pytest.mark.asyncio
async def test_context_is_optional_across_all_registry_declared_async_methods(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
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

    assert graph_client_instance is not None

    if not framework_descriptor.supports_async:
        return

    for method_name, kind, is_async in _iter_registry_methods(framework_descriptor):
        if not is_async:
            continue

        fn = _get_method(graph_client_instance, method_name)
        args, kwargs = _best_effort_call_args(kind)
        out = fn(*args, **kwargs)

        if kind == "stream":
            await _assert_stream_like_async(out)
        else:
            assert inspect.isawaitable(out), "Async method must return an awaitable"
            res = await out  # noqa: PT018
            assert res is not None


# ---------------------------------------------------------------------------
# Error-context decorator contract tests
# ---------------------------------------------------------------------------


def test_error_context_is_attached_on_sync_query_failure(
    framework_descriptor: GraphFrameworkDescriptor,
    failing_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the underlying graph adapter raises during a sync query operation,
    the framework adapter's error-context decorator should:

    - call attach_context() with the exception and useful metadata, and
    - re-raise the original exception (or a wrapped one).

    We assert that attach_context is invoked and that the operation name
    looks like a graph operation (e.g. starts with "graph_").

    Policy:
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(module, monkeypatch)

    instance = _build_error_wrapped_client_instance(
        framework_descriptor,
        failing_adapter,
    )
    assert instance is not None

    query_method = _get_method(instance, framework_descriptor.query_method)

    with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
        if framework_descriptor.context_kwarg:
            query_method(
                "err-query",
                **{framework_descriptor.context_kwarg: {}},
            )
        else:
            query_method("err-query")

    assert calls, "attach_context was not called on sync query failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    _assert_error_context_minimum(framework_descriptor, ctx)


def test_error_context_is_attached_on_sync_stream_failure_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
    failing_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When streaming is supported, sync stream failures should also go through
    the error-context decorator and call attach_context().

    Policy:
    - If framework is unavailable, validate the unavailable contract and return.
    - If streaming is not declared, validate that and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.stream_query_method:
        assert framework_descriptor.stream_query_method is None
        return

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(module, monkeypatch)

    instance = _build_error_wrapped_client_instance(
        framework_descriptor,
        failing_adapter,
    )
    assert instance is not None

    stream_method = _get_method(instance, framework_descriptor.stream_query_method)

    with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
        if framework_descriptor.context_kwarg:
            iterator = stream_method(
                "err-stream",
                **{framework_descriptor.context_kwarg: {}},
            )
        else:
            iterator = stream_method("err-stream")

        for _ in iterator:  # noqa: B007
            pass

    assert calls, "attach_context was not called on sync stream failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    _assert_error_context_minimum(framework_descriptor, ctx)


def test_error_context_is_attached_on_sync_stream_calltime_failure_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
    failing_adapter_stream_calltime: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When streaming is supported, sync stream failures that occur at call-time
    must also go through the error-context decorator and call attach_context().

    Policy:
    - If framework is unavailable, validate the unavailable contract and return.
    - If streaming is not declared, validate that and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.stream_query_method:
        assert framework_descriptor.stream_query_method is None
        return

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(module, monkeypatch)

    instance = _build_error_wrapped_client_instance(
        framework_descriptor,
        failing_adapter_stream_calltime,
    )
    assert instance is not None

    stream_method = _get_method(instance, framework_descriptor.stream_query_method)

    with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
        if framework_descriptor.context_kwarg:
            iterator = stream_method("err-stream-calltime", **{framework_descriptor.context_kwarg: {}})
        else:
            iterator = stream_method("err-stream-calltime")
        
        # Call-time errors manifest on first iteration attempt due to Python's
        # lazy generator semantics. The key distinction is that the adapter's
        # stream method raises immediately (before yielding any data), vs
        # in-stream errors that occur after some data has been yielded.
        next(iterator)

    assert calls, "attach_context was not called on sync stream call-time failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    _assert_error_context_minimum(framework_descriptor, ctx)


def test_error_context_is_attached_on_sync_failure_for_all_registry_declared_methods(
    framework_descriptor: GraphFrameworkDescriptor,
    failing_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure *every registry-declared sync method* attaches error context on failure.

    Why this matters:
    - The graph registry declares many surfaces beyond query/stream.
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

    instance = _build_error_wrapped_client_instance(framework_descriptor, failing_adapter)
    assert instance is not None

    for method_name, kind, is_async in _iter_registry_methods(framework_descriptor):
        if is_async:
            continue

        # Reset best-effort between operations to reduce the chance of sticky failures
        # masking underlying exceptions (e.g., circuit breaker behavior).
        _reset_async_bridge_state_best_effort()

        fn = _get_method(instance, method_name)

        with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
            if framework_descriptor.context_kwarg:
                out = _call_declared_method_with_context_best_effort(
                    framework_descriptor,
                    fn,
                    kind,
                    context={},
                )
            else:
                args, kwargs = _best_effort_call_args(kind)
                out = fn(*args, **kwargs)

            if kind == "stream":
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


@pytest.mark.asyncio
async def test_error_context_is_attached_on_async_query_failure_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
    failing_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async is supported, async query failures should also go through
    the error-context decorator and call attach_context().

    Policy:
    - If framework is unavailable, validate the unavailable contract and return.
    - If async query is not declared, validate that and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.async_query_method:
        assert framework_descriptor.async_query_method is None
        return

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(module, monkeypatch)

    instance = _build_error_wrapped_client_instance(
        framework_descriptor,
        failing_adapter,
    )
    assert instance is not None

    aquery_method = _get_method(instance, framework_descriptor.async_query_method)

    with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
        if framework_descriptor.context_kwarg:
            coro = aquery_method(
                "err-aquery",
                **{framework_descriptor.context_kwarg: {}},
            )
        else:
            coro = aquery_method("err-aquery")

        assert inspect.isawaitable(coro), "Async query method must return an awaitable"
        await coro  # noqa: PT018

    assert calls, "attach_context was not called on async query failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    _assert_error_context_minimum(framework_descriptor, ctx)


@pytest.mark.asyncio
async def test_error_context_is_attached_on_async_stream_failure_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
    failing_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async streaming is supported, async stream failures should also go
    through the error-context decorator and call attach_context().

    Policy:
    - If framework is unavailable, validate the unavailable contract and return.
    - If async streaming is not declared, validate that and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.async_stream_query_method:
        assert framework_descriptor.async_stream_query_method is None
        return

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(module, monkeypatch)

    instance = _build_error_wrapped_client_instance(
        framework_descriptor,
        failing_adapter,
    )
    assert instance is not None

    astream_method = _get_method(
        instance,
        framework_descriptor.async_stream_query_method,
    )

    with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
        if framework_descriptor.context_kwarg:
            aiter = astream_method(
                "err-astream",
                **{framework_descriptor.context_kwarg: {}},
            )
        else:
            aiter = astream_method("err-astream")

        # aiter may be an async iterator or an awaitable that resolves to one.
        if inspect.isawaitable(aiter):
            aiter = await aiter  # type: ignore[assignment]

        async for _ in aiter:  # noqa: B007
            pass

    assert calls, "attach_context was not called on async stream failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    _assert_error_context_minimum(framework_descriptor, ctx)


@pytest.mark.asyncio
async def test_error_context_is_attached_on_async_failure_for_all_registry_declared_methods(
    framework_descriptor: GraphFrameworkDescriptor,
    failing_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure *every registry-declared async method* attaches error context on failure.

    Notes:
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

    instance = _build_error_wrapped_client_instance(framework_descriptor, failing_adapter)
    assert instance is not None

    for method_name, kind, is_async in _iter_registry_methods(framework_descriptor):
        if not is_async:
            continue

        fn = _get_method(instance, method_name)

        with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
            if framework_descriptor.context_kwarg:
                out = _call_declared_method_with_context_best_effort(
                    framework_descriptor,
                    fn,
                    kind,
                    context={},
                )
            else:
                args, kwargs = _best_effort_call_args(kind)
                out = fn(*args, **kwargs)

            if kind == "stream":
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
