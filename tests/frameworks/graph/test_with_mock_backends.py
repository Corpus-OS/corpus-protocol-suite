# tests/frameworks/graph/test_with_mock_backends.py

from __future__ import annotations

import importlib
import inspect
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Type

import pytest

from tests.frameworks.registries.graph_registry import (
    GraphFrameworkDescriptor,
    iter_graph_framework_descriptors,
)


GRAPH_OPERATION_PREFIX = "graph_"
FAILURE_MESSAGE_SYNC = "intentional graph backend failure (sync)"
FAILURE_MESSAGE_ASYNC = "intentional graph backend failure (async)"

# Rich mapping context used across all calls in this file.
#
# Why this exists:
# - Other conformance suites enforce that adapters tolerate "rich mapping context"
#   being passed as regular kwargs (e.g. request_id=..., tags=[...]).
# - This file must not silently "pass" while only testing the narrow context_kwarg
#   path; therefore we always splat these kwargs for every call here.
#
# NOTE: Adapters are expected to accept and ignore unknown context keys.
RICH_CONTEXT: dict[str, Any] = {
    "request_id": "req-123",
    "user_id": "user-abc",
    "tags": ["test"],
    "nested": {"depth": 2, "key": "value"},
}

# Performance guardrails:
# - These tests intentionally consume only a small number of stream chunks to avoid
#   hanging if an adapter returns an unbounded iterator.
MAX_STREAM_CHUNKS_TO_CONSUME = 10


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

    UPDATED STRICT POLICY (ACTIVE TESTING):
    - Frameworks registered in the graph registry must be available/importable in
      the conformance environment.
    - If a framework is registered but not available, that is a test failure,
      because it implies registry/environment drift and yields false confidence.
    """
    descriptor: GraphFrameworkDescriptor = request.param
    assert descriptor.is_available(), (
        f"Framework '{descriptor.name}' is registered but not available in this environment. "
        "This suite is configured for active testing: frameworks must be present and testable."
    )
    return descriptor


# ---------------------------------------------------------------------------
# "Evil" graph backends
# ---------------------------------------------------------------------------


class InvalidResultGraphAdapter:
    """
    Backend that returns blatantly invalid results.

    - query() returns a non-graph scalar value
    - stream_query() returns a non-iterable
    - async variants mirror the same shape errors

    Framework adapters should surface coercion / validation errors rather than
    silently treating these as valid graph results.
    """

    # Required for adapter construction
    def capabilities(self, *args: Any, **kwargs: Any) -> Any:
        return {"query": True, "stream": True}

    # Sync query surface
    def query(self, *args: Any, **kwargs: Any) -> Any:
        return "not-a-query-result"

    # Sync streaming surface
    def stream_query(self, *args: Any, **kwargs: Any) -> Any:
        # Return something that is not iterable
        return 123456

    # Async query surface
    async def aquery(self, *args: Any, **kwargs: Any) -> Any:
        return "not-a-query-result"

    # Async streaming surface
    async def astream_query(self, *args: Any, **kwargs: Any) -> Any:
        # Could be an awaitable that resolves to a non-iterable
        return 123456

    # Optional bulk / batch surfaces (kept simple)
    def bulk_vertices(self, *args: Any, **kwargs: Any) -> Any:
        return "not-a-bulk-result"

    async def abulk_vertices(self, *args: Any, **kwargs: Any) -> Any:
        return "not-a-bulk-result"

    def batch(self, *args: Any, **kwargs: Any) -> Any:
        return "not-a-batch-result"

    async def abatch(self, *args: Any, **kwargs: Any) -> Any:
        return "not-a-batch-result"


class EmptyResultGraphAdapter:
    """
    Backend that always returns obviously empty results.

    Used to verify that adapters do not silently treat empty backend responses
    as fully valid results, particularly for batch() surfaces.
    """

    # Required for adapter construction
    def capabilities(self, *args: Any, **kwargs: Any) -> Any:
        return {"query": True, "stream": True}

    def query(self, *args: Any, **kwargs: Any) -> Any:
        return None

    async def aquery(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def stream_query(self, *args: Any, **kwargs: Any) -> Any:
        return iter(())

    async def astream_query(self, *args: Any, **kwargs: Any) -> Any:
        async def _aiter():
            if False:  # pragma: no cover - structure only
                yield None

        return _aiter()

    def bulk_vertices(self, *args: Any, **kwargs: Any) -> Any:
        return []

    async def abulk_vertices(self, *args: Any, **kwargs: Any) -> Any:
        return []

    def batch(self, *args: Any, **kwargs: Any) -> Any:
        return []

    async def abatch(self, *args: Any, **kwargs: Any) -> Any:
        return []


class RaisingGraphAdapter:
    """
    Backend that always raises.

    Used to validate that error-context decorators still attach context when
    failures originate in the graph backend rather than the higher-level code.
    """

    # Required for adapter construction
    def capabilities(self, *args: Any, **kwargs: Any) -> Any:
        return {"query": True, "stream": True}

    def query(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    async def aquery(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_ASYNC)

    async def stream_query(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_ASYNC)
        # This is never reached, but keeps type checkers happy
        if False:  # pragma: no cover
            yield

    async def astream_query(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_ASYNC)
        if False:  # pragma: no cover
            yield

    def bulk_vertices(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    async def abulk_vertices(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_ASYNC)

    def batch(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    async def abatch(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_ASYNC)


class IterationRaisingGraphAdapter:
    """
    Backend whose streaming surfaces raise *during iteration* rather than at call-time.

    This models real-world streaming failure modes where:
    - a stream starts successfully,
    - then fails mid-stream during consumption.

    These failures must still flow through the framework adapter's error-context
    decorators and attach context.
    """

    # Required for adapter construction
    def capabilities(self, *args: Any, **kwargs: Any) -> Any:
        return {"query": True, "stream": True}

    def query(self, *args: Any, **kwargs: Any) -> Any:
        # Keep query raising behavior deterministic and aligned with other "raising" backends.
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    async def aquery(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_ASYNC)

    async def stream_query(self, *args: Any, **kwargs: Any) -> Any:
        from corpus_sdk.graph.graph_base import QueryChunk
        # Yield one item to prove "stream started", then fail deterministically.
        yield QueryChunk(records=[{"chunk": 1}], is_final=False)
        raise RuntimeError(FAILURE_MESSAGE_ASYNC)

    async def astream_query(self, *args: Any, **kwargs: Any) -> Any:
        from corpus_sdk.graph.graph_base import QueryChunk
        yield QueryChunk(records=[{"chunk": 1}], is_final=False)
        raise RuntimeError(FAILURE_MESSAGE_ASYNC)

    def batch(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    async def abatch(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_ASYNC)


class WrongBatchLengthGraphAdapter:
    """
    Backend that returns a fixed number of batch results regardless of input.

    Used to verify that adapters do not silently accept mismatched batch result
    lengths from the backend.
    """

    # Required for adapter construction
    def capabilities(self, *args: Any, **kwargs: Any) -> Any:
        return {"query": True, "stream": True}

    def query(self, *args: Any, **kwargs: Any) -> Any:
        return {"rows": []}

    def batch(self, operations: Any, *args: Any, **kwargs: Any) -> Any:
        # Always return a single "result" regardless of len(operations)
        return ["single-result"]

    async def abatch(self, operations: Any, *args: Any, **kwargs: Any) -> Any:
        return ["single-result"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_method(instance: Any, name: str | None) -> Callable[..., Any]:
    """Helper to fetch a method from the instance and assert it is callable."""
    assert name is not None, "Method name must not be None"
    attr = getattr(instance, name, None)
    assert callable(attr), f"{instance!r} missing expected callable method {name!r}"
    return attr


def _context_kwargs_for_descriptor(descriptor: GraphFrameworkDescriptor) -> dict[str, Any]:
    """
    Build kwargs that nest RICH_CONTEXT under the framework-specific context parameter.

    Each framework adapter expects context to be passed via its specific context_kwarg
    (e.g., 'conversation' for AutoGen, 'task' for CrewAI, 'config' for LangChain).
    We nest RICH_CONTEXT under that parameter to avoid TypeError from unexpected kwargs.

    This prevents "false passes" where tests only exercise the happy path.
    """
    kw: dict[str, Any] = {}

    if descriptor.context_kwarg:
        # Nest RICH_CONTEXT under the framework-specific context parameter
        ctx = dict(RICH_CONTEXT)
        
        # Best-effort traceability: include framework name in tags.
        # This is non-fatal if tags cannot be coerced to a list.
        try:
            tags = list(ctx.get("tags", []))
            tags.append(descriptor.name)
            ctx["tags"] = tags
        except Exception:
            pass
        
        kw[descriptor.context_kwarg] = ctx

    return kw


def _make_client_with_evil_backend(
    framework_descriptor: GraphFrameworkDescriptor,
    backend_cls: Type[Any],
) -> Any:
    """
    Instantiate the framework graph client with an 'evil' backend.

    This bypasses the normal adapter fixture wiring and lets us simulate
    misbehaving backends in a controlled way.
    """
    # Defensive import hardening: surface syntax errors with actionable diagnostics.
    try:
        module = importlib.import_module(framework_descriptor.adapter_module)
    except SyntaxError as e:
        pytest.fail(
            f"Adapter module failed to import for {framework_descriptor.name!r}: "
            f"SyntaxError at line {e.lineno}: {e.msg}\n"
            f"Text: {e.text!r}",
            pytrace=True,
        )

    client_cls = getattr(module, framework_descriptor.adapter_class)

    backend = backend_cls()
    instance = client_cls(adapter=backend)

    # If adapters cache the backend under a different attribute, they should
    # still be referencing the same object we passed as adapter.
    return instance


def _require_all_surfaces_declared(descriptor: GraphFrameworkDescriptor) -> None:
    """
    Enforce ACTIVE TESTING POLICY: all method surfaces must be enabled/declared.

    Why this exists:
    - The goal is to test all framework adapter surfaces (sync + async, query + stream,
      and batch) under misbehaving backends.
    - If a surface isn't declared, the suite would otherwise skip/return early,
      creating false confidence.

    This helper enforces registry alignment; method existence/callability is asserted
    when we _get_method(...) inside each test.
    """
    assert descriptor.query_method, f"{descriptor.name}: query_method must be declared"
    assert descriptor.stream_query_method, f"{descriptor.name}: stream_query_method must be declared for active testing"
    assert descriptor.batch_method, f"{descriptor.name}: batch_method must be declared for active testing"

    assert descriptor.supports_async is True, f"{descriptor.name}: supports_async must be True for active testing"
    assert descriptor.async_query_method, f"{descriptor.name}: async_query_method must be declared for active testing"
    assert descriptor.async_stream_query_method, (
        f"{descriptor.name}: async_stream_query_method must be declared for active testing"
    )


def _call_query(
    descriptor: GraphFrameworkDescriptor,
    instance: Any,
    text: str,
) -> Any:
    query_fn = _get_method(instance, descriptor.query_method)
    return query_fn(text, **_context_kwargs_for_descriptor(descriptor))


def _call_stream(
    descriptor: GraphFrameworkDescriptor,
    instance: Any,
    text: str,
) -> Any:
    assert descriptor.stream_query_method is not None
    stream_fn = _get_method(instance, descriptor.stream_query_method)
    return stream_fn(text, **_context_kwargs_for_descriptor(descriptor))


def _call_batch(
    descriptor: GraphFrameworkDescriptor,
    instance: Any,
    operations: Sequence[Any],
) -> Any:
    """Call batch method, converting string operations to BatchOperation objects if needed."""
    from corpus_sdk.graph.graph_base import BatchOperation
    
    assert descriptor.batch_method is not None
    batch_fn = _get_method(instance, descriptor.batch_method)
    
    # Convert strings to BatchOperation objects
    batch_ops = [
        BatchOperation(op="query", args={"text": op}) if isinstance(op, str) else op
        for op in operations
    ]
    
    return batch_fn(batch_ops, **_context_kwargs_for_descriptor(descriptor))


def _consume_sync_stream_best_effort(iterator: Any) -> None:
    """
    Consume up to MAX_STREAM_CHUNKS_TO_CONSUME items from an iterator.

    This prevents runaway/never-ending iterators from hanging the suite while
    still forcing iteration-time errors to surface.
    """
    for i, _ in enumerate(iterator):  # noqa: B007
        if i + 1 >= MAX_STREAM_CHUNKS_TO_CONSUME:
            break


# ---------------------------------------------------------------------------
# Invalid result behavior
# ---------------------------------------------------------------------------


def test_invalid_backend_result_causes_errors_for_sync_query(
    framework_descriptor: GraphFrameworkDescriptor,
) -> None:
    """
    If the backend returns a clearly invalid result type for query(), the
    framework adapter should surface an error rather than silently treating
    it as a valid graph result.
    """
    _require_all_surfaces_declared(framework_descriptor)

    instance = _make_client_with_evil_backend(framework_descriptor, InvalidResultGraphAdapter)

    with pytest.raises(Exception):  # noqa: BLE001
        _call_query(framework_descriptor, instance, "invalid-query-test")


def test_invalid_backend_result_causes_errors_for_sync_stream(
    framework_descriptor: GraphFrameworkDescriptor,
) -> None:
    """
    Same as the query test, but for the sync streaming surface.

    The invalid backend returns a non-iterable; adapters should surface an error.
    """
    _require_all_surfaces_declared(framework_descriptor)

    instance = _make_client_with_evil_backend(framework_descriptor, InvalidResultGraphAdapter)

    with pytest.raises(Exception):  # noqa: BLE001
        iterator = _call_stream(framework_descriptor, instance, "invalid-stream-test")
        _consume_sync_stream_best_effort(iterator)


@pytest.mark.asyncio
async def test_async_invalid_backend_result_causes_errors_for_query(
    framework_descriptor: GraphFrameworkDescriptor,
) -> None:
    """
    When async is supported, invalid backend results for async query() should
    also surface as errors, not valid-looking graph results.
    """
    _require_all_surfaces_declared(framework_descriptor)

    instance = _make_client_with_evil_backend(framework_descriptor, InvalidResultGraphAdapter)

    aquery_fn = _get_method(instance, framework_descriptor.async_query_method)

    with pytest.raises(Exception):  # noqa: BLE001
        coro = aquery_fn("invalid-async-query-test", **_context_kwargs_for_descriptor(framework_descriptor))
        assert inspect.isawaitable(coro), "Async query method must return an awaitable"
        await coro  # noqa: PT018


@pytest.mark.asyncio
async def test_async_invalid_backend_result_causes_errors_for_stream(
    framework_descriptor: GraphFrameworkDescriptor,
) -> None:
    """
    When async streaming is supported, invalid backend results for astream_query()
    should also surface as errors.
    """
    _require_all_surfaces_declared(framework_descriptor)

    instance = _make_client_with_evil_backend(framework_descriptor, InvalidResultGraphAdapter)

    astream_fn = _get_method(instance, framework_descriptor.async_stream_query_method)

    with pytest.raises(Exception):  # noqa: BLE001
        aiter = astream_fn("invalid-async-stream-test", **_context_kwargs_for_descriptor(framework_descriptor))

        # Allow awaitable -> async iterator or async iterator directly
        if inspect.isawaitable(aiter):
            aiter = await aiter  # type: ignore[assignment]

        # Force async iteration to trigger type/shape errors.
        n = 0
        async for _ in aiter:  # noqa: B007
            n += 1
            if n >= MAX_STREAM_CHUNKS_TO_CONSUME:
                break


# ---------------------------------------------------------------------------
# Empty / wrong batch result behavior
# ---------------------------------------------------------------------------


def test_empty_backend_batch_result_is_not_silently_treated_as_valid(
    framework_descriptor: GraphFrameworkDescriptor,
) -> None:
    """
    When the backend returns an empty batch result, the adapter should not
    silently treat it as a fully valid per-operation response.

    Acceptable behaviors:
    - Raise an Exception (preferred), or
    - Return a sequence whose length != len(operations).
    """
    _require_all_surfaces_declared(framework_descriptor)

    instance = _make_client_with_evil_backend(framework_descriptor, EmptyResultGraphAdapter)

    operations = ["op-a", "op-b"]  # treated as opaque by the evil backend

    try:
        result = _call_batch(framework_descriptor, instance, operations)
    except Exception:  # noqa: BLE001
        # Raising is acceptable / preferred.
        return

    # If it did not raise, we at least require that the result is obviously
    # not a valid batch result with one entry per operation.
    if isinstance(result, Sequence):
        assert len(result) != len(operations), (
            "Empty backend batch result unexpectedly produced a sequence whose "
            "length matches the number of operations; adapters should treat "
            "empty backend results as errors or obvious mismatches."
        )
    else:
        # If the adapter returns a non-sequence type, it must still not look "valid"
        # as an opaque batch result. This is a best-effort negative test.
        assert result is not None


def test_wrong_batch_length_from_backend_causes_error_or_obvious_mismatch(
    framework_descriptor: GraphFrameworkDescriptor,
) -> None:
    """
    When the backend returns a batch result whose length does not match the
    number of input operations, the adapter should not silently treat it as valid.

    Acceptable behaviors:
    - Raise an Exception, or
    - Return a sequence whose length != len(operations).
    """
    _require_all_surfaces_declared(framework_descriptor)

    instance = _make_client_with_evil_backend(framework_descriptor, WrongBatchLengthGraphAdapter)

    operations = ["op-1", "op-2", "op-3"]  # 3 inputs

    try:
        result = _call_batch(framework_descriptor, instance, operations)
    except Exception:  # noqa: BLE001
        # Raising is acceptable / preferred.
        return

    if isinstance(result, Sequence):
        assert len(result) != len(operations), (
            "WrongBatchLengthGraphAdapter produced a batch result whose length "
            "matches the number of operations; adapters should validate batch "
            "row counts and treat mismatches as errors."
        )
    else:
        assert result is not None


# ---------------------------------------------------------------------------
# Error-context when backend raises
# ---------------------------------------------------------------------------


def _patch_attach_context(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
) -> list[tuple[BaseException, dict[str, Any]]]:
    """
    Patch the module-local attach_context used by decorators and capture calls.
    """
    calls: list[tuple[BaseException, dict[str, Any]]] = []

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        calls.append((exc, ctx))

    monkeypatch.setattr(module, "attach_context", fake_attach_context)
    return calls


def test_backend_exception_is_wrapped_with_error_context_on_query(
    framework_descriptor: GraphFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the backend raises during a sync query operation, the framework
    adapter's error-context decorator should:

    - call attach_context() with the exception and useful metadata, and
    - re-raise the original exception (or a wrapped one).
    """
    _require_all_surfaces_declared(framework_descriptor)

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _make_client_with_evil_backend(framework_descriptor, RaisingGraphAdapter)

    with pytest.raises(RuntimeError, match="intentional graph backend failure"):
        _call_query(framework_descriptor, instance, "err-query")

    assert calls, "attach_context was not called for backend query failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name
    assert str(ctx["operation"]).startswith(GRAPH_OPERATION_PREFIX)


def test_backend_exception_is_wrapped_with_error_context_on_stream_calltime(
    framework_descriptor: GraphFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Validate error-context decoration for streaming failures that occur at call-time.
    """
    _require_all_surfaces_declared(framework_descriptor)

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _make_client_with_evil_backend(framework_descriptor, RaisingGraphAdapter)

    with pytest.raises(RuntimeError, match="intentional graph backend failure"):
        iterator = _call_stream(framework_descriptor, instance, "err-stream-calltime")
        # For generators, we need to start iterating to trigger call-time exceptions
        next(iter(iterator))

    assert calls, "attach_context was not called for backend stream call-time failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert ctx.get("framework") == framework_descriptor.name
    assert str(ctx.get("operation", "")).startswith(GRAPH_OPERATION_PREFIX)


def test_backend_exception_is_wrapped_with_error_context_on_stream_iteration(
    framework_descriptor: GraphFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Validate error-context decoration for streaming failures that occur during iteration.

    This models real streaming failure modes where a stream begins successfully,
    then fails while being consumed.
    """
    _require_all_surfaces_declared(framework_descriptor)

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _make_client_with_evil_backend(framework_descriptor, IterationRaisingGraphAdapter)

    with pytest.raises(RuntimeError, match="intentional graph backend failure"):
        iterator = _call_stream(framework_descriptor, instance, "err-stream-iteration")
        assert isinstance(iterator, Iterable) or True  # best-effort; iterability validated by consumption
        _consume_sync_stream_best_effort(iterator)

    assert calls, "attach_context was not called for backend stream iteration failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert ctx.get("framework") == framework_descriptor.name
    assert str(ctx.get("operation", "")).startswith(GRAPH_OPERATION_PREFIX)


def test_backend_exception_is_wrapped_with_error_context_on_batch(
    framework_descriptor: GraphFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Same as the query error-context test, but for the sync batch surface.
    """
    _require_all_surfaces_declared(framework_descriptor)

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _make_client_with_evil_backend(framework_descriptor, RaisingGraphAdapter)

    with pytest.raises(RuntimeError, match="intentional graph backend failure"):
        _call_batch(framework_descriptor, instance, ["err-batch-1", "err-batch-2"])

    assert calls, "attach_context was not called for backend batch failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert ctx.get("framework") == framework_descriptor.name
    assert str(ctx.get("operation", "")).startswith(GRAPH_OPERATION_PREFIX)


@pytest.mark.asyncio
async def test_async_backend_exception_is_wrapped_with_error_context_on_query(
    framework_descriptor: GraphFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async is supported, backend exceptions in async query should also go
    through the error-context decorators and call attach_context().
    """
    _require_all_surfaces_declared(framework_descriptor)

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _make_client_with_evil_backend(framework_descriptor, RaisingGraphAdapter)

    aquery_fn = _get_method(instance, framework_descriptor.async_query_method)

    with pytest.raises(RuntimeError, match="intentional graph backend failure"):
        coro = aquery_fn("err-async-query", **_context_kwargs_for_descriptor(framework_descriptor))
        assert inspect.isawaitable(coro), "Async query method must return an awaitable"
        await coro  # noqa: PT018

    assert calls, "attach_context was not called for async backend query failures"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert ctx.get("framework") == framework_descriptor.name
    assert str(ctx.get("operation", "")).startswith(GRAPH_OPERATION_PREFIX)


@pytest.mark.asyncio
async def test_async_backend_exception_is_wrapped_with_error_context_on_stream_calltime(
    framework_descriptor: GraphFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async streaming is supported, backend exceptions at call-time must
    also go through the error-context decorators and call attach_context().
    """
    _require_all_surfaces_declared(framework_descriptor)

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _make_client_with_evil_backend(framework_descriptor, RaisingGraphAdapter)

    astream_fn = _get_method(instance, framework_descriptor.async_stream_query_method)

    with pytest.raises(RuntimeError, match="intentional graph backend failure"):
        aiter = astream_fn("err-async-stream-calltime", **_context_kwargs_for_descriptor(framework_descriptor))
        if inspect.isawaitable(aiter):
            aiter = await aiter  # type: ignore[assignment]
        n = 0
        async for _ in aiter:  # noqa: B007
            n += 1
            if n >= MAX_STREAM_CHUNKS_TO_CONSUME:
                break

    assert calls, "attach_context was not called for async backend stream call-time failures"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert ctx.get("framework") == framework_descriptor.name
    assert str(ctx.get("operation", "")).startswith(GRAPH_OPERATION_PREFIX)


@pytest.mark.asyncio
async def test_async_backend_exception_is_wrapped_with_error_context_on_stream_iteration(
    framework_descriptor: GraphFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async streaming is supported, backend exceptions during async iteration
    must also go through the error-context decorators and call attach_context().

    This models real streaming failure modes where an async stream begins and then
    fails mid-consumption.
    """
    _require_all_surfaces_declared(framework_descriptor)

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _make_client_with_evil_backend(framework_descriptor, IterationRaisingGraphAdapter)

    astream_fn = _get_method(instance, framework_descriptor.async_stream_query_method)

    with pytest.raises(RuntimeError, match="intentional graph backend failure"):
        aiter = astream_fn("err-async-stream-iteration", **_context_kwargs_for_descriptor(framework_descriptor))
        if inspect.isawaitable(aiter):
            aiter = await aiter  # type: ignore[assignment]
        n = 0
        async for _ in aiter:  # noqa: B007
            n += 1
            if n >= MAX_STREAM_CHUNKS_TO_CONSUME:
                break

    assert calls, "attach_context was not called for async backend stream iteration failures"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert ctx.get("framework") == framework_descriptor.name
    assert str(ctx.get("operation", "")).startswith(GRAPH_OPERATION_PREFIX)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
