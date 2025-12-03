# tests/frameworks/graph/test_with_mock_backends.py

from __future__ import annotations

import importlib
import inspect
from collections.abc import Sequence
from typing import Any, Callable, Type

import pytest

from tests.frameworks.registries.graph_registry import (
    GraphFrameworkDescriptor,
    iter_graph_framework_descriptors,
)


GRAPH_OPERATION_PREFIX = "graph_"
FAILURE_MESSAGE_SYNC = "intentional graph backend failure (sync)"
FAILURE_MESSAGE_ASYNC = "intentional graph backend failure (async)"


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

    Frameworks that are not actually available in the environment (e.g. the
    underlying LangChain / LlamaIndex / Semantic Kernel libraries are missing)
    are skipped via descriptor.is_available().
    """
    descriptor: GraphFrameworkDescriptor = request.param
    if not descriptor.is_available():
        pytest.skip(f"Framework '{descriptor.name}' not available in this environment")
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

    def query(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    async def aquery(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_ASYNC)

    def stream_query(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    async def astream_query(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_ASYNC)

    def bulk_vertices(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    async def abulk_vertices(self, *args: Any, **kwargs: Any) -> Any:
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


def _make_client_with_evil_backend(
    framework_descriptor: GraphFrameworkDescriptor,
    backend_cls: Type[Any],
) -> Any:
    """
    Instantiate the framework graph client with an 'evil' backend.

    This bypasses the normal graph_adapter fixture wiring and lets us simulate
    misbehaving backends in a controlled way.
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    client_cls = getattr(module, framework_descriptor.adapter_class)

    backend = backend_cls()
    instance = client_cls(graph_adapter=backend)

    # If adapters cache the backend under a different attribute, they should
    # still be referencing the same object we passed as graph_adapter.
    return instance


def _call_query(
    descriptor: GraphFrameworkDescriptor,
    instance: Any,
    text: str,
) -> Any:
    query_fn = _get_method(instance, descriptor.query_method)
    if descriptor.context_kwarg:
        return query_fn(text, **{descriptor.context_kwarg: {}})
    return query_fn(text)


def _call_stream(
    descriptor: GraphFrameworkDescriptor,
    instance: Any,
    text: str,
) -> Any:
    assert descriptor.stream_query_method is not None
    stream_fn = _get_method(instance, descriptor.stream_query_method)
    if descriptor.context_kwarg:
        return stream_fn(text, **{descriptor.context_kwarg: {}})
    return stream_fn(text)


def _call_batch(
    descriptor: GraphFrameworkDescriptor,
    instance: Any,
    operations: Sequence[Any],
) -> Any:
    assert descriptor.batch_method is not None
    batch_fn = _get_method(instance, descriptor.batch_method)
    if descriptor.context_kwarg:
        return batch_fn(operations, **{descriptor.context_kwarg: {}})
    return batch_fn(operations)


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
    instance = _make_client_with_evil_backend(
        framework_descriptor,
        InvalidResultGraphAdapter,
    )

    with pytest.raises(Exception):  # noqa: BLE001
        _call_query(framework_descriptor, instance, "invalid-query-test")


def test_invalid_backend_result_causes_errors_for_sync_stream_when_declared(
    framework_descriptor: GraphFrameworkDescriptor,
) -> None:
    """
    Same as the query test, but for the sync streaming surface when declared.
    """
    if not framework_descriptor.stream_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare sync streaming",
        )

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        InvalidResultGraphAdapter,
    )

    with pytest.raises(Exception):  # noqa: BLE001
        iterator = _call_stream(
            framework_descriptor,
            instance,
            "invalid-stream-test",
        )

        # Force iteration to trigger type/shape errors
        for _ in iterator:  # noqa: B007
            pass


@pytest.mark.asyncio
async def test_async_invalid_backend_result_causes_errors_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
) -> None:
    """
    When async is supported, invalid backend results for async query() should
    also surface as errors, not valid-looking graph results.
    """
    if not framework_descriptor.async_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async query",
        )

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        InvalidResultGraphAdapter,
    )

    aquery_fn = _get_method(
        instance,
        framework_descriptor.async_query_method,
    )

    with pytest.raises(Exception):  # noqa: BLE001
        if framework_descriptor.context_kwarg:
            coro = aquery_fn(
                "invalid-async-query-test",
                **{framework_descriptor.context_kwarg: {}},
            )
        else:
            coro = aquery_fn("invalid-async-query-test")

        assert inspect.isawaitable(coro), "Async query method must return an awaitable"
        await coro  # noqa: PT018


@pytest.mark.asyncio
async def test_async_invalid_backend_result_causes_errors_for_stream_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
) -> None:
    """
    When async streaming is supported, invalid backend results for
    astream_query() should also surface as errors.
    """
    if not framework_descriptor.async_stream_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async streaming",
        )

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        InvalidResultGraphAdapter,
    )

    astream_fn = _get_method(
        instance,
        framework_descriptor.async_stream_query_method,
    )

    with pytest.raises(Exception):  # noqa: BLE001
        if framework_descriptor.context_kwarg:
            aiter = astream_fn(
                "invalid-async-stream-test",
                **{framework_descriptor.context_kwarg: {}},
            )
        else:
            aiter = astream_fn("invalid-async-stream-test")

        # Allow awaitable -> async iterator or async iterator directly
        if inspect.isawaitable(aiter):
            aiter = await aiter  # type: ignore[assignment]

        async for _ in aiter:  # noqa: B007
            pass


# ---------------------------------------------------------------------------
# Empty batch result behavior
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
    if not framework_descriptor.batch_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare batch support",
        )

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        EmptyResultGraphAdapter,
    )

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


def test_wrong_batch_length_from_backend_causes_error_or_obvious_mismatch(
    framework_descriptor: GraphFrameworkDescriptor,
) -> None:
    """
    When the backend returns a batch result whose length does not match the
    number of input operations, the adapter should not silently treat it as
    valid.

    Acceptable behaviors:
    - Raise an Exception, or
    - Return a sequence whose length != len(operations).
    """
    if not framework_descriptor.batch_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare batch support",
        )

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        WrongBatchLengthGraphAdapter,
    )

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
    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        RaisingGraphAdapter,
    )

    with pytest.raises(RuntimeError, match="backend failure"):
        _call_query(framework_descriptor, instance, "err-query")

    assert calls, "attach_context was not called for backend query failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name
    assert str(ctx["operation"]).startswith(GRAPH_OPERATION_PREFIX)


def test_backend_exception_is_wrapped_with_error_context_on_batch_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Same as the query error-context test, but for the sync batch surface
    when declared.
    """
    if not framework_descriptor.batch_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare batch support",
        )

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        RaisingGraphAdapter,
    )

    with pytest.raises(RuntimeError, match="backend failure"):
        _call_batch(
            framework_descriptor,
            instance,
            ["err-batch-1", "err-batch-2"],
        )

    assert calls, "attach_context was not called for backend batch failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name
    assert str(ctx["operation"]).startswith(GRAPH_OPERATION_PREFIX)


@pytest.mark.asyncio
async def test_async_backend_exception_is_wrapped_with_error_context_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async is supported, backend exceptions in async query should also go
    through the error-context decorators and call attach_context().
    """
    if not framework_descriptor.async_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async query",
        )

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        RaisingGraphAdapter,
    )

    aquery_fn = _get_method(
        instance,
        framework_descriptor.async_query_method,
    )

    with pytest.raises(RuntimeError, match="backend failure"):
        if framework_descriptor.context_kwarg:
            coro = aquery_fn(
                "err-async-query",
                **{framework_descriptor.context_kwarg: {}},
            )
        else:
            coro = aquery_fn("err-async-query")

        assert inspect.isawaitable(coro), "Async query method must return an awaitable"
        await coro  # noqa: PT018

    assert calls, "attach_context was not called for async backend failures"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name
    assert str(ctx["operation"]).startswith(GRAPH_OPERATION_PREFIX)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
