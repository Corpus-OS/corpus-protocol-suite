# tests/frameworks/vector/test_with_mock_backends.py

from __future__ import annotations

import importlib
import inspect
from collections.abc import Sequence
from typing import Any, Callable, Type

import pytest

from tests.frameworks.registries.vector_registry import (
    VectorFrameworkDescriptor,
    iter_vector_framework_descriptors,
)

VECTOR_OPERATION_PREFIX = "vector_"
FAILURE_MESSAGE_SYNC = "intentional vector backend failure (sync)"
FAILURE_MESSAGE_ASYNC = "intentional vector backend failure (async)"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=list(iter_vector_framework_descriptors()),
    name="framework_descriptor",
)
def framework_descriptor_fixture(
    request: pytest.FixtureRequest,
) -> VectorFrameworkDescriptor:
    """
    Parameterized over all registered vector framework descriptors.

    Frameworks that are not actually available in the environment (e.g. the
    underlying LlamaIndex / Semantic Kernel libraries are missing)
    are skipped via descriptor.is_available().
    """
    descriptor: VectorFrameworkDescriptor = request.param
    if not descriptor.is_available():
        pytest.skip(f"Framework '{descriptor.name}' not available in this environment")
    return descriptor


# ---------------------------------------------------------------------------
# "Evil" vector backends
# ---------------------------------------------------------------------------


class InvalidResultVectorAdapter:
    """
    Backend that returns blatantly invalid results.

    - query() / query_mmr() return non-vector scalar values
    - query_stream() / astream_query() return non-iterables
    - async variants mirror the same shape errors

    Framework adapters should surface coercion / validation errors rather
    than silently treating these as valid vector results.
    """

    # Sync add / delete surfaces (kept simple; not directly asserted here)
    def add(self, *args: Any, **kwargs: Any) -> Any:
        return "not-an-add-result"

    def delete(self, *args: Any, **kwargs: Any) -> Any:
        return "not-a-delete-result"

    # Sync query surfaces
    def query(self, *args: Any, **kwargs: Any) -> Any:
        return "not-a-query-result"

    def query_stream(self, *args: Any, **kwargs: Any) -> Any:
        # Return something that is not iterable
        return 123456

    def query_mmr(self, *args: Any, **kwargs: Any) -> Any:
        return "not-an-mmr-result"

    # Async variants
    async def aadd(self, *args: Any, **kwargs: Any) -> Any:
        return "not-an-add-result"

    async def adelete(self, *args: Any, **kwargs: Any) -> Any:
        return "not-a-delete-result"

    async def aquery(self, *args: Any, **kwargs: Any) -> Any:
        return "not-a-query-result"

    async def astream_query(self, *args: Any, **kwargs: Any) -> Any:
        # Could be an awaitable that resolves to a non-iterable
        return 123456

    async def aquery_mmr(self, *args: Any, **kwargs: Any) -> Any:
        return "not-an-mmr-result"


class EmptyResultVectorAdapter:
    """
    Backend that always returns obviously empty results.

    Used to verify that adapters do not crash when backends return degenerate
    responses for query / stream / MMR surfaces.
    """

    # Sync surfaces
    def add(self, *args: Any, **kwargs: Any) -> Any:
        return []

    def delete(self, *args: Any, **kwargs: Any) -> Any:
        return []

    def query(self, *args: Any, **kwargs: Any) -> Any:
        return []

    def query_stream(self, *args: Any, **kwargs: Any) -> Any:
        return iter(())

    def query_mmr(self, *args: Any, **kwargs: Any) -> Any:
        return []

    # Async surfaces
    async def aadd(self, *args: Any, **kwargs: Any) -> Any:
        return []

    async def adelete(self, *args: Any, **kwargs: Any) -> Any:
        return []

    async def aquery(self, *args: Any, **kwargs: Any) -> Any:
        return []

    async def astream_query(self, *args: Any, **kwargs: Any) -> Any:
        async def _aiter():
            if False:  # pragma: no cover - structure only
                yield None

        return _aiter()

    async def aquery_mmr(self, *args: Any, **kwargs: Any) -> Any:
        return []


class RaisingVectorAdapter:
    """
    Backend that always raises.

    Used to validate that error-context decorators still attach context when
    failures originate in the vector backend rather than the higher-level code.
    """

    # Sync surfaces
    def add(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    def delete(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    def query(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    def query_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    def query_mmr(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    # Async surfaces
    async def aadd(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_ASYNC)

    async def adelete(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_ASYNC)

    async def aquery(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_ASYNC)

    async def astream_query(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_ASYNC)

    async def aquery_mmr(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_ASYNC)


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
    framework_descriptor: VectorFrameworkDescriptor,
    backend_cls: Type[Any],
) -> Any:
    """
    Instantiate the framework vector client with an 'evil' backend.

    This bypasses the normal adapter fixture wiring and lets us simulate
    misbehaving backends in a controlled way.
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    client_cls = getattr(module, framework_descriptor.adapter_class)

    backend = backend_cls()
    instance = client_cls(adapter=backend)

    # If adapters cache the backend under a different attribute, they should
    # still be referencing the same object we passed as `adapter`.
    return instance


def _call_query(
    descriptor: VectorFrameworkDescriptor,
    instance: Any,
    text: str,
) -> Any:
    query_fn = _get_method(instance, descriptor.query_method)
    if descriptor.context_kwarg:
        return query_fn(text, **{descriptor.context_kwarg: {}})
    return query_fn(text)


def _call_stream(
    descriptor: VectorFrameworkDescriptor,
    instance: Any,
    text: str,
) -> Any:
    assert descriptor.stream_query_method is not None
    stream_fn = _get_method(instance, descriptor.stream_query_method)
    if descriptor.context_kwarg:
        return stream_fn(text, **{descriptor.context_kwarg: {}})
    return stream_fn(text)


def _call_mmr(
    descriptor: VectorFrameworkDescriptor,
    instance: Any,
    text: str,
) -> Any:
    assert descriptor.mmr_query_method is not None
    mmr_fn = _get_method(instance, descriptor.mmr_query_method)
    if descriptor.context_kwarg:
        return mmr_fn(text, **{descriptor.context_kwarg: {}})
    return mmr_fn(text)


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


# ---------------------------------------------------------------------------
# Invalid result behavior (sync + async)
# ---------------------------------------------------------------------------


def test_invalid_backend_result_causes_errors_for_sync_query(
    framework_descriptor: VectorFrameworkDescriptor,
) -> None:
    """
    If the backend returns a clearly invalid result type for query(), the
    framework adapter should surface an error rather than silently treating
    it as a valid vector result.
    """
    instance = _make_client_with_evil_backend(
        framework_descriptor,
        InvalidResultVectorAdapter,
    )

    with pytest.raises(Exception):  # noqa: BLE001
        _call_query(framework_descriptor, instance, "invalid-query-test")


def test_invalid_backend_result_causes_errors_for_sync_stream_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
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
        InvalidResultVectorAdapter,
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


def test_invalid_backend_result_causes_errors_for_sync_mmr_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
) -> None:
    """
    When MMR is declared, invalid backend results for query_mmr() should also
    surface as errors, not be treated as valid-looking MMR results.
    """
    if not framework_descriptor.mmr_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare MMR support",
        )

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        InvalidResultVectorAdapter,
    )

    with pytest.raises(Exception):  # noqa: BLE001
        _call_mmr(framework_descriptor, instance, "invalid-mmr-test")


@pytest.mark.asyncio
async def test_async_invalid_backend_result_causes_errors_for_query_when_supported(
    framework_descriptor: VectorFrameworkDescriptor,
) -> None:
    """
    When async query is supported, invalid backend results for async query()
    should also surface as errors.
    """
    if not framework_descriptor.async_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async query",
        )

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        InvalidResultVectorAdapter,
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
    framework_descriptor: VectorFrameworkDescriptor,
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
        InvalidResultVectorAdapter,
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


@pytest.mark.asyncio
async def test_async_invalid_backend_result_causes_errors_for_mmr_when_supported(
    framework_descriptor: VectorFrameworkDescriptor,
) -> None:
    """
    When async MMR is supported, invalid backend results for async query_mmr()
    should also surface as errors.
    """
    if not framework_descriptor.async_mmr_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async MMR support",
        )

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        InvalidResultVectorAdapter,
    )

    ammr_fn = _get_method(
        instance,
        framework_descriptor.async_mmr_query_method,
    )

    with pytest.raises(Exception):  # noqa: BLE001
        if framework_descriptor.context_kwarg:
            coro = ammr_fn(
                "invalid-async-mmr-test",
                **{framework_descriptor.context_kwarg: {}},
            )
        else:
            coro = ammr_fn("invalid-async-mmr-test")

        assert inspect.isawaitable(coro), "Async MMR method must return an awaitable"
        await coro  # noqa: PT018


# ---------------------------------------------------------------------------
# Empty backend behavior (soft expectations)
# ---------------------------------------------------------------------------


def test_empty_backend_query_does_not_crash(
    framework_descriptor: VectorFrameworkDescriptor,
) -> None:
    """
    When the backend returns an obviously empty result for query(), the adapter
    should not crash. Empty results may be valid in vector space, so this test
    only asserts that the call completes without raising.
    """
    instance = _make_client_with_evil_backend(
        framework_descriptor,
        EmptyResultVectorAdapter,
    )

    _call_query(framework_descriptor, instance, "empty-query-test")
    # If we reach here, the test passes; no further shape assertion necessary.


def test_empty_backend_stream_does_not_crash_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
) -> None:
    """
    When streaming is declared, an empty backend stream should not cause errors.
    """
    if not framework_descriptor.stream_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare sync streaming",
        )

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        EmptyResultVectorAdapter,
    )

    iterator = _call_stream(
        framework_descriptor,
        instance,
        "empty-stream-test",
    )

    # Iterating over an empty stream should be fine.
    for _ in iterator:  # noqa: B007
        pass


def test_empty_backend_mmr_does_not_crash_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
) -> None:
    """
    When MMR is declared, an empty backend MMR result should not cause errors.
    """
    if not framework_descriptor.mmr_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare MMR support",
        )

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        EmptyResultVectorAdapter,
    )

    _call_mmr(framework_descriptor, instance, "empty-mmr-test")
    # Again, just asserting that the call completes successfully.


# ---------------------------------------------------------------------------
# Error-context when backend raises
# ---------------------------------------------------------------------------


def test_backend_exception_is_wrapped_with_error_context_on_query(
    framework_descriptor: VectorFrameworkDescriptor,
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
        RaisingVectorAdapter,
    )

    with pytest.raises(RuntimeError, match="backend failure"):
        _call_query(framework_descriptor, instance, "err-query")

    assert calls, "attach_context was not called for backend query failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name
    assert str(ctx["operation"]).startswith(VECTOR_OPERATION_PREFIX)


def test_backend_exception_is_wrapped_with_error_context_on_stream_when_supported(
    framework_descriptor: VectorFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Same as the query error-context test, but for the sync streaming surface
    when declared.
    """
    if not framework_descriptor.stream_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare sync streaming",
        )

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        RaisingVectorAdapter,
    )

    with pytest.raises(RuntimeError, match="backend failure"):
        iterator = _call_stream(
            framework_descriptor,
            instance,
            "err-stream",
        )

        for _ in iterator:  # noqa: B007
            pass

    assert calls, "attach_context was not called for backend stream failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name
    assert str(ctx["operation"]).startswith(VECTOR_OPERATION_PREFIX)


def test_backend_exception_is_wrapped_with_error_context_on_mmr_when_supported(
    framework_descriptor: VectorFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Same as the query error-context test, but for the sync MMR surface
    when declared.
    """
    if not framework_descriptor.mmr_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare MMR support",
        )

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        RaisingVectorAdapter,
    )

    with pytest.raises(RuntimeError, match="backend failure"):
        _call_mmr(
            framework_descriptor,
            instance,
            "err-mmr",
        )

    assert calls, "attach_context was not called for backend MMR failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name
    assert str(ctx["operation"]).startswith(VECTOR_OPERATION_PREFIX)


@pytest.mark.asyncio
async def test_async_backend_exception_is_wrapped_with_error_context_on_query_when_supported(
    framework_descriptor: VectorFrameworkDescriptor,
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
        RaisingVectorAdapter,
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

    assert calls, "attach_context was not called for async backend query failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name
    assert str(ctx["operation"]).startswith(VECTOR_OPERATION_PREFIX)


@pytest.mark.asyncio
async def test_async_backend_exception_is_wrapped_with_error_context_on_stream_when_supported(
    framework_descriptor: VectorFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async streaming is supported, backend exceptions in async stream
    should also go through the error-context decorators and call attach_context().
    """
    if not framework_descriptor.async_stream_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async streaming",
        )

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        RaisingVectorAdapter,
    )

    astream_fn = _get_method(
        instance,
        framework_descriptor.async_stream_query_method,
    )

    with pytest.raises(RuntimeError, match="backend failure"):
        if framework_descriptor.context_kwarg:
            aiter = astream_fn(
                "err-async-stream",
                **{framework_descriptor.context_kwarg: {}},
            )
        else:
            aiter = astream_fn("err-async-stream")

        if inspect.isawaitable(aiter):
            aiter = await aiter  # type: ignore[assignment]

        async for _ in aiter:  # noqa: B007
            pass

    assert calls, "attach_context was not called for async backend stream failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name
    assert str(ctx["operation"]).startswith(VECTOR_OPERATION_PREFIX)


@pytest.mark.asyncio
async def test_async_backend_exception_is_wrapped_with_error_context_on_mmr_when_supported(
    framework_descriptor: VectorFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async MMR is supported, backend exceptions in async MMR should also
    go through the error-context decorators and call attach_context().
    """
    if not framework_descriptor.async_mmr_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async MMR support",
        )

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        RaisingVectorAdapter,
    )

    ammr_fn = _get_method(
        instance,
        framework_descriptor.async_mmr_query_method,
    )

    with pytest.raises(RuntimeError, match="backend failure"):
        if framework_descriptor.context_kwarg:
            coro = ammr_fn(
                "err-async-mmr",
                **{framework_descriptor.context_kwarg: {}},
            )
        else:
            coro = ammr_fn("err-async-mmr")

        assert inspect.isawaitable(coro), "Async MMR method must return an awaitable"
        await coro  # noqa: PT018

    assert calls, "attach_context was not called for async backend MMR failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name
    assert str(ctx["operation"]).startswith(VECTOR_OPERATION_PREFIX)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
