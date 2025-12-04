# tests/frameworks/vector/test_contract_context_and_error_context.py

from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable

import pytest

from tests.frameworks.registries.vector_registry import (
    VectorFrameworkDescriptor,
    iter_vector_framework_descriptors,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAILURE_MESSAGE = "intentional failure from failing vector adapter"
VECTOR_OPERATION_PREFIX = "vector_"

RICH_CONTEXT = {
    "request_id": "req-123",
    "user_id": "user-abc",
    "tags": ["test"],
    "nested": {"key": "value", "depth": 2},
}

QUERY_TEXT = "ctx-query"
STREAM_QUERY_TEXT = "ctx-stream-query"
MMR_QUERY_TEXT = "ctx-mmr-query"


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
    underlying LlamaIndex / Semantic Kernel libraries are missing) are skipped
    via descriptor.is_available().
    """
    descriptor: VectorFrameworkDescriptor = request.param
    if not descriptor.is_available():
        pytest.skip(f"Framework '{descriptor.name}' not available in this environment")
    return descriptor


@pytest.fixture
def vector_client_instance(
    framework_descriptor: VectorFrameworkDescriptor,
    adapter: Any,
) -> Any:
    """
    Construct a concrete vector client/store instance for the given descriptor.

    Mirrors the construction pattern used in the other vector contract tests:
    each framework adapter is expected to take an `adapter` kwarg that wraps
    a Corpus vector protocol implementation.
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    client_cls = getattr(module, framework_descriptor.adapter_class)

    init_kwargs: dict[str, Any] = {"adapter": adapter}
    instance = client_cls(**init_kwargs)
    return instance


@pytest.fixture
def failing_adapter() -> Any:
    """
    A minimal vector adapter whose query / stream / MMR methods always fail.

    Used only for error-context tests to ensure the decorators invoke
    attach_context() and propagate the exception.
    """

    class FailingVectorAdapter:
        # Core query surfaces
        def query(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        async def aquery(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        # Streaming
        def query_stream(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        async def aquery_stream(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        # MMR
        def query_mmr(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        async def aquery_mmr(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

    return FailingVectorAdapter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_method(instance: Any, name: str | None) -> Callable[..., Any]:
    """
    Helper to fetch a method from the instance and assert it is callable.

    If name is None, this fails fast with a clear assertion message so we
    don't silently mis-test a missing surface.
    """
    assert name, "Expected a non-empty method name"
    attr = getattr(instance, name, None)
    assert callable(attr), f"{instance!r} missing expected callable method {name!r}"
    return attr


def _maybe_call_with_context(
    descriptor: VectorFrameworkDescriptor,
    fn: Callable[..., Any],
    *args: Any,
    context: Any,
) -> Any:
    """
    Call a vector client method, respecting descriptor.context_kwarg if present.

    This helper allows injecting either a valid Mapping context or an
    intentionally invalid context for robustness tests.
    """
    if descriptor.context_kwarg:
        return fn(*args, **{descriptor.context_kwarg: context})
    return fn(*args)


def _build_error_wrapped_client_instance(
    framework_descriptor: VectorFrameworkDescriptor,
    failing_adapter: Any,
) -> Any:
    """
    Construct a vector client instance wired to a failing vector adapter.

    Used only for error-context tests (we expect calls to raise).
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    client_cls = getattr(module, framework_descriptor.adapter_class)
    init_kwargs: dict[str, Any] = {"adapter": failing_adapter}
    return client_cls(**init_kwargs)


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
# Context contract tests
# ---------------------------------------------------------------------------


def test_rich_mapping_context_is_accepted_and_does_not_break_queries(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    If a framework declares a context_kwarg, it should:

    - accept a rich Mapping (with extra / nested keys),
    - not raise TypeError / ValueError,
    - still return a valid query result.

    Frameworks without a declared context_kwarg are skipped here.
    """
    if not framework_descriptor.context_kwarg:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare a context_kwarg",
        )

    query_method = _get_method(vector_client_instance, framework_descriptor.query_method)

    rich_context = {
        **RICH_CONTEXT,
        "tags": [*RICH_CONTEXT["tags"], framework_descriptor.name],
    }

    result = _maybe_call_with_context(
        framework_descriptor,
        query_method,
        QUERY_TEXT,
        context=rich_context,
    )
    assert result is not None


def test_invalid_context_type_is_tolerated_and_does_not_crash(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    Passing an invalid context type (non-Mapping) should not crash the adapter.

    The framework adapters are expected to either:
    - log a warning and ignore the context, or
    - gracefully treat it as "no context".

    In all cases, queries should still return results.
    """
    if not framework_descriptor.context_kwarg:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare a context_kwarg",
        )

    query_method = _get_method(vector_client_instance, framework_descriptor.query_method)

    invalid_contexts = ["not-a-mapping", 12345]

    for invalid_ctx in invalid_contexts:
        result = _maybe_call_with_context(
            framework_descriptor,
            query_method,
            QUERY_TEXT,
            context=invalid_ctx,
        )
        assert result is not None


def test_context_is_optional_and_omitting_it_still_works(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    Even when a framework supports a context kwarg, it must still work
    when no context is provided.
    """
    query_method = _get_method(vector_client_instance, framework_descriptor.query_method)

    result = query_method(QUERY_TEXT)
    assert result is not None


def test_context_on_mmr_queries_when_supported(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    When MMR is supported, the MMR query surface should also accept context
    without crashing and still return results.
    """
    if not framework_descriptor.supports_mmr or not framework_descriptor.mmr_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare MMR support",
        )

    if not framework_descriptor.context_kwarg:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare a context_kwarg",
        )

    mmr_method = _get_method(vector_client_instance, framework_descriptor.mmr_query_method)

    rich_context = {
        **RICH_CONTEXT,
        "tags": [*RICH_CONTEXT["tags"], "mmr", framework_descriptor.name],
    }

    result = _maybe_call_with_context(
        framework_descriptor,
        mmr_method,
        MMR_QUERY_TEXT,
        context=rich_context,
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Error-context decorator contract tests
# ---------------------------------------------------------------------------


def test_error_context_is_attached_on_sync_query_failure(
    framework_descriptor: VectorFrameworkDescriptor,
    failing_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the underlying vector adapter raises during a sync query operation,
    the framework adapter's error-context decorator should:

    - call attach_context() with the exception and useful metadata, and
    - re-raise the original exception (or a wrapped one).

    We assert that attach_context is invoked and that the operation name
    looks like a vector operation (e.g. starts with "vector_").
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _build_error_wrapped_client_instance(
        framework_descriptor,
        failing_adapter,
    )

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
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name
    assert str(ctx["operation"]).startswith(VECTOR_OPERATION_PREFIX)


def test_error_context_is_attached_on_sync_stream_failure_when_supported(
    framework_descriptor: VectorFrameworkDescriptor,
    failing_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When streaming is supported, sync stream failures should also go through
    the error-context decorator and call attach_context().
    """
    if not framework_descriptor.stream_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare sync streaming",
        )

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _build_error_wrapped_client_instance(
        framework_descriptor,
        failing_adapter,
    )

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
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name
    assert str(ctx["operation"]).startswith(VECTOR_OPERATION_PREFIX)


def test_error_context_is_attached_on_sync_mmr_failure_when_supported(
    framework_descriptor: VectorFrameworkDescriptor,
    failing_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When MMR is supported, sync MMR failures should also go through the
    error-context decorator and call attach_context().
    """
    if not framework_descriptor.supports_mmr or not framework_descriptor.mmr_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare MMR support",
        )

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _build_error_wrapped_client_instance(
        framework_descriptor,
        failing_adapter,
    )

    mmr_method = _get_method(instance, framework_descriptor.mmr_query_method)

    with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
        if framework_descriptor.context_kwarg:
            mmr_method(
                "err-mmr",
                **{framework_descriptor.context_kwarg: {}},
            )
        else:
            mmr_method("err-mmr")

    assert calls, "attach_context was not called on sync MMR failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name
    assert str(ctx["operation"]).startswith(VECTOR_OPERATION_PREFIX)


@pytest.mark.asyncio
async def test_error_context_is_attached_on_async_query_failure_when_supported(
    framework_descriptor: VectorFrameworkDescriptor,
    failing_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async query is declared, async query failures should also go through
    the error-context decorator and call attach_context().
    """
    if not framework_descriptor.async_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async query",
        )

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _build_error_wrapped_client_instance(
        framework_descriptor,
        failing_adapter,
    )

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
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name
    assert str(ctx["operation"]).startswith(VECTOR_OPERATION_PREFIX)


@pytest.mark.asyncio
async def test_error_context_is_attached_on_async_stream_failure_when_supported(
    framework_descriptor: VectorFrameworkDescriptor,
    failing_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async streaming is supported, async stream failures should also go
    through the error-context decorator and call attach_context().
    """
    if not framework_descriptor.async_stream_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async streaming",
        )

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _build_error_wrapped_client_instance(
        framework_descriptor,
        failing_adapter,
    )

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

        if inspect.isawaitable(aiter):
            aiter = await aiter  # type: ignore[assignment]

        async for _ in aiter:  # noqa: B007
            pass

    assert calls, "attach_context was not called on async stream failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name
    assert str(ctx["operation"]).startswith(VECTOR_OPERATION_PREFIX)


@pytest.mark.asyncio
async def test_error_context_is_attached_on_async_mmr_failure_when_supported(
    framework_descriptor: VectorFrameworkDescriptor,
    failing_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async MMR is supported, async MMR failures should also go through
    the error-context decorator and call attach_context().
    """
    if not framework_descriptor.supports_mmr:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare MMR support",
        )

    if not framework_descriptor.async_mmr_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async MMR support",
        )

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _build_error_wrapped_client_instance(
        framework_descriptor,
        failing_adapter,
    )

    ammr_method = _get_method(instance, framework_descriptor.async_mmr_query_method)

    with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
        if framework_descriptor.context_kwarg:
            coro = ammr_method(
                "err-ammr",
                **{framework_descriptor.context_kwarg: {}},
            )
        else:
            coro = ammr_method("err-ammr")

        assert inspect.isawaitable(coro), "Async MMR method must return an awaitable"
        await coro  # noqa: PT018

    assert calls, "attach_context was not called on async MMR failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["framework"] == framework_descriptor.name
    assert str(ctx["operation"]).startswith(VECTOR_OPERATION_PREFIX)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
