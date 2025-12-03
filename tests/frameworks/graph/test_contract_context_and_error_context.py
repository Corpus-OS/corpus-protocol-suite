# tests/frameworks/graph/test_contract_context_and_error_context.py

from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable

import pytest

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


@pytest.fixture
def graph_client_instance(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_adapter: Any,
) -> Any:
    """
    Construct a concrete graph client instance for the given descriptor.

    Mirrors the construction pattern used in the other graph contract tests:
    each framework adapter is expected to take a `graph_adapter` kwarg that
    wraps a Corpus GraphProtocolV1 implementation.
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    client_cls = getattr(module, framework_descriptor.adapter_class)

    init_kwargs: dict[str, Any] = {"graph_adapter": graph_adapter}
    instance = client_cls(**init_kwargs)
    return instance


@pytest.fixture
def failing_graph_adapter() -> Any:
    """
    A minimal graph adapter whose query methods always fail.

    Used only for error-context tests to ensure the decorators invoke
    attach_context() and propagate the exception.
    """

    class FailingGraphAdapter:
        def query(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        async def aquery(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

    return FailingGraphAdapter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _maybe_call_with_context(
    descriptor: GraphFrameworkDescriptor,
    fn: Callable[..., Any],
    query_text: str,
    context: Any,
) -> Any:
    """
    Call a graph client function, respecting descriptor.context_kwarg if present.

    This helper allows injecting either a valid Mapping context or an
    intentionally invalid context for robustness tests.
    """
    if descriptor.context_kwarg:
        return fn(query_text, **{descriptor.context_kwarg: context})
    return fn(query_text)


def _build_error_wrapped_client_instance(
    framework_descriptor: GraphFrameworkDescriptor,
    failing_graph_adapter: Any,
) -> Any:
    """
    Construct a graph client instance wired to a failing graph adapter.

    Used only for error-context tests (we expect calls to raise).
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    client_cls = getattr(module, framework_descriptor.adapter_class)

    init_kwargs: dict[str, Any] = {"graph_adapter": failing_graph_adapter}
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
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
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

    query_method = _get_method(graph_client_instance, framework_descriptor.query_method)

    rich_context = {
        **RICH_CONTEXT,
        "tags": [*RICH_CONTEXT["tags"], framework_descriptor.name],
    }

    query_text = "ctx-rich-query"

    # Should not raise; result shape is validated lightly here since other
    # contract tests cover detailed QueryResult semantics.
    result = _maybe_call_with_context(
        framework_descriptor,
        query_method,
        query_text,
        context=rich_context,
    )
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
    """
    if not framework_descriptor.context_kwarg:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare a context_kwarg",
        )

    query_method = _get_method(graph_client_instance, framework_descriptor.query_method)

    query_text = "ctx-invalid-query"

    invalid_contexts = ["not-a-mapping", 12345]

    for invalid_ctx in invalid_contexts:
        result = _maybe_call_with_context(
            framework_descriptor,
            query_method,
            query_text,
            context=invalid_ctx,
        )
        assert result is not None


def test_context_is_optional_and_omitting_it_still_works(
    framework_descriptor: GraphFrameworkDescriptor,
    graph_client_instance: Any,
) -> None:
    """
    Even when a framework supports a context kwarg, it must still work
    when no context is provided.
    """
    query_method = _get_method(graph_client_instance, framework_descriptor.query_method)

    query_text = "ctx-optional-query"

    # No context kwarg passed at all.
    result = query_method(query_text)
    assert result is not None


# ---------------------------------------------------------------------------
# Error-context decorator contract tests
# ---------------------------------------------------------------------------


def test_error_context_is_attached_on_sync_query_failure(
    framework_descriptor: GraphFrameworkDescriptor,
    failing_graph_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the underlying graph adapter raises during a sync query operation,
    the framework adapter's error-context decorator should:

    - call attach_context() with the exception and useful metadata, and
    - re-raise the original exception (or a wrapped one).

    We assert that attach_context is invoked and that the operation name
    looks like a graph operation (e.g. starts with "graph_").
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _build_error_wrapped_client_instance(
        framework_descriptor,
        failing_graph_adapter,
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
    assert str(ctx["operation"]).startswith(GRAPH_OPERATION_PREFIX)


def test_error_context_is_attached_on_sync_stream_failure_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
    failing_graph_adapter: Any,
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
        failing_graph_adapter,
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
    assert str(ctx["operation"]).startswith(GRAPH_OPERATION_PREFIX)


@pytest.mark.asyncio
async def test_error_context_is_attached_on_async_query_failure_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
    failing_graph_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async is supported, async query failures should also go through
    the error-context decorator and call attach_context().
    """
    if not framework_descriptor.async_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async query support",
        )

    module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context(monkeypatch, module)

    instance = _build_error_wrapped_client_instance(
        framework_descriptor,
        failing_graph_adapter,
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
    assert str(ctx["operation"]).startswith(GRAPH_OPERATION_PREFIX)


@pytest.mark.asyncio
async def test_error_context_is_attached_on_async_stream_failure_when_supported(
    framework_descriptor: GraphFrameworkDescriptor,
    failing_graph_adapter: Any,
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
        failing_graph_adapter,
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

        # aiter may be an async iterator or an awaitable that resolves to one.
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
    assert str(ctx["operation"]).startswith(GRAPH_OPERATION_PREFIX)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

