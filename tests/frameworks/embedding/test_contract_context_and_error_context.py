# tests/frameworks/embedding/test_contract_context_and_error_context.py

from __future__ import annotations

import importlib
import inspect
from collections.abc import Sequence
from typing import Any, Callable

import pytest

from tests.frameworks.registries.embedding_registry import (
    EmbeddingFrameworkDescriptor,
    iter_embedding_framework_descriptors,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=list(iter_embedding_framework_descriptors()),
    name="framework_descriptor",
)
def framework_descriptor_fixture(
    request: pytest.FixtureRequest,
) -> EmbeddingFrameworkDescriptor:
    """
    Parameterized over all registered embedding framework descriptors.

    Frameworks that are not actually available in the environment (e.g. the
    underlying LangChain / LlamaIndex / Semantic Kernel libraries are missing)
    are skipped via descriptor.is_available().
    """
    descriptor: EmbeddingFrameworkDescriptor = request.param
    if not descriptor.is_available():
        pytest.skip(f"Framework '{descriptor.name}' not available in this environment")
    return descriptor


@pytest.fixture
def embedding_adapter_instance(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    adapter: Any,
) -> Any:
    """
    Construct a concrete framework adapter instance for the given descriptor.

    Mirrors the construction pattern used in the other embedding contract tests.
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    adapter_cls = getattr(module, framework_descriptor.adapter_class)

    init_kwargs: dict[str, Any] = {"corpus_adapter": adapter}

    # Some adapters require a known embedding dimension up-front.
    if framework_descriptor.requires_embedding_dimension:
        init_kwargs.setdefault("embedding_dimension", 8)

    instance = adapter_cls(**init_kwargs)
    return instance


@pytest.fixture
def failing_corpus_adapter() -> Any:
    """
    A minimal corpus adapter whose embed() always fails.

    Used only for error-context tests to ensure the decorators invoke
    attach_context() and propagate the exception.
    """

    class FailingEmbeddingAdapter:
        def embed(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("intentional failure from failing adapter")

    return FailingEmbeddingAdapter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_method(instance: Any, name: str) -> Callable[..., Any]:
    """Helper to fetch a method from the instance and assert it is callable."""
    attr = getattr(instance, name, None)
    assert callable(attr), f"{instance!r} missing expected callable method {name!r}"
    return attr


def _assert_embedding_matrix_shape(
    result: Any,
    expected_rows: int,
) -> None:
    """
    Validate that a result looks like a 2D embedding matrix.

    - Must be a sequence
    - Must have expected_rows rows
    - Each row must be a sequence
    - Values (if present) must be numeric
    """
    assert isinstance(result, Sequence), f"Expected sequence, got {type(result).__name__}"
    assert len(result) == expected_rows, f"Expected {expected_rows} rows, got {len(result)}"

    for row in result:
        assert isinstance(row, Sequence), f"Row is not a sequence: {type(row).__name__}"
        for val in row:
            assert isinstance(val, (int, float)), f"Embedding value is not numeric: {val!r}"


def _assert_embedding_vector_shape(result: Any) -> None:
    """
    Validate that a result looks like a 1D embedding vector.

    - Must be a sequence
    - Values (if present) must be numeric
    """
    assert isinstance(result, Sequence), f"Expected sequence, got {type(result).__name__}"
    for val in result:
        assert isinstance(val, (int, float)), f"Embedding value is not numeric: {val!r}"


def _maybe_call_with_context(
    descriptor: EmbeddingFrameworkDescriptor,
    fn: Callable[..., Any],
    texts_or_text: Any,
    context: Any,
) -> Any:
    """
    Call an embedding function, respecting descriptor.context_kwarg if present.

    This helper allows injecting either a valid Mapping context or an
    intentionally invalid context for robustness tests.
    """
    if descriptor.context_kwarg:
        return fn(texts_or_text, **{descriptor.context_kwarg: context})
    return fn(texts_or_text)


def _build_error_wrapped_adapter_instance(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    failing_corpus_adapter: Any,
) -> Any:
    """
    Construct a framework adapter instance wired to a failing corpus adapter.

    Used only for error-context tests (we expect calls to raise).
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    adapter_cls = getattr(module, framework_descriptor.adapter_class)

    init_kwargs: dict[str, Any] = {"corpus_adapter": failing_corpus_adapter}

    if framework_descriptor.requires_embedding_dimension:
        init_kwargs.setdefault("embedding_dimension", 8)

    return adapter_cls(**init_kwargs)


# ---------------------------------------------------------------------------
# Context contract tests
# ---------------------------------------------------------------------------


def test_rich_mapping_context_is_accepted_and_does_not_break_embeddings(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    If a framework declares a context_kwarg, it should:

    - accept a rich Mapping (with extra / nested keys),
    - not raise TypeError / ValueError,
    - still return embeddings with valid shapes.

    Frameworks without a declared context_kwarg are skipped here.
    """
    if not framework_descriptor.context_kwarg:
        pytest.skip(f"Framework '{framework_descriptor.name}' does not declare a context_kwarg")

    batch_method = _get_method(embedding_adapter_instance, framework_descriptor.batch_method)
    query_method = _get_method(embedding_adapter_instance, framework_descriptor.query_method)

    rich_context = {
        "request_id": "req-123",
        "user_id": "user-abc",
        "tags": ["test", framework_descriptor.name],
        "nested": {"key": "value", "depth": 2},
    }

    texts = ["ctx-rich-alpha", "ctx-rich-beta"]
    query_text = "ctx-rich-query"

    batch_result = _maybe_call_with_context(
        framework_descriptor,
        batch_method,
        texts,
        context=rich_context,
    )
    _assert_embedding_matrix_shape(batch_result, expected_rows=len(texts))

    query_result = _maybe_call_with_context(
        framework_descriptor,
        query_method,
        query_text,
        context=rich_context,
    )
    _assert_embedding_vector_shape(query_result)


def test_invalid_context_type_is_tolerated_and_does_not_crash(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    Passing an invalid context type (non-Mapping) should not crash the adapter.

    The framework adapters are expected to either:
    - log a warning and ignore the context, or
    - gracefully treat it as "no context".

    In all cases, embeddings should still be returned.
    """
    if not framework_descriptor.context_kwarg:
        pytest.skip(f"Framework '{framework_descriptor.name}' does not declare a context_kwarg")

    batch_method = _get_method(embedding_adapter_instance, framework_descriptor.batch_method)
    query_method = _get_method(embedding_adapter_instance, framework_descriptor.query_method)

    texts = ["ctx-invalid-alpha", "ctx-invalid-beta"]
    query_text = "ctx-invalid-query"

    # Intentionally wrong types for context.
    invalid_context_for_batch = "not-a-mapping"
    invalid_context_for_query = 12345

    # Should not raise TypeError / ValueError
    batch_result = _maybe_call_with_context(
        framework_descriptor,
        batch_method,
        texts,
        context=invalid_context_for_batch,
    )
    _assert_embedding_matrix_shape(batch_result, expected_rows=len(texts))

    query_result = _maybe_call_with_context(
        framework_descriptor,
        query_method,
        query_text,
        context=invalid_context_for_query,
    )
    _assert_embedding_vector_shape(query_result)


def test_context_is_optional_and_omitting_it_still_works(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    Even when a framework supports a context kwarg, it must still work
    when no context is provided.
    """
    batch_method = _get_method(embedding_adapter_instance, framework_descriptor.batch_method)
    query_method = _get_method(embedding_adapter_instance, framework_descriptor.query_method)

    texts = ["ctx-optional-alpha", "ctx-optional-beta"]
    query_text = "ctx-optional-query"

    # No context kwarg passed at all.
    batch_result = batch_method(texts)
    _assert_embedding_matrix_shape(batch_result, expected_rows=len(texts))

    query_result = query_method(query_text)
    _assert_embedding_vector_shape(query_result)


# ---------------------------------------------------------------------------
# Error-context decorator contract tests
# ---------------------------------------------------------------------------


def test_error_context_is_attached_on_sync_batch_failure(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    failing_corpus_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the underlying corpus adapter raises during a sync batch operation,
    the framework adapter's error-context decorator should:

    - call attach_context() with the exception and useful metadata, and
    - re-raise the original exception (or a wrapped one).

    We assert that attach_context is invoked and that the operation name
    starts with "embedding_".
    """
    module = importlib.import_module(framework_descriptor.adapter_module)

    calls: list[tuple[BaseException, dict[str, Any]]] = []

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        calls.append((exc, ctx))

    # Patch the module-local attach_context reference used by decorators.
    monkeypatch.setattr(module, "attach_context", fake_attach_context)

    instance = _build_error_wrapped_adapter_instance(
        framework_descriptor,
        failing_corpus_adapter,
    )

    batch_method = _get_method(instance, framework_descriptor.batch_method)

    with pytest.raises(RuntimeError, match="intentional failure"):
        if framework_descriptor.context_kwarg:
            batch_method(["err-batch"], **{framework_descriptor.context_kwarg: {}})
        else:
            batch_method(["err-batch"])

    assert calls, "attach_context was not called on sync batch failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    # Decorators provide at least framework + operation.
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["operation"].startswith("embedding_")


def test_error_context_is_attached_on_sync_query_failure(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    failing_corpus_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Same as the batch failure test, but for the sync query operation.

    Ensures the query path is also wrapped by the error-context decorator.
    """
    module = importlib.import_module(framework_descriptor.adapter_module)

    calls: list[tuple[BaseException, dict[str, Any]]] = []

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        calls.append((exc, ctx))

    monkeypatch.setattr(module, "attach_context", fake_attach_context)

    instance = _build_error_wrapped_adapter_instance(
        framework_descriptor,
        failing_corpus_adapter,
    )

    query_method = _get_method(instance, framework_descriptor.query_method)

    with pytest.raises(RuntimeError, match="intentional failure"):
        if framework_descriptor.context_kwarg:
            query_method("err-query", **{framework_descriptor.context_kwarg: {}})
        else:
            query_method("err-query")

    assert calls, "attach_context was not called on sync query failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["operation"].startswith("embedding_")


@pytest.mark.asyncio
async def test_error_context_is_attached_on_async_batch_failure_when_supported(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    failing_corpus_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async is supported, async batch failures should also go through
    the error-context decorator and call attach_context().
    """
    if not framework_descriptor.supports_async:
        pytest.skip(f"Framework '{framework_descriptor.name}' does not declare async support")

    assert framework_descriptor.async_batch_method is not None

    module = importlib.import_module(framework_descriptor.adapter_module)

    calls: list[tuple[BaseException, dict[str, Any]]] = []

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        calls.append((exc, ctx))

    monkeypatch.setattr(module, "attach_context", fake_attach_context)

    instance = _build_error_wrapped_adapter_instance(
        framework_descriptor,
        failing_corpus_adapter,
    )

    abatch_method = _get_method(instance, framework_descriptor.async_batch_method)

    with pytest.raises(RuntimeError, match="intentional failure"):
        if framework_descriptor.context_kwarg:
            coro = abatch_method(["err-abatch"], **{framework_descriptor.context_kwarg: {}})
        else:
            coro = abatch_method(["err-abatch"])

        assert inspect.isawaitable(coro), "Async batch method must return an awaitable"
        await coro  # noqa: PT018

    assert calls, "attach_context was not called on async batch failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["operation"].startswith("embedding_")


@pytest.mark.asyncio
async def test_error_context_is_attached_on_async_query_failure_when_supported(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    failing_corpus_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async is supported, async query failures should also go through
    the error-context decorator and call attach_context().
    """
    if not framework_descriptor.supports_async:
        pytest.skip(f"Framework '{framework_descriptor.name}' does not declare async support")

    assert framework_descriptor.async_query_method is not None

    module = importlib.import_module(framework_descriptor.adapter_module)

    calls: list[tuple[BaseException, dict[str, Any]]] = []

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        calls.append((exc, ctx))

    monkeypatch.setattr(module, "attach_context", fake_attach_context)

    instance = _build_error_wrapped_adapter_instance(
        framework_descriptor,
        failing_corpus_adapter,
    )

    aquery_method = _get_method(instance, framework_descriptor.async_query_method)

    with pytest.raises(RuntimeError, match="intentional failure"):
        if framework_descriptor.context_kwarg:
            coro = aquery_method("err-aquery", **{framework_descriptor.context_kwarg: {}})
        else:
            coro = aquery_method("err-aquery")

        assert inspect.isawaitable(coro), "Async query method must return an awaitable"
        await coro  # noqa: PT018

    assert calls, "attach_context was not called on async query failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["operation"].startswith("embedding_")

