# tests/frameworks/embedding/test_with_mock_backends.py

from __future__ import annotations

import importlib
import inspect
from collections.abc import Sequence
from typing import Any, Callable, Type

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


# ---------------------------------------------------------------------------
# "Evil" translators
# ---------------------------------------------------------------------------


class InvalidShapeTranslator:
    """
    Translator that returns blatantly invalid shapes.

    - For any input, returns a non-sequence scalar value.
    This should trigger coercion errors in the framework adapters.
    """

    def embed(
        self,
        raw_texts: Any,
        *,
        op_ctx: Any | None = None,
        framework_ctx: Any | None = None,
    ) -> Any:
        return "not-an-embedding-matrix"

    async def arun_embed(
        self,
        raw_texts: Any,
        *,
        op_ctx: Any | None = None,
        framework_ctx: Any | None = None,
    ) -> Any:
        return "not-an-embedding-matrix"


class EmptyResultTranslator:
    """
    Translator that always returns an empty list.

    This is used to verify how adapters handle completely empty results
    from the underlying translation layer.
    """

    def embed(
        self,
        raw_texts: Any,
        *,
        op_ctx: Any | None = None,
        framework_ctx: Any | None = None,
    ) -> Any:
        return []

    async def arun_embed(
        self,
        raw_texts: Any,
        *,
        op_ctx: Any | None = None,
        framework_ctx: Any | None = None,
    ) -> Any:
        return []


class RaisingTranslator:
    """
    Translator that always raises.

    Used to validate that error-context decorators still attach context when
    failures originate in the translation layer, not the corpus adapter.
    """

    def embed(
        self,
        raw_texts: Any,
        *,
        op_ctx: Any | None = None,
        framework_ctx: Any | None = None,
    ) -> Any:
        raise RuntimeError("intentional translator failure (sync)")

    async def arun_embed(
        self,
        raw_texts: Any,
        *,
        op_ctx: Any | None = None,
        framework_ctx: Any | None = None,
    ) -> Any:
        raise RuntimeError("intentional translator failure (async)")


class WrongRowCountTranslator:
    """
    Translator that returns a fixed number of rows regardless of input length.

    Used to verify that adapters validate row-count consistency between the
    input texts and the returned embedding matrix.
    """

    def embed(
        self,
        raw_texts: Any,
        *,
        op_ctx: Any | None = None,
        framework_ctx: Any | None = None,
    ) -> Any:
        # Always a single row, regardless of how many texts were provided.
        return [[0.1, 0.2, 0.3]]

    async def arun_embed(
        self,
        raw_texts: Any,
        *,
        op_ctx: Any | None = None,
        framework_ctx: Any | None = None,
    ) -> Any:
        return [[0.1, 0.2, 0.3]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_method(instance: Any, name: str | None) -> Callable[..., Any]:
    """Helper to fetch a method from the instance and assert it is callable."""
    assert name is not None, "Method name must not be None"
    attr = getattr(instance, name, None)
    assert callable(attr), f"{instance!r} missing expected callable method {name!r}"
    return attr


def _make_adapter_with_evil_translator(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    adapter: Any,
    translator_cls: Type[Any],
) -> Any:
    """
    Instantiate the framework adapter and forcibly inject an 'evil' translator.

    This bypasses the normal create_embedding_translator wiring and lets us
    simulate misbehaving translation layers in a controlled way.
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    adapter_cls = getattr(module, framework_descriptor.adapter_class)

    init_kwargs: dict[str, Any] = {"corpus_adapter": adapter}
    if framework_descriptor.requires_embedding_dimension:
        init_kwargs.setdefault("embedding_dimension", 8)

    instance = adapter_cls(**init_kwargs)
    evil_translator = translator_cls()

    # For adapters that use a cached_property or PrivateAttr cache.
    if hasattr(instance, "_translator_cache"):
        setattr(instance, "_translator_cache", evil_translator)

    # For adapters that expose _translator as a cached_property (or similar).
    try:
        setattr(instance, "_translator", evil_translator)
    except Exception:
        # Not fatal – if the adapter only uses _translator_cache, that's fine.
        pass

    return instance


def _call_batch(
    descriptor: EmbeddingFrameworkDescriptor,
    instance: Any,
    texts: Sequence[str],
) -> Any:
    batch_fn = _get_method(instance, descriptor.batch_method)
    if descriptor.context_kwarg:
        return batch_fn(texts, **{descriptor.context_kwarg: {}})
    return batch_fn(texts)


def _call_query(
    descriptor: EmbeddingFrameworkDescriptor,
    instance: Any,
    text: str,
) -> Any:
    query_fn = _get_method(instance, descriptor.query_method)
    if descriptor.context_kwarg:
        return query_fn(text, **{descriptor.context_kwarg: {}})
    return query_fn(text)


# ---------------------------------------------------------------------------
# Invalid shape behavior
# ---------------------------------------------------------------------------


def test_invalid_translator_shape_causes_errors_for_batch_and_query(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    adapter: Any,
) -> None:
    """
    If the translator returns a clearly invalid shape (non-sequence scalar),
    the adapter should surface a coercion-style error rather than silently
    returning nonsense.

    We don't over-specify the exact exception type; any Exception is acceptable.
    """
    instance = _make_adapter_with_evil_translator(
        framework_descriptor,
        adapter,
        InvalidShapeTranslator,
    )

    texts = ["x", "y"]
    query_text = "z"

    # Batch should fail
    with pytest.raises(Exception):  # noqa: BLE001
        _call_batch(framework_descriptor, instance, texts)

    # Query should also fail
    with pytest.raises(Exception):  # noqa: BLE001
        _call_query(framework_descriptor, instance, query_text)


@pytest.mark.asyncio
async def test_async_invalid_translator_shape_causes_errors_when_supported(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    adapter: Any,
) -> None:
    """
    Same as above but for async methods when the framework declares async support.
    """
    if not framework_descriptor.supports_async:
        pytest.skip(f"Framework '{framework_descriptor.name}' does not declare async support")

    assert framework_descriptor.async_batch_method is not None
    assert framework_descriptor.async_query_method is not None

    instance = _make_adapter_with_evil_translator(
        framework_descriptor,
        adapter,
        InvalidShapeTranslator,
    )

    abatch_fn = _get_method(instance, framework_descriptor.async_batch_method)
    aquery_fn = _get_method(instance, framework_descriptor.async_query_method)

    texts = ["x-async", "y-async"]
    query_text = "z-async"

    # Batch async
    with pytest.raises(Exception):  # noqa: BLE001
        if framework_descriptor.context_kwarg:
            coro = abatch_fn(texts, **{framework_descriptor.context_kwarg: {}})
        else:
            coro = abatch_fn(texts)

        assert inspect.isawaitable(coro), "Async batch method must return an awaitable"
        await coro  # noqa: PT018

    # Query async
    with pytest.raises(Exception):  # noqa: BLE001
        if framework_descriptor.context_kwarg:
            coro = aquery_fn(query_text, **{framework_descriptor.context_kwarg: {}})
        else:
            coro = aquery_fn(query_text)

        assert inspect.isawaitable(coro), "Async query method must return an awaitable"
        await coro  # noqa: PT018


# ---------------------------------------------------------------------------
# Empty result behavior
# ---------------------------------------------------------------------------


def test_empty_translator_result_is_not_silently_treated_as_valid_embedding(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    adapter: Any,
) -> None:
    """
    When the translator returns an empty result, the adapter should not
    silently treat it as a valid embedding.

    Acceptable behaviors:
    - Raise an Exception (preferred; empty result is usually an error), or
    - Return a sequence whose row count does not match the input (still a bug
      but surfaced to the caller).

    This test simply asserts that we *don't* get a plausible embedding matrix
    with one row per input text.
    """
    instance = _make_adapter_with_evil_translator(
        framework_descriptor,
        adapter,
        EmptyResultTranslator,
    )

    texts = ["alpha", "beta"]

    try:
        result = _call_batch(framework_descriptor, instance, texts)
    except Exception:  # noqa: BLE001
        # Raising is acceptable and expected in many implementations.
        return

    # If it did not raise, we at least require that the result is obviously
    # not a valid 2D embedding matrix with correct row count.
    assert not isinstance(result, Sequence) or len(result) != len(texts), (
        "EmptyResultTranslator unexpectedly produced a valid-looking embedding "
        "matrix; adapters should treat empty translator outputs as an error."
    )


def test_translator_returning_wrong_row_count_causes_errors_or_obvious_mismatch(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    adapter: Any,
) -> None:
    """
    When the translator returns a matrix with the wrong number of rows, the
    adapter should not silently treat it as valid.

    Acceptable behaviors:
    - Raise an Exception, or
    - Return a sequence whose length != len(input_texts).
    """
    instance = _make_adapter_with_evil_translator(
        framework_descriptor,
        adapter,
        WrongRowCountTranslator,
    )

    texts = ["a", "b", "c"]  # 3 inputs

    try:
        result = _call_batch(framework_descriptor, instance, texts)
    except Exception:  # noqa: BLE001
        # Raising is acceptable / preferred.
        return

    # If not raised, it must at least be obviously wrong in row-count terms.
    assert not isinstance(result, Sequence) or len(result) != len(texts), (
        "WrongRowCountTranslator produced a matrix whose row count matches the "
        "input length. Adapters should validate translator row counts and treat "
        "mismatches as errors."
    )


# ---------------------------------------------------------------------------
# Error-context when translator raises
# ---------------------------------------------------------------------------


def test_translator_exception_is_wrapped_with_error_context_on_batch(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the translator itself raises during a batch operation, the framework
    adapter's error-context decorator should still call attach_context().

    This ensures that failures originating in the translation layer are just
    as observable as failures from the underlying corpus adapter.
    """
    module = importlib.import_module(framework_descriptor.adapter_module)

    calls: list[tuple[BaseException, dict[str, Any]]] = []

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        calls.append((exc, ctx))

    monkeypatch.setattr(module, "attach_context", fake_attach_context)

    instance = _make_adapter_with_evil_translator(
        framework_descriptor,
        adapter,
        RaisingTranslator,
    )

    texts = ["err-batch-1", "err-batch-2"]

    with pytest.raises(RuntimeError, match="translator failure"):
        _call_batch(framework_descriptor, instance, texts)

    assert calls, "attach_context was not called for translator batch failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["operation"].startswith("embedding_")


def test_translator_exception_is_wrapped_with_error_context_on_query(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Same as the batch test but for the sync query path.
    """
    module = importlib.import_module(framework_descriptor.adapter_module)

    calls: list[tuple[BaseException, dict[str, Any]]] = []

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        calls.append((exc, ctx))

    monkeypatch.setattr(module, "attach_context", fake_attach_context)

    instance = _make_adapter_with_evil_translator(
        framework_descriptor,
        adapter,
        RaisingTranslator,
    )

    with pytest.raises(RuntimeError, match="translator failure"):
        _call_query(framework_descriptor, instance, "err-query")

    assert calls, "attach_context was not called for translator query failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["operation"].startswith("embedding_")


@pytest.mark.asyncio
async def test_async_translator_exception_is_wrapped_with_error_context_when_supported(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async is supported, translator exceptions in async methods should
    also go through the error-context decorators and call attach_context().
    """
    if not framework_descriptor.supports_async:
        pytest.skip(f"Framework '{framework_descriptor.name}' does not declare async support")

    assert framework_descriptor.async_batch_method is not None
    assert framework_descriptor.async_query_method is not None

    module = importlib.import_module(framework_descriptor.adapter_module)

    calls: list[tuple[BaseException, dict[str, Any]]] = []

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        calls.append((exc, ctx))

    monkeypatch.setattr(module, "attach_context", fake_attach_context)

    instance = _make_adapter_with_evil_translator(
        framework_descriptor,
        adapter,
        RaisingTranslator,
    )

    abatch_fn = _get_method(instance, framework_descriptor.async_batch_method)
    aquery_fn = _get_method(instance, framework_descriptor.async_query_method)

    # Batch async
    with pytest.raises(RuntimeError, match="translator failure"):
        if framework_descriptor.context_kwarg:
            coro = abatch_fn(["err-async-batch"], **{framework_descriptor.context_kwarg: {}})
        else:
            coro = abatch_fn(["err-async-batch"])

        assert inspect.isawaitable(coro), "Async batch method must return an awaitable"
        await coro  # noqa: PT018

    # Query async
    with pytest.raises(RuntimeError, match="translator failure"):
        if framework_descriptor.context_kwarg:
            coro = aquery_fn("err-async-query", **{framework_descriptor.context_kwarg: {}})
        else:
            coro = aquery_fn("err-async-query")

        assert inspect.isawaitable(coro), "Async query method must return an awaitable"
        await coro  # noqa: PT018

    assert calls, "attach_context was not called for async translator failures"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert ctx["operation"].startswith("embedding_")

