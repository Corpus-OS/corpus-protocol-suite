# tests/frameworks/embedding/test_with_mock_backends.py

from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping as ABCMapping
from collections.abc import Sequence
from typing import Any, Callable, Optional, Type

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

    IMPORTANT POLICY (no pytest.skip):
    - We do not skip unavailable frameworks.
    - Tests must pass by asserting correct "unavailable" signaling when a framework
      is not installed, and must fully run when it is available.
    """
    descriptor: EmbeddingFrameworkDescriptor = request.param
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


def _assert_unavailable_contract(descriptor: EmbeddingFrameworkDescriptor) -> None:
    """
    Validate that an unavailable framework descriptor is behaving as expected.

    The test suite policy is "no skip": when unavailable, tests must pass by
    asserting correct unavailability signaling.
    """
    assert descriptor.is_available() is False

    # If availability_attr is set, adapter module should generally import and expose the flag.
    # If import fails or flag is missing/False, that is an acceptable "unavailable" signal.
    if descriptor.availability_attr:
        try:
            module = importlib.import_module(descriptor.adapter_module)
        except Exception:
            return
        flag = getattr(module, descriptor.availability_attr, None)
        assert flag is None or bool(flag) is False


def _get_method(instance: Any, name: str | None) -> Callable[..., Any]:
    """Helper to fetch a method from the instance and assert it is callable."""
    assert name is not None, "Method name must not be None"
    attr = getattr(instance, name, None)
    assert callable(attr), f"{instance!r} missing expected callable method {name!r}"
    return attr


def _context_payload(descriptor: EmbeddingFrameworkDescriptor) -> Any:
    """
    Build a minimal context payload for the framework, preferring registry-provided sample_context.

    Why:
    - Some frameworks (notably BaseEmbedding-style surfaces) accept context via **kwargs
      rather than a single context kwarg. A structured mapping helps us exercise those paths.
    """
    return dict(descriptor.sample_context or {})


def _call_with_context(
    descriptor: EmbeddingFrameworkDescriptor,
    fn: Callable[..., Any],
    texts_or_text: Any,
    *,
    context: Any,
) -> Any:
    """
    Call an embedding function with context in a robust, framework-agnostic way.

    Primary strategy:
      - If descriptor.context_kwarg is set, pass {context_kwarg: context}.

    Compatibility fallback:
      - If that raises TypeError due to an unexpected keyword argument, and context is a Mapping,
        retry by spreading the mapping into kwargs (useful for BaseEmbedding-style **kwargs surfaces).

    This avoids test skips while remaining resilient to framework method signature shapes.
    """
    if not descriptor.context_kwarg:
        return fn(texts_or_text)

    try:
        return fn(texts_or_text, **{descriptor.context_kwarg: context})
    except TypeError as e:
        msg = str(e)
        unexpected_kw = f"unexpected keyword argument '{descriptor.context_kwarg}'" in msg or (
            "unexpected keyword" in msg and descriptor.context_kwarg in msg
        )
        if unexpected_kw and isinstance(context, ABCMapping):
            return fn(texts_or_text, **dict(context))
        raise


def _inject_translator(instance: Any, evil_translator: Any) -> None:
    """
    Inject an 'evil' translator into an adapter instance across the known cache patterns.

    This deliberately supports the five framework adapters' internal caching shapes:

    - LangChain:       _translator_cache (PrivateAttr)
    - AutoGen/CrewAI:  cached_property "_translator" stored in instance.__dict__
    - LlamaIndex:      cached_property "_translator" stored in instance.__dict__
    - Semantic Kernel: property _translator reads from _translator_instance

    We avoid relying on a single attribute name, because framework adapters use
    different caching approaches and (importantly) some use properties that cannot
    be overwritten by setting instance attributes.
    """
    # 1) Pydantic / PrivateAttr cache pattern (LangChain)
    if hasattr(instance, "_translator_cache"):
        try:
            setattr(instance, "_translator_cache", evil_translator)
        except Exception:
            pass

    # 2) Semantic Kernel pattern: property reads from _translator_instance
    if hasattr(instance, "_translator_instance"):
        try:
            setattr(instance, "_translator_instance", evil_translator)
        except Exception:
            pass

    # 3) cached_property pattern: store the computed value in instance.__dict__
    # cached_property is a non-data descriptor; instance dict overrides it.
    try:
        instance.__dict__["_translator"] = evil_translator
    except Exception:
        pass

    # 4) Best-effort direct set for any adapters that use a plain attribute.
    # This can fail if _translator is a @property (data descriptor), which is fine.
    try:
        setattr(instance, "_translator", evil_translator)
    except Exception:
        pass


def _make_adapter_with_evil_translator(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    adapter: Any,
    translator_cls: Type[Any],
) -> Optional[Any]:
    """
    Instantiate the framework adapter and forcibly inject an 'evil' translator.

    This bypasses the normal create_embedding_translator wiring and lets us
    simulate misbehaving translation layers in a controlled way.

    IMPORTANT POLICY (no pytest.skip):
    - If a framework is unavailable, returns None and tests must treat that as a
      validated pass condition by asserting the unavailable contract.
    """
    if not framework_descriptor.is_available():
        return None

    module = importlib.import_module(framework_descriptor.adapter_module)
    adapter_cls = getattr(module, framework_descriptor.adapter_class)

    init_kwargs: dict[str, Any] = {"corpus_adapter": adapter}
    if framework_descriptor.requires_embedding_dimension:
        kw = framework_descriptor.embedding_dimension_kwarg or "embedding_dimension"
        init_kwargs.setdefault(kw, 8)

    instance = adapter_cls(**init_kwargs)
    evil_translator = translator_cls()

    _inject_translator(instance, evil_translator)
    return instance


def _call_batch(
    descriptor: EmbeddingFrameworkDescriptor,
    instance: Any,
    texts: Sequence[str],
) -> Any:
    batch_fn = _get_method(instance, descriptor.batch_method)
    ctx = _context_payload(descriptor)
    return _call_with_context(descriptor, batch_fn, texts, context=ctx)


def _call_query(
    descriptor: EmbeddingFrameworkDescriptor,
    instance: Any,
    text: str,
) -> Any:
    query_fn = _get_method(instance, descriptor.query_method)
    ctx = _context_payload(descriptor)
    return _call_with_context(descriptor, query_fn, text, context=ctx)


def _patch_attach_context(
    adapter_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[BaseException, dict[str, Any]]],
) -> None:
    """
    Patch attach_context in both:
      1) the adapter module (module-local reference used by decorators), and
      2) the shared corpus_sdk.core.error_context module.

    This ensures we observe context attachment even if an adapter references either symbol.
    """

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
        # Minimal environments may not import core module; module-local patch is still valuable.
        pass


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

    IMPORTANT POLICY (no pytest.skip):
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    instance = _make_adapter_with_evil_translator(
        framework_descriptor,
        adapter,
        InvalidShapeTranslator,
    )
    assert instance is not None

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

    IMPORTANT POLICY (no pytest.skip):
    - If framework is unavailable, validate the unavailable contract and return.
    - If async is not supported, validate that and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.supports_async:
        assert framework_descriptor.async_batch_method is None
        assert framework_descriptor.async_query_method is None
        return

    assert framework_descriptor.async_batch_method is not None
    assert framework_descriptor.async_query_method is not None

    instance = _make_adapter_with_evil_translator(
        framework_descriptor,
        adapter,
        InvalidShapeTranslator,
    )
    assert instance is not None

    abatch_fn = _get_method(instance, framework_descriptor.async_batch_method)
    aquery_fn = _get_method(instance, framework_descriptor.async_query_method)

    texts = ["x-async", "y-async"]
    query_text = "z-async"
    ctx = _context_payload(framework_descriptor)

    # Batch async should fail
    with pytest.raises(Exception):  # noqa: BLE001
        coro = _call_with_context(framework_descriptor, abatch_fn, texts, context=ctx)
        assert inspect.isawaitable(coro), "Async batch method must return an awaitable"
        await coro  # noqa: PT018

    # Query async should fail
    with pytest.raises(Exception):  # noqa: BLE001
        coro = _call_with_context(framework_descriptor, aquery_fn, query_text, context=ctx)
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

    IMPORTANT POLICY (no pytest.skip):
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    instance = _make_adapter_with_evil_translator(
        framework_descriptor,
        adapter,
        EmptyResultTranslator,
    )
    assert instance is not None

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

    IMPORTANT POLICY (no pytest.skip):
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    instance = _make_adapter_with_evil_translator(
        framework_descriptor,
        adapter,
        WrongRowCountTranslator,
    )
    assert instance is not None

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

    IMPORTANT POLICY (no pytest.skip):
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    module = importlib.import_module(framework_descriptor.adapter_module)

    calls: list[tuple[BaseException, dict[str, Any]]] = []
    _patch_attach_context(module, monkeypatch, calls)

    instance = _make_adapter_with_evil_translator(
        framework_descriptor,
        adapter,
        RaisingTranslator,
    )
    assert instance is not None

    texts = ["err-batch-1", "err-batch-2"]

    with pytest.raises(RuntimeError, match="translator failure"):
        _call_batch(framework_descriptor, instance, texts)

    assert calls, "attach_context was not called for translator batch failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert str(ctx["operation"]).startswith("embedding_")


def test_translator_exception_is_wrapped_with_error_context_on_query(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Same as the batch test but for the sync query path.

    IMPORTANT POLICY (no pytest.skip):
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    module = importlib.import_module(framework_descriptor.adapter_module)

    calls: list[tuple[BaseException, dict[str, Any]]] = []
    _patch_attach_context(module, monkeypatch, calls)

    instance = _make_adapter_with_evil_translator(
        framework_descriptor,
        adapter,
        RaisingTranslator,
    )
    assert instance is not None

    with pytest.raises(RuntimeError, match="translator failure"):
        _call_query(framework_descriptor, instance, "err-query")

    assert calls, "attach_context was not called for translator query failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert str(ctx["operation"]).startswith("embedding_")


@pytest.mark.asyncio
async def test_async_translator_exception_is_wrapped_with_error_context_when_supported(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async is supported, translator exceptions in async methods should
    also go through the error-context decorators and call attach_context().

    IMPORTANT POLICY (no pytest.skip):
    - If framework is unavailable, validate the unavailable contract and return.
    - If async is not supported, validate that and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.supports_async:
        assert framework_descriptor.async_batch_method is None
        assert framework_descriptor.async_query_method is None
        return

    assert framework_descriptor.async_batch_method is not None
    assert framework_descriptor.async_query_method is not None

    module = importlib.import_module(framework_descriptor.adapter_module)

    calls: list[tuple[BaseException, dict[str, Any]]] = []
    _patch_attach_context(module, monkeypatch, calls)

    instance = _make_adapter_with_evil_translator(
        framework_descriptor,
        adapter,
        RaisingTranslator,
    )
    assert instance is not None

    abatch_fn = _get_method(instance, framework_descriptor.async_batch_method)
    aquery_fn = _get_method(instance, framework_descriptor.async_query_method)
    ctx = _context_payload(framework_descriptor)

    # Batch async
    with pytest.raises(RuntimeError, match="translator failure"):
        coro = _call_with_context(framework_descriptor, abatch_fn, ["err-async-batch"], context=ctx)
        assert inspect.isawaitable(coro), "Async batch method must return an awaitable"
        await coro  # noqa: PT018

    # Query async
    with pytest.raises(RuntimeError, match="translator failure"):
        coro = _call_with_context(framework_descriptor, aquery_fn, "err-async-query", context=ctx)
        assert inspect.isawaitable(coro), "Async query method must return an awaitable"
        await coro  # noqa: PT018

    assert calls, "attach_context was not called for async translator failures"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert str(ctx["operation"]).startswith("embedding_")
