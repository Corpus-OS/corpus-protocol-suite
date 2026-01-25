# tests/frameworks/embedding/test_contract_context_and_error_context.py

from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping as ABCMapping
from collections.abc import Sequence
from typing import Any, Callable, Optional, Tuple

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


@pytest.fixture
def embedding_adapter_instance(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    adapter: Any,
) -> Any:
    """
    Construct a concrete framework adapter instance for the given descriptor.

    Mirrors the construction pattern used in the other embedding contract tests.

    IMPORTANT POLICY (no pytest.skip):
    - If a framework is unavailable, this fixture returns None and tests must
      treat that as a validated pass condition (not a skip).
    """
    if not framework_descriptor.is_available():
        return None

    module = importlib.import_module(framework_descriptor.adapter_module)
    adapter_cls = getattr(module, framework_descriptor.adapter_class)

    init_kwargs: dict[str, Any] = {"corpus_adapter": adapter}

    # Some adapters require a known embedding dimension up-front.
    if framework_descriptor.requires_embedding_dimension:
        kw = framework_descriptor.embedding_dimension_kwarg or "embedding_dimension"
        init_kwargs.setdefault(kw, 8)

    instance = adapter_cls(**init_kwargs)
    return instance


@pytest.fixture
def failing_corpus_adapter() -> Any:
    """
    A minimal corpus adapter whose embed() always fails.

    Used only for error-context tests to ensure the decorators invoke
    attach_context() and propagate the exception.

    IMPORTANT:
    - Implemented as async to exercise async-first protocol paths.
    - Includes embed_batch as a best-effort fallback for translators that may prefer it.
    """

    class FailingEmbeddingAdapter:
        async def embed(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("intentional failure from failing adapter")

        async def embed_batch(self, *args: Any, **kwargs: Any) -> Any:
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

    - Must be a non-string sequence
    - Must have expected_rows rows
    - Each row must be a non-string sequence
    - Values (if present) must be numeric
    """
    assert isinstance(result, Sequence) and not isinstance(
        result, (str, bytes)
    ), f"Expected sequence (non-str), got {type(result).__name__}"
    assert len(result) == expected_rows, f"Expected {expected_rows} rows, got {len(result)}"

    for row in result:
        assert isinstance(row, Sequence) and not isinstance(
            row, (str, bytes)
        ), f"Row is not a sequence (non-str): {type(row).__name__}"
        for val in row:
            assert isinstance(val, (int, float)), f"Embedding value is not numeric: {val!r}"


def _assert_embedding_vector_shape(result: Any) -> None:
    """
    Validate that a result looks like a 1D embedding vector.

    - Must be a non-string sequence
    - Values (if present) must be numeric
    """
    assert isinstance(result, Sequence) and not isinstance(
        result, (str, bytes)
    ), f"Expected sequence (non-str), got {type(result).__name__}"
    for val in result:
        assert isinstance(val, (int, float)), f"Embedding value is not numeric: {val!r}"


def _merge_rich_context(
    descriptor: EmbeddingFrameworkDescriptor,
) -> Any:
    """
    Build a "rich" Mapping context for the framework using registry-provided sample_context
    plus extra nested keys to ensure adapters tolerate unknown fields.
    """
    base = dict(descriptor.sample_context or {})
    base.update(
        {
            "request_id": base.get("request_id", "req-123"),
            "user_id": base.get("user_id", "user-abc"),
            "tags": base.get("tags", ["test", descriptor.name]),
            "nested": {"key": "value", "depth": 2, "framework": descriptor.name},
        }
    )
    return base


def _call_with_context(
    descriptor: EmbeddingFrameworkDescriptor,
    fn: Callable[..., Any],
    texts_or_text: Any,
    context: Any,
) -> Any:
    """
    Call an embedding function with context in a robust, framework-agnostic way.

    Primary strategy:
      - If descriptor.context_kwarg is set, pass {context_kwarg: context}.

    Compatibility fallback:
      - If that raises TypeError due to an unexpected keyword argument, and context is a Mapping,
        retry by spreading the mapping into kwargs (useful for BaseEmbedding-style **kwargs surfaces).

    This approach avoids test skips while remaining resilient to framework method signature shapes.
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


async def _maybe_await(value: Any) -> Any:
    """Await an awaitable value if needed; otherwise return it directly."""
    if inspect.isawaitable(value):
        return await value
    return value


def _build_error_wrapped_adapter_instance(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    failing_corpus_adapter: Any,
) -> Any:
    """
    Construct a framework adapter instance wired to a failing corpus adapter.

    Used only for error-context tests (we expect calls to raise).

    IMPORTANT POLICY (no pytest.skip):
    - If framework is unavailable, returns None and tests assert the unavailable contract.
    """
    if not framework_descriptor.is_available():
        return None

    module = importlib.import_module(framework_descriptor.adapter_module)
    adapter_cls = getattr(module, framework_descriptor.adapter_class)

    init_kwargs: dict[str, Any] = {"corpus_adapter": failing_corpus_adapter}

    if framework_descriptor.requires_embedding_dimension:
        kw = framework_descriptor.embedding_dimension_kwarg or "embedding_dimension"
        init_kwargs.setdefault(kw, 8)

    return adapter_cls(**init_kwargs)


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
        # If this import fails in a minimal environment, we still keep the module-local patch.
        pass


def _assert_unavailable_contract(descriptor: EmbeddingFrameworkDescriptor) -> None:
    """
    Validate that an unavailable framework descriptor is behaving as expected.

    The test suite policy is "no skip": when unavailable, tests must pass by
    asserting correct unavailability signaling.
    """
    assert descriptor.is_available() is False
    # If availability_attr is set, adapter module should generally import and expose the flag.
    # If not, the framework may be unavailable due to import error or missing attr.
    if descriptor.availability_attr:
        try:
            module = importlib.import_module(descriptor.adapter_module)
        except Exception:
            # Import failing is acceptable as an "unavailable" signal.
            return
        flag = getattr(module, descriptor.availability_attr, None)
        # Either missing (treated as unavailable) or False.
        assert flag is None or bool(flag) is False


def _assert_error_context_minimum(
    descriptor: EmbeddingFrameworkDescriptor,
    ctx: dict[str, Any],
) -> None:
    """
    Assert minimum error-context fields for conformance alignment.

    We intentionally enforce:
      - framework
      - operation
      - error_codes

    Operation name is framework-specific; we assert it is one of the expected
    embedding operation names or starts with "embedding_".
    """
    assert "framework" in ctx
    assert "operation" in ctx
    assert "error_codes" in ctx

    # Framework identity should generally match descriptor.name (or a stable label).
    # Keep this tolerant: adapters may choose framework labels that differ slightly.
    assert isinstance(ctx["framework"], str) and ctx["framework"]

    op = ctx["operation"]
    assert isinstance(op, str) and op

    # Accept both styles:
    # - embedding_<op> (common in several adapters)
    # - embedding_query / embedding_documents / embedding_text / embedding_texts (SK/LI patterns)
    allowed_exact = {
        "embedding_query",
        "embedding_documents",
        "embedding_text",
        "embedding_texts",
        "embedding_text_batch",
        "embedding_context_build",
        "context_build",
        "embedding_capabilities",
        "embedding_health",
        "capabilities",
        "health",
    }
    assert op.startswith("embedding_") or op in allowed_exact, (
        f"{descriptor.name}: unexpected operation name {op!r}; "
        f"expected prefix 'embedding_' or one of {sorted(allowed_exact)}"
    )


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

    IMPORTANT POLICY (no pytest.skip):
    - If framework is unavailable, validate the unavailable contract and return.
    - If framework does not declare a context_kwarg, validate that fact and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    if not framework_descriptor.context_kwarg:
        assert framework_descriptor.context_kwarg is None
        return

    batch_method = _get_method(embedding_adapter_instance, framework_descriptor.batch_method)
    query_method = _get_method(embedding_adapter_instance, framework_descriptor.query_method)

    rich_context = _merge_rich_context(framework_descriptor)

    texts = ["ctx-rich-alpha", "ctx-rich-beta"]
    query_text = "ctx-rich-query"

    batch_result = _call_with_context(
        framework_descriptor,
        batch_method,
        texts,
        context=rich_context,
    )
    _assert_embedding_matrix_shape(batch_result, expected_rows=len(texts))

    query_result = _call_with_context(
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

    IMPORTANT POLICY (no pytest.skip):
    - If framework is unavailable, validate the unavailable contract and return.
    - If framework does not declare a context_kwarg, validate that fact and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    if not framework_descriptor.context_kwarg:
        assert framework_descriptor.context_kwarg is None
        return

    batch_method = _get_method(embedding_adapter_instance, framework_descriptor.batch_method)
    query_method = _get_method(embedding_adapter_instance, framework_descriptor.query_method)

    texts = ["ctx-invalid-alpha", "ctx-invalid-beta"]
    query_text = "ctx-invalid-query"

    # Intentionally wrong types for context.
    invalid_context_for_batch = "not-a-mapping"
    invalid_context_for_query = 12345

    batch_result = _call_with_context(
        framework_descriptor,
        batch_method,
        texts,
        context=invalid_context_for_batch,
    )
    _assert_embedding_matrix_shape(batch_result, expected_rows=len(texts))

    query_result = _call_with_context(
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

    IMPORTANT POLICY (no pytest.skip):
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    batch_method = _get_method(embedding_adapter_instance, framework_descriptor.batch_method)
    query_method = _get_method(embedding_adapter_instance, framework_descriptor.query_method)

    texts = ["ctx-optional-alpha", "ctx-optional-beta"]
    query_text = "ctx-optional-query"

    batch_result = batch_method(texts)
    _assert_embedding_matrix_shape(batch_result, expected_rows=len(texts))

    query_result = query_method(query_text)
    _assert_embedding_vector_shape(query_result)


def test_alias_methods_exist_and_behave_consistently_when_declared(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    If a framework declares aliases in the registry, those alias methods should:
    - exist on the adapter instance
    - be callable
    - return valid shapes when called with the same input as the primary method

    We do not require exact float equality; shape + numeric contract is sufficient.

    IMPORTANT POLICY (no pytest.skip):
    - If framework is unavailable, validate the unavailable contract and return.
    - If no aliases are declared, validate that fact and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    if not framework_descriptor.aliases:
        assert framework_descriptor.aliases is None
        return

    rich_context = _merge_rich_context(framework_descriptor)

    # Choose representative inputs for both batch and query aliases.
    texts = ["alias-alpha", "alias-beta"]
    query_text = "alias-query"

    for alias_name, primary_name in framework_descriptor.aliases.items():
        alias_fn = _get_method(embedding_adapter_instance, alias_name)
        primary_fn = _get_method(embedding_adapter_instance, primary_name)

        # Determine whether this alias looks like a batch or query surface by name.
        is_batch = "document" in alias_name or "embeddings" in alias_name or "texts" in alias_name

        if is_batch:
            alias_out = _call_with_context(
                framework_descriptor,
                alias_fn,
                texts,
                context=rich_context,
            )
            primary_out = _call_with_context(
                framework_descriptor,
                primary_fn,
                texts,
                context=rich_context,
            )
            _assert_embedding_matrix_shape(alias_out, expected_rows=len(texts))
            _assert_embedding_matrix_shape(primary_out, expected_rows=len(texts))
        else:
            alias_out = _call_with_context(
                framework_descriptor,
                alias_fn,
                query_text,
                context=rich_context,
            )
            primary_out = _call_with_context(
                framework_descriptor,
                primary_fn,
                query_text,
                context=rich_context,
            )
            _assert_embedding_vector_shape(alias_out)
            _assert_embedding_vector_shape(primary_out)


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

    IMPORTANT POLICY (no pytest.skip):
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    module = importlib.import_module(framework_descriptor.adapter_module)

    calls: list[tuple[BaseException, dict[str, Any]]] = []
    _patch_attach_context(module, monkeypatch, calls)

    instance = _build_error_wrapped_adapter_instance(
        framework_descriptor,
        failing_corpus_adapter,
    )
    assert instance is not None

    batch_method = _get_method(instance, framework_descriptor.batch_method)

    with pytest.raises(RuntimeError, match="intentional failure"):
        if framework_descriptor.context_kwarg:
            batch_method(["err-batch"], **{framework_descriptor.context_kwarg: {}})
        else:
            batch_method(["err-batch"])

    assert calls, "attach_context was not called on sync batch failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    _assert_error_context_minimum(framework_descriptor, ctx)


def test_error_context_is_attached_on_sync_query_failure(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    failing_corpus_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Same as the batch failure test, but for the sync query operation.

    Ensures the query path is also wrapped by the error-context decorator.

    IMPORTANT POLICY (no pytest.skip):
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    module = importlib.import_module(framework_descriptor.adapter_module)

    calls: list[tuple[BaseException, dict[str, Any]]] = []
    _patch_attach_context(module, monkeypatch, calls)

    instance = _build_error_wrapped_adapter_instance(
        framework_descriptor,
        failing_corpus_adapter,
    )
    assert instance is not None

    query_method = _get_method(instance, framework_descriptor.query_method)

    with pytest.raises(RuntimeError, match="intentional failure"):
        if framework_descriptor.context_kwarg:
            query_method("err-query", **{framework_descriptor.context_kwarg: {}})
        else:
            query_method("err-query")

    assert calls, "attach_context was not called on sync query failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    _assert_error_context_minimum(framework_descriptor, ctx)


@pytest.mark.asyncio
async def test_error_context_is_attached_on_async_batch_failure_when_supported(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    failing_corpus_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async is supported, async batch failures should also go through
    the error-context decorator and call attach_context().

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

    module = importlib.import_module(framework_descriptor.adapter_module)

    calls: list[tuple[BaseException, dict[str, Any]]] = []
    _patch_attach_context(module, monkeypatch, calls)

    instance = _build_error_wrapped_adapter_instance(
        framework_descriptor,
        failing_corpus_adapter,
    )
    assert instance is not None

    abatch_method = _get_method(instance, framework_descriptor.async_batch_method)

    with pytest.raises(RuntimeError, match="intentional failure"):
        if framework_descriptor.context_kwarg:
            coro = abatch_method(["err-abatch"], **{framework_descriptor.context_kwarg: {}})
        else:
            coro = abatch_method(["err-abatch"])

        assert inspect.isawaitable(coro), (
            f"{framework_descriptor.name}: async batch method "
            f"{framework_descriptor.async_batch_method!r} must return an awaitable"
        )
        await coro  # noqa: PT018

    assert calls, "attach_context was not called on async batch failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    _assert_error_context_minimum(framework_descriptor, ctx)


@pytest.mark.asyncio
async def test_error_context_is_attached_on_async_query_failure_when_supported(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    failing_corpus_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When async is supported, async query failures should also go through
    the error-context decorator and call attach_context().

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

    assert framework_descriptor.async_query_method is not None

    module = importlib.import_module(framework_descriptor.adapter_module)

    calls: list[tuple[BaseException, dict[str, Any]]] = []
    _patch_attach_context(module, monkeypatch, calls)

    instance = _build_error_wrapped_adapter_instance(
        framework_descriptor,
        failing_corpus_adapter,
    )
    assert instance is not None

    aquery_method = _get_method(instance, framework_descriptor.async_query_method)

    with pytest.raises(RuntimeError, match="intentional failure"):
        if framework_descriptor.context_kwarg:
            coro = aquery_method("err-aquery", **{framework_descriptor.context_kwarg: {}})
        else:
            coro = aquery_method("err-aquery")

        assert inspect.isawaitable(coro), (
            f"{framework_descriptor.name}: async query method "
            f"{framework_descriptor.async_query_method!r} must return an awaitable"
        )
        await coro  # noqa: PT018

    assert calls, "attach_context was not called on async query failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    _assert_error_context_minimum(framework_descriptor, ctx)
