# tests/frameworks/embedding/test_contract_shapes_and_batching.py

from __future__ import annotations

import asyncio
import importlib
import inspect
import warnings
from collections.abc import Mapping as ABCMapping
from collections.abc import Sequence
from typing import Any, Callable, Optional

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

    Mirrors the construction pattern used in test_contract_interface_conformance.

    
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

    if descriptor.availability_attr:
        try:
            module = importlib.import_module(descriptor.adapter_module)
        except Exception:
            # Import failing is acceptable as an "unavailable" signal.
            return

        flag = getattr(module, descriptor.availability_attr, None)
        # Either missing (treated as unavailable) or False.
        assert flag is None or bool(flag) is False


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


def _get_batch_fn(
    embedding_adapter_instance: Any,
    descriptor: EmbeddingFrameworkDescriptor,
) -> Callable[..., Any]:
    """Convenience helper to fetch the sync batch embedding method."""
    return _get_method(embedding_adapter_instance, descriptor.batch_method)


def _get_query_fn(
    embedding_adapter_instance: Any,
    descriptor: EmbeddingFrameworkDescriptor,
) -> Callable[..., Any]:
    """Convenience helper to fetch the sync query embedding method."""
    return _get_method(embedding_adapter_instance, descriptor.query_method)


def _merge_rich_context(descriptor: EmbeddingFrameworkDescriptor) -> Any:
    """
    Build a "rich" Mapping context for the framework using registry-provided sample_context
    plus extra nested keys to ensure adapters tolerate unknown fields.

    NOTE:
    - This is intentionally not schema-validated. Framework adapters are expected to be
      forward-compatible with unknown context keys (ignored or stored in attrs/metadata).
    """
    base = dict(descriptor.sample_context or {})
    base.update(
        {
            "request_id": base.get("request_id", "req-shapes-1"),
            "user_id": base.get("user_id", "user-shapes"),
            "tags": base.get("tags", ["conformance", "shapes", descriptor.name]),
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


def _maybe_call_with_context(
    descriptor: EmbeddingFrameworkDescriptor,
    fn: Callable[..., Any],
    texts_or_text: Any,
) -> Any:
    """
    Call an embedding function, respecting descriptor.context_kwarg if present.

    Uses registry sample_context (merged into a rich Mapping) to exercise context translation
    in a stable way. Works for both batch (Sequence[str]) and query (str) methods.
    """
    ctx = _merge_rich_context(descriptor)
    return _call_with_context(descriptor, fn, texts_or_text, context=ctx)


def _first_nonempty_row_dim(matrix: Sequence[Sequence[float]]) -> Optional[int]:
    """Return the dimension of the first non-empty row, or None if all rows are empty."""
    for row in matrix:
        if len(row) > 0:
            return len(row)
    return None


def _rows_close(a: Sequence[float], b: Sequence[float], tol: float = 1e-6) -> bool:
    """Numerical equality check for two embedding vectors with tolerance."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if abs(x - y) > tol:
            return False
    return True


def _empty_text_behavior_is_acceptable(exc: BaseException) -> bool:
    """
    Decide whether an exception represents an acceptable "strict empty text" behavior.

    Some frameworks/adapters intentionally reject empty/whitespace strings. This contract
    suite treats those rejections as acceptable *only when they are explicit*, rather than
    crashing in surprising ways.

    We keep this conservative:
      - TypeError / ValueError are accepted as strict input validation signals.

    Additionally:
      - CORPUS SDK adapters may raise corpus_sdk.embedding.embedding_base.BadRequest with a
        specific error code when empty/whitespace-only items would be dropped but alignment
        must be preserved. This is also an explicit, deterministic validation signal and is
        treated as acceptable strict behavior.
    """
    # Traditional strict validation surfaces
    if isinstance(exc, (TypeError, ValueError)):
        return True

    # Accept CORPUS SDK BadRequest for the "empty texts in batch" contract case.
    # Import lazily to avoid hard failures in environments where the SDK layout differs.
    try:
        from corpus_sdk.embedding.embedding_base import BadRequest as CorpusBadRequest  # type: ignore
    except Exception:
        CorpusBadRequest = None  # type: ignore[assignment]

    # Prefer structured identification via the error code. Fall back to string scanning
    # only when code is not reliably accessible.
    if getattr(exc, "code", None) == "BAD_BATCH_EMPTY_TEXTS":
        return True

    if CorpusBadRequest is not None and isinstance(exc, CorpusBadRequest):
        if getattr(exc, "code", None) == "BAD_BATCH_EMPTY_TEXTS":
            return True
        if "BAD_BATCH_EMPTY_TEXTS" in str(exc):
            return True

    if "BAD_BATCH_EMPTY_TEXTS" in str(exc):
        return True

    return False


# ---------------------------------------------------------------------------
# Core shape + batching contract tests (sync)
# ---------------------------------------------------------------------------


def test_batch_output_row_count_matches_input_length(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    For any batch of N texts, the embedding matrix should have exactly N rows.

    This is the core batch shape contract shared by all embedding framework
    adapters.

    
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    batch_fn = _get_batch_fn(embedding_adapter_instance, framework_descriptor)

    for size in (1, 2, 5):
        texts = [f"row-count-test-{i}" for i in range(size)]
        result = _maybe_call_with_context(framework_descriptor, batch_fn, texts)
        _assert_embedding_matrix_shape(result, expected_rows=size)


def test_all_rows_have_consistent_dimension(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    All rows in the batch output should have the same dimensionality.

    This should hold for small batches of non-empty texts. Zero-dimensional
    embeddings are considered invalid by this contract.

    
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    texts = ["dim-a", "dim-b", "dim-c", "dim-d"]
    batch_fn = _get_batch_fn(embedding_adapter_instance, framework_descriptor)

    result = _maybe_call_with_context(framework_descriptor, batch_fn, texts)
    _assert_embedding_matrix_shape(result, expected_rows=len(texts))

    dim = _first_nonempty_row_dim(result)
    assert dim is not None and dim > 0, "All embedding rows are empty or zero-dimensional"

    for row in result:
        if len(row) > 0:
            assert len(row) == dim, "Embedding rows have inconsistent dimensions"


def test_query_vector_dimension_matches_batch_rows(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    The embedding dimension used for batch embeddings should match the
    dimension of the query embedding for the same framework.

    
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    texts = ["dim-match-a", "dim-match-b", "dim-match-c"]
    batch_fn = _get_batch_fn(embedding_adapter_instance, framework_descriptor)
    query_fn = _get_query_fn(embedding_adapter_instance, framework_descriptor)

    batch_result = _maybe_call_with_context(framework_descriptor, batch_fn, texts)
    _assert_embedding_matrix_shape(batch_result, expected_rows=len(texts))

    query_result = _maybe_call_with_context(
        framework_descriptor,
        query_fn,
        "dim-match-query",
    )
    _assert_embedding_vector_shape(query_result)

    batch_dim = _first_nonempty_row_dim(batch_result)
    query_dim = len(query_result)

    assert batch_dim is not None and batch_dim > 0, (
        "Cannot determine non-zero embedding dimension from batch outputs"
    )
    assert query_dim > 0, "Query embedding has zero dimension"

    assert batch_dim == query_dim, (
        f"Batch row dimension ({batch_dim}) does not match query embedding "
        f"dimension ({query_dim})"
    )


def test_single_element_batch_matches_query_shape(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    A single-element batch should have a shape compatible with a query embedding.

    Typically: batch_result[0] should be the same length as embed_query(text).

    
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    text = "single-element-shape"
    batch_fn = _get_batch_fn(embedding_adapter_instance, framework_descriptor)
    query_fn = _get_query_fn(embedding_adapter_instance, framework_descriptor)

    batch_result = _maybe_call_with_context(
        framework_descriptor,
        batch_fn,
        [text],
    )
    _assert_embedding_matrix_shape(batch_result, expected_rows=1)

    query_result = _maybe_call_with_context(
        framework_descriptor,
        query_fn,
        text,
    )
    _assert_embedding_vector_shape(query_result)

    batch_vec = batch_result[0]
    assert len(batch_vec) > 0, "Single-element batch row has zero dimension"
    assert len(query_result) > 0, "Query embedding has zero dimension"

    assert len(batch_vec) == len(query_result), (
        "Single-element batch row dimension does not match query embedding dimension"
    )


def test_mixed_empty_and_nonempty_texts_preserve_batch_length(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    Mixed batches containing empty/whitespace texts should still behave deterministically.

    ACCEPTED BEHAVIORS:
    - Lenient frameworks/adapters: return one embedding row per input text (no dropping).
    - Strict frameworks/adapters: raise TypeError/ValueError explicitly signaling invalid input.

    
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    texts = ["real text", "", "   ", "another real text"]
    batch_fn = _get_batch_fn(embedding_adapter_instance, framework_descriptor)

    try:
        result = _maybe_call_with_context(framework_descriptor, batch_fn, texts)
    except Exception as exc:  # noqa: BLE001
        assert _empty_text_behavior_is_acceptable(exc), (
            f"{framework_descriptor.name}: mixed empty/nonempty batch raised unexpected "
            f"{type(exc).__name__}: {exc}"
        )
        return

    # Lenient path: preserve length and basic matrix shape.
    _assert_embedding_matrix_shape(result, expected_rows=len(texts))


def test_duplicate_texts_produce_identical_rows_within_same_batch(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    Duplicate texts in a single batch should produce identical embedding rows.

    This checks basic determinism + input order preservation *within* a single
    batch call, without relying on cross-call reproducibility.

    
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    texts = ["dup-sample", "other-sample", "dup-sample"]
    batch_fn = _get_batch_fn(embedding_adapter_instance, framework_descriptor)

    result = _maybe_call_with_context(framework_descriptor, batch_fn, texts)
    _assert_embedding_matrix_shape(result, expected_rows=len(texts))

    first, _, third = result
    assert len(first) > 0 and len(third) > 0, (
        "Duplicate rows are zero-dimensional; cannot assert equality reliably"
    )

    assert _rows_close(first, third), (
        "Duplicate texts within the same batch did not produce numerically "
        "identical embeddings within tolerance"
    )


def test_large_batch_shape_is_respected(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    Larger batches (e.g., ~50 items) should still conform to shape expectations
    and not fragment into nested batches at the API surface.

    
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    size = 50
    texts = [f"large-batch-{i}" for i in range(size)]
    batch_fn = _get_batch_fn(embedding_adapter_instance, framework_descriptor)

    result = _maybe_call_with_context(framework_descriptor, batch_fn, texts)

    _assert_embedding_matrix_shape(result, expected_rows=size)

    dim = _first_nonempty_row_dim(result)
    assert dim is not None and dim > 0, (
        "All embedding rows empty or zero-dimensional in large batch"
    )

    for row in result:
        if len(row) > 0:
            assert len(row) == dim, "Inconsistent row dimension in large batch"


def test_batch_is_order_preserving_for_duplicates(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    Use duplicates to assert that positions are preserved.

    For texts = ["x", "y", "x"], we expect:
    - Same number of rows as inputs
    - Row 0 and row 2 correspond to the same text and should be equal

    This doesn't fully prove order preservation for *all* inputs, but catches
    obvious reordering / bucketing mistakes in framework adapters.

    
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    texts = ["order-a", "order-b", "order-a"]
    batch_fn = _get_batch_fn(embedding_adapter_instance, framework_descriptor)

    result = _maybe_call_with_context(framework_descriptor, batch_fn, texts)
    _assert_embedding_matrix_shape(result, expected_rows=len(texts))

    row0, _row1, row2 = result

    assert len(row0) > 0 and len(row2) > 0, (
        "Duplicate rows are zero-dimensional; cannot assert order-preserving equality"
    )

    assert _rows_close(row0, row2), (
        "Duplicate first/last texts did not yield numerically identical embeddings "
        "within tolerance; adapter may be reordering inputs"
    )


# ---------------------------------------------------------------------------
# Async variants (shape/batching parity)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_batch_shape_matches_sync_when_supported(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    When async is supported, the async batch method should return shapes
    consistent with the sync batch method for the same inputs.

    We only assert shape parity (row count + dimensions), not exact numeric
    equality across calls.

    
    - If framework is unavailable, validate the unavailable contract and return.
    - If async is not supported, validate that and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    if not framework_descriptor.supports_async:
        assert framework_descriptor.async_batch_method is None
        assert framework_descriptor.async_query_method is None
        return

    assert framework_descriptor.async_batch_method is not None
    abatch_fn = _get_method(embedding_adapter_instance, framework_descriptor.async_batch_method)

    batch_fn = _get_batch_fn(embedding_adapter_instance, framework_descriptor)

    texts = ["async-shape-a", "async-shape-b", "async-shape-c"]

    # Sync result
    #
    # IMPORTANT:
    # Many framework adapters intentionally refuse calling sync APIs from within an
    # active asyncio event loop to avoid deadlocks/hangs. Because this test itself
    # is async, we compute the sync reference result in a worker thread.
    sync_result = await asyncio.to_thread(
        _maybe_call_with_context,
        framework_descriptor,
        batch_fn,
        texts,
    )

    _assert_embedding_matrix_shape(sync_result, expected_rows=len(texts))
    sync_dim = _first_nonempty_row_dim(sync_result)
    assert sync_dim is not None and sync_dim > 0, (
        "Cannot determine non-zero embedding dimension from sync batch"
    )

    # Async result (use the same context strategy for parity)
    ctx = _merge_rich_context(framework_descriptor)
    async_coro = _call_with_context(framework_descriptor, abatch_fn, texts, context=ctx)

    assert inspect.isawaitable(async_coro), "Async batch method must return an awaitable"
    async_result = await async_coro

    _assert_embedding_matrix_shape(async_result, expected_rows=len(texts))
    async_dim = _first_nonempty_row_dim(async_result)
    assert async_dim is not None and async_dim > 0, (
        "Cannot determine non-zero embedding dimension from async batch"
    )

    assert sync_dim == async_dim, (
        f"Async batch dimension ({async_dim}) does not match sync batch dimension ({sync_dim})"
    )


@pytest.mark.asyncio
async def test_async_large_batch_shape_is_respected(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    Async large batches should also obey the same shape and row-count invariants
    as sync batches.

    
    - If framework is unavailable, validate the unavailable contract and return.
    - If async is not supported, validate that and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    if not framework_descriptor.supports_async:
        assert framework_descriptor.async_batch_method is None
        assert framework_descriptor.async_query_method is None
        return

    assert framework_descriptor.async_batch_method is not None
    abatch_fn = _get_method(embedding_adapter_instance, framework_descriptor.async_batch_method)

    size = 40
    texts = [f"async-large-{i}" for i in range(size)]

    ctx = _merge_rich_context(framework_descriptor)
    async_coro = _call_with_context(framework_descriptor, abatch_fn, texts, context=ctx)

    assert inspect.isawaitable(async_coro), "Async batch method must return an awaitable"
    result = await async_coro

    _assert_embedding_matrix_shape(result, expected_rows=size)

    dim = _first_nonempty_row_dim(result)
    assert dim is not None and dim > 0, (
        "All embedding rows empty or zero-dimensional in async large batch"
    )

    for row in result:
        if len(row) > 0:
            assert len(row) == dim, "Inconsistent row dimension in async large batch"
