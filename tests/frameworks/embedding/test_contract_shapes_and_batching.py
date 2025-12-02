# tests/frameworks/embedding/test_contract_shapes_and_batching.py

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

    Mirrors the construction pattern used in test_contract_interface_conformance.
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    adapter_cls = getattr(module, framework_descriptor.adapter_class)

    init_kwargs: dict[str, Any] = {"corpus_adapter": adapter}

    # Some adapters require a known embedding dimension up-front.
    if framework_descriptor.requires_embedding_dimension:
        init_kwargs.setdefault("embedding_dimension", 8)

    instance = adapter_cls(**init_kwargs)
    return instance


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


def _maybe_call_with_context(
    descriptor: EmbeddingFrameworkDescriptor,
    fn: Callable[..., Any],
    texts_or_text: Any,
) -> Any:
    """
    Call an embedding function, respecting descriptor.context_kwarg if present.

    Works for both batch (Sequence[str]) and query (str) methods.
    """
    if descriptor.context_kwarg:
        return fn(texts_or_text, **{descriptor.context_kwarg: {}})
    return fn(texts_or_text)


def _first_nonempty_row_dim(matrix: Sequence[Sequence[float]]) -> int | None:
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
    """
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
    """
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
    """
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
    """
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
    Mixed batches containing empty/whitespace texts should still return one
    embedding row per input text (no dropping, no crashes).
    """
    texts = ["real text", "", "   ", "another real text"]
    batch_fn = _get_batch_fn(embedding_adapter_instance, framework_descriptor)

    result = _maybe_call_with_context(framework_descriptor, batch_fn, texts)

    # Should preserve length and basic matrix shape.
    _assert_embedding_matrix_shape(result, expected_rows=len(texts))


def test_duplicate_texts_produce_identical_rows_within_same_batch(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    Duplicate texts in a single batch should produce identical embedding rows.

    This checks basic determinism + input order preservation *within* a single
    batch call, without relying on cross-call reproducibility.
    """
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
    """
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
    """
    texts = ["order-a", "order-b", "order-a"]
    batch_fn = _get_batch_fn(embedding_adapter_instance, framework_descriptor)

    result = _maybe_call_with_context(framework_descriptor, batch_fn, texts)
    _assert_embedding_matrix_shape(result, expected_rows=len(texts))

    row0, row1, row2 = result

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
    """
    if not framework_descriptor.supports_async:
        pytest.skip(f"Framework '{framework_descriptor.name}' does not declare async support")

    assert framework_descriptor.async_batch_method is not None
    abatch_fn = _get_method(embedding_adapter_instance, framework_descriptor.async_batch_method)

    batch_fn = _get_batch_fn(embedding_adapter_instance, framework_descriptor)

    texts = ["async-shape-a", "async-shape-b", "async-shape-c"]

    # Sync result
    sync_result = _maybe_call_with_context(framework_descriptor, batch_fn, texts)
    _assert_embedding_matrix_shape(sync_result, expected_rows=len(texts))
    sync_dim = _first_nonempty_row_dim(sync_result)
    assert sync_dim is not None and sync_dim > 0, (
        "Cannot determine non-zero embedding dimension from sync batch"
    )

    # Async result
    if framework_descriptor.context_kwarg:
        async_coro = abatch_fn(texts, **{framework_descriptor.context_kwarg: {}})
    else:
        async_coro = abatch_fn(texts)

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
    """
    if not framework_descriptor.supports_async:
        pytest.skip(f"Framework '{framework_descriptor.name}' does not declare async support")

    assert framework_descriptor.async_batch_method is not None
    abatch_fn = _get_method(embedding_adapter_instance, framework_descriptor.async_batch_method)

    size = 40
    texts = [f"async-large-{i}" for i in range(size)]

    if framework_descriptor.context_kwarg:
        async_coro = abatch_fn(texts, **{framework_descriptor.context_kwarg: {}})
    else:
        async_coro = abatch_fn(texts)

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

