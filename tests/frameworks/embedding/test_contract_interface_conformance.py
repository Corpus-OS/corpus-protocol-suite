# tests/frameworks/embedding/test_contract_interface_conformance.py

from __future__ import annotations

import asyncio
import importlib
import inspect
from collections.abc import Mapping, Sequence
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

    This uses the registry metadata to import the adapter class and instantiate
    it with the *generic* Corpus adapter provided by the top-level pytest
    plugin (see conftest.py).

    The adapter class is expected to implement the EmbeddingProtocolV1 surface
    that the framework-specific adapters wrap.
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    adapter_cls = getattr(module, framework_descriptor.adapter_class)

    # All embedding framework adapters take a corpus_adapter implementing the
    # embedding protocol surface. The global `adapter` fixture is pluggable via
    # CORPUS_ADAPTER and can point to the user's real adapter implementation.
    init_kwargs: dict[str, Any] = {"corpus_adapter": adapter}

    # Some adapters require a known embedding dimension up-front.
    if framework_descriptor.requires_embedding_dimension:
        init_kwargs.setdefault("embedding_dimension", 8)

    # Additional framework-specific kwargs can be added here if needed.

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
    """Validate that a result looks like a 2D embedding matrix."""
    assert isinstance(result, Sequence), f"Expected sequence, got {type(result).__name__}"
    assert len(result) == expected_rows, f"Expected {expected_rows} rows, got {len(result)}"
    for row in result:
        assert isinstance(row, Sequence), f"Row is not a sequence: {type(row).__name__}"
        # Allow empty rows, just enforce numeric-ish elements if present
        for val in row:
            assert isinstance(val, (int, float)), f"Embedding value is not numeric: {val!r}"


def _assert_embedding_vector_shape(result: Any) -> None:
    """Validate that a result looks like a 1D embedding vector."""
    assert isinstance(result, Sequence), f"Expected sequence, got {type(result).__name__}"
    for val in result:
        assert isinstance(val, (int, float)), f"Embedding value is not numeric: {val!r}"


def _run_async_if_needed(coro):
    """
    Run an async coroutine, handling existing event loops gracefully.
    """
    try:
        # Try to use asyncio.run() (creates new loop)
        return asyncio.run(coro)
    except RuntimeError:
        # If there's already a running loop (e.g., in pytest-asyncio test)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Core contract tests
# ---------------------------------------------------------------------------


def test_can_instantiate_framework_adapter(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    Each registered framework descriptor should be instantiable with the
    pluggable Corpus adapter and any inferred kwargs.
    """
    # Basic sanity: instance has the methods the descriptor claims exist.
    _get_method(embedding_adapter_instance, framework_descriptor.batch_method)
    _get_method(embedding_adapter_instance, framework_descriptor.query_method)

    if framework_descriptor.supports_async:
        assert framework_descriptor.async_batch_method is not None
        assert framework_descriptor.async_query_method is not None
        _get_method(embedding_adapter_instance, framework_descriptor.async_batch_method)
        _get_method(embedding_adapter_instance, framework_descriptor.async_query_method)


def test_async_methods_exist_when_supports_async_true(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    Ensure that when supports_async=True, both async methods actually exist.
    This catches registry-descriptor mismatches.
    """
    if not framework_descriptor.supports_async:
        pytest.skip("Framework does not declare async support")

    # These should not be None per registry
    assert framework_descriptor.async_batch_method is not None
    assert framework_descriptor.async_query_method is not None

    # And the methods should exist on the instance
    assert hasattr(embedding_adapter_instance, framework_descriptor.async_batch_method)
    assert hasattr(embedding_adapter_instance, framework_descriptor.async_query_method)

    # And be callable
    batch_method = getattr(embedding_adapter_instance, framework_descriptor.async_batch_method)
    query_method = getattr(embedding_adapter_instance, framework_descriptor.async_query_method)
    assert callable(batch_method)
    assert callable(query_method)


def test_sync_embedding_interface_conformance(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    Validate that sync batch and query methods accept simple text input and
    return embedding shapes that look like vectors / matrices of numbers.
    """
    texts = ["alpha text", "beta text"]
    query_text = "gamma query"

    batch_fn = _get_method(embedding_adapter_instance, framework_descriptor.batch_method)
    query_fn = _get_method(embedding_adapter_instance, framework_descriptor.query_method)

    # Call batch embedding
    if framework_descriptor.context_kwarg:
        batch_result = batch_fn(texts, **{framework_descriptor.context_kwarg: {}})
    else:
        batch_result = batch_fn(texts)
    _assert_embedding_matrix_shape(batch_result, expected_rows=len(texts))

    # Call query embedding
    if framework_descriptor.context_kwarg:
        query_result = query_fn(query_text, **{framework_descriptor.context_kwarg: {}})
    else:
        query_result = query_fn(query_text)
    _assert_embedding_vector_shape(query_result)


def test_single_element_batch(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    Test that single-element batches work correctly.
    """
    texts = ["single text"]

    batch_fn = _get_method(embedding_adapter_instance, framework_descriptor.batch_method)

    if framework_descriptor.context_kwarg:
        result = batch_fn(texts, **{framework_descriptor.context_kwarg: {}})
    else:
        result = batch_fn(texts)

    _assert_embedding_matrix_shape(result, expected_rows=1)


def test_empty_batch_handling(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    Test how frameworks handle empty batch requests.
    """
    batch_fn = _get_method(embedding_adapter_instance, framework_descriptor.batch_method)

    if framework_descriptor.context_kwarg:
        result = batch_fn([], **{framework_descriptor.context_kwarg: {}})
    else:
        result = batch_fn([])

    # Should return empty list, not crash
    assert isinstance(result, Sequence)
    assert len(result) == 0


@pytest.mark.asyncio
async def test_async_embedding_interface_conformance(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    Validate that async batch and query methods (when declared) accept text
    input and return embedding shapes compatible with the sync API.
    """
    if not framework_descriptor.supports_async:
        pytest.skip(f"Framework '{framework_descriptor.name}' does not declare async support")

    assert framework_descriptor.async_batch_method is not None
    assert framework_descriptor.async_query_method is not None

    texts = ["alpha async", "beta async"]
    query_text = "gamma async"

    abatch_fn = _get_method(embedding_adapter_instance, framework_descriptor.async_batch_method)
    aquery_fn = _get_method(embedding_adapter_instance, framework_descriptor.async_query_method)

    # Batch async
    if framework_descriptor.context_kwarg:
        batch_coro = abatch_fn(texts, **{framework_descriptor.context_kwarg: {}})
    else:
        batch_coro = abatch_fn(texts)

    batch_result = await batch_coro
    _assert_embedding_matrix_shape(batch_result, expected_rows=len(texts))

    # Query async
    if framework_descriptor.context_kwarg:
        query_coro = aquery_fn(query_text, **{framework_descriptor.context_kwarg: {}})
    else:
        query_coro = aquery_fn(query_text)

    query_result = await query_coro
    _assert_embedding_vector_shape(query_result)


def test_context_kwarg_is_accepted_when_declared(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    If a context_kwarg is declared in the descriptor, the corresponding
    embedding methods should accept that kwarg without raising TypeError.
    """
    if not framework_descriptor.context_kwarg:
        pytest.skip(f"Framework '{framework_descriptor.name}' does not declare a context_kwarg")

    ctx_kw = framework_descriptor.context_kwarg
    texts = ["ctx alpha", "ctx beta"]
    query_text = "ctx gamma"

    batch_fn = _get_method(embedding_adapter_instance, framework_descriptor.batch_method)
    query_fn = _get_method(embedding_adapter_instance, framework_descriptor.query_method)

    # Should not raise TypeError
    batch_result = batch_fn(texts, **{ctx_kw: {"test": "value"}})
    query_result = query_fn(query_text, **{ctx_kw: {"test": "value"}})

    _assert_embedding_matrix_shape(batch_result, expected_rows=len(texts))
    _assert_embedding_vector_shape(query_result)


def test_embedding_dimension_when_required(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    adapter: Any,
) -> None:
    """
    Test that frameworks requiring embedding_dimension enforce it.
    """
    if not framework_descriptor.requires_embedding_dimension:
        pytest.skip("Framework does not require embedding dimension")

    module = importlib.import_module(framework_descriptor.adapter_module)
    adapter_cls = getattr(module, framework_descriptor.adapter_class)

    # Create a minimal adapter WITHOUT get_embedding_dimension() to test enforcement
    from corpus_sdk.embedding.embedding_base import BatchEmbedResult, EmbeddingVector
    
    class MinimalAdapter:
        def embed(self, *args: Any, **kwargs: Any) -> list[list[float]]:
            # Return a batch-like result
            batch_spec = args[0] if args else kwargs.get('batch_spec', None)
            if batch_spec and hasattr(batch_spec, 'texts'):
                return [[0.0] * 8] * len(batch_spec.texts)
            return [[0.0] * 8]
        
        async def embed_batch(self, *args: Any, **kwargs: Any) -> BatchEmbedResult:
            # Return a BatchEmbedResult with proper EmbeddingVector objects
            batch_spec = args[0] if args else kwargs.get('batch_spec', None)
            if batch_spec and hasattr(batch_spec, 'texts'):
                embeddings = [
                    EmbeddingVector(
                        vector=[0.0] * 8,
                        text=text,
                        model="minimal-test-model",
                        dimensions=8,
                        index=i
                    )
                    for i, text in enumerate(batch_spec.texts)
                ]
                return BatchEmbedResult(
                    embeddings=embeddings,
                    model="minimal-test-model",
                    total_texts=len(batch_spec.texts)
                )
            return BatchEmbedResult(
                embeddings=[EmbeddingVector(vector=[0.0] * 8, text="", model="minimal-test-model", dimensions=8)],
                model="minimal-test-model",
                total_texts=1
            )
        
        async def aembed(self, *args: Any, **kwargs: Any) -> list[list[float]]:
            # Return a batch-like result
            batch_spec = args[0] if args else kwargs.get('batch_spec', None)
            if batch_spec and hasattr(batch_spec, 'texts'):
                return [[0.0] * 8] * len(batch_spec.texts)
            return [[0.0] * 8]
        
        async def aembed_batch(self, *args: Any, **kwargs: Any) -> BatchEmbedResult:
            # Return a BatchEmbedResult with proper EmbeddingVector objects
            batch_spec = args[0] if args else kwargs.get('batch_spec', None)
            if batch_spec and hasattr(batch_spec, 'texts'):
                embeddings = [
                    EmbeddingVector(
                        vector=[0.0] * 8,
                        text=text,
                        model="minimal-test-model",
                        dimensions=8,
                        index=i
                    )
                    for i, text in enumerate(batch_spec.texts)
                ]
                return BatchEmbedResult(
                    embeddings=embeddings,
                    model="minimal-test-model",
                    total_texts=len(batch_spec.texts)
                )
            return BatchEmbedResult(
                embeddings=[EmbeddingVector(vector=[0.0] * 8, text="", model="minimal-test-model", dimensions=8)],
                model="minimal-test-model",
                total_texts=1
            )

    minimal_adapter = MinimalAdapter()

    # Should fail without embedding_dimension when adapter lacks get_embedding_dimension()
    with pytest.raises((TypeError, ValueError)):
        adapter_cls(corpus_adapter=minimal_adapter)

    # Should succeed with embedding_dimension
    instance = adapter_cls(corpus_adapter=minimal_adapter, embedding_dimension=8)
    assert instance is not None
    
    # Should also succeed with the regular adapter that HAS get_embedding_dimension()
    instance2 = adapter_cls(corpus_adapter=adapter)
    assert instance2 is not None
    
    # Verify methods work
    batch_fn = _get_method(instance, framework_descriptor.batch_method)
    if framework_descriptor.context_kwarg:
        result = batch_fn(["test"], **{framework_descriptor.context_kwarg: {}})
    else:
        result = batch_fn(["test"])
    _assert_embedding_matrix_shape(result, expected_rows=1)


def test_method_signatures_consistent(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    Verify that sync and async methods have consistent signatures
    (same parameters except maybe the context kwarg).
    """
    sync_batch = _get_method(embedding_adapter_instance, framework_descriptor.batch_method)
    sync_query = _get_method(embedding_adapter_instance, framework_descriptor.query_method)

    if framework_descriptor.supports_async:
        async_batch = _get_method(embedding_adapter_instance, framework_descriptor.async_batch_method)
        async_query = _get_method(embedding_adapter_instance, framework_descriptor.async_query_method)

        # Compare signatures (excluding self)
        sync_batch_sig = inspect.signature(sync_batch)
        async_batch_sig = inspect.signature(async_batch)

        # Should have same parameters except maybe return annotation
        sync_params = list(sync_batch_sig.parameters.keys())[1:]  # Skip self
        async_params = list(async_batch_sig.parameters.keys())[1:]  # Skip self

        assert sync_params == async_params, "Sync/async batch methods have different parameters"

        # Same for query methods
        sync_query_sig = inspect.signature(sync_query)
        async_query_sig = inspect.signature(async_query)

        sync_query_params = list(sync_query_sig.parameters.keys())[1:]  # Skip self
        async_query_params = list(async_query_sig.parameters.keys())[1:]  # Skip self

        assert sync_query_params == async_query_params, "Sync/async query methods have different parameters"


# ---------------------------------------------------------------------------
# Capabilities / health passthrough contract
# ---------------------------------------------------------------------------


def test_capabilities_contract_if_declared(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    If a framework declares has_capabilities=True, it should expose a
    capabilities() method returning a mapping. Async variants (when present)
    should behave similarly.
    """
    if not framework_descriptor.has_capabilities:
        pytest.skip(f"Framework '{framework_descriptor.name}' does not expose capabilities")

    # Sync capabilities
    capabilities = getattr(embedding_adapter_instance, "capabilities", None)
    assert callable(capabilities), "capabilities() method is missing"
    caps_result = capabilities()
    assert isinstance(caps_result, Mapping), "capabilities() should return a mapping"

    # Async capabilities (best-effort)
    async_caps = getattr(embedding_adapter_instance, "acapabilities", None)
    if async_caps is not None and callable(async_caps):
        acaps_result = _run_async_if_needed(async_caps())
        assert isinstance(acaps_result, Mapping), "acapabilities() should return a mapping"


def test_health_contract_if_declared(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    If a framework declares has_health=True, it should expose a health()
    method returning a mapping. Async variants (when present) should behave
    similarly.
    """
    if not framework_descriptor.has_health:
        pytest.skip(f"Framework '{framework_descriptor.name}' does not expose health")

    # Sync health
    health = getattr(embedding_adapter_instance, "health", None)
    assert callable(health), "health() method is missing"
    health_result = health()
    assert isinstance(health_result, Mapping), "health() should return a mapping"

    # Async health (best-effort)
    async_health = getattr(embedding_adapter_instance, "ahealth", None)
    if async_health is not None and callable(async_health):
        ahealth_result = _run_async_if_needed(async_health())
        assert isinstance(ahealth_result, Mapping), "ahealth() should return a mapping"
