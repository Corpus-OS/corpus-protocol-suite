# tests/frameworks/embedding/test_contract_interface_conformance.py

from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping, Sequence
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

    This uses the registry metadata to import the adapter class and instantiate
    it with the *generic* Corpus adapter provided by the top-level pytest
    plugin (see conftest.py).

    
    - If a framework is unavailable, this fixture returns None and tests must
      treat that as a validated pass condition (not a skip).
    """
    if not framework_descriptor.is_available():
        return None

    module = importlib.import_module(framework_descriptor.adapter_module)
    adapter_cls = getattr(module, framework_descriptor.adapter_class)

    # All embedding framework adapters take a corpus_adapter implementing the
    # embedding protocol surface. The global `adapter` fixture is pluggable via
    # CORPUS_ADAPTER and can point to the user's real adapter implementation.
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
    - Values (if present) must be numeric-ish
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
    - Values (if present) must be numeric-ish
    """
    assert isinstance(result, Sequence) and not isinstance(
        result, (str, bytes)
    ), f"Expected sequence (non-str), got {type(result).__name__}"
    for val in result:
        assert isinstance(val, (int, float)), f"Embedding value is not numeric: {val!r}"


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
            return
        flag = getattr(module, descriptor.availability_attr, None)
        assert flag is None or bool(flag) is False


def _call_with_optional_context(
    descriptor: EmbeddingFrameworkDescriptor,
    fn: Callable[..., Any],
    arg: Any,
    ctx: Any,
) -> Any:
    """
    Call embedding method with context in a framework-agnostic way.

    Primary strategy:
      - If descriptor.context_kwarg is set, pass {context_kwarg: ctx}.

    Compatibility fallback:
      - If that raises TypeError due to an unexpected keyword argument, and ctx is a Mapping,
        retry by spreading ctx into kwargs (useful for BaseEmbedding-style **kwargs surfaces).
    """
    if not descriptor.context_kwarg:
        return fn(arg)

    try:
        return fn(arg, **{descriptor.context_kwarg: ctx})
    except TypeError as e:
        msg = str(e)
        unexpected_kw = f"unexpected keyword argument '{descriptor.context_kwarg}'" in msg or (
            "unexpected keyword" in msg and descriptor.context_kwarg in msg
        )
        if unexpected_kw and isinstance(ctx, Mapping):
            return fn(arg, **dict(ctx))
        raise


def _assert_awaitable(value: Any, *, descriptor: EmbeddingFrameworkDescriptor, method_name: str) -> None:
    """Assert that an async method returns an awaitable when invoked."""
    assert inspect.isawaitable(value), (
        f"{descriptor.name}: async method {method_name!r} must return an awaitable, "
        f"got {type(value).__name__}"
    )


def _empty_batch_expected_behavior(
    exc: BaseException,
) -> bool:
    """
    Determine whether an exception type is acceptable for empty-batch behavior.

    Empty-batch semantics differ across frameworks/adapters:
    - Some return []
    - Some raise a validation error

    We treat common "invalid input" errors as acceptable, but fail on unrelated exceptions.
    """
    allowed = (TypeError, ValueError)
    return isinstance(exc, allowed)


def _check_minimal_signature_contract(
    fn: Callable[..., Any],
    *,
    descriptor: EmbeddingFrameworkDescriptor,
    method_name: str,
) -> None:
    """
    Minimal signature sanity checks without enforcing exact parameter equality.

    Rationale:
    - Wrappers/decorators/framework base classes can legitimately alter signatures.
    - We only assert a stable contract: method is callable and can accept at least one
      positional arg (texts/text) and optionally framework context.
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    # For bound methods, the first param is usually "self"; but wrappers may not show it.
    # We require that there is at least one non-vararg parameter OR that it accepts *args.
    has_args = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    non_self_params = [p for p in params if p.name != "self"]
    assert has_args or len(non_self_params) >= 1, (
        f"{descriptor.name}: {method_name} signature does not accept a primary argument"
    )


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

    
    - If framework is unavailable, validate the unavailable contract and return.
    - If framework is available but instantiation fails, this should fail (real regression).
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

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

    
    - If framework is unavailable, validate the unavailable contract and return.
    - If supports_async=False, assert async methods are None and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    if not framework_descriptor.supports_async:
        assert framework_descriptor.async_batch_method is None
        assert framework_descriptor.async_query_method is None
        return

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

    
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    texts = ["alpha text", "beta text"]
    query_text = "gamma query"

    batch_fn = _get_method(embedding_adapter_instance, framework_descriptor.batch_method)
    query_fn = _get_method(embedding_adapter_instance, framework_descriptor.query_method)

    _check_minimal_signature_contract(batch_fn, descriptor=framework_descriptor, method_name=framework_descriptor.batch_method)
    _check_minimal_signature_contract(query_fn, descriptor=framework_descriptor, method_name=framework_descriptor.query_method)

    ctx = dict(framework_descriptor.sample_context or {})

    batch_result = _call_with_optional_context(framework_descriptor, batch_fn, texts, ctx)
    _assert_embedding_matrix_shape(batch_result, expected_rows=len(texts))

    query_result = _call_with_optional_context(framework_descriptor, query_fn, query_text, ctx)
    _assert_embedding_vector_shape(query_result)


def test_single_element_batch(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    Test that single-element batches work correctly.

    
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    texts = ["single text"]
    batch_fn = _get_method(embedding_adapter_instance, framework_descriptor.batch_method)

    ctx = dict(framework_descriptor.sample_context or {})
    result = _call_with_optional_context(framework_descriptor, batch_fn, texts, ctx)
    _assert_embedding_matrix_shape(result, expected_rows=1)


def test_empty_batch_handling(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    Test how frameworks handle empty batch requests.

    Acceptable behaviors:
    - return []
    - raise a common validation error (TypeError/ValueError)

    
    - If framework is unavailable, validate the unavailable contract and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    batch_fn = _get_method(embedding_adapter_instance, framework_descriptor.batch_method)
    ctx = dict(framework_descriptor.sample_context or {})

    try:
        result = _call_with_optional_context(framework_descriptor, batch_fn, [], ctx)
    except BaseException as e:
        assert _empty_batch_expected_behavior(e), (
            f"{framework_descriptor.name}: unexpected exception for empty batch: {type(e).__name__}: {e}"
        )
        return

    # If no exception, should return empty list/sequence
    assert isinstance(result, Sequence) and not isinstance(result, (str, bytes))
    assert len(result) == 0


@pytest.mark.asyncio
async def test_async_embedding_interface_conformance(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    Validate that async batch and query methods (when declared) accept text
    input and return embedding shapes compatible with the sync API.

    
    - If framework is unavailable, validate the unavailable contract and return.
    - If async is unsupported, validate that and return.
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
    assert framework_descriptor.async_query_method is not None

    texts = ["alpha async", "beta async"]
    query_text = "gamma async"

    abatch_fn = _get_method(embedding_adapter_instance, framework_descriptor.async_batch_method)
    aquery_fn = _get_method(embedding_adapter_instance, framework_descriptor.async_query_method)

    _check_minimal_signature_contract(abatch_fn, descriptor=framework_descriptor, method_name=framework_descriptor.async_batch_method)
    _check_minimal_signature_contract(aquery_fn, descriptor=framework_descriptor, method_name=framework_descriptor.async_query_method)

    ctx = dict(framework_descriptor.sample_context or {})

    batch_coro = _call_with_optional_context(framework_descriptor, abatch_fn, texts, ctx)
    _assert_awaitable(batch_coro, descriptor=framework_descriptor, method_name=framework_descriptor.async_batch_method)
    batch_result = await batch_coro
    _assert_embedding_matrix_shape(batch_result, expected_rows=len(texts))

    query_coro = _call_with_optional_context(framework_descriptor, aquery_fn, query_text, ctx)
    _assert_awaitable(query_coro, descriptor=framework_descriptor, method_name=framework_descriptor.async_query_method)
    query_result = await query_coro
    _assert_embedding_vector_shape(query_result)


def test_context_kwarg_is_accepted_when_declared(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    If a context_kwarg is declared in the descriptor, the corresponding
    embedding methods should accept that kwarg without raising TypeError.

    
    - If framework is unavailable, validate the unavailable contract and return.
    - If context_kwarg is not declared, assert that fact and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    if not framework_descriptor.context_kwarg:
        assert framework_descriptor.context_kwarg is None
        return

    ctx_kw = framework_descriptor.context_kwarg
    texts = ["ctx alpha", "ctx beta"]
    query_text = "ctx gamma"

    batch_fn = _get_method(embedding_adapter_instance, framework_descriptor.batch_method)
    query_fn = _get_method(embedding_adapter_instance, framework_descriptor.query_method)

    # Should not raise TypeError for mapping context.
    ctx = dict(framework_descriptor.sample_context or {})
    ctx.update({"test": "value"})

    batch_result = _call_with_optional_context(framework_descriptor, batch_fn, texts, ctx)
    query_result = _call_with_optional_context(framework_descriptor, query_fn, query_text, ctx)

    _assert_embedding_matrix_shape(batch_result, expected_rows=len(texts))
    _assert_embedding_vector_shape(query_result)

    # Also validate that the declared context_kwarg itself is accepted, when applicable.
    # If the adapter expects kwargs-spread only, this may raise; that is treated as a real mismatch
    # because the registry explicitly declares context_kwarg.
    try:
        if ctx_kw:
            batch_fn(texts, **{ctx_kw: {"test": "value"}})
            query_fn(query_text, **{ctx_kw: {"test": "value"}})
    except TypeError as e:
        raise AssertionError(
            f"{framework_descriptor.name}: declared context_kwarg={ctx_kw!r} but method rejected it: {e}"
        ) from e


def test_embedding_dimension_when_required(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    adapter: Any,
) -> None:
    """
    Test that frameworks requiring embedding dimension enforce it.

    
    - If framework is unavailable, validate the unavailable contract and return.
    - If framework does not require dimension, assert that fact and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.requires_embedding_dimension:
        assert framework_descriptor.requires_embedding_dimension is False
        return

    module = importlib.import_module(framework_descriptor.adapter_module)
    adapter_cls = getattr(module, framework_descriptor.adapter_class)

    kw = framework_descriptor.embedding_dimension_kwarg or "embedding_dimension"

    # Should fail without embedding dimension override (TypeError/ValueError).
    with pytest.raises((TypeError, ValueError)):
        adapter_cls(corpus_adapter=adapter)

    # Should succeed with it.
    instance = adapter_cls(corpus_adapter=adapter, **{kw: 8})
    assert instance is not None

    # Verify methods work.
    batch_fn = _get_method(instance, framework_descriptor.batch_method)
    ctx = dict(framework_descriptor.sample_context or {})

    try:
        result = _call_with_optional_context(framework_descriptor, batch_fn, ["test"], ctx)
        _assert_embedding_matrix_shape(result, expected_rows=1)
    except BaseException as e:
        raise AssertionError(
            f"{framework_descriptor.name}: dimension-required adapter instantiated successfully, "
            f"but batch method failed unexpectedly: {type(e).__name__}: {e}"
        ) from e


def test_alias_methods_exist_and_behave_consistently_when_declared(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    If a framework declares aliases in the registry, those alias methods should:
    - exist on the adapter instance
    - be callable
    - return valid shapes when called with the same input as the primary method

    
    - If framework is unavailable, validate the unavailable contract and return.
    - If no aliases are declared, assert that fact and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    if not framework_descriptor.aliases:
        assert framework_descriptor.aliases is None
        return

    ctx = dict(framework_descriptor.sample_context or {})
    texts = ["alias-alpha", "alias-beta"]
    query_text = "alias-query"

    for alias_name, primary_name in framework_descriptor.aliases.items():
        alias_fn = _get_method(embedding_adapter_instance, alias_name)
        primary_fn = _get_method(embedding_adapter_instance, primary_name)

        is_batch = "document" in alias_name or "embeddings" in alias_name or "texts" in alias_name

        if is_batch:
            alias_out = _call_with_optional_context(framework_descriptor, alias_fn, texts, ctx)
            primary_out = _call_with_optional_context(framework_descriptor, primary_fn, texts, ctx)
            # Alias may be async; only enforce awaitable for async-typed names.
            if inspect.isawaitable(alias_out):
                # Async alias: must be awaitable and yield matrix
                # NOTE: This file keeps sync tests sync; async alias surfaces are validated in async tests.
                # We still ensure it is awaitable here.
                _assert_awaitable(alias_out, descriptor=framework_descriptor, method_name=alias_name)
            else:
                _assert_embedding_matrix_shape(alias_out, expected_rows=len(texts))
                _assert_embedding_matrix_shape(primary_out, expected_rows=len(texts))
        else:
            alias_out = _call_with_optional_context(framework_descriptor, alias_fn, query_text, ctx)
            primary_out = _call_with_optional_context(framework_descriptor, primary_fn, query_text, ctx)
            if inspect.isawaitable(alias_out):
                _assert_awaitable(alias_out, descriptor=framework_descriptor, method_name=alias_name)
            else:
                _assert_embedding_vector_shape(alias_out)
                _assert_embedding_vector_shape(primary_out)


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

    
    - If framework is unavailable, validate the unavailable contract and return.
    - If has_capabilities=False, assert that fact and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    if not framework_descriptor.has_capabilities:
        assert framework_descriptor.has_capabilities is False
        return

    # Prefer registry-specified method names; fall back to conventional names.
    cap_name = framework_descriptor.capabilities_method or "capabilities"
    acap_name = framework_descriptor.async_capabilities_method or "acapabilities"

    capabilities = getattr(embedding_adapter_instance, cap_name, None)
    assert callable(capabilities), f"{framework_descriptor.name}: {cap_name}() method is missing"
    caps_result = capabilities()
    assert isinstance(caps_result, Mapping), f"{framework_descriptor.name}: {cap_name}() should return a mapping"

    async_caps = getattr(embedding_adapter_instance, acap_name, None)
    if async_caps is not None and callable(async_caps):
        # Validate it returns an awaitable, but execute in the async test below to avoid loop hazards.
        coro = async_caps()
        assert inspect.isawaitable(coro), f"{framework_descriptor.name}: {acap_name}() must return an awaitable"


@pytest.mark.asyncio
async def test_capabilities_async_contract_if_declared(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    Async companion for capabilities contract.

    Runs only when:
    - framework is available
    - has_capabilities=True
    - async capabilities method exists
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    if not framework_descriptor.has_capabilities:
        assert framework_descriptor.has_capabilities is False
        return

    acap_name = framework_descriptor.async_capabilities_method or "acapabilities"
    async_caps = getattr(embedding_adapter_instance, acap_name, None)
    if async_caps is None or not callable(async_caps):
        return

    coro = async_caps()
    _assert_awaitable(coro, descriptor=framework_descriptor, method_name=acap_name)
    acaps_result = await coro
    assert isinstance(acaps_result, Mapping), f"{framework_descriptor.name}: {acap_name}() should return a mapping"


def test_health_contract_if_declared(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    If a framework declares has_health=True, it should expose a health()
    method returning a mapping. Async variants (when present) should behave
    similarly.

    
    - If framework is unavailable, validate the unavailable contract and return.
    - If has_health=False, assert that fact and return.
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    if not framework_descriptor.has_health:
        assert framework_descriptor.has_health is False
        return

    # Prefer registry-specified method names; fall back to conventional names.
    health_name = framework_descriptor.health_method or "health"
    ahealth_name = framework_descriptor.async_health_method or "ahealth"

    health = getattr(embedding_adapter_instance, health_name, None)
    assert callable(health), f"{framework_descriptor.name}: {health_name}() method is missing"
    health_result = health()
    assert isinstance(health_result, Mapping), f"{framework_descriptor.name}: {health_name}() should return a mapping"

    async_health = getattr(embedding_adapter_instance, ahealth_name, None)
    if async_health is not None and callable(async_health):
        coro = async_health()
        assert inspect.isawaitable(coro), f"{framework_descriptor.name}: {ahealth_name}() must return an awaitable"


@pytest.mark.asyncio
async def test_health_async_contract_if_declared(
    framework_descriptor: EmbeddingFrameworkDescriptor,
    embedding_adapter_instance: Any,
) -> None:
    """
    Async companion for health contract.

    Runs only when:
    - framework is available
    - has_health=True
    - async health method exists
    """
    if not framework_descriptor.is_available():
        _assert_unavailable_contract(framework_descriptor)
        return

    assert embedding_adapter_instance is not None

    if not framework_descriptor.has_health:
        assert framework_descriptor.has_health is False
        return

    ahealth_name = framework_descriptor.async_health_method or "ahealth"
    async_health = getattr(embedding_adapter_instance, ahealth_name, None)
    if async_health is None or not callable(async_health):
        return

    coro = async_health()
    _assert_awaitable(coro, descriptor=framework_descriptor, method_name=ahealth_name)
    ahealth_result = await coro
    assert isinstance(ahealth_result, Mapping), f"{framework_descriptor.name}: {ahealth_name}() should return a mapping"
