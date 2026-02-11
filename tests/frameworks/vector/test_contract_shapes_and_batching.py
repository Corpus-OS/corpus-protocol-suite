# tests/frameworks/vector/test_contract_shapes_and_batching.py

from __future__ import annotations

import importlib
import inspect
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import pytest

from tests.frameworks.registries.vector_registry import (
    VectorFrameworkDescriptor,
    iter_vector_framework_descriptors,
)

# ---------------------------------------------------------------------------
# Constants (shared test inputs)
# ---------------------------------------------------------------------------

UPSERT_SINGLE_TEXTS = ["vec-upsert-single"]
UPSERT_MULTI_TEXTS = [f"vec-upsert-{i}" for i in range(5)]

UPSERT_SINGLE_METADATAS = [{"source": "single"}]
UPSERT_MULTI_METADATAS = [{"source": f"multi-{i}"} for i in range(5)]

UPSERT_SINGLE_IDS = ["vec-id-single"]
UPSERT_MULTI_IDS = [f"vec-id-{i}" for i in range(5)]

DELETE_SINGLE_IDS = ["vec-del-single"]
DELETE_MULTI_IDS = [f"vec-del-{i}" for i in range(5)]

# Keep dimensionality intentionally small: fast, deterministic, and valid.
TEST_DIMENSIONS = 3

# Query inputs must be protocol-valid across all wrappers: either a vector (list[float])
# or a QuerySpec-like mapping. Strings are intentionally NOT supported by the SDK.
QUERY_1 = [0.1, 0.2, 0.3]
QUERY_2 = [0.2, 0.3, 0.4]

# Batch query inputs: list of query objects (same shape as query()).
BATCH_QUERIES_1 = [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.4, 0.4, 0.4]]
BATCH_QUERIES_2 = [[0.2, 0.3, 0.4], [0.5, 0.5, 0.5], [0.9, 0.1, 0.2]]


def _base_framework_context(namespace: str) -> dict[str, Any]:
    """
    Base framework context used by all wrappers.

    IMPORTANT:
    - Provide both "namespace" and "collection" so DefaultVectorFrameworkTranslator.preferred_namespace
      works regardless of which alias wrappers prefer.
    - Include dimensions for create_namespace (strict requirement in the SDK).
    - Keep distance_metric stable and widely-supported.
    """
    return {
        "namespace": namespace,
        "collection": namespace,
        "dimensions": TEST_DIMENSIONS,
        "distance_metric": "cosine",
    }


def _vector_for_id(idx: int) -> list[float]:
    """
    Deterministic vectors with correct dimensionality.
    """
    base = float(idx + 1)
    return [base / 10.0, (base + 1.0) / 10.0, (base + 2.0) / 10.0]


def _make_documents(
    *,
    namespace: str,
    ids: Sequence[str],
    texts: Sequence[str],
    metadatas: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Create protocol-compatible document mappings for upsert.

    IMPORTANT:
    - We intentionally do NOT include "text" here.
      Some adapters correctly advertise that text storage is unsupported and will raise
      NotSupported if text is set. Bulk tests must remain compatible with vector+metadata-only adapters.
    - We include namespace per-document to avoid accidental default-namespace usage when wrappers
      prefer per-item namespace semantics.
    """
    out: list[dict[str, Any]] = []
    for idx, (doc_id, _text, metadata) in enumerate(zip(ids, texts, metadatas)):
        out.append(
            {
                "id": doc_id,
                "vector": _vector_for_id(idx),
                "metadata": metadata,
                "namespace": namespace,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_method(instance: Any, name: str | None) -> Callable[..., Any]:
    """
    Fetch a method from the instance and assert it is callable.

    If name is None, fail fast with a clear assertion message so we don't
    silently mis-test a missing surface.
    """
    assert name, "Expected a non-empty method name"
    attr = getattr(instance, name, None)
    assert callable(attr), f"{instance!r} missing expected callable method {name!r}"
    return attr


async def _maybe_await(value: Any) -> Any:
    """
    Await the returned value if it is awaitable; otherwise return it directly.

    This keeps tests compatible with both sync and async adapter surfaces
    while remaining strict about not swallowing errors.
    """
    if inspect.isawaitable(value):
        return await value
    return value


def _type_stability_assert(result1: Any, result2: Any, label: str) -> None:
    """
    Assert type stability across two results, when both are non-None.
    """
    assert result1 is not None
    assert result2 is not None
    assert type(result1) is type(result2), (
        f"{label} returned different result types across calls: "
        f"{type(result1).__name__} vs {type(result2).__name__}"
    )


def _inject_context_if_declared(
    descriptor: VectorFrameworkDescriptor,
    kwargs: dict[str, Any],
    context: Any,
) -> dict[str, Any]:
    """
    Inject framework-specific context into kwargs only when descriptor.context_kwarg is set.

    This helper is intentionally conservative so tests don't leak unexpected kwargs
    into adapters that do not declare a context surface.
    """
    if descriptor.context_kwarg:
        kwargs.setdefault(descriptor.context_kwarg, context)
    return kwargs


def _signature_accepts_kwarg(fn: Callable[..., Any], kw: str) -> bool:
    """
    Return True if the callable signature can accept `kw`, either via an explicit
    parameter name or a **kwargs catch-all.

    This avoids fragile behavioral calls when adapters are implemented in ways
    that do not expose Python signatures (builtins/extension methods).
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True

    params = sig.parameters.values()
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return True
    return kw in sig.parameters


def _import_client_class(desc: VectorFrameworkDescriptor) -> type:
    """
    Import the adapter module and resolve the adapter class with clear errors.

    This keeps failures attributable: registry metadata vs module/class issues.
    """
    try:
        module = importlib.import_module(desc.adapter_module)
    except Exception as e:  # noqa: BLE001
        raise AssertionError(
            f"{desc.name}: failed to import adapter_module {desc.adapter_module!r}: {e}"
        ) from e

    try:
        client_cls = getattr(module, desc.adapter_class)
    except Exception as e:  # noqa: BLE001
        raise AssertionError(
            f"{desc.name}: adapter_class {desc.adapter_class!r} not found in module "
            f"{desc.adapter_module!r}: {e}"
        ) from e

    if not isinstance(client_cls, type):
        raise AssertionError(
            f"{desc.name}: resolved adapter_class {desc.adapter_class!r} is not a class"
        )

    return client_cls


def _construct_client(desc: VectorFrameworkDescriptor, adapter: Any) -> Any:
    """
    Construct a concrete client instance using descriptor.adapter_init_kwarg.

    Tests MUST respect adapter_init_kwarg and must never hardcode "adapter".
    """
    client_cls = _import_client_class(desc)
    init_kwargs: dict[str, Any] = {desc.adapter_init_kwarg: adapter}
    try:
        return client_cls(**init_kwargs)
    except TypeError as e:
        raise AssertionError(
            f"{desc.name}: failed to construct client with adapter_init_kwarg="
            f"{desc.adapter_init_kwarg!r}: {e}"
        ) from e


def _call_with_fallbacks(
    fn: Callable[..., Any],
    attempts: Sequence[tuple[tuple[Any, ...], dict[str, Any]]],
) -> Any:
    """
    Call `fn` using a small set of explicit candidate calling conventions.

    Why this exists:
    - Different framework adapters may expose slightly different ergonomic wrappers
      over the same underlying protocol.
    - These tests want to validate batching behavior robustly without locking the
      contract to a single exact argument layout.

    Behavior:
    - Try each (args, kwargs) pair in order.
    - If a candidate raises TypeError (argument mismatch), try the next one.
    - For any other exception, propagate (it indicates a real runtime failure).
    - If all candidates TypeError, fail with a clear assertion.
    """
    last_type_error: Optional[TypeError] = None
    for args, kwargs in attempts:
        try:
            return fn(*args, **kwargs)
        except TypeError as e:
            last_type_error = e
            continue
    raise AssertionError(
        f"All call candidates failed with TypeError; last error: {last_type_error}"
    ) from last_type_error


def _extract_batch_results(value: Any) -> Optional[Sequence[Any]]:
    """
    Extract a per-query result sequence from a batch_query return value.

    Contract intent:
    - batch_query is a bulk API; it should return something containing per-query results.
    - Different wrappers may return:
      * list/tuple of results
      * mapping like {"results": [...]} or {"items": [...]}
      * object with attribute .results or .items

    We keep this extraction conservative:
    - If we can confidently find a sequence, return it.
    - Otherwise return None (the test will assert minimal properties only).
    """
    if isinstance(value, (list, tuple)):
        return value

    if isinstance(value, dict):
        for key in ("results", "items", "data", "output"):
            v = value.get(key)
            if isinstance(v, (list, tuple)):
                return v

    for attr in ("results", "items", "data", "output"):
        if hasattr(value, attr):
            v = getattr(value, attr)
            if isinstance(v, (list, tuple)):
                return v

    return None


async def _ensure_namespace_seeded(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
    *,
    namespace: str,
) -> None:
    """
    Create namespace (with required dimensions) and upsert one vector.

    Some adapters require the namespace to exist and contain at least one vector
    before query/batch_query will succeed.

    This is a test precondition setup step; it should not weaken production behavior.
    """
    ctx = _base_framework_context(namespace)

    create_fn = _get_method(
        vector_client_instance, framework_descriptor.create_namespace_method
    )
    await _maybe_await(
        _call_with_fallbacks(
            create_fn,
            attempts=[
                ((namespace,), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"name": namespace}, context=ctx))),
            ],
        )
    )

    # Seed with one vector so query paths don't fail with IndexNotReady or empty-index rules.
    seed_docs = _make_documents(
        namespace=namespace,
        ids=["seed-0"],
        texts=["seed"],
        metadatas=[{"source": "seed"}],
    )
    upsert_fn = _get_method(vector_client_instance, framework_descriptor.upsert_method)
    await _maybe_await(
        _call_with_fallbacks(
            upsert_fn,
            attempts=[
                ((seed_docs,), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"raw_documents": seed_docs}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"documents": seed_docs}, context=ctx))),
            ],
        )
    )


@dataclass(frozen=True)
class _UpsertItem:
    """
    Minimal structured upsert item fallback.

    Some wrappers accept structured items rather than mapping-based raw_documents/documents.
    This is kept for forward-compat and as a diagnostic option when wrappers evolve.
    """
    id: str
    text: str
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=list(iter_vector_framework_descriptors()), name="framework_descriptor")
def framework_descriptor_fixture(
    request: pytest.FixtureRequest,
) -> VectorFrameworkDescriptor:
    """
    Parameterized over all registered vector framework descriptors.

    This suite does not use pytest.skip; failures here are intended to be actionable:
    - registry metadata drift (wrong method/class/module names), or
    - adapter import/constructor incompatibility, or
    - batching/shape instability.
    """
    return request.param


@pytest.fixture
def vector_client_instance(
    framework_descriptor: VectorFrameworkDescriptor,
    adapter: Any,
) -> Any:
    """
    Construct a concrete vector client/store instance for the given descriptor.

    IMPORTANT:
    We must respect descriptor.adapter_init_kwarg (tests must not hardcode "adapter").
    """
    return _construct_client(framework_descriptor, adapter)


@pytest.fixture
def test_namespace(framework_descriptor: VectorFrameworkDescriptor, request: pytest.FixtureRequest) -> str:
    """
    Unique namespace per test + per framework.

    Why this matters:
    - Prevents cross-test contamination (especially in parallel CI).
    - Prevents "already exists" / "unknown namespace" flake from shared global state.
    - Makes failures attributable to the test in question.
    """
    # Use node name + framework name + short uuid to ensure uniqueness while staying readable.
    node = request.node.name.replace("[", "_").replace("]", "_")
    suffix = uuid.uuid4().hex[:8]
    return f"contract-{framework_descriptor.name}-{node}-{suffix}"


@pytest.fixture
async def seeded_namespace(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
    test_namespace: str,
) -> str:
    """
    Ensure a namespace exists and is seeded for query/batch_query/delete tests.

    We seed once per test function (cheap enough, and robust across differing adapter semantics).
    """
    await _ensure_namespace_seeded(
        framework_descriptor,
        vector_client_instance,
        namespace=test_namespace,
    )
    return test_namespace


# ---------------------------------------------------------------------------
# Surface / availability sanity (lightweight, no behavioral overreach)
# ---------------------------------------------------------------------------


def test_descriptor_declares_required_surfaces_as_non_empty_strings() -> None:
    """
    Registry descriptors must declare required method names as non-empty strings.
    """
    required = (
        "capabilities_method",
        "query_method",
        "upsert_method",
        "delete_method",
        "create_namespace_method",
        "delete_namespace_method",
        "health_method",
        "adapter_init_kwarg",
        "adapter_module",
        "adapter_class",
        "name",
    )

    for desc in iter_vector_framework_descriptors():
        for field_name in required:
            v = getattr(desc, field_name)
            assert isinstance(v, str) and v.strip(), f"{desc.name}: {field_name} must be a non-empty string"


def test_batch_query_descriptor_is_coherent_when_enabled() -> None:
    """
    If has_batch_query=True, batch_query_method must be a non-empty string.
    """
    for desc in iter_vector_framework_descriptors():
        if desc.has_batch_query:
            assert desc.batch_query_method is not None and desc.batch_query_method.strip(), (
                f"{desc.name}: has_batch_query=True requires batch_query_method"
            )


def test_client_exposes_required_methods_as_callables(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    Bidirectional sanity:
    The adapter instance must expose the registry-declared required surface.
    """
    _get_method(vector_client_instance, framework_descriptor.capabilities_method)
    _get_method(vector_client_instance, framework_descriptor.query_method)
    _get_method(vector_client_instance, framework_descriptor.upsert_method)
    _get_method(vector_client_instance, framework_descriptor.delete_method)
    _get_method(vector_client_instance, framework_descriptor.create_namespace_method)
    _get_method(vector_client_instance, framework_descriptor.delete_namespace_method)
    _get_method(vector_client_instance, framework_descriptor.health_method)


def test_batch_query_method_callable_presence_when_has_batch_query_true(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    Required by contract change request:
    a single test for batch_query_method callable presence when has_batch_query=True.
    """
    if framework_descriptor.has_batch_query:
        _get_method(vector_client_instance, framework_descriptor.batch_query_method)


# ---------------------------------------------------------------------------
# Capabilities / health: return non-None and remain stable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capabilities_returns_non_none(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    capabilities() should return a non-None value (shape is framework-defined).
    """
    fn = _get_method(vector_client_instance, framework_descriptor.capabilities_method)
    result = await _maybe_await(fn())
    assert result is not None


@pytest.mark.asyncio
async def test_health_returns_non_none(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    health() should return a non-None value (shape is framework-defined).
    """
    fn = _get_method(vector_client_instance, framework_descriptor.health_method)
    result = await _maybe_await(fn())
    assert result is not None


@pytest.mark.asyncio
async def test_capabilities_type_stable_across_calls(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    capabilities() should return a stable type across repeated calls.
    """
    fn = _get_method(vector_client_instance, framework_descriptor.capabilities_method)
    r1 = await _maybe_await(fn())
    r2 = await _maybe_await(fn())
    _type_stability_assert(r1, r2, label="capabilities()")


@pytest.mark.asyncio
async def test_health_type_stable_across_calls(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    health() should return a stable type across repeated calls.
    """
    fn = _get_method(vector_client_instance, framework_descriptor.health_method)
    r1 = await _maybe_await(fn())
    r2 = await _maybe_await(fn())
    _type_stability_assert(r1, r2, label="health()")


# ---------------------------------------------------------------------------
# Namespace ops: should not crash and should be type-stable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_namespace_does_not_crash(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
    test_namespace: str,
) -> None:
    """
    create_namespace() should accept a namespace name when provided with required context.

    IMPORTANT:
    The SDK requires "dimensions" for namespace creation. This test supplies it via
    the framework context to avoid hardcoding wrapper-specific signatures.
    """
    fn = _get_method(vector_client_instance, framework_descriptor.create_namespace_method)
    ctx = _base_framework_context(test_namespace)

    _ = await _maybe_await(
        _call_with_fallbacks(
            fn,
            attempts=[
                ((test_namespace,), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"name": test_namespace}, context=ctx))),
            ],
        )
    )
    # Shape is framework-defined; we assert only non-crash.


@pytest.mark.asyncio
async def test_delete_namespace_does_not_crash_when_namespace_exists(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
    test_namespace: str,
) -> None:
    """
    delete_namespace() should accept a namespace name without crashing when the namespace exists.

    We explicitly create the namespace first to avoid ambiguous "not found" semantics.
    """
    ctx = _base_framework_context(test_namespace)

    create_fn = _get_method(vector_client_instance, framework_descriptor.create_namespace_method)
    await _maybe_await(
        _call_with_fallbacks(
            create_fn,
            attempts=[
                ((test_namespace,), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"name": test_namespace}, context=ctx))),
            ],
        )
    )

    delete_fn = _get_method(vector_client_instance, framework_descriptor.delete_namespace_method)
    _ = await _maybe_await(
        _call_with_fallbacks(
            delete_fn,
            attempts=[
                ((test_namespace,), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"name": test_namespace}, context=ctx))),
            ],
        )
    )
    # Shape is framework-defined; we assert only non-crash.


@pytest.mark.asyncio
async def test_namespace_ops_type_stable_across_calls(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
    test_namespace: str,
) -> None:
    """
    Namespace op return types should remain stable across repeated calls.

    NOTE:
    Some adapters may return None for these operations; in that case we do not enforce type stability.
    """
    create_fn = _get_method(vector_client_instance, framework_descriptor.create_namespace_method)
    delete_fn = _get_method(vector_client_instance, framework_descriptor.delete_namespace_method)

    ctx = _base_framework_context(test_namespace)

    c1 = await _maybe_await(
        _call_with_fallbacks(
            create_fn,
            attempts=[
                ((test_namespace,), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"name": test_namespace}, context=ctx))),
            ],
        )
    )
    c2 = await _maybe_await(
        _call_with_fallbacks(
            create_fn,
            attempts=[
                ((test_namespace,), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"name": test_namespace}, context=ctx))),
            ],
        )
    )
    if c1 is not None and c2 is not None:
        _type_stability_assert(c1, c2, label="create_namespace()")

    d1 = await _maybe_await(
        _call_with_fallbacks(
            delete_fn,
            attempts=[
                ((test_namespace,), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"name": test_namespace}, context=ctx))),
            ],
        )
    )
    d2 = await _maybe_await(
        _call_with_fallbacks(
            delete_fn,
            attempts=[
                ((test_namespace,), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"name": test_namespace}, context=ctx))),
            ],
        )
    )
    if d1 is not None and d2 is not None:
        _type_stability_assert(d1, d2, label="delete_namespace()")


# ---------------------------------------------------------------------------
# Upsert batching: single vs multi batch acceptance + type stability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_accepts_single_and_multiple_items(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
    seeded_namespace: str,
) -> None:
    """
    The upsert surface should accept both single-item and multi-item batches without crashing.

    We intentionally do not over-specify the exact argument convention; instead we try a
    small, explicit set of common conventions used across wrappers.
    """
    upsert_fn = _get_method(vector_client_instance, framework_descriptor.upsert_method)
    ctx = _base_framework_context(seeded_namespace)

    single_docs = _make_documents(
        namespace=seeded_namespace,
        ids=UPSERT_SINGLE_IDS,
        texts=UPSERT_SINGLE_TEXTS,
        metadatas=UPSERT_SINGLE_METADATAS,
    )
    multi_docs = _make_documents(
        namespace=seeded_namespace,
        ids=UPSERT_MULTI_IDS,
        texts=UPSERT_MULTI_TEXTS,
        metadatas=UPSERT_MULTI_METADATAS,
    )

    single_result = await _maybe_await(
        _call_with_fallbacks(
            upsert_fn,
            attempts=[
                ((single_docs,), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"raw_documents": single_docs}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"documents": single_docs}, context=ctx))),
            ],
        )
    )

    multi_result = await _maybe_await(
        _call_with_fallbacks(
            upsert_fn,
            attempts=[
                ((multi_docs,), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"raw_documents": multi_docs}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"documents": multi_docs}, context=ctx))),
            ],
        )
    )

    # We do not require non-None returns, but if both are non-None, enforce type stability.
    if single_result is not None and multi_result is not None:
        _type_stability_assert(single_result, multi_result, label="upsert(single) vs upsert(multi)")


@pytest.mark.asyncio
async def test_upsert_result_type_stable_across_calls(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
    seeded_namespace: str,
) -> None:
    """
    Repeated upsert calls with similar batch shapes should return a stable result type.
    """
    upsert_fn = _get_method(vector_client_instance, framework_descriptor.upsert_method)
    ctx = _base_framework_context(seeded_namespace)

    docs_a = _make_documents(
        namespace=seeded_namespace,
        ids=UPSERT_SINGLE_IDS,
        texts=UPSERT_SINGLE_TEXTS,
        metadatas=UPSERT_SINGLE_METADATAS,
    )
    docs_b = _make_documents(
        namespace=seeded_namespace,
        ids=["vec-id-single-2"],
        texts=UPSERT_SINGLE_TEXTS,
        metadatas=UPSERT_SINGLE_METADATAS,
    )

    r1 = await _maybe_await(
        _call_with_fallbacks(
            upsert_fn,
            [
                ((docs_a,), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"raw_documents": docs_a}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"documents": docs_a}, context=ctx))),
            ],
        )
    )
    r2 = await _maybe_await(
        _call_with_fallbacks(
            upsert_fn,
            [
                ((docs_b,), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"raw_documents": docs_b}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"documents": docs_b}, context=ctx))),
            ],
        )
    )

    if r1 is not None and r2 is not None:
        _type_stability_assert(r1, r2, label="upsert() type stability")


# ---------------------------------------------------------------------------
# Delete batching: single vs multi acceptance + type stability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_accepts_single_and_multiple_ids(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
    seeded_namespace: str,
) -> None:
    """
    The delete surface should accept both single-id and multi-id batches without crashing.

    IMPORTANT:
    We ensure the namespace exists and is seeded first; some adapters reject deletes
    for unknown namespaces.
    """
    delete_fn = _get_method(vector_client_instance, framework_descriptor.delete_method)
    ctx = _base_framework_context(seeded_namespace)

    r1 = await _maybe_await(
        _call_with_fallbacks(
            delete_fn,
            [
                ((DELETE_SINGLE_IDS,), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
                ((DELETE_SINGLE_IDS[0],), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
            ],
        )
    )
    r2 = await _maybe_await(
        _call_with_fallbacks(
            delete_fn,
            [
                ((DELETE_MULTI_IDS,), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
            ],
        )
    )

    if r1 is not None and r2 is not None:
        _type_stability_assert(r1, r2, label="delete(single) vs delete(multi)")


@pytest.mark.asyncio
async def test_delete_result_type_stable_across_calls(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
    seeded_namespace: str,
) -> None:
    """
    Repeated delete calls should return a stable result type when non-None.
    """
    delete_fn = _get_method(vector_client_instance, framework_descriptor.delete_method)
    ctx = _base_framework_context(seeded_namespace)

    r1 = await _maybe_await(
        _call_with_fallbacks(
            delete_fn,
            [
                ((DELETE_SINGLE_IDS,), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
            ],
        )
    )
    r2 = await _maybe_await(
        _call_with_fallbacks(
            delete_fn,
            [
                ((["vec-del-single-2"],), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
                (("vec-del-single-2",), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
            ],
        )
    )

    if r1 is not None and r2 is not None:
        _type_stability_assert(r1, r2, label="delete() type stability")


# ---------------------------------------------------------------------------
# Query + batch_query: type stability, batching acceptance, and minimal parity checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_result_type_stable_across_calls(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
    seeded_namespace: str,
) -> None:
    """
    For simple similarity queries, the vector client should return the same *type*
    on repeated calls with similar inputs.

    This catches adapters that sometimes return different container types, which
    would break callers relying on type stability.
    """
    query_fn = _get_method(vector_client_instance, framework_descriptor.query_method)
    ctx = _base_framework_context(seeded_namespace)

    r1 = await _maybe_await(
        _call_with_fallbacks(
            query_fn,
            attempts=[
                ((QUERY_1,), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"raw_query": QUERY_1}, context=ctx))),
            ],
        )
    )
    r2 = await _maybe_await(
        _call_with_fallbacks(
            query_fn,
            attempts=[
                ((QUERY_2,), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"raw_query": QUERY_2}, context=ctx))),
            ],
        )
    )

    _type_stability_assert(r1, r2, label="query()")


@pytest.mark.asyncio
async def test_batch_query_accepts_multiple_queries_and_type_stable(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
    seeded_namespace: str,
) -> None:
    """
    When batch_query is declared at the wrapper surface, it should accept multiple queries
    and return a stable type across calls with similar input shapes.

    This is the primary "bulk queries" contract for batching.
    """
    if not framework_descriptor.has_batch_query:
        # Contract: if not declared, we do not force it.
        return

    bq_fn = _get_method(vector_client_instance, framework_descriptor.batch_query_method)
    ctx = _base_framework_context(seeded_namespace)

    r1 = await _maybe_await(
        _call_with_fallbacks(
            bq_fn,
            attempts=[
                ((BATCH_QUERIES_1,), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"raw_queries": BATCH_QUERIES_1}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"queries": BATCH_QUERIES_1}, context=ctx))),
            ],
        )
    )
    r2 = await _maybe_await(
        _call_with_fallbacks(
            bq_fn,
            attempts=[
                ((BATCH_QUERIES_2,), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"raw_queries": BATCH_QUERIES_2}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"queries": BATCH_QUERIES_2}, context=ctx))),
            ],
        )
    )

    if r1 is not None and r2 is not None:
        _type_stability_assert(r1, r2, label="batch_query() type stability")


@pytest.mark.asyncio
async def test_batch_query_result_cardinality_matches_input_when_extractable(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
    seeded_namespace: str,
) -> None:
    """
    Bulk contract (robust, not over-prescriptive):

    If batch_query returns an extractable per-query result sequence, its length should match
    the number of queries submitted. This protects callers relying on 1:1 correspondence.

    We do not fail adapters that return non-sequence batch structures; those are validated by
    type-stability tests and parity checks instead.
    """
    if not framework_descriptor.has_batch_query:
        return

    bq_fn = _get_method(vector_client_instance, framework_descriptor.batch_query_method)
    ctx = _base_framework_context(seeded_namespace)

    out = await _maybe_await(
        _call_with_fallbacks(
            bq_fn,
            attempts=[
                ((BATCH_QUERIES_1,), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"raw_queries": BATCH_QUERIES_1}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"queries": BATCH_QUERIES_1}, context=ctx))),
            ],
        )
    )

    results = _extract_batch_results(out)
    if results is not None:
        assert len(results) == len(BATCH_QUERIES_1), (
            "batch_query result count does not match input query count: "
            f"{len(results)} vs {len(BATCH_QUERIES_1)}"
        )


@pytest.mark.asyncio
async def test_batch_query_result_type_compatible_with_query_when_both_non_none(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
    seeded_namespace: str,
) -> None:
    """
    Minimal parity check:
    If both query() and batch_query() return non-None values, their return types
    should be compatible, because callers may switch between single and batched
    execution paths.

    Compatibility definition used here:
    - If we can extract a single per-query result from the batch response, the element type
      should match query() result type.
    """
    if not framework_descriptor.has_batch_query:
        return

    query_fn = _get_method(vector_client_instance, framework_descriptor.query_method)
    bq_fn = _get_method(vector_client_instance, framework_descriptor.batch_query_method)

    ctx = _base_framework_context(seeded_namespace)

    q_res = await _maybe_await(
        _call_with_fallbacks(
            query_fn,
            attempts=[
                ((QUERY_1,), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"raw_query": QUERY_1}, context=ctx))),
            ],
        )
    )
    bq_res = await _maybe_await(
        _call_with_fallbacks(
            bq_fn,
            attempts=[
                (([QUERY_1],), dict(_inject_context_if_declared(framework_descriptor, {}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"raw_queries": [QUERY_1]}, context=ctx))),
                (tuple(), dict(_inject_context_if_declared(framework_descriptor, {"queries": [QUERY_1]}, context=ctx))),
            ],
        )
    )

    if q_res is None or bq_res is None:
        return

    extracted = _extract_batch_results(bq_res)
    if extracted is not None and len(extracted) == 1:
        assert type(extracted[0]) is type(q_res), (
            "batch_query single-result element type does not match query() result type: "
            f"{type(extracted[0]).__name__} vs {type(q_res).__name__}"
        )


# ---------------------------------------------------------------------------
# Context kwarg behavior: signature acceptance and helper leakage guarantees
# ---------------------------------------------------------------------------


def test_context_kwarg_signature_acceptance_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    If a framework declares context_kwarg, major surfaces should accept it
    (either explicitly or via **kwargs).

    This is a surface-level check that avoids forcing deep protocol-spec construction.
    """
    if not framework_descriptor.context_kwarg:
        return

    ctx_kw = framework_descriptor.context_kwarg
    for method_name in (
        framework_descriptor.query_method,
        framework_descriptor.upsert_method,
        framework_descriptor.delete_method,
        framework_descriptor.health_method,
        framework_descriptor.capabilities_method,
    ):
        fn = _get_method(vector_client_instance, method_name)
        assert _signature_accepts_kwarg(fn, ctx_kw), (
            f"{framework_descriptor.name}: method {method_name!r} does not appear to accept "
            f"declared context_kwarg {ctx_kw!r} (no parameter and no **kwargs)"
        )


def test_context_injection_does_not_occur_when_context_kwarg_is_none() -> None:
    """
    Required by contract change request:
    a test that context injection does not occur when context_kwarg is None.
    """
    desc = VectorFrameworkDescriptor(
        name="no_context",
        adapter_module="test.module",
        adapter_class="TestClient",
        context_kwarg=None,
    )

    kwargs: dict[str, Any] = {"existing": 1}
    out = _inject_context_if_declared(desc, kwargs, context={"k": "v"})

    assert out is kwargs
    assert out == {"existing": 1}


def test_context_injection_occurs_when_context_kwarg_is_set() -> None:
    """
    Complementary sanity: when context_kwarg is set, the helper injects it.
    """
    desc = VectorFrameworkDescriptor(
        name="with_context",
        adapter_module="test.module",
        adapter_class="TestClient",
        context_kwarg="ctx",
    )

    kwargs: dict[str, Any] = {"existing": 1}
    out = _inject_context_if_declared(desc, kwargs, context={"k": "v"})

    assert out is kwargs  # helper mutates in-place by design
    assert out["existing"] == 1
    assert out["ctx"] == {"k": "v"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
