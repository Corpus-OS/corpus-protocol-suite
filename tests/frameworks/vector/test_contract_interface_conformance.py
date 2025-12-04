# tests/frameworks/vector/test_contract_interface_conformance.py

from __future__ import annotations

import asyncio
import importlib
import inspect
from collections.abc import Mapping
from typing import Any, Callable

import pytest

from tests.frameworks.registries.vector_registry import (
    VectorFrameworkDescriptor,
    iter_vector_framework_descriptors,
)


# ---------------------------------------------------------------------------
# Constants (shared test inputs)
# ---------------------------------------------------------------------------

ADD_TEXTS = ["vector-add-text-1", "vector-add-text-2"]
ADD_METADATAS = [{"source": "test-1"}, {"source": "test-2"}]
ADD_IDS = ["vector-id-1", "vector-id-2"]

DELETE_IDS = ["vector-delete-id-1", "vector-delete-id-2"]

SYNC_QUERY_TEXT = "vector-sync-query"
SYNC_STREAM_TEXT = "vector-sync-stream"
ASYNC_QUERY_TEXT = "vector-async-query"
ASYNC_STREAM_TEXT = "vector-async-stream"
MMR_QUERY_TEXT = "vector-mmr-query"
ASYNC_MMR_QUERY_TEXT = "vector-async-mmr-query"
CONTEXT_QUERY_TEXT = "vector-context-query"

TOP_K = 4
MMR_LAMBDA = 0.5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=list(iter_vector_framework_descriptors()),
    name="framework_descriptor",
)
def framework_descriptor_fixture(
    request: pytest.FixtureRequest,
) -> VectorFrameworkDescriptor:
    """
    Parameterized over all registered vector framework descriptors.

    Frameworks that are not actually available in the environment (e.g. the
    underlying LlamaIndex / Semantic Kernel libraries are missing) are skipped
    via descriptor.is_available().
    """
    descriptor: VectorFrameworkDescriptor = request.param
    if not descriptor.is_available():
        pytest.skip(f"Framework '{descriptor.name}' not available in this environment")
    return descriptor


@pytest.fixture
def vector_client_instance(
    framework_descriptor: VectorFrameworkDescriptor,
    adapter: Any,
) -> Any:
    """
    Construct a concrete vector client/store instance for the given descriptor.

    This uses the registry metadata to import the client class and instantiate
    it with the *generic* Corpus vector adapter provided by the top-level
    pytest plugin (see conftest.py).

    All vector framework adapters are expected to take a ProtocolV1
    implementation under the kwarg name `adapter`.
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    client_cls = getattr(module, framework_descriptor.adapter_class)

    init_kwargs: dict[str, Any] = {"adapter": adapter}
    instance = client_cls(**init_kwargs)
    return instance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_method(instance: Any, name: str | None) -> Callable[..., Any]:
    """
    Helper to fetch a method from the instance and assert it is callable.

    If name is None, this fails fast with a clear assertion message.
    """
    assert name, "Expected a non-empty method name"
    attr = getattr(instance, name, None)
    assert callable(attr), f"{instance!r} missing expected callable method {name!r}"
    return attr


def _run_async_if_needed(coro: Any) -> Any:
    """
    Run an async coroutine, handling existing event loops gracefully.

    Used for optional async surfaces (e.g. acapabilities/ahealth) in tests
    that are not themselves marked async.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # Fall back to the current event loop if one is already running.
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


def _maybe_inject_context(
    descriptor: VectorFrameworkDescriptor,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """
    Helper to inject the framework-specific context kwarg when declared.
    """
    if descriptor.context_kwarg:
        kwargs.setdefault(descriptor.context_kwarg, {})
    return kwargs


def _build_add_args(
    descriptor: VectorFrameworkDescriptor,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Build positional args + kwargs for a sync/async add call appropriate
    for the unified Corpus vector adapter contract.

    Contract (for all Corpus vector adapters):
        add_method(*, texts, metadatas=None, ids=None, **context)
        async_add_method: same parameters, async.
    """
    args: list[Any] = []
    kwargs: dict[str, Any] = {
        "texts": ADD_TEXTS,
        "metadatas": ADD_METADATAS,
        "ids": ADD_IDS,
    }
    _maybe_inject_context(descriptor, kwargs)
    return args, kwargs


def _build_delete_args(
    descriptor: VectorFrameworkDescriptor,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Build args/kwargs for sync/async delete calls.

    Contract:
        delete_method(*, ids=None, **context)
        async_delete_method: same parameters, async.

    We exercise the `ids` path in tests; adapters are free to also support
    additional filters (ref_doc_ids, where, etc.).
    """
    args: list[Any] = []
    kwargs: dict[str, Any] = {"ids": DELETE_IDS}
    _maybe_inject_context(descriptor, kwargs)
    return args, kwargs


def _build_query_args(
    descriptor: VectorFrameworkDescriptor,
    text: str,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Build args/kwargs for sync/async similarity query calls.

    Contract:
        query_method(query, k=TOP_K, **context)
        async_query_method: same parameters, async.
    """
    args: list[Any] = [text]
    kwargs: dict[str, Any] = {"k": TOP_K}
    _maybe_inject_context(descriptor, kwargs)
    return args, kwargs


def _build_stream_query_args(
    descriptor: VectorFrameworkDescriptor,
    text: str,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Build args/kwargs for sync/async streaming similarity query calls.

    Contract:
        stream_query_method(query, k=TOP_K, **context)
        async_stream_query_method: same parameters, async.
    """
    return _build_query_args(descriptor, text)


def _build_mmr_query_args(
    descriptor: VectorFrameworkDescriptor,
    text: str,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Build args/kwargs for sync/async MMR query calls.

    Contract:
        mmr_query_method(query, k=TOP_K, lambda_mult=MMR_LAMBDA, **context)
        async_mmr_query_method: same parameters, async.
    """
    args: list[Any] = [text]
    kwargs: dict[str, Any] = {"k": TOP_K, "lambda_mult": MMR_LAMBDA}
    _maybe_inject_context(descriptor, kwargs)
    return args, kwargs


# ---------------------------------------------------------------------------
# Core interface / surface contract tests
# ---------------------------------------------------------------------------


def test_can_instantiate_vector_client(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    Each registered framework descriptor should be instantiable with the
    pluggable Corpus vector adapter and any inferred kwargs.

    Sanity-check that the instance exposes the methods the descriptor claims.
    """
    # Required sync add & query methods
    _get_method(vector_client_instance, framework_descriptor.add_method)
    _get_method(vector_client_instance, framework_descriptor.query_method)

    # Optional delete method
    if framework_descriptor.delete_method:
        _get_method(vector_client_instance, framework_descriptor.delete_method)

    # Optional sync streaming (when declared)
    if framework_descriptor.stream_query_method:
        _get_method(
            vector_client_instance,
            framework_descriptor.stream_query_method,
        )

    # Optional MMR method (when declared)
    if framework_descriptor.mmr_query_method:
        _get_method(
            vector_client_instance,
            framework_descriptor.mmr_query_method,
        )

    # Async surfaces: not all-or-nothing; we exercise whatever is declared.
    if framework_descriptor.async_add_method:
        _get_method(
            vector_client_instance,
            framework_descriptor.async_add_method,
        )

    if framework_descriptor.async_delete_method:
        _get_method(
            vector_client_instance,
            framework_descriptor.async_delete_method,
        )

    if framework_descriptor.async_query_method:
        _get_method(
            vector_client_instance,
            framework_descriptor.async_query_method,
        )

    if framework_descriptor.async_stream_query_method:
        _get_method(
            vector_client_instance,
            framework_descriptor.async_stream_query_method,
        )

    if framework_descriptor.async_mmr_query_method:
        _get_method(
            vector_client_instance,
            framework_descriptor.async_mmr_query_method,
        )


def test_async_methods_exist_when_supports_async_true(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    Ensure that when supports_async=True, at least one async surface exists
    and is callable.

    Unlike graph, vector frameworks are allowed to be async-partial (e.g.
    async add/query but sync-only streaming). This test only asserts that
    async support is not a lie.
    """
    if not framework_descriptor.supports_async:
        pytest.skip("Framework does not declare async support")

    async_methods = [
        framework_descriptor.async_add_method,
        framework_descriptor.async_delete_method,
        framework_descriptor.async_query_method,
        framework_descriptor.async_stream_query_method,
        framework_descriptor.async_mmr_query_method,
    ]

    assert any(async_methods), (
        f"{framework_descriptor.name}: supports_async=True but no async "
        f"methods are declared"
    )

    for name in async_methods:
        if not name:
            continue
        fn = getattr(vector_client_instance, name, None)
        assert callable(fn), f"Async method {name!r} is not callable"


def test_sync_add_interface_conformance(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    Validate that the sync add method accepts batched test inputs and does
    not raise.

    Detailed persistence semantics are covered by framework-specific tests.
    """
    add_fn = _get_method(vector_client_instance, framework_descriptor.add_method)
    args, kwargs = _build_add_args(framework_descriptor)

    result = add_fn(*args, **kwargs)
    # Contract: may return None or framework-specific result; we only care
    # that the call succeeds.


@pytest.mark.asyncio
async def test_async_add_interface_conformance_when_supported(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    Validate that the async add method (when declared) accepts batched inputs
    and returns an awaitable that completes without error.
    """
    if not framework_descriptor.async_add_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async add",
        )

    aadd_fn = _get_method(
        vector_client_instance,
        framework_descriptor.async_add_method,
    )
    args, kwargs = _build_add_args(framework_descriptor)

    coro = aadd_fn(*args, **kwargs)
    assert inspect.isawaitable(coro), "Async add method must return an awaitable"
    await coro


def test_sync_delete_interface_conformance_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    Validate that the sync delete method (when declared) accepts a simple
    ids-based delete call and does not raise.
    """
    if not framework_descriptor.delete_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare delete_method",
        )

    delete_fn = _get_method(
        vector_client_instance,
        framework_descriptor.delete_method,
    )
    args, kwargs = _build_delete_args(framework_descriptor)

    result = delete_fn(*args, **kwargs)
    # We only assert that the call completes; semantics are framework-specific.


@pytest.mark.asyncio
async def test_async_delete_interface_conformance_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    Validate that the async delete method (when declared) accepts ids-based
    deletes and returns an awaitable.
    """
    if not framework_descriptor.async_delete_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async_delete_method",
        )

    adelete_fn = _get_method(
        vector_client_instance,
        framework_descriptor.async_delete_method,
    )
    args, kwargs = _build_delete_args(framework_descriptor)

    coro = adelete_fn(*args, **kwargs)
    assert inspect.isawaitable(coro), "Async delete method must return an awaitable"
    await coro


def test_sync_query_interface_conformance(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    Validate that the sync query method accepts a simple text query and
    returns a non-None result.

    Detailed result shape is covered by separate tests.
    """
    query_fn = _get_method(
        vector_client_instance,
        framework_descriptor.query_method,
    )
    args, kwargs = _build_query_args(
        framework_descriptor,
        SYNC_QUERY_TEXT,
    )

    result = query_fn(*args, **kwargs)
    assert result is not None


def test_sync_streaming_interface_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    Validate that the sync streaming method (when declared) accepts a text
    query and returns an iterable of chunks.

    Current vector frameworks declare sync-only streaming; async streaming is
    optional and tested separately when present.
    """
    if not framework_descriptor.stream_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare sync streaming",
        )

    stream_fn = _get_method(
        vector_client_instance,
        framework_descriptor.stream_query_method,
    )
    args, kwargs = _build_stream_query_args(
        framework_descriptor,
        SYNC_STREAM_TEXT,
    )

    iterator = stream_fn(*args, **kwargs)

    seen_any = False
    for _ in iterator:  # noqa: B007
        seen_any = True
        break

    # Contract: iterability is required; it's fine if no chunks are produced.
    assert iterator is not None
    assert isinstance(seen_any, bool)


@pytest.mark.asyncio
async def test_async_query_interface_conformance_when_supported(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    Validate that the async query method (when declared) accepts text input
    and returns a non-None result compatible with the sync API.
    """
    if not framework_descriptor.async_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async query",
        )

    aquery_fn = _get_method(
        vector_client_instance,
        framework_descriptor.async_query_method,
    )
    args, kwargs = _build_query_args(
        framework_descriptor,
        ASYNC_QUERY_TEXT,
    )

    coro = aquery_fn(*args, **kwargs)
    assert inspect.isawaitable(coro), "Async query method must return an awaitable"

    result = await coro
    assert result is not None


@pytest.mark.asyncio
async def test_async_streaming_interface_conformance_when_supported(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    Validate that the async streaming method (when declared) accepts text input
    and produces an async-iterable of chunks.

    The returned object may be an async iterator directly, or an awaitable
    that resolves to one.
    """
    if not framework_descriptor.async_stream_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async streaming",
        )

    astream_fn = _get_method(
        vector_client_instance,
        framework_descriptor.async_stream_query_method,
    )
    args, kwargs = _build_stream_query_args(
        framework_descriptor,
        ASYNC_STREAM_TEXT,
    )

    aiter = astream_fn(*args, **kwargs)
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    seen_any = False
    async for _ in aiter:  # noqa: B007
        seen_any = True
        break

    assert isinstance(seen_any, bool)


def test_sync_mmr_interface_conformance_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    Validate that the sync MMR method (when declared) accepts a text query
    and returns a non-None result.
    """
    if not framework_descriptor.mmr_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare mmr_query_method",
        )

    mmr_fn = _get_method(
        vector_client_instance,
        framework_descriptor.mmr_query_method,
    )
    args, kwargs = _build_mmr_query_args(
        framework_descriptor,
        MMR_QUERY_TEXT,
    )

    result = mmr_fn(*args, **kwargs)
    assert result is not None


@pytest.mark.asyncio
async def test_async_mmr_interface_conformance_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    Validate that the async MMR method (when declared) accepts a text query
    and returns a non-None result.
    """
    if not framework_descriptor.async_mmr_query_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async_mmr_query_method",
        )

    ammr_fn = _get_method(
        vector_client_instance,
        framework_descriptor.async_mmr_query_method,
    )
    args, kwargs = _build_mmr_query_args(
        framework_descriptor,
        ASYNC_MMR_QUERY_TEXT,
    )

    coro = ammr_fn(*args, **kwargs)
    assert inspect.isawaitable(coro), "Async MMR method must return an awaitable"

    result = await coro
    assert result is not None


def test_context_kwarg_is_accepted_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    If a context_kwarg is declared in the descriptor, the core query/add
    methods should accept that kwarg without raising TypeError.
    """
    if not framework_descriptor.context_kwarg:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare a context_kwarg",
        )

    ctx_kw = framework_descriptor.context_kwarg

    # Test against query method
    query_fn = _get_method(
        vector_client_instance,
        framework_descriptor.query_method,
    )
    args, kwargs = _build_query_args(
        framework_descriptor,
        CONTEXT_QUERY_TEXT,
    )
    kwargs[ctx_kw] = {"test": "value"}

    result = query_fn(*args, **kwargs)
    assert result is not None

    # Also verify add accepts the same context kwarg.
    add_fn = _get_method(
        vector_client_instance,
        framework_descriptor.add_method,
    )
    add_args, add_kwargs = _build_add_args(framework_descriptor)
    add_kwargs[ctx_kw] = {"test": "value"}
    add_fn(*add_args, **add_kwargs)


def test_method_signatures_consistent_between_sync_and_async(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    Verify that sync and async methods have consistent signatures
    (same parameters except maybe the return annotation), where both
    variants are declared.

    Covers add, delete, query, streaming, and MMR surfaces.
    """

    def _compare_signatures(sync_name: str | None, async_name: str | None) -> None:
        if not sync_name or not async_name:
            return

        sync_fn = _get_method(vector_client_instance, sync_name)
        async_fn = _get_method(vector_client_instance, async_name)

        sync_sig = inspect.signature(sync_fn)
        async_sig = inspect.signature(async_fn)

        # Skip "self" for bound methods
        sync_params = list(sync_sig.parameters.keys())[1:]
        async_params = list(async_sig.parameters.keys())[1:]

        assert (
            sync_params == async_params
        ), f"Signature mismatch between {sync_name!r} and {async_name!r}"

    # Add
    _compare_signatures(
        framework_descriptor.add_method,
        framework_descriptor.async_add_method,
    )

    # Delete
    _compare_signatures(
        framework_descriptor.delete_method,
        framework_descriptor.async_delete_method,
    )

    # Query
    _compare_signatures(
        framework_descriptor.query_method,
        framework_descriptor.async_query_method,
    )

    # Streaming
    _compare_signatures(
        framework_descriptor.stream_query_method,
        framework_descriptor.async_stream_query_method,
    )

    # MMR
    _compare_signatures(
        framework_descriptor.mmr_query_method,
        framework_descriptor.async_mmr_query_method,
    )


# ---------------------------------------------------------------------------
# Capabilities / health passthrough contract
# ---------------------------------------------------------------------------


def test_capabilities_contract_if_declared(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    If a framework declares has_capabilities=True, it should expose a
    capabilities() method returning a mapping. Async variants (when present)
    should behave similarly.
    """
    if not framework_descriptor.has_capabilities:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not expose capabilities",
        )

    capabilities = getattr(vector_client_instance, "capabilities", None)
    assert callable(capabilities), "capabilities() method is missing"

    caps_result = capabilities()
    assert isinstance(
        caps_result,
        Mapping,
    ), "capabilities() should return a mapping"

    async_caps = getattr(vector_client_instance, "acapabilities", None)
    if async_caps is not None and callable(async_caps):
        acaps_result = _run_async_if_needed(async_caps())
        assert isinstance(
            acaps_result,
            Mapping,
        ), "acapabilities() should return a mapping"


def test_health_contract_if_declared(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    If a framework declares has_health=True, it should expose a health()
    method returning a mapping. Async variants (when present) should behave
    similarly.
    """
    if not framework_descriptor.has_health:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not expose health",
        )

    health = getattr(vector_client_instance, "health", None)
    assert callable(health), "health() method is missing"

    health_result = health()
    assert isinstance(
        health_result,
        Mapping,
    ), "health() should return a mapping"

    async_health = getattr(vector_client_instance, "ahealth", None)
    if async_health is not None and callable(async_health):
        ahealth_result = _run_async_if_needed(async_health())
        assert isinstance(
            ahealth_result,
            Mapping,
        ), "ahealth() should return a mapping"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
