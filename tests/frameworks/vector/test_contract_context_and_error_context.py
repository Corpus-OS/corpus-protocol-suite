# tests/frameworks/vector/test_contract_context_and_error_context.py

from __future__ import annotations

import importlib
import inspect
import os
from typing import Any, Callable, Optional

import pytest

from tests.frameworks.registries.vector_registry import (
    VectorFrameworkDescriptor,
    iter_vector_framework_descriptors,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAILURE_MESSAGE = "intentional failure from failing vector adapter"
VECTOR_OPERATION_PREFIX = "vector."
RICH_CONTEXT = {
    "request_id": "req-123",
    "user_id": "user-abc",
    "tags": ["test"],
    "nested": {"key": "value", "depth": 2},
}

# In these tests, we only care that calls succeed/fail in the expected way.
# We intentionally keep payloads simple and framework-agnostic.
QUERY_INPUT = {"vector": [0.1, 0.2], "top_k": 3}
UPSERT_INPUT = {"id": "doc-1", "vector": [0.1, 0.2], "metadata": {"k": "v"}}
DELETE_INPUT = "doc-1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _seed_default_namespace_for_mock_adapter() -> None:
    """Ensure the mock vector adapter has a usable default namespace.

    The contract tests in this module are about context handling and centralized
    error-context attachment; they should not fail due to an intentionally
    unseeded in-memory mock namespace.
    """
    os.environ.setdefault("VECTOR_SEED_DEFAULT", "1")


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
    underlying LangChain / LlamaIndex / Semantic Kernel libs are missing)
    are skipped via descriptor.is_available().
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
    Construct a vector client instance for the given descriptor.

    IMPORTANT:
    We use descriptor.adapter_init_kwarg so conformance tests do not hardcode
    constructor argument names.
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    client_cls = getattr(module, framework_descriptor.adapter_class)

    init_kwargs: dict[str, Any] = {framework_descriptor.adapter_init_kwarg: adapter}
    return client_cls(**init_kwargs)


@pytest.fixture
def failing_adapter() -> Any:
    """
    A minimal vector adapter whose core VectorProtocolV1 methods always fail.

    This is used ONLY for error-context tests to ensure attach_context() is invoked
    and the exception propagates.
    """

    class FailingVectorAdapter:
        async def query(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        async def upsert(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        async def delete(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        async def capabilities(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        async def create_namespace(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        async def delete_namespace(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

        async def health(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(FAILURE_MESSAGE)

    return FailingVectorAdapter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_method(instance: Any, name: Optional[str]) -> Callable[..., Any]:
    """
    Fetch a method from the instance and assert it is callable.
    """
    assert name, "Expected a non-empty method name"
    attr = getattr(instance, name, None)
    assert callable(attr), f"{instance!r} missing expected callable method {name!r}"
    return attr


async def _maybe_await(value: Any) -> Any:
    """
    Await the value if it is awaitable, otherwise return it unchanged.

    This keeps tests robust across wrappers that expose sync methods wrapping
    async internals vs wrappers that expose native async methods.
    """
    if inspect.isawaitable(value):
        return await value
    return value


def _call_with_optional_context(
    descriptor: VectorFrameworkDescriptor,
    fn: Callable[..., Any],
    *args: Any,
    context: Any = None,
    **kwargs: Any,
) -> Any:
    """
    Call a method and inject context using descriptor.context_kwarg (if any).

    If context_kwarg is not declared, context is ignored.
    """
    if descriptor.context_kwarg and context is not None:
        kwargs = {**kwargs, descriptor.context_kwarg: context}
    return fn(*args, **kwargs)


def _build_error_wrapped_client_instance(
    framework_descriptor: VectorFrameworkDescriptor,
    failing_adapter: Any,
) -> Any:
    """
    Construct a vector client instance wired to a failing vector adapter.
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    client_cls = getattr(module, framework_descriptor.adapter_class)
    init_kwargs: dict[str, Any] = {
        framework_descriptor.adapter_init_kwarg: failing_adapter,
    }
    return client_cls(**init_kwargs)


def _patch_attach_context_centrally(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[BaseException, dict[str, Any]]]:
    """
    Patch the centralized attach_context imported by the common translator layer.

    New approach:
    Error context is attached in the framework-agnostic orchestration layer,
    not bespoke per-framework method wrappers.
    """
    import corpus_sdk.vector.framework_adapters.common.vector_translation as vt

    calls: list[tuple[BaseException, dict[str, Any]]] = []

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        calls.append((exc, ctx))

    monkeypatch.setattr(vt, "attach_context", fake_attach_context)
    return calls


# ---------------------------------------------------------------------------
# Context contract tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_accepts_context_mapping_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    If a framework declares context_kwarg, query should accept a rich Mapping
    and not fail due to context shape.
    """
    if not framework_descriptor.context_kwarg:
        pytest.skip(f"Framework '{framework_descriptor.name}' does not declare a context_kwarg")

    query_method = _get_method(vector_client_instance, framework_descriptor.query_method)

    rich_context = {**RICH_CONTEXT, "tags": [*RICH_CONTEXT["tags"], framework_descriptor.name]}

    result = _call_with_optional_context(
        framework_descriptor,
        query_method,
        QUERY_INPUT,
        context=rich_context,
    )
    out = await _maybe_await(result)
    assert out is not None


@pytest.mark.asyncio
async def test_query_context_is_optional_even_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    Query must still work when context is omitted.
    """
    query_method = _get_method(vector_client_instance, framework_descriptor.query_method)
    out = await _maybe_await(query_method(QUERY_INPUT))
    assert out is not None


@pytest.mark.asyncio
async def test_invalid_context_type_behavior_is_consistent(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    New approach expectation:
    - If context_kwarg is declared, invalid context types may raise (BadRequest)
      OR may be ignored by the wrapper if it sanitizes before passing to core.
    - This test enforces "no AttributeError explosions" and makes behavior explicit.

    We accept either:
    - success (context ignored/sanitized), OR
    - a clean exception type (ValueError/TypeError/BadRequest family)
      BUT we do not accept unhandled AttributeError-style crashes.
    """
    if not framework_descriptor.context_kwarg:
        pytest.skip(f"Framework '{framework_descriptor.name}' does not declare a context_kwarg")

    query_method = _get_method(vector_client_instance, framework_descriptor.query_method)

    invalid_contexts = ["not-a-mapping", 12345]

    for invalid_ctx in invalid_contexts:
        try:
            result = _call_with_optional_context(
                framework_descriptor,
                query_method,
                QUERY_INPUT,
                context=invalid_ctx,
            )
            out = await _maybe_await(result)
            assert out is not None
        except Exception as exc:  # noqa: BLE001
            assert not isinstance(exc, AttributeError), (
                f"{framework_descriptor.name}: invalid context caused AttributeError crash: {exc!r}"
            )


def test_context_injection_does_not_occur_when_context_kwarg_is_none() -> None:
    """
    Ensure _call_with_optional_context does not leak a 'context' kwarg when
    descriptor.context_kwarg is None.

    This is a pure unit test to prevent accidental kwargs injection into
    wrappers that do not support it.
    """

    class RecordingCallable:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            self.kwargs = dict(kwargs)
            return {"ok": True}

    fn = RecordingCallable()
    desc = VectorFrameworkDescriptor(
        name="unit_no_ctx",
        adapter_module="test.module",
        adapter_class="TestClass",
        context_kwarg=None,
    )

    out = _call_with_optional_context(desc, fn, QUERY_INPUT, context={"should": "not-pass"})
    assert out == {"ok": True}
    assert "context" not in fn.kwargs
    assert fn.kwargs == {}, "No kwargs should be injected when context_kwarg is None"


# ---------------------------------------------------------------------------
# Error-context contract tests (centralized attach_context)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_context_attached_on_query_failure(
    framework_descriptor: VectorFrameworkDescriptor,
    failing_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    On query failure, attach_context must be called with useful metadata and
    operation name starting with 'vector.'.
    """
    calls = _patch_attach_context_centrally(monkeypatch)
    instance = _build_error_wrapped_client_instance(framework_descriptor, failing_adapter)
    query_method = _get_method(instance, framework_descriptor.query_method)

    with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
        result = _call_with_optional_context(
            framework_descriptor,
            query_method,
            QUERY_INPUT,
            context={} if framework_descriptor.context_kwarg else None,
        )
        await _maybe_await(result)

    assert calls, "attach_context was not called on query failure"
    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert str(ctx.get("operation", "")).startswith(VECTOR_OPERATION_PREFIX)


@pytest.mark.asyncio
async def test_error_context_attached_on_upsert_failure(
    framework_descriptor: VectorFrameworkDescriptor,
    failing_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    On upsert failure, attach_context must be called.
    """
    calls = _patch_attach_context_centrally(monkeypatch)
    instance = _build_error_wrapped_client_instance(framework_descriptor, failing_adapter)
    upsert_method = _get_method(instance, framework_descriptor.upsert_method)

    with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
        result = _call_with_optional_context(
            framework_descriptor,
            upsert_method,
            UPSERT_INPUT,
            context={} if framework_descriptor.context_kwarg else None,
        )
        await _maybe_await(result)

    assert calls, "attach_context was not called on upsert failure"
    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert str(ctx.get("operation", "")).startswith(VECTOR_OPERATION_PREFIX)


@pytest.mark.asyncio
async def test_error_context_attached_on_delete_failure(
    framework_descriptor: VectorFrameworkDescriptor,
    failing_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    On delete failure, attach_context must be called.
    """
    calls = _patch_attach_context_centrally(monkeypatch)
    instance = _build_error_wrapped_client_instance(framework_descriptor, failing_adapter)
    delete_method = _get_method(instance, framework_descriptor.delete_method)

    with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
        result = _call_with_optional_context(
            framework_descriptor,
            delete_method,
            DELETE_INPUT,
            context={} if framework_descriptor.context_kwarg else None,
        )
        await _maybe_await(result)

    assert calls, "attach_context was not called on delete failure"
    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert str(ctx.get("operation", "")).startswith(VECTOR_OPERATION_PREFIX)


@pytest.mark.asyncio
async def test_error_context_attached_on_capabilities_failure(
    framework_descriptor: VectorFrameworkDescriptor,
    failing_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    On capabilities failure, attach_context must be called.
    """
    calls = _patch_attach_context_centrally(monkeypatch)
    instance = _build_error_wrapped_client_instance(framework_descriptor, failing_adapter)
    capabilities_method = _get_method(instance, framework_descriptor.capabilities_method)

    with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
        result = _call_with_optional_context(
            framework_descriptor,
            capabilities_method,
            context={} if framework_descriptor.context_kwarg else None,
        )
        await _maybe_await(result)

    assert calls, "attach_context was not called on capabilities failure"
    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert str(ctx.get("operation", "")).startswith(VECTOR_OPERATION_PREFIX)


@pytest.mark.asyncio
async def test_error_context_attached_on_health_failure(
    framework_descriptor: VectorFrameworkDescriptor,
    failing_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    On health failure, attach_context must be called.
    """
    calls = _patch_attach_context_centrally(monkeypatch)
    instance = _build_error_wrapped_client_instance(framework_descriptor, failing_adapter)
    health_method = _get_method(instance, framework_descriptor.health_method)

    with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
        result = _call_with_optional_context(
            framework_descriptor,
            health_method,
            context={} if framework_descriptor.context_kwarg else None,
        )
        await _maybe_await(result)

    assert calls, "attach_context was not called on health failure"
    exc, ctx = calls[-1]
    assert isinstance(exc, RuntimeError)
    assert "framework" in ctx
    assert "operation" in ctx
    assert str(ctx.get("operation", "")).startswith(VECTOR_OPERATION_PREFIX)


@pytest.mark.asyncio
async def test_error_context_attached_on_namespace_ops_failure(
    framework_descriptor: VectorFrameworkDescriptor,
    failing_adapter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    On create/delete namespace failures, attach_context must be called.

    We cover both namespace ops in one test to keep total count reasonable.
    """
    calls = _patch_attach_context_centrally(monkeypatch)
    instance = _build_error_wrapped_client_instance(framework_descriptor, failing_adapter)

    create_ns = _get_method(instance, framework_descriptor.create_namespace_method)
    delete_ns = _get_method(instance, framework_descriptor.delete_namespace_method)

    with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
        result = _call_with_optional_context(
            framework_descriptor,
            create_ns,
            "ns-test",
            context={} if framework_descriptor.context_kwarg else None,
        )
        await _maybe_await(result)

    with pytest.raises(RuntimeError, match=FAILURE_MESSAGE):
        result = _call_with_optional_context(
            framework_descriptor,
            delete_ns,
            "ns-test",
            context={} if framework_descriptor.context_kwarg else None,
        )
        await _maybe_await(result)

    assert calls, "attach_context was not called on namespace op failures"
    for exc, ctx in calls[-2:]:
        assert isinstance(exc, RuntimeError)
        assert "framework" in ctx
        assert "operation" in ctx
        assert str(ctx.get("operation", "")).startswith(VECTOR_OPERATION_PREFIX)


# ---------------------------------------------------------------------------
# Registry/descriptor alignment sanity tests (this file only)
# ---------------------------------------------------------------------------


def test_descriptor_method_names_are_non_empty_strings(framework_descriptor: VectorFrameworkDescriptor) -> None:
    """
    Ensure descriptor method name fields required by the registry contract
    are always non-empty strings.
    """
    required = [
        framework_descriptor.capabilities_method,
        framework_descriptor.query_method,
        framework_descriptor.upsert_method,
        framework_descriptor.delete_method,
        framework_descriptor.create_namespace_method,
        framework_descriptor.delete_namespace_method,
        framework_descriptor.health_method,
    ]
    for name in required:
        assert isinstance(name, str) and name.strip()


def test_client_exposes_required_methods(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    Ensure the constructed client exposes the required callable methods.
    """
    for attr_name in (
        framework_descriptor.capabilities_method,
        framework_descriptor.query_method,
        framework_descriptor.upsert_method,
        framework_descriptor.delete_method,
        framework_descriptor.create_namespace_method,
        framework_descriptor.delete_namespace_method,
        framework_descriptor.health_method,
    ):
        _get_method(vector_client_instance, attr_name)


def test_client_exposes_batch_query_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
    vector_client_instance: Any,
) -> None:
    """
    If the registry declares has_batch_query=True, the client must expose a
    callable batch_query_method.

    This is a surface-shape check only; runtime support is determined by
    capabilities().
    """
    if not framework_descriptor.has_batch_query:
        pytest.skip(f"Framework '{framework_descriptor.name}' does not declare batch query surface")

    assert framework_descriptor.batch_query_method, (
        f"{framework_descriptor.name}: has_batch_query=True but batch_query_method is not set"
    )
    _get_method(vector_client_instance, framework_descriptor.batch_query_method)


def test_adapter_init_kwarg_is_respected_with_nonstandard_kwarg() -> None:
    """
    Pure unit test that adapter_init_kwarg is respected during construction.

    This prevents tests from accidentally hardcoding 'adapter=...' and ensures
    registry-driven construction remains correct even for nonstandard wrappers.
    """

    class SyntheticClient:
        def __init__(self, corpus_adapter: Any) -> None:
            self._adapter = corpus_adapter

    sentinel_adapter = object()
    desc = VectorFrameworkDescriptor(
        name="unit_nonstandard_init_kwarg",
        adapter_module="test.module",
        adapter_class="SyntheticClient",
        adapter_init_kwarg="corpus_adapter",
    )

    init_kwargs: dict[str, Any] = {desc.adapter_init_kwarg: sentinel_adapter}
    instance = SyntheticClient(**init_kwargs)

    assert instance._adapter is sentinel_adapter


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
