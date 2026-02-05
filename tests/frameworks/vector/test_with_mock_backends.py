# tests/frameworks/vector/test_with_mock_backends.py

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type

import pytest

from tests.frameworks.registries.vector_registry import (
    VectorFrameworkDescriptor,
    iter_vector_framework_descriptors,
)

# IMPORTANT: wire/protocol operation names use "vector.<op>"
VECTOR_OPERATION_PREFIX = "vector."
FAILURE_MESSAGE_SYNC = "intentional vector backend failure (sync)"


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
    underlying LlamaIndex / Semantic Kernel libraries are missing)
    are skipped via descriptor.is_available().
    """
    descriptor: VectorFrameworkDescriptor = request.param
    if not descriptor.is_available():
        pytest.skip(f"Framework '{descriptor.name}' not available in this environment")
    return descriptor


# ---------------------------------------------------------------------------
# Helpers: robust "construct a minimal instance" for protocol result classes
# ---------------------------------------------------------------------------


def _construct_minimal(cls: Type[Any], **overrides: Any) -> Any:
    """
    Best-effort constructor for protocol result objects.

    This avoids hardcoding dataclass signatures across SDK changes:
    - Uses inspect.signature to fill required params with safe defaults
    - Applies overrides (e.g., matches=[], upserted_count=0)
    """
    import inspect

    sig = inspect.signature(cls)  # type: ignore[arg-type]
    kwargs: Dict[str, Any] = {}

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if name in overrides:
            continue

        if param.default is not inspect._empty:
            # Let default apply by not providing it.
            continue

        # Required parameter: choose a conservative default by annotation/name.
        ann = param.annotation
        if name in {"matches", "failures", "items", "results"}:
            kwargs[name] = []
        elif name.endswith("_count") or name in {"count"}:
            kwargs[name] = 0
        elif name in {"ok", "success"}:
            kwargs[name] = True
        elif ann in (int, float, bool, str):
            if ann is int:
                kwargs[name] = 0
            elif ann is float:
                kwargs[name] = 0.0
            elif ann is bool:
                kwargs[name] = False
            else:
                kwargs[name] = ""
        else:
            # Fallback: None is usually accepted for optional-ish fields.
            kwargs[name] = None

    kwargs.update(overrides)
    return cls(**kwargs)  # type: ignore[misc]


def _get_method(instance: Any, name: str | None) -> Callable[..., Any]:
    """Helper to fetch a method from the instance and assert it is callable."""
    assert name is not None, "Method name must not be None"
    attr = getattr(instance, name, None)
    assert callable(attr), f"{instance!r} missing expected callable method {name!r}"
    return attr


def _make_client_with_evil_backend(
    framework_descriptor: VectorFrameworkDescriptor,
    backend_cls: Type[Any],
) -> Any:
    """
    Instantiate the framework vector client with an 'evil' backend.

    IMPORTANT:
    The registry is the source of truth:
    - import via descriptor.adapter_module
    - class via descriptor.adapter_class
    - inject underlying adapter using descriptor.adapter_init_kwarg (NOT hardcoded)
    """
    module = importlib.import_module(framework_descriptor.adapter_module)
    client_cls = getattr(module, framework_descriptor.adapter_class)

    backend = backend_cls()
    init_kw = {framework_descriptor.adapter_init_kwarg: backend}
    instance = client_cls(**init_kw)

    return instance


def _maybe_with_context_kwargs(
    descriptor: VectorFrameworkDescriptor,
) -> Dict[str, Any]:
    """
    Build framework-specific context kwargs if the wrapper supports a context kwarg.
    """
    if descriptor.context_kwarg:
        # Always pass an empty context object; content is not under test here.
        return {descriptor.context_kwarg: {}}
    return {}


def _patch_attach_context_best_effort(
    monkeypatch: pytest.MonkeyPatch,
    adapter_module: Any,
) -> list[tuple[BaseException, dict[str, Any]]]:
    """
    Patch attach_context where it is most likely to be invoked for protocol-level errors.

    We patch BOTH:
      1) The adapter module (framework module-local attach_context if used by decorators)
      2) The shared VectorTranslator module (protocol/wire op error context)

    The test asserts we capture at least one call across these patch points.
    """
    calls: list[tuple[BaseException, dict[str, Any]]] = []

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        calls.append((exc, ctx))

    # 1) Framework adapter module-local attach_context (if present)
    if hasattr(adapter_module, "attach_context"):
        monkeypatch.setattr(adapter_module, "attach_context", fake_attach_context)

    # 2) Shared translator module attach_context (protocol-focused)
    try:
        vt_module = importlib.import_module(
            "corpus_sdk.vector.framework_adapters.common.vector_translation"
        )
    except Exception:
        vt_module = None

    if vt_module is not None and hasattr(vt_module, "attach_context"):
        monkeypatch.setattr(vt_module, "attach_context", fake_attach_context)

    return calls


# ---------------------------------------------------------------------------
# "Evil" protocol backends (VectorProtocolV1-shaped)
# ---------------------------------------------------------------------------


class InvalidResultVectorAdapter:
    """
    Backend that returns blatantly invalid result objects for protocol methods.

    Used to test that query operations properly validate result types.
    """

    async def capabilities(self, *args: Any, **kwargs: Any) -> Any:
        return "not-a-capabilities-result"

    async def health(self, *args: Any, **kwargs: Any) -> Any:
        return "not-a-health-result"

    async def query(self, *args: Any, **kwargs: Any) -> Any:
        return "not-a-query-result"

    async def batch_query(self, *args: Any, **kwargs: Any) -> Any:
        return "not-a-batch-query-result"

    async def upsert(self, *args: Any, **kwargs: Any) -> Any:
        return "not-an-upsert-result"

    async def delete(self, *args: Any, **kwargs: Any) -> Any:
        return "not-a-delete-result"

    async def create_namespace(self, *args: Any, **kwargs: Any) -> Any:
        return "not-a-create-namespace-result"

    async def delete_namespace(self, *args: Any, **kwargs: Any) -> Any:
        return "not-a-delete-namespace-result"


class EmptyResultVectorAdapter:
    """
    Backend that returns empty-but-valid protocol results.

    Used to verify wrappers do not crash on degenerate (empty) results that may
    be valid in vector space (e.g., zero matches).
    """

    def __init__(self) -> None:
        from corpus_sdk.vector.vector_base import (
            DeleteResult,
            QueryResult,
            UpsertResult,
            VectorCapabilities,
        )

        self._QueryResult = QueryResult
        self._VectorCapabilities = VectorCapabilities
        self._UpsertResult = UpsertResult
        self._DeleteResult = DeleteResult

    async def capabilities(self, *args: Any, **kwargs: Any) -> Any:
        # Minimal caps; rely on defaults where possible.
        return _construct_minimal(self._VectorCapabilities)

    async def health(self, *args: Any, **kwargs: Any) -> Any:
        # Health surface varies; commonly a bool/dict. Keep it harmless.
        return {"ok": True}

    async def query(self, *args: Any, **kwargs: Any) -> Any:
        return _construct_minimal(self._QueryResult, matches=[])

    async def batch_query(self, raw_queries: Any, *args: Any, **kwargs: Any) -> Any:
        # If caller asks empty batch, empty batch result is valid.
        try:
            if raw_queries is None:
                return []
            if isinstance(raw_queries, Sequence) and len(raw_queries) == 0:
                return []
        except Exception:
            pass
        # Otherwise return a parallel structure with empty QueryResults where possible.
        try:
            n = len(raw_queries)  # type: ignore[arg-type]
        except Exception:
            n = 1
        return [_construct_minimal(self._QueryResult, matches=[]) for _ in range(int(n))]

    async def upsert(self, *args: Any, **kwargs: Any) -> Any:
        return _construct_minimal(self._UpsertResult, upserted_count=0, failed_count=0, failures=[])

    async def delete(self, *args: Any, **kwargs: Any) -> Any:
        return _construct_minimal(self._DeleteResult)

    async def create_namespace(self, *args: Any, **kwargs: Any) -> Any:
        return {"created": True}

    async def delete_namespace(self, *args: Any, **kwargs: Any) -> Any:
        return {"deleted": True}


class RaisingVectorAdapter:
    """
    Backend that always raises.

    Used to validate that error-context attachment occurs when failures originate
    in the underlying protocol backend.
    """

    async def capabilities(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    async def health(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    async def query(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    async def batch_query(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    async def upsert(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    async def delete(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    async def create_namespace(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)

    async def delete_namespace(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(FAILURE_MESSAGE_SYNC)


# ---------------------------------------------------------------------------
# Protocol-call helpers (registry-driven)
# ---------------------------------------------------------------------------


def _call_capabilities(descriptor: VectorFrameworkDescriptor, instance: Any) -> Any:
    fn = _get_method(instance, descriptor.capabilities_method)
    # Capabilities typically accept only the framework context kwarg if any.
    return fn(**_maybe_with_context_kwargs(descriptor))


def _call_health(descriptor: VectorFrameworkDescriptor, instance: Any) -> Any:
    fn = _get_method(instance, descriptor.health_method)
    return fn(**_maybe_with_context_kwargs(descriptor))


def _call_query(descriptor: VectorFrameworkDescriptor, instance: Any, raw_query: Any) -> Any:
    fn = _get_method(instance, descriptor.query_method)
    return fn(raw_query, **_maybe_with_context_kwargs(descriptor))


def _call_batch_query(
    descriptor: VectorFrameworkDescriptor, instance: Any, raw_queries: Any
) -> Any:
    assert descriptor.batch_query_method is not None
    fn = _get_method(instance, descriptor.batch_query_method)
    return fn(raw_queries, **_maybe_with_context_kwargs(descriptor))


def _call_upsert(descriptor: VectorFrameworkDescriptor, instance: Any, raw_docs: Any) -> Any:
    fn = _get_method(instance, descriptor.upsert_method)
    return fn(raw_docs, **_maybe_with_context_kwargs(descriptor))


def _call_delete(
    descriptor: VectorFrameworkDescriptor, instance: Any, raw_filter_or_ids: Any
) -> Any:
    fn = _get_method(instance, descriptor.delete_method)
    return fn(raw_filter_or_ids, **_maybe_with_context_kwargs(descriptor))


def _call_create_namespace(
    descriptor: VectorFrameworkDescriptor, instance: Any, name: str, dimensions: int = 384
) -> Any:
    fn = _get_method(instance, descriptor.create_namespace_method)
    # Namespace creation requires dimensions - pass via context if framework expects it
    ctx_kwargs = _maybe_with_context_kwargs(descriptor)
    if descriptor.context_kwarg and descriptor.context_kwarg in ctx_kwargs:
        # Add dimensions to framework context
        ctx_kwargs[descriptor.context_kwarg]["dimensions"] = dimensions
    return fn(name, **ctx_kwargs)


def _call_delete_namespace(
    descriptor: VectorFrameworkDescriptor, instance: Any, name: str
) -> Any:
    fn = _get_method(instance, descriptor.delete_namespace_method)
    return fn(name, **_maybe_with_context_kwargs(descriptor))


# ---------------------------------------------------------------------------
# Invalid result behavior (protocol-focused)
# ---------------------------------------------------------------------------


def test_invalid_backend_result_causes_errors_for_query(
    framework_descriptor: VectorFrameworkDescriptor,
) -> None:
    """
    If the backend returns a clearly invalid result type for query(), the
    wrapper/translator should surface an error rather than silently accepting it.
    """
    instance = _make_client_with_evil_backend(
        framework_descriptor,
        InvalidResultVectorAdapter,
    )

    with pytest.raises(Exception):  # noqa: BLE001
        _call_query(framework_descriptor, instance, {"vector": [0.0], "top_k": 1})


def test_invalid_backend_result_causes_errors_for_batch_query_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
) -> None:
    """
    If batch_query exists on the wrapper surface (as declared by the registry),
    invalid backend results for batch_query() should surface as errors.
    """
    if not framework_descriptor.has_batch_query:
        pytest.skip(f"Framework '{framework_descriptor.name}' does not expose batch_query surface")

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        InvalidResultVectorAdapter,
    )

    with pytest.raises(Exception):  # noqa: BLE001
        _call_batch_query(framework_descriptor, instance, [{"vector": [0.0], "top_k": 1}])


# ---------------------------------------------------------------------------
# Empty backend behavior (soft expectations)
# ---------------------------------------------------------------------------


def test_empty_backend_query_does_not_crash(
    framework_descriptor: VectorFrameworkDescriptor,
) -> None:
    """
    When the backend returns an empty-but-valid result for query(), the adapter
    should not crash. Empty matches may be valid.
    """
    instance = _make_client_with_evil_backend(
        framework_descriptor,
        EmptyResultVectorAdapter,
    )

    _call_query(framework_descriptor, instance, {"vector": [0.0], "top_k": 1})


def test_empty_backend_batch_query_does_not_crash_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
) -> None:
    """
    When batch_query is declared, an empty batch should not crash.
    """
    if not framework_descriptor.has_batch_query:
        pytest.skip(f"Framework '{framework_descriptor.name}' does not expose batch_query surface")

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        EmptyResultVectorAdapter,
    )

    # Empty batch in -> empty batch out is valid.
    _call_batch_query(framework_descriptor, instance, [])


# ---------------------------------------------------------------------------
# Error-context when backend raises (protocol-focused)
# ---------------------------------------------------------------------------


def _assert_error_context_calls(
    calls: list[tuple[BaseException, dict[str, Any]]],
    *,
    framework_name: str,
) -> None:
    assert calls, "attach_context was not called for backend failure"

    exc, ctx = calls[-1]
    assert isinstance(exc, BaseException)

    # We require these stable observability keys.
    assert "framework" in ctx
    assert "operation" in ctx

    assert ctx["framework"] == framework_name

    # Operation should follow base wire naming: "vector.<op>"
    op_val = str(ctx["operation"])
    assert op_val.startswith(VECTOR_OPERATION_PREFIX), f"operation={op_val!r} did not start with {VECTOR_OPERATION_PREFIX!r}"


def test_backend_exception_is_wrapped_with_error_context_on_query(
    framework_descriptor: VectorFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the backend raises during a sync query operation, the protocol layer
    should attach error context and re-raise the exception.
    """
    adapter_module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context_best_effort(monkeypatch, adapter_module)

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        RaisingVectorAdapter,
    )

    with pytest.raises(RuntimeError, match="backend failure"):
        _call_query(framework_descriptor, instance, {"vector": [0.0], "top_k": 1})

    _assert_error_context_calls(calls, framework_name=framework_descriptor.name)


def test_backend_exception_is_wrapped_with_error_context_on_batch_query_when_declared(
    framework_descriptor: VectorFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Same as query error-context test, but for batch_query when the wrapper surface exists.
    """
    if not framework_descriptor.has_batch_query:
        pytest.skip(f"Framework '{framework_descriptor.name}' does not expose batch_query surface")

    adapter_module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context_best_effort(monkeypatch, adapter_module)

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        RaisingVectorAdapter,
    )

    with pytest.raises(RuntimeError, match="backend failure"):
        _call_batch_query(framework_descriptor, instance, [{"vector": [0.0], "top_k": 1}])

    _assert_error_context_calls(calls, framework_name=framework_descriptor.name)


def test_backend_exception_is_wrapped_with_error_context_on_capabilities_and_health(
    framework_descriptor: VectorFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Validate context attachment for capabilities() and health() failures.
    """
    adapter_module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context_best_effort(monkeypatch, adapter_module)

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        RaisingVectorAdapter,
    )

    with pytest.raises(RuntimeError, match="backend failure"):
        _call_capabilities(framework_descriptor, instance)

    _assert_error_context_calls(calls, framework_name=framework_descriptor.name)

    # Reset calls and test health separately (clearer failure attribution)
    calls.clear()

    with pytest.raises(RuntimeError, match="backend failure"):
        _call_health(framework_descriptor, instance)

    _assert_error_context_calls(calls, framework_name=framework_descriptor.name)


def test_backend_exception_is_wrapped_with_error_context_on_upsert_delete_and_namespace_ops(
    framework_descriptor: VectorFrameworkDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Validate context attachment for upsert/delete/namespace operations failures.
    """
    adapter_module = importlib.import_module(framework_descriptor.adapter_module)
    calls = _patch_attach_context_best_effort(monkeypatch, adapter_module)

    instance = _make_client_with_evil_backend(
        framework_descriptor,
        RaisingVectorAdapter,
    )

    with pytest.raises(RuntimeError, match="backend failure"):
        _call_upsert(framework_descriptor, instance, {"id": "test-id", "vector": [0.1, 0.2], "metadata": {}})
    _assert_error_context_calls(calls, framework_name=framework_descriptor.name)

    calls.clear()
    with pytest.raises(RuntimeError, match="backend failure"):
        _call_delete(framework_descriptor, instance, {"ids": []})
    _assert_error_context_calls(calls, framework_name=framework_descriptor.name)

    calls.clear()
    with pytest.raises(RuntimeError, match="backend failure"):
        _call_create_namespace(framework_descriptor, instance, "ns-err")
    _assert_error_context_calls(calls, framework_name=framework_descriptor.name)

    calls.clear()
    with pytest.raises(RuntimeError, match="backend failure"):
        _call_delete_namespace(framework_descriptor, instance, "ns-err")
    _assert_error_context_calls(calls, framework_name=framework_descriptor.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
