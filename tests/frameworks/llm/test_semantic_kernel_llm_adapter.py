# tests/frameworks/llm/test_semantic_kernel_llm_adapter.py

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Dict, List

import pytest

import corpus_sdk.llm.framework_adapters.semantic_kernel as sk_llm_module
from corpus_sdk.llm.framework_adapters.semantic_kernel import (
    CorpusSemanticKernelLLM,
    SEMANTIC_KERNEL_AVAILABLE,
    configure_semantic_kernel_llm,
    register_with_semantic_kernel_llm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm(adapter: Any, **kwargs: Any) -> CorpusSemanticKernelLLM:
    """
    Construct a CorpusSemanticKernelLLM instance from the generic adapter.

    We assume the public API uses `llm_adapter=` for consistency with
    other LLM framework adapters.
    """
    return CorpusSemanticKernelLLM(llm_adapter=adapter, **kwargs)


def _assert_llm_text_like(result: Any) -> None:
    """
    Validate that an LLM output is "text-like".

    For Semantic Kernel we intentionally allow:
    - bare strings, or
    - objects with a `.content`, `.text`, or `.value` attribute (SK-style).
    """
    assert result is not None

    if isinstance(result, str):
        return

    for attr in ("content", "text", "value"):
        val = getattr(result, attr, None)
        if isinstance(val, str):
            return

    raise AssertionError(
        f"LLM result is not text-like: {type(result).__name__}",
    )


def _collect_first_n(iterable, n: int) -> list[Any]:
    """Collect at most n items from a (sync) iterable."""
    out: list[Any] = []
    for item in iterable:
        out.append(item)
        if len(out) >= n:
            break
    return out


async def _acollect_first_n(aiterable, n: int) -> list[Any]:
    """Collect at most n items from an async iterable."""
    out: list[Any] = []
    async for item in aiterable:  # noqa: B007
        out.append(item)
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# Construction / basic configuration
# ---------------------------------------------------------------------------


def test_constructor_rejects_adapter_without_complete() -> None:
    """
    CorpusSemanticKernelLLM should enforce that llm_adapter exposes a
    `complete`-style surface; otherwise __init__ should raise TypeError.
    """

    class BadAdapter:
        # deliberately missing `complete` / LLMProtocol surface
        def __init__(self) -> None:
            pass

    with pytest.raises(TypeError) as exc_info:
        CorpusSemanticKernelLLM(llm_adapter=BadAdapter())

    msg = str(exc_info.value)
    # The exact wording may differ, but it should mention LLMProtocol/required methods.
    assert "LLMProtocol" in msg or "must implement" in msg


def test_constructor_accepts_valid_llm_adapter(adapter: Any) -> None:
    """
    A valid llm_adapter implementing the LLM protocol should be accepted
    and stored as-is on the model.
    """
    llm = _make_llm(adapter, model="sk-llm-model")

    assert llm is not None
    assert getattr(llm, "model", None) == "sk-llm-model"


def test_configure_and_register_helpers_return_llm(adapter: Any) -> None:
    """
    configure_semantic_kernel_llm and register_with_semantic_kernel_llm should
    both return CorpusSemanticKernelLLM instances wired to the given adapter.
    """
    llm1 = configure_semantic_kernel_llm(
        llm_adapter=adapter,
        model="cfg-sk-llm-model",
    )
    assert isinstance(llm1, CorpusSemanticKernelLLM)

    llm2 = register_with_semantic_kernel_llm(
        llm_adapter=adapter,
        model="reg-sk-llm-model",
    )
    assert isinstance(llm2, CorpusSemanticKernelLLM)


def test_SEMANTIC_KERNEL_AVAILABLE_is_bool() -> None:
    """
    SEMANTIC_KERNEL_AVAILABLE flag should always be a boolean, regardless of
    whether Semantic Kernel is actually installed.
    """
    assert isinstance(SEMANTIC_KERNEL_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# Semantic Kernel interface compatibility
# ---------------------------------------------------------------------------


def test_semantic_kernel_interface_compatibility(adapter: Any) -> None:
    """
    Verify that CorpusSemanticKernelLLM implements the expected Semantic Kernel
    chat completion interface when SK is available, and that core surfaces
    exist regardless.
    """
    llm = _make_llm(adapter, model="iface-sk-llm-model")

    # Core methods we expect on the adapter
    assert hasattr(llm, "complete")
    assert hasattr(llm, "acomplete")
    assert hasattr(llm, "stream_complete")
    assert hasattr(llm, "astream_complete")

    if not SEMANTIC_KERNEL_AVAILABLE:
        pytest.skip(
            "Semantic Kernel is not available; cannot assert SK base class compatibility",
        )

    # Try the modern ChatCompletionClientBase import; if that fails, skip.
    try:
        from semantic_kernel.connectors.ai.chat_completion_client_base import (  # type: ignore[import]
            ChatCompletionClientBase,
        )
    except Exception:
        pytest.skip(
            "SEMANTIC_KERNEL_AVAILABLE is True but could not import ChatCompletionClientBase",
        )

    assert isinstance(
        llm,
        ChatCompletionClientBase,
    ), "CorpusSemanticKernelLLM should subclass ChatCompletionClientBase when available"


# ---------------------------------------------------------------------------
# SK context translation / ContextTranslator mapping
# ---------------------------------------------------------------------------


def test_sk_context_passed_to_context_translation_for_complete(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Verify that the `sk_context` kwarg is passed through to
    ContextTranslator.from_semantic_kernel when issuing a completion.
    """
    captured: Dict[str, Any] = {}

    def fake_from_semantic_kernel(
        sk_context: Dict[str, Any],
        framework_version: str | None = None,
    ) -> Any:
        captured["sk_context"] = sk_context
        captured["framework_version"] = framework_version
        # Return a real OperationContext instance so type checks pass.
        return sk_llm_module.OperationContext()

    monkeypatch.setattr(
        sk_llm_module.ContextTranslator,
        "from_semantic_kernel",
        staticmethod(fake_from_semantic_kernel),
    )

    llm = _make_llm(
        adapter,
        model="cfg-sk-llm",
        framework_version="sk-llm-fw-test",
    )

    sk_ctx = {
        "kernel_name": "test-kernel",
        "skill_name": "test-skill",
        "function_name": "test-function",
        "tenant": "tenant-123",
    }

    result = llm.complete("Hello from SK", sk_context=sk_ctx)
    _assert_llm_text_like(result)

    assert captured.get("sk_context") is sk_ctx
    assert captured.get("framework_version") == "sk-llm-fw-test"


# ---------------------------------------------------------------------------
# Error-context decorator behavior (framework + SK metadata)
# ---------------------------------------------------------------------------


def test_error_context_includes_semantic_kernel_metadata_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When an error occurs during a sync SK LLM operation, error context should
    include Semantic Kernel metadata via attach_context().
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(
        sk_llm_module,
        "attach_context",
        fake_attach_context,
    )

    # Minimal LLMProtocol adapter; methods won't actually be used directly.
    class MinimalAdapter:
        def complete(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return "ok"

        def stream(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return iter(())

        def count_tokens(self, *args: Any, **kwargs: Any) -> int:  # noqa: ARG002
            return 0

        def capabilities(self) -> Dict[str, Any]:
            return {}

        def health(self) -> Dict[str, Any]:
            return {}

    llm = CorpusSemanticKernelLLM(llm_adapter=MinimalAdapter())

    # Force the error at the translator layer so the decorator is exercised.
    class FailingTranslator:
        def complete(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from semantic kernel llm adapter")

    llm._translator = FailingTranslator()  # type: ignore[attr-defined]

    sk_ctx = {"kernel_name": "err-kernel", "skill_name": "err-skill"}

    with pytest.raises(RuntimeError, match="test error from semantic kernel llm adapter"):
        llm.complete("trigger failure", sk_context=sk_ctx)

    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "semantic_kernel"
    op = str(captured_context.get("operation", ""))
    assert op.startswith("llm_")

    # Best-effort SK-specific metadata assertions (optional but encouraged).
    if "kernel_name" in captured_context:
        assert captured_context["kernel_name"] == "err-kernel"
    if "skill_name" in captured_context:
        assert captured_context["skill_name"] == "err-skill"


@pytest.mark.asyncio
async def test_error_context_includes_semantic_kernel_metadata_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Same as the sync error-context test but for the async acomplete path.
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(
        sk_llm_module,
        "attach_context",
        fake_attach_context,
    )

    class MinimalAdapter:
        async def acomplete(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return "ok"

        def complete(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return "ok"

        def stream(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return iter(())

        def count_tokens(self, *args: Any, **kwargs: Any) -> int:  # noqa: ARG002
            return 0

        def capabilities(self) -> Dict[str, Any]:
            return {}

        def health(self) -> Dict[str, Any]:
            return {}

    llm = CorpusSemanticKernelLLM(llm_adapter=MinimalAdapter())

    class FailingTranslator:
        async def arun_complete(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test async error from semantic kernel llm adapter")

    llm._translator = FailingTranslator()  # type: ignore[attr-defined]

    sk_ctx = {"kernel_name": "async-kernel", "skill_name": "async-skill"}

    with pytest.raises(
        RuntimeError,
        match="test async error from semantic kernel llm adapter",
    ):
        await llm.acomplete("trigger async failure", sk_context=sk_ctx)

    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "semantic_kernel"
    op = str(captured_context.get("operation", ""))
    assert op.startswith("llm_")
    if "kernel_name" in captured_context:
        assert captured_context["kernel_name"] == "async-kernel"
    if "skill_name" in captured_context:
        assert captured_context["skill_name"] == "async-skill"


# ---------------------------------------------------------------------------
# Sync / async semantics
# ---------------------------------------------------------------------------


def test_sync_complete_basic(adapter: Any) -> None:
    """
    Basic smoke test for sync complete() behavior: it should accept simple
    text input and return a text-like result.
    """
    llm = configure_semantic_kernel_llm(
        llm_adapter=adapter,
        model="sync-sk-llm-model",
    )

    prompt = "Say hello in one short sentence."
    result = llm.complete(prompt)
    _assert_llm_text_like(result)


@pytest.mark.asyncio
async def test_async_acomplete_basic(adapter: Any) -> None:
    """
    Async acomplete() should be a coroutine function and produce a text-like
    result compatible with sync complete().
    """
    llm = configure_semantic_kernel_llm(
        llm_adapter=adapter,
        model="async-sk-llm-model",
    )

    assert hasattr(llm, "acomplete")
    assert inspect.iscoroutinefunction(llm.acomplete)

    prompt = "Give a very short async SK response."
    result = await llm.acomplete(prompt)
    _assert_llm_text_like(result)


@pytest.mark.asyncio
async def test_async_and_sync_complete_same_output_type(adapter: Any) -> None:
    """
    Sync complete and async acomplete for the same prompt should produce
    outputs of the same *type* (string vs SK object), even if contents differ.
    """
    llm = configure_semantic_kernel_llm(
        llm_adapter=adapter,
        model="type-parity-sk-llm",
    )

    prompt = "Explain type parity briefly."

    sync_result = llm.complete(prompt)
    async_result = await llm.acomplete(prompt)

    assert type(sync_result) is type(
        async_result,
    ), (
        f"complete()/acomplete() returned different types: "
        f"{type(sync_result).__name__} vs {type(async_result).__name__}"
    )


# ---------------------------------------------------------------------------
# Streaming semantics
# ---------------------------------------------------------------------------


def test_stream_complete_basic(adapter: Any) -> None:
    """
    stream_complete() should return an iterable of chunks. We don't enforce a
    specific chunk shape, only that collected chunks are text-like.
    """
    llm = configure_semantic_kernel_llm(
        llm_adapter=adapter,
        model="stream-sk-llm-model",
    )

    iterator = llm.stream_complete("Stream a short SK reply.")
    chunks = _collect_first_n(iterator, 5)

    assert isinstance(chunks, list)
    for chunk in chunks:
        _assert_llm_text_like(chunk)


@pytest.mark.asyncio
async def test_async_astream_complete_basic(adapter: Any) -> None:
    """
    astream_complete() should return an async-iterable of chunks, either
    directly or via an awaitable resolving to one.
    """
    llm = configure_semantic_kernel_llm(
        llm_adapter=adapter,
        model="astream-sk-llm-model",
    )

    aiter = llm.astream_complete("Async stream a short SK reply.")
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    assert hasattr(aiter, "__aiter__")

    chunks = await _acollect_first_n(aiter, 5)
    assert isinstance(chunks, list)
    for chunk in chunks:
        _assert_llm_text_like(chunk)


# ---------------------------------------------------------------------------
# Token counting semantics (translator + fallback)
# ---------------------------------------------------------------------------


def test_get_num_tokens_uses_translator_when_available(
    adapter: Any,
) -> None:
    """
    get_num_tokens() should use the translator-based token counting when
    available and return the integer it produces.
    """
    llm = _make_llm(adapter, model="count-sk-llm")

    class DummyTranslator:
        def count_tokens_for_messages(
            self,
            *args: Any,
            **kwargs: Any,  # noqa: ARG002
        ) -> Mapping[str, Any]:
            # Return a mapping to exercise the dict path.
            return {"tokens": 42}

    llm._translator = DummyTranslator()  # type: ignore[attr-defined]

    tokens = llm.get_num_tokens("hello semantic kernel")
    assert isinstance(tokens, int)
    assert tokens == 42


def test_get_num_tokens_falls_back_on_error(adapter: Any) -> None:
    """
    When translator-based token counting fails, get_num_tokens() should fall
    back to a heuristic and still return a positive integer.
    """
    llm = _make_llm(adapter, model="count-fallback-sk-llm")

    class FailingTranslator:
        def count_tokens_for_messages(
            self,
            *args: Any,
            **kwargs: Any,  # noqa: ARG002
        ) -> int:
            raise RuntimeError("forced token count failure")

    llm._translator = FailingTranslator()  # type: ignore[attr-defined]

    tokens = llm.get_num_tokens("this is some SK text for counting")
    assert isinstance(tokens, int)
    assert tokens > 0


# ---------------------------------------------------------------------------
# Capabilities / health passthrough (basic)
# ---------------------------------------------------------------------------


def test_capabilities_and_health_basic() -> None:
    """
    Capabilities and health should be surfaced as mappings when provided by
    the underlying adapter. Detailed structure is tested elsewhere.
    """

    class CapHealthAdapter:
        def complete(self, prompt: str, **_: Any) -> str:
            return f"echo: {prompt}"

        def stream(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return iter(())

        def count_tokens(self, *args: Any, **kwargs: Any) -> int:  # noqa: ARG002
            return 0

        def capabilities(self) -> Dict[str, Any]:
            return {"ok": True}

        async def acapabilities(self) -> Dict[str, Any]:
            return {"ok_async": True}

        def health(self) -> Dict[str, Any]:
            return {"status": "healthy"}

        async def ahealth(self) -> Dict[str, Any]:
            return {"status_async": "healthy"}

    llm = CorpusSemanticKernelLLM(llm_adapter=CapHealthAdapter())

    caps = llm.capabilities()
    assert isinstance(caps, Mapping)
    assert caps.get("ok") is True

    health = llm.health()
    assert isinstance(health, Mapping)
    assert health.get("status") == "healthy"


@pytest.mark.asyncio
async def test_async_capabilities_and_health_fallback_to_sync() -> None:
    """
    acapabilities/ahealth should fall back to sync capabilities()/health()
    when only sync methods are implemented on the underlying adapter.
    """

    class SyncOnlyAdapter:
        def complete(self, prompt: str, **_: Any) -> str:
            return prompt

        def stream(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return iter(())

        def count_tokens(self, *args: Any, **kwargs: Any) -> int:  # noqa: ARG002
            return 0

        def capabilities(self) -> Dict[str, Any]:
            return {"via_sync_caps": True}

        def health(self) -> Dict[str, Any]:
            return {"via_sync_health": True}

    llm = CorpusSemanticKernelLLM(llm_adapter=SyncOnlyAdapter())

    acaps = await llm.acapabilities()
    assert isinstance(acaps, Mapping)
    assert acaps.get("via_sync_caps") is True

    ahealth = await llm.ahealth()
    assert isinstance(ahealth, Mapping)
    assert ahealth.get("via_sync_health") is True


def test_capabilities_and_health_return_empty_when_missing() -> None:
    """
    If the underlying adapter has no capabilities()/health(), the SK LLM
    adapter should return an empty dict rather than raising.
    """

    class NoCapHealthAdapter:
        def complete(self, prompt: str, **_: Any) -> str:
            return prompt

        def stream(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return iter(())

        def count_tokens(self, *args: Any, **kwargs: Any) -> int:  # noqa: ARG002
            return 0

    llm = CorpusSemanticKernelLLM(llm_adapter=NoCapHealthAdapter())

    caps = llm.capabilities()
    assert isinstance(caps, Mapping)
    assert caps == {}

    health = llm.health()
    assert isinstance(health, Mapping)
    assert health == {}


# ---------------------------------------------------------------------------
# Resource management (context managers)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_closes_underlying_adapter() -> None:
    """
    __enter__/__exit__ and __aenter__/__aexit__ should call close/aclose on
    the underlying LLM adapter when those methods exist.
    """

    class ClosingAdapter:
        def __init__(self) -> None:
            self.closed = False
            self.aclosed = False

        def complete(self, prompt: str, **_: Any) -> str:
            return prompt

        def stream(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return iter(())

        def count_tokens(self, *args: Any, **kwargs: Any) -> int:  # noqa: ARG002
            return 0

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.aclosed = True

    adapter = ClosingAdapter()

    # Sync context manager
    with CorpusSemanticKernelLLM(llm_adapter=adapter) as llm:
        _ = llm.complete("smoke-test")

    assert adapter.closed is True

    # Async context manager
    adapter2 = ClosingAdapter()
    llm2 = CorpusSemanticKernelLLM(llm_adapter=adapter2)

    async with llm2:
        _ = await llm2.acomplete("smoke-test-async")

    assert adapter2.aclosed is True


# ---------------------------------------------------------------------------
# Semantic Kernel registration helper (minimal wiring)
# ---------------------------------------------------------------------------


def test_register_with_semantic_kernel_llm_returns_llm(adapter: Any) -> None:
    """
    register_with_semantic_kernel_llm should return a CorpusSemanticKernelLLM
    instance. We intentionally don't assume any particular SK controller
    pattern here.
    """
    llm = register_with_semantic_kernel_llm(
        llm_adapter=adapter,
        model="sk-register-llm",
    )
    assert isinstance(llm, CorpusSemanticKernelLLM)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
