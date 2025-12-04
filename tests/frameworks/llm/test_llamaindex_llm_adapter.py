# tests/frameworks/llm/test_llamaindex_llm_adapter.py

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Dict, List

import pytest

import corpus_sdk.llm.framework_adapters.llamaindex as llamaindex_llm_module
from corpus_sdk.llm.framework_adapters.llamaindex import (
    CorpusLlamaIndexLLM,
    LLAMAINDEX_AVAILABLE,
    configure_llamaindex_llm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm(adapter: Any, **kwargs: Any) -> CorpusLlamaIndexLLM:
    """
    Construct a CorpusLlamaIndexLLM instance from the generic LLM adapter.

    The fixture is named `adapter` for consistency across frameworks; we pass
    it through to the llamaindex adapter via the `llm_adapter` kwarg.
    """
    return CorpusLlamaIndexLLM(llm_adapter=adapter, **kwargs)


def _assert_llm_text_like(result: Any) -> None:
    """
    Validate that an LLM output is "text-like".

    We intentionally allow either:
    - bare strings, or
    - objects with a `.content` or `.text` attribute (LlamaIndex-style outputs).
    """
    assert result is not None

    if isinstance(result, str):
        return

    content = getattr(result, "content", None)
    text = getattr(result, "text", None)
    assert isinstance(content, str) or isinstance(
        text, str
    ), f"LLM result is not text-like: {type(result).__name__}"


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
# Construction / config behavior
# ---------------------------------------------------------------------------


def test_constructor_rejects_adapter_without_llm_methods() -> None:
    """
    CorpusLlamaIndexLLM should validate that llm_adapter implements an
    LLMProtocolV1-style interface; otherwise __init__ should raise TypeError.
    """

    class BadAdapter:
        # deliberately missing LLMProtocolV1 methods like complete/stream/etc.
        def __init__(self) -> None:
            pass

    with pytest.raises(TypeError) as exc_info:
        CorpusLlamaIndexLLM(llm_adapter=BadAdapter())

    msg = str(exc_info.value)
    # We don't hard-code framework-specific error codes; just assert that the
    # message mentions LLMProtocolV1 or "missing methods".
    assert "LLMProtocolV1" in msg or "missing methods" in msg


def test_constructor_accepts_valid_llm_adapter(adapter: Any) -> None:
    """
    A valid LLMProtocolV1-compatible adapter should be accepted and stored.
    """
    llm = CorpusLlamaIndexLLM(
        llm_adapter=adapter,
        model="llama-llm-model",
    )

    assert llm is not None
    assert llm.model == "llama-llm-model"


def test_config_rejects_bad_temperature_and_max_tokens(adapter: Any) -> None:
    """
    Temperature and max_tokens should be validated and raise ValueError when
    obviously invalid.
    """
    # Temperature below 0.0
    with pytest.raises(ValueError):
        configure_llamaindex_llm(
            llm_adapter=adapter,
            model="bad-temp",
            temperature=-0.1,
        )

    # max_tokens <= 0
    with pytest.raises(ValueError):
        configure_llamaindex_llm(
            llm_adapter=adapter,
            model="bad-max-tokens",
            max_tokens=0,
        )


def test_configure_llamaindex_llm_returns_llm(adapter: Any) -> None:
    """
    configure_llamaindex_llm should return a CorpusLlamaIndexLLM instance wired
    to the given adapter and model.
    """
    llm = configure_llamaindex_llm(
        llm_adapter=adapter,
        model="cfg-llm-model",
    )

    assert isinstance(llm, CorpusLlamaIndexLLM)
    # We can't reliably introspect the underlying adapter attribute name, but
    # we can at least ensure model is set.
    assert llm.model == "cfg-llm-model"


def test_LLAMAINDEX_AVAILABLE_is_bool() -> None:
    """
    LLAMAINDEX_AVAILABLE flag should always be a boolean, regardless of
    whether LlamaIndex is actually installed.
    """
    assert isinstance(LLAMAINDEX_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# LlamaIndex interface compatibility (shape-based)
# ---------------------------------------------------------------------------


def test_llamaindex_llm_interface_shape(adapter: Any) -> None:
    """
    Verify that CorpusLlamaIndexLLM exposes the expected LlamaIndex-style
    chat surfaces:

    - complete / acomplete
    - stream_complete / astream_complete

    We use structural checks only (no hard dependency on LlamaIndex base
    classes) to keep this test optional-dependency-safe.
    """
    llm = _make_llm(adapter, model="iface-llm")

    # Core methods should always exist
    assert hasattr(llm, "complete")
    assert hasattr(llm, "acomplete")
    assert hasattr(llm, "stream_complete")
    assert hasattr(llm, "astream_complete")

    assert callable(llm.complete)
    assert callable(llm.stream_complete)
    assert inspect.iscoroutinefunction(llm.acomplete)
    assert inspect.iscoroutinefunction(llm.astream_complete)


# ---------------------------------------------------------------------------
# Context translation / LlamaIndex context mapping
# ---------------------------------------------------------------------------


def test_llamaindex_context_passed_to_context_translator_for_complete(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Verify that a `config`/context-like kwarg is passed through to
    ContextTranslator.from_llamaindex when invoking the LLM.

    We patch ContextTranslator + OperationContext inside the module to capture
    arguments but still let the rest of the code run normally.
    """
    captured: Dict[str, Any] = {}

    # Patch OperationContext so isinstance checks still pass.
    class DummyOperationContext:
        def __init__(self, **kwargs: Any) -> None:
            self.attrs = kwargs

    monkeypatch.setattr(
        llamaindex_llm_module,
        "OperationContext",
        DummyOperationContext,
    )

    # Build a fake ContextTranslator with from_llamaindex.
    class FakeContextTranslator:
        @staticmethod
        def from_llamaindex(
            config: Dict[str, Any],
            framework_version: str | None = None,
        ) -> DummyOperationContext:
            captured["config"] = config
            captured["framework_version"] = framework_version
            return DummyOperationContext(config=config)

    monkeypatch.setattr(
        llamaindex_llm_module,
        "ContextTranslator",
        FakeContextTranslator,
    )

    llm = _make_llm(
        adapter,
        model="ctx-llm",
        framework_version="llama-fw-test",
    )

    config = {
        "request_id": "req-123",
        "query_id": "q-456",
        "tenant": "tenant-xyz",
    }

    result = llm.complete("Hello from LlamaIndex", config=config)
    _assert_llm_text_like(result)

    assert captured.get("config") is config
    assert captured.get("framework_version") == "llama-fw-test"


def test_context_translation_failure_attaches_error_context(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If ContextTranslator.from_llamaindex fails, the adapter should:

    - Attach error context via attach_context(framework="llamaindex",
      operation="llm_context_translation", ...)
    - Re-raise the underlying error (wrapped or not).
    """
    captured_ctx: Dict[str, Any] = {}

    # Patch attach_context to capture metadata.
    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(
        llamaindex_llm_module,
        "attach_context",
        fake_attach_context,
    )

    # Patch ContextTranslator.from_llamaindex to raise.
    class FakeContextTranslator:
        @staticmethod
        def from_llamaindex(
            config: Dict[str, Any],
            framework_version: str | None = None,
        ) -> Any:  # noqa: ARG002
            raise RuntimeError("ctx boom from llamaindex")

    monkeypatch.setattr(
        llamaindex_llm_module,
        "ContextTranslator",
        FakeContextTranslator,
    )

    llm = _make_llm(
        adapter,
        model="ctx-fail-llm",
        framework_version="llama-fw-ctx-fail",
    )

    with pytest.raises(RuntimeError, match="ctx boom from llamaindex"):
        llm.complete("Trigger context failure", config={"foo": "bar"})

    assert captured_ctx.get("framework") == "llamaindex"
    assert captured_ctx.get("operation") == "llm_context_translation"
    assert captured_ctx.get("framework_version") == "llama-fw-ctx-fail"


# ---------------------------------------------------------------------------
# Sync / async semantics
# ---------------------------------------------------------------------------


def test_sync_complete_basic(adapter: Any) -> None:
    """
    Basic smoke test for sync complete() behavior: it should accept simple
    text input and return a text-like object.
    """
    llm = configure_llamaindex_llm(
        llm_adapter=adapter,
        model="sync-llm-model",
    )

    prompt = "Say hello in one short sentence."
    result = llm.complete(prompt)
    _assert_llm_text_like(result)


@pytest.mark.asyncio
async def test_async_acomplete_basic(adapter: Any) -> None:
    """
    Async acomplete() should be a coroutine function and produce results
    compatible with sync complete().
    """
    llm = configure_llamaindex_llm(
        llm_adapter=adapter,
        model="async-llm-model",
    )

    assert hasattr(llm, "acomplete")
    assert inspect.iscoroutinefunction(llm.acomplete)

    prompt = "Give a very short async response."
    result = await llm.acomplete(prompt)
    _assert_llm_text_like(result)


@pytest.mark.asyncio
async def test_async_and_sync_complete_same_output_type(adapter: Any) -> None:
    """
    Sync complete and async acomplete for the same prompt should produce
    outputs of the same *type* (string vs response object), even if contents
    differ.
    """
    llm = configure_llamaindex_llm(
        llm_adapter=adapter,
        model="type-parity-llm",
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


def test_sync_stream_complete_basic(adapter: Any) -> None:
    """
    stream_complete() should return an iterable of chunks. We don't enforce a
    specific chunk shape, only that:

    - The object is iterable
    - Collected chunks are non-None and text-like.
    """
    llm = configure_llamaindex_llm(
        llm_adapter=adapter,
        model="stream-llm-model",
    )

    iterator = llm.stream_complete("Stream a short reply.")
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
    llm = configure_llamaindex_llm(
        llm_adapter=adapter,
        model="astream-llm-model",
    )

    aiter = llm.astream_complete("Async stream a short reply.")
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    # Validate async-iterability
    assert hasattr(aiter, "__aiter__")

    chunks = await _acollect_first_n(aiter, 5)
    assert isinstance(chunks, list)
    for chunk in chunks:
        _assert_llm_text_like(chunk)


def test_config_is_accepted_by_stream_and_astream(adapter: Any) -> None:
    """
    stream_complete() should accept the same `config` kwarg as complete().
    We only assert that the call does not raise and yields an iterable.
    """
    llm = _make_llm(adapter, model="stream-cfg-llm")

    config = {"request_id": "stream-run", "tags": ["stream"]}

    iterator = llm.stream_complete("Stream with config", config=config)
    chunks = _collect_first_n(iterator, 3)
    assert isinstance(chunks, list)


@pytest.mark.asyncio
async def test_config_is_accepted_by_astream_complete_async(adapter: Any) -> None:
    llm = _make_llm(adapter, model="astream-cfg-llm")

    config = {"request_id": "astream-run", "tags": ["astream"]}

    aiter = llm.astream_complete("Async stream with config", config=config)
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    chunks = await _acollect_first_n(aiter, 3)
    assert isinstance(chunks, list)


# ---------------------------------------------------------------------------
# Error-context decorator behavior (framework-focused)
# ---------------------------------------------------------------------------


def test_error_context_includes_llamaindex_metadata_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When an error occurs during a sync LLM operation, error context should
    include LlamaIndex-specific metadata via attach_context().
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(
        llamaindex_llm_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingAdapter:
        def complete(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from llamaindex llm adapter")

    llm = CorpusLlamaIndexLLM(llm_adapter=FailingAdapter())

    with pytest.raises(RuntimeError, match="test error from llamaindex llm adapter"):
        llm.complete("trigger failure")

    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "llamaindex"
    op = str(captured_context.get("operation", ""))
    assert op.startswith("llm_")


@pytest.mark.asyncio
async def test_error_context_includes_llamaindex_metadata_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Same as the sync error-context test but for the async acomplete path.
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(
        llamaindex_llm_module,
        "attach_context",
        fake_attach_context,
    )

    class FailingAdapter:
        async def acomplete(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from llamaindex llm adapter")

    llm = CorpusLlamaIndexLLM(llm_adapter=FailingAdapter())

    with pytest.raises(RuntimeError, match="test error from llamaindex llm adapter"):
        await llm.acomplete("trigger async failure")

    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "llamaindex"
    op = str(captured_context.get("operation", ""))
    assert op.startswith("llm_")


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

        def capabilities(self) -> Dict[str, Any]:
            return {"ok": True}

        async def acapabilities(self) -> Dict[str, Any]:
            return {"ok_async": True}

        def health(self) -> Dict[str, Any]:
            return {"status": "healthy"}

        async def ahealth(self) -> Dict[str, Any]:
            return {"status_async": "healthy"}

    llm = CorpusLlamaIndexLLM(llm_adapter=CapHealthAdapter())

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

        def capabilities(self) -> Dict[str, Any]:
            return {"via_sync_caps": True}

        def health(self) -> Dict[str, Any]:
            return {"via_sync_health": True}

    llm = CorpusLlamaIndexLLM(llm_adapter=SyncOnlyAdapter())

    acaps = await llm.acapabilities()
    assert isinstance(acaps, Mapping)
    assert acaps.get("via_sync_caps") is True

    ahealth = await llm.ahealth()
    assert isinstance(ahealth, Mapping)
    assert ahealth.get("via_sync_health") is True


def test_capabilities_and_health_return_empty_when_missing() -> None:
    """
    If the underlying adapter has no capabilities()/health(), the LlamaIndex
    LLM adapter should return an empty dict rather than raising.
    """

    class NoCapHealthAdapter:
        def complete(self, prompt: str, **_: Any) -> str:
            return prompt

    llm = CorpusLlamaIndexLLM(llm_adapter=NoCapHealthAdapter())

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

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.aclosed = True

    adapter = ClosingAdapter()

    # Sync context manager
    with CorpusLlamaIndexLLM(llm_adapter=adapter) as llm:
        _ = llm.complete("smoke-test")

    assert adapter.closed is True

    # Async context manager
    adapter2 = ClosingAdapter()
    llm2 = CorpusLlamaIndexLLM(llm_adapter=adapter2)

    async with llm2:
        _ = await llm2.acomplete("smoke-test-async")

    assert adapter2.aclosed is True


# ---------------------------------------------------------------------------
# Token counting (if implemented)
# ---------------------------------------------------------------------------


def test_get_num_tokens_uses_translator_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    get_num_tokens should delegate to the translator's
    count_tokens_for_messages when it returns an integer or mapping.
    """

    class DummyAdapter:
        def complete(self, prompt: str, **_: Any) -> str:
            return prompt

        def stream(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return iter(())

        def count_tokens(self, *args: Any, **kwargs: Any) -> int:  # noqa: ARG002
            return 42

        def capabilities(self) -> Dict[str, Any]:
            return {}

        def health(self) -> Dict[str, Any]:
            return {}

    llm = CorpusLlamaIndexLLM(llm_adapter=DummyAdapter())

    # Patch translator on the instance
    class DummyTranslator:
        def count_tokens_for_messages(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return {"tokens": 123}

    monkeypatch.setattr(llm, "_translator", DummyTranslator())

    tokens = llm.get_num_tokens("hello world")
    assert isinstance(tokens, int)
    assert tokens == 123


def test_get_num_tokens_falls_back_to_heuristic_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When translator.count_tokens_for_messages fails, get_num_tokens should
    fall back to a heuristic character-based estimate.
    """

    class DummyAdapter:
        def complete(self, prompt: str, **_: Any) -> str:
            return prompt

        def stream(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return iter(())

        def count_tokens(self, *args: Any, **kwargs: Any) -> int:  # noqa: ARG002
            return 1

        def capabilities(self) -> Dict[str, Any]:
            return {}

        def health(self) -> Dict[str, Any]:
            return {}

    llm = CorpusLlamaIndexLLM(llm_adapter=DummyAdapter())

    class FailingTranslator:
        def count_tokens_for_messages(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("no tokens for you")

    monkeypatch.setattr(llm, "_translator", FailingTranslator())

    text = "this is a reasonably long text for heuristic counting"
    tokens = llm.get_num_tokens(text)
    assert isinstance(tokens, int)
    assert tokens > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
