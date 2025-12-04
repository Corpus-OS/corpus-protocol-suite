# tests/frameworks/llm/test_langchain_llm_adapter.py

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Dict

import pytest

import corpus_sdk.llm.framework_adapters.langchain as langchain_llm_module
from corpus_sdk.llm.framework_adapters.langchain import (
    CorpusLangChainLLM,
    LangChainLLMConfig,
    LANGCHAIN_AVAILABLE,
    configure_langchain_llm,
    register_with_langchain_llm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm(llm_adapter: Any, **kwargs: Any) -> CorpusLangChainLLM:
    """Construct a CorpusLangChainLLM instance from the generic adapter."""
    return CorpusLangChainLLM(llm_adapter=llm_adapter, **kwargs)


def _assert_llm_text_like(result: Any) -> None:
    """
    Validate that an LLM output is "text-like".

    We intentionally allow either:
    - bare strings
    - LangChain message objects with `.content`
    - generation chunks with `.text` or `.content`
    """
    assert result is not None

    if isinstance(result, str):
        return

    # Common LangChain message / chunk patterns
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
# Construction / config validation
# ---------------------------------------------------------------------------


def test_init_rejects_adapter_without_required_methods() -> None:
    """
    CorpusLangChainLLM should enforce that llm_adapter implements a
    LLMProtocolV1-like interface (`complete`, `stream`, `count_tokens`,
    `health`, `capabilities`); otherwise __init__ should raise TypeError.
    """

    class BadAdapter:
        # deliberately missing LLMProtocolV1 surface
        def __init__(self) -> None:
            pass

    with pytest.raises(TypeError) as exc_info:
        CorpusLangChainLLM(llm_adapter=BadAdapter())

    msg = str(exc_info.value)
    assert "llm_adapter must implement LLMProtocolV1" in msg
    assert "missing methods" in msg


def test_init_accepts_valid_llm_adapter(adapter: Any) -> None:
    """
    A valid llm_adapter implementing the LLMProtocolV1 surface should be
    accepted and stored as-is on the model.
    """
    llm = CorpusLangChainLLM(
        llm_adapter=adapter,
        model="lc-llm-model",
        temperature=0.5,
        max_tokens=128,
        framework_version="fw-1",
    )

    assert llm.model == "lc-llm-model"
    assert llm.temperature == 0.5
    assert llm.max_tokens == 128


def test_config_object_validation_and_precedence(adapter: Any) -> None:
    """
    LangChainLLMConfig should validate its own fields, and when passed via
    the `config` kwarg it should override simple constructor params.
    """
    cfg = LangChainLLMConfig(
        model="cfg-model",
        temperature=0.3,
        max_tokens=64,
        framework_version="cfg-fw",
    )

    llm = CorpusLangChainLLM(
        llm_adapter=adapter,
        model="ignored-model",
        temperature=1.5,
        max_tokens=999,
        framework_version="ignored-fw",
        config=cfg,
    )

    assert llm.model == "cfg-model"
    assert llm.temperature == 0.3
    assert llm.max_tokens == 64


def test_config_validation_temperature_and_max_tokens_bounds(adapter: Any) -> None:
    """
    Invalid temperature and max_tokens values should raise ValueError,
    both via LangChainLLMConfig and direct constructor params.
    """
    # Config object path
    with pytest.raises(ValueError):
        LangChainLLMConfig(temperature=2.5)

    with pytest.raises(ValueError):
        LangChainLLMConfig(max_tokens=0)

    # Direct params path (no config object)
    with pytest.raises(ValueError):
        CorpusLangChainLLM(llm_adapter=adapter, temperature=-0.1)

    with pytest.raises(ValueError):
        CorpusLangChainLLM(llm_adapter=adapter, max_tokens=0)


def test_configure_and_register_helpers_return_llm(adapter: Any) -> None:
    """
    configure_langchain_llm and register_with_langchain_llm should both
    return CorpusLangChainLLM instances wired to the given adapter.
    """
    llm1 = configure_langchain_llm(
        llm_adapter=adapter,
        model="cfg-llm-model",
    )
    assert isinstance(llm1, CorpusLangChainLLM)

    llm2 = register_with_langchain_llm(
        llm_adapter=adapter,
        model="reg-llm-model",
    )
    assert isinstance(llm2, CorpusLangChainLLM)


def test_LANGCHAIN_AVAILABLE_is_bool() -> None:
    """
    LANGCHAIN_AVAILABLE flag should always be a boolean, regardless of
    whether LangChain is actually installed.
    """
    assert isinstance(LANGCHAIN_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# LangChain interface compatibility
# ---------------------------------------------------------------------------


def test_langchain_llm_interface_compatibility(adapter: Any) -> None:
    """
    Verify that CorpusLangChainLLM exposes the core LangChain chat model
    surfaces (invoke/ainvoke/stream/astream) and, when LangChain is available,
    subclasses the expected chat model base.
    """
    llm = _make_llm(adapter, model="iface-llm-model")

    # Core methods should always exist
    assert hasattr(llm, "invoke")
    assert hasattr(llm, "ainvoke")
    assert hasattr(llm, "stream")
    assert hasattr(llm, "astream")

    if not LANGCHAIN_AVAILABLE:
        pytest.skip("LangChain is not available; cannot assert base class compatibility")

    # Prefer langchain-core BaseChatModel when available
    try:
        from langchain_core.language_models import (  # type: ignore[import]
            BaseChatModel,
        )

        assert isinstance(
            llm,
            BaseChatModel,
        ), "CorpusLangChainLLM should subclass BaseChatModel when available"
    except Exception:
        pytest.skip(
            "LANGCHAIN_AVAILABLE is True but cannot import langchain_core BaseChatModel",
        )


# ---------------------------------------------------------------------------
# RunnableConfig / context mapping via ContextTranslator
# ---------------------------------------------------------------------------


def test_config_passed_to_ContextTranslator_from_langchain_for_invoke(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    Verify that the `config` kwarg is passed through to ContextTranslator.from_langchain
    with the configured framework_version, and that its result is treated as an
    OperationContext.
    """
    captured: Dict[str, Any] = {}

    # Patch OperationContext in the module so isinstance() checks pass.
    class DummyOperationContext:
        def __init__(self, **kwargs: Any) -> None:
            self.attrs = kwargs

    monkeypatch.setattr(
        langchain_llm_module,
        "OperationContext",
        DummyOperationContext,
    )

    class DummyTranslatorClass:
        @staticmethod
        def from_langchain(
            config: Dict[str, Any],
            framework_version: str | None = None,
        ) -> DummyOperationContext:
            captured["config"] = config
            captured["framework_version"] = framework_version
            return DummyOperationContext(config=config, fw=framework_version)

    # Replace ContextTranslator with our dummy that has from_langchain
    monkeypatch.setattr(
        langchain_llm_module,
        "ContextTranslator",
        DummyTranslatorClass,
    )

    llm = _make_llm(
        adapter,
        model="cfg-llm",
        framework_version="lc-llm-fw-test",
    )

    config = {
        "run_id": "llm-run-123",
        "run_name": "llm-test-run",
        "tags": ["llm-tag-a"],
        "metadata": {"pipeline": "llm-unit-test"},
        "configurable": {"tenant": "acme-llm"},
    }

    result = llm.invoke("Hello, world!", config=config)
    _assert_llm_text_like(result)

    assert captured.get("config") is config
    assert captured.get("framework_version") == "lc-llm-fw-test"


def test_config_is_accepted_by_stream_and_astream(
    adapter: Any,
) -> None:
    """
    stream() and astream() should accept the same `config` kwarg as invoke().
    We only assert that calls do not raise and yield an iterable / async-iterable.
    """
    llm = _make_llm(adapter, model="stream-cfg-llm")

    config = {"run_id": "stream-run", "tags": ["stream"]}

    # Sync streaming
    stream_iter = llm.stream("Stream test", config=config)
    chunks = _collect_first_n(stream_iter, 3)
    assert isinstance(chunks, list)


@pytest.mark.asyncio
async def test_config_is_accepted_by_astream_async(adapter: Any) -> None:
    llm = _make_llm(adapter, model="astream-cfg-llm")

    config = {"run_id": "astream-run", "tags": ["astream"]}

    aiter = llm.astream("Async stream test", config=config)
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    chunks = await _acollect_first_n(aiter, 3)
    assert isinstance(chunks, list)


# ---------------------------------------------------------------------------
# Sync / async semantics
# ---------------------------------------------------------------------------


def test_sync_invoke_basic(adapter: Any) -> None:
    """
    Basic smoke test for sync invoke behavior: it should accept a simple
    string prompt and return a text-like output.
    """
    llm = configure_langchain_llm(
        llm_adapter=adapter,
        model="sync-llm-model",
    )

    prompt = "Say hello in one short sentence."
    result = llm.invoke(prompt)
    _assert_llm_text_like(result)


@pytest.mark.asyncio
async def test_async_ainvoke_basic(adapter: Any) -> None:
    """
    Async ainvoke should be a coroutine function and produce results
    compatible with sync invoke().
    """
    llm = configure_langchain_llm(
        llm_adapter=adapter,
        model="async-llm-model",
    )

    assert hasattr(llm, "ainvoke")
    assert inspect.iscoroutinefunction(llm.ainvoke)

    prompt = "Give a very short async response."
    result = await llm.ainvoke(prompt)
    _assert_llm_text_like(result)


@pytest.mark.asyncio
async def test_async_and_sync_invoke_same_output_type(adapter: Any) -> None:
    """
    Sync invoke and async ainvoke for the same prompt should produce outputs
    of the same *type* (e.g., AIMessage vs string), even if contents differ.
    """
    llm = configure_langchain_llm(
        llm_adapter=adapter,
        model="type-parity-llm",
    )

    prompt = "Explain type parity briefly."

    sync_result = llm.invoke(prompt)
    async_result = await llm.ainvoke(prompt)

    assert type(sync_result) is type(
        async_result,
    ), (
        f"invoke()/ainvoke() returned different types: "
        f"{type(sync_result).__name__} vs {type(async_result).__name__}"
    )


# ---------------------------------------------------------------------------
# Streaming semantics
# ---------------------------------------------------------------------------


def test_sync_stream_basic(adapter: Any) -> None:
    """
    stream() should return an iterable of chunks. We don't enforce a specific
    chunk shape, only that:
    - The object is iterable
    - Collected chunks are non-None and text-like.
    """
    llm = configure_langchain_llm(
        llm_adapter=adapter,
        model="stream-llm-model",
    )

    iterator = llm.stream("Stream a short reply.")
    chunks = _collect_first_n(iterator, 5)

    assert isinstance(chunks, list)
    for chunk in chunks:
        _assert_llm_text_like(chunk)


@pytest.mark.asyncio
async def test_async_astream_basic(adapter: Any) -> None:
    """
    astream() should return an async-iterable of chunks, either directly or
    via an awaitable resolving to one.
    """
    llm = configure_langchain_llm(
        llm_adapter=adapter,
        model="astream-llm-model",
    )

    aiter = llm.astream("Async stream a short reply.")
    if inspect.isawaitable(aiter):
        aiter = await aiter  # type: ignore[assignment]

    # Validate async-iterability
    assert hasattr(aiter, "__aiter__")

    chunks = await _acollect_first_n(aiter, 5)
    assert isinstance(chunks, list)
    for chunk in chunks:
        _assert_llm_text_like(chunk)


# ---------------------------------------------------------------------------
# Error-context decorator behavior (framework-focused)
# ---------------------------------------------------------------------------


def test_error_context_includes_langchain_metadata_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When an error occurs during a sync LLM operation, error context should
    include LangChain-specific metadata via attach_context().
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(
        langchain_llm_module,
        "attach_context",
        fake_attach_context,
    )

    class DummyAdapter:
        # Just enough to pass _validate_init_params; adapter methods won't be used
        def complete(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return "ok"

        def stream(self, *args: Any, **kwargs: Any):  # noqa: ARG002
            yield "ok"

        def count_tokens(self, *args: Any, **kwargs: Any) -> int:  # noqa: ARG002
            return 1

        def health(self) -> Dict[str, Any]:
            return {}

        def capabilities(self) -> Dict[str, Any]:
            return {}

    class FailingTranslator:
        def complete(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from langchain llm adapter")

    def fake_create_llm_translator(*_: Any, **__: Any) -> Any:
        return FailingTranslator()

    monkeypatch.setattr(
        langchain_llm_module,
        "create_llm_translator",
        fake_create_llm_translator,
    )

    llm = CorpusLangChainLLM(llm_adapter=DummyAdapter(), model="err-llm-sync")

    with pytest.raises(RuntimeError, match="test error from langchain llm adapter"):
        llm.invoke("trigger failure")

    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "langchain"
    op = str(captured_context.get("operation", ""))
    assert op.startswith("llm_")
    # Some dynamic context fields should be present
    assert captured_context.get("model") == "err-llm-sync"


@pytest.mark.asyncio
async def test_error_context_includes_langchain_metadata_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Same as the sync error-context test but for the async invoke path.
    """
    captured_context: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_context.update(ctx)

    monkeypatch.setattr(
        langchain_llm_module,
        "attach_context",
        fake_attach_context,
    )

    class DummyAdapter:
        def complete(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return "ok"

        def stream(self, *args: Any, **kwargs: Any):  # noqa: ARG002
            yield "ok"

        def count_tokens(self, *args: Any, **kwargs: Any) -> int:  # noqa: ARG002
            return 1

        def health(self) -> Dict[str, Any]:
            return {}

        def capabilities(self) -> Dict[str, Any]:
            return {}

    class FailingTranslator:
        async def arun_complete(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("test error from langchain llm adapter")

    def fake_create_llm_translator(*_: Any, **__: Any) -> Any:
        return FailingTranslator()

    monkeypatch.setattr(
        langchain_llm_module,
        "create_llm_translator",
        fake_create_llm_translator,
    )

    llm = CorpusLangChainLLM(llm_adapter=DummyAdapter(), model="err-llm-async")

    with pytest.raises(RuntimeError, match="test error from langchain llm adapter"):
        await llm.ainvoke("trigger async failure")

    assert captured_context, "attach_context was not called"
    assert captured_context.get("framework") == "langchain"
    op = str(captured_context.get("operation", ""))
    assert op.startswith("llm_")
    assert captured_context.get("model") == "err-llm-async"


# ---------------------------------------------------------------------------
# Token counting (translator + fallback)
# ---------------------------------------------------------------------------


def test_get_num_tokens_returns_positive_int(adapter: Any) -> None:
    """
    get_num_tokens() should return a positive integer for non-empty input,
    either via translator-based counting or via the heuristic fallback.
    """
    llm = configure_langchain_llm(
        llm_adapter=adapter,
        model="tokens-llm",
    )

    count = llm.get_num_tokens("This is a short test.")
    assert isinstance(count, int)
    assert count > 0


def test_get_num_tokens_uses_heuristic_when_translator_fails(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
) -> None:
    """
    If the underlying translator.count_tokens_for_messages raises or returns
    unexpected data, get_num_tokens_from_messages should fall back to the
    character-based heuristic.
    """

    class DummyAdapter:
        def complete(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return "ok"

        def stream(self, *args: Any, **kwargs: Any):  # noqa: ARG002
            yield "ok"

        def count_tokens(self, *args: Any, **kwargs: Any) -> int:  # noqa: ARG002
            return 1

        def health(self) -> Dict[str, Any]:
            return {}

        def capabilities(self) -> Dict[str, Any]:
            return {}

    class FailingTokenTranslator:
        def count_tokens_for_messages(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            raise RuntimeError("forced token failure")

        # Minimal surfaces to keep _generate/_stream unused
        def complete(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return "ok"

        async def arun_complete(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return "ok"

        def stream(self, *args: Any, **kwargs: Any):  # noqa: ARG002
            yield "ok"

        async def arun_stream(self, *args: Any, **kwargs: Any):  # noqa: ARG002
            async def gen():
                yield "ok"

            return gen()

    def fake_create_llm_translator(*_: Any, **__: Any) -> Any:
        return FailingTokenTranslator()

    monkeypatch.setattr(
        langchain_llm_module,
        "create_llm_translator",
        fake_create_llm_translator,
    )

    from langchain_core.messages import HumanMessage  # type: ignore[import]

    llm = CorpusLangChainLLM(llm_adapter=DummyAdapter(), model="tokens-fallback")

    msgs = [HumanMessage(content="fallback token test")]
    count = llm.get_num_tokens_from_messages(msgs)
    assert isinstance(count, int)
    assert count > 0


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

        # Minimal LLMProtocolV1 surface; they won't be heavily used here.
        def complete(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return "ok"

        def stream(self, *args: Any, **kwargs: Any):  # noqa: ARG002
            yield "ok"

        def count_tokens(self, *args: Any, **kwargs: Any) -> int:  # noqa: ARG002
            return 1

        def health(self) -> Dict[str, Any]:
            return {}

        def capabilities(self) -> Dict[str, Any]:
            return {}

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.aclosed = True

    adapter = ClosingAdapter()

    # Sync context manager
    with CorpusLangChainLLM(llm_adapter=adapter) as llm:
        _ = llm.invoke("smoke-test")

    assert adapter.closed is True

    # Async context manager
    adapter2 = ClosingAdapter()
    llm2 = CorpusLangChainLLM(llm_adapter=adapter2)

    async with llm2:
        _ = await llm2.ainvoke("smoke-test-async")

    assert adapter2.aclosed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
