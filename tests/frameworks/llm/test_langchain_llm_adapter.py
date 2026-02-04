# tests/frameworks/llm/test_langchain_llm_adapter.py
# SPDX-License-Identifier: Apache-2.0
"""
Full LangChain framework adapter test suite for CorpusLangChainLLM.

Goals
-----
This suite is intentionally *framework-focused* and covers:

- LangChain semantics (message normalization, callbacks, stop, tools, streaming)
- Sync/async behavior and event-loop safety guards
- "Translator-only" routing for health/capabilities/token counting
- Direct integration tests that exercise: LangChain adapter → LLMTranslator → LLMProtocolV1
- End-to-end tests using LangChain public entrypoints (invoke/ainvoke/stream/astream)

Notes
-----
- Many tests require `langchain-core`. If it is not installed, the suite will
  skip tests that cannot run.
- The adapter module itself is designed to be importable without LangChain;
  we include guarded import tests using importlib.reload + monkeypatched imports.

Structure
---------
- Shared fixtures: FakeLLMProtocolV1 + lightweight callback managers
- Unit tests: adapter behavior in isolation (with translator mocked/spied as needed)
- Integration tests: real LLMTranslator + fake protocol adapter
- E2E tests: call BaseChatModel public methods (invoke/ainvoke/stream/astream)

This file intentionally contains 60+ distinct tests (target: 57–65).
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import types
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import pytest


# -----------------------------------------------------------------------------
# Optional LangChain imports (tests that require it will be skipped if missing)
# -----------------------------------------------------------------------------
_LANGCHAIN_AVAILABLE = True
try:
    from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
except Exception:  # pragma: no cover
    _LANGCHAIN_AVAILABLE = False
    AIMessage = AIMessageChunk = BaseMessage = HumanMessage = object  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# Imports under test
# -----------------------------------------------------------------------------
# We import these at module scope for the "normal" path. Import-guard tests
# will reload under a simulated missing-langchain environment.
from corpus_sdk.llm.framework_adapters.langchain import (  # type: ignore
    CorpusLangChainLLM,
    LangChainLLMConfig,
    LANGCHAIN_AVAILABLE,
    INIT_CONFIG_ERROR,
    SYNC_WRAPPER_CALLED_IN_EVENT_LOOP,
)

from corpus_sdk.llm.framework_adapters.common.llm_translation import (  # type: ignore
    create_llm_translator,
    LLMTranslator,
)

from corpus_sdk.llm.llm_base import (  # type: ignore
    LLMCapabilities,
    LLMChunk,
    LLMCompletion,
    OperationContext,
)


# =============================================================================
# Test Harness: Fake protocol adapter + helpers
# =============================================================================

@dataclass
class _CallRecord:
    name: str
    kwargs: Dict[str, Any]


class FakeLLMProtocolV1:
    """
    Fake LLMProtocolV1 adapter with deterministic, scriptable behavior.

    Matches the signatures expected by LLMTranslator in corpus_sdk:

    - complete(...) -> LLMCompletion  (async)
    - stream(...)   -> AsyncIterator[LLMChunk]  (async generator)
    - count_tokens(text, ...) -> int/float/str (async)
    - health(ctx=...) -> Mapping[str, Any]      (async)
    - capabilities() -> LLMCapabilities         (async)

    Captures inbound parameters for assertions.
    """

    def __init__(self) -> None:
        self.calls: List[_CallRecord] = []

        # Script controls
        self.next_completion: Optional[LLMCompletion] = None
        self.next_stream_chunks: List[LLMChunk] = []
        self.next_count_tokens: Any = 42
        self.next_health: Mapping[str, Any] = {"ok": True}
        self.next_capabilities: Optional[LLMCapabilities] = None

        # Error injection controls
        self.raise_on_complete: Optional[BaseException] = None
        self.raise_on_stream_at_index: Optional[int] = None
        self.raise_on_stream_exc: Optional[BaseException] = None
        self.raise_on_count_tokens: Optional[BaseException] = None
        self.raise_on_health: Optional[BaseException] = None
        self.raise_on_capabilities: Optional[BaseException] = None

        # Cleanup signals
        self.stream_closed: bool = False

    def _record(self, name: str, **kwargs: Any) -> None:
        self.calls.append(_CallRecord(name=name, kwargs=dict(kwargs)))

    # --- Protocol methods expected by CorpusLangChainLLM validation ---
    async def complete(
        self,
        *,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> LLMCompletion:
        self._record(
            "complete",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop_sequences,
            model=model,
            system_message=system_message,
            tools=tools,
            tool_choice=tool_choice,
            ctx=ctx,
        )
        if self.raise_on_complete is not None:
            raise self.raise_on_complete

        if self.next_completion is not None:
            return self.next_completion

        # Sensible default
        return LLMCompletion(text="ok", model=model or "default", model_family=None, usage=None, finish_reason="stop", tool_calls=[])

    async def stream(
        self,
        *,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[LLMChunk]:
        self._record(
            "stream",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop_sequences,
            model=model,
            system_message=system_message,
            tools=tools,
            tool_choice=tool_choice,
            ctx=ctx,
        )

        async def _agen() -> AsyncIterator[LLMChunk]:
            try:
                for idx, ch in enumerate(self.next_stream_chunks):
                    if self.raise_on_stream_at_index is not None and idx == self.raise_on_stream_at_index:
                        raise self.raise_on_stream_exc or RuntimeError("stream failure")
                    yield ch
            finally:
                # Signal that the underlying async generator was closed/finished.
                self.stream_closed = True

        return _agen()

    async def count_tokens(
        self,
        *,
        text: str,
        model: Optional[str] = None,
        ctx: Optional[OperationContext] = None,
    ) -> Any:
        self._record("count_tokens", text=text, model=model, ctx=ctx)
        if self.raise_on_count_tokens is not None:
            raise self.raise_on_count_tokens
        return self.next_count_tokens

    async def health(self, *, ctx: Optional[OperationContext] = None) -> Mapping[str, Any]:
        self._record("health", ctx=ctx)
        if self.raise_on_health is not None:
            raise self.raise_on_health
        return dict(self.next_health)

    async def capabilities(self) -> LLMCapabilities:
        self._record("capabilities")
        if self.raise_on_capabilities is not None:
            raise self.raise_on_capabilities
        if self.next_capabilities is not None:
            return self.next_capabilities

        # Minimal, permissive default: your real LLMCapabilities may have more fields.
        # We populate commonly-used ones; dataclass defaults handle the rest.
        return LLMCapabilities(
            supports_streaming=True,
            supports_tools=True,
            supports_json_mode=True,
            supports_vision=False,
            supports_audio=False,
            max_context_tokens=None,
            provider=None,
            models=None,
        )


# -----------------------------------------------------------------------------
# Lightweight callback managers (sync + async) for framework semantics tests
# -----------------------------------------------------------------------------

class SyncRunManager:
    """
    Minimal CallbackManagerForLLMRun-like object.

    Captures callback invocations from CorpusLangChainLLM._generate/_stream.
    """
    def __init__(self) -> None:
        self.events: List[Tuple[str, Any]] = []
        self.raise_on_new_token: Optional[BaseException] = None

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        self.events.append(("start", {"args": args, "kwargs": kwargs}))

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.events.append(("token", {"token": token, "kwargs": kwargs}))
        if self.raise_on_new_token is not None:
            raise self.raise_on_new_token

    def on_llm_end(self, result: Any) -> None:
        self.events.append(("end", result))

    def on_llm_error(self, exc: BaseException) -> None:
        self.events.append(("error", exc))


class AsyncRunManager:
    """
    Minimal AsyncCallbackManagerForLLMRun-like object.

    Captures callback invocations from CorpusLangChainLLM._agenerate/_astream.
    """
    def __init__(self) -> None:
        self.events: List[Tuple[str, Any]] = []
        self.raise_on_new_token: Optional[BaseException] = None

    async def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        self.events.append(("start", {"args": args, "kwargs": kwargs}))

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.events.append(("token", {"token": token, "kwargs": kwargs}))
        if self.raise_on_new_token is not None:
            raise self.raise_on_new_token

    async def on_llm_end(self, result: Any) -> None:
        self.events.append(("end", result))

    async def on_llm_error(self, exc: BaseException) -> None:
        self.events.append(("error", exc))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _mk_completion(text: str = "hello", model: str = "m") -> LLMCompletion:
    return LLMCompletion(
        text=text,
        model=model,
        model_family=None,
        usage=None,
        finish_reason="stop",
        tool_calls=[],
    )


def _mk_chunk(text: str, model: str = "m", is_final: bool = False) -> LLMChunk:
    return LLMChunk(
        text=text,
        is_final=is_final,
        model=model,
        usage_so_far=None,
        tool_calls=[],
    )


async def _collect_async_iter(ait: AsyncIterator[Any], limit: Optional[int] = None) -> List[Any]:
    out: List[Any] = []
    i = 0
    async for item in ait:
        out.append(item)
        i += 1
        if limit is not None and i >= limit:
            break
    return out


def _collect_sync_iter(it: Iterator[Any], limit: Optional[int] = None) -> List[Any]:
    out: List[Any] = []
    for idx, item in enumerate(it):
        out.append(item)
        if limit is not None and idx + 1 >= limit:
            break
    return out


def _require_langchain() -> None:
    if not _LANGCHAIN_AVAILABLE:
        pytest.skip("langchain-core is not installed; skipping LangChain-specific tests.")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def fake_adapter() -> FakeLLMProtocolV1:
    return FakeLLMProtocolV1()


@pytest.fixture()
def model(fake_adapter: FakeLLMProtocolV1) -> CorpusLangChainLLM:
    _require_langchain()
    return CorpusLangChainLLM(llm_adapter=fake_adapter, model="test-model", temperature=0.3, max_tokens=123)


@pytest.fixture()
def basic_messages() -> List[BaseMessage]:
    _require_langchain()
    return [HumanMessage(content="hi")]


# =============================================================================
# 1) Import + optional dependency behavior (2 tests)
# =============================================================================

def test_langchain_module_import_guard_simulated_missing_langchain(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Simulate missing langchain_core and ensure the adapter module can still import.

    We do this by reloading the module under a patched import environment that
    raises ImportError for langchain_core.*.
    """
    module_name = "corpus_sdk.llm.framework_adapters.langchain"
    original = sys.modules.get(module_name)

    real_import = __import__

    def guarded_import(name: str, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("langchain_core"):
            raise ImportError("simulated missing langchain_core")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", guarded_import)

    # Remove module so reload re-executes import logic
    if module_name in sys.modules:
        del sys.modules[module_name]

    m = importlib.import_module(module_name)
    assert hasattr(m, "LANGCHAIN_AVAILABLE")
    assert m.LANGCHAIN_AVAILABLE is False

    # Restore prior loaded module to avoid affecting subsequent tests
    if original is not None:
        sys.modules[module_name] = original


def test_init_raises_without_langchain_installed_simulated(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Under simulated missing langchain_core, instantiation should raise ImportError.
    """
    module_name = "corpus_sdk.llm.framework_adapters.langchain"
    original = sys.modules.get(module_name)

    real_import = __import__

    def guarded_import(name: str, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("langchain_core"):
            raise ImportError("simulated missing langchain_core")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", guarded_import)

    if module_name in sys.modules:
        del sys.modules[module_name]
    m = importlib.import_module(module_name)

    assert m.LANGCHAIN_AVAILABLE is False
    with pytest.raises(ImportError):
        m.CorpusLangChainLLM(llm_adapter=FakeLLMProtocolV1())

    if original is not None:
        sys.modules[module_name] = original


# =============================================================================
# 2) Initialization validation + invariants (3 tests)
# =============================================================================

def test_init_validates_temperature_range() -> None:
    _require_langchain()
    with pytest.raises(ValueError) as e:
        LangChainLLMConfig(temperature=3.0)
    assert INIT_CONFIG_ERROR in str(e.value)


def test_init_validates_max_tokens_positive() -> None:
    _require_langchain()
    with pytest.raises(ValueError) as e:
        LangChainLLMConfig(max_tokens=0)
    assert INIT_CONFIG_ERROR in str(e.value)


def test_init_requires_llmprotocol_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_langchain()

    class MissingMethods:
        # missing: stream, count_tokens, health, capabilities
        async def complete(self, **kwargs: Any) -> Any:
            return _mk_completion("x")

    with pytest.raises(TypeError) as e:
        CorpusLangChainLLM(llm_adapter=MissingMethods())  # type: ignore[arg-type]
    assert INIT_CONFIG_ERROR in str(e.value)


# =============================================================================
# 3) Message normalization & validation (8 tests)
# =============================================================================

def test_normalize_accepts_base_messages(model: CorpusLangChainLLM, basic_messages: List[BaseMessage]) -> None:
    out = model._normalize_messages(basic_messages)  # type: ignore[attr-defined]
    assert len(out) == 1
    assert isinstance(out[0], BaseMessage)


def test_normalize_accepts_mapping_role_user(model: CorpusLangChainLLM) -> None:
    msgs = model._normalize_messages([{"role": "user", "content": "hi"}])  # type: ignore[attr-defined]
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "hi"


def test_normalize_accepts_mapping_role_assistant(model: CorpusLangChainLLM) -> None:
    msgs = model._normalize_messages([{"role": "assistant", "content": "yo"}])  # type: ignore[attr-defined]
    assert isinstance(msgs[0], AIMessage)
    assert msgs[0].content == "yo"


def test_normalize_accepts_mapping_type_field(model: CorpusLangChainLLM) -> None:
    msgs = model._normalize_messages([{"type": "ai", "content": "ok"}])  # type: ignore[attr-defined]
    assert isinstance(msgs[0], AIMessage)


def test_normalize_unknown_role_defaults_to_human(model: CorpusLangChainLLM) -> None:
    msgs = model._normalize_messages([{"role": "weird", "content": "hi"}])  # type: ignore[attr-defined]
    assert isinstance(msgs[0], HumanMessage)


def test_normalize_rejects_empty_list(model: CorpusLangChainLLM) -> None:
    with pytest.raises(ValueError):
        model._normalize_messages([])  # type: ignore[attr-defined]


def test_validate_rejects_empty_content_when_enabled(model: CorpusLangChainLLM) -> None:
    msgs = [HumanMessage(content="")]
    with pytest.raises(ValueError):
        model._validate_messages(msgs)  # type: ignore[attr-defined]


def test_validate_skips_when_validate_inputs_false(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    cfg = LangChainLLMConfig(validate_inputs=False)
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, config=cfg)
    # Should not raise
    m._validate_messages([HumanMessage(content="")])  # type: ignore[attr-defined]


# =============================================================================
# 4) Context translation from LangChain config (6 tests)
# =============================================================================

def test_context_none_yields_default_ctx(model: CorpusLangChainLLM) -> None:
    ctx, params, fw = model._build_context_and_params(operation="generate", stream=False)  # type: ignore[attr-defined]
    assert isinstance(ctx, OperationContext)
    assert isinstance(params, dict)
    assert fw["framework"] == "langchain"


def test_context_passthrough_operationcontext(model: CorpusLangChainLLM) -> None:
    oc = OperationContext(request_id="rid", attrs={"x": 1})
    ctx, _, _ = model._build_context_and_params(operation="generate", stream=False, config=oc)  # type: ignore[attr-defined]
    assert ctx.request_id == "rid"
    assert ctx.attrs.get("x") == 1


def test_context_invalid_config_type_tolerated(model: CorpusLangChainLLM) -> None:
    ctx, _, _ = model._build_context_and_params(operation="generate", stream=False, config=123)  # type: ignore[attr-defined]
    assert isinstance(ctx, OperationContext)


def test_framework_ctx_contains_expected_keys(model: CorpusLangChainLLM) -> None:
    ctx, params, fw = model._build_context_and_params(operation="generate", stream=False)  # type: ignore[attr-defined]
    assert fw["framework"] == "langchain"
    assert "operation" in fw and fw["operation"] == "generate"
    assert fw["stream"] is False
    assert "model" in fw


def test_framework_ctx_includes_stop_sequences(model: CorpusLangChainLLM) -> None:
    _, _, fw = model._build_context_and_params(operation="generate", stream=False, stop=["a", "b"])  # type: ignore[attr-defined]
    assert fw["stop_sequences"] == ["a", "b"]


def test_framework_ctx_includes_langchain_config_when_present(model: CorpusLangChainLLM) -> None:
    cfg = {"tags": ["x"], "metadata": {"k": "v"}}
    _, _, fw = model._build_context_and_params(operation="generate", stream=False, config=cfg)  # type: ignore[attr-defined]
    assert fw.get("langchain_config") == cfg


# =============================================================================
# 5) Generation behavior (sync + async) (10 tests)
# =============================================================================

def test_generate_calls_translator_complete(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_completion = _mk_completion("hello", model="test-model")

    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="test-model")
    res = m._generate([HumanMessage(content="hi")])  # type: ignore[attr-defined]
    assert hasattr(res, "generations")
    assert res.generations[0].message.content == "hello"
    assert any(c.name == "complete" for c in fake_adapter.calls)


@pytest.mark.asyncio
async def test_agenerate_calls_translator_arun_complete(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_completion = _mk_completion("hello-async", model="test-model")

    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="test-model")
    res = await m._agenerate([HumanMessage(content="hi")])  # type: ignore[attr-defined]
    assert res.generations[0].message.content == "hello-async"
    assert any(c.name == "complete" for c in fake_adapter.calls)


def test_generate_forwards_stop_sequences(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    _ = m._generate([HumanMessage(content="hi")], stop=["STOP"])  # type: ignore[attr-defined]
    call = next(c for c in fake_adapter.calls if c.name == "complete")
    assert call.kwargs["stop_sequences"] == ["STOP"]


@pytest.mark.asyncio
async def test_agenerate_forwards_stop_sequences(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    _ = await m._agenerate([HumanMessage(content="hi")], stop=["STOP"])  # type: ignore[attr-defined]
    call = next(c for c in fake_adapter.calls if c.name == "complete")
    assert call.kwargs["stop_sequences"] == ["STOP"]


def test_generate_forwards_sampling_params(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m", temperature=0.1, max_tokens=10)
    _ = m._generate([HumanMessage(content="hi")], temperature=0.9, max_tokens=99, top_p=0.5, frequency_penalty=0.2, presence_penalty=0.3)  # type: ignore[attr-defined]
    call = next(c for c in fake_adapter.calls if c.name == "complete")
    assert call.kwargs["temperature"] == 0.9
    assert call.kwargs["max_tokens"] == 99
    assert call.kwargs["top_p"] == 0.5
    assert call.kwargs["frequency_penalty"] == 0.2
    assert call.kwargs["presence_penalty"] == 0.3


@pytest.mark.asyncio
async def test_agenerate_forwards_sampling_params(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m", temperature=0.1, max_tokens=10)
    _ = await m._agenerate([HumanMessage(content="hi")], temperature=0.9, max_tokens=99, top_p=0.5, frequency_penalty=0.2, presence_penalty=0.3)  # type: ignore[attr-defined]
    call = next(c for c in fake_adapter.calls if c.name == "complete")
    assert call.kwargs["temperature"] == 0.9
    assert call.kwargs["max_tokens"] == 99


def test_generate_forwards_system_message_override(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    _ = m._generate([HumanMessage(content="hi")], system_message="SYS")  # type: ignore[attr-defined]
    call = next(c for c in fake_adapter.calls if c.name == "complete")
    assert call.kwargs["system_message"] == "SYS"


def test_generate_forwards_tools_and_tool_choice_string(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    tools = [{
        "type": "function",
        "function": {"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}},
    }]
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    _ = m._generate([HumanMessage(content="hi")], tools=tools, tool_choice="auto")  # type: ignore[attr-defined]
    call = next(c for c in fake_adapter.calls if c.name == "complete")
    assert call.kwargs["tools"] == tools
    assert call.kwargs["tool_choice"] == "auto"


def test_generate_forwards_tool_choice_dict(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    tools = [{
        "type": "function",
        "function": {"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}},
    }]
    tool_choice = {"type": "function", "function": {"name": "t"}}
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    _ = m._generate([HumanMessage(content="hi")], tools=tools, tool_choice=tool_choice)  # type: ignore[attr-defined]
    call = next(c for c in fake_adapter.calls if c.name == "complete")
    assert call.kwargs["tool_choice"] == tool_choice


# =============================================================================
# 6) Streaming behavior (sync + async) (10 tests)
# =============================================================================

def test_stream_yields_incremental_chunks(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_stream_chunks = [_mk_chunk("he"), _mk_chunk("ll"), _mk_chunk("o", is_final=True)]
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    chunks = list(m._stream([HumanMessage(content="hi")]))  # type: ignore[attr-defined]
    assert len(chunks) == 3
    assert "".join([c.message.content for c in chunks]) == "hello"


@pytest.mark.asyncio
async def test_astream_yields_incremental_chunks(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_stream_chunks = [_mk_chunk("a"), _mk_chunk("b"), _mk_chunk("c", is_final=True)]
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    out: List[str] = []
    async for ch in m._astream([HumanMessage(content="hi")]):  # type: ignore[attr-defined]
        out.append(ch.message.content)
    assert "".join(out) == "abc"


def test_stream_calls_sync_callbacks_in_order(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_stream_chunks = [_mk_chunk("x"), _mk_chunk("y", is_final=True)]
    rm = SyncRunManager()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    _ = list(m._stream([HumanMessage(content="hi")], run_manager=rm))  # type: ignore[attr-defined]
    assert [e[0] for e in rm.events][0] == "start"
    assert [e[0] for e in rm.events].count("token") == 2
    assert [e[0] for e in rm.events][-1] == "end"


@pytest.mark.asyncio
async def test_astream_calls_async_callbacks_in_order(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_stream_chunks = [_mk_chunk("x"), _mk_chunk("y", is_final=True)]
    rm = AsyncRunManager()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    _ = [c async for c in m._astream([HumanMessage(content="hi")], run_manager=rm)]  # type: ignore[attr-defined]
    assert [e[0] for e in rm.events][0] == "start"
    assert [e[0] for e in rm.events].count("token") == 2
    assert [e[0] for e in rm.events][-1] == "end"


def test_stream_callback_failure_cancels_stream(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_stream_chunks = [_mk_chunk("1"), _mk_chunk("2"), _mk_chunk("3", is_final=True)]
    rm = SyncRunManager()
    rm.raise_on_new_token = RuntimeError("callback failed")
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    chunks = list(m._stream([HumanMessage(content="hi")], run_manager=rm))  # type: ignore[attr-defined]
    # Stream stops early due to callback failure
    assert "".join([c.message.content for c in chunks]) in {"1", "12"}


@pytest.mark.asyncio
async def test_astream_callback_failure_cancels_stream(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_stream_chunks = [_mk_chunk("1"), _mk_chunk("2"), _mk_chunk("3", is_final=True)]
    rm = AsyncRunManager()
    rm.raise_on_new_token = RuntimeError("callback failed")
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    out: List[str] = []
    async for ch in m._astream([HumanMessage(content="hi")], run_manager=rm):  # type: ignore[attr-defined]
        out.append(ch.message.content)
    assert "".join(out) in {"1", "12"}


@pytest.mark.asyncio
async def test_astream_early_break_closes_underlying_generator(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_stream_chunks = [_mk_chunk("a"), _mk_chunk("b"), _mk_chunk("c"), _mk_chunk("d", is_final=True)]
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    ait = m._astream([HumanMessage(content="hi")])  # type: ignore[attr-defined]
    got = await _collect_async_iter(ait, limit=2)
    assert len(got) == 2
    # Force generator close via aclose if available
    if hasattr(ait, "aclose"):
        await ait.aclose()  # type: ignore[attr-defined]
    assert fake_adapter.stream_closed is True


def test_stream_forwards_stop_sequences(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_stream_chunks = [_mk_chunk("x", is_final=True)]
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    _ = list(m._stream([HumanMessage(content="hi")], stop=["STOP"]))  # type: ignore[attr-defined]
    call = next(c for c in fake_adapter.calls if c.name == "stream")
    assert call.kwargs["stop_sequences"] == ["STOP"]


@pytest.mark.asyncio
async def test_astream_forwards_stop_sequences(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_stream_chunks = [_mk_chunk("x", is_final=True)]
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    _ = [c async for c in m._astream([HumanMessage(content="hi")], stop=["STOP"])]  # type: ignore[attr-defined]
    call = next(c for c in fake_adapter.calls if c.name == "stream")
    assert call.kwargs["stop_sequences"] == ["STOP"]


# =============================================================================
# 7) Event-loop safety (2 tests)
# =============================================================================

@pytest.mark.asyncio
async def test_generate_raises_in_event_loop(model: CorpusLangChainLLM, basic_messages: List[BaseMessage]) -> None:
    with pytest.raises(RuntimeError) as e:
        model._generate(basic_messages)  # type: ignore[attr-defined]
    assert SYNC_WRAPPER_CALLED_IN_EVENT_LOOP in str(e.value)


@pytest.mark.asyncio
async def test_stream_raises_in_event_loop(model: CorpusLangChainLLM, basic_messages: List[BaseMessage]) -> None:
    with pytest.raises(RuntimeError) as e:
        list(model._stream(basic_messages))  # type: ignore[attr-defined]
    assert SYNC_WRAPPER_CALLED_IN_EVENT_LOOP in str(e.value)


# =============================================================================
# 8) Health & capabilities (translator-only contract) (8 tests)
# =============================================================================

def test_health_calls_translator_health_only(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")

    # Patch translator.health to bypass adapter.health entirely.
    def fake_health() -> Mapping[str, Any]:
        return {"patched": True}

    m._translator.health = fake_health  # type: ignore[method-assign]
    out = m.health()
    assert out["patched"] is True
    assert not any(c.name == "health" for c in fake_adapter.calls)


@pytest.mark.asyncio
async def test_ahealth_calls_translator_ahealth_when_present(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")

    async def fake_ahealth() -> Mapping[str, Any]:
        return {"patched_async": True}

    m._translator.ahealth = fake_ahealth  # type: ignore[attr-defined]
    out = await m.ahealth()
    assert out["patched_async"] is True
    assert not any(c.name == "health" for c in fake_adapter.calls)


@pytest.mark.asyncio
async def test_ahealth_falls_back_to_thread_when_no_ahealth(monkeypatch: pytest.MonkeyPatch, fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")

    # Ensure translator has no ahealth attribute
    if hasattr(m._translator, "ahealth"):
        delattr(m._translator, "ahealth")  # type: ignore[attr-defined]

    # Patch translator.health to return quickly; ensures fallback calls it.
    m._translator.health = lambda: {"fallback": True}  # type: ignore[method-assign]
    out = await m.ahealth()
    assert out["fallback"] is True


def test_capabilities_calls_translator_capabilities_only(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    m._translator.capabilities = lambda: {"caps": True}  # type: ignore[method-assign]
    out = m.capabilities()
    assert out["caps"] is True
    assert not any(c.name == "capabilities" for c in fake_adapter.calls)


@pytest.mark.asyncio
async def test_acapabilities_calls_translator_acapabilities_when_present(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")

    async def fake_acaps() -> Mapping[str, Any]:
        return {"caps_async": True}

    m._translator.acapabilities = fake_acaps  # type: ignore[attr-defined]
    out = await m.acapabilities()
    assert out["caps_async"] is True
    assert not any(c.name == "capabilities" for c in fake_adapter.calls)


@pytest.mark.asyncio
async def test_acapabilities_falls_back_to_thread_when_no_acapabilities(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")

    if hasattr(m._translator, "acapabilities"):
        delattr(m._translator, "acapabilities")  # type: ignore[attr-defined]

    m._translator.capabilities = lambda: {"caps_fallback": True}  # type: ignore[method-assign]
    out = await m.acapabilities()
    assert out["caps_fallback"] is True


def test_health_raises_typeerror_if_translator_returns_non_mapping(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    m._translator.health = lambda: "bad"  # type: ignore[method-assign]
    with pytest.raises(TypeError):
        _ = m.health()


def test_capabilities_raises_typeerror_if_translator_returns_non_mapping(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    m._translator.capabilities = lambda: "bad"  # type: ignore[method-assign]
    with pytest.raises(TypeError):
        _ = m.capabilities()


# =============================================================================
# 9) Token counting (6 tests)
# =============================================================================

def test_token_count_calls_translator_count_tokens_for_messages(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    # Spy by patching translator method
    called: Dict[str, Any] = {"n": 0}

    def fake_count_tokens_for_messages(*args: Any, **kwargs: Any) -> Any:
        called["n"] += 1
        return 7

    m._translator.count_tokens_for_messages = fake_count_tokens_for_messages  # type: ignore[method-assign]
    n = m.get_num_tokens_from_messages([HumanMessage(content="hi")])
    assert n == 7
    assert called["n"] == 1


def test_token_count_accepts_int_result(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    m._translator.count_tokens_for_messages = lambda **kwargs: 9  # type: ignore[method-assign]
    assert m.get_num_tokens_from_messages([HumanMessage(content="hi")]) == 9


def test_token_count_accepts_mapping_tokens_key(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    m._translator.count_tokens_for_messages = lambda **kwargs: {"tokens": 11}  # type: ignore[method-assign]
    assert m.get_num_tokens_from_messages([HumanMessage(content="hi")]) == 11


def test_token_count_accepts_mapping_count_key(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    m._translator.count_tokens_for_messages = lambda **kwargs: {"count": 12}  # type: ignore[method-assign]
    assert m.get_num_tokens_from_messages([HumanMessage(content="hi")]) == 12


def test_token_count_accepts_mapping_total_tokens_key(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    m._translator.count_tokens_for_messages = lambda **kwargs: {"total_tokens": 13}  # type: ignore[method-assign]
    assert m.get_num_tokens_from_messages([HumanMessage(content="hi")]) == 13


def test_token_count_raises_on_unexpected_return_type(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    m._translator.count_tokens_for_messages = lambda **kwargs: object()  # type: ignore[method-assign]
    with pytest.raises(TypeError):
        _ = m.get_num_tokens_from_messages([HumanMessage(content="hi")])


# =============================================================================
# 10) Error context attachment / propagation (6 tests)
# =============================================================================

def test_generate_propagates_exception_and_calls_on_llm_error(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.raise_on_complete = RuntimeError("boom")
    rm = SyncRunManager()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    with pytest.raises(RuntimeError):
        _ = m._generate([HumanMessage(content="hi")], run_manager=rm)  # type: ignore[attr-defined]
    assert any(ev[0] == "error" for ev in rm.events)


@pytest.mark.asyncio
async def test_agenerate_propagates_exception_and_calls_on_llm_error(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.raise_on_complete = RuntimeError("boom")
    rm = AsyncRunManager()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    with pytest.raises(RuntimeError):
        _ = await m._agenerate([HumanMessage(content="hi")], run_manager=rm)  # type: ignore[attr-defined]
    assert any(ev[0] == "error" for ev in rm.events)


def test_stream_propagates_exception_and_calls_on_llm_error(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_stream_chunks = [_mk_chunk("a"), _mk_chunk("b"), _mk_chunk("c")]
    fake_adapter.raise_on_stream_at_index = 1
    fake_adapter.raise_on_stream_exc = RuntimeError("stream boom")
    rm = SyncRunManager()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    with pytest.raises(RuntimeError):
        _ = list(m._stream([HumanMessage(content="hi")], run_manager=rm))  # type: ignore[attr-defined]
    assert any(ev[0] == "error" for ev in rm.events)


@pytest.mark.asyncio
async def test_astream_propagates_exception_and_calls_on_llm_error(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_stream_chunks = [_mk_chunk("a"), _mk_chunk("b"), _mk_chunk("c")]
    fake_adapter.raise_on_stream_at_index = 1
    fake_adapter.raise_on_stream_exc = RuntimeError("stream boom")
    rm = AsyncRunManager()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    with pytest.raises(RuntimeError):
        _ = [c async for c in m._astream([HumanMessage(content="hi")], run_manager=rm)]  # type: ignore[attr-defined]
    assert any(ev[0] == "error" for ev in rm.events)


def test_error_context_decorator_does_not_break_success_path(model: CorpusLangChainLLM) -> None:
    """
    Smoke test: ensure decorated methods still work on success.
    """
    out = model.capabilities()
    assert isinstance(out, Mapping)


@pytest.mark.asyncio
async def test_error_context_decorator_does_not_break_success_path_async(model: CorpusLangChainLLM) -> None:
    out = await model.acapabilities()
    assert isinstance(out, Mapping)


# =============================================================================
# Direct integration tests (adapter + real LLMTranslator + fake protocol) (5 tests)
# =============================================================================

def test_integration_generate_end_to_end_basic(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_completion = _mk_completion("integration-ok", model="m")
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    res = m._generate([HumanMessage(content="hi")])  # type: ignore[attr-defined]
    assert res.generations[0].message.content == "integration-ok"
    call = next(c for c in fake_adapter.calls if c.name == "complete")
    assert call.kwargs["messages"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_integration_agenerate_end_to_end_with_stop_and_system(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_completion = _mk_completion("integration-async", model="m")
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    _ = await m._agenerate([HumanMessage(content="hi")], stop=["STOP"], system_message="SYS")  # type: ignore[attr-defined]
    call = next(c for c in fake_adapter.calls if c.name == "complete")
    assert call.kwargs["stop_sequences"] == ["STOP"]
    assert call.kwargs["system_message"] == "SYS"


def test_integration_generate_with_tools_forwarding(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    tools = [{
        "type": "function",
        "function": {"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}},
    }]
    fake_adapter.next_completion = _mk_completion("tools-ok", model="m")
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    _ = m._generate([HumanMessage(content="hi")], tools=tools, tool_choice="auto")  # type: ignore[attr-defined]
    call = next(c for c in fake_adapter.calls if c.name == "complete")
    assert call.kwargs["tools"] == tools
    assert call.kwargs["tool_choice"] == "auto"


def test_integration_stream_end_to_end_yields_chunks(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_stream_chunks = [_mk_chunk("he"), _mk_chunk("y", is_final=True)]
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    out = "".join([c.message.content for c in m._stream([HumanMessage(content="hi")])])  # type: ignore[attr-defined]
    assert out == "hey"


@pytest.mark.asyncio
async def test_integration_astream_end_to_end_yields_chunks_and_callbacks(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_stream_chunks = [_mk_chunk("a"), _mk_chunk("b", is_final=True)]
    rm = AsyncRunManager()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    out = ""
    async for c in m._astream([HumanMessage(content="hi")], run_manager=rm):  # type: ignore[attr-defined]
        out += c.message.content
    assert out == "ab"
    assert [e[0] for e in rm.events][0] == "start"
    assert [e[0] for e in rm.events].count("token") == 2
    assert [e[0] for e in rm.events][-1] == "end"


# =============================================================================
# E2E tests (LangChain public entrypoints) (6 tests)
# =============================================================================

def test_e2e_invoke_returns_ai_message(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_completion = _mk_completion("hello", model="m")
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    out = m.invoke([HumanMessage(content="hi")])
    assert isinstance(out, AIMessage)
    assert out.content == "hello"


@pytest.mark.asyncio
async def test_e2e_ainvoke_returns_ai_message(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_completion = _mk_completion("hello-async", model="m")
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    out = await m.ainvoke([HumanMessage(content="hi")])
    assert isinstance(out, AIMessage)
    assert out.content == "hello-async"


def test_e2e_stream_emits_incremental_chunks(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_stream_chunks = [_mk_chunk("he"), _mk_chunk("ll"), _mk_chunk("o", is_final=True)]
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    parts: List[str] = []
    for ch in m.stream([HumanMessage(content="hi")]):
        # LangChain yields message chunks; content may be string
        parts.append(getattr(ch, "content", "") or "")
    assert "".join(parts) == "hello"


@pytest.mark.asyncio
async def test_e2e_astream_emits_incremental_chunks(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_stream_chunks = [_mk_chunk("a"), _mk_chunk("b"), _mk_chunk("c", is_final=True)]
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    parts: List[str] = []
    async for ch in m.astream([HumanMessage(content="hi")]):
        parts.append(getattr(ch, "content", "") or "")
    assert "".join(parts) == "abc"


@pytest.mark.asyncio
async def test_e2e_astream_early_cancel_closes_generator(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_stream_chunks = [_mk_chunk("a"), _mk_chunk("b"), _mk_chunk("c"), _mk_chunk("d", is_final=True)]
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")

    ait = m.astream([HumanMessage(content="hi")])
    # consume a couple
    got: List[Any] = []
    async for i, ch in _enumerate_async(ait):
        got.append(ch)
        if i >= 1:
            break
    # explicitly close if possible
    if hasattr(ait, "aclose"):
        await ait.aclose()  # type: ignore[attr-defined]
    assert fake_adapter.stream_closed is True
    assert len(got) == 2


def test_e2e_tools_forwarded(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    tools = [{
        "type": "function",
        "function": {"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}},
    }]
    fake_adapter.next_completion = _mk_completion("tooling", model="m")
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    out = m.invoke([HumanMessage(content="hi")], tools=tools, tool_choice="auto")
    assert isinstance(out, AIMessage)
    call = next(c for c in fake_adapter.calls if c.name == "complete")
    assert call.kwargs["tools"] == tools
    assert call.kwargs["tool_choice"] == "auto"


# Helper used by E2E cancellation test
async def _enumerate_async(ait: AsyncIterator[Any]) -> AsyncIterator[Tuple[int, Any]]:
    idx = 0
    async for item in ait:
        yield idx, item
        idx += 1


# =============================================================================
# Additional coverage: small but important edge cases (10 tests)
# =============================================================================

def test_identifying_params_includes_framework_version(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m", framework_version="1.2.3")
    p = m._identifying_params
    assert p["framework_version"] == "1.2.3"


def test_llm_type_is_corpus(model: CorpusLangChainLLM) -> None:
    assert model._llm_type == "corpus"


def test_generate_start_callback_includes_run_id(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    rm = SyncRunManager()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    _ = m._generate([HumanMessage(content="hi")], run_manager=rm)  # type: ignore[attr-defined]
    start = next(ev for ev in rm.events if ev[0] == "start")
    assert "run_id" in start[1]["kwargs"]


@pytest.mark.asyncio
async def test_agenerate_start_callback_includes_run_id(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    rm = AsyncRunManager()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    _ = await m._agenerate([HumanMessage(content="hi")], run_manager=rm)  # type: ignore[attr-defined]
    start = next(ev for ev in rm.events if ev[0] == "start")
    assert "run_id" in start[1]["kwargs"]


def test_stream_end_callback_contains_streaming_marker(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    rm = SyncRunManager()
    fake_adapter.next_stream_chunks = [_mk_chunk("x", is_final=True)]
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    _ = list(m._stream([HumanMessage(content="hi")], run_manager=rm))  # type: ignore[attr-defined]
    end = next(ev for ev in rm.events if ev[0] == "end")[1]
    # The adapter emits a synthetic end ChatResult with generation_info markers
    gen_info = end.generations[0].generation_info
    assert gen_info.get("streaming") is True
    assert gen_info.get("completed") is True


@pytest.mark.asyncio
async def test_astream_end_callback_contains_streaming_marker(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    rm = AsyncRunManager()
    fake_adapter.next_stream_chunks = [_mk_chunk("x", is_final=True)]
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="m")
    _ = [c async for c in m._astream([HumanMessage(content="hi")], run_manager=rm)]  # type: ignore[attr-defined]
    end = next(ev for ev in rm.events if ev[0] == "end")[1]
    gen_info = end.generations[0].generation_info
    assert gen_info.get("streaming") is True
    assert gen_info.get("completed") is True


def test_generate_uses_default_model_when_not_overridden(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="default-model")
    _ = m._generate([HumanMessage(content="hi")])  # type: ignore[attr-defined]
    call = next(c for c in fake_adapter.calls if c.name == "complete")
    assert call.kwargs["model"] == "default-model"


def test_generate_allows_model_override_in_kwargs(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="default-model")
    _ = m._generate([HumanMessage(content="hi")], model="override-model")  # type: ignore[attr-defined]
    call = next(c for c in fake_adapter.calls if c.name == "complete")
    assert call.kwargs["model"] == "override-model"


def test_stream_allows_model_override_in_kwargs(fake_adapter: FakeLLMProtocolV1) -> None:
    _require_langchain()
    fake_adapter.next_stream_chunks = [_mk_chunk("x", is_final=True)]
    m = CorpusLangChainLLM(llm_adapter=fake_adapter, model="default-model")
    _ = list(m._stream([HumanMessage(content="hi")], model="override-model"))  # type: ignore[attr-defined]
    call = next(c for c in fake_adapter.calls if c.name == "stream")
    assert call.kwargs["model"] == "override-model"


def test_get_num_tokens_uses_humanmessage_wrapper(model: CorpusLangChainLLM) -> None:
    # Uses get_num_tokens_from_messages internally; we just sanity-check it runs.
    n = model.get_num_tokens("hi")
    assert isinstance(n, int)


@pytest.mark.asyncio
async def test_aget_num_tokens_from_messages_delegates_sync(model: CorpusLangChainLLM) -> None:
    n = await model.aget_num_tokens_from_messages([HumanMessage(content="hi")])
    assert isinstance(n, int)


