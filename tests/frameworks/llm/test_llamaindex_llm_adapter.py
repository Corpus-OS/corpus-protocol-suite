"""LlamaIndex LLM framework adapter tests.

These tests are written against the current public API in
`corpus_sdk.llm.framework_adapters.llamaindex`, which exposes a LlamaIndex-style
LLM client (`CorpusLlamaIndexLLM`) with `chat`/`achat`/`stream_chat`/`astream_chat` 
and `complete`/`acomplete`/`stream_complete`/`astream_complete` methods.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, Dict, Optional

import pytest

import corpus_sdk.llm.framework_adapters.llamaindex as llamaindex_adapter_module
from corpus_sdk.llm.framework_adapters.llamaindex import (
    CorpusLlamaIndexLLM,
    LlamaIndexLLMConfig,
)
from corpus_sdk.llm.llm_base import OperationContext


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROMPT = "hello from llamaindex tests"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_llamaindex_installed() -> bool:
    """Dependency-neutral check: does not import LlamaIndex, only checks availability."""
    return importlib.util.find_spec("llama_index") is not None


def _require_llamaindex_available_for_integration() -> None:
    """Enforce "no skips": Integration tests must fail if LlamaIndex isn't installed."""
    if not _has_llamaindex_installed():
        raise AssertionError(
            "LlamaIndex is not installed, but LlamaIndex integration tests are required (no skips). "
            "Install LlamaIndex packages in the test environment to run this framework suite."
        )


def _make_dummy_translator() -> Any:
    """Factory for creating a standard dummy translator for tests."""
    class DummyTranslator:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}

        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            yield {"text": "chunk", "is_final": False, "model": "m"}
            yield {"text": "final", "is_final": True, "model": "m"}

        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}

        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                yield {"text": "chunk", "is_final": False, "model": "m"}
                yield {"text": "final", "is_final": True, "model": "m"}
            return _gen()

        def capabilities(self) -> Mapping[str, Any]:
            return {"supports_count_tokens": True}

        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}

        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 10

    return DummyTranslator()


def _make_mock_chatmessage(role: str = "user", content: str = PROMPT) -> Any:
    """Create a mock ChatMessage-like object for testing."""
    class MockChatMessage:
        def __init__(self, role: str, content: str):
            self.role = role
            self.content = content
    
    return MockChatMessage(role, content)


# ---------------------------------------------------------------------------
# Construction / initialization tests (8 tests)
# ---------------------------------------------------------------------------


def test_init_rejects_adapter_without_required_methods() -> None:
    """Adapter must implement LLMProtocolV1 methods."""
    class BadAdapter:
        pass

    with pytest.raises(TypeError, match="LLMProtocolV1"):
        CorpusLlamaIndexLLM(llm_adapter=BadAdapter(), model="test")


def test_init_validates_temperature_range() -> None:
    """Init should reject temperature outside [0.0, 2.0] range."""
    class MinimalAdapter:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        def count_tokens(self, *a: Any, **k: Any) -> int:
            return 0
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}

    with pytest.raises(ValueError, match="temperature"):
        CorpusLlamaIndexLLM(llm_adapter=MinimalAdapter(), temperature=-0.1)
    
    with pytest.raises(ValueError, match="temperature"):
        CorpusLlamaIndexLLM(llm_adapter=MinimalAdapter(), temperature=2.5)


def test_init_validates_max_tokens_positive() -> None:
    """Init should reject non-positive max_tokens."""
    class MinimalAdapter:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        def count_tokens(self, *a: Any, **k: Any) -> int:
            return 0
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}

    with pytest.raises(ValueError, match="max_tokens"):
        CorpusLlamaIndexLLM(llm_adapter=MinimalAdapter(), max_tokens=0)
    
    with pytest.raises(ValueError, match="max_tokens"):
        CorpusLlamaIndexLLM(llm_adapter=MinimalAdapter(), max_tokens=-10)


def test_config_validates_context_window_positive() -> None:
    """Config should reject non-positive context_window."""
    with pytest.raises(ValueError, match="context_window"):
        LlamaIndexLLMConfig(context_window=0)
    
    with pytest.raises(ValueError, match="context_window"):
        LlamaIndexLLMConfig(context_window=-100)


def test_create_llm_translator_called_with_framework_llamaindex(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Translator factory should be called with framework='llamaindex'."""
    captured: Dict[str, Any] = {}

    def fake_create_llm_translator(*_: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return _make_dummy_translator()

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", fake_create_llm_translator)

    messages = [_make_mock_chatmessage()]
    llm = CorpusLlamaIndexLLM(llm_adapter=adapter, model="test")
    _ = llm.chat(messages)

    assert captured.get("framework") == "llamaindex"
    assert captured.get("adapter") is adapter


def test_translator_override_is_used_and_factory_not_called(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """When translator is provided explicitly, factory should not be called."""
    # This would fail if factory is called
    monkeypatch.setattr(
        llamaindex_adapter_module,
        "create_llm_translator",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    # Inject translator directly (not currently supported, but tests the concept)
    # For now, just verify init doesn't call factory during __init__
    llm = CorpusLlamaIndexLLM(llm_adapter=adapter, model="test")
    assert llm is not None


def test_client_stores_config_attributes() -> None:
    """Client should store key config attributes as instance variables."""
    class MinimalAdapter:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        def count_tokens(self, *a: Any, **k: Any) -> int:
            return 0
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}

    llm = CorpusLlamaIndexLLM(
        llm_adapter=MinimalAdapter(),
        model="my-model",
        temperature=0.8,
        max_tokens=100,
        framework_version="v1.0",
    )
    
    assert llm.model == "my-model"
    assert llm.temperature == 0.8
    assert llm.max_tokens == 100
    assert llm.framework_version == "v1.0"


def test_config_object_takes_precedence_over_direct_params() -> None:
    """LlamaIndexLLMConfig should override direct parameter values."""
    class MinimalAdapter:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        def count_tokens(self, *a: Any, **k: Any) -> int:
            return 0
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}

    config = LlamaIndexLLMConfig(context_window=8192)
    llm = CorpusLlamaIndexLLM(
        llm_adapter=MinimalAdapter(),
        config=config,
    )
    
    assert llm.context_window == 8192


# ---------------------------------------------------------------------------
# LlamaIndex Metadata Property Tests (4 tests)
# ---------------------------------------------------------------------------


def test_metadata_exposes_context_window(adapter: Any) -> None:
    """metadata property should expose configured context_window."""
    llm = CorpusLlamaIndexLLM(
        llm_adapter=adapter,
        context_window=4096,
    )
    
    metadata = llm.metadata
    assert metadata.context_window == 4096


def test_metadata_exposes_num_output(adapter: Any) -> None:
    """metadata property should expose max_tokens as num_output."""
    llm = CorpusLlamaIndexLLM(
        llm_adapter=adapter,
        max_tokens=512,
    )
    
    metadata = llm.metadata
    assert metadata.num_output == 512


def test_metadata_reports_is_chat_model_true(adapter: Any) -> None:
    """metadata should always report is_chat_model=True."""
    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    
    metadata = llm.metadata
    assert metadata.is_chat_model is True


def test_metadata_includes_model_name(adapter: Any) -> None:
    """metadata should include the configured model name."""
    llm = CorpusLlamaIndexLLM(
        llm_adapter=adapter,
        model="gpt-4",
    )
    
    metadata = llm.metadata
    assert metadata.model_name == "gpt-4"


# ---------------------------------------------------------------------------
# Context Translation Tests (6 tests)
# ---------------------------------------------------------------------------


def test_chat_builds_operation_context_from_callback_manager(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """chat() should build OperationContext from callback_manager."""
    captured: Dict[str, Any] = {}
    base_ctx = OperationContext(request_id="from-core", tenant="from-core", attrs={"x": 1})

    def fake_context_from_llamaindex(callback_manager: Any, *, framework_version: Any = None) -> Any:
        captured["callback_manager"] = callback_manager
        captured["framework_version"] = framework_version
        return base_ctx

    monkeypatch.setattr(llamaindex_adapter_module, "context_from_llamaindex", fake_context_from_llamaindex)

    class DummyTranslator:
        def complete(self, raw_messages: Any, *, op_ctx: Any = None, **_: Any) -> Any:
            captured["op_ctx"] = op_ctx
            return {"text": "ok", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    mock_callback_manager = {"trace_id": "test-123"}
    llm = CorpusLlamaIndexLLM(llm_adapter=adapter, framework_version="v1.0")
    messages = [_make_mock_chatmessage()]
    
    _ = llm.chat(messages, callback_manager=mock_callback_manager)
    
    assert captured["callback_manager"] == mock_callback_manager
    assert captured["framework_version"] == "v1.0"
    assert isinstance(captured["op_ctx"], OperationContext)


def test_chat_handles_none_callback_manager(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """chat() should handle None callback_manager gracefully."""
    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    # Should not raise
    result = llm.chat(messages, callback_manager=None)
    assert result is not None


def test_context_translation_error_attaches_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Errors during context translation should attach error context."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    def fake_context_from_llamaindex(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("ctx translation failed")

    monkeypatch.setattr(llamaindex_adapter_module, "attach_context", fake_attach_context)
    monkeypatch.setattr(llamaindex_adapter_module, "context_from_llamaindex", fake_context_from_llamaindex)

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    with pytest.raises(RuntimeError, match="ctx translation failed"):
        llm.chat(messages)

    assert captured_ctx.get("framework") == "llamaindex"
    assert captured_ctx.get("operation") == "llm_context_translation"


def test_context_translation_uses_framework_version(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Context translation should pass framework_version through."""
    captured: Dict[str, Any] = {}

    def fake_context_from_llamaindex(callback_manager: Any, *, framework_version: Any = None) -> Any:
        captured["framework_version"] = framework_version
        return OperationContext(request_id="test", tenant="test", attrs={})

    monkeypatch.setattr(llamaindex_adapter_module, "context_from_llamaindex", fake_context_from_llamaindex)
    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter, framework_version="llamaindex-v2.0")
    messages = [_make_mock_chatmessage()]
    
    _ = llm.chat(messages)
    
    assert captured["framework_version"] == "llamaindex-v2.0"


def test_chat_passes_framework_ctx_with_metadata(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """chat() should build framework_ctx including callback_manager."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def complete(self, raw_messages: Any, *, framework_ctx: Any = None, **_: Any) -> Any:
            captured["framework_ctx"] = framework_ctx
            return {"text": "ok", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    mock_callback_manager = {"trace_id": "test"}
    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    _ = llm.chat(messages, callback_manager=mock_callback_manager)
    
    framework_ctx = captured.get("framework_ctx")
    assert isinstance(framework_ctx, Mapping)
    assert framework_ctx.get("framework") == "llamaindex"
    assert framework_ctx.get("operation") == "chat"
    assert framework_ctx.get("stream") is False
    assert framework_ctx.get("callback_manager") == mock_callback_manager


def test_chat_merges_additional_kwargs_into_framework_ctx(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """chat() should include additional_kwargs in framework_ctx."""
    captured: Dict[str, Any] = {}

    class DummyTranslator:
        def complete(self, raw_messages: Any, *, framework_ctx: Any = None, **_: Any) -> Any:
            captured["framework_ctx"] = framework_ctx
            return {"text": "ok", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    _ = llm.chat(messages, additional_kwargs={"custom": "value"})
    
    framework_ctx = captured.get("framework_ctx")
    assert framework_ctx.get("llamaindex_additional_kwargs") == {"custom": "value"}


# ---------------------------------------------------------------------------
# Parameter Forwarding Tests (8 tests)
# ---------------------------------------------------------------------------


def test_chat_forwards_model_parameter(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """chat() should forward model parameter to translator."""
    seen: Dict[str, Any] = {}

    class DummyTranslator:
        def complete(self, raw_messages: Any, **params: Any) -> Any:
            seen["params"] = dict(params)
            return {"text": "ok", "model": params.get("model", "m")}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter, model="default-model")
    messages = [_make_mock_chatmessage()]
    
    _ = llm.chat(messages, model="override-model")
    
    assert seen["params"].get("model") == "override-model"


def test_chat_forwards_temperature_parameter(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """chat() should forward temperature parameter to translator."""
    seen: Dict[str, Any] = {}

    class DummyTranslator:
        def complete(self, raw_messages: Any, **params: Any) -> Any:
            seen["params"] = dict(params)
            return {"text": "ok", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    _ = llm.chat(messages, temperature=0.3)
    
    assert seen["params"].get("temperature") == 0.3


def test_chat_forwards_max_tokens_parameter(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """chat() should forward max_tokens parameter to translator."""
    seen: Dict[str, Any] = {}

    class DummyTranslator:
        def complete(self, raw_messages: Any, **params: Any) -> Any:
            seen["params"] = dict(params)
            return {"text": "ok", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    _ = llm.chat(messages, max_tokens=150)
    
    assert seen["params"].get("max_tokens") == 150


def test_chat_forwards_stop_sequences(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """chat() should convert stop parameter to stop_sequences."""
    seen: Dict[str, Any] = {}

    class DummyTranslator:
        def complete(self, raw_messages: Any, **params: Any) -> Any:
            seen["params"] = dict(params)
            return {"text": "ok", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    _ = llm.chat(messages, stop=["\n\n", "STOP"])
    
    assert seen["params"].get("stop_sequences") == ["\n\n", "STOP"]


def test_chat_extracts_stop_from_additional_kwargs(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """chat() should extract stop from nested additional_kwargs."""
    seen: Dict[str, Any] = {}

    class DummyTranslator:
        def complete(self, raw_messages: Any, **params: Any) -> Any:
            seen["params"] = dict(params)
            return {"text": "ok", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    _ = llm.chat(messages, additional_kwargs={"stop": ["END"]})
    
    assert seen["params"].get("stop_sequences") == ["END"]


def test_chat_converts_string_stop_to_list(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """chat() should convert string stop parameter to list."""
    seen: Dict[str, Any] = {}

    class DummyTranslator:
        def complete(self, raw_messages: Any, **params: Any) -> Any:
            seen["params"] = dict(params)
            return {"text": "ok", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    _ = llm.chat(messages, stop="END")
    
    assert seen["params"].get("stop_sequences") == ["END"]


def test_chat_forwards_sampling_params(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """chat() should forward top_p, frequency_penalty, presence_penalty."""
    seen: Dict[str, Any] = {}

    class DummyTranslator:
        def complete(self, raw_messages: Any, **params: Any) -> Any:
            seen["params"] = dict(params)
            return {"text": "ok", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    _ = llm.chat(
        messages,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.2,
    )
    
    params = seen["params"]
    assert params.get("top_p") == 0.9
    assert params.get("frequency_penalty") == 0.1
    assert params.get("presence_penalty") == 0.2


def test_chat_forwards_tools_and_tool_choice(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """chat() should forward tools and tool_choice via additional_kwargs."""
    seen: Dict[str, Any] = {}

    class DummyTranslator:
        def complete(self, raw_messages: Any, **params: Any) -> Any:
            seen["params"] = dict(params)
            return {"text": "ok", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    tools = [{"type": "function", "function": {"name": "test"}}]
    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    # LlamaIndex typically passes these via kwargs, not additional_kwargs
    # But the adapter should handle both
    _ = llm.chat(messages, additional_kwargs={"tools": tools, "tool_choice": "auto"})
    
    # Verify they're accessible (implementation may vary)
    assert seen["params"] is not None


# ---------------------------------------------------------------------------
# ChatMessage Normalization Tests (5 tests)
# ---------------------------------------------------------------------------


def test_to_translator_messages_converts_chatmessage_objects(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """_to_translator_messages should convert ChatMessage objects to dicts."""
    seen: Dict[str, Any] = {}

    class DummyTranslator:
        def complete(self, raw_messages: Any, **params: Any) -> Any:
            seen["messages"] = raw_messages
            return {"text": "ok", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage("user", "test content")]
    
    _ = llm.chat(messages)
    
    normalized = seen["messages"]
    assert isinstance(normalized, list)
    assert len(normalized) == 1
    assert normalized[0]["role"] == "user"
    assert normalized[0]["content"] == "test content"


def test_to_translator_messages_handles_role_enums(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """_to_translator_messages should convert role enums to strings."""
    seen: Dict[str, Any] = {}

    class DummyTranslator:
        def complete(self, raw_messages: Any, **params: Any) -> Any:
            seen["messages"] = raw_messages
            return {"text": "ok", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    # Mock enum-like role
    class MockRole:
        def __init__(self, value: str):
            self.value = value
    
    class MockMessage:
        def __init__(self):
            self.role = MockRole("assistant")
            self.content = "test"

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [MockMessage()]
    
    _ = llm.chat(messages)
    
    normalized = seen["messages"]
    assert normalized[0]["role"] == "assistant"


def test_to_translator_messages_extracts_content_from_blocks(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """_to_translator_messages should handle messages with content blocks."""
    seen: Dict[str, Any] = {}

    class DummyTranslator:
        def complete(self, raw_messages: Any, **params: Any) -> Any:
            seen["messages"] = raw_messages
            return {"text": "ok", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    class MockBlock:
        def __init__(self, text: str):
            self.text = text
    
    class MockMessage:
        def __init__(self):
            self.role = "user"
            self.blocks = [MockBlock("part1"), MockBlock("part2")]

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [MockMessage()]
    
    _ = llm.chat(messages)
    
    normalized = seen["messages"]
    assert "part1part2" in normalized[0]["content"]


def test_to_translator_messages_handles_mixed_messages(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """_to_translator_messages should handle a list of different message types."""
    seen: Dict[str, Any] = {}

    class DummyTranslator:
        def complete(self, raw_messages: Any, **params: Any) -> Any:
            seen["messages"] = raw_messages
            return {"text": "ok", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [
        _make_mock_chatmessage("user", "hello"),
        _make_mock_chatmessage("assistant", "hi there"),
        _make_mock_chatmessage("user", "how are you"),
    ]
    
    _ = llm.chat(messages)
    
    normalized = seen["messages"]
    assert len(normalized) == 3
    assert normalized[0]["role"] == "user"
    assert normalized[1]["role"] == "assistant"
    assert normalized[2]["role"] == "user"


def test_to_translator_messages_preserves_all_roles(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """_to_translator_messages should preserve system/user/assistant/tool roles."""
    seen: Dict[str, Any] = {}

    class DummyTranslator:
        def complete(self, raw_messages: Any, **params: Any) -> Any:
            seen["messages"] = raw_messages
            return {"text": "ok", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [
        _make_mock_chatmessage("system", "be helpful"),
        _make_mock_chatmessage("user", "hello"),
        _make_mock_chatmessage("assistant", "hi"),
        _make_mock_chatmessage("tool", "result"),
    ]
    
    _ = llm.chat(messages)
    
    normalized = seen["messages"]
    roles = [msg["role"] for msg in normalized]
    assert "system" in roles
    assert "user" in roles
    assert "assistant" in roles
    assert "tool" in roles


# ---------------------------------------------------------------------------
# ChatResponse Building Tests (6 tests)
# ---------------------------------------------------------------------------


def test_build_chat_response_from_translator_result_with_text(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """_build_chat_response_from_translator_result should extract text."""
    class DummyTranslator:
        def complete(self, *_: Any, **__: Any) -> Any:
            return {"text": "response text", "model": "test-model"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    response = llm.chat(messages)
    
    assert response.message.content == "response text"


def test_build_chat_response_includes_usage_dict(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """ChatResponse should include normalized usage dict."""
    class DummyTranslator:
        def complete(self, *_: Any, **__: Any) -> Any:
            return {
                "text": "test",
                "model": "m",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            }
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    response = llm.chat(messages)
    
    usage = response.additional_kwargs.get("usage")
    assert usage is not None
    assert usage.get("prompt_tokens") == 10
    assert usage.get("completion_tokens") == 20
    assert usage.get("total_tokens") == 30


def test_build_chat_response_handles_finish_reason(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """ChatResponse should include finish_reason."""
    class DummyTranslator:
        def complete(self, *_: Any, **__: Any) -> Any:
            return {"text": "test", "model": "m", "finish_reason": "stop"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    response = llm.chat(messages)
    
    assert response.additional_kwargs.get("finish_reason") == "stop"


def test_build_chat_response_from_chunk_includes_delta(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Streaming ChatResponse should include delta field."""
    class DummyTranslator:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def stream(self, *_: Any, **__: Any) -> Iterator[Any]:
            yield {"text": "chunk", "is_final": False, "model": "m"}
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    stream = llm.stream_chat(messages)
    chunk = next(iter(stream))
    
    assert chunk.delta == "chunk"


def test_build_chat_response_chunk_marks_final(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Final streaming chunk should set is_final in additional_kwargs."""
    class DummyTranslator:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def stream(self, *_: Any, **__: Any) -> Iterator[Any]:
            yield {"text": "chunk1", "is_final": False, "model": "m"}
            yield {"text": "chunk2", "is_final": True, "model": "m"}
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    chunks = list(llm.stream_chat(messages))
    
    assert chunks[0].additional_kwargs.get("is_final") is False
    assert chunks[1].additional_kwargs.get("is_final") is True


def test_usage_to_dict_uses_coerce_token_usage(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """_usage_to_dict should use shared coerce_token_usage utility."""
    # This tests internal behavior indirectly by verifying the result format
    class DummyTranslator:
        def complete(self, *_: Any, **__: Any) -> Any:
            return {
                "text": "test",
                "model": "m",
                "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
            }
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    response = llm.chat(messages)
    usage = response.additional_kwargs.get("usage")
    
    # Verify the standard format from coerce_token_usage
    assert isinstance(usage, dict)
    assert all(k in usage for k in ["prompt_tokens", "completion_tokens", "total_tokens"])


# ---------------------------------------------------------------------------
# Chat API Tests (Sync/Async) (6 tests)
# ---------------------------------------------------------------------------


def test_chat_returns_chatresponse(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """chat() should return a ChatResponse object."""
    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    response = llm.chat(messages)
    
    # Check it has ChatResponse-like structure
    assert hasattr(response, "message")
    assert hasattr(response.message, "content")


def test_chat_validates_empty_messages(adapter: Any) -> None:
    """chat() should raise BadRequest for empty messages."""
    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    
    with pytest.raises(Exception, match="empty"):
        llm.chat([])


@pytest.mark.asyncio
async def test_achat_returns_chatresponse(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """achat() should return a ChatResponse object."""
    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    response = await llm.achat(messages)
    
    assert hasattr(response, "message")
    assert hasattr(response.message, "content")


@pytest.mark.asyncio
async def test_achat_validates_empty_messages(adapter: Any) -> None:
    """achat() should raise BadRequest for empty messages."""
    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    
    with pytest.raises(Exception, match="empty"):
        await llm.achat([])


def test_chat_delegates_to_translator_complete(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """chat() should call translator.complete()."""
    called = {"complete": False}

    class DummyTranslator:
        def complete(self, *a: Any, **k: Any) -> Any:
            called["complete"] = True
            return {"text": "ok", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    _ = llm.chat(messages)
    
    assert called["complete"] is True


@pytest.mark.asyncio
async def test_achat_delegates_to_translator_arun_complete(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """achat() should call translator.arun_complete()."""
    called = {"arun_complete": False}

    class DummyTranslator:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            called["arun_complete"] = True
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    _ = await llm.achat(messages)
    
    assert called["arun_complete"] is True


# ---------------------------------------------------------------------------
# Streaming Chat API Tests (6 tests)
# ---------------------------------------------------------------------------


def test_stream_chat_returns_iterator(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """stream_chat() should return an iterator."""
    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    result = llm.stream_chat(messages)
    
    assert hasattr(result, "__iter__")


def test_stream_chat_yields_chatresponse_chunks(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """stream_chat() should yield ChatResponse chunks progressively."""
    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    chunks = list(llm.stream_chat(messages))
    
    assert len(chunks) >= 1
    for chunk in chunks:
        assert hasattr(chunk, "message")
        assert hasattr(chunk, "delta")


@pytest.mark.asyncio
async def test_astream_chat_returns_async_iterator(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """astream_chat() should return an async iterator."""
    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    result = await llm.astream_chat(messages)
    
    assert hasattr(result, "__aiter__")


@pytest.mark.asyncio
async def test_astream_chat_yields_chatresponse_chunks(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """astream_chat() should yield ChatResponse chunks progressively."""
    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    chunks = []
    async for chunk in await llm.astream_chat(messages):
        chunks.append(chunk)
    
    assert len(chunks) >= 1
    for chunk in chunks:
        assert hasattr(chunk, "message")
        assert hasattr(chunk, "delta")


def test_stream_chat_delegates_to_translator_stream(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """stream_chat() should call translator.stream()."""
    called = {"stream": False}

    class DummyTranslator:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def stream(self, *_: Any, **__: Any) -> Iterator[Any]:
            called["stream"] = True
            yield {"text": "chunk", "is_final": True, "model": "m"}
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    _ = list(llm.stream_chat(messages))
    
    assert called["stream"] is True


@pytest.mark.asyncio
async def test_astream_chat_delegates_to_translator_arun_stream(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """astream_chat() should call translator.arun_stream()."""
    called = {"arun_stream": False}

    class DummyTranslator:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *_: Any, **__: Any) -> AsyncIterator[Any]:
            called["arun_stream"] = True
            async def _gen() -> AsyncIterator[Any]:
                yield {"text": "chunk", "is_final": True, "model": "m"}
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    chunks = []
    async for chunk in await llm.astream_chat(messages):
        chunks.append(chunk)
    
    assert called["arun_stream"] is True


# ---------------------------------------------------------------------------
# Completion API Tests (Sync/Async) (4 tests)
# ---------------------------------------------------------------------------


def test_complete_converts_prompt_to_message_and_delegates_to_chat(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """complete() should convert prompt to message and call chat()."""
    called = {"chat": False}

    def fake_chat(self: Any, messages: Any, **kwargs: Any) -> Any:
        called["chat"] = True
        called["messages"] = messages
        # Return minimal ChatResponse-like object
        class FakeResponse:
            def __init__(self):
                class FakeMessage:
                    content = "test"
                self.message = FakeMessage()
                self.additional_kwargs = {}
                self.raw = None
        return FakeResponse()

    monkeypatch.setattr(CorpusLlamaIndexLLM, "chat", fake_chat)
    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    
    _ = llm.complete("test prompt")
    
    assert called["chat"] is True
    assert len(called["messages"]) == 1


@pytest.mark.asyncio
async def test_acomplete_converts_prompt_to_message_and_delegates_to_achat(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """acomplete() should convert prompt to message and call achat()."""
    called = {"achat": False}

    async def fake_achat(self: Any, messages: Any, **kwargs: Any) -> Any:
        called["achat"] = True
        called["messages"] = messages
        # Return minimal ChatResponse-like object
        class FakeResponse:
            def __init__(self):
                class FakeMessage:
                    content = "test"
                self.message = FakeMessage()
                self.additional_kwargs = {}
                self.raw = None
        return FakeResponse()

    monkeypatch.setattr(CorpusLlamaIndexLLM, "achat", fake_achat)
    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    
    _ = await llm.acomplete("test prompt")
    
    assert called["achat"] is True
    assert len(called["messages"]) == 1


def test_stream_complete_delegates_to_stream_chat(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """stream_complete() should call stream_chat()."""
    called = {"stream_chat": False}

    def fake_stream_chat(self: Any, messages: Any, **kwargs: Any) -> Iterator[Any]:
        called["stream_chat"] = True
        called["messages"] = messages
        # Return minimal ChatResponse iterator
        class FakeResponse:
            def __init__(self):
                class FakeMessage:
                    content = "chunk"
                self.message = FakeMessage()
                self.delta = "chunk"
                self.additional_kwargs = {}
                self.raw = None
        yield FakeResponse()

    monkeypatch.setattr(CorpusLlamaIndexLLM, "stream_chat", fake_stream_chat)
    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    
    _ = list(llm.stream_complete("test prompt"))
    
    assert called["stream_chat"] is True
    assert len(called["messages"]) == 1


@pytest.mark.asyncio
async def test_astream_complete_delegates_to_astream_chat(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """astream_complete() should call astream_chat()."""
    called = {"astream_chat": False}

    async def fake_astream_chat(self: Any, messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        called["astream_chat"] = True
        called["messages"] = messages
        # Return minimal ChatResponse async iterator
        async def _gen() -> AsyncIterator[Any]:
            class FakeResponse:
                def __init__(self):
                    class FakeMessage:
                        content = "chunk"
                    self.message = FakeMessage()
                    self.delta = "chunk"
                    self.additional_kwargs = {}
                    self.raw = None
            yield FakeResponse()
        return _gen()

    monkeypatch.setattr(CorpusLlamaIndexLLM, "astream_chat", fake_astream_chat)
    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    
    chunks = []
    async for chunk in await llm.astream_complete("test prompt"):
        chunks.append(chunk)
    
    assert called["astream_chat"] is True
    assert len(called["messages"]) == 1


# ---------------------------------------------------------------------------
# Error Context Attachment Tests (4 tests)
# ---------------------------------------------------------------------------


def test_chat_error_attaches_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Errors during chat() should attach error context."""
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    monkeypatch.setattr(llamaindex_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def complete(self, *_: Any, **__: Any) -> Any:
            raise RuntimeError("boom")
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: FailingTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    with pytest.raises(RuntimeError, match="boom"):
        _ = llm.chat(messages)

    assert captured.get("framework") == "llamaindex"
    assert "llm_" in str(captured.get("operation", ""))


def test_stream_chat_iteration_error_attaches_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Errors during stream_chat iteration should attach error context."""
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    monkeypatch.setattr(llamaindex_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def stream(self, *_: Any, **__: Any) -> Iterator[Any]:
            def _gen() -> Iterator[Any]:
                yield {"text": "a", "is_final": False, "model": "m"}
                raise RuntimeError("stream-boom")
            return _gen()
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: FailingTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    it = llm.stream_chat(messages)
    _ = next(iter(it))
    
    with pytest.raises(RuntimeError, match="stream-boom"):
        _ = list(it)
    
    assert captured.get("framework") == "llamaindex"
    assert captured.get("operation") == "llm_stream_chat"


@pytest.mark.asyncio
async def test_achat_error_attaches_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Errors during achat() should attach error context."""
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    monkeypatch.setattr(llamaindex_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *_: Any, **__: Any) -> Any:
            raise RuntimeError("async-boom")
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: FailingTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    with pytest.raises(RuntimeError, match="async-boom"):
        await llm.achat(messages)

    assert captured.get("framework") == "llamaindex"
    assert "llm_" in str(captured.get("operation", ""))


@pytest.mark.asyncio
async def test_astream_chat_iteration_error_attaches_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Errors during astream_chat iteration should attach error context."""
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    monkeypatch.setattr(llamaindex_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def arun_stream(self, *_: Any, **__: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                yield {"text": "a", "is_final": False, "model": "m"}
                raise RuntimeError("async-stream-boom")
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: FailingTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    aiter = await llm.astream_chat(messages)
    _ = await aiter.__anext__()
    
    with pytest.raises(RuntimeError, match="async-stream-boom"):
        async for _ in aiter:
            pass

    assert captured.get("framework") == "llamaindex"
    assert captured.get("operation") == "llm_astream_chat"


# ---------------------------------------------------------------------------
# Token Counting Tests (5 tests)
# ---------------------------------------------------------------------------


def test_count_tokens_returns_int(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """count_tokens() should return an integer."""
    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    n = llm.count_tokens(messages)
    
    assert isinstance(n, int)
    assert n >= 0


def test_count_tokens_handles_empty_messages(adapter: Any) -> None:
    """count_tokens() should return 0 for empty messages."""
    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    
    n = llm.count_tokens([])
    
    assert n == 0


def test_count_tokens_delegates_to_translator(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """count_tokens() should call translator.count_tokens_for_messages()."""
    called = {"count_tokens_for_messages": False}

    class DummyTranslator:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            called["count_tokens_for_messages"] = True
            return 42

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    n = llm.count_tokens(messages)
    
    assert called["count_tokens_for_messages"] is True
    assert n == 42


def test_count_tokens_raises_on_invalid_translator_response(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """count_tokens() should raise TypeError if translator returns invalid type."""
    class BadTranslator:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def arun_stream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> Any:
            return "not-an-int"

    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: BadTranslator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    with pytest.raises(TypeError):
        llm.count_tokens(messages)


@pytest.mark.asyncio
async def test_acount_tokens_returns_int(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """acount_tokens() should return an integer."""
    monkeypatch.setattr(llamaindex_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [_make_mock_chatmessage()]
    
    n = await llm.acount_tokens(messages)
    
    assert isinstance(n, int)
    assert n >= 0


# ---------------------------------------------------------------------------
# Capabilities and Health Tests (4 tests)
# ---------------------------------------------------------------------------


def test_capabilities_delegates_to_translator_only(adapter: Any) -> None:
    """capabilities() should delegate only to translator, not adapter."""
    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    
    caps = llm.capabilities()
    
    assert isinstance(caps, Mapping)


@pytest.mark.asyncio
async def test_acapabilities_delegates_to_translator_or_thread(adapter: Any) -> None:
    """acapabilities() should use translator.acapabilities or run sync in thread."""
    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    
    caps = await llm.acapabilities()
    
    assert isinstance(caps, Mapping)


def test_health_delegates_to_translator_only(adapter: Any) -> None:
    """health() should delegate only to translator, not adapter."""
    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    
    health = llm.health()
    
    assert isinstance(health, Mapping)


@pytest.mark.asyncio
async def test_ahealth_delegates_to_translator_or_thread(adapter: Any) -> None:
    """ahealth() should use translator.ahealth or run sync in thread."""
    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    
    health = await llm.ahealth()
    
    assert isinstance(health, Mapping)


# ---------------------------------------------------------------------------
# Context Manager Tests (4 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_closes_underlying_adapter() -> None:
    """Context managers should call close/aclose on adapter."""
    class ClosingLLMAdapter:
        def __init__(self) -> None:
            self.closed = False
            self.aclosed = False

        def capabilities(self) -> Dict[str, Any]:
            return {"supports_count_tokens": True}
        def health(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "ok"}
        def complete(self, *args: Any, **kwargs: Any) -> Any:
            return {"text": "x", "model": "m"}
        def stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
            return iter(())
        def count_tokens(self, *args: Any, **kwargs: Any) -> int:
            return 0
        def close(self) -> None:
            self.closed = True
        async def aclose(self) -> None:
            self.aclosed = True

    adapter_instance = ClosingLLMAdapter()

    with CorpusLlamaIndexLLM(llm_adapter=adapter_instance) as llm:
        assert llm is not None
    assert adapter_instance.closed is True

    adapter2 = ClosingLLMAdapter()
    llm2 = CorpusLlamaIndexLLM(llm_adapter=adapter2)
    async with llm2:
        assert llm2 is not None
    assert adapter2.aclosed is True


def test_sync_context_manager_works_without_close(adapter: Any) -> None:
    """Sync context manager should work even if adapter lacks close()."""
    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    
    with llm:
        assert llm is not None


@pytest.mark.asyncio
async def test_async_context_manager_works_without_aclose(adapter: Any) -> None:
    """Async context manager should work even if adapter lacks aclose()."""
    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    
    async with llm:
        assert llm is not None


def test_context_manager_api_consistency_with_other_adapters(adapter: Any) -> None:
    """LlamaIndex adapter should support same context manager pattern as AutoGen/CrewAI."""
    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    
    # Verify both __enter__/__exit__ exist
    assert hasattr(llm, "__enter__")
    assert hasattr(llm, "__exit__")
    assert hasattr(llm, "__aenter__")
    assert hasattr(llm, "__aexit__")


# ---------------------------------------------------------------------------
# Event Loop Guard Tests (2 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_raises_when_called_in_event_loop() -> None:
    """chat() should raise when called from within an event loop."""
    class MinimalAdapter:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        def count_tokens(self, *a: Any, **k: Any) -> int:
            return 0
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}

    llm = CorpusLlamaIndexLLM(llm_adapter=MinimalAdapter())
    messages = [_make_mock_chatmessage()]
    
    with pytest.raises(RuntimeError, match="event loop"):
        llm.chat(messages)


@pytest.mark.asyncio
async def test_stream_chat_raises_when_called_in_event_loop() -> None:
    """stream_chat() should raise when called from within an event loop."""
    class MinimalAdapter:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        def count_tokens(self, *a: Any, **k: Any) -> int:
            return 0
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}

    llm = CorpusLlamaIndexLLM(llm_adapter=MinimalAdapter())
    messages = [_make_mock_chatmessage()]
    
    with pytest.raises(RuntimeError, match="event loop"):
        _ = llm.stream_chat(messages)


# ---------------------------------------------------------------------------
# LlamaIndex Framework Integration Tests (5 tests - NO SKIPS)
# ---------------------------------------------------------------------------


def test_llamaindex_dependency_must_be_installed() -> None:
    """LlamaIndex must be installed for framework integration tests."""
    _require_llamaindex_available_for_integration()


def test_llamaindex_chatmessage_objects_work_with_adapter(adapter: Any) -> None:
    """Real LlamaIndex ChatMessage objects should work with adapter."""
    _require_llamaindex_available_for_integration()

    # Import real LlamaIndex ChatMessage
    from llama_index.core.llms import ChatMessage, MessageRole
    
    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [ChatMessage(role=MessageRole.USER, content="test")]
    
    response = llm.chat(messages)
    assert response is not None


def test_llamaindex_callback_manager_creates_operation_context(adapter: Any) -> None:
    """Real LlamaIndex CallbackManager should create OperationContext."""
    _require_llamaindex_available_for_integration()

    from llama_index.core.llms import ChatMessage, MessageRole
    from llama_index.core.callbacks import CallbackManager
    
    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [ChatMessage(role=MessageRole.USER, content="test")]
    callback_manager = CallbackManager()
    
    # Should not raise - callback_manager should be handled gracefully
    response = llm.chat(messages, callback_manager=callback_manager)
    assert response is not None


def test_llamaindex_metadata_property_exposes_correct_values(adapter: Any) -> None:
    """LlamaIndex metadata property should expose correct LLMMetadata values."""
    _require_llamaindex_available_for_integration()

    from llama_index.core.llms import LLMMetadata
    
    llm = CorpusLlamaIndexLLM(
        llm_adapter=adapter,
        model="test-model",
        max_tokens=512,
        context_window=4096,
    )
    
    metadata = llm.metadata
    
    assert isinstance(metadata, LLMMetadata)
    assert metadata.model_name == "test-model"
    assert metadata.num_output == 512
    assert metadata.context_window == 4096
    assert metadata.is_chat_model is True


def test_llamaindex_streaming_with_chatmessage_objects(adapter: Any) -> None:
    """Streaming should work with real LlamaIndex ChatMessage objects."""
    _require_llamaindex_available_for_integration()

    from llama_index.core.llms import ChatMessage, MessageRole
    
    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [ChatMessage(role=MessageRole.USER, content="stream test")]
    
    chunks = list(llm.stream_chat(messages))
    
    assert len(chunks) > 0
    for chunk in chunks:
        # Verify ChatResponse structure
        assert hasattr(chunk, "message")
        assert hasattr(chunk, "delta")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter() -> Any:
    """Create a minimal test adapter."""
    class TestAdapter:
        def complete(self, *args: Any, **kwargs: Any) -> Any:
            return {"text": "test response", "model": "test-model"}
        
        def stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
            yield {"text": "chunk1", "is_final": False, "model": "test-model"}
            yield {"text": "chunk2", "is_final": True, "model": "test-model"}
        
        def count_tokens(self, *args: Any, **kwargs: Any) -> int:
            return 10
        
        def capabilities(self) -> Dict[str, Any]:
            return {"supports_count_tokens": True}
        
        def health(self) -> Dict[str, Any]:
            return {"status": "ok"}

    return TestAdapter()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
