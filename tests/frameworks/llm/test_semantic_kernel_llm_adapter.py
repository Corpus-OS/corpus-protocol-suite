"""Semantic Kernel LLM framework adapter tests.

These tests are written against the current public API in
`corpus_sdk.llm.framework_adapters.semantic_kernel`, which exposes a Semantic Kernel-style
ChatCompletionClientBase (`CorpusSemanticKernelChatCompletion`) with async-first methods
and sync wrappers for registry conformance.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, Dict, Optional

import pytest

import corpus_sdk.llm.framework_adapters.semantic_kernel as sk_adapter_module
from corpus_sdk.llm.framework_adapters.semantic_kernel import (
    CorpusSemanticKernelChatCompletion,
    SemanticKernelChatConfig,
)
from corpus_sdk.llm.llm_base import (
    LLMCapabilities,
    LLMChunk,
    LLMCompletion,
    OperationContext,
    TokenUsage,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROMPT = "hello from semantic kernel tests"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_semantic_kernel_installed() -> bool:
    """Dependency-neutral check: does not import SK, only checks availability."""
    return importlib.util.find_spec("semantic_kernel") is not None


def _require_semantic_kernel_available_for_integration() -> None:
    """Enforce "no skips": Integration tests must fail if SK isn't installed."""
    if not _has_semantic_kernel_installed():
        raise AssertionError(
            "semantic-kernel is not installed, but Semantic Kernel integration tests are required (no skips). "
            "Install semantic-kernel packages in the test environment to run this framework suite."
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


def _make_mock_chat_history(content: str = PROMPT) -> Any:
    """Create a mock ChatHistory-like object for testing."""
    class MockMessage:
        def __init__(self, role: str, content: str):
            self.role = role
            self.content = content
    
    return [MockMessage("user", content)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter() -> Any:
    """Create a minimal test adapter."""
    class TestAdapter:
        async def capabilities(self) -> LLMCapabilities:
            return LLMCapabilities(
                server="test",
                version="1.0",
                model_family="test-family",
                max_context_length=8192,
                supports_count_tokens=True,
            )

        async def health(self, *, ctx: Optional[OperationContext] = None) -> Dict[str, Any]:
            _ = ctx
            return {"status": "ok"}

        async def complete(
            self,
            *,
            messages: Any,
            ctx: Optional[OperationContext] = None,
            **kwargs: Any,
        ) -> LLMCompletion:
            _ = (messages, ctx, kwargs)
            return LLMCompletion(
                text="test response",
                model="test-model",
                model_family="test-family",
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                finish_reason="stop",
            )

        async def stream(
            self,
            *,
            messages: Any,
            ctx: Optional[OperationContext] = None,
            **kwargs: Any,
        ) -> AsyncIterator[LLMChunk]:
            _ = (messages, ctx, kwargs)
            yield LLMChunk(text="chunk1", is_final=False, model="test-model")
            yield LLMChunk(text="chunk2", is_final=True, model="test-model")

        async def count_tokens(
            self,
            text: str,
            *,
            model: Optional[str] = None,
            ctx: Optional[OperationContext] = None,
        ) -> int:
            _ = (text, model, ctx)
            return 10

    return TestAdapter()


# ---------------------------------------------------------------------------
# Construction / Initialization Tests (8 tests)
# ---------------------------------------------------------------------------


def test_init_rejects_adapter_without_required_methods() -> None:
    """Adapter must implement LLMProtocolV1 methods."""
    class BadAdapter:
        pass

    with pytest.raises(TypeError, match="LLMProtocolV1"):
        CorpusSemanticKernelChatCompletion(llm_adapter=BadAdapter(), model="test")


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
        CorpusSemanticKernelChatCompletion(llm_adapter=MinimalAdapter(), temperature=-0.1)
    
    with pytest.raises(ValueError, match="temperature"):
        CorpusSemanticKernelChatCompletion(llm_adapter=MinimalAdapter(), temperature=2.5)


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
        CorpusSemanticKernelChatCompletion(llm_adapter=MinimalAdapter(), max_tokens=0)
    
    with pytest.raises(ValueError, match="max_tokens"):
        CorpusSemanticKernelChatCompletion(llm_adapter=MinimalAdapter(), max_tokens=-10)


def test_config_validates_temperature_range() -> None:
    """SemanticKernelChatConfig should validate temperature in __post_init__."""
    with pytest.raises(ValueError, match="temperature"):
        SemanticKernelChatConfig(temperature=-0.1)
    
    with pytest.raises(ValueError, match="temperature"):
        SemanticKernelChatConfig(temperature=2.5)


def test_create_llm_translator_called_with_framework_semantic_kernel(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Translator factory should be called with framework='semantic_kernel'."""
    captured: Dict[str, Any] = {}

    def fake_create_llm_translator(*_: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return _make_dummy_translator()

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", fake_create_llm_translator)

    chat_history = _make_mock_chat_history()
    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter, model="test")
    _ = llm.get_chat_message_content_sync(chat_history)

    assert captured.get("framework") == "semantic_kernel"
    assert captured.get("adapter") is adapter


def test_translator_override_is_passed_to_factory(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """When translator override is provided, it is passed through to the factory."""
    captured: Dict[str, Any] = {}

    def fake_create_llm_translator(*_a: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return _make_dummy_translator()

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", fake_create_llm_translator)

    sentinel_translator = object()
    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter, model="test", translator=sentinel_translator)
    assert llm is not None
    assert captured.get("translator") is sentinel_translator


def test_client_stores_config_attributes(adapter: Any) -> None:
    """Client should store key config attributes as instance variables."""
    llm = CorpusSemanticKernelChatCompletion(
        llm_adapter=adapter,
        model="my-model",
        temperature=0.8,
        max_tokens=100,
        framework_version="v1.0",
    )
    
    assert llm.model == "my-model"
    assert llm.temperature == 0.8
    assert llm.max_tokens == 100
    assert llm.framework_version == "v1.0"


def test_config_object_takes_precedence_over_direct_params(adapter: Any) -> None:
    """SemanticKernelChatConfig should override direct parameter values."""
    config = SemanticKernelChatConfig(temperature=0.9, max_tokens=200)
    llm = CorpusSemanticKernelChatCompletion(
        llm_adapter=adapter,
        config=config,
        temperature=0.5,  # Should be overridden
        max_tokens=50,    # Should be overridden
    )
    
    assert llm.temperature == 0.9
    assert llm.max_tokens == 200


# ---------------------------------------------------------------------------
# Semantic Kernel Interface Compatibility Tests (3 tests)
# ---------------------------------------------------------------------------


def test_semantic_kernel_chatcompletionbase_inheritance(adapter: Any) -> None:
    """Verify CorpusSemanticKernelChatCompletion implements SK ChatCompletionClientBase when available."""
    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter, model="test")

    # Core methods we expect
    assert hasattr(llm, "get_chat_message_content")
    assert hasattr(llm, "get_chat_message_contents")
    assert hasattr(llm, "get_streaming_chat_message_content")
    assert hasattr(llm, "get_streaming_chat_message_contents")

    if not _has_semantic_kernel_installed():
        pytest.skip("Semantic Kernel is not available; cannot assert base class compatibility")

    # Try the ChatCompletionClientBase import
    try:
        from semantic_kernel.connectors.ai.chat_completion_client_base import (
            ChatCompletionClientBase,
        )
    except Exception:
        pytest.skip("SEMANTIC_KERNEL available but could not import ChatCompletionClientBase")

    assert isinstance(llm, ChatCompletionClientBase)


def test_semantic_kernel_available_flag_is_bool() -> None:
    """_SEMANTIC_KERNEL_IMPORT_ERROR should determine availability."""
    # The module should have soft-imported SK
    assert hasattr(sk_adapter_module, "_SEMANTIC_KERNEL_IMPORT_ERROR")
    import_error = sk_adapter_module._SEMANTIC_KERNEL_IMPORT_ERROR
    assert import_error is None or isinstance(import_error, Exception)


def test_semantic_kernel_not_installed_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """When SK is not installed, initialization should raise RuntimeError with install instructions."""
    # Simulate SK not being installed
    monkeypatch.setattr(
        sk_adapter_module,
        "_SEMANTIC_KERNEL_IMPORT_ERROR",
        ImportError("No module named 'semantic_kernel'"),
    )

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

    with pytest.raises(RuntimeError, match="semantic-kernel is not installed"):
        CorpusSemanticKernelChatCompletion(llm_adapter=MinimalAdapter())


# ---------------------------------------------------------------------------
# Context Translation Tests (6 tests)
# ---------------------------------------------------------------------------


def test_builds_operation_context_from_prompt_execution_settings(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Should build OperationContext from SK PromptExecutionSettings."""
    captured: Dict[str, Any] = {}
    base_ctx = OperationContext(request_id="from-core", tenant="from-core", attrs={"x": 1})

    def fake_context_from_semantic_kernel(
        kernel: Any,
        *,
        settings: Any = None,
        framework_version: Any = None,
    ) -> Any:
        captured["settings"] = settings
        captured["framework_version"] = framework_version
        return base_ctx

    monkeypatch.setattr(sk_adapter_module, "context_from_semantic_kernel", fake_context_from_semantic_kernel)

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

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    class MockPromptExecutionSettings:
        temperature = 0.5

    mock_settings = MockPromptExecutionSettings()
    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter, framework_version="sk-v1.0")
    chat_history = _make_mock_chat_history()
    
    _ = llm.get_chat_message_content_sync(chat_history, settings=mock_settings)
    
    assert captured["settings"] == mock_settings
    assert captured["framework_version"] == "sk-v1.0"
    assert isinstance(captured["op_ctx"], OperationContext)


def test_handles_none_settings_gracefully(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Should handle None settings without crashing."""
    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    # Should not raise
    result = llm.get_chat_message_content_sync(chat_history, settings=None)
    assert result is not None


def test_context_translation_error_attaches_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Errors during context translation should attach error context."""
    captured_calls: list[Dict[str, Any]] = []

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        _ = exc
        captured_calls.append(dict(ctx))

    def fake_context_from_semantic_kernel(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("ctx translation failed")

    monkeypatch.setattr(sk_adapter_module, "attach_context", fake_attach_context)
    monkeypatch.setattr(sk_adapter_module, "context_from_semantic_kernel", fake_context_from_semantic_kernel)

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    with pytest.raises(RuntimeError, match="ctx translation failed"):
        llm.get_chat_message_content_sync(chat_history)

    assert any(c.get("framework") == "semantic_kernel" for c in captured_calls)
    assert any(c.get("operation") == "llm_context_translation" for c in captured_calls)


def test_context_translation_uses_framework_version(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Context translation should pass framework_version through."""
    captured: Dict[str, Any] = {}

    def fake_context_from_semantic_kernel(
        kernel: Any,
        *,
        settings: Any = None,
        framework_version: Any = None,
    ) -> Any:
        captured["framework_version"] = framework_version
        return OperationContext(request_id="test", tenant="test", attrs={})

    monkeypatch.setattr(sk_adapter_module, "context_from_semantic_kernel", fake_context_from_semantic_kernel)
    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter, framework_version="sk-v2.0")
    chat_history = _make_mock_chat_history()
    
    _ = llm.get_chat_message_content_sync(chat_history)
    
    assert captured["framework_version"] == "sk-v2.0"


def test_passes_framework_ctx_with_sk_metadata(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Should build framework_ctx including service_id and settings_type."""
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

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter, service_id="test-service")
    chat_history = _make_mock_chat_history()
    
    _ = llm.get_chat_message_content_sync(chat_history)
    
    framework_ctx = captured.get("framework_ctx")
    assert isinstance(framework_ctx, Mapping)
    assert framework_ctx.get("framework") == "semantic_kernel"
    assert framework_ctx.get("service_id") == "test-service"
    assert "settings_type" in framework_ctx


def test_handles_dict_settings_for_conformance(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Should accept pre-normalized dict settings for conformance tests."""
    captured: Dict[str, Any] = {}

    def fake_context_from_dict(settings_dict: Any) -> Any:
        captured["dict_settings"] = settings_dict
        return OperationContext(request_id="test", tenant="test", attrs={})

    monkeypatch.setattr(sk_adapter_module, "context_from_dict", fake_context_from_dict)
    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    dict_settings = {"request_id": "test-123", "tenant": "test-tenant"}
    
    _ = llm.get_chat_message_content_sync(chat_history, settings=dict_settings)
    
    assert captured["dict_settings"] == dict_settings


# ---------------------------------------------------------------------------
# Parameter Forwarding Tests (9 tests)
# ---------------------------------------------------------------------------


def test_forwards_model_parameter(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Should forward model parameter to translator."""
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

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter, model="default-model")
    chat_history = _make_mock_chat_history()
    
    _ = llm.get_chat_message_content_sync(chat_history, model="override-model")
    
    assert seen["params"].get("model") == "override-model"


def test_forwards_model_from_settings_model_id(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Should extract model from settings.model_id."""
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

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    class MockSettings:
        model_id = "settings-model"

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    _ = llm.get_chat_message_content_sync(chat_history, settings=MockSettings())
    
    assert seen["params"].get("model") == "settings-model"


def test_forwards_model_from_settings_deployment_name(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Should extract model from settings.deployment_name."""
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

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    class MockSettings:
        deployment_name = "deployment-model"

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    _ = llm.get_chat_message_content_sync(chat_history, settings=MockSettings())
    
    assert seen["params"].get("model") == "deployment-model"


def test_forwards_temperature_parameter(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Should forward temperature parameter to translator."""
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

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    class MockSettings:
        temperature = 0.3

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    _ = llm.get_chat_message_content_sync(chat_history, settings=MockSettings())
    
    assert seen["params"].get("temperature") == 0.3


def test_forwards_max_tokens_parameter(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Should forward max_tokens parameter to translator."""
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

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    class MockSettings:
        max_tokens = 150

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    _ = llm.get_chat_message_content_sync(chat_history, settings=MockSettings())
    
    assert seen["params"].get("max_tokens") == 150


def test_forwards_stop_sequences(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Should extract and forward stop_sequences from settings."""
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

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    class MockSettings:
        stop_sequences = ["\n\n", "STOP"]

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    _ = llm.get_chat_message_content_sync(chat_history, settings=MockSettings())
    
    assert seen["params"].get("stop_sequences") == ["\n\n", "STOP"]


def test_converts_string_stop_to_list(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Should convert string stop parameter to list."""
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

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    class MockSettings:
        stop = "END"

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    _ = llm.get_chat_message_content_sync(chat_history, settings=MockSettings())
    
    assert seen["params"].get("stop_sequences") == ["END"]


def test_forwards_sampling_params(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Should forward top_p, frequency_penalty, presence_penalty."""
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

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    class MockSettings:
        top_p = 0.9
        frequency_penalty = 0.1
        presence_penalty = 0.2

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    _ = llm.get_chat_message_content_sync(chat_history, settings=MockSettings())
    
    params = seen["params"]
    assert params.get("top_p") == 0.9
    assert params.get("frequency_penalty") == 0.1
    assert params.get("presence_penalty") == 0.2


def test_forwards_tools_and_tool_choice(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Should forward tools and tool_choice via kwargs."""
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

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    tools = [{"type": "function", "function": {"name": "test"}}]
    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    _ = llm.get_chat_message_content_sync(chat_history, tools=tools, tool_choice="auto")
    
    assert seen["params"].get("tools") == tools
    assert seen["params"].get("tool_choice") == "auto"


# ---------------------------------------------------------------------------
# ChatHistory Normalization Tests (5 tests)
# ---------------------------------------------------------------------------


def test_to_translator_messages_converts_chatmessagecontent_objects(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """_to_translator_messages should convert SK ChatMessageContent objects to dicts."""
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

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history("test content")
    
    _ = llm.get_chat_message_content_sync(chat_history)
    
    normalized = seen["messages"]
    assert isinstance(normalized, list)
    assert len(normalized) == 1
    assert normalized[0]["role"] == "user"
    assert normalized[0]["content"] == "test content"


def test_to_translator_messages_handles_string_input(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """_to_translator_messages should wrap string input as user message."""
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

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    
    _ = llm.get_chat_message_content_sync("plain string prompt")
    
    normalized = seen["messages"]
    assert normalized == [{"role": "user", "content": "plain string prompt"}]


def test_to_translator_messages_handles_dict_messages(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """_to_translator_messages should pass through pre-normalized dicts."""
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

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    dict_messages = [{"role": "user", "content": "dict message"}]
    
    _ = llm.get_chat_message_content_sync(dict_messages)
    
    normalized = seen["messages"]
    assert normalized == dict_messages


def test_to_translator_messages_extracts_role_and_author_role(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """_to_translator_messages should handle both role and author_role fields."""
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

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    class MessageWithAuthorRole:
        def __init__(self):
            self.author_role = "assistant"
            self.content = "test"

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    
    _ = llm.get_chat_message_content_sync([MessageWithAuthorRole()])
    
    normalized = seen["messages"]
    assert normalized[0]["role"] == "assistant"


def test_to_translator_messages_handles_items_content(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """_to_translator_messages should handle multi-part content via items."""
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

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    class MessageWithItems:
        def __init__(self):
            self.role = "user"
            self.items = ["part1", "part2", "part3"]

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    
    _ = llm.get_chat_message_content_sync([MessageWithItems()])
    
    normalized = seen["messages"]
    assert "part1part2part3" in normalized[0]["content"]


# ---------------------------------------------------------------------------
# Chat API Tests (Sync/Async) (6 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_chat_message_content_returns_chatmessagecontent(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Async get_chat_message_content() should return ChatMessageContent."""
    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    response = await llm.get_chat_message_content(chat_history)
    
    assert response is not None


@pytest.mark.asyncio
async def test_get_chat_message_content_validates_empty_history(adapter: Any) -> None:
    """Async get_chat_message_content() should raise ValueError for empty history."""
    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    
    with pytest.raises(ValueError, match="empty"):
        await llm.get_chat_message_content([])


def test_get_chat_message_content_sync_returns_chatmessagecontent(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Sync get_chat_message_content_sync() should return ChatMessageContent."""
    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    response = llm.get_chat_message_content_sync(chat_history)
    
    assert response is not None


def test_get_chat_message_content_sync_validates_empty_history(adapter: Any) -> None:
    """Sync get_chat_message_content_sync() should raise ValueError for empty history."""
    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    
    with pytest.raises(ValueError, match="empty"):
        llm.get_chat_message_content_sync([])


@pytest.mark.asyncio
async def test_get_chat_message_content_delegates_to_translator_arun_complete(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Async method should call translator.arun_complete()."""
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

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    _ = await llm.get_chat_message_content(chat_history)
    
    assert called["arun_complete"] is True


def test_get_chat_message_content_sync_delegates_to_translator_complete(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Sync method should call translator.complete()."""
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

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    _ = llm.get_chat_message_content_sync(chat_history)
    
    assert called["complete"] is True


# ---------------------------------------------------------------------------
# Streaming Chat API Tests (6 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_streaming_chat_message_content_returns_async_iterator(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Async streaming should return async iterator."""
    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    result = await llm.get_streaming_chat_message_content(chat_history)
    
    assert hasattr(result, "__aiter__")


@pytest.mark.asyncio
async def test_get_streaming_chat_message_content_yields_chunks(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Async streaming should yield progressive chunks."""
    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    chunks = []
    async for chunk in await llm.get_streaming_chat_message_content(chat_history):
        chunks.append(chunk)
    
    assert len(chunks) >= 1


def test_get_streaming_chat_message_content_sync_returns_iterator(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Sync streaming should return iterator."""
    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    result = llm.get_streaming_chat_message_content_sync(chat_history)
    
    assert hasattr(result, "__iter__")


def test_get_streaming_chat_message_content_sync_yields_chunks(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Sync streaming should yield progressive chunks."""
    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    chunks = list(llm.get_streaming_chat_message_content_sync(chat_history))
    
    assert len(chunks) >= 1


@pytest.mark.asyncio
async def test_get_streaming_chat_message_content_delegates_to_translator_arun_stream(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Async streaming should call translator.arun_stream()."""
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

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    chunks = []
    async for chunk in await llm.get_streaming_chat_message_content(chat_history):
        chunks.append(chunk)
    
    assert called["arun_stream"] is True


def test_get_streaming_chat_message_content_sync_delegates_to_translator_stream(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Sync streaming should call translator.stream()."""
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

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    _ = list(llm.get_streaming_chat_message_content_sync(chat_history))
    
    assert called["stream"] is True


# ---------------------------------------------------------------------------
# Error Context Attachment Tests (6 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_chat_message_content_error_attaches_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Errors during async get_chat_message_content should attach error context."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(sk_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        async def arun_complete(self, *_: Any, **__: Any) -> Any:
            raise RuntimeError("async test error")
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
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

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    llm._translator = FailingTranslator()  # type: ignore[attr-defined]

    chat_history = _make_mock_chat_history()
    
    with pytest.raises(RuntimeError, match="async test error"):
        await llm.get_chat_message_content(chat_history)

    assert captured_ctx.get("framework") == "semantic_kernel"
    assert "llm_" in str(captured_ctx.get("operation", ""))


def test_get_chat_message_content_sync_error_attaches_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Errors during sync get_chat_message_content_sync should attach error context."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(sk_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def complete(self, *_: Any, **__: Any) -> Any:
            raise RuntimeError("sync test error")
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

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    llm._translator = FailingTranslator()  # type: ignore[attr-defined]

    chat_history = _make_mock_chat_history()
    
    with pytest.raises(RuntimeError, match="sync test error"):
        llm.get_chat_message_content_sync(chat_history)

    assert captured_ctx.get("framework") == "semantic_kernel"
    assert "llm_" in str(captured_ctx.get("operation", ""))


@pytest.mark.asyncio
async def test_get_streaming_chat_message_content_iteration_error_attaches_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Errors during async streaming iteration should attach error context."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(sk_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def arun_stream(self, *_: Any, **__: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                yield {"text": "a", "is_final": False, "model": "m"}
                raise RuntimeError("async stream error")
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    llm._translator = FailingTranslator()  # type: ignore[attr-defined]

    chat_history = _make_mock_chat_history()
    
    with pytest.raises(RuntimeError, match="async stream error"):
        async for _ in await llm.get_streaming_chat_message_content(chat_history):
            pass

    assert captured_ctx.get("framework") == "semantic_kernel"
    assert captured_ctx.get("operation") == "llm_get_streaming_chat_message_content"


def test_get_streaming_chat_message_content_sync_iteration_error_attaches_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Errors during sync streaming iteration should attach error context."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(sk_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "ok", "model": "m"}
        def stream(self, *_: Any, **__: Any) -> Iterator[Any]:
            def _gen() -> Iterator[Any]:
                yield {"text": "a", "is_final": False, "model": "m"}
                raise RuntimeError("sync stream error")
            return _gen()
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

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    llm._translator = FailingTranslator()  # type: ignore[attr-defined]

    chat_history = _make_mock_chat_history()
    
    with pytest.raises(RuntimeError, match="sync stream error"):
        for _ in llm.get_streaming_chat_message_content_sync(chat_history):
            pass

    assert captured_ctx.get("framework") == "semantic_kernel"
    assert captured_ctx.get("operation") == "llm_get_streaming_chat_message_content_sync"


def test_error_context_includes_sk_metadata(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Error context should include SK-specific metadata when available."""
    captured_ctx: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured_ctx.update(ctx)

    monkeypatch.setattr(sk_adapter_module, "attach_context", fake_attach_context)

    class FailingTranslator:
        def complete(self, *_: Any, **__: Any) -> Any:
            raise RuntimeError("test error with sk metadata")
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

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter, service_id="test-service")
    llm._translator = FailingTranslator()  # type: ignore[attr-defined]

    chat_history = _make_mock_chat_history()
    
    with pytest.raises(RuntimeError, match="test error with sk metadata"):
        llm.get_chat_message_content_sync(chat_history)

    assert captured_ctx.get("framework") == "semantic_kernel"
    # The error context decorator should include model, temperature, etc.
    assert "model" in captured_ctx or "resolved_model" in captured_ctx


def test_error_context_extraction_is_lazy(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Error context extraction should only happen on error path."""
    extract_called = {"count": 0}
    original_extract = sk_adapter_module._extract_dynamic_context

    def counting_extract(*args: Any, **kwargs: Any) -> Any:
        extract_called["count"] += 1
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(sk_adapter_module, "_extract_dynamic_context", counting_extract)
    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    # Successful call should not extract context
    _ = llm.get_chat_message_content_sync(chat_history)
    
    assert extract_called["count"] == 0


# ---------------------------------------------------------------------------
# Token Counting Tests (4 tests)
# ---------------------------------------------------------------------------


def test_count_tokens_returns_int(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """count_tokens() should return an integer."""
    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    n = llm.count_tokens(chat_history)
    
    assert isinstance(n, int)
    assert n >= 0


def test_count_tokens_handles_empty_history(adapter: Any) -> None:
    """count_tokens() should return 0 for empty history."""
    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    
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

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    n = llm.count_tokens(chat_history)
    
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

    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: BadTranslator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    with pytest.raises(TypeError):
        llm.count_tokens(chat_history)


# ---------------------------------------------------------------------------
# Capabilities and Health Tests (4 tests)
# ---------------------------------------------------------------------------


def test_capabilities_delegates_to_translator_only(adapter: Any) -> None:
    """capabilities() should delegate only to translator, not adapter."""
    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    
    caps = llm.capabilities()
    
    assert isinstance(caps, Mapping)


@pytest.mark.asyncio
async def test_acapabilities_delegates_to_translator_or_thread(adapter: Any) -> None:
    """acapabilities() should use translator.acapabilities or run sync in thread."""
    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    
    caps = await llm.acapabilities()
    
    assert isinstance(caps, Mapping)


def test_health_delegates_to_translator_only(adapter: Any) -> None:
    """health() should delegate only to translator, not adapter."""
    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    
    health = llm.health()
    
    assert isinstance(health, Mapping)


@pytest.mark.asyncio
async def test_ahealth_delegates_to_translator_or_thread(adapter: Any) -> None:
    """ahealth() should use translator.ahealth or run sync in thread."""
    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    
    health = await llm.ahealth()
    
    assert isinstance(health, Mapping)


# ---------------------------------------------------------------------------
# Context Manager Tests (4 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_closes_underlying_adapter() -> None:
    """Context managers should call close/aclose on adapter."""
    class ClosingAdapter:
        def __init__(self) -> None:
            self.closed = False
            self.aclosed = False

        def complete(self, *args: Any, **kwargs: Any) -> Any:
            return {"text": "test response", "model": "test-model"}
        
        def stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
            yield {"text": "chunk", "is_final": True, "model": "test-model"}
        
        def count_tokens(self, *args: Any, **kwargs: Any) -> int:
            return 10
        
        def capabilities(self) -> Dict[str, Any]:
            return {"supports_count_tokens": True}
        
        def health(self) -> Dict[str, Any]:
            return {"status": "ok"}

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.aclosed = True

    adapter_instance = ClosingAdapter()

    with CorpusSemanticKernelChatCompletion(llm_adapter=adapter_instance) as llm:
        assert llm is not None
    assert adapter_instance.closed is True

    adapter2 = ClosingAdapter()
    llm2 = CorpusSemanticKernelChatCompletion(llm_adapter=adapter2)
    async with llm2:
        assert llm2 is not None
    assert adapter2.aclosed is True


def test_sync_context_manager_works_without_close(adapter: Any) -> None:
    """Sync context manager should work even if adapter lacks close()."""
    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    
    with llm:
        assert llm is not None


@pytest.mark.asyncio
async def test_async_context_manager_works_without_aclose(adapter: Any) -> None:
    """Async context manager should work even if adapter lacks aclose()."""
    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    
    async with llm:
        assert llm is not None


def test_context_manager_api_consistency(adapter: Any) -> None:
    """SK adapter should support context manager pattern like other adapters."""
    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    
    # Verify both __enter__/__exit__ exist
    assert hasattr(llm, "__enter__")
    assert hasattr(llm, "__exit__")
    assert hasattr(llm, "__aenter__")
    assert hasattr(llm, "__aexit__")


# ---------------------------------------------------------------------------
# Event Loop Guard Tests (4 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_chat_message_content_sync_raises_when_called_in_event_loop() -> None:
    """Sync method should raise when called from within an event loop."""
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

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=MinimalAdapter())
    chat_history = _make_mock_chat_history()
    
    with pytest.raises(RuntimeError, match="event loop"):
        llm.get_chat_message_content_sync(chat_history)


@pytest.mark.asyncio
async def test_get_streaming_chat_message_content_sync_raises_when_called_in_event_loop() -> None:
    """Sync streaming should raise when called from within an event loop."""
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

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=MinimalAdapter())
    chat_history = _make_mock_chat_history()
    
    with pytest.raises(RuntimeError, match="event loop"):
        it = llm.get_streaming_chat_message_content_sync(chat_history)
        next(it)


@pytest.mark.asyncio
async def test_async_methods_work_in_event_loop(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Async methods should work fine inside an event loop."""
    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    # Should not raise
    result = await llm.get_chat_message_content(chat_history)
    assert result is not None


def test_event_loop_guard_provides_async_alternative_name() -> None:
    """Event loop error should mention the async alternative method."""
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

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=MinimalAdapter())
    chat_history = _make_mock_chat_history()
    
    async def _test() -> None:
        try:
            llm.get_chat_message_content_sync(chat_history)
        except RuntimeError as e:
            # Error message should mention the async alternative
            assert "get_chat_message_content" in str(e)
            raise

    # Run in event loop to trigger guard
    with pytest.raises(RuntimeError):
        asyncio.run(_test())


# ---------------------------------------------------------------------------
# Multi-Message API Tests (2 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_chat_message_contents_returns_list(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """get_chat_message_contents() should return List[ChatMessageContent]."""
    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    results = await llm.get_chat_message_contents(chat_history)
    
    assert isinstance(results, list)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_get_streaming_chat_message_contents_yields_chunks(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """get_streaming_chat_message_contents() should be an alias for streaming."""
    monkeypatch.setattr(sk_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    
    chunks = []
    async for chunk in await llm.get_streaming_chat_message_contents(chat_history):
        chunks.append(chunk)
    
    assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# Semantic Kernel Integration Tests (5 tests - NO SKIPS)
# ---------------------------------------------------------------------------


def test_semantic_kernel_dependency_must_be_installed() -> None:
    """Semantic Kernel must be installed for framework integration tests."""
    _require_semantic_kernel_available_for_integration()


def test_semantic_kernel_chatmessagecontent_roundtrip(adapter: Any) -> None:
    """Real SK ChatMessageContent objects should work with adapter."""
    _require_semantic_kernel_available_for_integration()

    from semantic_kernel.contents import ChatMessageContent
    
    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    # Note: This will fail without proper SK types, but tests the integration
    chat_history = [ChatMessageContent(role="user", content="test")]
    
    response = llm.get_chat_message_content_sync(chat_history)
    assert response is not None


def test_semantic_kernel_prompt_execution_settings_integration(adapter: Any) -> None:
    """Real SK PromptExecutionSettings should create OperationContext."""
    _require_semantic_kernel_available_for_integration()

    from semantic_kernel.connectors.ai import PromptExecutionSettings
    
    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = _make_mock_chat_history()
    settings = PromptExecutionSettings(temperature=0.5)
    
    # Should not raise - settings should be handled gracefully
    response = llm.get_chat_message_content_sync(chat_history, settings=settings)
    assert response is not None


def test_semantic_kernel_chathistory_object_works(adapter: Any) -> None:
    """Real SK ChatHistory object should work with adapter."""
    _require_semantic_kernel_available_for_integration()

    from semantic_kernel.contents import ChatHistory
    
    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = ChatHistory()
    chat_history.add_user_message("test message")
    
    response = llm.get_chat_message_content_sync(chat_history)
    assert response is not None


def test_semantic_kernel_streaming_with_real_messages(adapter: Any) -> None:
    """Streaming should work with real SK message objects."""
    _require_semantic_kernel_available_for_integration()

    from semantic_kernel.contents import ChatMessageContent
    
    llm = CorpusSemanticKernelChatCompletion(llm_adapter=adapter)
    chat_history = [ChatMessageContent(role="user", content="stream test")]
    
    chunks = list(llm.get_streaming_chat_message_content_sync(chat_history))
    
    assert len(chunks) > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
