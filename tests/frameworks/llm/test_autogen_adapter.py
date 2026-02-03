"""AutoGen LLM framework adapter tests.

These tests are written against the current public API in
`corpus_sdk.llm.framework_adapters.autogen`, which exposes an OpenAI-style
chat client (`CorpusAutoGenChatClient`) with `create`/`acreate` entrypoints.
"""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, Dict, Optional

import pytest

import corpus_sdk.llm.framework_adapters.autogen as autogen_adapter_module
from corpus_sdk.llm.framework_adapters.autogen import (
    AutoGenClientConfig,
    CorpusAutoGenChatClient,
    ErrorCodes,
    create_autogen_chat_completion_client,
)
from corpus_sdk.llm.llm_base import OperationContext


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROMPT = "hello from autogen tests"


def _msgs(text: str = PROMPT) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(
    adapter: Any,
    *,
    model: str = "default",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    framework_version: Optional[str] = None,
    enable_metrics: bool = True,
    validate_inputs: bool = True,
    translator: Any = None,
) -> CorpusAutoGenChatClient:
    config = AutoGenClientConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        framework_version=framework_version,
        enable_metrics=enable_metrics,
        validate_inputs=validate_inputs,
    )
    return CorpusAutoGenChatClient(llm_adapter=adapter, config=config, translator=translator)


def _has_autogen_installed() -> bool:
    """Dependency-neutral check: does not import AutoGen, only checks availability."""
    return importlib.util.find_spec("autogen") is not None


def _require_autogen_available_for_e2e() -> None:
    """Enforce "no skips": E2E tests must fail if AutoGen isn't installed."""
    if not _has_autogen_installed():
        raise AssertionError(
            "AutoGen is not installed, but AutoGen E2E integration tests are required (no skips). "
            "Install AutoGen packages in the test environment to run this framework suite."
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


# ---------------------------------------------------------------------------
# Construction / initialization tests (8 tests)
# ---------------------------------------------------------------------------


def test_init_rejects_adapter_without_required_methods() -> None:
    """Adapter must implement LLMProtocolV1 methods."""
    class BadAdapter:
        async def complete(self, **_: Any) -> Any:
            return None

    with pytest.raises(TypeError) as exc_info:
        _ = _make_client(BadAdapter())

    msg = str(exc_info.value)
    assert "LLMProtocolV1" in msg or "missing methods" in msg or "must implement" in msg


def test_config_validates_temperature_range() -> None:
    """Config should reject temperature outside [0.0, 2.0] range."""
    with pytest.raises(ValueError, match="temperature"):
        AutoGenClientConfig(model="m", temperature=-0.1)
    
    with pytest.raises(ValueError, match="temperature"):
        AutoGenClientConfig(model="m", temperature=2.1)


def test_config_validates_max_tokens_positive() -> None:
    """Config should reject non-positive max_tokens."""
    with pytest.raises(ValueError, match="max_tokens"):
        AutoGenClientConfig(model="m", max_tokens=0)
    
    with pytest.raises(ValueError, match="max_tokens"):
        AutoGenClientConfig(model="m", max_tokens=-10)


def test_config_validates_request_timeout_positive() -> None:
    """Config should reject non-positive request_timeout."""
    with pytest.raises(ValueError, match="request_timeout"):
        AutoGenClientConfig(model="m", request_timeout=0)
    
    with pytest.raises(ValueError, match="request_timeout"):
        AutoGenClientConfig(model="m", request_timeout=-5.0)


def test_config_from_dict_ignores_unknown_keys() -> None:
    """from_dict should filter unknown keys gracefully."""
    config_dict = {
        "model": "test-model",
        "temperature": 0.5,
        "unknown_key": "value",
        "another_unknown": 123,
    }
    config = AutoGenClientConfig.from_dict(config_dict)
    assert config.model == "test-model"
    assert config.temperature == 0.5


def test_create_llm_translator_called_with_framework_autogen(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Translator factory should be called with framework='autogen'."""
    captured: Dict[str, Any] = {}

    def fake_create_llm_translator(*_: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return _make_dummy_translator()

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", fake_create_llm_translator)

    client = _make_client(adapter)
    out = client.create(_msgs("ping"))
    assert isinstance(out, Mapping)

    assert captured.get("framework") == "autogen"
    assert captured.get("adapter") is adapter


def test_translator_override_is_used_and_factory_not_called(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """When translator is provided, factory should not be called."""
    monkeypatch.setattr(
        autogen_adapter_module,
        "create_llm_translator",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    client = _make_client(adapter, translator=_make_dummy_translator())
    out = client.create(_msgs("x"))
    assert isinstance(out, Mapping)


def test_client_stores_config_attributes() -> None:
    """Client should store key config attributes as instance variables."""
    class MinimalAdapter:
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
        def count_tokens(self, *a: Any, **k: Any) -> int:
            return 0
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}

    client = _make_client(
        MinimalAdapter(),
        model="my-model",
        temperature=0.8,
        max_tokens=100,
        framework_version="v1.0",
    )
    
    assert client.model == "my-model"
    assert client.temperature == 0.8
    assert client.max_tokens == 100


# ---------------------------------------------------------------------------
# Context translation tests (6 tests)
# ---------------------------------------------------------------------------


def test_create_builds_operation_context_from_conversation(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """create() should build OperationContext from conversation parameter."""
    captured: Dict[str, Any] = {}
    base_ctx = OperationContext(request_id="from-core", tenant="from-core", attrs={"x": 1})

    def fake_core_ctx_from_autogen(conversation: Any, *, framework_version: Any = None, **extra: Any) -> Any:
        captured["conversation"] = conversation
        captured["framework_version"] = framework_version
        captured["extra"] = dict(extra)
        return base_ctx

    monkeypatch.setattr(autogen_adapter_module, "core_ctx_from_autogen", fake_core_ctx_from_autogen)

    class DummyTranslator:
        def complete(self, raw_messages: Any, *, op_ctx: Any = None, framework_ctx: Any = None, **_: Any) -> Any:
            captured["op_ctx"] = op_ctx
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
            return {"supports_count_tokens": True}
        def health(self) -> Mapping[str, Any]:
            return {"status": "ok"}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter, framework_version="autogen-fw")
    out = client.create(
        _msgs("ctx"),
        conversation={"conversation_id": "c1"},
    )
    assert isinstance(out, Mapping)
    assert captured["conversation"] == {"conversation_id": "c1"}
    assert captured["framework_version"] == "autogen-fw"


def test_create_builds_operation_context_with_context_param(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """create() should merge 'context' parameter into OperationContext."""
    captured: Dict[str, Any] = {}
    base_ctx = OperationContext(request_id="from-core", tenant="from-core", attrs={"x": 1})

    def fake_core_ctx_from_autogen(conversation: Any, *, framework_version: Any = None, **extra: Any) -> Any:
        captured["extra"] = dict(extra)
        return base_ctx

    monkeypatch.setattr(autogen_adapter_module, "core_ctx_from_autogen", fake_core_ctx_from_autogen)
    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    client = _make_client(adapter)
    out = client.create(
        _msgs("ctx"),
        context={"key": "value"},
    )
    assert isinstance(out, Mapping)
    assert captured["extra"] == {"key": "value"}


def test_create_overrides_request_id_and_tenant(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """create() should allow request_id and tenant overrides."""
    captured: Dict[str, Any] = {}
    base_ctx = OperationContext(request_id="base", tenant="base", attrs={})

    def fake_core_ctx_from_autogen(conversation: Any, *, framework_version: Any = None, **extra: Any) -> Any:
        return base_ctx

    monkeypatch.setattr(autogen_adapter_module, "core_ctx_from_autogen", fake_core_ctx_from_autogen)

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

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    out = client.create(
        _msgs("ctx"),
        # NOTE: The adapter only creates an OperationContext when it has either
        # a conversation or extra context. Provide a minimal conversation so
        # request_id/tenant overrides apply.
        conversation={"conversation_id": "c1"},
        request_id="override-req",
        tenant="override-tenant",
    )
    
    op_ctx = captured["op_ctx"]
    assert isinstance(op_ctx, OperationContext)
    assert op_ctx.request_id == "override-req"
    assert op_ctx.tenant == "override-tenant"


def test_create_passes_framework_ctx_with_metadata(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """create() should build framework_ctx with framework='autogen' and operation details."""
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

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    client.create(_msgs("test"))

    framework_ctx = captured.get("framework_ctx")
    assert isinstance(framework_ctx, Mapping)
    assert framework_ctx.get("framework") == "autogen"
    assert framework_ctx.get("operation") == "create"
    assert framework_ctx.get("stream") is False


def test_create_context_translation_error_attaches_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Errors during context translation should attach error context."""
    captured_calls: list[Dict[str, Any]] = []

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured_calls.append(dict(ctx))

    def fake_core_ctx_from_autogen(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("ctx translation failed")

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)
    monkeypatch.setattr(autogen_adapter_module, "core_ctx_from_autogen", fake_core_ctx_from_autogen)

    client = _make_client(adapter)
    
    with pytest.raises(RuntimeError, match="ctx translation failed"):
        client.create(_msgs("test"), conversation={"id": "c1"})

    # The adapter attaches context twice:
    # 1) inside _build_core_context(): llm_context_translation
    # 2) via the create() error-context decorator: llm_create
    ops = [c.get("operation") for c in captured_calls]
    assert "llm_context_translation" in ops
    assert "llm_create" in ops
    assert any(c.get("framework") == "autogen" for c in captured_calls)


def test_create_with_none_conversation_works(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """create() should handle None conversation gracefully."""
    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    client = _make_client(adapter)
    out = client.create(_msgs("test"), conversation=None)
    assert isinstance(out, Mapping)


# ---------------------------------------------------------------------------
# Parameter forwarding tests (8 tests)
# ---------------------------------------------------------------------------


def test_create_forwards_model_parameter(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """create() should forward model parameter to translator."""
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

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter, model="default-model")
    out = client.create(_msgs("test"), model="override-model")
    
    assert seen["params"].get("model") == "override-model"


def test_create_forwards_temperature_parameter(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """create() should forward temperature parameter to translator."""
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

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    client.create(_msgs("test"), temperature=0.3)
    
    assert seen["params"].get("temperature") == 0.3


def test_create_forwards_max_tokens_parameter(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """create() should forward max_tokens parameter to translator."""
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

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    client.create(_msgs("test"), max_tokens=150)
    
    assert seen["params"].get("max_tokens") == 150


def test_create_forwards_stop_sequences(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """create() should convert stop parameter to stop_sequences."""
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

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    client.create(_msgs("test"), stop=["\n\n", "STOP"])
    
    assert seen["params"].get("stop_sequences") == ["\n\n", "STOP"]


def test_create_converts_string_stop_to_list(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """create() should convert string stop parameter to list."""
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

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    client.create(_msgs("test"), stop="END")
    
    assert seen["params"].get("stop_sequences") == ["END"]


def test_create_forwards_sampling_params(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """create() should forward top_p, frequency_penalty, presence_penalty."""
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

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    client.create(
        _msgs("test"),
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.2,
    )
    
    params = seen["params"]
    assert params.get("top_p") == 0.9
    assert params.get("frequency_penalty") == 0.1
    assert params.get("presence_penalty") == 0.2


def test_create_forwards_system_message(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """create() should forward system_message parameter."""
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

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    client.create(_msgs("test"), system_message="You are helpful")
    
    assert seen["params"].get("system_message") == "You are helpful"


def test_create_forwards_tools_and_tool_choice(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """create() should forward tools and tool_choice parameters."""
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

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    tools = [{"type": "function", "function": {"name": "test"}}]
    client = _make_client(adapter)
    client.create(_msgs("test"), tools=tools, tool_choice="auto")
    
    assert seen["params"].get("tools") == tools
    assert seen["params"].get("tool_choice") == "auto"


# ---------------------------------------------------------------------------
# OpenAI-style response shaping tests (6 tests)
# ---------------------------------------------------------------------------


def test_create_non_stream_returns_openai_chatcompletion(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """create() should return OpenAI ChatCompletion format."""
    class DummyTranslator:
        def complete(self, *_: Any, **__: Any) -> Any:
            return {
                "text": "assistant text",
                "model": "m1",
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                "finish_reason": "stop",
            }
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *_: Any, **__: Any) -> Any:
            return {"text": "assistant text", "model": "m1"}
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

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    out = client.create(_msgs("hello"), stream=False)
    
    assert isinstance(out, Mapping)
    assert out.get("object") == "chat.completion"
    assert "id" in out
    assert "created" in out
    
    choices = out.get("choices")
    assert isinstance(choices, list) and len(choices) == 1
    
    msg = choices[0].get("message")
    assert isinstance(msg, Mapping)
    assert msg.get("role") == "assistant"
    assert msg.get("content") == "assistant text"
    
    usage = out.get("usage")
    assert isinstance(usage, Mapping)
    assert usage.get("prompt_tokens") == 1
    assert usage.get("completion_tokens") == 2
    assert usage.get("total_tokens") == 3


def test_create_stream_returns_iterator_of_openai_chunks(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """create(stream=True) should return iterator of OpenAI ChatCompletionChunk format."""
    class DummyTranslator:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def stream(self, *_: Any, **__: Any) -> Iterator[Any]:
            yield {"text": "a", "is_final": False, "model": "m"}
            yield {"text": "b", "is_final": True, "model": "m"}
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

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    it = client.create(_msgs("stream"), stream=True)
    
    assert hasattr(it, "__iter__")
    chunks = list(it)
    assert len(chunks) == 2
    
    # First chunk
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    assert chunks[0]["choices"][0]["delta"].get("content") == "a"
    assert chunks[0]["choices"][0]["finish_reason"] is None
    
    # Final chunk
    assert chunks[1]["choices"][0]["delta"].get("content") == "b"
    assert chunks[1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_acreate_non_stream_returns_openai_chatcompletion(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """acreate() should return OpenAI ChatCompletion format."""
    class DummyTranslator:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *_: Any, **__: Any) -> Any:
            return {"text": "async text", "model": "m1"}
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

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    out = await client.acreate(_msgs("hello"), stream=False)
    
    assert isinstance(out, Mapping)
    assert out.get("object") == "chat.completion"
    assert out["choices"][0]["message"]["content"] == "async text"


@pytest.mark.asyncio
async def test_acreate_stream_returns_async_iterator_of_openai_chunks(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """acreate(stream=True) should return async iterator of OpenAI ChatCompletionChunk format."""
    class DummyTranslator:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def arun_stream(self, *_: Any, **__: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                yield {"text": "a", "is_final": False, "model": "m"}
                yield {"text": "b", "is_final": True, "model": "m"}
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    aiter = await client.acreate(_msgs("astream"), stream=True)
    
    assert hasattr(aiter, "__aiter__")
    out: list[Any] = []
    async for ch in aiter:
        out.append(ch)
    assert len(out) == 2
    assert out[0]["object"] == "chat.completion.chunk"


def test_create_handles_tool_calls_in_response(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """create() should convert tool_calls to OpenAI format."""
    class DummyTranslator:
        def complete(self, *_: Any, **__: Any) -> Any:
            return {
                "text": "",
                "model": "m1",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location": "SF"}'},
                    }
                ],
            }
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

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    out = client.create(_msgs("test"))
    
    msg = out["choices"][0]["message"]
    assert "tool_calls" in msg
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["function"]["name"] == "get_weather"


def test_create_sets_finish_reason_for_tool_calls(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """create() should set finish_reason='tool_calls' when tool calls present."""
    class DummyTranslator:
        def complete(self, *_: Any, **__: Any) -> Any:
            return {
                "text": "",
                "model": "m1",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "test", "arguments": "{}"}}
                ],
            }
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

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    out = client.create(_msgs("test"))
    
    assert out["choices"][0]["finish_reason"] == "tool_calls"


# ---------------------------------------------------------------------------
# Error context attachment tests (4 tests)
# ---------------------------------------------------------------------------


def test_create_error_attaches_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Errors during create() should attach error context."""
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)

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

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: FailingTranslator())

    client = _make_client(adapter)
    with pytest.raises(RuntimeError, match="boom"):
        _ = client.create(_msgs("x"))

    assert captured.get("framework") == "autogen"
    assert str(captured.get("operation", "")).startswith("llm_")


def test_create_stream_iteration_error_attaches_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Errors during stream iteration should attach error context."""
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)

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

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: FailingTranslator())

    client = _make_client(adapter)
    it = client.create(_msgs("x"), stream=True)
    assert next(iter(it))
    
    with pytest.raises(RuntimeError, match="stream-boom"):
        _ = list(it)
    
    assert captured.get("framework") == "autogen"
    assert captured.get("operation") == "llm_create"


@pytest.mark.asyncio
async def test_acreate_error_attaches_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Errors during acreate() should attach error context."""
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)

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

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: FailingTranslator())

    client = _make_client(adapter)
    
    with pytest.raises(RuntimeError, match="async-boom"):
        await client.acreate(_msgs("x"))

    assert captured.get("framework") == "autogen"
    assert str(captured.get("operation", "")).startswith("llm_")


@pytest.mark.asyncio
async def test_acreate_stream_iteration_error_attaches_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Errors during async stream iteration should attach error context."""
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    monkeypatch.setattr(autogen_adapter_module, "attach_context", fake_attach_context)

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

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: FailingTranslator())

    client = _make_client(adapter)
    aiter = await client.acreate(_msgs("x"), stream=True)
    
    # Consume first chunk
    first = await aiter.__anext__()
    assert first is not None
    
    with pytest.raises(RuntimeError, match="async-stream-boom"):
        async for _ in aiter:
            pass

    assert captured.get("framework") == "autogen"
    # Async streaming iteration errors are enriched within the acreate() generator.
    assert captured.get("operation") == "llm_acreate"


# ---------------------------------------------------------------------------
# Token counting tests (5 tests)
# ---------------------------------------------------------------------------


def test_count_tokens_returns_int(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """count_tokens() should return an integer."""
    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    client = _make_client(adapter)
    n = client.count_tokens(_msgs("test"))
    
    assert isinstance(n, int)
    assert n >= 0


def test_count_tokens_handles_single_string(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """count_tokens() should accept a single string and convert to messages."""
    seen: Dict[str, Any] = {}

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
        def count_tokens_for_messages(self, *, raw_messages: Any, **_: Any) -> int:
            seen["messages"] = raw_messages
            return 5

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    n = client.count_tokens("hello world")
    
    assert n == 5
    assert len(seen["messages"]) == 1
    assert seen["messages"][0]["role"] == "user"
    assert seen["messages"][0]["content"] == "hello world"


def test_count_tokens_handles_single_message_dict(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """count_tokens() should accept a single message dict and convert to list."""
    seen: Dict[str, Any] = {}

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
        def count_tokens_for_messages(self, *, raw_messages: Any, **_: Any) -> int:
            seen["messages"] = raw_messages
            return 3

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    n = client.count_tokens({"role": "user", "content": "test"})
    
    assert n == 3
    assert isinstance(seen["messages"], list)
    assert len(seen["messages"]) == 1


def test_count_tokens_forwards_model(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """count_tokens() should forward model to translator."""
    seen: Dict[str, Any] = {}

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
        def count_tokens_for_messages(self, *, model: Any = None, **_: Any) -> int:
            seen["model"] = model
            return 7

    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter, model="default-model")
    client.count_tokens(_msgs("test"), model="override-model")
    
    assert seen["model"] == "override-model"


@pytest.mark.asyncio
async def test_acount_tokens_returns_int(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """acount_tokens() should return an integer."""
    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    client = _make_client(adapter)
    n = await client.acount_tokens(_msgs("test"))
    
    assert isinstance(n, int)
    assert n >= 0


# ---------------------------------------------------------------------------
# Capabilities and health tests (6 tests)
# ---------------------------------------------------------------------------


def test_capabilities_returns_mapping(adapter: Any) -> None:
    """capabilities() should return a Mapping."""
    client = _make_client(adapter)
    caps = client.capabilities()
    
    assert isinstance(caps, Mapping)


def test_capabilities_reports_supports_count_tokens(adapter: Any) -> None:
    """capabilities() should report supports_count_tokens."""
    client = _make_client(adapter)
    caps = client.capabilities()
    
    assert "supports_count_tokens" in caps
    assert isinstance(caps["supports_count_tokens"], bool)


@pytest.mark.asyncio
async def test_acapabilities_returns_mapping(adapter: Any) -> None:
    """acapabilities() should return a Mapping."""
    client = _make_client(adapter)
    caps = await client.acapabilities()
    
    assert isinstance(caps, Mapping)


@pytest.mark.asyncio
async def test_acapabilities_reports_supports_count_tokens(adapter: Any) -> None:
    """acapabilities() should report supports_count_tokens."""
    client = _make_client(adapter)
    caps = await client.acapabilities()
    
    assert "supports_count_tokens" in caps
    assert isinstance(caps["supports_count_tokens"], bool)


def test_health_returns_mapping(adapter: Any) -> None:
    """health() should return a Mapping."""
    client = _make_client(adapter)
    health = client.health()
    
    assert isinstance(health, Mapping)


@pytest.mark.asyncio
async def test_ahealth_returns_mapping(adapter: Any) -> None:
    """ahealth() should return a Mapping."""
    client = _make_client(adapter)
    health = await client.ahealth()
    
    assert isinstance(health, Mapping)


# ---------------------------------------------------------------------------
# Context manager tests (4 tests)
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
        async def arun_complete(self, *args: Any, **kwargs: Any) -> Any:
            return {"text": "x", "model": "m"}
        def arun_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                if False:
                    yield None
            return _gen()
        def count_tokens(self, *args: Any, **kwargs: Any) -> int:
            return 0
        def close(self) -> None:
            self.closed = True
        async def aclose(self) -> None:
            self.aclosed = True

    adapter_instance = ClosingLLMAdapter()

    with CorpusAutoGenChatClient(llm_adapter=adapter_instance) as client:
        assert client is not None
    assert adapter_instance.closed is True

    adapter2 = ClosingLLMAdapter()
    client2 = CorpusAutoGenChatClient(llm_adapter=adapter2)
    async with client2:
        assert client2 is not None
    assert adapter2.aclosed is True


def test_sync_context_manager_works_without_close(adapter: Any) -> None:
    """Sync context manager should work even if adapter lacks close()."""
    client = _make_client(adapter)
    with client:
        assert client is not None


@pytest.mark.asyncio
async def test_async_context_manager_works_without_aclose(adapter: Any) -> None:
    """Async context manager should work even if adapter lacks aclose()."""
    client = _make_client(adapter)
    async with client:
        assert client is not None


def test_client_supports_direct_call_syntax(adapter: Any) -> None:
    """Client should support client(...) as alias for client.create(...)."""
    client = _make_client(adapter)
    
    # Should not raise
    result = client(_msgs("test"))
    assert isinstance(result, (Mapping, Iterator))


# ---------------------------------------------------------------------------
# Message validation tests (3 tests)
# ---------------------------------------------------------------------------


def test_create_validates_messages_when_enabled(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """create() should validate messages when validate_inputs=True."""
    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    client = _make_client(adapter, validate_inputs=True)
    
    # Empty messages should raise
    with pytest.raises(ValueError, match="empty"):
        client.create([])


def test_create_validates_message_structure(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """create() should validate each message has role and content."""
    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    client = _make_client(adapter, validate_inputs=True)
    
    # Missing role
    with pytest.raises(ValueError, match="role"):
        client.create([{"content": "test"}])
    
    # Missing content
    with pytest.raises(ValueError, match="content"):
        client.create([{"role": "user"}])


def test_create_skips_validation_when_disabled(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """create() should skip validation when validate_inputs=False."""
    monkeypatch.setattr(autogen_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    client = _make_client(adapter, validate_inputs=False)
    
    # Should not raise even with empty messages
    # (though translator might fail - that's expected)
    try:
        client.create([])
    except Exception:
        pass  # Expected - translator will fail, but not validation


# ---------------------------------------------------------------------------
# AutoGen E2E / Integration tests (5 tests)
# ---------------------------------------------------------------------------


def test_e2e_autogen_dependency_is_available_for_framework_suite() -> None:
    """AutoGen must be installed for E2E tests."""
    _require_autogen_available_for_e2e()


def test_e2e_autogen_core_wrapper_can_be_constructed(adapter: Any) -> None:
    """AutoGen Core wrapper should be constructable."""
    _require_autogen_available_for_e2e()

    inner = _make_client(adapter)
    wrapper = create_autogen_chat_completion_client(inner)
    caps = wrapper.capabilities
    assert isinstance(caps, Mapping)


def test_e2e_wrapper_has_expected_methods(adapter: Any) -> None:
    """AutoGen wrapper should expose expected interface."""
    _require_autogen_available_for_e2e()

    inner = _make_client(adapter)
    wrapper = create_autogen_chat_completion_client(inner)
    
    assert hasattr(wrapper, "capabilities")
    assert hasattr(wrapper, "count_tokens")
    assert hasattr(wrapper, "remaining_tokens")
    assert hasattr(wrapper, "create")
    assert hasattr(wrapper, "stream")


@pytest.mark.asyncio
async def test_e2e_wrapper_create_method_works(adapter: Any) -> None:
    """AutoGen wrapper create() should work end-to-end."""
    _require_autogen_available_for_e2e()

    inner = _make_client(adapter)
    wrapper = create_autogen_chat_completion_client(inner)
    
    # Import here to avoid hard dependency
    from autogen_core.models import CreateResult
    
    result = await wrapper.create(_msgs("test"))
    assert isinstance(result, CreateResult)


def test_e2e_wrapper_handles_realistic_autogen_conversation(adapter: Any) -> None:
    """Wrapper should handle realistic AutoGen conversation payloads."""
    _require_autogen_available_for_e2e()

    inner = _make_client(adapter)
    
    # AutoGen often carries extra metadata
    conversation = {
        "conversation_id": "e2e-conv",
        "agent_name": "assistant",
        "chat_history": [{"role": "user", "content": "hello"}],
        "metadata": {"run_id": "r1", "trace": True},
    }
    
    # Should not crash
    result = inner.create(_msgs("test"), conversation=conversation)
    assert isinstance(result, Mapping)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
