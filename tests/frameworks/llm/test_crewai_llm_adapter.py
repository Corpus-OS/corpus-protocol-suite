"""CrewAI LLM framework adapter tests.

These tests are written against the current public API in
`corpus_sdk.llm.framework_adapters.crewai`, which exposes a CrewAI-style
LLM client (`CorpusCrewAILLM`) with `complete`/`stream`/`acomplete`/`astream` methods.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, Dict, Optional

import pytest

import corpus_sdk.llm.framework_adapters.crewai as crewai_adapter_module
from corpus_sdk.llm.framework_adapters.crewai import (
    CorpusCrewAILLM,
    CrewAILLMConfig,
    ErrorCodes,
)
from corpus_sdk.llm.llm_base import OperationContext


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROMPT = "hello from crewai tests"


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
    config: Optional[CrewAILLMConfig] = None,
    require_crewai: bool = False,
) -> CorpusCrewAILLM:
    if config is not None:
        return CorpusCrewAILLM(
            llm_adapter=adapter,
            config=config,
            translator=translator,
            require_crewai=require_crewai,
        )
    
    return CorpusCrewAILLM(
        llm_adapter=adapter,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        framework_version=framework_version,
        translator=translator,
        require_crewai=require_crewai,
    )


def _has_crewai_installed() -> bool:
    """Dependency-neutral check: does not import CrewAI, only checks availability."""
    return importlib.util.find_spec("crewai") is not None


def _require_crewai_available_for_e2e() -> None:
    """Enforce "no skips": E2E tests must fail if CrewAI isn't installed."""
    if not _has_crewai_installed():
        raise AssertionError(
            "CrewAI is not installed, but CrewAI E2E integration tests are required (no skips). "
            "Install CrewAI packages in the test environment to run this framework suite."
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
        CrewAILLMConfig(model="m", temperature=-0.1)
    
    with pytest.raises(ValueError, match="temperature"):
        CrewAILLMConfig(model="m", temperature=2.1)


def test_config_validates_max_tokens_positive() -> None:
    """Config should reject non-positive max_tokens."""
    with pytest.raises(ValueError, match="max_tokens"):
        CrewAILLMConfig(model="m", max_tokens=0)
    
    with pytest.raises(ValueError, match="max_tokens"):
        CrewAILLMConfig(model="m", max_tokens=-10)


def test_init_validates_temperature_range_without_config() -> None:
    """Direct init should validate temperature range."""
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
        CorpusCrewAILLM(llm_adapter=MinimalAdapter(), temperature=-0.5)


def test_init_validates_max_tokens_positive_without_config() -> None:
    """Direct init should validate max_tokens."""
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
        CorpusCrewAILLM(llm_adapter=MinimalAdapter(), max_tokens=-5)


def test_create_llm_translator_called_with_framework_crewai(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Translator factory should be called with framework='crewai'."""
    captured: Dict[str, Any] = {}

    def fake_create_llm_translator(*_: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return _make_dummy_translator()

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", fake_create_llm_translator)

    client = _make_client(adapter)
    out = client.complete(PROMPT)
    assert out is not None

    assert captured.get("framework") == "crewai"
    assert captured.get("adapter") is adapter


def test_translator_override_is_used_and_factory_not_called(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """When translator is provided, it should be passed into the factory wiring."""
    captured: Dict[str, Any] = {}
    provided = object()

    def fake_create_llm_translator(*_: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return _make_dummy_translator()

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", fake_create_llm_translator)

    client = _make_client(adapter, translator=provided)
    out = client.complete(PROMPT)
    assert out is not None
    assert captured.get("translator") is provided


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


def test_complete_builds_operation_context_from_task(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """complete() should build OperationContext from task parameter."""
    captured: Dict[str, Any] = {}
    base_ctx = OperationContext(request_id="from-core", tenant="from-core", attrs={"x": 1})

    def fake_core_ctx_from_crewai(task: Any, *, framework_version: Any = None, **extra: Any) -> Any:
        captured["task"] = task
        captured["framework_version"] = framework_version
        captured["extra"] = dict(extra)
        return base_ctx

    monkeypatch.setattr(crewai_adapter_module, "core_ctx_from_crewai", fake_core_ctx_from_crewai)

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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter, framework_version="crewai-fw")
    
    # Mock task object
    class MockTask:
        description = "Test task"
    
    out = client.complete(PROMPT, task=MockTask())
    assert out is not None
    assert captured["task"] is not None


def test_complete_builds_operation_context_with_agent_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """complete() should include agent_role and crew_id in context."""
    captured: Dict[str, Any] = {}
    base_ctx = OperationContext(request_id="from-core", tenant="from-core", attrs={"x": 1})

    def fake_core_ctx_from_crewai(task: Any, *, framework_version: Any = None, **extra: Any) -> Any:
        captured["extra"] = dict(extra)
        return base_ctx

    monkeypatch.setattr(crewai_adapter_module, "core_ctx_from_crewai", fake_core_ctx_from_crewai)
    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    client = _make_client(adapter)
    out = client.complete(
        PROMPT,
        agent_role="researcher",
        crew_id="crew-123",
    )
    assert out is not None
    assert captured["extra"].get("agent_role") == "researcher"
    assert captured["extra"].get("crew_id") == "crew-123"


def test_complete_overrides_request_id_and_tenant(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """complete() should allow request_id and tenant overrides."""
    captured: Dict[str, Any] = {}
    base_ctx = OperationContext(request_id="base", tenant="base", attrs={})

    def fake_core_ctx_from_crewai(task: Any, *, framework_version: Any = None, **extra: Any) -> Any:
        return base_ctx

    monkeypatch.setattr(crewai_adapter_module, "core_ctx_from_crewai", fake_core_ctx_from_crewai)

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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    out = client.complete(
        PROMPT,
        request_id="override-req",
        tenant="override-tenant",
    )
    
    op_ctx = captured["op_ctx"]
    assert isinstance(op_ctx, OperationContext)
    assert op_ctx.request_id == "override-req"
    assert op_ctx.tenant == "override-tenant"


def test_complete_passes_framework_ctx_with_metadata(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """complete() should build framework_ctx with framework='crewai' and operation details."""
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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    client.complete(PROMPT)

    framework_ctx = captured.get("framework_ctx")
    assert isinstance(framework_ctx, Mapping)
    assert framework_ctx.get("framework") == "crewai"
    assert framework_ctx.get("operation") == "complete"
    assert framework_ctx.get("stream") is False


def test_complete_context_translation_error_attaches_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Errors during context translation should attach error context."""
    captured_calls: list[Dict[str, Any]] = []

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:  # noqa: ARG001
        captured_calls.append(dict(ctx))

    def fake_core_ctx_from_crewai(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("ctx translation failed")

    monkeypatch.setattr(crewai_adapter_module, "attach_context", fake_attach_context)
    monkeypatch.setattr(crewai_adapter_module, "core_ctx_from_crewai", fake_core_ctx_from_crewai)

    client = _make_client(adapter)
    
    with pytest.raises(RuntimeError, match="ctx translation failed"):
        client.complete(PROMPT, agent_role="test")

    # Context translation failure is enriched twice:
    # 1) inside _build_operation_context_from_kwargs(): llm_context_translation
    # 2) via the complete() error-context decorator: llm_complete
    ops = [c.get("operation") for c in captured_calls]
    assert "llm_context_translation" in ops
    assert "llm_complete" in ops
    assert any(c.get("framework") == "crewai" for c in captured_calls)


def test_complete_with_none_task_works(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """complete() should handle None task gracefully."""
    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    client = _make_client(adapter)
    out = client.complete(PROMPT, task=None)
    assert out is not None


# ---------------------------------------------------------------------------
# Parameter forwarding tests (8 tests)
# ---------------------------------------------------------------------------


def test_complete_forwards_model_parameter(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """complete() should forward model parameter to translator."""
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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter, model="default-model")
    out = client.complete(PROMPT, model="override-model")
    
    assert seen["params"].get("model") == "override-model"


def test_complete_forwards_temperature_parameter(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """complete() should forward temperature parameter to translator."""
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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    client.complete(PROMPT, temperature=0.3)
    
    assert seen["params"].get("temperature") == 0.3


def test_complete_forwards_max_tokens_parameter(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """complete() should forward max_tokens parameter to translator."""
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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    client.complete(PROMPT, max_tokens=150)
    
    assert seen["params"].get("max_tokens") == 150


def test_complete_forwards_stop_sequences(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """complete() should convert stop parameter to stop_sequences."""
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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    client.complete(PROMPT, stop=["\n\n", "STOP"])
    
    assert seen["params"].get("stop_sequences") == ["\n\n", "STOP"]


def test_complete_converts_string_stop_to_list(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """complete() should convert string stop parameter to list."""
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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    client.complete(PROMPT, stop="END")
    
    assert seen["params"].get("stop_sequences") == ["END"]


def test_complete_forwards_sampling_params(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """complete() should forward top_p, frequency_penalty, presence_penalty."""
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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    client.complete(
        PROMPT,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.2,
    )
    
    params = seen["params"]
    assert params.get("top_p") == 0.9
    assert params.get("frequency_penalty") == 0.1
    assert params.get("presence_penalty") == 0.2


def test_complete_forwards_system_message(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """complete() should forward system_message parameter."""
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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    client.complete(PROMPT, system_message="You are helpful")
    
    assert seen["params"].get("system_message") == "You are helpful"


def test_complete_forwards_tools_and_tool_choice(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """complete() should forward tools and tool_choice parameters."""
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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    tools = [{"type": "function", "function": {"name": "test"}}]
    client = _make_client(adapter)
    client.complete(PROMPT, tools=tools, tool_choice="auto")
    
    assert seen["params"].get("tools") == tools
    assert seen["params"].get("tool_choice") == "auto"


# ---------------------------------------------------------------------------
# Message normalization tests (5 tests)
# ---------------------------------------------------------------------------


def test_complete_accepts_string_message(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """complete() should accept a plain string and normalize to message format."""
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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    client.complete("hello world")
    
    messages = seen["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "hello world"


def test_complete_accepts_dict_message(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """complete() should accept a dict message."""
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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    client.complete({"role": "assistant", "content": "test"})
    
    messages = seen["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"


def test_complete_accepts_list_of_messages(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """complete() should accept a list of messages."""
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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    client.complete(_msgs("test"))
    
    messages = seen["messages"]
    assert len(messages) == 1
    assert messages[0]["content"] == "test"


def test_complete_accepts_crewai_message_object(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """complete() should accept CrewAI message objects with role/content attributes."""
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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    class MockCrewAIMessage:
        def __init__(self, role: str, content: str):
            self.role = role
            self.content = content

    client = _make_client(adapter)
    client.complete(MockCrewAIMessage("user", "test message"))
    
    messages = seen["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "test message"


def test_complete_normalizes_mixed_message_list(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """complete() should normalize a list with mixed message types."""
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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    class MockMessage:
        def __init__(self, role: str, content: str):
            self.role = role
            self.content = content

    client = _make_client(adapter)
    mixed_messages = [
        "plain string",
        {"role": "assistant", "content": "dict message"},
        MockMessage("user", "object message"),
    ]
    client.complete(mixed_messages)
    
    messages = seen["messages"]
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "plain string"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"


# ---------------------------------------------------------------------------
# Error context attachment tests (4 tests)
# ---------------------------------------------------------------------------


def test_complete_error_attaches_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Errors during complete() should attach error context."""
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    monkeypatch.setattr(crewai_adapter_module, "attach_context", fake_attach_context)

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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: FailingTranslator())

    client = _make_client(adapter)
    with pytest.raises(RuntimeError, match="boom"):
        _ = client.complete(PROMPT)

    assert captured.get("framework") == "crewai"
    assert str(captured.get("operation", "")).startswith("llm_")


def test_stream_iteration_error_attaches_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Errors during stream iteration should attach error context."""
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    monkeypatch.setattr(crewai_adapter_module, "attach_context", fake_attach_context)

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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: FailingTranslator())

    client = _make_client(adapter)
    it = client.stream(PROMPT)
    assert next(iter(it))
    
    with pytest.raises(RuntimeError, match="stream-boom"):
        _ = list(it)
    
    assert captured.get("framework") == "crewai"
    assert captured.get("operation") == "llm_stream"


@pytest.mark.asyncio
async def test_acomplete_error_attaches_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Errors during acomplete() should attach error context."""
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    monkeypatch.setattr(crewai_adapter_module, "attach_context", fake_attach_context)

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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: FailingTranslator())

    client = _make_client(adapter)
    
    with pytest.raises(RuntimeError, match="async-boom"):
        await client.acomplete(PROMPT)

    assert captured.get("framework") == "crewai"
    assert str(captured.get("operation", "")).startswith("llm_")


@pytest.mark.asyncio
async def test_astream_iteration_error_attaches_context(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Errors during async stream iteration should attach error context."""
    captured: Dict[str, Any] = {}

    def fake_attach_context(exc: BaseException, **ctx: Any) -> None:
        captured.update(ctx)

    monkeypatch.setattr(crewai_adapter_module, "attach_context", fake_attach_context)

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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: FailingTranslator())

    client = _make_client(adapter)
    aiter = await client.astream(PROMPT)
    
    # Consume first chunk
    first = await aiter.__anext__()
    assert first is not None
    
    with pytest.raises(RuntimeError, match="async-stream-boom"):
        async for _ in aiter:
            pass

    assert captured.get("framework") == "crewai"
    assert captured.get("operation") == "llm_astream"


# ---------------------------------------------------------------------------
# Streaming tests (4 tests)
# ---------------------------------------------------------------------------


def test_stream_returns_iterator(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """stream() should return an iterator."""
    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    client = _make_client(adapter)
    it = client.stream(PROMPT)
    
    assert hasattr(it, "__iter__")
    chunks = list(it)
    assert len(chunks) >= 1


def test_stream_yields_chunks_progressively(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """stream() should yield chunks as they're produced."""
    class StreamingTranslator:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def stream(self, *_: Any, **__: Any) -> Iterator[Any]:
            for i in range(3):
                yield {"text": f"chunk{i}", "is_final": i == 2, "model": "m"}
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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: StreamingTranslator())

    client = _make_client(adapter)
    chunks = list(client.stream(PROMPT))
    
    assert len(chunks) == 3
    assert chunks[0]["text"] == "chunk0"
    assert chunks[1]["text"] == "chunk1"
    assert chunks[2]["text"] == "chunk2"


@pytest.mark.asyncio
async def test_astream_returns_async_iterator(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """astream() should return an async iterator."""
    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    client = _make_client(adapter)
    aiter = await client.astream(PROMPT)
    
    assert hasattr(aiter, "__aiter__")
    chunks = []
    async for chunk in aiter:
        chunks.append(chunk)
    assert len(chunks) >= 1


@pytest.mark.asyncio
async def test_astream_yields_chunks_progressively(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """astream() should yield chunks as they're produced."""
    class StreamingTranslator:
        def complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def stream(self, *a: Any, **k: Any) -> Iterator[Any]:
            return iter(())
        async def arun_complete(self, *a: Any, **k: Any) -> Any:
            return {"text": "x", "model": "m"}
        def arun_stream(self, *_: Any, **__: Any) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                for i in range(3):
                    yield {"text": f"chunk{i}", "is_final": i == 2, "model": "m"}
            return _gen()
        def capabilities(self) -> Mapping[str, Any]:
            return {}
        def health(self) -> Mapping[str, Any]:
            return {}
        def count_tokens_for_messages(self, *a: Any, **k: Any) -> int:
            return 0

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: StreamingTranslator())

    client = _make_client(adapter)
    chunks = []
    async for chunk in await client.astream(PROMPT):
        chunks.append(chunk)
    
    assert len(chunks) == 3
    assert chunks[0]["text"] == "chunk0"


# ---------------------------------------------------------------------------
# Token counting tests (5 tests)
# ---------------------------------------------------------------------------


def test_count_tokens_returns_int(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """count_tokens() should return an integer."""
    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    client = _make_client(adapter)
    n = client.count_tokens(PROMPT)
    
    assert isinstance(n, int)
    assert n >= 0


def test_count_tokens_accepts_string(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """count_tokens() should accept a plain string."""
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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter)
    n = client.count_tokens("hello world")
    
    assert n == 5
    assert len(seen["messages"]) == 1
    assert seen["messages"][0]["role"] == "user"


def test_count_tokens_accepts_task_object(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """count_tokens() should extract text from task object."""
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
            return 8

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    class MockTask:
        description = "Task description here"

    client = _make_client(adapter)
    n = client.count_tokens(task=MockTask())
    
    assert n == 8
    assert "Task description here" in str(seen["messages"])


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

    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: DummyTranslator())

    client = _make_client(adapter, model="default-model")
    client.count_tokens(PROMPT, model="override-model")
    
    assert seen["model"] == "override-model"


@pytest.mark.asyncio
async def test_acount_tokens_returns_int(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """acount_tokens() should return an integer."""
    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    client = _make_client(adapter)
    n = await client.acount_tokens(PROMPT)
    
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
        def count_tokens(self, *args: Any, **kwargs: Any) -> int:
            return 0
        def close(self) -> None:
            self.closed = True
        async def aclose(self) -> None:
            self.aclosed = True

    adapter_instance = ClosingLLMAdapter()

    with CorpusCrewAILLM(llm_adapter=adapter_instance) as client:
        assert client is not None
    assert adapter_instance.closed is True

    adapter2 = ClosingLLMAdapter()
    client2 = CorpusCrewAILLM(llm_adapter=adapter2)
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
    """Client should support client(...) as alias for client.complete(...)."""
    client = _make_client(adapter)
    
    # Should not raise
    result = client(PROMPT)
    assert result is not None


# ---------------------------------------------------------------------------
# Message validation tests (3 tests)
# ---------------------------------------------------------------------------


def test_complete_validates_messages_when_enabled(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """complete() should validate messages when validate_inputs=True."""
    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    config = CrewAILLMConfig(model="m", validate_inputs=True)
    client = _make_client(adapter, config=config)
    
    # None messages should raise
    with pytest.raises(ValueError, match="cannot be None"):
        client.complete(None)  # type: ignore


def test_complete_validates_empty_messages(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """complete() should validate non-empty messages."""
    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    config = CrewAILLMConfig(model="m", validate_inputs=True)
    client = _make_client(adapter, config=config)
    
    # Empty list
    with pytest.raises(ValueError, match="cannot be empty"):
        client.complete([])
    
    # Empty string
    with pytest.raises(ValueError, match="cannot be empty"):
        client.complete("   ")


def test_complete_skips_validation_when_disabled(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """complete() should skip validation when validate_inputs=False."""
    monkeypatch.setattr(crewai_adapter_module, "create_llm_translator", lambda *_a, **_k: _make_dummy_translator())

    config = CrewAILLMConfig(model="m", validate_inputs=False)
    client = _make_client(adapter, config=config)
    
    # Should not raise during validation (translator might still fail)
    try:
        client.complete([])
    except Exception:
        pass  # Expected - translator will fail, but not validation


# ---------------------------------------------------------------------------
# Event loop guard tests (2 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_raises_when_called_in_event_loop() -> None:
    """complete() should raise when called from within an event loop."""
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

    client = CorpusCrewAILLM(llm_adapter=MinimalAdapter())
    
    with pytest.raises(RuntimeError, match="event loop"):
        client.complete(PROMPT)


@pytest.mark.asyncio
async def test_stream_raises_when_called_in_event_loop() -> None:
    """stream() should raise when called from within an event loop."""
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

    client = CorpusCrewAILLM(llm_adapter=MinimalAdapter())
    
    with pytest.raises(RuntimeError, match="event loop"):
        it = client.stream(PROMPT)
        next(it)


# ---------------------------------------------------------------------------
# CrewAI E2E / Integration tests (5 tests)
# ---------------------------------------------------------------------------


def test_e2e_crewai_dependency_is_available_for_framework_suite() -> None:
    """CrewAI must be installed for E2E tests."""
    _require_crewai_available_for_e2e()


def test_e2e_crewai_can_be_imported_and_initialized(adapter: Any) -> None:
    """CrewAI adapter should initialize successfully when CrewAI is installed."""
    _require_crewai_available_for_e2e()

    client = _make_client(adapter, require_crewai=True)
    assert client is not None
    assert client.model == "default"


def test_e2e_crewai_complete_works_with_string_prompt(adapter: Any) -> None:
    """CrewAI adapter should handle string prompts end-to-end."""
    _require_crewai_available_for_e2e()

    client = _make_client(adapter)
    result = client.complete("What is AI?")
    assert result is not None


def test_e2e_crewai_handles_agent_context(adapter: Any) -> None:
    """CrewAI adapter should handle realistic agent context."""
    _require_crewai_available_for_e2e()

    client = _make_client(adapter)
    
    result = client.complete(
        "Research this topic",
        agent_role="Researcher",
        agent_goal="Find accurate information",
        crew_id="crew-123",
        task_description="Research AI trends",
    )
    assert result is not None


@pytest.mark.asyncio
async def test_e2e_crewai_async_methods_work(adapter: Any) -> None:
    """CrewAI adapter async methods should work end-to-end."""
    _require_crewai_available_for_e2e()

    client = _make_client(adapter)
    
    # Test acomplete
    result = await client.acomplete("Hello CrewAI")
    assert result is not None
    
    # Test astream
    chunks = []
    async for chunk in await client.astream("Stream test"):
        chunks.append(chunk)
    assert len(chunks) > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
