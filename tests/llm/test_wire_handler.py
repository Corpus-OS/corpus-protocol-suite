# SPDX-License-Identifier: Apache-2.0
"""
LLM Conformance — Wire-level envelopes & routing.

Asserts (SPECIFICATION.md refs):
  • §4.1         — Canonical op strings and envelope shapes (STRICT wire boundary)
  • §8.3, §8.4   — LLM operations mapped via `llm.<op>` wire contract
  • §6.1         — OperationContext constructed from wire ctx (ignore unknowns)
  • §12.1, §12.4 — Normalized error envelopes for LLMAdapterError subclasses
  • §11.2, §13   — SIEM-safe behavior (no tenant leakage via Wire handler)

Covers:
  • llm.capabilities → unary success envelope
  • llm.complete / llm.count_tokens / llm.health → unary success envelopes
  • llm.stream via handle_stream() → STREAMING chunk envelopes
  • Context translation: wire ctx → OperationContext passed into BaseLLMAdapter
  • Unknown op → NotSupported → normalized error envelope
  • Missing/invalid op → BadRequest → normalized error envelope
  • Strictness: envelope MUST include op, ctx, args; ctx/args MUST be objects
  • LLMAdapterError → mapped with correct code/error/message/details
  • Unexpected Exception → mapped to UNAVAILABLE with stable, non-leaky message ("internal error")
  • Streaming error: final error envelope emitted and stream terminates
"""

import pytest
from typing import Any, Dict, List, Mapping, Optional, AsyncIterator

from corpus_sdk.llm.llm_base import (
    LLMCapabilities,
    LLMCompletion,
    LLMChunk,
    TokenUsage,
    OperationContext,
    LLMAdapterError,
    BadRequest,
    NotSupported,
    Unavailable,
    BaseLLMAdapter,
    WireLLMHandler,
)

pytestmark = pytest.mark.asyncio


class TrackingMockLLMAdapter(BaseLLMAdapter):
    """
    Test adapter for exercising WireLLMHandler with tracking capabilities.

    IMPORTANT:
      - We intentionally rely on BaseLLMAdapter for validation/gates.
      - We implement only _do_* hooks.
    """

    def __init__(self) -> None:
        super().__init__()
        self.last_ctx: Optional[OperationContext] = None
        self.last_call: Optional[str] = None
        self.last_args: Dict[str, Any] = {}
        self._caps = LLMCapabilities(
            server="mock",
            version="1.0.0",
            model_family="mock",
            max_context_length=4096,
            supports_streaming=True,
            supports_roles=True,
            supports_json_output=False,
            supports_tools=False,
            supports_parallel_tool_calls=False,
            supports_tool_choice=False,
            idempotent_writes=True,
            supports_multi_tenant=True,
            supports_system_message=True,
            supports_deadline=True,
            supports_count_tokens=True,
            supported_models=("mock-model", "mock-model-2"),
        )

    # --- helpers -------------------------------------------------------------

    def _store(self, op: str, ctx: Optional[OperationContext], **kwargs: Any) -> None:
        self.last_call = op
        self.last_ctx = ctx
        self.last_args = dict(kwargs)

    # --- backend hooks w/ tracking ------------------------------------------

    async def _do_capabilities(self) -> LLMCapabilities:
        self._store("capabilities", None)
        return self._caps

    async def _do_complete(
        self,
        *,
        messages,
        max_tokens=None,
        temperature=None,
        top_p=None,
        frequency_penalty=None,
        presence_penalty=None,
        stop_sequences=None,
        model=None,
        system_message=None,
        tools=None,
        tool_choice=None,
        ctx: Optional[OperationContext] = None,
    ) -> LLMCompletion:
        self._store(
            "complete",
            ctx,
            messages=messages,
            model=model,
            system_message=system_message,
        )
        return LLMCompletion(
            text="Mock completion response",
            model=model or "mock-model",
            model_family="mock",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="stop",
        )

    async def _do_stream(
        self,
        *,
        messages,
        max_tokens=None,
        temperature=None,
        top_p=None,
        frequency_penalty=None,
        presence_penalty=None,
        stop_sequences=None,
        model=None,
        system_message=None,
        tools=None,
        tool_choice=None,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[LLMChunk]:
        self._store(
            "stream",
            ctx,
            messages=messages,
            model=model,
            system_message=system_message,
        )

        # Yield multiple chunks including final
        yield LLMChunk(text="First ", is_final=False, model=model or "mock-model")
        yield LLMChunk(text="chunk ", is_final=False, model=model or "mock-model")
        yield LLMChunk(text="content", is_final=False, model=model or "mock-model")
        yield LLMChunk(text="", is_final=True, model=model or "mock-model")

    async def _do_count_tokens(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        ctx: Optional[OperationContext] = None,
    ) -> int:
        self._store("count_tokens", ctx, text=text, model=model)
        return len(str(text).split())  # Simple token approximation

    async def _do_health(
        self,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> Mapping[str, Any]:
        self._store("health", ctx)
        return {"ok": True, "server": "mock", "version": "1.0.0"}


class ErrorAdapter(TrackingMockLLMAdapter):
    """
    Adapter that always raises a specific LLMAdapterError for testing mapping.
    """

    def __init__(self, exc: LLMAdapterError):
        super().__init__()
        self._exc = exc

    async def _do_complete(self, **kwargs: Any) -> LLMCompletion:
        raise self._exc


class BoomAdapter(TrackingMockLLMAdapter):
    """
    Adapter that raises unexpected exceptions for testing error hardening.
    """

    async def _do_complete(self, **kwargs: Any) -> LLMCompletion:
        raise RuntimeError("boom")


class StreamErrorAdapter(TrackingMockLLMAdapter):
    """
    Adapter that raises an LLMAdapterError mid-stream to verify stream error envelopes terminate.
    """

    async def _do_stream(self, **kwargs: Any) -> AsyncIterator[LLMChunk]:
        yield LLMChunk(text="partial", is_final=False, model="mock-model")
        raise Unavailable("backend unavailable", retry_after_ms=42, details={"hint": "retry later"})


# ---------------------------------------------------------------------------
# Success-path envelopes
# ---------------------------------------------------------------------------

async def test_wire_contract_capabilities_success_envelope():
    a = TrackingMockLLMAdapter()
    h = WireLLMHandler(a)

    res = await h.handle({"op": "llm.capabilities", "ctx": {}, "args": {}})

    assert res["ok"] is True
    assert res["code"] == "OK"
    assert isinstance(res["ms"], (int, float)) and res["ms"] >= 0
    assert isinstance(res["result"], dict)

    out = res["result"]
    assert out["server"] == "mock"
    assert out["version"] == "1.0.0"
    assert out["model_family"] == "mock"
    assert "supported_models" in out
    assert len(out["supported_models"]) >= 1


async def test_wire_contract_complete_roundtrip_and_context_plumbing():
    a = TrackingMockLLMAdapter()
    h = WireLLMHandler(a)

    ctx_wire = {
        "request_id": "req_wire_llm",
        "idempotency_key": "idem-llm",
        "deadline_ms": 9999999999999,
        "traceparent": "00-abc-xyz-01",
        "tenant": "acme-tenant",
        "attrs": {"k": "v"},
        "ignore_me": "extra",  # MUST be ignored by ctx mapping
    }
    args = {"messages": [{"role": "user", "content": "hi"}], "model": "mock-model"}

    res = await h.handle({"op": "llm.complete", "ctx": ctx_wire, "args": args})

    # Envelope shape
    assert res["ok"] is True
    assert res["code"] == "OK"
    assert isinstance(res["ms"], (int, float)) and res["ms"] >= 0

    out = res["result"]
    assert isinstance(out["text"], str) and out["text"]
    assert out["model"] == "mock-model"
    assert out["model_family"] == "mock"
    assert "usage" in out

    # Context propagation via WireLLMHandler -> OperationContext passed into adapter hook
    assert a.last_call == "complete"
    assert isinstance(a.last_ctx, OperationContext)
    assert a.last_ctx.request_id == "req_wire_llm"
    assert a.last_ctx.idempotency_key == "idem-llm"
    assert a.last_ctx.traceparent == "00-abc-xyz-01"
    assert a.last_ctx.tenant == "acme-tenant"
    # unknown ctx field ignored (should not appear inside attrs)
    assert "ignore_me" not in (a.last_ctx.attrs or {})


async def test_wire_contract_count_tokens_and_health_envelopes():
    a = TrackingMockLLMAdapter()
    h = WireLLMHandler(a)

    # count_tokens
    ct_res = await h.handle(
        {
            "op": "llm.count_tokens",
            "ctx": {"request_id": "ct1"},
            "args": {"text": "hello world", "model": "mock-model"},
        }
    )
    assert ct_res["ok"] is True
    assert ct_res["code"] == "OK"
    assert isinstance(ct_res["result"], int)
    assert ct_res["result"] >= 0

    # health
    health_res = await h.handle({"op": "llm.health", "ctx": {"request_id": "h1"}, "args": {}})
    assert health_res["ok"] is True
    assert health_res["code"] == "OK"
    hr = health_res["result"]
    assert hr["server"] == "mock"
    assert hr["version"] == "1.0.0"
    assert isinstance(hr["ok"], bool)


# ---------------------------------------------------------------------------
# Streaming via handle_stream
# ---------------------------------------------------------------------------

async def test_wire_contract_stream_success_chunks_and_context():
    a = TrackingMockLLMAdapter()
    h = WireLLMHandler(a)

    env = {
        "op": "llm.stream",
        "ctx": {"request_id": "stream-1", "tenant": "stream-tenant"},
        "args": {"messages": [{"role": "user", "content": "stream me"}], "model": "mock-model"},
    }

    frames: List[Dict[str, Any]] = []
    async for envelope in h.handle_stream(env):
        frames.append(envelope)

    assert len(frames) >= 2

    for env_out in frames:
        assert env_out["ok"] is True
        assert env_out["code"] == "STREAMING"
        assert "chunk" in env_out
        ch = env_out["chunk"]
        assert isinstance(ch, dict)
        assert "text" in ch
        assert "is_final" in ch

    finals = [f for f in frames if f["chunk"].get("is_final")]
    assert len(finals) == 1, "Expected exactly one final chunk"
    assert frames[-1]["chunk"].get("is_final") is True, "Final chunk must be last"

    # Ensure context was passed through to adapter stream path
    assert a.last_call == "stream"
    assert isinstance(a.last_ctx, OperationContext)
    assert a.last_ctx.request_id == "stream-1"
    assert a.last_ctx.tenant == "stream-tenant"


# ---------------------------------------------------------------------------
# Wire strictness
# ---------------------------------------------------------------------------

async def test_wire_strictness_missing_required_keys_maps_to_bad_request():
    a = TrackingMockLLMAdapter()
    h = WireLLMHandler(a)

    res1 = await h.handle({"op": "llm.capabilities", "args": {}})
    assert res1["ok"] is False
    assert res1["code"] == "BAD_REQUEST"
    assert res1["error"] == "BadRequest"
    assert "missing required 'ctx'" in res1["message"]

    res2 = await h.handle({"op": "llm.capabilities", "ctx": {}})
    assert res2["ok"] is False
    assert res2["code"] == "BAD_REQUEST"
    assert res2["error"] == "BadRequest"
    assert "missing required 'args'" in res2["message"]


async def test_wire_strictness_ctx_and_args_must_be_objects():
    a = TrackingMockLLMAdapter()
    h = WireLLMHandler(a)

    res1 = await h.handle({"op": "llm.capabilities", "ctx": "nope", "args": {}})
    assert res1["ok"] is False
    assert res1["code"] == "BAD_REQUEST"
    assert res1["error"] == "BadRequest"
    assert "ctx must be an object" in res1["message"]

    res2 = await h.handle({"op": "llm.capabilities", "ctx": {}, "args": "nope"})
    assert res2["ok"] is False
    assert res2["code"] == "BAD_REQUEST"
    assert res2["error"] == "BadRequest"
    assert "args must be an object" in res2["message"]


# ---------------------------------------------------------------------------
# Error mapping semantics
# ---------------------------------------------------------------------------

async def test_wire_contract_unknown_op_maps_to_not_supported():
    a = TrackingMockLLMAdapter()
    h = WireLLMHandler(a)

    res = await h.handle({"op": "llm.nope", "ctx": {}, "args": {}})

    assert res["ok"] is False
    assert res["code"] == "NOT_SUPPORTED"
    assert res["error"] == "NotSupported"
    assert "unknown or non-unary operation" in res["message"]


async def test_wire_contract_missing_or_invalid_op_maps_to_bad_request():
    a = TrackingMockLLMAdapter()
    h = WireLLMHandler(a)

    res = await h.handle({"ctx": {}, "args": {}})

    assert res["ok"] is False
    assert res["code"] == "BAD_REQUEST"
    assert res["error"] == "BadRequest"
    assert "missing or invalid 'op'" in res["message"]


async def test_wire_contract_maps_llm_adapter_error_to_normalized_envelope():
    exc = BadRequest("bad llm call")
    a = ErrorAdapter(exc)
    h = WireLLMHandler(a)

    res = await h.handle(
        {
            "op": "llm.complete",
            "ctx": {"request_id": "err-llm"},
            "args": {"messages": [{"role": "user", "content": "x"}], "model": "mock-model"},
        }
    )

    assert res["ok"] is False
    assert res["code"] == "BAD_REQUEST"
    assert res["error"] == "BadRequest"
    assert res["message"] == "bad llm call"
    assert "details" in res  # JSON-safe details present (may be null)
    assert "retry_after_ms" in res
    assert isinstance(res["ms"], (int, float)) and res["ms"] >= 0


async def test_wire_contract_maps_unexpected_exception_to_unavailable_stable_message():
    a = BoomAdapter()
    h = WireLLMHandler(a)

    res = await h.handle(
        {
            "op": "llm.complete",
            "ctx": {"request_id": "boom"},
            "args": {"messages": [{"role": "user", "content": "hi"}], "model": "mock-model"},
        }
    )

    assert res["ok"] is False
    assert res["code"] == "UNAVAILABLE"
    assert res["error"] == "RuntimeError"
    # Unknown exception: stable message; do not echo raw exception text (may leak internals).
    assert res["message"] == "internal error"
    assert res.get("details") is None


async def test_wire_stream_error_envelope_terminates_stream():
    a = StreamErrorAdapter()
    h = WireLLMHandler(a)

    env = {
        "op": "llm.stream",
        "ctx": {"request_id": "stream-err", "tenant": "t1"},
        "args": {"messages": [{"role": "user", "content": "x"}], "model": "mock-model"},
    }

    frames: List[Dict[str, Any]] = []
    async for f in h.handle_stream(env):
        frames.append(f)

    assert len(frames) >= 2, "Expected at least one chunk and a final error envelope"

    # First frame is a streaming chunk envelope
    assert frames[0]["ok"] is True
    assert frames[0]["code"] == "STREAMING"
    assert "chunk" in frames[0]

    # Last frame is an error envelope and terminates the stream
    last = frames[-1]
    assert last["ok"] is False
    assert last["code"] == "UNAVAILABLE"
    assert last["error"] == "Unavailable"
    assert isinstance(last.get("message"), str) and last["message"]
    assert last.get("retry_after_ms") == 42
    assert isinstance(last.get("ms"), (int, float)) and last["ms"] >= 0
