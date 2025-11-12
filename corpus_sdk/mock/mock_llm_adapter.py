# SPDX-License-Identifier: Apache-2.0
"""
Mock LLM adapter used in Corpus SDK example scripts.

Implements BaseLLMAdapter methods for demonstration purposes only.
Simulates latency, token counting, streaming behavior, stop sequences, max_tokens,
parameter validation, deadline semantics, and deterministic behavior for tests.
"""

from __future__ import annotations
import asyncio
import hashlib
import random
from typing import AsyncIterator, List, Mapping, Optional, Any
from dataclasses import dataclass

from corpus_sdk.llm.llm_base import (
    BaseLLMAdapter,
    LLMCompletion,
    LLMChunk,
    TokenUsage,
    LLMCapabilities,
    OperationContext as LLMContext,
    # normalized error taxonomy
    Unavailable,
    ResourceExhausted,
    BadRequest,
    DeadlineExceeded,
)

# Demo-only helpers (not used by tests unless running __main__)
from examples.common.ctx import make_ctx
from examples.common.printing import print_json, print_kv, box


# -----------------------------
# Small helpers
# -----------------------------

def _stable_seed(*parts: str) -> int:
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    # use 48 bits to keep randint fast & stable
    return int(h[:12], 16)

def _tokenize(s: str) -> List[str]:
    # ultra-silly tokenizer = whitespace split
    return s.split()

def _join_tokens(tokens: List[str]) -> str:
    return " ".join(tokens)

def _approx_usage(prompt_text: str, completion_text: str) -> TokenUsage:
    p = len(_tokenize(prompt_text))
    c = len(_tokenize(completion_text))
    return TokenUsage(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)


_ALLOWED_ROLES = {"system", "user", "assistant"}


@dataclass
class MockLLMAdapter(BaseLLMAdapter):
    """A mock LLM adapter for protocol demonstrations."""

    name: str = "mock-llm"
    failure_rate: float = 0.1  # 10% chance of simulated failure

    def __post_init__(self) -> None:
        # Ensure BaseLLMAdapter infra is initialized and tighten stream deadline checks
        super().__init__(mode="thin", stream_deadline_check_every_n_chunks=1)

    # --- override: align DEADLINE code with tests ----------------------------
    def _preflight_deadline(self, ctx: Optional[LLMContext]) -> None:
        """
        Same logic as base, but raise DeadlineExceeded with code 'DEADLINE'
        (some tests assert this exact code on pre-expired budgets).
        """
        if ctx and ctx.deadline_ms is not None:
            import time as _t
            now_ms = int(_t.time() * 1000)
            if now_ms >= ctx.deadline_ms:
                raise DeadlineExceeded(
                    "deadline already exceeded",
                    code="DEADLINE",
                    details={"remaining_ms": 0},
                )

    # ----- local validation helpers -----------------------------------------

    def _validate_roles(self, messages: List[Mapping[str, Any]]) -> None:
        for m in messages:
            role = str(m.get("role", ""))
            if role not in _ALLOWED_ROLES:
                raise BadRequest(f"unknown role: {role!r}")

    def _validate_sampling_params_local(
        self,
        *,
        temperature: Optional[float],
        top_p: Optional[float],
        frequency_penalty: Optional[float],
        presence_penalty: Optional[float],
    ) -> None:
        """
        Mirror spec ranges here explicitly so this adapter passes even if called
        outside BaseLLMAdapter.complete/stream (belt & suspenders).
        """
        if temperature is not None and not (0.0 <= temperature <= 2.0):
            raise BadRequest("temperature must be within [0.0, 2.0]")
        if top_p is not None and not (0.0 < top_p <= 1.0):
            raise BadRequest("top_p must be within (0.0, 1.0]")
        if frequency_penalty is not None and not (-2.0 <= frequency_penalty <= 2.0):
            raise BadRequest("frequency_penalty must be within [-2.0, 2.0]")
        if presence_penalty is not None and not (-2.0 <= presence_penalty <= 2.0]):
            raise BadRequest("presence_penalty must be within [-2.0, 2.0]")

    # ----- completion --------------------------------------------------------

    async def _do_complete(
        self,
        *,
        messages: List[Mapping[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        ctx: Optional[LLMContext] = None,
    ) -> LLMCompletion:
        """Pretend to complete a chat turn with occasional simulated failures."""
        # Schema: reject unknown roles (beyond base shape checks)
        self._validate_roles(messages)
        # Explicit sampling validation (base already does this in public method)
        self._validate_sampling_params_local(
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        # ----- deterministic randomness for reproducible tests -----
        seed = _stable_seed(
            str(model or "mock-model"),
            str(system_message or ""),
            repr(messages),
            str(max_tokens),
            str(temperature),
            str(top_p),
            str(frequency_penalty),
            str(presence_penalty),
            ",".join(stop_sequences or []),
            str(getattr(ctx, "request_id", "")),
        )
        rnd = random.Random(seed)

        # ----- simulated transient failures -----
        last_content = str(messages[-1].get("content", "")) if messages else ""
        if rnd.random() < self.failure_rate:
            if "overload" in last_content.lower():
                raise Unavailable("Mocked service overload", retry_after_ms=2000, code="OVERLOAD")
            raise ResourceExhausted("Mocked rate limit", retry_after_ms=1000, code="RATE_LIMIT")

        # ----- build a simple prompt representation -----
        prompt_parts: List[str] = []
        if system_message:
            prompt_parts.append(f"[system] {system_message}")
        for m in messages:
            prompt_parts.append(f"[{m.get('role','')}] {m.get('content','')}")
        prompt_text = "\n".join(prompt_parts)

        # ----- generate mock completion -----
        base = (last_content.strip() or "ok")
        words = _tokenize(base)
        # add a deterministic flourish so it isn't a pure echo
        suffix = ["(mock)", f"[{model or 'mock-model'}]"]
        gen_tokens = words + suffix

        # temperature: if > 0, deterministically duplicate or drop some tokens
        if (temperature or 0.0) > 0:
            mutated = []
            for t in gen_tokens:
                r = rnd.random()
                if r < min(0.05 * float(temperature), 0.2):
                    mutated.extend([t, t])
                elif r < min(0.10 * float(temperature), 0.4):
                    continue
                else:
                    mutated.append(t)
            gen_tokens = mutated or gen_tokens

        # respect max_tokens (cap completion length)
        cut_by_max = False
        if max_tokens is not None:
            lim = max(0, int(max_tokens))
            if len(gen_tokens) > lim:
                cut_by_max = True
            gen_tokens = gen_tokens[:lim]

        # apply stop sequences (truncate on first match)
        cut_by_stop = False
        if stop_sequences:
            completion_so_far = _join_tokens(gen_tokens)
            cut_at = None
            for s in stop_sequences:
                if not s:
                    continue
                i = completion_so_far.find(s)
                if i != -1:
                    cut_at = i if cut_at is None else min(cut_at, i)
            if cut_at is not None:
                completion_so_far = completion_so_far[:cut_at].rstrip()
                gen_tokens = _tokenize(completion_so_far)
                cut_by_stop = True

        completion_text = _join_tokens(gen_tokens)

        # Simulate a small processing delay
        await asyncio.sleep(0.03)

        usage = _approx_usage(prompt_text, completion_text)
        finish_reason = "stop"
        if cut_by_max and not cut_by_stop:
            finish_reason = "length"

        return LLMCompletion(
            text=completion_text if completion_text else "",
            model=model or "mock-model",
            model_family="mock",
            usage=usage,
            finish_reason=finish_reason,
        )

    # ----- streaming ---------------------------------------------------------

    async def _do_stream(
        self,
        *,
        messages: List[Mapping[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        ctx: Optional[LLMContext] = None,
    ) -> AsyncIterator[LLMChunk]:
        """Simulate token streaming with progressive token counts and stop sequence handling."""
        # Schema: reject unknown roles
        self._validate_roles(messages)
        # Explicit sampling validation
        self._validate_sampling_params_local(
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        # Deterministic plan; match completion tokenization for parity
        seed = _stable_seed(
            str(model or "mock-model"),
            str(system_message or ""),
            repr(messages),
            str(max_tokens),
            str(temperature),
            str(top_p),
            str(frequency_penalty),
            str(presence_penalty),
            ",".join(stop_sequences or []),
            str(getattr(ctx, "request_id", "")),
            "stream",
        )
        rnd = random.Random(seed)

        last_content = str(messages[-1].get("content", "")) if messages else ""
        if rnd.random() < self.failure_rate:
            if "overload" in last_content.lower():
                raise Unavailable("Mocked service overload", retry_after_ms=2000, code="OVERLOAD")
            raise ResourceExhausted("Mocked rate limit", retry_after_ms=1000, code="RATE_LIMIT")

        prompt_parts: List[str] = []
        if system_message:
            prompt_parts.append(f"[system] {system_message}")
        for m in messages:
            prompt_parts.append(f"[{m.get('role','')}] {m.get('content','')}")
        prompt_text = "\n".join(prompt_parts)

        # Match completion suffix to keep parity test happy
        base = (last_content or "ok").strip()
        gen_tokens = _tokenize(base) + ["(mock)", f"[{model or 'mock-model'}]"]

        # temperature-driven deterministic mutation (same as in complete)
        if (temperature or 0.0) > 0:
            mutated = []
            for t in gen_tokens:
                r = rnd.random()
                if r < min(0.05 * float(temperature), 0.2):
                    mutated.extend([t, t])
                elif r < min(0.10 * float(temperature), 0.4):
                    continue
                else:
                    mutated.append(t)
            gen_tokens = mutated or gen_tokens

        # respect max_tokens
        if max_tokens is not None:
            gen_tokens = gen_tokens[: max(0, int(max_tokens))]

        # Stream one token at a time; enforce stop_sequences while streaming.
        emitted: List[str] = []

        async def _emit_usage_and_chunk(tok: str) -> LLMChunk:
            # Build partial and usage snapshot
            partial = _join_tokens(emitted)
            usage = _approx_usage(prompt_text, partial)
            return LLMChunk(
                text=tok,
                is_final=False,
                model=model or "mock-model",
                usage_so_far=usage,
            )

        for i, tok in enumerate(gen_tokens, start=1):
            # Before emitting, check if adding this token would cross a stop sequence.
            candidate = _join_tokens(emitted + [tok])
            if stop_sequences:
                cut_at = None
                for s in stop_sequences:
                    if not s:
                        continue
                    idx = candidate.find(s)
                    if idx != -1:
                        cut_at = idx if cut_at is None else min(cut_at, idx)
                if cut_at is not None:
                    # Truncate candidate to the cut point and stop emitting non-final chunks.
                    candidate = candidate[:cut_at].rstrip()
                    emitted = _tokenize(candidate)
                    break

            emitted.append(tok)
            # Add a trailing space for readability except maybe last; consumers concatenate anyway.
            yield await _emit_usage_and_chunk(tok + (" " if i < len(gen_tokens) else ""))

        # Ensure at least one non-final chunk for edge cases (e.g., max_tokens==0 or full stop at first token)
        if not emitted:
            # Emit a minimal, non-empty delta to satisfy "multiple chunks" + non-empty text semantics
            yield await _emit_usage_and_chunk("[noop]")

        # final sentinel chunk: terminal marker with empty text per canonical semantics
        final_text = _join_tokens(emitted)
        final_usage = _approx_usage(prompt_text, final_text)
        yield LLMChunk(
            text="",            # final chunk carries no new text
            is_final=True,
            model=model or "mock-model",
            usage_so_far=final_usage,
        )

    # ----- token counting ----------------------------------------------------

    async def count_tokens(  # override public to allow empty-string per tests
        self,
        text: str,
        *,
        model: Optional[str] = None,
        ctx: Optional[LLMContext] = None,
    ) -> int:
        """
        Allow empty-string counting (tests expect 0..10).
        Delegate to base for non-empty text to keep instrumentation.
        """
        if isinstance(text, str) and text == "":
            # Minimal overhead for empty prompt
            return 0
        return await super().count_tokens(text, model=model, ctx=ctx)

    async def _do_count_tokens(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        ctx: Optional[LLMContext] = None,
    ) -> int:
        """Mock token counting with word-based approximation (+3 overhead)."""
        await asyncio.sleep(0.005)
        return len(_tokenize(text)) + 3

    # ----- capabilities/health ----------------------------------------------

    async def _do_capabilities(self) -> LLMCapabilities:
        """Report mock model capabilities."""
        return LLMCapabilities(
            server="mock",
            version="1.0.0",
            model_family="mock",
            max_context_length=4096,
            supports_streaming=True,
            supports_roles=True,
            supports_json_output=False,
            supports_parallel_tool_calls=False,
            idempotent_writes=False,
            supports_multi_tenant=True,
            supports_system_message=True,
            supports_deadline=True,
            supports_count_tokens=True,
            supported_models=("mock-model", "mock-model-pro"),
        )

    async def _do_health(self, *, ctx: Optional[LLMContext] = None) -> Mapping[str, object]:
        """Mock health check with occasional degraded status."""
        if random.random() < 0.2:  # 20% chance of degraded
            return {"ok": False, "status": "degraded", "server": "mock", "version": "1.0.0"}
        return {"ok": True, "status": "healthy", "server": "mock", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Demo usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Run this module directly to see mock adapter behavior in action."""

    async def _demo() -> None:
        random.seed(42)  # Deterministic for reproducible demos
        box("MockLLMAdapter Demo")
        adapter = MockLLMAdapter(failure_rate=0.3)  # Higher failure rate for demo
        ctx = make_ctx(LLMContext, tenant="demo", request_id="demo-req-1")

        # --- Capabilities example ---
        print("\n=== CAPABILITIES ===")
        caps = await adapter.capabilities()
        print_json(caps.__dict__)

        # --- Health check example ---
        print("\n=== HEALTH CHECK ===")
        health = await adapter.health(ctx=ctx)
        print_kv(health)

        # --- Complete example (with stops/max_tokens) ---
        print("\n=== COMPLETE ===")
        try:
            result = await adapter.complete(
                messages=[{"role": "user", "content": "hello world please show me something nice"}],
                model="mock-model",
                max_tokens=6,
                stop_sequences=["something"],
                system_message="You are helpful.",
                ctx=ctx,
            )
            print_kv({"Output": result.text, "FinishReason": result.finish_reason})
            print_json(result.usage.__dict__)
        except Exception as e:
            print_kv({"Error": str(e), "Type": type(e).__name__})

        # --- Stream example (with stop sequence) ---
        print("\n=== STREAM ===")
        try:
            async for chunk in adapter.stream(
                messages=[{"role": "user", "content": "stream this message and then STOP please"}],
                model="mock-model-pro",
                temperature=0.7,
                stop_sequences=["STOP"],
                ctx=ctx,
            ):
                # Show emitted text for non-final chunks; denote final
                if chunk.is_final:
                    print("\n[final]", chunk.usage_so_far.__dict__)
                else:
                    print(chunk.text, end="", flush=True)
            print("\n[done]")
        except Exception as e:
            print(f"\nStream error: {e}")

        # --- Token counting example ---
        print("\n=== TOKEN COUNTING ===")
        try:
            count = await adapter.count_tokens("This is a test sentence", ctx=ctx)
            print_kv({"Text": "This is a test sentence", "Tokens": count})
            count_empty = await adapter.count_tokens("", ctx=ctx)
            print_kv({"Text": "<empty>", "Tokens": count_empty})
        except Exception as e:
            print_kv({"Error": str(e)})

    asyncio.run(_demo())
