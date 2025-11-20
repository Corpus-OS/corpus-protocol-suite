# SPDX-License-Identifier: Apache-2.0
"""
Mock LLM adapter used in Corpus SDK example scripts and conformance tests.

Implements BaseLLMAdapter methods for demonstration purposes only.
Simulates latency, token counting, streaming behavior, stop sequences, max_tokens,
deadline semantics, and deterministic behavior for tests.

Key properties:
- Deterministic given the same inputs (including ctx.request_id)
- Shared planning path for complete() and stream() so final text matches
- Default failure_rate=0.0 for conformance (can be raised for demos)
- Deterministic health where ctx.attrs["health"] can force degraded/error
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Mapping, Optional, Sequence

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
    NotSupported,
)

# Demo-only helpers (not required by tests)
try:  # guard so import doesn’t explode if examples package isn’t present
    from examples.common.ctx import make_ctx
    from examples.common.printing import print_json, print_kv, box
except Exception:  # pragma: no cover - demo convenience only
    make_ctx = None
    print_json = None
    print_kv = None
    box = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

_ALLOWED_ROLES = {"system", "user", "assistant"}


def _stable_seed(*parts: str) -> int:
    """Stable 48-bit seed from a list of string parts."""
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    # use 48 bits to keep randint fast & stable
    return int(h[:12], 16)


def _tokenize(s: str) -> List[str]:
    # Ultra-simple tokenizer: whitespace split
    return s.split()


def _join_tokens(tokens: Sequence[str]) -> str:
    return " ".join(tokens)


def _approx_usage(prompt_text: str, completion_text: str) -> TokenUsage:
    p = len(_tokenize(prompt_text))
    c = len(_tokenize(completion_text))
    return TokenUsage(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)


@dataclass
class MockLLMAdapter(BaseLLMAdapter):
    """A mock LLM adapter for protocol demonstrations & conformance tests."""

    name: str = "mock-llm"
    # Default 0.0 so conformance runs are deterministic and non-flaky.
    failure_rate: float = 0.0

    def __post_init__(self) -> None:
        # Ensure BaseLLMAdapter infra is initialized and check deadlines frequently in streams.
        super().__init__(mode="thin", stream_deadline_check_every_n_chunks=1)

    # -----------------------------------------------------------------------
    # Deadline preflight override
    # -----------------------------------------------------------------------
    def _preflight_deadline(self, ctx: Optional[LLMContext]) -> None:
        """
        Same logic as the base class but we explicitly set code='DEADLINE'
        so tests that assert this exact code on pre-expired budgets pass.
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

    # -----------------------------------------------------------------------
    # Local validation helper
    # -----------------------------------------------------------------------
    def _validate_roles(self, messages: List[Mapping[str, Any]]) -> None:
        """Reject unknown roles beyond the base schema checks."""
        for m in messages:
            role = str(m.get("role", ""))
            if role not in _ALLOWED_ROLES:
                raise BadRequest(f"unknown role: {role!r}")

    # -----------------------------------------------------------------------
    # Shared planning for complete() and stream()
    # -----------------------------------------------------------------------
    def _make_rng(
        self,
        *,
        model: Optional[str],
        system_message: Optional[str],
        messages: List[Mapping[str, Any]],
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        frequency_penalty: Optional[float],
        presence_penalty: Optional[float],
        stop_sequences: Optional[List[str]],
        ctx: Optional[LLMContext],
    ) -> random.Random:
        """Build a deterministic RNG that is identical for complete + stream."""
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
        return random.Random(seed)

    def _build_base_tokens(self, *, last_content: str, model: Optional[str]) -> List[str]:
        """
        Core token plan before sampling / temperature effects.
        Always adds a deterministic suffix so it's not pure echo.
        """
        base = (last_content.strip() or "ok")
        words = _tokenize(base)
        suffix = ["(mock)", f"[{model or 'mock-model'}]"]
        return words + suffix

    def _apply_temperature(
        self,
        gen_tokens: List[str],
        *,
        temperature: Optional[float],
        rnd: random.Random,
    ) -> List[str]:
        """Deterministic temperature effects (dup/drop some tokens)."""
        t = float(temperature or 0.0)
        if t <= 0.0:
            return gen_tokens

        mutated: List[str] = []
        for tok in gen_tokens:
            r = rnd.random()
            dup_thresh = min(0.05 * t, 0.2)
            drop_thresh = min(0.10 * t, 0.4)
            if r < dup_thresh:
                mutated.extend([tok, tok])
            elif r < drop_thresh:
                continue
            else:
                mutated.append(tok)

        return mutated or gen_tokens

    def _apply_max_and_stops(
        self,
        gen_tokens: List[str],
        *,
        max_tokens: Optional[int],
        stop_sequences: Optional[List[str]],
    ) -> tuple[List[str], bool, bool]:
        """
        Apply max_tokens then stop sequences. Returns:
        (final_tokens, cut_by_max, cut_by_stop).

        Also ensures that final_tokens is never empty, to satisfy tests that
        aggregate text must be non-empty.
        """
        tokens = list(gen_tokens)
        cut_by_max = False
        cut_by_stop = False

        # max_tokens
        if max_tokens is not None:
            lim = max(0, int(max_tokens))
            if len(tokens) > lim:
                cut_by_max = True
            tokens = tokens[:lim]

        # stop sequences (string-based, then re-tokenize)
        if stop_sequences:
            completion_text = _join_tokens(tokens)
            cut_at: Optional[int] = None
            for s in stop_sequences:
                if not s:
                    continue
                idx = completion_text.find(s)
                if idx != -1:
                    cut_at = idx if cut_at is None else min(cut_at, idx)
            if cut_at is not None:
                completion_text = completion_text[:cut_at].rstrip()
                tokens = _tokenize(completion_text)
                cut_by_stop = True

        # Ensure non-empty final tokens to keep aggregate non-empty.
        if not tokens:
            tokens = ["(mock)"]
            cut_by_max = False
            cut_by_stop = False

        return tokens, cut_by_max, cut_by_stop

    def _plan_completion_text(
        self,
        *,
        messages: List[Mapping[str, Any]],
        model: Optional[str],
        system_message: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        frequency_penalty: Optional[float],
        presence_penalty: Optional[float],
        stop_sequences: Optional[List[str]],
        ctx: Optional[LLMContext],
    ) -> tuple[str, str, str]:
        """
        Compute prompt_text, completion_text, and finish_reason.
        Used by both complete() and stream() so their aggregate text matches.
        """
        # Build prompt representation
        prompt_parts: List[str] = []
        if system_message:
            prompt_parts.append(f"[system] {system_message}")
        for m in messages:
            prompt_parts.append(f"[{m.get('role','')}] {m.get('content','')}")
        prompt_text = "\n".join(prompt_parts)

        last_content = str(messages[-1].get("content", "")) if messages else ""

        rnd = self._make_rng(
            model=model,
            system_message=system_message,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop_sequences,
            ctx=ctx,
        )

        base_tokens = self._build_base_tokens(last_content=last_content, model=model)
        gen_tokens = self._apply_temperature(base_tokens, temperature=temperature, rnd=rnd)

        final_tokens, cut_by_max, cut_by_stop = self._apply_max_and_stops(
            gen_tokens,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
        )

        completion_text = _join_tokens(final_tokens)

        finish_reason = "stop"
        if cut_by_max and not cut_by_stop:
            finish_reason = "length"

        return prompt_text, completion_text, finish_reason

    def _maybe_simulate_failure(
        self,
        *,
        messages: List[Mapping[str, Any]],
        model: Optional[str],
        system_message: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        frequency_penalty: Optional[float],
        presence_penalty: Optional[float],
        stop_sequences: Optional[List[str]],
        ctx: Optional[LLMContext],
    ) -> None:
        """
        Optional failure injection for demos/tests. By default failure_rate=0.0
        so conformance runs are stable.
        """
        if self.failure_rate <= 0.0:
            return

        rnd = self._make_rng(
            model=model,
            system_message=system_message,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop_sequences,
            ctx=ctx,
        )

        last_content = str(messages[-1].get("content", "")) if messages else ""
        if rnd.random() < self.failure_rate:
            if "overload" in last_content.lower():
                raise Unavailable("Mocked service overload", retry_after_ms=2000)
            raise ResourceExhausted("Mocked rate limit", retry_after_ms=1000)

    # -----------------------------------------------------------------------
    # Completion
    # -----------------------------------------------------------------------
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
        """Pretend to complete a chat turn with deterministic behavior."""
        # Role validation (beyond base schema checks)
        self._validate_roles(messages)

        # May raise Unavailable/ResourceExhausted for demo purposes.
        self._maybe_simulate_failure(
            messages=messages,
            model=model,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop_sequences,
            ctx=ctx,
        )

        prompt_text, completion_text, finish_reason = self._plan_completion_text(
            messages=messages,
            model=model,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop_sequences,
            ctx=ctx,
        )

        # Simulate processing delay
        await asyncio.sleep(0.03)

        usage = _approx_usage(prompt_text, completion_text)

        return LLMCompletion(
            text=completion_text,
            model=model or "mock-model",
            model_family="mock",
            usage=usage,
            finish_reason=finish_reason,
        )

    # -----------------------------------------------------------------------
    # Streaming
    # -----------------------------------------------------------------------
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
        """Simulate token streaming with progressive usage and final sentinel chunk."""
        # Role validation
        self._validate_roles(messages)

        # Same failure semantics as complete()
        self._maybe_simulate_failure(
            messages=messages,
            model=model,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop_sequences,
            ctx=ctx,
        )

        prompt_text, completion_text, _finish_reason = self._plan_completion_text(
            messages=messages,
            model=model,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop_sequences,
            ctx=ctx,
        )

        final_tokens = _tokenize(completion_text)

        # Emit at least one non-final chunk.
        if not final_tokens:
            final_tokens = ["(mock)"]

        emitted_tokens: List[str] = []
        model_name = model or "mock-model"

        for i, tok in enumerate(final_tokens, start=1):
            emitted_tokens.append(tok)
            partial_text = _join_tokens(emitted_tokens)
            usage_so_far = _approx_usage(prompt_text, partial_text)

            # small pacing delay to simulate streaming
            await asyncio.sleep(0.01)

            yield LLMChunk(
                text=tok + (" " if i < len(final_tokens) else ""),
                is_final=False,
                model=model_name,
                usage_so_far=usage_so_far,
            )

        # Final sentinel chunk: no new text, just terminal marker & final usage.
        final_usage = _approx_usage(prompt_text, _join_tokens(emitted_tokens))
        yield LLMChunk(
            text="",
            is_final=True,
            model=model_name,
            usage_so_far=final_usage,
        )

    # -----------------------------------------------------------------------
    # Stream override to fix base class async iterator issue
    # -----------------------------------------------------------------------
    async def stream(
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
        """Override stream to return async iterator directly instead of coroutine."""
        # Check for deadline expiration before starting
        self._preflight_deadline(ctx)
        
        # Simple validation
        self._validate_messages(messages)
        if not messages:
            raise BadRequest("messages cannot be empty")
            
        caps = await self.capabilities()
        if not caps.supports_streaming:
            raise NotSupported("stream is not supported by this adapter")
            
        # Direct call to _do_stream which returns AsyncIterator correctly
        async for chunk in self._do_stream(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop_sequences,
            model=model,
            system_message=system_message,
            ctx=ctx,
        ):
            yield chunk

    # -----------------------------------------------------------------------
    # Token counting
    # -----------------------------------------------------------------------
    async def _do_count_tokens(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        ctx: Optional[LLMContext] = None,
    ) -> int:
        """Mock token counting with a word-based approximation (+3 overhead)."""
        if not text:
            return 0
        await asyncio.sleep(0.005)
        return len(_tokenize(text)) + 3

    # -----------------------------------------------------------------------
    # Capabilities & Health
    # -----------------------------------------------------------------------
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
        """
        Mock health check.

        Deterministic shape; allow ctx.attrs["health"] to force degraded/error.
        - health="degraded" -> ok=False, status="degraded"
        - health="error"    -> ok=False, status="error"
        - anything else     -> ok=True,  status="healthy"
        """
        status_hint = (ctx and ctx.attrs.get("health")) or "ok"
        if status_hint == "degraded":
            return {"ok": False, "status": "degraded", "server": "mock", "version": "1.0.0"}
        if status_hint == "error":
            return {"ok": False, "status": "error", "server": "mock", "version": "1.0.0"}
        return {"ok": True, "status": "healthy", "server": "mock", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Demo usage (optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover - manual demo only
    async def _demo() -> None:
        if not (make_ctx and print_json and print_kv and box):
            print("Demo helpers not available; run within examples environment.")
            return

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

        # --- Stream example (with stop sequences) ---
        print("\n=== STREAM ===")
        try:
            async for chunk in adapter.stream(
                messages=[{"role": "user", "content": "stream this message and then STOP please"}],
                model="mock-model-pro",
                temperature=0.7,
                stop_sequences=["STOP"],
                ctx=ctx,
            ):
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
