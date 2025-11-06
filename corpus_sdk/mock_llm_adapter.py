# SPDX-License-Identifier: Apache-2.0
"""
Mock LLM adapter used in Corpus SDK example scripts.

Implements BaseLLMAdapter methods for demonstration purposes only.
Simulates latency, token counting, streaming behavior, stop sequences, and max_tokens.
"""

from __future__ import annotations
import asyncio
import hashlib
import random
from typing import AsyncIterator, List, Mapping, Optional
from dataclasses import dataclass

from corpus_sdk.llm.llm_base import (
    BaseLLMAdapter,
    LLMCompletion,
    LLMChunk,
    TokenUsage,
    LLMCapabilities,
    OperationContext as LLMContext,
)
from corpus_sdk.examples.common.errors import Unavailable, ResourceExhausted
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.printing import print_json, print_kv, box


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


@dataclass
class MockLLMAdapter(BaseLLMAdapter):
    """A mock LLM adapter for protocol demonstrations."""

    name: str = "mock-llm"
    failure_rate: float = 0.1  # 10% chance of simulated failure

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
        # Start from the last user content, echo with a prefix; vary a bit with temperature
        base = last_content.strip() or "ok"
        words = _tokenize(base)
        # add a deterministic flourish so it isn't a pure echo
        suffix = ["(mock)", f"[{model or 'mock-model'}]"]
        gen_tokens = words + suffix

        # temperature: if > 0, randomly duplicate or drop some tokens deterministically
        if (temperature or 0.0) > 0:
            mutated = []
            for t in gen_tokens:
                r = rnd.random()
                if r < min(0.05 * float(temperature), 0.2):
                    # duplicate
                    mutated.extend([t, t])
                elif r < min(0.10 * float(temperature), 0.4):
                    # drop
                    continue
                else:
                    mutated.append(t)
            gen_tokens = mutated or gen_tokens

        # respect max_tokens (cap completion length)
        if max_tokens is not None:
            max_tokens = max(0, int(max_tokens))
            gen_tokens = gen_tokens[:max_tokens]

        # apply stop sequences (truncate on first match)
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

        completion_text = _join_tokens(gen_tokens)

        # Simulate a small processing delay
        await asyncio.sleep(0.03)

        usage = _approx_usage(prompt_text, completion_text)
        return LLMCompletion(
            text=completion_text if completion_text else "",
            model=model or "mock-model",
            model_family="mock",
            usage=usage,
            finish_reason="stop",
        )

    async def _do_stream(
        self,
        *,
        messages: List[Mapping[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        ctx: Optional[LLMContext] = None,
    ) -> AsyncIterator[LLMChunk]:
        """Simulate token streaming with progressive token counts."""
        # Deterministic plan same as complete()
        seed = _stable_seed(
            str(model or "mock-model"),
            str(system_message or ""),
            repr(messages),
            str(max_tokens),
            str(temperature),
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

        base = (last_content or "ok").strip()
        gen_tokens = _tokenize(base) + ["(stream)", f"[{model or 'mock-model'}]"]

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

        if max_tokens is not None:
            gen_tokens = gen_tokens[: max(0, int(max_tokens))]

        # Stream one token at a time; keep progressive usage accurate
        emitted: List[str] = []
        for i, tok in enumerate(gen_tokens, start=1):
            emitted.append(tok)
            partial = _join_tokens(emitted)
            usage = _approx_usage(prompt_text, partial)
            await asyncio.sleep(0.015)
            yield LLMChunk(text=tok + (" " if i < len(gen_tokens) else ""), usage_so_far=usage, is_final=False)

        # final sentinel chunk (no extra text but includes final metadata)
        final_text = _join_tokens(emitted)
        final_usage = _approx_usage(prompt_text, final_text)
        yield LLMChunk(
            text="",
            is_final=True,
            model=model or "mock-model",
            usage_so_far=final_usage,
        )

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
        """Mock health check with occasional failures."""
        # Keep this simple and stable; callers just need a shape
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
            print_kv({"Output": result.text})
            print_json(result.usage.__dict__)
        except Exception as e:
            print_kv({"Error": str(e), "Type": type(e).__name__})

        # --- Stream example ---
        print("\n=== STREAM ===")
        try:
            async for chunk in adapter.stream(
                messages=[{"role": "user", "content": "stream this message"}],
                model="mock-model-pro",
                temperature=0.7,
                ctx=ctx,
            ):
                print(chunk.text, end="", flush=True)
            print("\n[done]")
        except Exception as e:
            print(f"\nStream error: {e}")

        # --- Token counting example ---
        print("\n=== TOKEN COUNTING ===")
        try:
            count = await adapter.count_tokens("This is a test sentence", ctx=ctx)
            print_kv({"Text": "This is a test sentence", "Tokens": count})
        except Exception as e:
            print_kv({"Error": str(e)})

    asyncio.run(_demo())
