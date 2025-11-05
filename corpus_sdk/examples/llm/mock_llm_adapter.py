# SPDX-License-Identifier: Apache-2.0
"""
Mock LLM adapter used in Corpus SDK example scripts.

Implements BaseLLMAdapter methods for demonstration purposes only.
Simulates latency, token counting, and streaming behavior.
"""

from __future__ import annotations
import asyncio
import random
from typing import AsyncIterator, Optional
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


@dataclass
class MockLLMAdapter(BaseLLMAdapter):
    """A mock LLM adapter for protocol demonstrations."""

    name: str = "mock-llm"
    failure_rate: float = 0.1  # 10% chance of simulated failure

    async def _do_complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        **_: object,
    ) -> LLMCompletion:
        """Pretend to complete a chat turn with occasional simulated failures."""
        # Simulate random failures for demonstration
        if random.random() < self.failure_rate:
            if "overload" in messages[-1]["content"]:
                raise Unavailable("Mocked service overload", retry_after_ms=2000)
            else:
                raise ResourceExhausted("Mocked rate limit", retry_after_ms=1000)
        
        joined = " ".join([m["content"] for m in messages])
        await asyncio.sleep(0.05)  # Simulate processing time
        return LLMCompletion(
            text=f"[mock:{model or 'default'}] response to: {joined}",
            model=model or "mock-model",
            model_family="mock",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="stop",
        )

    async def _do_stream(
        self,
        *,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        **_: object,
    ) -> AsyncIterator[LLMChunk]:
        """Simulate token streaming with progressive token counts."""
        full_text = f"[streamed:{model or 'mock'}] " + messages[-1]["content"]
        words = full_text.split()
        
        for i, word in enumerate(words):
            await asyncio.sleep(0.02)
            yield LLMChunk(
                text=word + " ",
                usage_so_far=TokenUsage(
                    prompt_tokens=10, 
                    completion_tokens=i + 1, 
                    total_tokens=10 + i + 1
                )
            )
        
        yield LLMChunk(
            text="[end]", 
            is_final=True,
            model=model or "mock-model",
            usage_so_far=TokenUsage(prompt_tokens=10, completion_tokens=len(words), total_tokens=10 + len(words))
        )

    async def _do_count_tokens(
        self, text: str, *, model: Optional[str] = None, **_: object
    ) -> int:
        """Mock token counting with word-based approximation."""
        await asyncio.sleep(0.01)  # Simulate tokenizer processing
        return len(text.split()) + 3  # Add overhead for special tokens

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
            supported_models=("mock-model", "mock-model-pro")
        )

    async def _do_health(self, **_: object) -> dict[str, object]:
        """Mock health check with occasional failures."""
        if random.random() < 0.2:  # 20% chance of unhealthy
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
        ctx = make_ctx(LLMContext, tenant="demo")

        # --- Capabilities example ---
        print("\n=== CAPABILITIES ===")
        caps = await adapter.capabilities()
        print_json(caps.__dict__)

        # --- Health check example ---
        print("\n=== HEALTH CHECK ===")
        health = await adapter.health(ctx=ctx)
        print_kv(health)

        # --- Complete example ---
        print("\n=== COMPLETE ===")
        try:
            result = await adapter.complete(
                messages=[{"role": "user", "content": "hello world"}], 
                model="mock-model",
                ctx=ctx
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
                ctx=ctx
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
