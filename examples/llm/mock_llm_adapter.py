from __future__ import annotations
import asyncio
from typing import AsyncIterator, Optional
from dataclasses import dataclass

from corpus_sdk.llm.llm_base import BaseLLMAdapter, LLMCompletion, LLMChunk, TokenUsage
from corpus_sdk.examples.common.ctx import make_ctx


@dataclass
class MockLLMAdapter(BaseLLMAdapter):
    """A mock LLM adapter for protocol demonstrations."""

    name: str = "mock-llm"

    async def _do_complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        **_: object,
    ) -> LLMCompletion:
        """Pretend to complete a chat turn."""
        joined = " ".join([m["content"] for m in messages])
        await asyncio.sleep(0.05)
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
        """Simulate token streaming."""
        full_text = f"[streamed:{model or 'mock'}] " + messages[-1]["content"]
        for word in full_text.split():
            await asyncio.sleep(0.02)
            yield LLMChunk(text=word + " ")
        yield LLMChunk(text="[end]", is_final=True)

    async def _do_count_tokens(
        self, text: str, *, model: Optional[str] = None, **_: object
    ) -> int:
        """Mock token counting."""
        return len(text.split())

    async def _do_capabilities(self) -> dict:
        """Report simple model capabilities."""
        return {
            "models": [
                {
                    "name": "mock-llm-1",
                    "family": "mock",
                    "context_window": 4096,
                    "supports_tools": False,
                }
            ],
            "sampling": {"temperature_range": [0.0, 2.0]},
        }


if __name__ == "__main__":
    async def _demo() -> None:
        adapter = MockLLMAdapter()
        ctx = make_ctx("demo")
        result = await adapter.complete(
            messages=[{"role": "user", "content": "hello"}], ctx=ctx
        )
        print(result.text)

    asyncio.run(_demo())
