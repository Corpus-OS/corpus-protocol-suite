import asyncio
from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.examples.common.ctx import make_ctx

async def main() -> None:
    adapter = MockLLMAdapter()
    ctx = make_ctx("basic-complete")
    result = await adapter.complete(
        messages=[{"role": "user", "content": "Summarize Corpus SDK"}],
        model="mock-llm-1",
        ctx=ctx,
    )
    print(f"Response: {result.text}")
    print(f"Usage: {result.usage.total_tokens} tokens")

if __name__ == "__main__":
    asyncio.run(main())
