import asyncio
from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.examples.common.ctx import make_ctx

async def main() -> None:
    adapter = MockLLMAdapter()
    ctx = make_ctx("stream-demo")
    async for chunk in adapter.stream(
        messages=[{"role": "user", "content": "stream this please"}], ctx=ctx
    ):
        print(chunk.text, end="", flush=True)
    print("\n✅ Stream finished.")

if __name__ == "__main__":
    asyncio.run(main())
