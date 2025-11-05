import asyncio
from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.retry import retry_async

async def main() -> None:
    adapter = MockLLMAdapter()
    ctx = make_ctx("standalone")
    
    @retry_async(max_attempts=3, base_delay=0.1)
    async def resilient_complete():
        return await adapter.complete(
            messages=[{"role": "user", "content": "retry if transient"}],
            ctx=ctx,
        )
    
    result = await resilient_complete()
    print(result.text)

if __name__ == "__main__":
    asyncio.run(main())
