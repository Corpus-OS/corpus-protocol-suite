import asyncio
from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.common.errors import TransientNetwork

async def main() -> None:
    adapter = MockLLMAdapter()
    ctx = make_ctx("deadline-demo", deadline_ms=100)
    try:
        await asyncio.wait_for(
            adapter.complete(
                messages=[{"role": "user", "content": "slow operation"}],
                ctx=ctx,
            ),
            timeout=0.05,
        )
    except asyncio.TimeoutError:
        print("⏰ Deadline exceeded (client-side).")
    except TransientNetwork:
        print("⚠️ Simulated transient issue.")
    else:
        print("✅ Completed within deadline.")

if __name__ == "__main__":
    asyncio.run(main())
