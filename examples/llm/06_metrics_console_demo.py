import asyncio
from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.examples.common.metrics_console import ConsoleMetricsSink
from corpus_sdk.examples.common.ctx import make_ctx

async def main() -> None:
    adapter = MockLLMAdapter(metrics=ConsoleMetricsSink())
    ctx = make_ctx("metrics-demo")
    _ = await adapter.complete(
        messages=[{"role": "user", "content": "test metrics"}],
        ctx=ctx,
    )
    print("✅ Metrics emitted to console.")

if __name__ == "__main__":
    asyncio.run(main())
