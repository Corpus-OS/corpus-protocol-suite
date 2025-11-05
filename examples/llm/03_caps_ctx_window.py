import asyncio
from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter

async def main() -> None:
    adapter = MockLLMAdapter()
    caps = await adapter.capabilities()
    print("Discovered model capabilities:")
    for m in caps["models"]:
        print(f"- {m['name']} (window={m['context_window']})")

if __name__ == "__main__":
    asyncio.run(main())
