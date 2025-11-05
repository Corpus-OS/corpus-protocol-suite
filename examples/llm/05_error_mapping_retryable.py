import asyncio
from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.common.errors import ResourceExhausted, TransientNetwork
from corpus_sdk.examples.common.ctx import make_ctx

async def main() -> None:
    adapter = MockLLMAdapter()
    ctx = make_ctx("error-demo")

    try:
        raise ResourceExhausted("Rate limit hit")
    except (ResourceExhausted, TransientNetwork) as e:
        print(f"Retryable error: {e.__class__.__name__}")
    except Exception as e:
        print(f"Non-retryable error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
