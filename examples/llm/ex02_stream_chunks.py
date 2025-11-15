# SPDX-License-Identifier: Apache-2.0
"""
Example 02 — Streaming LLM Responses

Demonstrates how to stream incremental chunks from the MockLLMAdapter.
Each chunk simulates partial tokens being emitted from a model.
"""

import asyncio
from examples.llm.mock_llm_adapter import MockLLMAdapter
from examples.common.ctx import make_ctx
from corpus_sdk.llm.llm_base import OperationContext
from examples.common.printing import box


async def main() -> None:
    box("Example 02 — Streaming LLM Responses")

    adapter = MockLLMAdapter()
    ctx = make_ctx(OperationContext, request_id="stream-demo", tenant="demo")

    print("Streaming response:\n")

    async for chunk in adapter.stream(
        messages=[{"role": "user", "content": "Stream this response, please"}],
        model="mock-model",
        ctx=ctx,
    ):
        print(chunk.text, end="", flush=True)

    print("\n✅ Stream finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
