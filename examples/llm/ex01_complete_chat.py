# SPDX-License-Identifier: Apache-2.0
"""
Example 01 — Basic LLM Completion

Demonstrates a simple completion request using the MockLLMAdapter.
Shows how to create an OperationContext, run a completion, and
display the model response and token usage.
"""

import asyncio

from examples.llm.mock_llm_adapter import MockLLMAdapter
from examples.common.ctx import make_ctx
from examples.common.printing import box, print_kv
from corpus_sdk.llm.llm_base import OperationContext


async def main() -> None:
    box("Example 01 — Basic LLM Completion")

    # Initialize the mock adapter (no network calls)
    adapter = MockLLMAdapter()

    # Create a standardized operation context
    ctx = make_ctx(OperationContext, request_id="example-01", tenant="demo")

    # Run a simple completion call
    result = await adapter.complete(
        messages=[{"role": "user", "content": "Summarize the Corpus SDK"}],
        model="mock-model",
        ctx=ctx,
    )

    # Display results
    print_kv({
        "Model": result.model,
        "Response": result.text,
        "Tokens Used": result.usage.total_tokens,
    })


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
