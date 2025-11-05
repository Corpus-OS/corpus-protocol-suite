# SPDX-License-Identifier: Apache-2.0
"""
Basic completion example using the MockLLMAdapter.

Demonstrates a simple complete() call with a context and prints
the result and token usage.
"""

import asyncio
from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.llm.llm_base import OperationContext
from corpus_sdk.examples.common.printing import print_kv, box


async def main() -> None:
    adapter = MockLLMAdapter()
    ctx = make_ctx(OperationContext, request_id="basic-complete")

    result = await adapter.complete(
        messages=[{"role": "user", "content": "Summarize Corpus SDK"}],
        model="mock-model",
        ctx=ctx,
    )

    box("Basic Completion Example")
    print_kv({
        "Response": result.text,
        "Tokens": result.usage.total_tokens,
        "Model": result.model,
    })


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
