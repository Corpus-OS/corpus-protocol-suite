# SPDX-License-Identifier: Apache-2.0
"""
Example 04 — Standalone Resilience with Retry

Demonstrates using the generic retry_async helper to make an operation resilient
to transient errors (e.g., Unavailable or ResourceExhausted) from the mock adapter.
"""

import asyncio
from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.retry import retry_async, RetryPolicy
from corpus_sdk.llm.llm_base import OperationContext
from corpus_sdk.examples.common.printing import box, print_kv


async def main() -> None:
    box("Example 04 — Standalone Resilience with Retry")

    adapter = MockLLMAdapter(failure_rate=0.3)  # Simulated 30% transient failure rate
    ctx = make_ctx(OperationContext, request_id="resilient-demo", tenant="demo")

    policy = RetryPolicy(max_attempts=4, base_ms=200, max_ms=2000)

    async def run_completion():
        return await adapter.complete(
            messages=[{"role": "user", "content": "retry if transient"}],
            model="mock-model",
            ctx=ctx,
        )

    try:
        # Execute with retries on retryable errors
        result = await retry_async(run_completion, policy=policy)
        print_kv({
            "Final Status": "Success",
            "Response": result.text,
            "Total Tokens": result.usage.total_tokens,
        })
    except Exception as e:
        print_kv({
            "Final Status": "Failed",
            "Error": str(e),
            "Type": type(e).__name__,
        })


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
