# SPDX-License-Identifier: Apache-2.0
"""
Example 07 — Deadlines and Timeouts

Demonstrates how to use OperationContext deadlines together with asyncio
timeouts to enforce request budgets and simulate client-side cancellation.
"""

import asyncio
from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.errors import TransientNetwork
from corpus_sdk.llm.llm_base import OperationContext
from corpus_sdk.examples.common.printing import box


async def main() -> None:
    box("Example 07 — Deadlines and Timeouts")

    adapter = MockLLMAdapter()
    # Set a very short deadline to simulate client timeout
    ctx = make_ctx(OperationContext, request_id="deadline-demo", tenant="demo", deadline_ms=100)

    try:
        await asyncio.wait_for(
            adapter.complete(
                messages=[{"role": "user", "content": "slow operation"}],
                ctx=ctx,
            ),
            timeout=0.05,  # asyncio-level timeout (client enforced)
        )
    except asyncio.TimeoutError:
        print("⏰ Deadline exceeded (client-side).")
    except TransientNetwork:
        print("⚠️ Simulated transient issue.")
    else:
        print("✅ Completed within deadline.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
