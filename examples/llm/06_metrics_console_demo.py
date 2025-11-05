# SPDX-License-Identifier: Apache-2.0
"""
Example 06 — Metrics Emission Demo

Demonstrates attaching a ConsoleMetrics sink to the MockLLMAdapter.
Each adapter call automatically emits structured metric lines to stdout.
"""

import asyncio
from corpus_sdk.examples.llm.mock_llm_adapter import MockLLMAdapter
from corpus_sdk.examples.common.metrics_console import ConsoleMetrics
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.llm.llm_base import OperationContext
from corpus_sdk.examples.common.printing import box


async def main() -> None:
    box("Example 06 — Metrics Emission Demo")

    adapter = MockLLMAdapter(metrics=ConsoleMetrics())
    ctx = make_ctx(OperationContext, request_id="metrics-demo", tenant="demo")

    _ = await adapter.complete(
        messages=[{"role": "user", "content": "test metrics emission"}],
        ctx=ctx,
    )

    print("\n✅ Metrics emitted to console (look for [OBS]/[CTR]/[GAU] lines).")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
