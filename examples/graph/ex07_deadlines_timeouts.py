# SPDX-License-Identifier: Apache-2.0
"""
Demonstrates: deadline preflight + timeout during op
Expected: preflight DeadlineExceeded; then timeout during query
"""
import asyncio, random, time
from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.graph.graph_base import OperationContext as GraphContext, DeadlineExceeded
from corpus_sdk.examples.common.printing import box, print_kv

async def main():
    random.seed(107)
    box("ex07_deadlines_timeouts")
    adapter = MockGraphAdapter()
    now_ms = int(time.time() * 1000)

    # Expired at start → preflight fail
    expired_ctx = GraphContext(deadline_ms=now_ms - 100)
    try:
        await adapter.query(dialect="cypher", text="MATCH (n) RETURN n", ctx=expired_ctx)
    except DeadlineExceeded:
        print_kv({"expired_preflight": "EXPECTED ✓"})

    # Timeout during operation (mock query sleeps ~20ms): set tiny positive budget then burn time
    tight_ctx = GraphContext(deadline_ms=int(time.time() * 1000) + 1)
    await asyncio.sleep(0.01)  # consume budget
    try:
        await adapter.query(dialect="cypher", text="MATCH (n) RETURN n", ctx=tight_ctx)
    except DeadlineExceeded:
        print_kv({"timeout_during_op": "EXPECTED ✓"})

    print_kv({"lesson": "always pass ctx.deadline_ms; base enforces preflight + per-op timeouts"})

if __name__ == "__main__":
    asyncio.run(main())
