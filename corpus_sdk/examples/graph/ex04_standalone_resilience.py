# SPDX-License-Identifier: Apache-2.0
"""
Demonstrates: mode='standalone' (deadline policy, limiter, breaker, read-path caches)
Expected: first query normal; second hits cache; metrics visible if ConsoleMetrics is used
"""
import asyncio, random
from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.graph.graph_base import OperationContext as GraphContext
from corpus_sdk.examples.common.metrics_console import ConsoleMetrics
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.printing import box, print_kv

async def main():
    random.seed(104)
    box("ex04_standalone_resilience")
    metrics = ConsoleMetrics()
    adapter = MockGraphAdapter(failure_rate=0.25, mode="standalone", metrics=metrics)
    ctx = make_ctx(GraphContext, tenant="demo-tenant")

    rows1 = await adapter.query(dialect="cypher", text="MATCH (n) RETURN n LIMIT 2", ctx=ctx)
    rows2 = await adapter.query(dialect="cypher", text="MATCH (n) RETURN n LIMIT 2", ctx=ctx)
    print_kv({"rows_first": len(rows1), "rows_second": len(rows2), "lesson": "standalone caches read-path ops"})

if __name__ == "__main__":
    asyncio.run(main())

