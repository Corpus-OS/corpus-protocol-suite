# SPDX-License-Identifier: Apache-2.0
"""
Demonstrates: SIEM-safe metrics (observe/counter) w/ tenant hashing + deadline buckets
Expected: console metrics lines for query + stream, no PII
"""
import asyncio, random
from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.examples.common.metrics_console import ConsoleMetrics
from corpus_sdk.graph.graph_base import OperationContext as GraphContext
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.printing import box, print_kv

async def main():
    random.seed(106)
    box("ex06_metrics_console_demo")
    metrics = ConsoleMetrics()
    adapter = MockGraphAdapter(metrics=metrics)
    ctx = make_ctx(GraphContext, tenant="tenant-a")

    await adapter.query(dialect="cypher", text="MATCH (n) RETURN n LIMIT 2", ctx=ctx)
    async for _ in adapter.stream_query(dialect="cypher", text="MATCH (n) RETURN n LIMIT 3", ctx=ctx):
        pass

    print_kv({"lesson": "metrics include hashed tenant + deadline bucket; never PII"})

if __name__ == "__main__":
    asyncio.run(main())

