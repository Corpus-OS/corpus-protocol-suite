# SPDX-License-Identifier: Apache-2.0
"""
Demonstrates: read-path caches in standalone (caps/schema/query), tenant-scoped keys
Expected: second calls faster; cache_hits counter increments
"""
import asyncio, random, time
from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.examples.common.metrics_console import ConsoleMetrics
from corpus_sdk.graph.graph_base import OperationContext as GraphContext
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.printing import box, print_kv

async def main():
    random.seed(109)
    box("ex09_cache_behavior_standalone")
    metrics = ConsoleMetrics()
    adapter = MockGraphAdapter(mode="standalone", metrics=metrics)

    ctx = make_ctx(GraphContext, tenant="tenant-1")

    t0 = time.perf_counter(); await adapter.capabilities(); t1 = time.perf_counter()
    await adapter.capabilities(); t2 = time.perf_counter()
    print_kv({"caps_first_ms": int((t1 - t0)*1000), "caps_second_ms": int((t2 - t1)*1000)})

    await adapter.get_schema(ctx=ctx)
    await adapter.get_schema(ctx=ctx)

    await adapter.query(dialect="cypher", text="MATCH (n) RETURN n LIMIT 1", ctx=ctx)
    await adapter.query(dialect="cypher", text="MATCH (n) RETURN n LIMIT 1", ctx=ctx)

    print_kv({"lesson": "cache keys are tenant-scoped; repeated reads hit cache"})

if __name__ == "__main__":
    asyncio.run(main())
