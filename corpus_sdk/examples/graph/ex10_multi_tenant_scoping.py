# SPDX-License-Identifier: Apache-2.0
"""
Demonstrates: multi-tenant isolation in caches & metrics
Expected: same query under two tenants → distinct cache keys; metrics show hashed tenant tags
"""
import asyncio, random
from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.examples.common.metrics_console import ConsoleMetrics
from corpus_sdk.graph.graph_base import OperationContext as GraphContext
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.printing import box, print_kv

async def main():
    random.seed(110)
    box("ex10_multi_tenant_scoping")
    adapter = MockGraphAdapter(mode="standalone", metrics=ConsoleMetrics())

    ctx_a = make_ctx(GraphContext, tenant="tenant-A")
    ctx_b = make_ctx(GraphContext, tenant="tenant-B")

    await adapter.query(dialect="cypher", text="MATCH (n) RETURN n LIMIT 1", ctx=ctx_a)
    await adapter.query(dialect="cypher", text="MATCH (n) RETURN n LIMIT 1", ctx=ctx_b)

    print_kv({"lesson": "no PII in metrics; tenant scopes cache keys separately"})

if __name__ == "__main__":
    asyncio.run(main())

