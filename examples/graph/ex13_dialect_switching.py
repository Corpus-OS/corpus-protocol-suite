# SPDX-License-Identifier: Apache-2.0
"""
Demonstrates: dialect selection based on capabilities
Expected: picks a supported dialect, runs a tiny query
"""
import asyncio, random
from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.graph.graph_base import OperationContext as GraphContext
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.printing import box, print_kv

async def main():
    random.seed(113)
    box("ex13_dialect_switching")
    adapter = MockGraphAdapter()
    ctx = make_ctx(GraphContext, tenant="demo-tenant")

    caps = await adapter.capabilities()
    dialect = "cypher" if "cypher" in caps.dialects else caps.dialects[0]
    await adapter.query(dialect=dialect, text="MATCH (n) RETURN n LIMIT 1", ctx=ctx)
    print_kv({"chosen_dialect": dialect, "lesson": "discover then select supported dialect"})

if __name__ == "__main__":
    asyncio.run(main())
