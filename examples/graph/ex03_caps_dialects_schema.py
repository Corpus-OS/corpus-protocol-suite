# SPDX-License-Identifier: Apache-2.0
"""
Demonstrates: capabilities, dialect gating, schema retrieval
Expected: prints supported dialects, schema; NotSupported for gremlin
"""
import asyncio, random
from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.graph.graph_base import OperationContext as GraphContext, NotSupported
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.printing import box, print_kv, print_json

async def main():
    random.seed(103)
    box("ex03_caps_dialects_schema")
    adapter = MockGraphAdapter()
    ctx = make_ctx(GraphContext, tenant="demo-tenant")

    caps = await adapter.capabilities()
    print_kv({"dialects": caps.dialects, "supports_schema_ops": caps.supports_schema_ops})
    schema = await adapter.get_schema(ctx=ctx)
    print_json(schema)

    try:
        await adapter.query(dialect="gremlin", text="g.V().limit(1)", ctx=ctx)
    except NotSupported as e:
        print_kv({"expected_not_supported": "✓", "err": str(e)})
    print_kv({"lesson": "gate queries to adapter-supported dialects"})

if __name__ == "__main__":
    asyncio.run(main())
