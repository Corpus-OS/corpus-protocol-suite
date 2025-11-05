# SPDX-License-Identifier: Apache-2.0
"""
Demonstrates: bulk_vertices + batch ops + hitting max_batch_ops
Expected: creates a small bulk set; large bulk raises BadRequest with suggestion; batch mixes results
"""
import asyncio, random
from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.graph.graph_base import OperationContext as GraphContext, BatchOperations, BadRequest
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.printing import box, print_kv, print_json

async def main():
    random.seed(108)
    box("ex08_bulk_and_batch")
    adapter = MockGraphAdapter()
    ctx = make_ctx(GraphContext, tenant="demo-tenant")

    verts = [("Doc", {"id": f"d{i}"}) for i in range(20)]
    ids = await adapter.bulk_vertices(verts, ctx=ctx)
    print_kv({"bulk_created": len(ids)})

    try:
        big = [("User", {"i": i}) for i in range(2000)]
        await adapter.bulk_vertices(big, ctx=ctx)
    except BadRequest as e:
        print_kv({"expected_batch_limit": "✓", "msg": str(e)})

    ops = [
        BatchOperations.create_vertex_op("User", {"name": "Ada"}),
        BatchOperations.create_edge_op("READ", "v:User:1", "v:Doc:1", {}),
        {"type": "unknown_op"},
    ]
    res = await adapter.batch(ops, ctx=ctx)
    print_json(res)
    print_kv({"lesson": "respect max_batch_ops; batch returns per-op results"})

if __name__ == "__main__":
    asyncio.run(main())

