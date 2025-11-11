# SPDX-License-Identifier: Apache-2.0
"""
Demonstrates: CRUD + basic query + capabilities
Expected: prints created IDs, a small rowset, then clean deletes
"""
import asyncio, random, argparse
from examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.graph.graph_base import OperationContext as GraphContext
from examples.common.ctx import make_ctx
from examples.common.printing import box, print_kv, print_json

async def main():
    random.seed(101)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", default="demo-tenant")
    args = ap.parse_args()

    box("ex01_create_read_basics")
    adapter = MockGraphAdapter()
    ctx = make_ctx(GraphContext, tenant=args.tenant)

    caps = await adapter.capabilities()
    print_kv({"server": caps.server, "dialects": caps.dialects})

    v1 = await adapter.create_vertex("User", {"name": "Ada"}, ctx=ctx)
    v2 = await adapter.create_vertex("User", {"name": "Grace"}, ctx=ctx)
    e  = await adapter.create_edge("FOLLOWS", str(v1), str(v2), {"since": 2021}, ctx=ctx)
    print_kv({"v1": v1, "v2": v2, "edge": e})

    rows = await adapter.query(dialect="cypher", text="MATCH (u:User) RETURN u LIMIT 3", ctx=ctx)
    print_json(rows)

    await adapter.delete_edge(str(e), ctx=ctx)
    await adapter.delete_vertex(str(v2), ctx=ctx)
    print_kv({"lesson": "CRUD works; query returns deterministic mock rows"})

if __name__ == "__main__":
    asyncio.run(main())

