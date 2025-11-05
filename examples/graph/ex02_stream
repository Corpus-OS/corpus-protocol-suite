# SPDX-License-Identifier: Apache-2.0
"""
Demonstrates: streaming semantics + early cancel
Expected: consumes first 3 rows then stops cleanly
"""
import asyncio, random, argparse
from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.graph.graph_base import OperationContext as GraphContext
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.printing import box, print_kv

async def main():
    random.seed(102)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", default="demo-tenant")
    ap.add_argument("--dialect", default="cypher")
    args = ap.parse_args()

    box("ex02_stream_rows")
    adapter = MockGraphAdapter()
    ctx = make_ctx(GraphContext, tenant=args.tenant)

    count = 0
    async for row in adapter.stream_query(
        dialect=args.dialect,
        text="MATCH (u:User)-[:READ]->(d:Doc) RETURN u,d LIMIT 5",
        params=None,
        ctx=ctx,
    ):
        count += 1
        print(row)
        if count == 3:
            break

    print_kv({"rows_consumed": count, "lesson": "early consumer cancel closes stream cleanly"})

if __name__ == "__main__":
    asyncio.run(main())

