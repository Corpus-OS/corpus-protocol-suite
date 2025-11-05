# SPDX-License-Identifier: Apache-2.0
"""
Demonstrates: parameter binding (injection-safe shape)
Expected: query runs with 'weird' email safely as a bound param
"""
import asyncio, random
from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.graph.graph_base import OperationContext as GraphContext
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.printing import box, print_kv

async def main():
    random.seed(112)
    box("ex12_parameter_injection_safety")
    adapter = MockGraphAdapter()
    ctx = make_ctx(GraphContext, tenant="demo-tenant")
    await adapter.query(
        dialect="cypher",
        text="MATCH (u:User {email: $email}) RETURN u",
        params={"email": "'; DROP TABLE users; --"},
        ctx=ctx,
    )
    print_kv({"lesson": "always pass params map; let adapter bind safely"})

if __name__ == "__main__":
    asyncio.run(main())
