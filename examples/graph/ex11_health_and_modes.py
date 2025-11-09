# SPDX-License-Identifier: Apache-2.0
"""
Demonstrates: health surface & simple readiness logic
Expected: prints status + readiness decision
"""
import asyncio, random
from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.graph.graph_base import OperationContext as GraphContext, HealthStatus
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.printing import box, print_kv

def readiness(status: str) -> str:
    if status == HealthStatus.OK: return "ready"
    if status == HealthStatus.DEGRADED: return "ready_with_caution"
    return "not_ready"

async def main():
    random.seed(111)
    box("ex11_health_and_modes")
    adapter = MockGraphAdapter()
    ctx = make_ctx(GraphContext, tenant="demo-tenant")
    h = await adapter.health(ctx=ctx)
    print_kv({"status": h["status"], "decision": readiness(h["status"]), "server": h["server"], "version": h["version"]})
    print_kv({"lesson": "treat DEGRADED as warn-level: keep serving, alert ops"})

if __name__ == "__main__":
    asyncio.run(main())
