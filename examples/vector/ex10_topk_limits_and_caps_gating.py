# SPDX-License-Identifier: Apache-2.0
"""
Vector ex10 — top_k limits & capabilities gating
Spec refs: §9.3 (query.top_k must be >0 and ≤ limits.max_top_k), §6.2 (capabilities)

Demonstrates:
  • top_k validation against capabilities.max_top_k
"""

import asyncio
from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter
from corpus_sdk.vector.vector_base import (
    Vector, UpsertSpec, QuerySpec, NamespaceSpec, OperationContext, BadRequest
)
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.printing import box, print_kv

async def main():
    box("Vector ex10 — top_k gating")
    adapter = MockVectorAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, tenant="ex10-tenant", timeout_ms=10_000)
    ns = "ex10"

    await adapter.create_namespace(NamespaceSpec(namespace=ns, dimensions=3), ctx=ctx)
    await adapter.upsert(UpsertSpec(vectors=[
        Vector(id="1", vector=[1,0,0], namespace=ns),
        Vector(id="2", vector=[0,1,0], namespace=ns),
        Vector(id="3", vector=[0.7,0.7,0], namespace=ns),
    ], namespace=ns), ctx=ctx)

    caps = await adapter.capabilities()
    ok_topk = min(3, caps.max_top_k or 3)
    res = await adapter.query(QuerySpec(vector=[0.8,0.6,0], top_k=ok_topk, namespace=ns), ctx=ctx)
    print_kv({"ok_query_matches": len(res.matches)})

    # Exceed limits to trigger BadRequest per base gating
    try:
        too_high = (caps.max_top_k or 1000) + 1
        await adapter.query(QuerySpec(vector=[0.8,0.6,0], top_k=too_high, namespace=ns), ctx=ctx)
    except BadRequest as e:
        print_kv({"caught": type(e).__name__, "message": str(e)})

    print("\n[lesson] ex10: top_k must be positive and ≤ capabilities.max_top_k when present.")

if __name__ == "__main__":
    asyncio.run(main())
