# SPDX-License-Identifier: Apache-2.0
"""
Vector ex06 — Standalone read-path cache behavior
Spec refs: §5.3 (Implementation Profiles), §11.6 (Caching Guidance)

Demonstrates:
  • mode='standalone' enables a small read-path cache
  • Second identical query likely hits cache (faster)
"""

import asyncio, time
from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter
from corpus_sdk.vector.vector_base import (
    Vector, UpsertSpec, QuerySpec, NamespaceSpec, OperationContext
)
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.printing import box, print_kv

async def main():
    box("Vector ex06 — Standalone Cache")
    adapter = MockVectorAdapter(failure_rate=0.0, mode="standalone")
    ctx = make_ctx(OperationContext, tenant="ex06-tenant", timeout_ms=10_000)
    ns = "ex06"

    await adapter.create_namespace(NamespaceSpec(namespace=ns, dimensions=3), ctx=ctx)
    await adapter.upsert(UpsertSpec(vectors=[
        Vector(id="x", vector=[1,0,0], metadata={"k":"v"}, namespace=ns),
        Vector(id="y", vector=[0,1,0], metadata={"k":"v"}, namespace=ns),
    ], namespace=ns), ctx=ctx)

    q = QuerySpec(vector=[0.9,0.1,0.0], top_k=1, namespace=ns)
    t0 = time.monotonic()
    await adapter.query(q, ctx=ctx)   # miss, populate
    t1 = time.monotonic()
    await adapter.query(q, ctx=ctx)   # cache hit (standalone)
    t2 = time.monotonic()

    print_kv({"first_ms": round((t1-t0)*1000,2), "second_ms": round((t2-t1)*1000,2)})
    print("\n[lesson] ex06: standalone mode caches queries keyed by vector/params/tenant.")

if __name__ == "__main__":
    asyncio.run(main())

