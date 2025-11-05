# SPDX-License-Identifier: Apache-2.0
"""
Vector ex08 — Rate limiter & Circuit breaker (standalone)
Spec refs: §5.3 (Profiles), §12.3 (Circuit Breaking), §12.2 (Backoff & Jitter)

Demonstrates:
  • Token-bucket limiter gates concurrency
  • SimpleCircuitBreaker opens after repeated failures (simulated overload)
"""

import asyncio, random
from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter
from corpus_sdk.vector.vector_base import (
    Vector, UpsertSpec, QuerySpec, NamespaceSpec, OperationContext, Unavailable
)
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.printing import box, print_kv

async def noisy_query(adapter, ns, i):
    ctx = make_ctx(OperationContext, tenant=f"ex08-tenant", timeout_ms=250)
    try:
        await adapter.query(QuerySpec(vector=[1,0,0], top_k=1, namespace=ns), ctx=ctx)
        return "OK"
    except Exception as e:
        return type(e).__name__

async def main():
    box("Vector ex08 — Limiter & Breaker")
    random.seed(42)
    adapter = MockVectorAdapter(failure_rate=0.35, mode="standalone")
    ns = "ex08"
    ctx_ok = make_ctx(OperationContext, tenant="ex08-tenant", timeout_ms=2000)
    await adapter.create_namespace(NamespaceSpec(namespace=ns, dimensions=3), ctx=ctx_ok)
    await adapter.upsert(UpsertSpec(vectors=[Vector(id="seed", vector=[1,0,0], namespace=ns)], namespace=ns), ctx=ctx_ok)

    results = await asyncio.gather(*(noisy_query(adapter, ns, i) for i in range(30)))
    # Basic summary: how many OK vs errors (breaker should cause Unavailable / CIRCUIT_OPEN eventually)
    ok = sum(1 for r in results if r == "OK")
    errs = len(results) - ok
    print_kv({"ok": ok, "errors": errs, "unique_error_types": sorted(set(r for r in results if r != "OK"))})
    print("\n[lesson] ex08: limiter smooths, breaker opens after consecutive failures.")

if __name__ == "__main__":
    asyncio.run(main())

