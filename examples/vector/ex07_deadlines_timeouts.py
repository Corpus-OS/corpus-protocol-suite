# SPDX-License-Identifier: Apache-2.0
"""
Vector ex07 — Deadlines & Timeouts
Spec refs: §6.1 (Operation Context deadlines), §12.4 (DeadlineExceeded)

Demonstrates:
  • Pre-expired deadline fails fast (preflight)
  • Tight deadline enforced by SimpleDeadline (standalone mode)
"""

import asyncio, time
from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter
from corpus_sdk.vector.vector_base import (
    Vector, UpsertSpec, QuerySpec, NamespaceSpec, OperationContext, DeadlineExceeded
)
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.printing import box, print_kv

async def main():
    box("Vector ex07 — Deadlines")
    adapter = MockVectorAdapter(failure_rate=0.0, mode="standalone")
    ns = "ex07"

    # Pre-expired
    pre_ctx = make_ctx(OperationContext, tenant="ex07-tenant", deadline_ms=int(time.time()*1000) - 1)
    try:
        await adapter.create_namespace(NamespaceSpec(namespace=ns, dimensions=3), ctx=pre_ctx)
    except DeadlineExceeded as e:
        print_kv({"preflight_deadline": "EXPECTED", "error": type(e).__name__})

    # Tight budget during op
    ctx_ok = make_ctx(OperationContext, tenant="ex07-tenant", timeout_ms=1000)
    await adapter.create_namespace(NamespaceSpec(namespace=ns, dimensions=3), ctx=ctx_ok)
    await adapter.upsert(UpsertSpec(vectors=[Vector(id="a", vector=[1,0,0], namespace=ns)], namespace=ns), ctx=ctx_ok)

    ctx_tight = make_ctx(OperationContext, tenant="ex07-tenant", timeout_ms=5)  # ~5ms budget
    await asyncio.sleep(0.01)  # burn budget
    try:
        await adapter.query(QuerySpec(vector=[0.9,0.1,0.0], top_k=1, namespace=ns), ctx=ctx_tight)
    except DeadlineExceeded as e:
        print_kv({"timeout_during_query": "EXPECTED", "error": type(e).__name__})

    print("\n[lesson] ex07: fail-fast on expired budgets; strict enforcement in standalone mode.")

if __name__ == "__main__":
    asyncio.run(main())

