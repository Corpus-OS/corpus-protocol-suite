# SPDX-License-Identifier: Apache-2.0
"""
Vector ex01 — Basic query flow
Spec refs: §9.3 (Operations: query), §9.2 (Data Types), §6.1 (Operation Context)

Demonstrates:
  • Namespace creation
  • Upsert of a few vectors
  • Basic similarity query with top_k
  • SIEM-safe context via make_ctx()
"""

import asyncio
from examples.vector.mock_vector_adapter import MockVectorAdapter
from corpus_sdk.vector.vector_base import (
    Vector, QuerySpec, UpsertSpec, NamespaceSpec, OperationContext
)
from examples.common.ctx import make_ctx
from examples.common.printing import box, print_kv

async def main():
    box("Vector ex01 — Basic query")
    adapter = MockVectorAdapter(failure_rate=0.0)  # deterministic
    ctx = make_ctx(OperationContext, tenant="ex01-tenant", timeout_ms=10_000)

    # Create namespace (3 dims, cosine)
    await adapter.create_namespace(NamespaceSpec(namespace="ex01", dimensions=3, distance_metric="cosine"), ctx=ctx)

    # Upsert a few vectors
    vecs = [
        Vector(id="a", vector=[1.0, 0.0, 0.0], metadata={"label": "alpha"}, namespace="ex01"),
        Vector(id="b", vector=[0.0, 1.0, 0.0], metadata={"label": "beta"}, namespace="ex01"),
        Vector(id="c", vector=[0.6, 0.8, 0.0], metadata={"label": "gamma"}, namespace="ex01"),
    ]
    up = await adapter.upsert(UpsertSpec(vectors=vecs, namespace="ex01"), ctx=ctx)
    print_kv({"upserted": up.upserted_count, "failed": up.failed_count})

    # Query (top 2)
    spec = QuerySpec(vector=[0.7, 0.7, 0.0], top_k=2, namespace="ex01")
    res = await adapter.query(spec, ctx=ctx)
    print_kv({"matches": len(res.matches), "total_matches": res.total_matches})
    for m in res.matches:
        print(f"  id={m.vector.id} score={m.score:.3f} dist={m.distance:.3f} meta={m.vector.metadata}")

    print("\n[lesson] ex01: query returns VectorMatch list with scores/distances; context via make_ctx().")

if __name__ == "__main__":
    asyncio.run(main())

