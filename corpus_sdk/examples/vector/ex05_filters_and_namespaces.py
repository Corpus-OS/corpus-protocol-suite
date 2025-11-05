# SPDX-License-Identifier: Apache-2.0
"""
Vector ex05 — Filtering & Include Flags
Spec refs: §9.3 (query semantics: pre-search filtering), §9.2 (QuerySpec flags)

Demonstrates:
  • Metadata equality filter
  • include_metadata / include_vectors toggles
"""

import asyncio
from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter
from corpus_sdk.vector.vector_base import (
    Vector, UpsertSpec, QuerySpec, NamespaceSpec, OperationContext
)
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.printing import box, print_kv

async def main():
    box("Vector ex05 — Filters & Flags")
    adapter = MockVectorAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, tenant="ex05-tenant", timeout_ms=10_000)
    ns = "ex05"

    await adapter.create_namespace(NamespaceSpec(namespace=ns, dimensions=3), ctx=ctx)

    vecs = [
        Vector(id="A", vector=[1,0,0], metadata={"lang": "en", "tier": "gold"}, namespace=ns),
        Vector(id="B", vector=[0,1,0], metadata={"lang": "fr", "tier": "silver"}, namespace=ns),
        Vector(id="C", vector=[0.7,0.7,0], metadata={"lang": "en", "tier": "silver"}, namespace=ns),
    ]
    await adapter.upsert(UpsertSpec(vectors=vecs, namespace=ns), ctx=ctx)

    spec = QuerySpec(
        vector=[0.7,0.6,0.0],
        top_k=3,
        namespace=ns,
        filter={"lang":"en"},
        include_metadata=False,
        include_vectors=False
    )
    res = await adapter.query(spec, ctx=ctx)
    print_kv({"matches": len(res.matches), "include_vectors": False, "include_metadata": False})
    for m in res.matches:
        print(f"  id={m.vector.id} vector_len={len(m.vector.vector)} meta={m.vector.metadata}")

    print("\n[lesson] ex05: filters narrow candidates; flags suppress raw vector/metadata in results.")

if __name__ == "__main__":
    asyncio.run(main())

