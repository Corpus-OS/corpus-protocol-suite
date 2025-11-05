# SPDX-License-Identifier: Apache-2.0
"""
Vector ex09 — Metadata delete & Health
Spec refs: §9.3 (delete semantics), §7.6/§9.3 (health normalization), §13 (Observability)

Demonstrates:
  • Bulk delete by metadata filter
  • Health report shape (ok/server/version/namespaces)
"""

import asyncio
from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter
from corpus_sdk.vector.vector_base import (
    Vector, UpsertSpec, DeleteSpec, NamespaceSpec, OperationContext
)
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.printing import box, print_kv, print_json

async def main():
    box("Vector ex09 — Delete by filter & Health")
    adapter = MockVectorAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, tenant="ex09-tenant", timeout_ms=10_000)
    ns = "ex09"

    await adapter.create_namespace(NamespaceSpec(namespace=ns, dimensions=3), ctx=ctx)
    await adapter.upsert(UpsertSpec(vectors=[
        Vector(id="u1", vector=[1,0,0], metadata={"kind":"tmp"}, namespace=ns),
        Vector(id="u2", vector=[0,1,0], metadata={"kind":"perm"}, namespace=ns),
        Vector(id="u3", vector=[0.9,0.1,0], metadata={"kind":"tmp"}, namespace=ns),
    ], namespace=ns), ctx=ctx)

    d = await adapter.delete(DeleteSpec(ids=[], filter={"kind":"tmp"}, namespace=ns), ctx=ctx)
    print_kv({"deleted_tmp": d.deleted_count, "failed": d.failed_count})

    h = await adapter.health(ctx=ctx)
    print_json(h)
    print("\n[lesson] ex09: delete by filter works; health returns stable shape (ok/server/version/namespaces).")

if __name__ == "__main__":
    asyncio.run(main())

