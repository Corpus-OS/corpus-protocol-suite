# SPDX-License-Identifier: Apache-2.0
"""
Vector ex02 — Upsert & Delete
Spec refs: §9.3 (Operations: upsert, delete), §12.5 (Partial Failure Contracts)

Demonstrates:
  • Upsert vectors
  • Delete by IDs and by filter (equality)
  • Result counters and failures
"""

import asyncio
from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter
from corpus_sdk.vector.vector_base import (
    Vector, UpsertSpec, DeleteSpec, NamespaceSpec, OperationContext
)
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.printing import box, print_kv

async def main():
    box("Vector ex02 — Upsert & Delete")
    adapter = MockVectorAdapter(failure_rate=0.0)
    ctx = make_ctx(OperationContext, tenant="ex02-tenant", timeout_ms=10_000)
    ns = "ex02"

    await adapter.create_namespace(NamespaceSpec(namespace=ns, dimensions=3, distance_metric="cosine"), ctx=ctx)

    vecs = [
        Vector(id="1", vector=[1, 0, 0], metadata={"tag": "keep"}, namespace=ns),
        Vector(id="2", vector=[0, 1, 0], metadata={"tag": "delete"}, namespace=ns),
        Vector(id="3", vector=[0, 0.9, 0.1], metadata={"tag": "delete"}, namespace=ns),
    ]
    up = await adapter.upsert(UpsertSpec(vectors=vecs, namespace=ns), ctx=ctx)
    print_kv({"upserted": up.upserted_count, "failed": up.failed_count})

    # Delete by ID
    d1 = await adapter.delete(DeleteSpec(ids=["2"], namespace=ns), ctx=ctx)
    print_kv({"deleted_by_id": d1.deleted_count, "failed_by_id": d1.failed_count})

    # Delete by filter
    d2 = await adapter.delete(DeleteSpec(ids=[], filter={"tag": "delete"}, namespace=ns), ctx=ctx)
    print_kv({"deleted_by_filter": d2.deleted_count, "failed_by_filter": d2.failed_count})

    print("\n[lesson] ex02: deletes support IDs or equality filters; results report counts and failures.")

if __name__ == "__main__":
    asyncio.run(main())

