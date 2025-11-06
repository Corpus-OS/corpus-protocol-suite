# SPDX-License-Identifier: Apache-2.0
"""
Vector ex11 — Error mapping & retry loop (retryable vs non-retryable)
Spec refs: §12.1 (Retry Semantics), §12.2 (Backoff & Jitter), §12.4 (Error Mapping Table),
           §9.5 (Vector-specific errors)

Demonstrates:
  • Handling retryable errors: ResourceExhausted, Unavailable, TransientNetwork, IndexNotReady
  • Not retrying on non-retryable: BadRequest, NotSupported, DimensionMismatch
  • Honoring retry_after_ms when present
"""

import asyncio, random, time
from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter
from corpus_sdk.vector.vector_base import (
    Vector, UpsertSpec, QuerySpec, NamespaceSpec, OperationContext,
    ResourceExhausted, Unavailable, TransientNetwork, IndexNotReady,
    BadRequest, NotSupported, DimensionMismatch, DeadlineExceeded
)
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.printing import box, print_kv

RETRYABLE = (ResourceExhausted, Unavailable, TransientNetwork, IndexNotReady)
NON_RETRYABLE = (BadRequest, NotSupported, DimensionMismatch)

async def backoff_sleep(attempt, retry_after_ms=None):
    if retry_after_ms:
        await asyncio.sleep(retry_after_ms / 1000.0)
        return
    # full jitter backoff per §12.2
    cap = min(5.0, 0.2 * (2 ** attempt))
    await asyncio.sleep(random.random() * cap)

async def robust_query(adapter, spec, ctx):
    # bounded attempts example
    for attempt in range(5):
        try:
            return await adapter.query(spec, ctx=ctx)
        except DeadlineExceeded:
            # conditionally retryable only if you can extend budget or reduce work (§12.1)
            raise
        except RETRYABLE as e:
            await backoff_sleep(attempt, getattr(e, "retry_after_ms", None))
            continue
        except NON_RETRYABLE:
            raise
    raise Unavailable("exhausted retries")

async def main():
    box("Vector ex11 — Error mapping & retry loop")
    random.seed(1234)

    # Simulate occasional overload & rate-limits
    adapter = MockVectorAdapter(failure_rate=0.25, mode="standalone")
    ns = "ex11"
    ctx = make_ctx(OperationContext, tenant="ex11-tenant", timeout_ms=10_000)

    # Create ns and upsert seed
    await adapter.create_namespace(NamespaceSpec(namespace=ns, dimensions=3), ctx=ctx)
    await adapter.upsert(UpsertSpec(vectors=[Vector(id="seed", vector=[1,0,0], namespace=ns)], namespace=ns), ctx=ctx)

    # Run a robust query with retry policy aligned to §12
    spec = QuerySpec(vector=[1,0,0], top_k=1, namespace=ns)
    try:
        res = await robust_query(adapter, spec, ctx)
        print_kv({"matches": len(res.matches)})
    except Exception as e:
        print_kv({"final_error": type(e).__name__, "message": str(e)})

    # Demonstrate non-retryable DimensionMismatch
    try:
        await adapter.query(QuerySpec(vector=[1,0], top_k=1, namespace=ns), ctx=ctx)
    except DimensionMismatch as e:
        print_kv({"non_retryable": type(e).__name__, "message": str(e)})

    print("\n[lesson] ex11: retry only on §12.1 retryable classes; honor retry_after_ms; stop on non-retryables.")

if __name__ == "__main__":
    asyncio.run(main())

