# SPDX-License-Identifier: Apache-2.0
"""
Demonstrates: normalized errors + retry hints
Expected: shows Unavailable/ResourceExhausted/BadRequest surfaces with retry_after_ms/suggestions
"""
import asyncio, random
from corpus_sdk.examples.graph.mock_graph_adapter import MockGraphAdapter
from corpus_sdk.graph.graph_base import OperationContext as GraphContext, AdapterError, BadRequest
from corpus_sdk.examples.common.ctx import make_ctx
from corpus_sdk.examples.common.printing import box, print_kv

async def main():
    random.seed(105)  # deterministic failure sequence
    box("ex05_error_mapping_retryable")
    adapter = MockGraphAdapter(failure_rate=0.7)  # force errors
    ctx = make_ctx(GraphContext, tenant="demo-tenant")

    async def try_call(name, coro):
        try:
            await coro
            print_kv({name: "OK"})
        except AdapterError as e:
            print_kv({
                "op": name, "type": type(e).__name__, "code": e.code,
                "retry_after_ms": e.retry_after_ms, "throttle_scope": getattr(e, "throttle_scope", None)
            })

    # transient failures (expect Unavailable / ResourceExhausted to appear deterministically with seed)
    await try_call("query", adapter.query(dialect="cypher", text="MATCH (n) RETURN n LIMIT 1", ctx=ctx))
    await try_call("create_vertex", adapter.create_vertex("User", {"name": "X"}, ctx=ctx))

    # BadRequest (deterministic)
    try:
        await adapter.create_vertex("", {"name": "bad"}, ctx=ctx)
    except BadRequest as e:
        print_kv({"op": "create_vertex(empty_label)", "type": "BadRequest", "msg": str(e)})

    print_kv({"lesson": "use error fields to decide retry/backoff vs. fix input"})

if __name__ == "__main__":
    asyncio.run(main())

