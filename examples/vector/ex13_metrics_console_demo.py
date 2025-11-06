# SPDX-License-Identifier: Apache-2.0
"""
Vector — Metrics Console Demo (ex06)

What this shows
---------------
• Emits SIEM-safe metrics for vector ops via ConsoleMetrics
  (Spec §6.4 Observability Interfaces, §13.1–13.3 Observability & Tracing)
• Uses standalone mode so the base adapter enforces deadline, limiter, breaker,
  and small read-path cache for query() (Spec §5.3 Implementation Profiles)
• Demonstrates cache hit on repeated query (see printed metrics counters)

Expected console highlights
---------------------------
• observe() lines for: create_namespace, upsert, query (x2), health
• counter() lines such as: vectors_upserted, queries, cache_hits (≥1 on 2nd query)
• No raw tenant ID in metrics (hashed only) (Spec §6.1 tenant privacy, §15 Privacy)

References
----------
• Vector operations contract: Spec §9.3
• Error taxonomy mapping (when failures occur): Spec §12.4
"""

from __future__ import annotations
import asyncio
import random

# Consistent shared helpers
from corpus_sdk.examples.common.ctx import make_ctx  # generic context helpers
from corpus_sdk.examples.common.printing import box, print_kv
from corpus_sdk.examples.common.metrics_console import ConsoleMetrics

# Vector SDK imports
from corpus_sdk.vector.vector_base import (
    OperationContext as VectorContext,
    QuerySpec,
    UpsertSpec,
    NamespaceSpec,
    Vector,
)
from corpus_sdk.examples.vector.mock_vector_adapter import MockVectorAdapter


async def main() -> None:
    random.seed(42)  # deterministic demo output
    box("ex06_metrics_console_demo")

    # Metrics sink (canonical name) + standalone mode for local policies (Spec §5.3, §13)
    metrics = ConsoleMetrics()
    adapter = MockVectorAdapter(metrics=metrics, mode="standalone", failure_rate=0.0)

    # Build an op context with a short but comfortable budget (Spec §6.1)
    ctx = make_ctx(VectorContext, tenant="demo-metrics", timeout_ms=5_000)

    # 1) Create a namespace (Spec §9.3 create_namespace)
    ns = "demo.vec"
    await adapter.create_namespace(
        NamespaceSpec(namespace=ns, dimensions=3, distance_metric="cosine"), ctx=ctx
    )
    print_kv({"namespace": ns, "dimensions": 3, "metric": "cosine"})

    # 2) Upsert a few vectors (Spec §9.3 upsert)
    vecs = [
        Vector(id="a", vector=[1.0, 0.0, 0.0], metadata={"label": "alpha"}, namespace=ns),
        Vector(id="b", vector=[0.0, 1.0, 0.0], metadata={"label": "beta"}, namespace=ns),
        Vector(id="c", vector=[0.7, 0.7, 0.0], metadata={"label": "gamma"}, namespace=ns),
    ]
    up_res = await adapter.upsert(UpsertSpec(vectors=vecs, namespace=ns), ctx=ctx)
    print_kv({"upserted": up_res.upserted_count, "failed": up_res.failed_count})

    # 3) Query twice to demonstrate cache hit in standalone mode (Spec §5.3 caching guidance)
    q = QuerySpec(
        vector=[0.8, 0.6, 0.0],
        top_k=2,
        namespace=ns,
        include_metadata=True,   # OK to include; vectors remain omitted by default
        include_vectors=False,   # privacy-friendly default (Spec §9.3 semantics)
    )

    res1 = await adapter.query(q, ctx=ctx)
    print_kv({
        "query_1_total_matches": res1.total_matches,
        "query_1_top_ids": [m.vector.id for m in res1.matches],
    })

    # Second identical query should be served from the small TTL cache (see metrics: cache_hits)
    res2 = await adapter.query(q, ctx=ctx)
    print_kv({
        "query_2_total_matches": res2.total_matches,
        "query_2_top_ids": [m.vector.id for m in res2.matches],
        "note": "Second call should increment cache_hits counter",
    })

    # 4) Health probe (Spec §9.3 health + Spec §7.6/§9 health normalization)
    h = await adapter.health(ctx=ctx)
    print_kv({"health_ok": h.get("ok"), "server": h.get("server"), "version": h.get("version")})

    # Lesson learned line (consistent across examples)
    print_kv({"lesson": "Metrics emitted for all ops; cache hit observed on repeated query"})


if __name__ == "__main__":
    asyncio.run(main())

