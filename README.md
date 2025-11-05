# corpus_sdk

A protocol-first, vendor-neutral SDK for interoperable AI/data backends — **LLM**, **Embedding**, **Vector**, and **Graph** — with consistent error taxonomies, capability discovery, SIEM-safe metrics, and deadline propagation. Designed to compose cleanly under an external control plane (router, scheduler, rate limiter) while remaining usable in a lightweight **standalone** mode for development and simple services.

---

## Table of Contents

1. [Why `corpus_sdk`](#why-corpus_sdk)
2. [Features at a Glance](#features-at-a-glance)
3. [Install](#install)
4. [Modes: `thin` vs `standalone`](#modes-thin-vs-standalone)
5. [Core Concepts](#core-concepts)
6. [Quickstart](#quickstart)

   * [Embeddings](#embeddings-quickstart)
   * [LLM](#llm-quickstart)
   * [Vector](#vector-quickstart)
   * [Graph](#graph-quickstart)
7. [Error Taxonomy](#error-taxonomy)
8. [Metrics & Observability](#metrics--observability)
9. [Deadlines & Timeouts](#deadlines--timeouts)
10. [Caching](#caching)
11. [Rate Limiting & Circuit Breaking](#rate-limiting--circuit-breaking)
12. [Capabilities](#capabilities)
13. [Example Adapters](#example-adapters)
14. [Security & Privacy](#security--privacy)
15. [Performance Notes](#performance-notes)
16. [Versioning & Compatibility](#versioning--compatibility)
17. [Testing](#testing)
18. [Troubleshooting](#troubleshooting)
19. [FAQ](#faq)
20. [Contributing](#contributing)
21. [License](#license)
22. [Roadmap](#roadmap)
23. [Appendix](#appendix)

---

## Why `corpus_sdk`

Modern AI platforms juggle multiple LLM, embedding, vector, and graph backends. Each vendor has unique APIs, error schemes, rate limits, and capabilities — making cross-provider integration brittle and costly. `corpus_sdk` provides:

* **Stable, runtime-checkable protocols** across domains.
* **Normalized errors** with retry hints and scopes.
* **SIEM-safe metrics** (low-cardinality; tenant hashed).
* **Deadline propagation** for cancelation & cost control.
* **Two modes**: compose under your own router (**thin**) or use lightweight infra (**standalone**).

---

## Features at a Glance

* Async-first, production-hardened bases that validate inputs and instrument operations.
* Capability discovery to guide routing/planning.
* Strict error taxonomy per domain (Embedding/LLM/Vector/Graph).
* Metrics hooks that never leak PII (tenant hashing).
* Optional in-memory cache (Embedding + LLM complete), rate limiter, and simple circuit breaker in **standalone** mode.
* Everything ships in a **single file per domain** (protocols + base) to keep adoption friction low. You can split them later if desired.

---

## Install

```bash
pip install corpus_sdk
```

* Python ≥ 3.9 recommended.
* No heavy runtime dependencies; bring your own metrics sink or use the provided `NoopMetrics`.

---

## Modes: `thin` vs `standalone`

`corpus_sdk` can operate in two mutually exclusive modes:

* **`thin` (default)**
  All infra hooks are **no-ops**. Use this when you already have a control plane (router/scheduler/limiter/caching/circuit breaker). Prevents **double-stacking** resiliency.

* **`standalone`**
  Enables a small set of helpers: deadline enforcement, a simple circuit breaker, a tiny token-bucket limiter, and an in-memory TTL cache (for deterministic, safe ops). Ideal for demos, dev, and light workloads.

> If you run in **standalone** without a metrics sink, the SDK will emit a warning advising you to provide one before production use.

---

## Core Concepts

### Protocol vs Base

* **Protocol**: A runtime-checkable interface (e.g., `EmbeddingProtocolV1`) that defines *what* an adapter must implement.
* **Base**: A concrete class (e.g., `BaseEmbeddingAdapter`) that implements validation, deadlines, metrics, caching (where safe), and error normalization. You implement the `_do_*` hooks to talk to your provider.

### OperationContext

A small struct propagated across operations:

* `request_id`, `idempotency_key`, `deadline_ms`, `traceparent`, `tenant`, `attrs`.
* Never logged raw; tenants are hashed before recording to metrics.

### Capabilities

Each domain exposes a `*Capabilities` object (e.g., `LLMCapabilities`) that describes supported features, limits (context length, batch size), and flags such as `supports_deadline`, `supports_streaming`, etc.

---

## Quickstart

> **Note**: In all examples, swap `Example*Adapter` with your actual adapter class that inherits the corresponding base and implements `_do_*` hooks.

### Embeddings Quickstart

```python
from corpus_sdk.adapter_sdk.embedding_base import (
    BaseEmbeddingAdapter, EmbedSpec, OperationContext, EmbeddingVector,
    EmbeddingCapabilities, BatchEmbedSpec, BatchEmbedResult
)

class ExampleEmbeddingAdapter(BaseEmbeddingAdapter):
    async def _do_capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            server="example-embeddings",
            version="1.0.0",
            supported_models=("example-embed-001",),
            max_batch_size=128,
            max_text_length=8192,
            supports_normalization=True,
            normalizes_at_source=False,
            supports_deadline=True,
            supports_token_counting=False
        )

    async def _do_embed(self, spec: EmbedSpec, *, ctx: OperationContext | None):
        # Fake embedding for demonstration
        vec = [0.1, 0.2, 0.3]
        return type("EmbedResult", (), {})(
            embedding=EmbeddingVector(vector=vec, text=spec.text, model=spec.model, dimensions=len(vec)),
            model=spec.model,
            text=spec.text,
            tokens_used=None,
            truncated=False
        )

    async def _do_embed_batch(self, spec: BatchEmbedSpec, *, ctx: OperationContext | None):
        vecs = [[0.1, 0.2, 0.3] for _ in spec.texts]
        return BatchEmbedResult(
            embeddings=[
                EmbeddingVector(vector=v, text=t, model=spec.model, dimensions=len(v))
                for v, t in zip(vecs, spec.texts)
            ],
            model=spec.model,
            total_texts=len(spec.texts),
            total_tokens=None,
            failed_texts=[]
        )

    async def _do_count_tokens(self, text: str, model: str, *, ctx: OperationContext | None) -> int:
        return len(text.split())

    async def _do_health(self, *, ctx: OperationContext | None):
        return {"ok": True, "server": "example-embeddings", "version": "1.0.0", "models": {"example-embed-001": "ok"}}

adapter = ExampleEmbeddingAdapter()  # default mode="thin"
ctx = OperationContext(request_id="req-1", tenant="acme")

res = await adapter.embed(EmbedSpec(text="hello world", model="example-embed-001"), ctx=ctx)
print(res.embedding.vector)
```

### LLM Quickstart

```python
from corpus_sdk.adapter_sdk.llm_base import (
    BaseLLMAdapter, OperationContext, LLMCompletion, TokenUsage, LLMCapabilities
)

class ExampleLLMAdapter(BaseLLMAdapter):
    async def _do_capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            server="example-llm",
            version="1.0.0",
            model_family="gpt-4",
            max_context_length=8192,
            supports_streaming=True,
            supports_roles=True,
            supports_json_output=False,
            supports_parallel_tool_calls=False,
            idempotent_writes=False,
            supports_multi_tenant=True,
            supports_system_message=True,
        )

    async def _do_complete(self, **kwargs):
        usage = TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10)
        return LLMCompletion(
            text="Hello from example-llm!",
            model="example-llm-001",
            model_family="gpt-4",
            usage=usage,
            finish_reason="stop"
        )

    async def _do_stream(self, **kwargs):
        # Yield two chunks for demonstration
        from corpus_sdk.adapter_sdk.llm_base import LLMChunk
        yield LLMChunk(text="Hello ", is_final=False)
        yield LLMChunk(text="world!", is_final=True)

    async def _do_count_tokens(self, text: str, *, model: str | None, ctx: OperationContext | None) -> int:
        return len(text.split())

    async def _do_health(self, *, ctx: OperationContext | None):
        return {"ok": True, "server": "example-llm", "version": "1.0.0"}

adapter = ExampleLLMAdapter()
ctx = OperationContext(request_id="req-2", tenant="acme")

resp = await adapter.complete(messages=[{"role": "user", "content": "Say hi"}], ctx=ctx)
print(resp.text)
```

### Vector Quickstart

```python
from corpus_sdk.adapter_sdk.vector_base import (
    BaseVectorAdapter, VectorCapabilities, QuerySpec, QueryResult, Vector, VectorMatch,
    UpsertSpec, UpsertResult, DeleteSpec, DeleteResult, NamespaceSpec, NamespaceResult, OperationContext, VectorID
)

class ExampleVectorAdapter(BaseVectorAdapter):
    async def _do_capabilities(self) -> VectorCapabilities:
        return VectorCapabilities(server="example-vector", version="1.0.0", max_dimensions=3)

    async def _do_query(self, spec: QuerySpec, *, ctx: OperationContext | None) -> QueryResult:
        v = Vector(id=VectorID("v1"), vector=[0.1, 0.2, 0.3], metadata={"label": "demo"}, namespace=spec.namespace)
        return QueryResult(matches=[VectorMatch(vector=v, score=0.99, distance=0.01)], query_vector=spec.vector, namespace=spec.namespace, total_matches=1)

    async def _do_upsert(self, spec: UpsertSpec, *, ctx: OperationContext | None) -> UpsertResult:
        return UpsertResult(upserted_count=len(spec.vectors), failed_count=0, failures=[])

    async def _do_delete(self, spec: DeleteSpec, *, ctx: OperationContext | None) -> DeleteResult:
        return DeleteResult(deleted_count=len(spec.ids), failed_count=0, failures=[])

    async def _do_create_namespace(self, spec: NamespaceSpec, *, ctx: OperationContext | None) -> NamespaceResult:
        return NamespaceResult(success=True, namespace=spec.namespace, details={"created": True})

    async def _do_delete_namespace(self, namespace: str, *, ctx: OperationContext | None) -> NamespaceResult:
        return NamespaceResult(success=True, namespace=namespace, details={"deleted": True})

    async def _do_health(self, *, ctx: OperationContext | None) -> dict:
        return {"ok": True, "server": "example-vector", "version": "1.0.0", "namespaces": {"default": "ok"}}

adapter = ExampleVectorAdapter()
ctx = OperationContext(request_id="req-3", tenant="acme")

result = await adapter.query(QuerySpec(vector=[0.1, 0.2, 0.3], top_k=1), ctx=ctx)
print(result.matches[0].score)
```

### Graph Quickstart

```python
from corpus_sdk.adapter_sdk.graph_base import (
    BaseGraphAdapter, GraphCapabilities, OperationContext, GraphID, BatchOperations
)

class ExampleGraphAdapter(BaseGraphAdapter):
    async def _do_capabilities(self) -> GraphCapabilities:
        return GraphCapabilities(server="example-graph", version="1.0.0", dialects=("cypher",))

    async def _do_create_vertex(self, label: str, props: dict, *, ctx: OperationContext | None) -> GraphID:
        return GraphID("v-1")

    async def _do_create_edge(self, label: str, from_id: str, to_id: str, props: dict, *, ctx: OperationContext | None) -> GraphID:
        return GraphID("e-1")

    async def _do_delete_vertex(self, vertex_id: str, *, ctx: OperationContext | None) -> None:
        return None

    async def _do_delete_edge(self, edge_id: str, *, ctx: OperationContext | None) -> None:
        return None

    async def _do_query(self, *, dialect: str, text: str, params: dict, ctx: OperationContext | None):
        return [{"ok": True, "dialect": dialect}]

    async def _do_stream_query(self, *, dialect: str, text: str, params: dict, ctx: OperationContext | None):
        yield {"row": 1}
        yield {"row": 2}

    async def _do_bulk_vertices(self, vertices, *, ctx: OperationContext | None):
        return [GraphID(f"v-{i}") for i, _ in enumerate(vertices, 1)]

    async def _do_batch(self, ops, *, ctx: OperationContext | None):
        return [{"ok": True, "type": op["type"]} for op in ops]

    async def _do_get_schema(self, *, ctx: OperationContext | None):
        return {"nodes": ["User"], "edges": ["FOLLOWS"]}

    async def _do_health(self, *, ctx: OperationContext | None):
        return {"status": "ok", "server": "example-graph", "version": "1.0.0", "details": {}}

adapter = ExampleGraphAdapter()
ctx = OperationContext(request_id="req-4", tenant="acme")

vertex_id = await adapter.create_vertex("User", {"name": "Ada"}, ctx=ctx)
print(vertex_id)
```

---

## Error Taxonomy

All domains use normalized, structured exceptions with optional guidance fields:

* `BadRequest`, `AuthError`, `ResourceExhausted`, `TransientNetwork`, `Unavailable`, `NotSupported` (+ domain-specific like `TextTooLong`, `ModelOverloaded`, `DimensionMismatch`, `IndexNotReady`).
* Optional fields: `retry_after_ms`, `throttle_scope`/`resource_scope`, `suggested_*_reduction`, `details`.

This enables consistent handling (e.g., retry budgets, UI messaging) regardless of provider.

---

## Metrics & Observability

* `MetricsSink.observe(component, op, ms, ok, code, extra)` for latencies.
* `MetricsSink.counter(component, name, value, extra)` for counters.
* **Low cardinality only** (no PII). Tenants are hashed (first 12 chars of SHA-256).
* Bases record per-op timing and outcome; adapters can emit additional counters.

---

## Deadlines & Timeouts

* All bases accept `OperationContext.deadline_ms`.
* **Thin**: passes through; **Standalone**: enforced via deadline policy.
* Timeouts map to `DeadlineExceeded` (LLM/Embedding) or propagate as `Unavailable`/domain error as applicable.
* Streaming ops periodically check deadlines and terminate cleanly.

---

## Caching

* **Embeddings**: deterministic key includes `(model, normalize, tokenizer/version if present, text hash)`.
* **LLM (complete only)**: key includes `(model, system hash, messages hash, params like temperature/top_p/penalties/max_tokens/stop_sequences)`.
* **Vectors/Graph**: cache is not applied at the base (generally backend/router concern).
* **Thin**: cache no-op; **Standalone**: in-mem TTL cache (short TTL).

---

## Rate Limiting & Circuit Breaking

* Minimal interfaces allow plugging in enterprise infra.
* **Thin**: no-op.
* **Standalone**: simple token bucket + simple circuit breaker (fail-open/closed per mode semantics) to protect demos and small services.

---

## Capabilities

Each adapter declares capabilities for routing/planning:

* Embeddings: models, max text length, batch size, normalization flags, token counting support, deadline support.
* LLM: model family, context size, streaming, roles, JSON output, parallel tool calls, deadline support.
* Vector: max dimensions, supported distance metrics, metadata filtering, namespaces, batch sizes.
* Graph: dialects (`cypher`, `opencypher`, `gremlin`, `gql`), schema ops, transactions, streaming, bulk ops.

Routers can preflight requests (e.g., token counts vs context size; batch sizing) based on these.

---

## Example Adapters

* Reference adapters show how to override `_do_*` methods to call a vendor API, translate errors into normalized exceptions, and report minimal usage data.
* You can keep your **production adapters closed-source** while exposing a public example for the community.

---

## Security & Privacy

* No raw tenant IDs in metrics logs — all tenant IDs are hashed client-side.
* No secrets stored in the bases; adapters accept credentials via constructor or environment.
* Cache keys avoid embedding PII (content hashed).
* Multi-tenant isolation is supported through `OperationContext.tenant` and namespace fields where applicable.

---

## Performance Notes

* Async-first design avoids blocking the event loop; keep heavy CPU tasks off the loop.
* Respect `max_batch_size` and context windows from capabilities.
* Use **thin** mode under a robust router to prevent duplicate resiliency layers.
* For vectors, prefer server-side filtering; avoid returning large vectors unless `include_vectors=True`.

---

## Versioning & Compatibility

* Protocols follow SemVer:

  * **Patch**: clarifications; non-breaking.
  * **Minor**: additive fields/capabilities.
  * **Major**: breaking changes.
* Protocol version constants:

  * `EMBEDDING_PROTOCOL_VERSION`
  * `LLM_PROTOCOL_VERSION`
  * `VECTOR_PROTOCOL_VERSION`
  * `GRAPH_PROTOCOL_VERSION`

---

## Testing

* Unit tests for: validation, capability gating, error mapping, deadlines, caching keys.
* Streaming tests for partial yields, cancellations, and deadline mid-stream.
* Integration tests for example adapters in both **thin** and **standalone** modes.
* Property tests for cache key determinism and message hashing.

---

## Troubleshooting

* **Double-stacked resiliency** (timeouts vs rate limits firing twice): ensure adapters run in **thin** mode under your router.
* **Circuit open** in standalone: reduce concurrency or switch to **thin** and move CB to your infra.
* **Cache surprises**: verify normalization flag and all sampling params are included in keys.
* **Health check failures**: inspect adapter-specific `_do_health` and backend reachability.

---

## FAQ

**Q: Should I use `standalone` in production?**
A: It’s intended for development and light workloads. For scalable production, use **thin** and delegate resiliency to your control plane.

**Q: Can I split protocols and bases into separate files?**
A: Yes. We ship them together for convenience. You can refactor the module layout as you see fit.

**Q: How do I add vendor-specific features?**
A: Extend your adapter’s `_do_*` implementations; expose additional configuration through your adapter’s constructor. Keep the protocol surface stable.

**Q: How do I avoid PII in logs/metrics?**
A: Use `OperationContext.tenant` and let the base hash it; avoid logging raw prompts or documents at the adapter layer.

---

## Contributing

* Follow PEP-8/ruff/black; type hints required.
* Include tests for new features; update README where appropriate.
* Maintain low-cardinality metrics; never add PII to `extra` fields.
* Observe SemVer: call out any breaking changes.

---

## License

Apache-2.0. See `LICENSE` file for details. SPDX headers are included at the top of source files.

---

## Roadmap

* Additional optional capability flags (e.g., function/tool calling schemas).
* Reference metrics exporter examples (Prometheus/OpenTelemetry bridge).
* More example adapters (public endpoints for demos).

---

## Appendix

### Error Mapping Cookbook (Examples)

* **HTTP 401/403** → `AuthError` with `details={"endpoint": "...", "hint": "check credential scope"}`
* **HTTP 429** → `ResourceExhausted` with `retry_after_ms` from headers; set `throttle_scope` (`tenant`/`model`).
* **Vendor timeout / canceled** → `DeadlineExceeded` (LLM/Embedding) or `Unavailable` with `details={"kind":"timeout"}`.
* **Context length exceeded** → `BadRequest` or `TextTooLong` (Embedding) with `suggested_*` guidance.

### Cache Key Compositions

* **Embedding**: `embed:{model}:{normalize}:{sha256(text)}` (+ tokenizer/version if applicable).
* **LLM complete**: `llm:complete:{model}:{sha256(system)}:{sha256(messages)}:{temperature}:{top_p}:{freq_pen}:{pres_pen}:{max_tokens}:{sha256(stop_sequences_json)}`.

### Metrics Field Reference (Common)

* `component`: `"embedding" | "llm" | "vector" | "graph"`
* `op`: e.g., `"embed"`, `"complete"`, `"query"`, `"create_vertex"`
* `ms`: latency in milliseconds
* `ok`: boolean
* `code`: `"OK"` or error class name
* `extra`: low-cardinality map; may include `"tenant"`, `"model"`, `"batch_size"`, `"rows"`, `"dialect"`

---

**Tip:** Keep adapters simple; put retries, scheduling, multi-tenant rate limits, cost controls, and circuit breaking in your control plane. Use **thin** mode to ensure the SDK composes cleanly without double-stacking resiliency.
