# Corpus SDK

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.9+-blue)

Reference implementation of the **Corpus Protocol Suite** — a **wire-first, vendor-neutral** SDK for interoperable AI/data backends: **LLM**, **Embedding**, **Vector**, and **Graph**.

Unlike in-process-only frameworks, Corpus defines **stable wire-level contracts** (ops, envelopes, errors, capabilities) so applications, routers, and providers can interoperate over the network. This SDK implements those protocols for Python with:

- Consistent error taxonomies
- Capability discovery
- SIEM-safe metrics
- Deadline & idempotency propagation
- Async-first, production-ready bases for adapters
- Canonical JSON envelopes and reserved `op` strings aligned with the public spec

Designed to compose cleanly under any external control plane (router, scheduler, rate limiter) while remaining usable in a lightweight **standalone** mode.

> **Open-Core Model**
>
> - The **Corpus Protocol Suite** and this **Corpus SDK** are **fully open source** (Apache-2.0).
> - **Corpus Router** and **official production adapters** are **commercial** offerings built *on top of* the same public protocols. They are optional; any compatible router/control plane can be used.
> - Using Corpus SDK or implementing the Corpus Protocols does **not** lock you into Corpus Router. The protocols are vendor-neutral by design.

---

## 📚 Documentation Layout

This repo ships both the SDK and the protocol documentation.

**Spec (normative):** `docs/spec/`

- `SPECIFICATION.md` – Corpus Protocol Suite specification (all domains, cross-cutting behavior).
- `PROTOCOL.md` – Wire-level envelopes, streaming semantics, canonical `op` registry.
- `ERRORS.md` – Canonical error taxonomy & mapping rules.
- `METRICS.md` – Metrics schema & SIEM-safe observability.
- `SCHEMA.md` – JSON/type shapes.
- `VERSIONING.md` – Semantic versioning & compatibility rules.

**Guides (how-to):** `docs/guides/`

- `QUICK_START.md` – Longer quick start & end-to-end flows.
- `IMPLEMENTATION.md` – How to implement adapters.
- `ADAPTER_RECIPES.md` – Real-world scenarios and multi-cloud workflows.
- `CONFORMANCE_GUIDE.md` – How to run & interpret conformance suites.

**Conformance (testing):** `docs/conformance/`

- `LLM_CONFORMANCE.md`
- `EMBEDDING_CONFORMANCE.md`
- `VECTOR_CONFORMANCE.md`
- `GRAPH_CONFORMANCE.md`
- `SCHEMA_CONFORMANCE.md`
- `BEHAVIORAL_CONFORMANCE.md`
- `CERTIFICATION.md`

Start here (README) + `docs/guides/QUICK_START.md`, then dive into `docs/spec/` and `docs/conformance/` when you need the full details.

---

## Table of Contents

1. [Why Corpus SDK](#why-corpus-sdk)
2. [How Corpus Compares](#how-corpus-compares)
3. [When Not to Use Corpus](#when-not-to-use-corpus)
4. [Features at a Glance](#features-at-a-glance)
5. [Install](#install)
6. [⚡ 5-Minute Quick Start](#-5-minute-quick-start)
7. [Modes: `thin` vs `standalone`](#modes-thin-vs-standalone)
8. [Core Concepts](#core-concepts)
9. [Quickstart](#quickstart)
   - [Embeddings](#embeddings-quickstart)
   - [LLM](#llm-quickstart)
   - [Vector](#vector-quickstart)
   - [Graph](#graph-quickstart)
10. [Error Taxonomy & Observability](#error-taxonomy--observability)
11. [Performance](#performance)
12. [Testing & Conformance](#testing--conformance)
13. [Troubleshooting](#troubleshooting)
14. [Contributing](#contributing)
15. [License & Commercial Options](#license--commercial-options)

---

## Why Corpus SDK

Modern AI platforms juggle multiple LLM, embedding, vector, and graph backends. Each vendor has unique APIs, error schemes, rate limits, and capabilities — making cross-provider integration brittle and costly.

### The Core Problem: AI Infrastructure Chaos

- **Provider Proliferation**: Dozens of LLM providers, vector databases, and graph databases with incompatible APIs  
- **Duplicate Integration**: Enterprises rewriting the same error handling, observability, and resilience patterns for each provider  
- **Vendor Lock-in**: Applications tightly coupled to specific AI infrastructure choices  
- **Operational Complexity**: Inconsistent monitoring, logging, and error handling across AI services  

**Corpus SDK provides:**

- **Stable, runtime-checkable protocols** across domains  
- **Normalized errors** with retry hints and scopes  
- **SIEM-safe metrics** (low-cardinality; tenant hashed)  
- **Deadline propagation** for cancellation & cost control  
- **Two modes**: compose under your own router (**thin**) or use lightweight infra (**standalone**)  
- A **wire-first protocol** that can be implemented by any language/runtime, with this SDK as the reference impl

---

## How Corpus Compares

### Who is this for?

- **For app developers** – Build on **LangChain, LlamaIndex, Semantic Kernel, AutoGen, CrewAI, or MCP** and still talk to your backends through the same **Corpus protocols**. Swap frameworks or providers without rewriting business logic or error handling.

- **For framework maintainers** – Implement one Corpus adapter per protocol (LLM / Vector / Graph / Embedding) and instantly support any backend that passes the Corpus conformance tests. Fewer bespoke integrations, fewer “this provider behaves differently” bugs.

- **For backend vendors** – Implement `llm/v1`, `embedding/v1`, `vector/v1`, or `graph/v1` once, run the open test suite (`docs/conformance/`), and your service “just works” with multiple frameworks and MCP tools. Golden samples + conformance tests give you a clear definition of correctness.

- **For platform / infra teams** – Get unified observability: normalized error codes, deadlines, and metrics across all frameworks and providers. One set of dashboards, alerts, and SLOs that cover LLM, vector, and graph traffic end-to-end.

- **For MCP users** – The Corpus MCP server exposes your protocols as standard MCP tools (LLM, vector search, graph query, etc.). Any MCP client (including ChatGPT) can call into your existing infra with consistent behavior and safety guarantees.

- **For security & compliance** – A shared, SIEM-safe error taxonomy and context model (tenant hashing, attrs) makes it easier to audit, trace, and reason about behavior across multiple services without leaking sensitive identifiers.

- **For OSS contributors** – The repo includes schemas, golden wire messages, and per-protocol test suites (`tests/llm`, `tests/vector`, `tests/graph`, `tests/embedding`) so new backends and frameworks can validate behavior and evolve the standards in the open.

- **For everyone tired of glue code** – Instead of N×M custom integrations between frameworks and providers, you get one stable protocol layer in the middle. Integrate once, interoperate everywhere.

### How Corpus Compares

| Aspect                    | LangChain/LlamaIndex | OpenRouter | MCP                  | **Corpus SDK**                        |
|---------------------------|----------------------|-----------|----------------------|--------------------------------------|
| **Scope**                 | Application framework | LLM unification | Tools & data sources | **AI infrastructure protocols**      |
| **Domains Covered**       | LLM + Tools          | LLM only  | Tools + Data         | **LLM + Vector + Graph + Embedding** |
| **Error Standardization** | Partial              | Limited   | N/A                  | **Comprehensive taxonomy**           |
| **Multi-Provider Routing**| Basic               | Managed service | N/A              | **Protocol for any router**         |
| **Observability**         | Basic                | Limited   | N/A                  | **Built-in metrics + tracing**      |
| **Installation**          | Heavy dependencies   | Service API | Early stage        | **Lightweight, async-first**        |
| **Vendor Neutrality**     | High                 | Service-dependent | High           | **Protocol-first, no lock-in**      |

**When to use each:**

- **LangChain/LlamaIndex**: Building complex AI applications with tool orchestration  
- **OpenRouter**: Quick LLM unification without infrastructure changes  
- **MCP**: Standardizing tools and data sources for AI applications  
- **Corpus SDK**: Standardizing entire AI infrastructure stack with production observability  

### Unified Integration: Frameworks as Corpus Adapters

The key advantage of Corpus’s protocol-first, **wire-level** approach is that **LangChain, LlamaIndex, Semantic Kernel, AutoGen, CrewAI, OpenRouter, and MCP can all be integrated as adapters within the Corpus ecosystem**:

```python
# LangChain as a Corpus LLM adapter
class LangChainLLMAdapter(BaseLLMAdapter):
    async def _do_complete(self, messages, **kwargs):
        # Wrap LangChain LLM with Corpus standardization
        llm = ChatOpenAI(model=kwargs["model"])
        result = await llm.ainvoke(messages)
        return self._normalize_langchain_result(result)

# OpenRouter as a Corpus LLM adapter  
class OpenRouterAdapter(BaseLLMAdapter):
    async def _do_complete(self, messages, **kwargs):
        # Standardize OpenRouter API with Corpus error handling
        response = await self._call_openrouter(messages, kwargs["model"])
        return self._normalize_openrouter_result(response)

# MCP as a Corpus Tools adapter
class MCPToolsAdapter(BaseLLMAdapter):
    async def _do_complete(self, messages, **kwargs):
        # Use MCP servers as tools within Corpus LLM flow
        mcp_tools = await self._get_mcp_tools()
        return await self._complete_with_tools(messages, mcp_tools)
````

**Benefits of this approach:**

* **Standardized observability**: All adapters emit the same metrics and error taxonomy
* **Consistent routing**: Mix and match providers, frameworks, and protocols under one routing layer
* **Production reliability**: All integrations inherit Corpus’s deadline propagation, retry logic, and circuit breaking
* **Vendor neutrality**: Switch between LangChain, LlamaIndex, Semantic Kernel, CrewAI, AutoGen, OpenRouter, or direct providers without changing application code

Instead of choosing one framework, use **Corpus** as the unifying layer that standardizes them all.

---

## When Not to Use Corpus

You probably don’t need `corpus_sdk` or Corpus Router if:

* **You’re single-provider and happy**: One LLM/vector/graph backend, and you’re fine with their SDKs and breaking changes.
* **No governance/compliance pressure**: No per-tenant isolation, budgets, audit trails, or data residency constraints.
* **No cross-domain orchestration**: You’re not coordinating LLM + Vector + Graph + Embedding as a unified substrate.
* **You want infra logic in-app**: You prefer to hard-code routing, retries, backoff, and failover directly.
* **It’s a quick throwaway prototype**: Lock-in, metrics, and resilience aren’t worth thinking about (yet).

If any of these stop being true, `corpus_sdk` is the incremental next step; **Corpus Router** becomes relevant once you need centralized, explainable, multi-provider routing.

---

## Features at a Glance

* **Async-first, production-hardened** bases that validate inputs and instrument operations
* **Capability discovery** to guide routing/planning
* **Strict error taxonomy** per domain (Embedding/LLM/Vector/Graph)
* **Metrics hooks** that never leak PII (tenant hashing)
* **Optional in-memory cache** (Embedding + LLM complete), rate limiter, and simple circuit breaker in **standalone** mode
* **Wire-first protocol design** with canonical JSON envelopes for transport-agnostic interoperability
* **Canonical `op` registry** aligned with the Corpus Protocol Suite for consistent routing and interoperability
* **Lifecycle management** with async context manager support for clean resource cleanup
* Everything ships in **single files per domain** (protocols + base) to keep adoption friction low

---

## Install

```bash
pip install corpus_sdk
```

* Python ≥ 3.9 recommended
* No heavy runtime dependencies; bring your own metrics sink or use the provided `NoopMetrics`.

---

## ⚡ 5-Minute Quick Start

```python
# Simplest possible working example - get started in under 5 minutes
from corpus_sdk.llm.llm_base import BaseLLMAdapter, OperationContext

class QuickAdapter(BaseLLMAdapter):
    async def _do_complete(self, messages, **kwargs):
        return {"text": "Hello from Corpus!", "model": "quick-demo"}

# Use it immediately
adapter = QuickAdapter()
ctx = OperationContext(request_id="test-123")
result = await adapter.complete(messages=[{"role": "user", "content": "Hi"}], ctx=ctx)
print(result.text)  # "Hello from Corpus!"
```

A more complete quick-start with all four protocols lives in `docs/guides/QUICK_START.md`.

---

## Modes: `thin` vs `standalone`

Corpus SDK can operate in two mutually exclusive modes:

* **`thin` (default)**
  All infra hooks are **no-ops**. Use this when you already have a control plane (router/scheduler/limiter/caching/circuit breaker). Prevents **double-stacking** resiliency.

* **`standalone`**
  Enables a small set of helpers:

  * Deadline enforcement
  * Simple circuit breaker
  * Tiny token-bucket limiter
  * In-memory TTL cache (for deterministic, safe ops)

If you run in **standalone** without a metrics sink, the SDK will emit a warning advising you to provide one before production use.

---

## Core Concepts

* **Protocol vs Base** – Protocols define the required behavior; bases implement validation, deadlines, observability, and error normalization. You implement `_do_*` hooks.
* **OperationContext** – Carries `request_id`, `idempotency_key`, `deadline_ms`, `traceparent`, `tenant`, and optional cache hints across all operations.
* **Wire Protocol** – Canonical envelopes (`op`, `ctx`, `args`) and response shapes (`ok`, `code`, `result`) defined in `docs/spec/PROTOCOL.md`.
* **Corpus-Compatible** – Implementations that honor the envelopes, reserved `op` strings, and error taxonomy described in `docs/spec/` and validated by `docs/conformance/`.

---

## Quickstart

> **Note**: In all examples, swap `Example*Adapter` with your actual adapter class that inherits the corresponding base and implements `_do_*` hooks.

### Embeddings Quickstart

```python
from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter, EmbedSpec, OperationContext, EmbeddingVector,
    EmbeddingCapabilities, BatchEmbedSpec, BatchEmbedResult, EmbedResult
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

    async def _do_embed(self, spec: EmbedSpec, *, ctx: OperationContext | None) -> EmbedResult:
        vec = [0.1, 0.2, 0.3]
        return EmbedResult(
            embedding=EmbeddingVector(
                vector=vec,
                text=spec.text,
                model=spec.model,
                dimensions=len(vec)
            ),
            model=spec.model,
            text=spec.text,
            tokens_used=None,
            truncated=False
        )

    async def _do_embed_batch(self, spec: BatchEmbedSpec, *, ctx: OperationContext | None) -> BatchEmbedResult:
        vecs = [[0.1, 0.2, 0.3] for _ in spec.texts]
        return BatchEmbedResult(
            embeddings=[
                EmbeddingVector(
                    vector=v,
                    text=t,
                    model=spec.model,
                    dimensions=len(v)
                )
                for v, t in zip(vecs, spec.texts)
            ],
            model=spec.model,
            total_texts=len(spec.texts),
            total_tokens=None,
            failed_texts=[]
        )

    async def _do_count_tokens(
        self,
        text: str,
        model: str,
        *,
        ctx: OperationContext | None
    ) -> int:
        return len(text.split())

    async def _do_health(self, *, ctx: OperationContext | None) -> dict:
        return {
            "ok": True,
            "server": "example-embeddings",
            "version": "1.0.0",
            "models": {"example-embed-001": "ok"}
        }

# Usage with lifecycle management
async with ExampleEmbeddingAdapter() as adapter:
    ctx = OperationContext(request_id="req-1", tenant="acme")
    res = await adapter.embed(
        EmbedSpec(text="hello world", model="example-embed-001"),
        ctx=ctx
    )
    print(res.embedding.vector)
# Adapter automatically cleaned up
```

### LLM Quickstart

```python
from corpus_sdk.llm.llm_base import (
    BaseLLMAdapter, OperationContext, LLMCompletion,
    TokenUsage, LLMCapabilities, LLMChunk, LLMStreamResult
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

    async def _do_complete(self, messages, model, **kwargs) -> LLMCompletion:
        usage = TokenUsage(
            prompt_tokens=5,
            completion_tokens=5,
            total_tokens=10
        )
        return LLMCompletion(
            text="Hello from example-llm!",
            model=model,
            model_family="gpt-4",
            usage=usage,
            finish_reason="stop"
        )

    async def _do_stream(self, messages, model, **kwargs) -> LLMStreamResult:
        async def generate_chunks():
            yield LLMChunk(text="Hello ", is_final=False)
            yield LLMChunk(text="world!", is_final=True)
        
        return LLMStreamResult(chunks=generate_chunks())

    async def _do_count_tokens(
        self,
        text: str,
        *,
        model: str | None,
        ctx: OperationContext | None
    ) -> int:
        return len(text.split())

    async def _do_health(self, *, ctx: OperationContext | None) -> dict:
        return {
            "ok": True,
            "server": "example-llm",
            "version": "1.0.0"
        }

# Usage with lifecycle management
async with ExampleLLMAdapter() as adapter:
    ctx = OperationContext(request_id="req-2", tenant="acme")
    resp = await adapter.complete(
        messages=[{"role": "user", "content": "Say hi"}],
        model="example-llm-001",
        ctx=ctx
    )
    print(resp.text)
# Adapter automatically cleaned up
```

### Vector Quickstart

```python
from corpus_sdk.vector.vector_base import (
    BaseVectorAdapter, VectorCapabilities, QuerySpec, QueryResult,
    Vector, VectorMatch, UpsertSpec, UpsertResult, DeleteSpec,
    DeleteResult, NamespaceSpec, NamespaceResult, OperationContext, VectorID
)

class ExampleVectorAdapter(BaseVectorAdapter):
    async def _do_capabilities(self) -> VectorCapabilities:
        return VectorCapabilities(
            server="example-vector",
            version="1.0.0",
            max_dimensions=3
        )

    async def _do_query(
        self,
        spec: QuerySpec,
        *,
        ctx: OperationContext | None
    ) -> QueryResult:
        v = Vector(
            id=VectorID("v1"),
            vector=[0.1, 0.2, 0.3],
            metadata={"label": "demo"},
            namespace=spec.namespace
        )
        return QueryResult(
            matches=[VectorMatch(vector=v, score=0.99, distance=0.01)],
            query_vector=spec.vector,
            namespace=spec.namespace,
            total_matches=1
        )

    async def _do_upsert(
        self,
        spec: UpsertSpec,
        *,
        ctx: OperationContext | None
    ) -> UpsertResult:
        return UpsertResult(
            upserted_count=len(spec.vectors),
            failed_count=0,
            failures=[]
        )

    async def _do_delete(
        self,
        spec: DeleteSpec,
        *,
        ctx: OperationContext | None
    ) -> DeleteResult:
        return DeleteResult(
            deleted_count=len(spec.ids),
            failed_count=0,
            failures=[]
        )

    async def _do_create_namespace(
        self,
        spec: NamespaceSpec,
        *,
        ctx: OperationContext | None
    ) -> NamespaceResult:
        return NamespaceResult(
            success=True,
            namespace=spec.namespace,
            details={"created": True}
        )

    async def _do_delete_namespace(
        self,
        namespace: str,
        *,
        ctx: OperationContext | None
    ) -> NamespaceResult:
        return NamespaceResult(
            success=True,
            namespace=namespace,
            details={"deleted": True}
        )

    async def _do_health(self, *, ctx: OperationContext | None) -> dict:
        return {
            "ok": True,
            "server": "example-vector",
            "version": "1.0.0",
            "namespaces": {"default": "ok"}
        }

# Usage
adapter = ExampleVectorAdapter()
ctx = OperationContext(request_id="req-3", tenant="acme")

result = await adapter.query(
    QuerySpec(vector=[0.1, 0.2, 0.3], top_k=1),
    ctx=ctx
)
print(result.matches[0].score)
```

### Graph Quickstart

```python
from corpus_sdk.graph.graph_base import (
    BaseGraphAdapter, GraphCapabilities, GraphQuerySpec,
    UpsertNodesSpec, UpsertEdgesSpec, Node, Edge, GraphID,
    OperationContext, GraphQueryResult, UpsertNodesResult,
    UpsertEdgesResult, DeleteNodesResult, DeleteEdgesResult,
    BulkVerticesResult, BatchResult, SchemaResult
)

class ExampleGraphAdapter(BaseGraphAdapter):
    async def _do_capabilities(self) -> GraphCapabilities:
        return GraphCapabilities(
            server="example-graph",
            version="1.0.0",
            supported_query_dialects=("cypher",),
            supports_stream_query=True,
            supports_bulk_vertices=True,
            supports_batch=True,
            supports_schema=True
        )

    async def _do_query(
        self,
        spec: GraphQuerySpec,
        *,
        ctx: OperationContext | None
    ) -> GraphQueryResult:
        return GraphQueryResult(
            records=[{"id": 1, "name": "Ada"}],
            summary={"rows": 1},
            dialect=spec.dialect,
            namespace=spec.namespace
        )

    async def _do_stream_query(
        self,
        spec: GraphQuerySpec,
        *,
        ctx: OperationContext | None
    ):
        yield GraphQueryResult(records=[{"id": 1}], is_final=False)
        yield GraphQueryResult(records=[{"id": 2}], is_final=True, summary={"rows": 2})

    async def _do_upsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        ctx: OperationContext | None
    ) -> UpsertNodesResult:
        return UpsertNodesResult(
            upserted_count=len(spec.nodes),
            failed_count=0,
            failures=[]
        )

    async def _do_upsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        ctx: OperationContext | None
    ) -> UpsertEdgesResult:
        return UpsertEdgesResult(
            upserted_count=len(spec.edges),
            failed_count=0,
            failures=[]
        )

    async def _do_delete_nodes(
        self, 
        ids: list[GraphID], 
        *, 
        ctx: OperationContext | None
    ) -> DeleteNodesResult:
        return DeleteNodesResult(
            deleted_count=len(ids),
            failed_count=0,
            failures=[]
        )

    async def _do_delete_edges(
        self, 
        ids: list[GraphID], 
        *, 
        ctx: OperationContext | None
    ) -> DeleteEdgesResult:
        return DeleteEdgesResult(
            deleted_count=len(ids),
            failed_count=0,
            failures=[]
        )

    async def _do_bulk_vertices(
        self, 
        cursor: str | None, 
        limit: int, 
        *, 
        ctx: OperationContext | None
    ) -> BulkVerticesResult:
        return BulkVerticesResult(
            nodes=[],
            next_cursor=None,
            has_more=False
        )

    async def _do_batch(
        self, 
        ops: list, 
        *, 
        ctx: OperationContext | None
    ) -> BatchResult:
        return BatchResult(results=[{"ok": True} for _ in ops])

    async def _do_get_schema(self, *, ctx: OperationContext | None) -> SchemaResult:
        return SchemaResult(
            nodes={"User": {"properties": {}}},
            edges={"FOLLOWS": {}},
            metadata={"version": "1.0"}
        )

    async def _do_health(self, *, ctx: OperationContext | None) -> dict:
        return {
            "ok": True,
            "server": "example-graph",
            "version": "1.0.0"
        }

# Usage with lifecycle management
async with ExampleGraphAdapter() as adapter:
    ctx = OperationContext(request_id="req-4", tenant="acme")

    # Create nodes
    result = await adapter.upsert_nodes(
        UpsertNodesSpec(nodes=[
            Node(
                id=GraphID("user:1"),
                labels=("User",),
                properties={"name": "Ada"}
            )
        ]),
        ctx=ctx
    )
    print(f"Upserted {result.upserted_count} nodes")
# Adapter automatically cleaned up
```

Additional “real-world” multi-cloud and RAG scenarios have been moved to `docs/guides/ADAPTER_RECIPES.md` to keep the README focused.

---

## Error Taxonomy & Observability

All domains share a **normalized error taxonomy** (`BadRequest`, `AuthError`, `ResourceExhausted`, `TransientNetwork`, `Unavailable`, `NotSupported`, `DeadlineExceeded`, plus domain-specific ones like `TextTooLong`, `ModelOverloaded`, `DimensionMismatch`, `IndexNotReady`).

See `docs/spec/ERRORS.md` for the full mapping and `docs/spec/METRICS.md` for metrics details.

Errors carry **machine-actionable hints** (e.g., `retry_after_ms`, `throttle_scope`) so routers and control planes can react consistently across providers.

A `MetricsSink` protocol lets you plug in your own metrics backend; the bases:

* Emit one `observe` per operation (or per stream lifecycle).
* Hash tenants before recording.
* Avoid logging prompts, vectors, or raw tenant IDs.

---

## Performance

Performance notes are covered in detail in `docs/spec/SPECIFICATION.md` (§ Performance Characteristics), but at a high level:

* **Overhead** of the bases is typically **<10ms** relative to vendor SDK calls:

  * Validation: <1ms
  * Metrics: <0.1ms
  * Cache lookup (standalone): <0.5ms
* **Async-first** design avoids blocking; designed for high concurrency.
* **Batch operations** (`embed_batch`, vector upserts, graph batch) are preferred for throughput.

Benchmarks and deployment patterns for higher scale live in `docs/guides/IMPLEMENTATION.md` and `docs/guides/ADAPTER_RECIPES.md`.

---

## Testing & Conformance

### One-Command Conformance Testing

See `docs/conformance/CONFORMANCE_GUIDE.md` for full details. Summary:

#### Make targets (from repo root)

```bash
# Test ALL protocols at once (LLM + Vector + Graph + Embedding)
make test-all-conformance

# Test specific protocols
make test-llm-conformance
make test-vector-conformance
make test-graph-conformance
make test-embedding-conformance
```

#### Corpus SDK CLI

> `[project.scripts] corpus-sdk = "corpus_sdk.cli:main"`

```bash
# Show help / usage
corpus-sdk

# Run ALL protocol conformance suites
corpus-sdk test-all-conformance
corpus-sdk verify

# Run specific protocol suites
corpus-sdk test-llm-conformance
corpus-sdk test-vector-conformance
corpus-sdk test-graph-conformance
corpus-sdk test-embedding-conformance

# Filtered verify (run only selected protocols)
corpus-sdk verify -p llm -p vector
corpus-sdk verify -p embedding
```

#### Direct: pytest

```bash
# Run everything with coverage
pytest tests/ -v --cov=corpus_sdk --cov-report=html

# Run specific protocol suites
pytest tests/llm/ -v
pytest tests/vector/ -v
pytest tests/graph/ -v
pytest tests/embedding/ -v
```

#### Corpus Protocol Suite Badge

![LLM Protocol](https://img.shields.io/badge/CorpusLLM%20Protocol-100%25%20Conformant-brightgreen)
![Vector Protocol](https://img.shields.io/badge/CorpusVector%20Protocol-100%25%20Conformant-brightgreen)
![Graph Protocol](https://img.shields.io/badge/CorpusGraph%20Protocol-100%25%20Conformant-brightgreen)
![Embedding Protocol](https://img.shields.io/badge/CorpusEmbedding%20Protocol-100%25%20Conformant-brightgreen)

Requirements for “Corpus-Compatible” certification are in `docs/conformance/CERTIFICATION.md`.

---

## Troubleshooting

### Common Issues

**Problem: Double-stacked resiliency (timeouts/limits firing twice)**

*Solution*: Ensure adapters run in thin mode under your router

*Check*: `mode="thin"` in adapter constructor

**Problem: Circuit breaker opens frequently in standalone mode**

*Solution*: Reduce concurrency or switch to thin mode with external circuit breaker

*Check*: `failure_threshold` and `recovery_after_s` settings

**Problem: Cache returns stale results**

*Solution*: Verify all sampling parameters are included in cache key

*Check*: `cache_ttl_s` setting, normalization flag consistency

**Problem: Health check failures**

*Solution*: Inspect adapter-specific `_do_health` implementation

*Check*: Backend reachability, credentials, network configuration

**Problem: `DeadlineExceeded` on fast operations**

*Solution*: Check `deadline_ms` is absolute epoch time, not relative

*Check*: System clock synchronization (NTP)

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("corpus_sdk").setLevel(logging.DEBUG)
```
---
## FAQ

### General

**Q: Is the SDK fully open source while the router is commercial?**

**A:** Yes. The SDK (protocols + bases + example adapters) is **open source** under Apache-2.0. **Corpus Router** and **official adapters** are **commercial** (managed cloud or on-prem).

**Q: Will you maintain official adapters for major providers (OpenAI, Anthropic, Pinecone, etc.)?**

**A:** Yes. We maintain **closed-source, production-grade adapters** for major providers as part of Corpus Router subscriptions.

**Q: Can Corpus Router run on-premises or is it cloud-only?**

**A:** Both. Corpus Router is available as a **managed cloud** service and as an **on-prem** deployment for regulated/air-gapped environments.

**Q: Do I have to use Corpus Router?**

**A:** No. The SDK composes with any router/control plane. Corpus Router is optional and adheres to the same public protocols.

**Q: Can I split protocols and bases into separate files?**

**A:** Yes. We ship them together for convenience; you can refactor module layout as you see fit.

### **MCP/LangChain/OpenRouter Comparison**

**Q: How does Corpus compare to LangChain/LlamaIndex?**

**A:** LangChain and LlamaIndex are **application-level frameworks** for building AI applications, while Corpus is an **infrastructure protocol** for standardizing backend services. You can use Corpus SDK underneath LangChain/LlamaIndex to get provider-agnostic LLM, embedding, vector, and graph operations with consistent error handling and observability.

**Q: How does Corpus compare to Model Context Protocol (MCP)?**

**A:** MCP focuses on standardizing **tools and data sources** for AI applications, while Corpus standardizes **core AI infrastructure services** (LLM, Vector, Graph, Embedding). They're complementary - you could use MCP for tool integration and Corpus for backend service abstraction.

**Q: How does Corpus compare to OpenRouter?**

**A:** OpenRouter provides a unified API for **LLM providers only**, while Corpus covers **four domains** (LLM, Vector, Graph, Embedding) with standardized error handling, metrics, and capabilities discovery. Corpus is a protocol you can implement anywhere, while OpenRouter is a specific service.

### Technical

**Q: Why async-only?**

**A:** Modern AI workloads require high concurrency. Async-first design prevents blocking the event loop. Sync wrappers can be built on top if needed.

**Q: How do I handle streaming with deadlines?**

**A:** Bases check deadlines periodically during streaming. Set `deadline_ms` in `OperationContext` and the base handles enforcement.

**Q: Can I use my own cache/metrics/limiter?**

**A:** Yes. All infrastructure components are pluggable via Protocol interfaces. Provide your implementations to the base constructor.

**Q: What happens if my adapter raises a non-normalized error?**

**A:** Bases catch unexpected exceptions and record them as `UnhandledException` in metrics. Wrap provider errors in normalized exceptions for proper handling.

**Q: How do I test my adapter?**

**A:** Use the protocol as a contract. Verify your adapter satisfies `isinstance(adapter, ProtocolV1)` and test all `_do_*` method implementations.

---

## Contributing

```bash
git clone https://github.com/corpus/corpus-sdk.git
cd corpus-sdk
pip install -e ".[dev]"
pytest
```

Guidelines:

* **Follow PEP-8** – Use ruff/black.
* **Type hints required** – All public APIs must be typed.
* **Include tests** – New features need corresponding coverage.
* **Update docs** – Especially relevant files under `docs/spec/` or `docs/guides/`.
* **Maintain low-cardinality metrics** – Never add PII to `extra` fields.
* **Observe SemVer** – Call out breaking changes and update `docs/spec/VERSIONING.md`.

We especially welcome **community adapter contributions** (e.g., new LLM/vector/graph backends implemented against the Corpus Protocol Suite).

---

## License & Commercial Options

* License: **Apache-2.0**. See `LICENSE`. SPDX headers are included at the top of source files.
* Implementation of the **Corpus Protocol Suite** is free and encouraged. Independent implementations SHOULD preserve wire compatibility if they refer to themselves as **Corpus-Compatible**. See `docs/conformance/CERTIFICATION.md`.

### SDK vs Platform

| Need                                | Solution                                   | Cost           |
| ----------------------------------- | ------------------------------------------ | -------------- |
| Learning / prototyping              | `corpus_sdk` + example adapters            | **Free (OSS)** |
| Production with your own infra      | `corpus_sdk` + your adapters               | **Free (OSS)** |
| Production with official adapters   | `corpus_sdk` + **Official Adapters**       | **Commercial** |
| Enterprise multi-provider (managed) | `corpus_sdk` + **Corpus Router (Managed)** | **Commercial** |
| Enterprise multi-provider (on-prem) | `corpus_sdk` + **Corpus Router (On-Prem)** | **Commercial** |

**Router details & architecture** live in a separate doc (see `docs/guides/ROUTER_OVERVIEW.md` or corpus.io) to keep this README focused on the SDK and protocols.

**Contact**

* Sales & commercial: `sales@corpus.io`
* Technical & community: `discussions@corpus.io`
* Partnerships: `partners@corpus.io`

---

**Built by the Corpus team** — aiming to make **wire-level AI infrastructure** something you integrate once and then stop thinking about.

```
