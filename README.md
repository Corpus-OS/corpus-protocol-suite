# Corpus OS Protocol and SDK

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![LLM Protocol](https://img.shields.io/badge/LLM%20Protocol-100%25%20Conformant-brightgreen)
![Vector Protocol](https://img.shields.io/badge/Vector%20Protocol-100%25%20Conformant-brightgreen)
![Graph Protocol](https://img.shields.io/badge/Graph%20Protocol-100%25%20Conformant-brightgreen)
![Embedding Protocol](https://img.shields.io/badge/Embedding%20Protocol-100%25%20Conformant-brightgreen)

Reference implementation of the **Corpus OS Protocol Suite** — a **wire-first, vendor-neutral** SDK for interoperable AI/data backends across four domains: **LLM**, **Embedding**, **Vector**, and **Graph**.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Your App / Agents / RAG Pipelines                                   │
│  (LangChain · LlamaIndex · Semantic Kernel · CrewAI · AutoGen · MCP) │
├──────────────────────────────────────────────────────────────────────┤
│  Corpus OS Protocol Suite (this SDK)                                 │
│  One protocol · One error taxonomy · One metrics model               │
├──────────┬──────────────┬────────────┬───────────────────────────────┤
│ LLM/v1   │ Embedding/v1 │ Vector/v1  │ Graph/v1                      │
├──────────┴──────────────┴────────────┴───────────────────────────────┤
│  Any Provider: OpenAI · Anthropic · Pinecone · Neo4j · ...           │
└──────────────────────────────────────────────────────────────────────┘
```

**Keep your frameworks. Standardize your infra.**

> **Open-Core Model** — The Corpus OS Protocol Suite and this SDK are **fully open source** (Apache-2.0). CORPUS Router and official production adapters are **commercial**, optional, and built on the same public protocols. Using this SDK does **not** lock you into CORPUS Router.

---

## Table of Contents

1. [Why CORPUS](#why-corpus)
2. [How CORPUS Compares](#how-corpus-compares)
3. [When Not to Use CORPUS](#when-not-to-use-corpus)
4. [Install](#install)
5. [Quick Start](#-quick-start)
6. [Domain Examples](#domain-examples)
7. [Core Concepts](#core-concepts)
8. [Error Taxonomy & Observability](#error-taxonomy--observability)
9. [Performance & Configuration](#performance--configuration)
10. [Testing & Conformance](#testing--conformance)
11. [Documentation Layout](#-documentation-layout)
12. [FAQ](#faq)
13. [Contributing](#contributing)
14. [License & Commercial Options](#license--commercial-options)

---

## Why CORPUS

Modern AI platforms juggle multiple LLM, embedding, vector, and graph backends. Each vendor ships unique APIs, error schemes, rate limits, and capabilities — making cross-provider integration brittle and costly.

**The problem:**

- **Provider proliferation** — Dozens of incompatible APIs across AI infrastructure
- **Duplicate integration** — Same error handling, observability, and resilience patterns rewritten per provider
- **Vendor lock-in** — Applications tightly coupled to specific backend choices
- **Operational complexity** — Inconsistent monitoring and debugging across services

**CORPUS SDK provides:**

- **Stable, runtime-checkable protocols** across all four domains
- **Normalized errors** with retry hints and machine-actionable scopes
- **SIEM-safe metrics** (low-cardinality, tenant-hashed, no PII)
- **Deadline propagation** for cancellation and cost control
- **Two modes** — compose under your own router (`thin`) or use lightweight built-in infra (`standalone`)
- **Wire-first design** — canonical JSON envelopes implementable in any language, with this SDK as reference

Corpus OS is **not** a replacement for LangChain, LlamaIndex, Semantic Kernel, CrewAI, AutoGen, or MCP. Use those for orchestration, agents, tools, and RAG pipelines. Use Corpus OS to standardize the **infrastructure layer underneath them**. Your app teams keep their frameworks. Your platform team gets one protocol, one error taxonomy, and one observability model across everything.

---

## How CORPUS Compares

| Aspect | LangChain / LlamaIndex | OpenRouter | MCP | **Corpus OS** |
|---|---|---|---|---|
| **Scope** | Application framework | LLM unification | Tools & data sources | **AI infrastructure protocols** |
| **Domains** | LLM + Tools | LLM only | Tools + Data | **LLM + Vector + Graph + Embedding** |
| **Error Standardization** | Partial | Limited | N/A | **Comprehensive taxonomy** |
| **Multi-Provider Routing** | Basic | Managed service | N/A | **Protocol for any router** |
| **Observability** | Basic | Limited | N/A | **Built-in metrics + tracing** |
| **Vendor Neutrality** | High | Service-dependent | High | **Protocol-first, no lock-in** |

### Who is this for?

- **App developers** — Keep using your framework of choice. Talk to all backends through Corpus OS protocols. Swap providers without rewriting error handling.
- **Framework maintainers** — Implement one CORPUS adapter per protocol. Instantly support any conformant backend.
- **Backend vendors** — Implement `llm/v1`, `embedding/v1`, `vector/v1`, or `graph/v1` once, run the conformance suite, and your service works with every framework.
- **Platform / infra teams** — Unified observability: normalized error codes, deadlines, and metrics. One set of dashboards and SLOs across all AI traffic.
- **MCP users** — The Corpus OS MCP server exposes protocols as standard MCP tools. Any MCP client can call into your infra with consistent behavior.

### Integration Patterns

| Pattern | How It Works | What You Get |
|---|---|---|
| Framework → Corpus OS → Providers | Framework uses Corpus OS as client | Unified errors/metrics across providers |
| Corpus OS → Framework-as-adapter → Providers | Framework wrapped as Corpus OS adapter | Reuse existing chains/indices as "providers" |
| Mixed | Both of the above | Gradual migration, no big-bang rewrites |

Large teams typically run all three patterns at once.

---

## When Not to Use CORPUS

You probably don't need Corpus OS if:

- **Single-provider and happy** — One backend, fine with their SDK and breaking changes.
- **No governance pressure** — No per-tenant isolation, budgets, audit trails, or data residency.
- **No cross-domain orchestration** — Not coordinating LLM + Vector + Graph + Embedding together.
- **Quick throwaway prototype** — Lock-in, metrics, and resilience aren't worth thinking about yet.

If any of these stop being true, `corpus_sdk` is the incremental next step.

---

## Install

```bash
pip install corpus_sdk
```

Python ≥ 3.10 recommended. No heavy runtime dependencies.

---

## ⚡ Quick Start

```python
from corpus_sdk.llm.llm_base import BaseLLMAdapter, OperationContext

class QuickAdapter(BaseLLMAdapter):
    async def _do_complete(self, messages, **kwargs):
        return {"text": "Hello from CORPUS!", "model": "quick-demo"}

adapter = QuickAdapter()
ctx = OperationContext(request_id="test-123")
result = await adapter.complete(
    messages=[{"role": "user", "content": "Hi"}], ctx=ctx
)
print(result.text)  # "Hello from CORPUS!"
```

A complete quick start with all four protocols is in [`docs/guides/QUICK_START.md`](docs/guides/QUICK_START.md).

---

## Domain Examples

> **Minimal viable adapter:** Implement `_do_capabilities`, your core operation (`_do_embed`, `_do_complete`, `_do_query`, etc.), and `_do_health`. All other methods have safe no-op defaults — you only override what you need.

> In all examples, swap `Example*Adapter` with your actual adapter class that inherits the corresponding base and implements `_do_*` hooks.

<details>
<summary><strong>Embeddings</strong></summary>

```python
from corpus_sdk.embedding.embedding_base import (
    BaseEmbeddingAdapter, EmbedSpec, OperationContext,
    EmbeddingVector, EmbeddingCapabilities, EmbedResult
)

class ExampleEmbeddingAdapter(BaseEmbeddingAdapter):
    async def _do_capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            server="example-embeddings", version="1.0.0",
            supported_models=("example-embed-001",),
            max_batch_size=128, max_text_length=8192,
            supports_normalization=True, normalizes_at_source=False,
            supports_deadline=True, supports_token_counting=False,
        )

    async def _do_embed(self, spec: EmbedSpec, *, ctx=None) -> EmbedResult:
        vec = [0.1, 0.2, 0.3]
        return EmbedResult(
            embedding=EmbeddingVector(vector=vec, text=spec.text,
                                      model=spec.model, dimensions=len(vec)),
            model=spec.model, text=spec.text,
            tokens_used=None, truncated=False,
        )

    async def _do_health(self, *, ctx=None) -> dict:
        return {"ok": True, "server": "example-embeddings", "version": "1.0.0"}

# Usage
async with ExampleEmbeddingAdapter() as adapter:
    ctx = OperationContext(request_id="req-1", tenant="acme")
    res = await adapter.embed(
        EmbedSpec(text="hello world", model="example-embed-001"), ctx=ctx
    )
    print(res.embedding.vector)
```
</details>

<details>
<summary><strong>LLM</strong></summary>

```python
from corpus_sdk.llm.llm_base import (
    BaseLLMAdapter, OperationContext, LLMCompletion,
    TokenUsage, LLMCapabilities
)

class ExampleLLMAdapter(BaseLLMAdapter):
    async def _do_capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            server="example-llm", version="1.0.0",
            model_family="gpt-4", max_context_length=8192,
            supports_streaming=True, supports_roles=True,
            supports_json_output=False, supports_parallel_tool_calls=False,
            idempotent_writes=False, supports_multi_tenant=True,
            supports_system_message=True,
        )

    async def _do_complete(self, messages, model, **kwargs) -> LLMCompletion:
        return LLMCompletion(
            text="Hello from example-llm!", model=model,
            model_family="gpt-4",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            finish_reason="stop",
        )

    async def _do_health(self, *, ctx=None) -> dict:
        return {"ok": True, "server": "example-llm", "version": "1.0.0"}

# Usage
async with ExampleLLMAdapter() as adapter:
    ctx = OperationContext(request_id="req-2", tenant="acme")
    resp = await adapter.complete(
        messages=[{"role": "user", "content": "Say hi"}],
        model="example-llm-001", ctx=ctx,
    )
    print(resp.text)
```
</details>

<details>
<summary><strong>Vector</strong></summary>

```python
from corpus_sdk.vector.vector_base import (
    BaseVectorAdapter, VectorCapabilities, QuerySpec,
    QueryResult, Vector, VectorMatch, OperationContext, VectorID
)

class ExampleVectorAdapter(BaseVectorAdapter):
    async def _do_capabilities(self) -> VectorCapabilities:
        return VectorCapabilities(
            server="example-vector", version="1.0.0", max_dimensions=3
        )

    async def _do_query(self, spec: QuerySpec, *, ctx=None) -> QueryResult:
        v = Vector(id=VectorID("v1"), vector=[0.1, 0.2, 0.3],
                   metadata={"label": "demo"}, namespace=spec.namespace)
        return QueryResult(
            matches=[VectorMatch(vector=v, score=0.99, distance=0.01)],
            query_vector=spec.vector, namespace=spec.namespace, total_matches=1,
        )

    async def _do_health(self, *, ctx=None) -> dict:
        return {"ok": True, "server": "example-vector", "version": "1.0.0"}

# Usage
adapter = ExampleVectorAdapter()
ctx = OperationContext(request_id="req-3", tenant="acme")
result = await adapter.query(QuerySpec(vector=[0.1, 0.2, 0.3], top_k=1), ctx=ctx)
print(result.matches[0].score)
```
</details>

<details>
<summary><strong>Graph</strong></summary>

```python
from corpus_sdk.graph.graph_base import (
    BaseGraphAdapter, GraphCapabilities, UpsertNodesSpec,
    Node, GraphID, OperationContext, GraphQuerySpec, GraphQueryResult
)

class ExampleGraphAdapter(BaseGraphAdapter):
    async def _do_capabilities(self) -> GraphCapabilities:
        return GraphCapabilities(
            server="example-graph", version="1.0.0",
            supported_query_dialects=("cypher",),
            supports_stream_query=True, supports_bulk_vertices=True,
            supports_batch=True, supports_schema=True,
        )

    async def _do_query(self, spec: GraphQuerySpec, *, ctx=None) -> GraphQueryResult:
        return GraphQueryResult(
            records=[{"id": 1, "name": "Ada"}],
            summary={"rows": 1}, dialect=spec.dialect,
            namespace=spec.namespace,
        )

    async def _do_health(self, *, ctx=None) -> dict:
        return {"ok": True, "server": "example-graph", "version": "1.0.0"}

# Usage
async with ExampleGraphAdapter() as adapter:
    ctx = OperationContext(request_id="req-4", tenant="acme")
    result = await adapter.upsert_nodes(
        UpsertNodesSpec(nodes=[
            Node(id=GraphID("user:1"), labels=("User",),
                 properties={"name": "Ada"})
        ]),
        ctx=ctx,
    )
    print(f"Upserted {result.upserted_count} nodes")
```
</details>

Full implementations with batch operations, streaming, and multi-cloud scenarios are in [`docs/guides/ADAPTER_RECIPES.md`](docs/guides/ADAPTER_RECIPES.md).

---

## Core Concepts

- **Protocol vs Base** — Protocols define required behavior. Bases implement validation, deadlines, observability, and error normalization. You implement `_do_*` hooks.
- **OperationContext** — Carries `request_id`, `idempotency_key`, `deadline_ms`, `traceparent`, `tenant`, and cache hints across all operations.
- **Wire Protocol** — Canonical envelopes (`op`, `ctx`, `args`) and response shapes (`ok`, `code`, `result`) defined in [`docs/spec/PROTOCOL.md`](docs/spec/PROTOCOL.md).
- **Corpus OS-Compatible** — Implementations that honor the envelopes, reserved `op` strings, and error taxonomy. Validated by the conformance suite.

---

## Error Taxonomy & Observability

All domains share a **normalized error taxonomy**: `BadRequest`, `AuthError`, `ResourceExhausted`, `TransientNetwork`, `Unavailable`, `NotSupported`, `DeadlineExceeded`, plus domain-specific errors like `TextTooLong`, `ModelOverloaded`, `DimensionMismatch`, and `IndexNotReady`.

Errors carry **machine-actionable hints** (`retry_after_ms`, `throttle_scope`) so routers and control planes can react consistently across providers. A pluggable `MetricsSink` protocol lets you bring your own metrics backend. Bases emit one `observe` per operation, hash tenants before recording, and never log prompts, vectors, or raw tenant IDs.

Full details in [`docs/spec/ERRORS.md`](docs/spec/ERRORS.md) and [`docs/spec/METRICS.md`](docs/spec/METRICS.md).

---

## Performance & Configuration

Base overhead is typically **<10 ms** relative to vendor SDK calls: validation <1 ms, metrics <0.1 ms, cache lookup (standalone) <0.5 ms. Async-first design avoids blocking and supports high concurrency. Batch operations (`embed_batch`, vector upserts, graph batch) are preferred for throughput.

Benchmarks and deployment patterns in [`docs/guides/IMPLEMENTATION.md`](docs/guides/IMPLEMENTATION.md).

### Modes: `thin` vs `standalone`

Once you're ready for production, choose a mode:

| Mode | Infra Hooks | When to Use |
|---|---|---|
| **`thin`** (default) | All no-ops | You have an external control plane (router, scheduler, limiter) |
| **`standalone`** | Deadline enforcement, circuit breaker, token-bucket limiter, in-memory TTL cache | Lightweight deployments without external infra |

Use `thin` under a router to prevent double-stacking resiliency. Use `standalone` for prototyping or single-service deployments.

---

## Testing & Conformance

### One-Command Testing

```bash
# All protocols at once
make test-all-conformance

# Specific protocols
make test-llm-conformance
make test-vector-conformance
make test-graph-conformance
make test-embedding-conformance
```

### CLI

```bash
corpus-sdk verify                        # All protocols
corpus-sdk verify -p llm -p vector       # Selected protocols
corpus-sdk test-llm-conformance          # Single protocol
```

### Direct pytest

```bash
pytest tests/ -v --cov=corpus_sdk --cov-report=html
```

Requirements for "CORPUS-Compatible" certification are in [`docs/conformance/CERTIFICATION.md`](docs/conformance/CERTIFICATION.md).

---

## 📚 Documentation Layout

**Spec (normative):** [`docs/spec/`](docs/spec/)

| File | Contents |
|---|---|
| `SPECIFICATION.md` | Full protocol suite specification (all domains, cross-cutting behavior) |
| `PROTOCOL.md` | Wire-level envelopes, streaming semantics, canonical `op` registry |
| `ERRORS.md` | Canonical error taxonomy & mapping rules |
| `METRICS.md` | Metrics schema & SIEM-safe observability |
| `SCHEMA.md` | JSON/type shapes |
| `VERSIONING.md` | Semantic versioning & compatibility rules |

**Guides (how-to):** [`docs/guides/`](docs/guides/)

| File | Contents |
|---|---|
| `QUICK_START.md` | End-to-end flows for all four protocols |
| `IMPLEMENTATION.md` | How to implement adapters |
| `ADAPTER_RECIPES.md` | Real-world multi-cloud and RAG scenarios |
| `CONFORMANCE_GUIDE.md` | How to run & interpret conformance suites |

**Conformance (testing):** [`docs/conformance/`](docs/conformance/) — Per-protocol test specs, schema conformance, behavioral conformance, and certification requirements.

---

## FAQ

<details>
<summary><strong>Is the SDK open source?</strong></summary>

Yes. The SDK (protocols, bases, example adapters) is open source under Apache-2.0. CORPUS Router and official production adapters are commercial.
</details>

<details>
<summary><strong>Do I have to use CORPUS Router?</strong></summary>

No. The SDK composes with any router or control plane. CORPUS Router is optional and adheres to the same public protocols.
</details>

<details>
<summary><strong>How does CORPUS relate to LangChain / LlamaIndex / MCP / OpenRouter?</strong></summary>

They're complementary. LangChain/LlamaIndex are application frameworks. MCP standardizes tools and data sources. OpenRouter unifies LLM providers. CORPUS standardizes the infrastructure layer (LLM + Vector + Graph + Embedding) underneath all of them with consistent errors, metrics, and capabilities discovery.
</details>

<details>
<summary><strong>Why async-only?</strong></summary>

Modern AI workloads require high concurrency. Async-first prevents blocking the event loop. Sync wrappers can be built on top if needed.
</details>

<details>
<summary><strong>What happens if my adapter raises a non-normalized error?</strong></summary>

Bases catch unexpected exceptions and record them as `UnhandledException` in metrics. Wrap provider errors in normalized exceptions for proper handling.
</details>

<details>
<summary><strong>Can CORPUS Router run on-prem?</strong></summary>

Yes. Available as managed cloud or on-prem deployment for regulated and air-gapped environments.
</details>

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Double-stacked resiliency (timeouts firing twice) | Ensure `mode="thin"` under your router |
| Circuit breaker opens frequently | Reduce concurrency or switch to `thin` with external circuit breaker |
| Cache returns stale results | Verify sampling params in cache key; check `cache_ttl_s` |
| `DeadlineExceeded` on fast operations | Ensure `deadline_ms` is absolute epoch time, not relative. Check NTP sync. |
| Health check failures | Inspect `_do_health` implementation; verify backend reachability and credentials |

```python
# Debug mode
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("corpus_sdk").setLevel(logging.DEBUG)
```

---

## Contributing

```bash
git clone https://github.com/corpus/corpus-sdk.git
cd corpus-sdk
pip install -e ".[dev]"
pytest
```

Follow PEP-8 (ruff/black). Type hints required on all public APIs. Include tests for new features. Maintain low-cardinality metrics — never add PII to `extra` fields. Observe SemVer.

We especially welcome **community adapter contributions** for new LLM, vector, graph, and embedding backends.

Community questions: [GitHub Discussions](https://github.com/corpus/corpus-sdk/discussions) preferred.

---

## License & Commercial Options

**License:** Apache-2.0 ([`LICENSE`](LICENSE))

| Need | Solution | Cost |
|---|---|---|
| Learning / prototyping | `corpus_sdk` + example adapters | **Free (OSS)** |
| Production with your own infra | `corpus_sdk` + your adapters | **Free (OSS)** |
| Production with official adapters | `corpus_sdk` + Official Adapters | **Commercial** |
| Enterprise multi-provider | `corpus_sdk` + CORPUS Router (Managed or On-Prem) | **Commercial** |

**Contact:** [sales@corpusos.com](mailto:sales@corpusos.com) · [partners@corpusos.com](mailto:partners@corpusos.com)

---

**Built by the Corpus OS team** — wire-level AI infrastructure you integrate once and stop thinking about.
