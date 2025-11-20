# Corpus Router Overview

> **One routing layer. Four AI domains. Learns what works.**  
> Built on the **Corpus Protocol Suite** (wire-level, vendor-neutral).

Corpus Router is an **enterprise orchestration layer** for AI infrastructure across:

- **LLM providers**
- **Embedding services**
- **Vector databases**
- **Graph databases**

It sits between your applications and your providers, speaking the **same wire protocol** as `corpus_sdk`:

- Canonical envelopes: `op`, `ctx`, `args` → `ok`, `code`, `result`
- Standardized error taxonomy
- Unified metrics and deadlines

If the **Corpus SDK** is the reference implementation of the **protocols**, **Corpus Router** is the reference implementation of a **control plane** that uses those protocols for **routing, governance, and observability**.

---

## Core Features (All Tiers)

Across all deployments, Corpus Router provides:

- **Universal interface across four domains**  
  One API for LLM providers, vector databases, graph databases, and embedding
  systems. Switch backends without changing application code—only routing config.

- **Multi-provider routing & failover**  
  Route requests across providers in any domain, with automatic failover when
  services or regions are unhealthy.

- **Request/response validation**  
  Validate envelopes, schemas, limits, and policies *before* expensive provider
  calls using the shapes from `docs/spec/SCHEMA.md`.

- **Unified observability & logging**  
  A single metrics and tracing model (`docs/spec/METRICS.md`) across LLM, Vector,
  Graph, and Embedding traffic, regardless of provider or framework.

- **Cost tracking & attribution**  
  Per-tenant / per-team view of usage and cost, mapped to normalized operations
  (`llm.complete`, `vector.query`, etc.), not vendor-specific APIs.

- **Deadline propagation & cancellation**  
  Deadlines from `ctx.deadline_ms` are enforced and propagated downstream,
  preventing runaway calls and wasted budget.

These are all built on top of the same **wire-level contracts** defined in
`docs/spec/SPECIFICATION.md` and `docs/spec/PROTOCOL.md`.

---

## How Router Relates to Corpus SDK & Protocols

- **Corpus Protocol Suite** (in `docs/spec/`)  
  Defines the **wire-level contracts** for LLM, Embedding, Vector, and Graph:
  - `SPECIFICATION.md` – overall spec and cross-protocol behavior
  - `PROTOCOL.md` – envelopes, streaming semantics, `op` registry
  - `ERRORS.md`, `METRICS.md`, `SCHEMA.md`, `VERSIONING.md`

- **Corpus SDK**  
  - Python reference implementation of those protocols.
  - You build **adapters** against `*BaseAdapter` classes.
  - Lives in this repo, documented in this README and `docs/guides/`.

- **Corpus Router** (commercial)  
  - Runs as a service (managed or on-prem).
  - Speaks the **same wire protocol** as defined in `docs/spec/`.
  - Uses **Corpus-compatible adapters** (or your own) to talk to providers.
  - Adds:
    - Multi-provider routing
    - Policies (budgets, allowlists, data residency)
    - Self-learning optimization
    - Centralized metrics & traces across **all four domains**

The Router does *not* change the protocol. It **consumes** the same wire format
and capabilities you already use via `corpus_sdk`.

---

## Core Responsibilities

At a high level, Corpus Router:

1. **Terminates Corpus Protocol requests**

   - Accepts JSON envelopes (`op`, `ctx`, `args`) over HTTP/HTTP2/WebSocket.
   - Validates against schemas from `docs/spec/SCHEMA.md`.

2. **Applies policies**

   - Budget enforcement, rate limits, provider allowlists.
   - Data residency & compliance constraints.
   - Per-tenant isolation rules.

3. **Selects a backend**

   - Uses **capabilities** + **historical metrics**.
   - Considers latency, cost, error rates, and quality signals.
   - Can use static routing, weighted routing, or self-learning policies.

4. **Calls providers via adapters**

   - Uses adapters built on `corpus_sdk` bases (or equivalent).
   - Adapters must pass the **conformance suites** in `docs/conformance/`.

5. **Normalizes the response**

   - Maps provider errors into the canonical taxonomy (`ERRORS.md`).
   - Emits metrics according to `METRICS.md`.
   - Returns a normalized response envelope to the caller.

---

## Architecture Overview

**Key idea**: The Router is a **protocol-native control plane**. It does not invent a new API; it **only** routes Corpus Protocol traffic.

```text
┌───────────────────────────────┐
│         Your Apps             │
│  (LangChain, LlamaIndex,      │
│   Semantic Kernel, CrewAI,    │
│   AutoGen, MCP, custom)       │
└──────────────┬────────────────┘
               │ Corpus Protocol (wire)
               ▼
       ┌───────────────────┐
       │   Corpus Router   │
       │  • Policies       │
       │  • Self-learning  │
       │  • Multi-tenant   │
       │  • Observability  │
       └────────┬──────────┘
                │
     ┌──────────┼─────────────────────────┐
     ▼          ▼                         ▼
┌────────┐ ┌────────────┐         ┌──────────────┐
│  LLM   │ │  Vector DB  │  ...   │ Graph / Embed│
│  Adpts │ │  Adapters   │        │  Adapters    │
└────────┘ └────────────┘         └──────────────┘
   │           │                         │
   ▼           ▼                         ▼
 OpenAI   Pinecone/Qdrant/...   Neo4j/TigerGraph/... + more
 Anthropic
 Mistral
 Cohere
 ...
````

* All adapters are **Corpus-compatible**: they implement the same ops and envelopes defined in `docs/spec/PROTOCOL.md`.
* Your own services can be added as providers as soon as they pass conformance (see `docs/conformance/*`).

---

## Frameworks as Adapters

Frameworks like **LangChain**, **LlamaIndex**, **Semantic Kernel**, **CrewAI**, and **AutoGen** are not just clients of the Router — they can also be **providers** that Router routes *into*.

You can wrap an existing framework as a **Corpus adapter**, so Router treats it like any other backend:

### Router → Framework → Providers

```text
┌──────────────┐
│ Your App     │
└──────┬───────┘
       │ Corpus Protocol
       ▼
┌──────────────┐
│   Router     │
└──────┬───────┘
       │
       ├─→ OpenAI Adapter (direct)
       ├─→ Anthropic Adapter (direct)
       └─→ LangChain Adapter ──→ LangChain LLM / tools / chains
                                  └─→ Multiple providers
```

**Why use frameworks as adapters?**

* **Leverage existing investments**
  Reuse LangChain chains, LlamaIndex indexes, or Semantic Kernel skills as
  Router-managed providers instead of rewriting everything as raw adapters.

* **Framework-specific features**
  Access LangChain’s tool-calling, LlamaIndex’s query engines, or AutoGen/CrewAI
  agent orchestration **through** Router’s unified interface, with normalized
  errors and metrics.

* **Gradual migration**
  Start with frameworks as adapters; move specific flows to direct
  Corpus adapters over time as needed.

### Example: LangChain as an LLM adapter

```python
from corpus_sdk.llm.llm_base import (
    BaseLLMAdapter,
    LLMCapabilities,
)
from langchain.chat_models import ChatOpenAI

class LangChainLLMAdapter(BaseLLMAdapter):
    """Wraps LangChain's LLM interface as a Corpus-compatible adapter."""

    def __init__(self, model: str = "gpt-4"):
        super().__init__()
        self._llm = ChatOpenAI(model=model)

    async def _do_capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            server="langchain",
            version="0.1.0",
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

    async def _do_complete(self, messages, model: str | None = None, **kwargs):
        # 1. Translate Corpus messages → LangChain format
        lc_messages = self._to_langchain_messages(messages)

        # 2. Call LangChain
        result = await self._llm.ainvoke(lc_messages)

        # 3. Translate back into a Corpus `LLMCompletion`
        return self._from_langchain_result(result)

    # You implement these helpers to match your message/result shapes:
    def _to_langchain_messages(self, messages):
        ...

    def _from_langchain_result(self, result):
        ...
```

**Same pattern works for:**

* **LlamaIndex** as a Vector or Graph adapter (wrap their `VectorStore` / query engine).
* **Semantic Kernel** as an LLM / Embedding adapter.
* **CrewAI / AutoGen** as a “meta-LLM” adapter that runs multi-agent workflows behind a single `llm.complete` call.

Result: Router can route to **framework-based** stacks and **raw provider** stacks side-by-side, with the same policies, metrics, and error taxonomy.

> The next section describes the “opposite” pattern — frameworks acting as **clients** of Router. Both patterns can coexist:
> • Use **frameworks as adapters** when you want Router to treat them as providers.
> • Use **frameworks as clients** when they call Router as a backend.

---

## Integration with Frameworks (LangChain, LlamaIndex, CrewAI, AutoGen, SK, MCP)

Corpus is a **wire protocol**, not an application framework.

* Frameworks like **LangChain**, **LlamaIndex**, **CrewAI**, **AutoGen**, **Semantic Kernel**, and **MCP** are *clients* or *orchestrators*.
* Corpus Router is a **protocol-native backend** they can talk to over HTTP, using the standard envelopes.

Any framework or client that can:

* Send/receive JSON over HTTP
* Respect the Corpus envelopes and operations

…can talk to the Router without being “Corpus-native” internally.

You get two main integration patterns (complementary to the adapter pattern above):

### 1. Framework → Corpus SDK → Router

* Use `corpus_sdk` as the **client** to the Router.
* Swap vendor-specific LangChain/LlamaIndex/AutoGen/... LLM & Vector classes
  with thin wrappers that call Corpus operations via the SDK.

Example flow (conceptual):

```python
# Inside your framework integration
from corpus_sdk.llm.llm_base import BaseLLMAdapter, OperationContext

class RouterLLMAdapter(BaseLLMAdapter):
    """
    Adapter that talks to Corpus Router over the wire instead of
    a single provider SDK.
    """
    async def _do_complete(self, messages, **kwargs):
        # Serialize as Corpus Protocol envelope and send to Router endpoint.
        # Router then decides which provider to call.
        ...
```

You can do the same for:

* LlamaIndex `VectorStore`
* Semantic Kernel `TextEmbeddingGeneration`
* CrewAI / AutoGen tool calls
* Any other framework component that can call Python / HTTP

### 2. Router as MCP Server / Tool Backend

The Router can be exposed as an MCP server that:

* Maps **MCP tools** to **Corpus ops** (`llm.complete`, `vector.query`,
  `graph.query`, `embedding.embed`, …).
* Lets MCP clients (including ChatGPT) access your AI infrastructure **through
  the Router**, inheriting:

  * Normalized errors
  * Budgets & policies
  * Multi-provider failover
  * Unified observability

---

## Domains & Operations

The Router works across **four domains**, each with a set of canonical operations
(from `docs/spec/PROTOCOL.md`):

* **LLM**

  * `llm.complete`
  * `llm.stream`
  * `llm.count_tokens`
  * `llm.capabilities`
  * `llm.health`

* **Embedding**

  * `embedding.embed`
  * `embedding.embed_batch`
  * `embedding.count_tokens`
  * `embedding.capabilities`
  * `embedding.health`

* **Vector**

  * `vector.query`
  * `vector.upsert`
  * `vector.delete`
  * `vector.create_namespace`
  * `vector.delete_namespace`
  * `vector.capabilities`
  * `vector.health`

* **Graph**

  * `graph.query`
  * `graph.stream_query`
  * `graph.upsert_nodes`
  * `graph.upsert_edges`
  * `graph.delete_nodes`
  * `graph.delete_edges`
  * `graph.bulk_vertices`
  * `graph.batch`
  * `graph.get_schema`
  * `graph.capabilities`
  * `graph.health`

The Router treats these as **opaque operations** defined by the spec. It doesn’t
invent its own semantics; it routes based on:

* `op`
* `ctx` (tenant, deadlines, attributes)
* Provider capabilities
* Historical metrics

---

## Policy Engine

Policies define what is allowed and how requests should be shaped.

Typical policy dimensions:

* **Budgets**

  * Per-tenant, per-team, per-project dollar ceilings.
  * Per-domain budgets (LLM vs Vector vs Graph vs Embedding).

* **Rate Limits / Concurrency**

  * Requests per second per tenant.
  * Tokens-per-minute for selected models.
  * Concurrency caps per provider.

* **Provider Allow/Deny Lists**

  * “In prod, only these providers/models.”
  * “In EU, only EU-hosted providers.”

* **Data Residency & Compliance**

  * Route based on `ctx.attrs.region` or tenant metadata.
  * Enforce “EU traffic → EU-resident providers only”.

* **Latency / SLO Targets**

  * Different SLOs per tenant/tier.
  * Use SLO violations as routing signals.

Policies can be:

* **Static** (YAML / JSON config)
* **API-driven** (for dynamic updates)
* **Tenant-aware** (per-tenant overrides)

Router evaluates policies **before** calling any provider, so non-compliant
requests are rejected quickly and safely.

---

## Self-Learning Routing (Privacy-Preserving)

The Router’s self-learning engine is **optional** but powerful:

* Learns routing decisions from:

  * Latency
  * Cost
  * Error patterns
  * Optional external quality scores (if you feed them)

* Works across all four domains:

  * Best LLM for this type of prompt
  * Best vector DB for this embedding/model pair
  * Best graph DB for a given workload pattern
  * Best embedding provider for a particular corpus size/latency budget

**Privacy guarantees:**

* Learns from **metadata only** (timings, sizes, error codes, etc.).
* Does **not** persist raw prompts, embeddings, vectors, or tenant identifiers.
* Uses the SIEM-safe model (`tenant_hash`, content hashes) defined in
  `docs/spec/METRICS.md` & `docs/spec/SPECIFICATION.md` (Privacy & Security
  sections).

**Per-tenant models:**

* Each tenant can have their own routing profile.
* Tenant A’s decisions are not influenced by Tenant B’s traffic unless
  explicitly configured.

**Guardrails:**

* Self-learning always runs **inside** policy constraints.

  * No routing to disallowed providers.
  * No overrunning budgets.
  * No violating residency rules.

---

## Multi-Tenancy & Isolation

Multi-tenancy is a first-class requirement in the Corpus Protocol Suite (see
`SPECIFICATION.md` § Security & Privacy) and the Router follows the same rules:

* **Tenant isolation**

  * Routing, rate limiting, circuit breaking, and caching are all scoped by
    tenant.
  * No cross-tenant cache sharing unless explicitly configured in a safer
    aggregate layer.

* **Tenant-aware context**

  * Uses `ctx.tenant` and derived `tenant_hash` as primary keys for metrics and
    policy decisions.
  * Never logs raw tenant identifiers.

* **Per-tenant routing “profiles”**

  * Different strategies per tenant:

    * `cost_optimized`
    * `low_latency`
    * `high_quality`
    * `compliance_focused`
  * Per-tenant domain sub-strategies:

    * e.g., cheap LLM + premium vector DB, or vice versa.

---

## Observability & Metrics

Router uses the same **metrics taxonomy** as described in `docs/spec/METRICS.md`:

* **Core metrics**

  * `ops_total{component,op,code}`
  * `latency_ms{component,op,code,quantile}`

* **Domain-specific metrics**

  * Tokens, matches returned, batch size, etc.

* **Tenant-safe**

  * Uses `tenant_hash` instead of raw IDs.
  * No prompts / vectors / full texts in metrics.

As a result:

* You can build **one set of dashboards** that covers:

  * All frameworks (LangChain, LlamaIndex, etc.)
  * All providers (LLMs, vectors, graphs, embeddings)
  * All tenants

* The Router becomes the single place to:

  * Debug routing decisions
  * Investigate SLO violations
  * Analyze cost/latency trade-offs

---

## Streaming & Deadlines

Router fully respects the streaming and deadline semantics from
`docs/spec/PROTOCOL.md` and `docs/spec/SPECIFICATION.md`:

* **Streaming**

  * For `llm.stream`, `graph.stream_query`, etc.:

    * Emits a sequence of `data` frames followed by exactly one terminal `end`
      or `error`.
    * Never sends `data` after terminal frames.
  * Supports transports:

    * NDJSON over HTTP
    * SSE
    * WebSocket
    * (And gRPC/HTTP/2 bindings where deployed)

* **Deadlines**

  * Uses `ctx.deadline_ms` as the **absolute time budget**.
  * Enforces or tightens downstream timeouts.
  * Maps upstream/backend timeouts into `DEADLINE_EXCEEDED` or `UNAVAILABLE`
    with proper retry hints.

This makes Router safe to use under strict latency SLOs and in large multi-hop
flows.

---

## Deployment Options

**Managed Service**

* Corpus-hosted Router.
* Connect your providers via:

  * Official adapters (commercial).
  * Your own adapters (built on `corpus_sdk` and deployed in your infra).
* Management plane, dashboards, and policy editing UIs.

**On-Prem / Private Cloud**

* Router deployed in your VPC / cluster (including air-gapped environments).
* Same protocol and behavior.
* Integrates with your:

  * IdP / SSO
  * SIEM
  * Metrics stack
  * Secret management

In both modes:

* The **wire protocol** remains the same.
* Your applications and frameworks talk to Router just like they would talk to
  any other Corpus-compatible backend.

---

## Relationship to Conformance & Certification

Router’s own adapters and internal components are tested using the same
conformance suites documented in `docs/conformance/`:

* `LLM_CONFORMANCE.md`
* `EMBEDDING_CONFORMANCE.md`
* `VECTOR_CONFORMANCE.md`
* `GRAPH_CONFORMANCE.md`
* `SCHEMA_CONFORMANCE.md`
* `BEHAVIORAL_CONFORMANCE.md`
* `CERTIFICATION.md`

For how to run and interpret these suites, see
`docs/guides/CONFORMANCE_GUIDE.md`.

If you build your own providers or internal services that speak the Corpus
Protocol Suite and pass those tests, they can be **plugged into the Router** as
first-class backends.

---

## When to Use Router vs Just SDK

* Use **`corpus_sdk` alone** when:

  * You have a small number of providers.
  * You’re okay managing routing, retries, and budgets inside your own code or
    sidecar.
  * You primarily need a clean, typed, vendor-neutral SDK.

* Add **Corpus Router** when:

  * You have **multiple providers** per domain (LLM, Vector, Graph, Embedding).
  * You need **centralized policies** (budget, compliance, residency).
  * You want **self-learning routing** and **cross-tenant analytics**.
  * You want to decouple **application code** from **infrastructure decisions**.

---

## Support & SLAs (Commercial)

Corpus Router is offered with enterprise support, SLAs, and 24/7 coverage for
production deployments:

* Managed and on-prem deployment options.
* Assistance with adapter implementation and conformance.
* Integration with your existing observability and security tooling.

See product documentation and pricing pages for the current support and SLA
details.

---

## Where to Go Next

Inside this repo:

* **Protocols & behavior**

  * `docs/spec/SPECIFICATION.md`
  * `docs/spec/PROTOCOL.md`
  * `docs/spec/ERRORS.md`
  * `docs/spec/METRICS.md`

* **How to implement adapters**

  * `docs/guides/IMPLEMENTATION.md`
  * `docs/guides/ADAPTER_RECIPES.md` (multi-cloud, RAG, etc.)
  * `docs/guides/QUICK_START.md`

* **Conformance & certification**

  * `docs/conformance/CONFORMANCE_GUIDE.md`
  * `docs/conformance/CERTIFICATION.md`

Outside this repo (commercial):

* Corpus Router product pages, pricing, and deployment docs.
* Contact: `sales@corpus.io` / `partners@corpus.io`.

---

**Summary:**
Corpus Router is the **protocol-native intelligent control plane** for the Corpus Suite: it
understands the same wire protocol as `corpus_sdk`, routes across multiple
providers and domains (including framework-based stacks like LangChain or LlamaIndex),
enforces policies, learns what works, and keeps your
applications decoupled from infrastructure churn—without inventing a new API
surface on top.

```
