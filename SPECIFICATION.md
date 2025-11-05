# Corpus Protocol Suite Specification

## Abstract

This specification defines the Corpus Protocol Suite: a vendor-neutral set of production-grade interfaces for **graph databases**, **large language models**, **vector databases**, and **text embeddings**. The suite standardizes contracts for heterogeneous AI infrastructure with built-in observability, error handling, and operational rigor. Protocols are minimal yet expressive, async-first, and extensible via negotiated capabilities. This document includes normative contracts, wire-compatible data shapes, an error taxonomy and mapping, resilience semantics, privacy and security guidance, and compatibility/versioning rules for enterprise-scale deployments.

> **Keywords:** Graph Database, Large Language Model, Vector Search, **Embeddings**, Observability, Multi-Tenancy, Capability Discovery, Semantic Versioning, SIEM-Safe Telemetry, BCP 14

## Status of This Memo

This document is not an Internet Standards Track specification; it is published for informational/standards-style guidance for the Corpus Protocol Suite. Distribution of this memo is unlimited.

## Copyright Notice

Copyright © 2025 Corpus Protocol Suite.
SPDX-License-Identifier: Apache-2.0

---

## Table of Contents

1. Introduction
2. Requirements Language
3. Terminology
4. Conventions and Notation
5. Architecture Overview
6. Common Foundation
7. Graph Protocol V1 Specification
8. LLM Protocol V1 Specification
9. Vector Protocol V1 Specification
10. **Embedding Protocol V1 Specification**
11. Cross-Protocol Patterns
12. Error Handling and Resilience
13. Observability and Monitoring
14. Security Considerations
15. Privacy Considerations
16. Performance Characteristics
17. Implementation Guidelines
18. Versioning and Compatibility
19. IANA Considerations
20. References
     20.1 Normative References
     20.2 Informative References
21. Author’s Address
    **Appendix A** — End-to-End Example (Normative)
    **Appendix B** — Capability Shapes (Illustrative)
    **Appendix C** — Wire-Level Envelopes (Optional)
    **Appendix D** — Content Redaction Patterns (Normative)
    **Appendix E** — Implementation Status (Non-Normative)
    **Appendix F** — Change Log / Revision History (Non-Normative)

---

## 1. Introduction

### 1.1. Motivation

The proliferation of AI infrastructure has created a fragmented landscape of proprietary APIs and inconsistent interfaces. Fragmentation increases integration complexity, reduces operational visibility, and creates vendor lock-in. Enterprise teams need cohesive, auditable, and performance-predictable interfaces that allow swapping providers without rewriting core application code or telemetry pipelines.

### 1.2. Scope

This specification defines four complementary protocols:

* **Graph Protocol V1** — Vertex/edge CRUD, traversal, and multi-dialect query execution.
* **LLM Protocol V1** — Chat-style completion, streaming tokens, usage accounting.
* **Vector Protocol V1** — Vector upsert/delete, similarity search, and namespace management.
* **Embedding Protocol V1** — Text embedding generation (single/batch), token counting, capability discovery, and health reporting.

All protocols share a **Common Foundation** (context propagation, capability discovery, error taxonomy, observability, resilience).

### 1.3. Design Philosophy

* **Minimal Surface Area (MUST).** Only essential operations are standardized. Vendor extensions appear via capabilities, not new methods.
* **Async-First (MUST).** All operations are non-blocking and concurrency-safe.
* **Production-Hardened (MUST).** Observability, error taxonomy, and resilience are first-class.
* **Extensible (SHOULD).** Capability negotiation enables optional features without breaking compatibility.
* **Type-Safe (SHOULD).** Strong typing and runtime validation minimize undefined behavior.
* **Privacy by Design (MUST).** SIEM-safe telemetry, data minimization, and redaction are defaults.

---

## 2. Requirements Language

The key words “MUST”, “MUST NOT”, “REQUIRED”, “SHALL”, “SHALL NOT”, “SHOULD”, “SHOULD NOT”, “RECOMMENDED”, “NOT RECOMMENDED”, “MAY”, and “OPTIONAL” in this document are to be interpreted as described in BCP 14 [RFC2119] [RFC8174] when, and only when, they appear in all capitals.

---

## 3. Terminology

**Adapter** — Concrete implementation of a protocol for a specific provider/backend.
**Protocol** — Interface contract that adapters MUST implement.
**Operation Context** — Metadata container for tracing, deadlines, and tenancy.
**Capabilities** — Dynamically discoverable features and limits of an adapter.
**SIEM-Safe** — Observability that excludes PII and uses privacy-preserving identifiers.
**Idempotency Key** — Client-provided token guaranteeing idempotent semantics.
**Tenant Isolation** — Logical separation of data/control plane in multi-tenant deployments.
**Backpressure** — Cooperative throttling to keep systems within safe operating limits.

---

## 4. Conventions and Notation

* JSON keys are **case-sensitive**; unknown keys **MUST** be ignored by clients and servers.
* Durations are expressed in **milliseconds** unless otherwise specified.
* Examples are **non-normative** unless explicitly marked **(Normative)**.
* Field names use **lower_snake_case** unless specified.
* Error `code` values use `UPPER_SNAKE_CASE`.
* “tenant_hash” denotes a deterministic, irreversible hash of the tenant identifier.
* Unless otherwise stated, scores are **higher is better** (cosine/dot); distance metrics MAY be inverted to scores.

---

## 5. Architecture Overview

### 5.1. Protocol Relationships

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Graph Protocol│    │   LLM Protocol  │    │  Vector Protocol│    │ Embedding Proto │
│  • CRUD/Query   │    │  • Completion   │    │  • Search       │    │  • Single/Batch │
│  • Dialects     │    │  • Streaming    │    │  • Upsert/Delete│    │  • Token Count  │
│                 │    │  • Token Usage  │    │  • Namespaces   │    │  • Health       │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         └───────────────────────┼───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │  Common Foundation  │
                    │ • Context           │
                    │ • Errors            │
                    │ • Observability     │
                    │ • Resilience        │
                    └─────────────────────┘
```

### 5.2. Layered Architecture

```
┌─────────────────────────────────────────┐
│            Application Layer            │
│ (Orchestrates Graph/LLM/Vector/Embed)   │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│             Protocol Layer              │
│ (Uniform contracts with capabilities)   │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│              Adapter Layer              │
│ (Provider-specific implementations)     │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│               Backend Layer             │
│ (Databases, models, vector & embed svc) │
└─────────────────────────────────────────┘
```

---

## 6. Common Foundation

### 6.1. Operation Context

```python
from dataclasses import dataclass
from typing import Any, Mapping, Optional

@dataclass(frozen=True)
class OperationContext:
    request_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    deadline_ms: Optional[int] = None
    traceparent: Optional[str] = None  # W3C Trace Context
    tenant: Optional[str] = None       # never logged raw
    attrs: Optional[Mapping[str, Any]] = None
```

**Normative behavior:**

* `request_id` (SHOULD) uniquely ties together end-to-end operations.
* `idempotency_key` (MAY) be supplied on mutating operations; when present, adapters **MUST** guarantee exactly-once external effects or return a consistent prior result.
* `deadline_ms` (SHOULD) be treated as an absolute deadline; adapters **MUST** propagate remaining budget downstream when possible.
* `traceparent` (MUST) be forwarded unchanged for distributed tracing.
* `tenant` (MUST) enforce isolation; telemetry **MUST NOT** log raw tenant values.

### 6.2. Capability Discovery

`capabilities()` **MUST** return a structured description:

```json
{
  "server": "example-backend",
  "version": "5.17.1",
  "protocol": "graph/v1",
  "features": {
    "dialects": ["cypher", "opencypher"],
    "supports_txn": true,
    "supports_schema_ops": true,
    "supports_pagination": false
  },
  "limits": {
    "max_batch_ops": 5000,
    "rate_limit_qps": 200,
    "concurrency": 256,
    "max_query_length": 100000
  },
  "extensions": {
    "vendor:neo4j.routing": "cluster",
    "vendor:read_your_writes": true
  }
}
```

* Unknown `features`, `limits`, or `extensions` keys **MUST** be ignored by clients.
* `extensions` keys **MUST** be namespaced (e.g., `vendor:foo.bar`).
* Semantic changes to published keys **MUST NOT** occur without a **major** protocol version change.

### 6.3. Error Taxonomy

```
AdapterError (base)
├─ BadRequest              # 400 client errors (validation, schema)
├─ AuthError               # 401/403 authentication/authorization
├─ ResourceExhausted       # 429 quotas, rate limits (retry-after)
├─ TransientNetwork        # 5xx gateway/timeouts; retryable
├─ Unavailable             # 503 backend temporarily unavailable
└─ NotSupported            # 501/400 operation unsupported
```

LLM, Vector, and Embedding add specific subtypes (see §§8.5, 9.5, 10.3).

Errors **MUST** include machine-readable metadata when applicable:

```json
{
  "error": "ResourceExhausted",
  "message": "Rate limit exceeded",
  "code": "RATE_LIMIT",
  "retry_after_ms": 1200,
  "throttle_scope": "tenant:acme:llm",
  "hint": "Reduce concurrency or increase backoff"
}
```

### 6.4. Observability Interfaces

```python
from typing import Protocol, Mapping, Optional, Any

class MetricsSink(Protocol):
    def observe(
        self, *, component: str, op: str, ms: float, ok: bool,
        code: str = "OK", extra: Optional[Mapping[str, Any]] = None
    ) -> None: ...
    def counter(
        self, *, component: str, name: str, value: int = 1,
        extra: Optional[Mapping[str, Any]] = None
    ) -> None: ...
```

* `component` **MUST** be one of `graph|llm|vector|embedding`.
* Adapters **MUST** emit at least one `observe` per operation.
* Telemetry **MUST** be SIEM-safe: no raw tenant IDs, request bodies, prompts, vectors, or source texts.

---

## 7. Graph Protocol V1 Specification

### 7.1. Overview

Vendor-neutral interface for graph databases (Cypher, OpenCypher, Gremlin, GQL), standardizing CRUD, queries (sync/streaming), batch operations, and (optionally) schema management.

### 7.2. Data Types

```python
from typing import NewType, Tuple, Optional, Mapping, Any, Iterable, List, AsyncIterator

GraphID = NewType('GraphID', str)

@dataclass(frozen=True)
class GraphCapabilities:
    server: str
    version: str
    dialects: Tuple[str, ...] = ("cypher",)
    supports_txn: bool = True
    supports_schema_ops: bool = True
    max_batch_ops: Optional[int] = None
    max_query_length: Optional[int] = None
    read_after_write_consistency: str = "eventual"  # "session" | "strong"
```

### 7.3. Operations

#### 7.3.1. Vertex/Edge CRUD

```python
async def create_vertex(label: str, props: Mapping[str, Any], *, ctx: Optional[OperationContext]=None) -> GraphID
async def delete_vertex(vertex_id: GraphID, *, ctx: Optional[OperationContext]=None) -> None

async def create_edge(
    label: str, from_id: GraphID, to_id: GraphID, props: Mapping[str, Any], *,
    ctx: Optional[OperationContext]=None
) -> GraphID
async def delete_edge(edge_id: GraphID, *, ctx: Optional[OperationContext]=None) -> None
```

**Semantics (Normative):**

* `props` keys **MUST** be strings; values **MUST** be JSON-serializable.
* Create operations **SHOULD** accept `idempotency_key` and **MUST** be idempotent when provided.
* Deletes **MUST** be idempotent: deleting a non-existent ID returns success.

#### 7.3.2. Queries

```python
async def query(*, dialect: str, text: str, params: Optional[Mapping[str, Any]]=None,
                ctx: Optional[OperationContext]=None) -> List[Mapping[str, Any]]

async def stream_query(*, dialect: str, text: str, params: Optional[Mapping[str, Any]]=None,
                       ctx: Optional[OperationContext]=None) -> AsyncIterator[Mapping[str, Any]]
```

* `dialect` **MUST** be one advertised in capabilities.
* `params` **MUST** be bound safely; adapters **MUST** prevent injection by disallowing string interpolation.
* `stream_query` yields rows progressively; iterator close **MUST** free resources.

#### 7.3.3. Batch Operations

```python
async def bulk_vertices(
    vertices: Iterable[Tuple[str, Mapping[str, Any]]], *,
    ctx: Optional[OperationContext]=None
) -> List[GraphID]

async def batch(
    ops: Iterable[Mapping[str, Any]], *,
    ctx: Optional[OperationContext]=None
) -> List[Mapping[str, Any]]
```

* Item-level atomicity: partial failures **MUST** be reported per item.
* `batch` supports vendor-optimized pipelines; supported op codes **MUST** be documented in `capabilities.extensions`.

### 7.4. Dialects

Supported (non-exhaustive): **Cypher**, **OpenCypher**, **Gremlin**, **GQL (ISO)**. Unknown dialects **MUST** yield `NotSupported`.

### 7.5. Schema Operations (Optional)

If `supports_schema_ops=true`, adapters MAY expose:

```python
async def create_index(label: str, property: str, unique: bool=False, *, ctx: Optional[OperationContext]=None) -> None
async def drop_index(label: str, property: str, *, ctx: Optional[OperationContext]=None) -> None
```

Index creation events **MUST NOT** log sample property values.

---

## 8. LLM Protocol V1 Specification

### 8.1. Overview

Standardized interface for chat-style completions with synchronous and streaming modes, token accounting, and deterministic sampling control. Capability-based model discovery provides portability.

### 8.2. Data Types

```python
@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class LLMCompletion:
    text: str
    model: str
    model_family: str       # e.g., "gpt", "llama", "mistral"
    usage: TokenUsage
    finish_reason: str      # "stop" | "length" | "tool_call" | "content_filter"

@dataclass
class LLMChunk:
    text: str
    is_final: bool = False
    model: Optional[str] = None
    usage_so_far: Optional[TokenUsage] = None
```

### 8.3. Operations

```python
async def complete(
    *, messages: list[dict[str, str]],
    max_tokens: Optional[int]=None, temperature: Optional[float]=None,
    top_p: Optional[float]=None, model: Optional[str]=None,
    system_message: Optional[str]=None, frequency_penalty: Optional[float]=None,
    presence_penalty: Optional[float]=None, ctx: Optional[OperationContext]=None
) -> LLMCompletion

async def stream(
    *, messages: list[dict[str, str]],
    max_tokens: Optional[int]=None, temperature: Optional[float]=None,
    model: Optional[str]=None, system_message: Optional[str]=None,
    ctx: Optional[OperationContext]=None
) -> AsyncIterator[LLMChunk]

async def count_tokens(text: str, *, model: Optional[str]=None,
                       ctx: Optional[OperationContext]=None) -> int
```

**Message format (MUST):**

```json
[
  {"role":"system","content":"You are a helpful assistant."},
  {"role":"user","content":"Summarize this..."}
]
```

* Roles: `system|user|assistant|tool`. Unknown roles **MUST** yield `BadRequest`.
* `complete` **MUST** return deterministic `usage` when the provider supplies it; when not, adapters **MUST** estimate and set `usage.total_tokens >= 0` with `extensions.usage_estimated=true`.
* `stream` **MUST** deliver ordered chunks and set `is_final=true` on the terminal chunk.

### 8.4. Model Discovery

`capabilities()` **MUST** include:

```json
{
  "models": [
    {"name":"gpt-4.1-mini","family":"gpt","context_window":128000,"supports_tools":true},
    {"name":"llama3-70b","family":"llama","context_window":8192,"supports_tools":false}
  ],
  "sampling": {"temperature_range":[0.0,2.0],"top_p_range":[0.0,1.0]}
}
```

### 8.5. LLM-Specific Errors

* `ModelOverloaded` (subtype `Unavailable`): transient capacity pressure.
* `ContentFiltered` (subtype `BadRequest`): provider content policy triggered.
* Mitigation metadata MAY include `suggested_token_reduction` and `retry_after_ms`.

### 8.6. Tool Use / Function Calls (Optional)

Adapters MAY support tool calls via structured messages:

```json
{"role":"assistant","content":"","tool_calls":[{"name":"getWeather","arguments":{"city":"Paris"}}]}
```

If unsupported, adapters **MUST** return `NotSupported` and advertise `extensions.supports_tools=false`.

---

## 9. Vector Protocol V1 Specification

### 9.1. Overview

Standardized vector storage, search, and namespace isolation across providers with configurable distance metrics and metadata filtering.

### 9.2. Data Types

```python
from typing import Any, Optional, NewType, List, Dict
from dataclasses import dataclass

VectorID = NewType('VectorID', str)

@dataclass(frozen=True)
class Vector:
    id: VectorID
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None
    namespace: Optional[str] = None

@dataclass(frozen=True)
class VectorMatch:
    id: VectorID
    score: float
    metadata: Optional[Dict[str, Any]] = None

@dataclass(frozen=True)
class QueryResult:
    matches: List[VectorMatch]
    query_vector: List[float]
    namespace: str
    total_matches: int
```

**Spec types:**

```python
@dataclass(frozen=True)
class QuerySpec:
    vector: List[float]
    top_k: int
    namespace: str
    filter: Optional[Dict[str, Any]] = None
    metric: str = "cosine"        # "cosine" | "euclidean" | "dot"
    include_metadata: bool = True

@dataclass(frozen=True)
class UpsertSpec:
    vectors: List[Vector]
    namespace: str
    create_if_missing: bool = True

@dataclass(frozen=True)
class DeleteSpec:
    ids: Optional[List[VectorID]] = None
    namespace: str = ""
    filter: Optional[Dict[str, Any]] = None
```

### 9.3. Operations

```python
async def query(spec: QuerySpec, *, ctx: Optional[OperationContext]=None) -> QueryResult
async def upsert(spec: UpsertSpec, *, ctx: Optional[OperationContext]=None) -> dict
async def delete(spec: DeleteSpec, *, ctx: Optional[OperationContext]=None) -> dict
async def create_namespace(spec: dict, *, ctx: Optional[OperationContext]=None) -> dict
async def delete_namespace(namespace: str, *, ctx: Optional[OperationContext]=None) -> dict
```

**Semantics (Normative):**

* Vector dimension **MUST** match the configured index dimension; mismatches **MUST** raise `DimensionMismatch`.
* `query.top_k` **MUST** be > 0 and bounded by `limits.max_top_k` if present.
* Metadata filters **MUST** be applied pre-search where supported; otherwise adapters **MUST** document post-filter behavior.
* Scores **MUST** be consistent with `metric` and the “higher is better” convention unless explicitly documented.

### 9.4. Distance Metrics

* **Cosine** (default)
* **Euclidean (L2)**
* **Dot Product**

Adapters **MUST** advertise supported metrics and scoring conventions.

### 9.5. Vector-Specific Errors

* `DimensionMismatch` — query/upsert vector dimension incompatible (non-retryable).
* `IndexNotReady` — background index build incomplete (retryable with `retry_after_ms`).

---

## 10. Embedding Protocol V1 Specification

### 10.1. Overview

Vendor-neutral, production-grade interface for generating text embeddings (single/batch), counting tokens, discovering capabilities, and health reporting. Aligns with the Common Foundation and resilience semantics.

**Deliberate non-goals (Informative):** preprocessing/chunking, fine-tuning, provider-specific vector post-processing; those belong to adjacent layers.

### 10.2. Core Types (Conceptual)

**EmbeddingVector** — `{ vector: float[], text: string, model: string, dimensions: int }`
**EmbeddingResult** — `{ embeddings: EmbeddingVector[], model: string, total_tokens?: int, processing_time_ms?: number }`

Batch input convenience:
**EmbeddingBatch** — `{ texts: string[], model: string, truncate?: bool=true, normalize?: bool=false }`

### 10.3. Errors (Embedding-Specific)

* `TextTooLong` (subtype of `BadRequest`) — input exceeds model maximum when `truncate=false`.
* `ModelNotAvailable` (subtype of `NotSupported` or `Unavailable`) — requested model not present/disabled.

**Mitigation hints (SHOULD):** `retry_after_ms`, `resource_scope` (`"model"|"token_limit"|"rate_limit"`), `suggested_batch_reduction` (percentage).

### 10.4. Capabilities

Adapters MUST declare:

* `server`, `version`, `supported_models` (list of strings).
* Optional limits: `max_batch_size`, `max_text_length`, `max_dimensions`.
* Flags: `supports_normalization`, `supports_truncation`, `supports_token_counting`, `idempotent_operations`, `supports_multi_tenant`.

### 10.5. Operations (Normative)

**Specs (conceptual):**

* **EmbedSpec** — `{ text, model, truncate?: true, normalize?: false }`
* **BatchEmbedSpec** — `{ texts, model, truncate?: true, normalize?: false }`

**Interface expectations:**

* `capabilities()` returns **EmbeddingCapabilities**.
* `embed(spec)` returns **EmbeddingResult** with exactly one **EmbeddingVector**.
* `embed_batch(spec)` returns **EmbeddingResult** with a vector per input text, and a per-item failure list if partials occur.
* `count_tokens(text, model)` returns an integer (if unsupported, return `NotSupported`).
* `health()` returns `{ ok, server, version, models }`.

**Semantics:**

* `model` MUST be present in `supported_models`.
* If `normalize=true` and unsupported, return `NotSupported`.
* `embed_batch` MUST enforce `max_batch_size` and validate each `text`.
* If `truncate=false` and a text exceeds `max_text_length`, raise `TextTooLong`.

---

## 11. Cross-Protocol Patterns

### 11.1. Unified Error Handling

Centralize error mapping to the normalized taxonomy; use mitigation hints (`retry_after_ms`, `suggested_*`) for adaptive clients.

### 11.2. Consistent Observability

Emit `observe` and `counter` across components (`graph|llm|vector|embedding`) with common labels (`op`, `code`, `tenant_hash`). Never log raw prompts, vectors, source texts, or tenant identifiers.

### 11.3. Context Propagation

Create one `OperationContext` at ingress and pass through all protocol calls; update remaining time budget between calls.

### 11.4. Idempotency and Exactly-Once

Where `idempotency_key` is accepted, mutations MUST be exactly-once or return the prior committed result.

### 11.5. Pagination and Streaming

Graph MAY stream rows; LLM MUST stream with a terminal chunk; Vector pagination is capability-gated; Embedding is request/response (no streaming) in V1.

---

## 12. Error Handling and Resilience

### 12.1. Retry Semantics

**Retryable:** `TransientNetwork`, `ResourceExhausted` (respect `retry_after_ms`), `Unavailable`, `IndexNotReady`.
**Non-Retryable:** `BadRequest`, `AuthError`, `NotSupported`, `DimensionMismatch`, `ContentFiltered`, `TextTooLong` (unless truncation enabled).

### 12.2. Backoff and Jitter (RECOMMENDED)

Exponential backoff (100–500 ms base, ×2 factor, 10–30 s cap) with **full jitter**. Prefer server-provided `retry_after_ms` when present.

### 12.3. Circuit Breaking

Fail fast with `Unavailable` when breaker is open; optionally include `retry_after_ms` to reduce thundering herds.

### 12.4. Error Mapping Table (Normative)

| Error Class        | HTTP Mapping | Retryable | Client Guidance                                                |
| ------------------ | ------------ | --------- | -------------------------------------------------------------- |
| BadRequest         | 400          | No        | Fix parameters; do not retry                                   |
| AuthError          | 401/403      | No        | Refresh credentials; verify scopes                             |
| ResourceExhausted  | 429          | Yes       | Back off; **honor** `retry_after_ms`; reduce concurrency/batch |
| TransientNetwork   | 502/504      | Yes       | Exponential backoff + jitter; consider failover                |
| Unavailable        | 503          | Yes       | Trip/bias breaker; failover if possible                        |
| NotSupported       | 501/400      | No        | Probe with `capabilities()`; use alternative feature           |
| DimensionMismatch* | 400          | No        | Align dimensions to index                                      |
| IndexNotReady*     | 503          | Yes       | Retry after `retry_after_ms`                                   |
| ModelOverloaded**  | 503          | Yes       | Reduce rate; try alternate family/model                        |
| ContentFiltered**  | 400          | No        | Sanitize/adjust prompt                                         |
| TextTooLong***     | 400          | No        | Enable truncation or split text                                |

* Vector-specific. ** LLM-specific. *** Embedding-specific.

### 12.5. Partial Failure Contracts

Batch APIs MUST report per-item status. Non-atomic batches MUST NOT fail the entire batch due to a single item.

### 12.6. Backpressure Integration

Expose per-tenant semaphores to bound concurrency:

```python
async with backpressure.acquire(f"{tenant}:{component}:{op}"):
    return await adapter.operation(...)
```

---

## 13. Observability and Monitoring

### 13.1. Metrics Taxonomy (MUST)

**Operational:** Latency (p50/p90/p99) by `component+op+code`, error rate by class, concurrency/queue length.
**Business:** LLM tokens processed, vector upserts/searches, graph ops executed, **texts embedded**.
**Resource:** Cache hit ratios, rate-limit utilization, breaker state.

### 13.2. Structured Logging (MUST)

SIEM-safe logs (examples):

```json
{
  "kind": "vector.audit",
  "op": "query",
  "tenant_hash": "7d9f53d2f1ab",
  "trace_id": "abc-123",
  "status": "ok",
  "latency_ms": 45.2,
  "vectors_searched": 1,
  "matches_returned": 10
}
```

```json
{
  "kind": "embedding.audit",
  "op": "embed_batch",
  "tenant_hash": "7d9f53d2f1ab",
  "trace_id": "def-456",
  "status": "ok",
  "latency_ms": 37.8,
  "texts": 32,
  "model": "example-embed-1"
}
```

### 13.3. Distributed Tracing (SHOULD)

Propagate `traceparent`. Use standard span attributes (`component`, `op`, `tenant_hash`, `model`, counts). For LLM streaming, emit child events per chunk.

---

## 14. Security Considerations

### 14.1. Tenant Isolation (MUST)

* **Graph:** separate DBs/schemas or RBAC-scoped labels.
* **LLM:** per-tenant keys or dedicated instances.
* **Vector:** per-tenant namespaces/collections with ACLs.
* **Embedding:** per-tenant API keys; no cross-tenant caches without isolation keys.

### 14.2. Authentication and Authorization (MUST)

Credentials managed at adapter init; rotate via secret stores; never emit secrets in telemetry or errors.

### 14.3. Threat Model (SHOULD)

Address idempotency-key spoofing, prompt/graph injection, vector/embedding poisoning, and unbounded traversals via rate limiting, schema constraints, and timeouts.

---

## 15. Privacy Considerations

Do not log prompts, source texts, vectors, or raw tenant IDs. Hash tenant identifiers, time-bound log retention (≤30 days recommended), and require explicit, access-controlled opt-in for content retention. Provide DSAR-compatible export/delete pathways where applicable.

---

## 16. Performance Characteristics

### 16.1. Latency Targets (Indicative)

* **Graph:** CRUD 1–10 ms; queries 10–1000 ms; batch 100–5000 ms.
* **LLM:** token counting 1–5 ms; completion 100–30000 ms; streaming progressive.
* **Vector:** search 1–100 ms; batch upsert 10–1000 ms; index 1000–60000 ms.
* **Embedding:** single 5–50 ms; batch 10–1000 ms; token counting 1–5 ms.

Adapters SHOULD surface p90/p99 per op in `capabilities().limits`.

### 16.2. Concurrency Limits

Expose `concurrency`, `rate_limit_qps`, `max_batch_ops/top_k`, and memory considerations in capabilities.

### 16.3. Caching Strategies

* **LLM:** cache keyed by normalized messages + sampling params.
* **Vector:** cache identical `QuerySpec` results.
* **Graph:** fingerprint query+params.
* **Embedding:** content-addressable cache `(model, normalized_text)`; record only hashes in telemetry.

---

## 17. Implementation Guidelines

### 17.1. Adapter Pattern

Use base classes to centralize validation, error normalization, and metrics; focus provider code on business logic.

### 17.2. Validation (MUST)

Reject empty labels/texts, negative `top_k`, NaN/Inf vectors; enforce JSON-serializable `props/metadata`; validate message roles and `max_tokens` vs. window; enforce embedding `max_text_length` and `max_batch_size`.

### 17.3. Testing

**Unit:** dimension mismatch, role/parameter validation, error mapping, batching limits.
**Integration:** end-to-end pipelines (Graph → LLM → Vector → Embedding).
**Chaos:** simulate `Unavailable`, timeouts, and rate-limit storms; verify backoff and breaker behavior; ensure idempotence.

---

## 18. Versioning and Compatibility

### 18.1. Semantic Versioning (MUST)

**MAJOR** (breaking), **MINOR** (additive), **PATCH** (non-breaking fixes/docs).

### 18.2. Version Identification and Negotiation

Clients MAY specify `X-Adapter-Protocol: {component}/v{major}`. Adapters MUST reject incompatible majors with `NotSupported` and SHOULD advertise supported versions in `capabilities.protocol`.

### 18.3. Backward Compatibility

Guaranteed for additive parameters/methods and new capability flags. Not guaranteed for changing required params, removing elements, or altering error semantics outside a major.

### 18.4. Deprecation Policy

Announce, warn at runtime (where feasible), maintain ≥1 major, then remove in the subsequent major.

---

## 19. IANA Considerations

No IANA actions required.

---

## 20. References

### 20.1. Normative References

* **[RFC2119]** S. Bradner, “Key words for use in RFCs to Indicate Requirement Levels,” BCP 14.
* **[RFC8174]** B. Leiba, “Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words,” BCP 14 update.
* **[W3C-Trace-Context]** W3C Recommendation, “Trace Context.”
* **[OpenTelemetry-Spec]** OpenTelemetry Specification.
* **[SemVer]** Semantic Versioning 2.0.0.

### 20.2. Informative References

* Corpus GitHub Repository — [https://github.com/adapter-sdk](https://github.com/adapter-sdk)

---

## 21. Author’s Address

Corpus Working Group
Email: [standards@adaptersdk.org](mailto:standards@adaptersdk.org)
GitHub: [https://github.com/adapter-sdk/standards](https://github.com/adapter-sdk/standards)

---

# Appendix A — End-to-End Example (Normative)

```python
import time, random, asyncio
ctx = OperationContext(
    request_id="req_01HZX...",
    idempotency_key="idem_64f0...",
    deadline_ms=int(time.time()*1000)+30000,
    traceparent="00-4bf9...-00f0...-01",
    tenant="acme-corp",
    attrs={"user":"u_12345"}
)

# 1) Graph query for related documents
graph_rows = await graph_adapter.query(
    dialect="cypher",
    text="MATCH (u:User {id:$uid})-[:READ]->(d:Doc) RETURN d.id AS doc_id LIMIT 20",
    params={"uid":"u_12345"},
    ctx=ctx
)

# 2) LLM summarization
summary = await llm_adapter.complete(
    messages=[{"role":"system","content":"Summarize tersely."},
              {"role":"user","content":f"Summarize docs: {[r['doc_id'] for r in graph_rows]}"}],
    max_tokens=256,
    temperature=0.2,
    model="gpt-4.1-mini",
    ctx=ctx
)

# 3) Embedding + Vector search
embedding = embed(summary.text)  # implementation-specific
qr = await vector_adapter.query(
    QuerySpec(vector=embedding, top_k=10, namespace="acme.docs",
              filter={"doc_type":"kb","lang":{"$in":["en"]}}),
    ctx=ctx
)
```

**Resilience loop (backoff + jitter):**

```python
for attempt in range(5):
    try:
        res = await vector_adapter.upsert(UpsertSpec(...), ctx=ctx)
        break
    except ResourceExhausted as e:
        await asyncio.sleep((e.retry_after_ms or 500)/1000)
    except (TransientNetwork, Unavailable):
        sleep = min(2**attempt * 0.2, 5.0) * random.random()
        await asyncio.sleep(sleep)
```

---

# Appendix B — Capability Shapes (Illustrative)

**Graph**

```json
{
  "server": "janusgraph",
  "version": "1.0.0",
  "protocol": "graph/v1",
  "features": {"dialects":["gremlin"],"supports_txn":true,"supports_schema_ops":false},
  "limits": {"max_batch_ops": 2000, "concurrency": 128},
  "extensions": {"vendor:storage":"cassandra","read_after_write_consistency":"session"}
}
```

**LLM**

```json
{
  "server": "chat-gateway",
  "version": "2024-10-01",
  "protocol": "llm/v1",
  "features": {"supports_tools":true,"supports_streaming":true},
  "limits": {"max_ctx_window": 128000, "max_rps": 50},
  "models": [
    {"name":"gpt-4.1-mini","family":"gpt","context_window":128000,"supports_tools":true},
    {"name":"mistral-large","family":"mistral","context_window":32000,"supports_tools":false}
  ],
  "sampling":{"temperature_range":[0.0,2.0],"top_p_range":[0.0,1.0]}
}
```

**Vector**

```json
{
  "server": "pinecone",
  "version": "2.2",
  "protocol": "vector/v1",
  "features": {"supports_pagination":true,"supports_filters":true},
  "limits": {"dimension":1536,"max_top_k":1000,"max_batch":1000},
  "extensions": {"metric_default":"cosine"}
}
```

**Embedding**

```json
{
  "server": "embed-service",
  "version": "2025-01-15",
  "protocol": "embedding/v1",
  "supported_models": ["example-embed-1", "example-embed-2"],
  "max_batch_size": 512,
  "max_text_length": 16000,
  "max_dimensions": 1536,
  "supports_normalization": true,
  "supports_truncation": true,
  "supports_token_counting": true,
  "idempotent_operations": true,
  "supports_multi_tenant": true
}
```

---

# Appendix C — Wire-Level Envelopes (Optional)

**Request**

```json
{
  "op": "embedding.embed_batch",
  "ctx": {"request_id":"req_abc","deadline_ms":1730312345123,"tenant":"acme"},
  "args": {"texts":["a","b","c"],"model":"example-embed-1","normalize":true}
}
```

**Response**

```json
{
  "ok": true,
  "code": "OK",
  "ms": 38.4,
  "result": {"embeddings":[{"dimensions":1536,"model":"example-embed-1"}], "model":"example-embed-1"}
}
```

**Error**

```json
{
  "ok": false,
  "code": "TEXT_TOO_LONG",
  "error": "TextTooLong",
  "message": "Input exceeds model maximum",
  "retry_after_ms": null,
  "resource_scope": "token_limit"
}
```

---

# Appendix D — Content Redaction Patterns (Normative)

* Replace user/tenant identifiers with irreversible hashes before logging.
* Replace prompts and graph query text with SHA-256 fingerprints; store full content **only** when explicit debug sampling is enabled and access-controlled.
* For vectors **and embeddings**, log only dimension and norm statistics (mean/std); **never** raw vectors or source texts.
* Telemetry exporters **MUST** implement field-level redaction lists configurable per deployment.

---

# Appendix E — Implementation Status (Non-Normative)

* **Reference Adapters:** at least one open-source adapter per protocol family is RECOMMENDED for interoperability testing.
* **Interop Suite:** a conformance test suite SHOULD validate error mapping, capability negotiation, and streaming semantics.
* **Release Quality Bar:** adapters SHOULD demonstrate stability under soak tests (24h) and chaos scenarios (network partitions, rate-limit storms).

---

# Appendix F — Change Log / Revision History (Non-Normative)

* **v1.1 — Embedding Added:** Added Embedding Protocol V1 (§10), updated Common Foundation to include `embedding` component, expanded Observability, Security, Privacy, and Error Mapping to cover embeddings.
* **v1.0 — Initial RFC-Style:** Introduced BCP 14 requirements language, IANA Considerations, split Normative/Informative references, explicit Privacy Considerations, Conventions and Notation, error-mapping table, capability namespacing rules, and appendices for examples, redaction, and wire envelopes.
