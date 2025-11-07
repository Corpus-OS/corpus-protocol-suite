## Corpus SDK Specification

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

* [1. Introduction](#1-introduction)

  * [1.1. Motivation](#11-motivation)
  * [1.2. Scope](#12-scope)
  * [1.3. Design Philosophy](#13-design-philosophy)
* [2. Requirements Language](#2-requirements-language)
* [3. Terminology](#3-terminology)
* [4. Conventions and Notation](#4-conventions-and-notation)

  * [4.1. Wire-First Canonical Form (Normative)](#41-wire-first-canonical-form-normative)

    * [4.1.1. Envelopes and Content Types (MUST)](#411-envelopes-and-content-types-must)
    * [4.1.2. Version Identification (MUST)](#412-version-identification-must)
    * [4.1.3. Streaming Frames (MUST where applicable)](#413-streaming-frames-must-where-applicable)
    * [4.1.4. Transport Bindings for Streaming (Normative)](#414-transport-bindings-for-streaming-normative)
    * [4.1.5. Compatibility and Unknown Fields (MUST)](#415-compatibility-and-unknown-fields-must)
    * [4.1.6. Operation Registry (Normative)](#416-operation-registry-normative)
* [5. Architecture Overview](#5-architecture-overview)

  * [5.1. Protocol Relationships](#51-protocol-relationships)
  * [5.2. Layered Architecture](#52-layered-architecture)
  * [5.3. Implementation Profiles (Informative)](#53-implementation-profiles-informative)
* [6. Common Foundation](#6-common-foundation)

  * [6.1. Operation Context](#61-operation-context)
  * [6.2. Capability Discovery](#62-capability-discovery)
  * [6.3. Error Taxonomy](#63-error-taxonomy)
  * [6.4. Observability Interfaces](#64-observability-interfaces)
* [7. Graph Protocol V1.0 Specification](#7-graph-protocol-v10-specification)

  * [7.1. Overview](#71-overview)
  * [7.2. Data Types](#72-data-types)
  * [7.3. Operations](#73-operations)

    * [7.3.1. Vertex/Edge CRUD](#731-vertexedge-crud)
    * [7.3.2. Queries](#732-queries)

      * [Streaming Finalization (Normative)](#streaming-finalization-normative)
    * [7.3.3. Batch Operations](#733-batch-operations)
  * [7.4. Dialects](#74-dialects)
  * [7.5. Schema Operations (Optional)](#75-schema-operations-optional)
  * [7.6. Health](#76-health)
* [8. LLM Protocol V1.0 Specification](#8-llm-protocol-v10-specification)

  * [8.1. Overview](#81-overview)
  * [8.2. Data Types](#82-data-types)
  * [8.3. Operations](#83-operations)
  * [8.4. Model Discovery](#84-model-discovery)
  * [8.5. LLM-Specific Errors](#85-llm-specific-errors)
* [9. Vector Protocol V1.0 Specification](#9-vector-protocol-v10-specification)

  * [9.1. Overview](#91-overview)
  * [9.2. Data Types](#92-data-types)
  * [9.3. Operations](#93-operations)
  * [9.4. Distance Metrics](#94-distance-metrics)
  * [9.5. Vector-Specific Errors](#95-vector-specific-errors)
* [10. Embedding Protocol V1.0 Specification](#10-embedding-protocol-v10-specification)

  * [10.1. Overview](#101-overview)
  * [10.2. Data Types (Formal)](#102-data-types-formal)
  * [10.3. Operations (Normative Signatures)](#103-operations-normative-signatures)
  * [10.4. Errors (Embedding-Specific)](#104-errors-embedding-specific)
  * [10.5. Capabilities](#105-capabilities)
  * [10.6. Semantics](#106-semantics)
* [11. Cross-Protocol Patterns](#11-cross-protocol-patterns)

  * [11.1. Unified Error Handling](#111-unified-error-handling)
  * [11.2. Consistent Observability](#112-consistent-observability)
  * [11.3. Context Propagation](#113-context-propagation)
  * [11.4. Idempotency and Exactly-Once](#114-idempotency-and-exactly-once)
  * [11.5. Pagination and Streaming](#115-pagination-and-streaming)
  * [11.6. Caching (Implementation Guidance)](#116-caching-implementation-guidance)
  * [11.7. Best-Effort Distributed Transactions](#117-best-effort-distributed-transactions)
* [12. Error Handling and Resilience](#12-error-handling-and-resilience)

  * [12.1. Retry Semantics](#121-retry-semantics)
  * [12.2. Backoff and Jitter (RECOMMENDED)](#122-backoff-and-jitter-recommended)
  * [12.3. Circuit Breaking](#123-circuit-breaking)
  * [12.4. Error Mapping Table (Normative)](#124-error-mapping-table-normative)
  * [12.5. Partial Failure Contracts](#125-partial-failure-contracts)
  * [12.6. Backpressure Integration](#126-backpressure-integration)
* [13. Observability and Monitoring](#13-observability-and-monitoring)

  * [13.1. Metrics Taxonomy (MUST)](#131-metrics-taxonomy-must)
  * [13.2. Structured Logging (MUST)](#132-structured-logging-must)
  * [13.3. Distributed Tracing (SHOULD)](#133-distributed-tracing-should)
* [14. Security Considerations](#14-security-considerations)

  * [14.1. Tenant Isolation (MUST)](#141-tenant-isolation-must)
  * [14.2. Authentication and Authorization (MUST)](#142-authentication-and-authorization-must)
  * [14.3. Threat Model (SHOULD)](#143-threat-model-should)
  * [14.4. Mitigation Matrix (Normative)](#144-mitigation-matrix-normative)
* [15. Privacy Considerations](#15-privacy-considerations)
* [16. Performance Characteristics](#16-performance-characteristics)

  * [16.1. Latency Targets (Indicative)](#161-latency-targets-indicative)
  * [16.2. Concurrency Limits](#162-concurrency-limits)
  * [16.3. Caching Strategies](#163-caching-strategies)
* [17. Implementation Guidelines](#17-implementation-guidelines)

  * [17.1. Adapter Pattern](#171-adapter-pattern)
  * [17.2. Validation (MUST)](#172-validation-must)
  * [17.3. Testing](#173-testing)
* [18. Versioning and Compatibility](#18-versioning-and-compatibility)

  * [18.1. Semantic Versioning (MUST)](#181-semantic-versioning-must)
  * [18.2. Version Identification and Negotiation](#182-version-identification-and-negotiation)
  * [18.3. Backward Compatibility](#183-backward-compatibility)
  * [18.4. Deprecation Policy](#184-deprecation-policy)
* [19. IANA Considerations](#19-iana-considerations)
* [20. References](#20-references)

  * [20.1. Normative References](#201-normative-references)
  * [20.2. Informative References](#202-informative-references)
* [21. Author’s Address](#21-authors-address)
* [Appendix A — End-to-End Example (Normative)](#appendix-a--end-to-end-example-normative)
* [Appendix B — Capability Shapes (Illustrative)](#appendix-b--capability-shapes-illustrative)
* [Appendix C — Wire-Level Envelopes](#appendix-c--wire-level-envelopes)
* [Appendix D — Content Redaction Patterns (Normative)](#appendix-d--content-redaction-patterns-normative)
* [Appendix E — Implementation Status (Non-Normative)](#appendix-e--implementation-status-non-normative)
* [Appendix F — Change Log / Revision History (Non-Normative)](#appendix-f--change-log--revision-history-non-normative)

---

## 1. Introduction

### 1.1. Motivation

The proliferation of AI infrastructure has created a fragmented landscape of proprietary APIs and inconsistent interfaces. Fragmentation increases integration complexity, reduces operational visibility, and creates vendor lock-in. Enterprise teams need cohesive, auditable, and performance-predictable interfaces that allow swapping providers without rewriting core application code or telemetry pipelines.

### 1.2. Scope

This specification defines four complementary protocols:

* **Graph Protocol V1.0** — Vertex/edge CRUD, traversal, and multi-dialect query execution.
* **LLM Protocol V1.0** — Chat-style completion, streaming tokens, usage accounting.
* **Vector Protocol V1.0** — Vector upsert/delete, similarity search, and namespace management.
* **Embedding Protocol V1.0** — Text embedding generation (single/batch), token counting, capability discovery, and health reporting.

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
**Operation Context** — Metadata container for tracing, deadlines, tenancy, and cache hints.
**Capabilities** — Dynamically discoverable features and limits of an adapter.
**SIEM-Safe** — Observability that excludes PII and uses privacy-preserving identifiers.
**Idempotency Key** — Client-provided token guaranteeing idempotent semantics.
**Tenant Isolation** — Logical separation of data/control plane in multi-tenant deployments.
**Backpressure** — Cooperative throttling to keep systems within safe operating limits.
**Streaming Frame** — Single JSON object carrying a `data`, `end`, or `error` event in a streaming operation.
**Batch Partial Failure** — Outcome where some batch items succeed and others fail, with per-item status reported.

---

## 4. Conventions and Notation

* JSON keys are **case-sensitive**; unknown keys **MUST** be ignored by clients and servers.
* Durations are expressed in **milliseconds** unless otherwise specified.
* Examples are **non-normative** unless explicitly marked **(Normative)**.
* Field names use **lower_snake_case** unless specified.
* Error `code` values use `UPPER_SNAKE_CASE`.
* `tenant_hash` denotes a deterministic, irreversible hash of the tenant identifier.
* Unless otherwise stated, scores are **higher is better** (cosine/dot); distance metrics MAY be inverted to scores.

### 4.1. Wire-First Canonical Form (Normative)

This specification is **language-agnostic**. Python code is illustrative only. The **canonical interface** is the wire contract defined here and applied throughout.

#### 4.1.1. Envelopes and Content Types (MUST)

Non-streaming operations:

* `Content-Type: application/json`
* UTF-8 JSON documents.

**Request envelope:**

```json
{
  "op": "<component>.<operation>",
  "ctx": {},
  "args": {}
}
```

* `op` examples: `llm.complete`, `llm.stream`, `graph.query`, `vector.query`, `embedding.embed_batch`.

**Success response envelope:**

```json
{
  "ok": true,
  "code": "OK",
  "ms": 12.3,
  "result": {}
}
```

**Error response envelope:**

```json
{
  "ok": false,
  "code": "DEADLINE_EXCEEDED",
  "error": "DeadlineExceeded",
  "message": "Operation budget exhausted",
  "retry_after_ms": null,
  "details": {
    "provider_error_id": "..."
  }
}
```

* `code` MUST be a normalized error/OK code.
* `error` MUST be the normalized error class name when `ok=false`.

#### 4.1.2. Version Identification (MUST)

* Each protocol instance declares a protocol identifier: `{component}/v1.0`

  * Example: `"protocol": "vector/v1.0"`.
* Clients MAY send `X-Adapter-Protocol: {component}/v1.0`.
* All `v1.x` revisions of this specification are wire-compatible with `{component}/v1.0`.
* Breaking changes require `{component}/v2.0`.

#### 4.1.3. Streaming Frames (MUST where applicable)

For streaming operations (`llm.stream`, `graph.stream_query`):

Each frame is a JSON object:

**Data frame:**

```json
{
  "event": "data",
  "data": {}
}
```

**Terminal success frame:**

```json
{
  "event": "end",
  "code": "OK"
}
```

**Terminal error frame:**

```json
{
  "event": "error",
  "code": "UNAVAILABLE",
  "error": "Unavailable",
  "message": "..."
}
```

Rules:

* Exactly one terminal frame (`event: "end"` or `event: "error"`) per stream.
* No `data` frames after the terminal frame.
* Terminal error frame MUST use normalized error codes/classes.

#### 4.1.4. Transport Bindings for Streaming (Normative)

The logical frame model binds to transports:

**HTTP/1.1 + NDJSON:**

* Headers:

  * `Content-Type: application/x-ndjson`
  * `Transfer-Encoding: chunked`
  * `X-Protocol-Streaming: chunked-json`
* Each line is one JSON frame from §4.1.3.

**Server-Sent Events (SSE):**

* `Content-Type: text/event-stream`
* For each frame:

  * `event: <event>`
  * `data: {json}`
  * blank line.

**WebSocket:**

* Optional header: `X-Protocol-Streaming: websocket-json`
* Each message is one JSON frame.

**gRPC / HTTP/2 streaming:**

* Each streamed message logically corresponds to a frame from §4.1.3.

Adapters MAY support one or more transports. Supported transports SHOULD be advertised via:

```json
"extensions": {
  "streaming_transports": ["ndjson", "sse", "websocket"]
}
```

#### 4.1.5. Compatibility and Unknown Fields (MUST)

* Unknown keys (top-level or nested) MUST be ignored.
* Clients MUST NOT rely on field ordering.
* Numeric fields MUST respect type/range constraints defined in relevant sections.

#### 4.1.6. Operation Registry (Normative)

The following `op` strings **MUST** be used for the corresponding protocol operations and are reserved for V1.0:

| Section / Operation      | `op` String               |
| ------------------------ | ------------------------- |
| 7.3.1 `create_vertex`    | `graph.create_vertex`     |
| 7.3.1 `delete_vertex`    | `graph.delete_vertex`     |
| 7.3.1 `create_edge`      | `graph.create_edge`       |
| 7.3.1 `delete_edge`      | `graph.delete_edge`       |
| 7.3.2 `query`            | `graph.query`             |
| 7.3.2 `stream_query`     | `graph.stream_query`      |
| 7.3.3 `bulk_vertices`    | `graph.bulk_vertices`     |
| 7.3.3 `batch`            | `graph.batch`             |
| 7.5   `create_index`     | `graph.create_index`      |
| 7.5   `drop_index`       | `graph.drop_index`        |
| 7.6   `health`           | `graph.health`            |
| 8.3   `complete`         | `llm.complete`            |
| 8.3   `stream`           | `llm.stream`              |
| 8.3   `count_tokens`     | `llm.count_tokens`        |
| 8.4   `capabilities`     | `llm.capabilities`        |
| 9.3   `query`            | `vector.query`            |
| 9.3   `upsert`           | `vector.upsert`           |
| 9.3   `delete`           | `vector.delete`           |
| 9.3   `create_namespace` | `vector.create_namespace` |
| 9.3   `delete_namespace` | `vector.delete_namespace` |
| 9.x   `capabilities`     | `vector.capabilities`     |
| 10.3  `capabilities`     | `embedding.capabilities`  |
| 10.3  `embed`            | `embedding.embed`         |
| 10.3  `embed_batch`      | `embedding.embed_batch`   |
| 10.3  `count_tokens`     | `embedding.count_tokens`  |
| 10.3  `health`           | `embedding.health`        |

Adapters MAY expose additional `op` values via namespaced extensions (e.g., `vendorX.llm.complete_raw`), but MUST NOT alter the semantics of the reserved `op` strings above.

---

## 5. Architecture Overview

### 5.1. Protocol Relationships

```text
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

```text
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

### 5.3. Implementation Profiles (Informative)

Profiles define behavior; they MUST NOT change wire contracts.

* **Thin (default):** No-op caches/limiters/breakers; propagate deadlines downstream.
* **Standalone:** Local enforcement of deadlines, circuit breaker, token-bucket limiter, short-TTL in-memory caches for Embedding + LLM `complete`.

---

## 6. Common Foundation

### 6.1. Operation Context

```python
from dataclasses import dataclass
from typing import Any, Mapping, Optional, List

@dataclass(frozen=True)
class OperationContext:
    request_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    deadline_ms: Optional[int] = None
    traceparent: Optional[str] = None  # W3C Trace Context
    tenant: Optional[str] = None       # never logged raw
    attrs: Optional[Mapping[str, Any]] = None

    # Cache coordination (advisory; wire-compatible)
    cache_scope: Optional[str] = None        # "tenant" | "global" | "session"; default "tenant"
    cache_tags: Optional[List[str]] = None   # tags for cache keying/invalidation
```

**Normative behavior:**

* `request_id` SHOULD uniquely identify the operation end-to-end.
* `idempotency_key` MAY be supplied for mutating ops; when present:

  * Adapters MUST provide exactly-once external effects or return the prior committed result.
* `deadline_ms` SHOULD be treated as an absolute deadline:

  * If elapsed at call time → fail fast (`DeadlineExceeded` or `Unavailable` per component).
  * Remaining budget SHOULD be propagated downstream.
* `traceparent` MUST be forwarded unchanged.
* `tenant` MUST drive isolation; raw tenant MUST NOT be logged.
* `attrs` treated as a map; if absent, treated as empty.
* `cache_scope`:

  * If unset, treated as `"tenant"`.
  * MUST NOT weaken tenant isolation (no cross-tenant sharing unless explicitly configured).
* `cache_tags`:

  * MUST NOT contain secrets or raw tenant identifiers.
  * MAY be used when `cache.supports_tags=true`.
* `cache_scope` and `cache_tags` are **advisory**:

  * Adapters and backends MAY ignore these hints without violating this specification.

### 6.2. Capability Discovery

`capabilities()` MUST return:

```json
{
  "server": "example-backend",
  "version": "5.17.1",
  "protocol": "graph/v1.0",
  "features": {
    "dialects": ["cypher", "opencypher"],
    "supports_txn": true,
    "supports_schema_ops": true,
    "supports_streaming": false,
    "supports_deadline": true
  },
  "limits": {
    "max_batch_ops": 5000,
    "rate_limit_qps": 200,
    "concurrency": 256,
    "max_query_length": 100000
  },
  "cache": {
    "supports_tags": true,
    "max_ttl_ms": 3600000,
    "invalidation_webhook": false
  },
  "extensions": {
    "vendor:neo4j.routing": "cluster",
    "vendor:read_your_writes": true,
    "tag_model_in_metrics": false
  }
}
```

Rules:

* Unknown `features`, `limits`, `cache`, `extensions` keys MUST be ignored.
* `extensions` keys MUST be namespaced (`vendor:foo.bar`).
* `cache.supports_tags=true` signals support for `cache_tags`.
* Semantic changes to existing keys require a major version bump.

### 6.3. Error Taxonomy

```text
AdapterError (base)
├─ BadRequest              # 400 client errors (validation, schema)
├─ AuthError               # 401/403 authentication/authorization
├─ ResourceExhausted       # 429 quotas, rate limits
├─ TransientNetwork        # 5xx gateway/timeouts; retryable
├─ Unavailable             # 503 backend temporarily unavailable
├─ NotSupported            # 501/400 operation unsupported
└─ DeadlineExceeded        # 504 budget/deadline exhausted
```

Component-specific subtypes:

* LLM: `ModelOverloaded`, `ContentFiltered`.
* Vector: `DimensionMismatch`, `IndexNotReady`.
* Embedding: `TextTooLong`, `ModelNotAvailable`.

Errors MUST use normalized `code` and SHOULD include hints:

```json
{
  "ok": false,
  "code": "RATE_LIMIT",
  "error": "ResourceExhausted",
  "message": "Rate limit exceeded",
  "retry_after_ms": 1200,
  "throttle_scope": "tenant:acme:llm",
  "details": {
    "provider_error_id": "..."
  }
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

Requirements:

* `component` MUST be one of: `graph`, `llm`, `vector`, `embedding`.
* At least one `observe` per operation.
* Streaming operations MUST emit exactly one final `observe` for the overall outcome.
* Telemetry MUST be SIEM-safe:

  * Never log raw tenant IDs, prompts, vectors, or source texts.
* `extra` SHOULD use low-cardinality keys: `tenant_hash`, `deadline_bucket`, `cache_hit`, `rows`, `matches_returned`, `batch_size`, and optionally `model` when allowed.

---

## 7. Graph Protocol V1.0 Specification

### 7.1. Overview

Vendor-neutral interface for graph databases (Cypher, OpenCypher, Gremlin, GQL), covering CRUD, queries (sync/streaming), batch operations, and optional schema management.

### 7.2. Data Types

```python
from dataclasses import dataclass
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
    retryable_codes: Tuple[str, ...] = ()
    rate_limit_unit: str = "requests_per_second"  # or "tokens_per_minute"
    max_qps: Optional[int] = None
    idempotent_writes: bool = False
    supports_multi_tenant: bool = False
    supports_streaming: bool = False
    supports_bulk_ops: bool = False
    max_query_length: Optional[int] = None
```

### 7.3. Operations

#### 7.3.1. Vertex/Edge CRUD

```python
async def create_vertex(
    label: str,
    props: Mapping[str, Any],
    *,
    ctx: Optional[OperationContext] = None
) -> GraphID

async def delete_vertex(
    vertex_id: GraphID,
    *,
    ctx: Optional[OperationContext] = None
) -> None

async def create_edge(
    label: str,
    from_id: GraphID,
    to_id: GraphID,
    props: Mapping[str, Any],
    *,
    ctx: Optional[OperationContext] = None
) -> GraphID

async def delete_edge(
    edge_id: GraphID,
    *,
    ctx: Optional[OperationContext] = None
) -> None
```

Semantics:

* `props` keys MUST be strings; values MUST be JSON-serializable.
* Create ops SHOULD accept `idempotency_key` and MUST be idempotent when provided.
* Deletes MUST be idempotent: deleting a non-existent ID is success.

#### 7.3.2. Queries

```python
async def query(
    *,
    dialect: str,
    text: str,
    params: Optional[Mapping[str, Any]] = None,
    ctx: Optional[OperationContext] = None
) -> List[Mapping[str, Any]]

async def stream_query(
    *,
    dialect: str,
    text: str,
    params: Optional[Mapping[str, Any]] = None,
    ctx: Optional[OperationContext] = None
) -> AsyncIterator[Mapping[str, Any]]
```

* `dialect` MUST be from capabilities.
* `params` MUST be bound safely (no string interpolation).
* On deadline expiry:

  * Thin profile SHOULD surface `Unavailable`.
  * Standalone profile SHOULD surface `DeadlineExceeded`.

##### Streaming Finalization (Normative)

When exposed over the wire:

1. Emit zero or more `event:"data"` frames with rows.
2. Emit exactly one terminal frame:

   * `event:"end", code:"OK"` on success; or
   * `event:"error", ...` with normalized error.

As async iterator:

* `StopAsyncIteration` MUST release resources.
* Implementation MUST emit one final `observe` with `ok`/`code` reflecting the terminal state.
* No multiple terminal outcomes.

#### 7.3.3. Batch Operations

```python
async def bulk_vertices(
    vertices: Iterable[Tuple[str, Mapping[str, Any]]],
    *,
    ctx: Optional[OperationContext] = None
) -> List[GraphID]

async def batch(
    ops: Iterable[Mapping[str, Any]],
    *,
    ctx: Optional[OperationContext] = None
) -> List[Mapping[str, Any]]
```

* `max_batch_ops` from capabilities MUST be enforced.
* Partial failures:

  * MUST be reported per item (see §12.5).
  * MUST NOT fail entire batch solely due to one bad item when others can commit safely.
* Supported batch op codes MUST be discoverable via `capabilities.extensions`.

### 7.4. Dialects

Supported examples: `cypher`, `opencypher`, `gremlin`, `gql`.
Unknown dialect → `NotSupported`.

### 7.5. Schema Operations (Optional)

If `supports_schema_ops=true`:

```python
async def create_index(
    label: str,
    property: str,
    unique: bool = False,
    *,
    ctx: Optional[OperationContext] = None
) -> None

async def drop_index(
    label: str,
    property: str,
    *,
    ctx: Optional[OperationContext] = None
) -> None
```

Must not log sample values.

### 7.6. Health

`health()` SHOULD:

* Return `{ "ok": true, "server": "...", "version": "..." }` on success.
* On failure, normalize to `Unavailable("health check failed")` while including known `server`/`version` if possible.

---

## 8. LLM Protocol V1.0 Specification

### 8.1. Overview

Standardized interface for chat-style completions, streaming, token accounting, and model discovery with consistent observability and error semantics.

### 8.2. Data Types

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class LLMCompletion:
    text: str
    model: str
    model_family: str        # e.g. "gpt", "llama", "mistral"
    usage: TokenUsage
    finish_reason: str       # "stop" | "length" | "tool_call" | "content_filter"

@dataclass
class LLMChunk:
    text: str
    is_final: bool = False
    model: Optional[str] = None
    usage_so_far: Optional[TokenUsage] = None
```

### 8.3. Operations

```python
from typing import Optional, AsyncIterator, List, Dict

async def complete(
    *,
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    model: Optional[str] = None,
    system_message: Optional[str] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    ctx: Optional[OperationContext] = None
) -> LLMCompletion

async def stream(
    *,
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    system_message: Optional[str] = None,
    ctx: Optional[OperationContext] = None
) -> AsyncIterator[LLMChunk]

async def count_tokens(
    text: str,
    *,
    model: Optional[str] = None,
    ctx: Optional[OperationContext] = None
) -> int
```

Message format:

```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Summarize this..."}
]
```

Validation:

* `role ∈ {system,user,assistant,tool}`; else `BadRequest`.
* `temperature ∈ [0,2]`, `top_p ∈ (0,1]`,
  `frequency_penalty, presence_penalty ∈ [-2,2]`; else `BadRequest`.
* `count_tokens`:

  * If `supports_count_tokens=false`, MUST return `NotSupported`.

Streaming:

* Over wire: use frames in §4.1.3–4.1.4.
* In-process iterator:

  * Last chunk MUST have `is_final=true`.
  * One terminal `observe` MUST be emitted.

Deadline:

* If deadline pre-expired or elapsed during generation:

  * Return `DeadlineExceeded`.
  * For `stream`, terminate with error frame or final chunk indicating error.

### 8.4. Model Discovery

`capabilities()` MUST include:

```json
{
  "protocol": "llm/v1.0",
  "models": [
    {
      "name": "gpt-4.1-mini",
      "family": "gpt",
      "context_window": 128000,
      "supports_tools": true
    }
  ],
  "sampling": {
    "temperature_range": [0.0, 2.0],
    "top_p_range": [0.0, 1.0]
  },
  "features": {
    "supports_streaming": true,
    "supports_roles": true,
    "supports_json_output": false,
    "supports_parallel_tool_calls": false,
    "supports_deadline": true,
    "supports_count_tokens": true
  },
  "limits": {
    "max_context_length": 128000
  }
}
```

Clients MAY preflight `prompt_tokens + max_tokens <= max_context_length`.

### 8.5. LLM-Specific Errors

* `ModelOverloaded` (`Unavailable`): retry with backoff or alternate model.
* `ContentFiltered` (`BadRequest`): non-retryable without changing input.
* `DeadlineExceeded`: conditionally retryable per §12.1 (reduce work/extend deadline).

---

## 9. Vector Protocol V1.0 Specification

### 9.1. Overview

Standardized vector storage, similarity search, and namespace isolation with metadata filtering, distance metrics, and consistent errors.

### 9.2. Data Types

```python
from dataclasses import dataclass
from typing import Any, Optional, NewType, List, Dict

VectorID = NewType('VectorID', str)

@dataclass(frozen=True)
class Vector:
    id: VectorID
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None
    namespace: Optional[str] = None

@dataclass(frozen=True)
class VectorMatch:
    vector: Vector
    score: float
    distance: float

@dataclass(frozen=True)
class QueryResult:
    matches: List[VectorMatch]
    query_vector: List[float]
    namespace: str
    total_matches: int
```

Spec helpers:

```python
@dataclass(frozen=True)
class QuerySpec:
    vector: List[float]
    top_k: int
    namespace: str = "default"
    filter: Optional[Dict[str, Any]] = None
    include_metadata: bool = True
    include_vectors: bool = False  # if False, raw vectors MAY be omitted

@dataclass(frozen=True)
class UpsertSpec:
    vectors: List[Vector]
    namespace: str = "default"

@dataclass(frozen=True)
class DeleteSpec:
    ids: List[VectorID]
    namespace: str = "default"
    filter: Optional[Dict[str, Any]] = None
```

### 9.3. Operations

```python
from typing import Optional, Dict, Any

async def query(
    spec: QuerySpec,
    *,
    ctx: Optional[OperationContext] = None
) -> QueryResult

async def upsert(
    spec: UpsertSpec,
    *,
    ctx: Optional[OperationContext] = None
) -> Dict[str, Any]

async def delete(
    spec: DeleteSpec,
    *,
    ctx: Optional[OperationContext] = None
) -> Dict[str, Any]

async def create_namespace(
    spec: Dict[str, Any],
    *,
    ctx: Optional[OperationContext] = None
) -> Dict[str, Any]

async def delete_namespace(
    namespace: str,
    *,
    ctx: Optional[OperationContext] = None
) -> Dict[str, Any]
```

Semantics:

* Vector dimensions MUST match index dimension; else `DimensionMismatch`.
* `top_k > 0` and MUST respect `limits.max_top_k`.
* Filters MUST be applied pre-search when supported; if post-filtering is used, behavior MUST be documented.
* Scores adopt “higher is better” unless clearly documented otherwise.
* `include_vectors=false`:

  * Raw vectors MAY be omitted; IDs/metadata still returned.

### 9.4. Distance Metrics

Adapters MUST advertise:

* Supported metrics (`cosine`, `euclidean`, `dot`).
* Whether they return scores, distances, or both; and which direction is “better”.

### 9.5. Vector-Specific Errors

* `DimensionMismatch` → HTTP 400, non-retryable.
* `IndexNotReady` → HTTP 503, retryable with `retry_after_ms`.

---

## 10. Embedding Protocol V1.0 Specification

### 10.1. Overview

Vendor-neutral interface for generating embeddings (single/batch), counting tokens, and health checking, with deterministic behavior and partial-failure reporting.

### 10.2. Data Types (Formal)

```python
from dataclasses import dataclass
from typing import List, Optional, Mapping, Any

@dataclass(frozen=True)
class EmbeddingVector:
    vector: List[float]
    text: Optional[str]      # MAY be omitted or redacted
    model: str
    dimensions: int

@dataclass(frozen=True)
class EmbeddingFailure:
    index: int               # index in input list
    error: str               # normalized code, e.g. "TEXT_TOO_LONG"
    message: Optional[str] = None
    details: Optional[Mapping[str, Any]] = None  # SIEM-safe; no full text

@dataclass(frozen=True)
class EmbeddingResult:
    embeddings: List[EmbeddingVector]
    model: str
    total_tokens: Optional[int] = None
    processing_time_ms: Optional[float] = None
    failures: Optional[List[EmbeddingFailure]] = None  # REQUIRED if any failures
```

```python
@dataclass(frozen=True)
class EmbeddingCapabilities:
    server: str
    version: str
    supported_models: List[str]
    max_batch_size: Optional[int] = None
    max_text_length: Optional[int] = None
    max_dimensions: Optional[int] = None
    supports_normalization: bool = False
    normalizes_at_source: bool = False
    supports_truncation: bool = True
    supports_token_counting: bool = True
    supports_deadline: bool = True
    idempotent_operations: bool = True
    supports_multi_tenant: bool = True
```

### 10.3. Operations (Normative Signatures)

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen=True)
class EmbedSpec:
    text: str
    model: str
    truncate: bool = True
    normalize: bool = False

@dataclass(frozen=True)
class BatchEmbedSpec:
    texts: List[str]
    model: str
    truncate: bool = True
    normalize: bool = False

async def capabilities(
    *,
    ctx: Optional[OperationContext] = None
) -> EmbeddingCapabilities

async def embed(
    spec: EmbedSpec,
    *,
    ctx: Optional[OperationContext] = None
) -> EmbeddingResult

async def embed_batch(
    spec: BatchEmbedSpec,
    *,
    ctx: Optional[OperationContext] = None
) -> EmbeddingResult

async def count_tokens(
    text: str,
    *,
    model: Optional[str] = None,
    ctx: Optional[OperationContext] = None
) -> int

async def health(
    *,
    ctx: Optional[OperationContext] = None
) -> Mapping[str, Any]  # { "ok": bool, "server": str, "version": str, "models": [...] }
```

### 10.4. Errors (Embedding-Specific)

* `TextTooLong` (`BadRequest`):

  * Raised when `truncate=false` and input exceeds `max_text_length`.
* `ModelNotAvailable` (`NotSupported` or `Unavailable`):

  * Model missing, disabled, or capacity-constrained.
* `DeadlineExceeded`:

  * Budget exhausted; conditionally retryable per §12.1 only if request is reduced.

Implementations SHOULD include hints:

```json
{
  "ok": false,
  "code": "TEXT_TOO_LONG",
  "error": "TextTooLong",
  "message": "Input exceeds 16000 tokens",
  "details": {
    "max_text_length": 16000,
    "suggested_batch_reduction": 0.5
  }
}
```

### 10.5. Capabilities

Adapters MUST declare:

* `server`, `version`, `supported_models`.
* Optional: `max_batch_size`, `max_text_length`, `max_dimensions`.
* Booleans: `supports_normalization`, `normalizes_at_source`, `supports_truncation`, `supports_token_counting`, `supports_deadline`, `idempotent_operations`, `supports_multi_tenant`.

### 10.6. Semantics

* `model` MUST be in `supported_models`.
* If `normalize=true` but unsupported → `NotSupported`.
* `embed_batch`:

  * MUST validate each text.
  * MUST enforce `max_batch_size` if present.
  * On per-item failures, MUST populate `failures` with indices; overall response MAY be `ok=true` with `code="PARTIAL_SUCCESS"` at envelope level.
* If `truncate=false` and text too long → `TextTooLong`.
* Deadline pre-expired or elapsed → `DeadlineExceeded`.

---

## 11. Cross-Protocol Patterns

### 11.1. Unified Error Handling

* ALL adapters MUST map provider-specific errors to common taxonomy (§6.3).
* Error envelopes MUST include `code` and `message`.
* SHOULD include:

  * `retry_after_ms` for throttling.
  * `throttle_scope`.
  * `suggested_batch_reduction` when relevant.
  * `details.provider_error_id` when available.

### 11.2. Consistent Observability

* Use `MetricsSink` across `graph|llm|vector|embedding`.
* MUST:

  * Emit exactly one final `observe` for each operation.
  * For streams, final `observe` reflects overall success/failure.
  * Include `deadline_bucket` from `<1s|<5s|<15s|<60s|>=60s`.
* SHOULD:

  * Include `cache_hit`, `rows`, `matches_returned`, `batch_size`.
* MUST NOT:

  * Log raw prompts, vectors, graph queries, or raw tenant IDs.

### 11.3. Context Propagation

* Single `OperationContext` flows through:

  * Graph → LLM → Embedding → Vector, etc.
* MUST:

  * Forward `traceparent` unchanged.
* SHOULD:

  * Recompute remaining deadline between calls.

### 11.4. Idempotency and Exactly-Once

* For operations accepting `idempotency_key`:

  * MUST treat duplicate keys as safe replays.
  * MUST either:

    * Reuse previous result, or
    * Ensure no duplicate side-effects.
* SHOULD:

  * Record SIEM-safe idempotency audits.

### 11.5. Pagination and Streaming

* Graph:

  * MAY stream query results; MUST obey streaming semantics.
* LLM:

  * `stream` MUST obey frame semantics and final chunk semantics.
* Vector:

  * Pagination support is capability-gated; if unsupported, MUST publish limits.
* Embedding:

  * V1.0 is request/response only (no streaming).

### 11.6. Caching (Implementation Guidance)

* Good candidates: Embedding, deterministic LLM completions.
* Router-level preferred for: Graph, Vector.
* Cache key MUST:

  * Use content hashes (e.g., `sha256`), not raw text.
  * Include `model`, parameters, and `tenant_hash` when `cache_scope="tenant"`.
* When `cache.supports_tags=true`:

  * Adapters SHOULD use `cache_tags` for grouping entries for invalidation.
* `cache_scope` and `cache_tags` are hints:

  * Backends MAY ignore them without violating the spec.
* Caches MUST respect tenant isolation.

### 11.7. Best-Effort Distributed Transactions

To coordinate cross-protocol operations (e.g., embed + upsert + graph edge):

```python
from typing import Protocol, Callable, List, Any

class TransactionCoordinator(Protocol):
    async def with_transaction(
        self,
        ops: List[Callable[[], Any]],
        ctx: OperationContext
    ) -> List[Any]:
        """
        Execute operations with best-effort atomicity across providers.

        Requirements:
        - On full success: return list of results.
        - On failure:
          - MUST raise CompositeError containing per-op status.
          - SHOULD attempt provider-level rollback where supported.
          - MUST NOT violate tenant isolation.
          - MUST NOT assume global ACID across vendors.
        """
```

This pattern is RECOMMENDED but OPTIONAL. Implementations MUST document which operations support compensation and how.

---

## 12. Error Handling and Resilience

### 12.1. Retry Semantics

**Retryable:**

* `TransientNetwork`
* `ResourceExhausted` (honor `retry_after_ms`)
* `Unavailable`
* `IndexNotReady`

**Conditionally Retryable: `DeadlineExceeded`**

A client MAY retry only if it changes the request:

1. Extend `deadline_ms` in new `OperationContext`.
2. LLM: reduce `max_tokens`, simplify prompt, or adjust settings to reduce work.
3. Embedding: reduce `batch_size`, shorten texts, or set `truncate=true`.
4. Vector: reduce `top_k`, simplify filters.
5. Graph: reduce traversal depth/complexity or limit result rows.

Without such changes, `DeadlineExceeded` MUST be treated as non-retryable.

**Non-Retryable:**

* `BadRequest`, `AuthError`, `NotSupported`
* `DimensionMismatch`, `ContentFiltered`, `TextTooLong` (unless enabling truncation or splitting text)
* Other validation errors.

### 12.2. Backoff and Jitter (RECOMMENDED)

* Exponential backoff:

  * Base: 100–500 ms
  * Factor: ×2
  * Cap: 10–30 s
* Use **full jitter**.
* Prefer server `retry_after_ms` when present.

### 12.3. Circuit Breaking

* On repeated `Unavailable`/`TransientNetwork`:

  * MAY open circuit.
  * While open:

    * Fail fast with normalized `Unavailable("circuit open")`.
    * SHOULD include indicative `retry_after_ms`.
* Circuit breakers are per-tenant and per-operation where possible.

### 12.4. Error Mapping Table (Normative)

| Error Class        | HTTP    | Retryable   | Client Guidance                                               |
| ------------------ | ------- | ----------- | ------------------------------------------------------------- |
| BadRequest         | 400     | No          | Fix request; do not retry                                     |
| AuthError          | 401/403 | No          | Refresh credentials; fix scopes                               |
| ResourceExhausted  | 429     | Yes         | Back off; honor `retry_after_ms`; reduce concurrency/batch    |
| TransientNetwork   | 502/504 | Yes         | Retry with backoff + jitter; consider failover                |
| Unavailable        | 503     | Yes         | Retry; use breaker/failover                                   |
| NotSupported       | 400/501 | No          | Use `capabilities()`; switch feature/model                    |
| DimensionMismatch* | 400     | No          | Fix vector dimension                                          |
| IndexNotReady*     | 503     | Yes         | Retry after `retry_after_ms`                                  |
| ModelOverloaded**  | 503     | Yes         | Retry with backoff; switch model/family                       |
| ContentFiltered**  | 400     | No          | Sanitize input                                                |
| TextTooLong***     | 400     | No          | Truncate or chunk inputs                                      |
| DeadlineExceeded   | 504     | Conditional | Retry only if reducing work or extending deadline (see §12.1) |

* Vector-specific
** LLM-specific
*** Embedding-specific

### 12.5. Partial Failure Contracts

For non-atomic batch operations (e.g., `embed_batch`, vector `upsert`, graph `batch`):

* Transport-level `ok` may be `true` with `code="PARTIAL_SUCCESS"` when:

  * At least one item succeeded.
  * At least one item failed.
* Item-level failures MUST include:

  * `index` (input position),
  * `code` (normalized),
  * optional `message`,
  * optional SIEM-safe `details`.
* Successful items MUST appear in normal result collections.
* MUST NOT drop failed items silently.

`EmbeddingResult.failures` is the normative pattern; other batch APIs SHOULD mirror it.

### 12.6. Backpressure Integration

Implement cooperative backpressure, e.g.:

```python
async with backpressure.acquire(f"{ctx.tenant or 'public'}:{component}:{op}"):
    return await adapter.operation(...)
```

* Limits SHOULD be per-tenant where applicable.
* On saturation, surface `ResourceExhausted` or `Unavailable`.

---

## 13. Observability and Monitoring

### 13.1. Metrics Taxonomy (MUST)

Track:

* Latency (p50/p90/p99) by `component`, `op`, `code`.
* Error rate by normalized class.
* Concurrency and queue length.
* LLM: tokens processed.
* Vector: upserts, queries.
* Graph: CRUD/query volume.
* Embedding: texts/embeddings processed.
* Cache hit ratios; breaker state.

### 13.2. Structured Logging (MUST)

Examples:

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
  "status": "partial_success",
  "latency_ms": 37.8,
  "texts": 32,
  "model": "example-embed-1",
  "failures": 1
}
```

Logs MUST be SIEM-safe per §15.

### 13.3. Distributed Tracing (SHOULD)

* Propagate `traceparent`.
* Use spans with attributes:

  * `component`, `op`, `tenant_hash`, `model`, counts.
* For streams:

  * Child events per chunk are OPTIONAL; final span status MUST match final stream outcome.

---

## 14. Security Considerations

### 14.1. Tenant Isolation (MUST)

* Graph: separate schemas/DBs or strong RBAC.
* LLM: per-tenant credentials; no cross-tenant training without explicit policy.
* Vector: namespaces/collections scoped per tenant.
* Embedding: per-tenant API keys; caches keyed by tenant hash.

### 14.2. Authentication and Authorization (MUST)

* Credentials provisioned at adapter initialization.
* MUST NOT be logged or echoed.
* Authorization checks MUST align with tenant isolation and operation type.

### 14.3. Threat Model (SHOULD)

Implementations SHOULD address:

* Prompt and query injection.
* Vector/embedding poisoning.
* Idempotency-key spoofing.
* Resource exhaustion / DoS.
* Cross-tenant leakage.
* Tool/model misuse.

### 14.4. Mitigation Matrix (Normative)

| Threat Category                | Component         | MUST                                                   | SHOULD                                                           |
| ------------------------------ | ----------------- | ------------------------------------------------------ | ---------------------------------------------------------------- |
| Injection (prompt/query)       | LLM, Graph        | Parameterized bindings; strict role/dialect validation | Allowlist dialects/models; use templates and escape user content |
| Data exfiltration via logs     | All               | SIEM-safe logs; no prompts/vectors/raw tenants         | Field redaction; retention ≤30 days                              |
| Idempotency-key spoofing       | All mutating      | Treat duplicates as replays; exactly-once semantics    | Use HMAC/nonce-scoped keys with TTL                              |
| Poisoning (vectors/embeddings) | Vector, Embedding | Validate dimensions; reject NaN/Inf; enforce ACLs      | Outlier detection; quarantine suspicious batches                 |
| Resource exhaustion / DoS      | All               | Enforce deadlines; per-tenant rate limits              | Adaptive backoff; tuned circuit breakers                         |
| Cross-tenant leakage           | All               | Strict tenant isolation; tenant in cache keys          | Periodic isolation tests; policy-as-code in CI/CD                |
| Tool/model misuse              | LLM               | Enforce param ranges; map filters to `ContentFiltered` | Guardrails and policies for model fallback                       |
| Unbounded traversals           | Graph             | Depth/row limits; timeouts; schema/RBAC                | Configurable caps; anomaly alerts on traversals                  |

---

## 15. Privacy Considerations

* MUST NOT log:

  * Raw prompts, source texts, vectors, or tenant IDs.
* SHOULD:

  * Hash tenant identifiers.
  * Limit log retention (≤30 days recommended).
  * Provide DSAR-aligned mechanisms for deletion/exports where applicable.
* Content retention for training/analytics MUST be explicit, access-controlled, and policy-governed.

---

## 16. Performance Characteristics

### 16.1. Latency Targets (Indicative)

* Graph:

  * CRUD: 1–10 ms
  * Queries: 10–1000 ms
  * Batch: 100–5000 ms
* LLM:

  * Token counting: 1–5 ms
  * Completion: 100–30000 ms (model-dependent)
* Vector:

  * Search: 1–100 ms
  * Batch upsert: 10–1000 ms
* Embedding:

  * Single: 5–50 ms
  * Batch: 10–1000 ms

Adapters SHOULD expose observed p90/p99 via `capabilities.limits` or metrics.

### 16.2. Concurrency Limits

* `capabilities.limits` SHOULD include:

  * `concurrency`
  * `rate_limit_qps`
  * `max_batch_ops` / `max_top_k`
* Clients SHOULD respect published limits.

### 16.3. Caching Strategies

* Embeddings:

  * Content-addressable by `(model, normalize, sha256(text))`.
* LLM:

  * Cache deterministic outputs only; keys include sampling params & system hash.
* Vector:

  * Prefer router-layer caching; key by full query spec hash.
* Graph:

  * Cache read-only queries keyed by `dialect + text + params` hash.
* Caches MUST respect tenant isolation.

---

## 17. Implementation Guidelines

### 17.1. Adapter Pattern

* Centralize:

  * Validation
  * Error normalization
  * Metrics/tracing
* Provider-specific shims:

  * Translate to provider SDK.
* `capabilities()`:

  * SHOULD be implemented and cached with short TTL.

### 17.2. Validation (MUST)

Adapters MUST:

* Reject:

  * Empty labels where disallowed.
  * Negative `top_k`.
  * NaN/Inf vectors.
* Enforce:

  * JSON-serializable `props/metadata`.
  * Valid message roles & sampling ranges.
  * Context-length limits where known.
  * Embedding `max_text_length` / `max_batch_size`.
  * Vector dimension matching.
* Apply:

  * `TextTooLong`, `DimensionMismatch`, etc., per contract.

### 17.3. Testing

* Unit:

  * Validation, error mapping, idempotency, partial-failure encoding.
* Integration:

  * Graph → LLM → Embedding → Vector flows.
* Chaos:

  * Inject `Unavailable`, rate-limits, timeouts.
  * Verify retries, breakers, and that idempotency + streaming cleanup work.

---

## 18. Versioning and Compatibility

### 18.1. Semantic Versioning (MUST)

Use `MAJOR.MINOR.PATCH`:

* MAJOR: breaking.
* MINOR: additive, compatible.
* PATCH: fixes, clarifications.

### 18.2. Version Identification and Negotiation

* Clients MAY send: `X-Adapter-Protocol: {component}/v1.0`.
* Adapters MUST:

  * Reject unsupported major versions with `NotSupported`.
  * Advertise supported versions in `capabilities.protocol`.

### 18.3. Backward Compatibility

* Additive fields/flags/methods within `v1.x` MUST be backward compatible.
* Behavior changes requiring clients to change MUST wait for `v2.0`.

### 18.4. Deprecation Policy

* Announce deprecations.
* MAY emit runtime warnings (e.g., headers, logs).
* Maintain deprecated features for at least one minor version before removal in next major.

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

* Corpus GitHub Repository — `https://github.com/adapter-sdk`

---

## 21. Author’s Address

Corpus Working Group
Email: [standards@adaptersdk.org](mailto:standards@adaptersdk.org)
GitHub: `https://github.com/adapter-sdk/standards`

---

## Appendix A — End-to-End Example (Normative)

```python
import time, random, asyncio
from typing import Any, Mapping

# Construct a shared context
now_ms = int(time.time() * 1000)
ctx = OperationContext(
    request_id="req_01HZX...",
    idempotency_key="idem_64f0...",
    deadline_ms=now_ms + 30000,
    traceparent="00-4bf9...-00f0...-01",
    tenant="acme-corp",
    attrs={"user": "u_12345"},
    cache_scope="tenant",
    cache_tags=["kb", "user:u_12345"]
)

# 1) Graph query for related documents
graph_rows = await graph_adapter.query(
    dialect="cypher",
    text=(
        "MATCH (u:User {id:$uid})-[:READ]->(d:Doc) "
        "RETURN d.id AS doc_id LIMIT 20"
    ),
    params={"uid": "u_12345"},
    ctx=ctx
)

doc_ids = [r["doc_id"] for r in graph_rows]

# 2) LLM summarization of those docs
summary = await llm_adapter.complete(
    messages=[
        {"role": "system", "content": "Summarize tersely."},
        {
            "role": "user",
            "content": f"Summarize docs: {doc_ids}"
        }
    ],
    max_tokens=256,
    temperature=0.2,
    model="gpt-4.1-mini",
    ctx=ctx
)

# 3) Embedding + Vector search for similar docs
embed_result = await embedding_adapter.embed(
    EmbedSpec(
        text=summary.text,
        model="example-embed-1",
        truncate=True,
        normalize=True
    ),
    ctx=ctx
)

embedding_vec = embed_result.embeddings[0].vector

qr = await vector_adapter.query(
    QuerySpec(
        vector=embedding_vec,
        top_k=10,
        namespace="acme.docs",
        filter={"doc_type": "kb", "lang": {"$in": ["en"]}},
        include_metadata=True,
        include_vectors=False
    ),
    ctx=ctx
)

# 4) Resilient upsert with backoff (illustrative)
for attempt in range(5):
    try:
        await vector_adapter.upsert(
            UpsertSpec(
                vectors=[
                    Vector(
                        id="doc:new",
                        vector=embedding_vec,
                        metadata={"source": "summary"},
                        namespace="acme.docs"
                    )
                ]
            ),
            ctx=ctx
        )
        break
    except ResourceExhausted as e:
        delay_ms = getattr(e, "retry_after_ms", None) or 500
        await asyncio.sleep(delay_ms / 1000)
    except (TransientNetwork, Unavailable):
        sleep = min((2 ** attempt) * 0.2, 5.0) * random.random()
        await asyncio.sleep(sleep)
    except DeadlineExceeded:
        # Further retries without changing workload/deadline would violate §12.1
        break
```

---

## Appendix B — Capability Shapes (Illustrative)

### Graph

```json
{
  "server": "janusgraph",
  "version": "1.0.0",
  "protocol": "graph/v1.0",
  "features": {
    "dialects": ["gremlin"],
    "supports_txn": true,
    "supports_schema_ops": false,
    "supports_streaming": true,
    "supports_deadline": true
  },
  "limits": {
    "max_batch_ops": 2000,
    "concurrency": 128
  },
  "cache": {
    "supports_tags": true,
    "max_ttl_ms": 600000,
    "invalidation_webhook": false
  },
  "extensions": {
    "vendor:storage": "cassandra",
    "read_after_write_consistency": "session"
  }
}
```

### LLM

```json
{
  "server": "chat-gateway",
  "version": "2024-10-01",
  "protocol": "llm/v1.0",
  "features": {
    "supports_tools": true,
    "supports_streaming": true,
    "supports_deadline": true,
    "supports_count_tokens": true
  },
  "limits": {
    "max_context_length": 128000
  },
  "sampling": {
    "temperature_range": [0.0, 2.0],
    "top_p_range": [0.0, 1.0]
  },
  "cache": {
    "supports_tags": true,
    "max_ttl_ms": 300000,
    "invalidation_webhook": true
  },
  "models": [
    {
      "name": "gpt-4.1-mini",
      "family": "gpt",
      "context_window": 128000,
      "supports_tools": true
    }
  ]
}
```

### Vector

```json
{
  "server": "pinecone",
  "version": "2.2",
  "protocol": "vector/v1.0",
  "features": {
    "supports_pagination": true,
    "supports_filters": true,
    "supports_deadline": true
  },
  "limits": {
    "dimension": 1536,
    "max_top_k": 1000,
    "max_batch": 1000
  },
  "cache": {
    "supports_tags": false
  },
  "extensions": {
    "metric_default": "cosine"
  }
}
```

### Embedding

```json
{
  "server": "embed-service",
  "version": "2025-01-15",
  "protocol": "embedding/v1.0",
  "supported_models": [
    "example-embed-1",
    "example-embed-2"
  ],
  "max_batch_size": 512,
  "max_text_length": 16000,
  "max_dimensions": 1536,
  "supports_normalization": true,
  "normalizes_at_source": false,
  "supports_truncation": true,
  "supports_token_counting": true,
  "supports_deadline": true,
  "idempotent_operations": true,
  "supports_multi_tenant": true,
  "cache": {
    "supports_tags": true,
    "max_ttl_ms": 3600000
  }
}
```

---

## Appendix C — Wire-Level Envelopes

These examples apply the canonical rules in §4.1.

**Embedding Batch Request**

```json
{
  "op": "embedding.embed_batch",
  "ctx": {
    "request_id": "req_abc",
    "deadline_ms": 1730312345123,
    "tenant": "acme",
    "cache_scope": "tenant",
    "cache_tags": ["kb"]
  },
  "args": {
    "texts": ["a", "b", "c"],
    "model": "example-embed-1",
    "truncate": true,
    "normalize": true
  }
}
```

**Embedding Batch Partial-Success Response**

```json
{
  "ok": true,
  "code": "PARTIAL_SUCCESS",
  "ms": 38.4,
  "result": {
    "model": "example-embed-1",
    "embeddings": [
      {
        "vector": [0.01, 0.02],
        "model": "example-embed-1",
        "dimensions": 2
      },
      {
        "vector": [0.03, 0.04],
        "model": "example-embed-1",
        "dimensions": 2
      }
    ],
    "failures": [
      {
        "index": 2,
        "error": "TEXT_TOO_LONG",
        "message": "Input exceeds max_text_length"
      }
    ]
  }
}
```

**Streaming LLM over NDJSON**

HTTP headers:

```text
Content-Type: application/x-ndjson
Transfer-Encoding: chunked
X-Protocol-Streaming: chunked-json
```

Body (each line = one frame):

```json
{"event":"data","data":{"text":"Hello","is_final":false}}
{"event":"data","data":{"text":" world","is_final":false}}
{"event":"end","code":"OK"}
```

**Error Envelope Example**

```json
{
  "ok": false,
  "code": "DEADLINE_EXCEEDED",
  "error": "DeadlineExceeded",
  "message": "Operation budget exhausted",
  "retry_after_ms": null,
  "details": {
    "resource_scope": "time_budget"
  }
}
```

---

## Appendix D — Content Redaction Patterns (Normative)

* Replace user/tenant identifiers with irreversible hashes before logging.
* Replace prompts and graph queries with hashes; full content only in tightly controlled debug modes.
* For vectors and embeddings:

  * Log only dimensions and aggregate stats (e.g., norms).
  * NEVER log raw vectors or source texts.
* Telemetry exporters MUST implement configurable redaction rules.

---

## Appendix E — Implementation Status (Non-Normative)

* Reference adapters for each protocol family are RECOMMENDED.
* An interop test suite SHOULD validate:

  * Error normalization
  * Capability discovery
  * Streaming semantics
  * Partial-failure handling.

---

## Appendix F — Change Log / Revision History (Non-Normative)

This appendix is **append-only**. Each entry is immutable once published to avoid ambiguity about the meaning of a given version.

* **v1.0.0-rc1 — 2025-01-10**

  * Introduced canonical wire-first envelopes and streaming frame model (§4.1).
  * Added normative transport bindings for streaming (§4.1.4).
  * Defined common error taxonomy and initial mappings (§6.3, §12.4).
  * Established baseline security, privacy, and observability requirements.

* **v1.0.0-rc2 — 2025-01-24**

  * Clarified deadline semantics and conditional retry rules (§12.1).
  * Strengthened streaming finalization guarantees across Graph and LLM (§4.1.3, §7.3.2, §8.3).
  * Added cache coordination hints to `OperationContext` and `capabilities.cache` (§6.1, §6.2).
  * Introduced explicit partial-failure patterns for batch operations (§10.2, §12.5).

* **v1.0.0 — 2025-02-07 (Initial Public Stable Release)**

  * Locked the operation registry mapping protocol sections to `op` strings (§4.1.6).
  * Confirmed advisory nature of `cache_scope` / `cache_tags` while preserving tenant isolation (§6.1, §11.6).
  * Finalized Embedding Protocol with `EmbeddingFailure` and `EmbeddingResult.failures` as normative (§10.2).
  * Documented best-effort `TransactionCoordinator` pattern for cross-protocol workflows (§11.7).
  * Declared V1.0 wire contracts stable; subsequent 1.0.x releases are limited to non-breaking clarifications.

---

**End of Document**
