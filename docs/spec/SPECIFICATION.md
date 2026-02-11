# CORPUS SPECIFICATION

**specification_version:** `1.0.0`   

## Abstract

This specification defines the Corpus Protocol Suite: a vendor-neutral set of production-grade interfaces for **graph databases**, **large language models**, **vector databases**, and **text embeddings**. The suite standardizes contracts for heterogeneous AI infrastructure with built-in observability, error handling, and operational rigor. Protocols are minimal yet expressive, async-first, and extensible via negotiated capabilities. This document includes normative contracts, wire-compatible data shapes, an error taxonomy and mapping, resilience semantics, privacy and security guidance, and compatibility/versioning rules for enterprise-scale deployments.

> **Keywords:** Graph Database, Large Language Model, Vector Search, **Embeddings**, Observability, Multi-Tenancy, Capability Discovery, Semantic Versioning, SIEM-Safe Telemetry, BCP 14

## Status of This Memo

This document is not an Internet Standards Track specification; it is published for informational/standards-style guidance for the Corpus Protocol Suite. Distribution of this memo is unlimited.

## Copyright Notice

Copyright © 2026 Interoperable Intelligence Inc.
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

  * [4.1. Relationship to SCHEMA.md and PROTOCOLS.md (Normative)](#41-relationship-to-schemamd-and-protocolsmd-normative)
  * [4.2. Wire-First Canonical Form (Normative)](#42-wire-first-canonical-form-normative)

    * [4.2.1. Envelopes and Content Types (MUST)](#421-envelopes-and-content-types-must)
    * [4.2.2. Version Identification (MUST)](#422-version-identification-must)
    * [4.2.3. Streaming Frames (MUST where applicable)](#423-streaming-frames-must-where-applicable)
    * [4.2.4. Transport Bindings for Streaming (Normative)](#424-transport-bindings-for-streaming-normative)
    * [4.2.5. Compatibility and Unknown Fields (MUST)](#425-compatibility-and-unknown-fields-must)
    * [4.2.6. Operation Registry (Normative)](#426-operation-registry-normative)
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

    * [7.3.1. Node/Edge CRUD](#731-nodeedge-crud)
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
* [21. Author's Address](#21-authors-address)
* [Appendix A — End-to-End Example (Normative)](#appendix-a--end-to-end-example-normative)
* [Appendix B — Capability Shapes (Illustrative)](#appendix-b--capability-shapes-illustrative)
* [Appendix C — Wire-Level Envelopes](#appendix-c--wire-level-envelopes)
* [Appendix D — Content Redaction Patterns (Normative)](#appendix-d--content-redaction-patterns-normative)
* [Appendix E — Implementation Status (Non-Normative)](#appendix-e--implementation-status-non-normative)
* [Appendix F — Change Log / Revision History (Non-Normative)](#appendix-f--change-log--revision-history-non-normative)
* [Appendix G — Migration from Existing APIs (Informative)](#appendix-g--migration-from-existing-apis-informative)

---

## 1. Introduction

### 1.1. Motivation

The AI infrastructure landscape is fragmenting in two directions simultaneously. At the framework layer, LangChain, LlamaIndex, AutoGen, CrewAI, and Semantic Kernel each define their own abstractions for orchestrating AI operations. At the provider layer, OpenAI, Anthropic, Cohere, Pinecone, Weaviate, Neo4j, and countless others expose incompatible APIs. This creates an N×M integration problem: every framework must maintain adapters for every provider, and every new provider means updates across all frameworks. The result is duplicated effort, inconsistent error handling, incompatible observability, and operational blindness across the stack.

Corpus solves this by defining vendor-neutral protocol standards that sit beneath frameworks and above providers—the TCP/IP for AI infrastructure. Frameworks adopt common protocols once and get compatibility with all conformant providers. Providers implement protocols once and gain compatibility with all conformant frameworks. Enterprises get unified observability, consistent error semantics, and genuine portability across their entire AI stack—without abandoning existing tooling.​​​​​​​​​​​​​​​​

### 1.2. Scope

This specification defines four complementary protocols:

* **Graph Protocol V1.0** — Node/edge CRUD, traversal, and multi-dialect query execution.
* **LLM Protocol V1.0** — Chat-style completion, streaming tokens, usage accounting.
* **Vector Protocol V1.0** — Vector upsert/delete, similarity search, and namespace management.
* **Embedding Protocol V1.0** — Text embedding generation (single/batch), token counting, capability discovery, and health reporting.

All protocols share a **Common Foundation** for context propagation, capability discovery, error taxonomy, observability, and resilience.

### 1.3. Design Philosophy

* **Minimal Surface Area (MUST).** Only essential operations are standardized. Vendor extensions appear via capabilities, not new methods.
* **Async-First (MUST).** All operations are non-blocking and concurrency-safe.
* **Production-Hardened (MUST).** Observability, error taxonomy, and resilience are first-class.
* **Extensible (SHOULD).** Capability negotiation enables optional features without breaking compatibility.
* **Type-Safe (SHOULD).** Strong typing and runtime validation minimize undefined behavior.
* **Privacy by Design (MUST).** SIEM-safe telemetry, data minimization, and redaction are defaults.

---

## 2. Requirements Language

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in BCP 14 [RFC2119] [RFC8174] when, and only when, they appear in all capitals.

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

* JSON keys are **case-sensitive**.
* Unknown keys in **permissive objects** (e.g., `ctx`, capability objects, health results, and arguments whose schemas allow `additionalProperties: true`) **MUST** be ignored by clients and servers.
* For **strict objects** (e.g., core response envelopes and types whose schemas use `additionalProperties: false`), behavior is governed by SCHEMA.md; unknown keys MAY cause validation to fail.
* Durations are expressed in **milliseconds** unless otherwise specified.
* Examples are **non-normative** unless explicitly marked **(Normative)**.
* Field names use **lower_snake_case** unless specified.
* Error `code` values use `UPPER_SNAKE_CASE`.
* `tenant_hash` denotes a deterministic, irreversible hash of the tenant identifier (recommended: SHA-256 with per-deployment salt).
* Unless otherwise stated, scores are **higher is better** when represented as scores.
* Numeric fields MUST respect type/range constraints defined in relevant sections.

### 4.1. Numeric Types (Normative)

Unless otherwise specified:

* Integer fields are signed 64-bit.
* Float fields are IEEE-754 double precision.
* All durations are non-negative integers in milliseconds.
* Implementations MUST reject NaN, +Inf, −Inf, and out-of-range numeric values with `BadRequest`.

**Note:** SCHEMA.md uses JSON Schema `integer` and `number` types without explicit bit-width; these specifications provide recommended implementations.

### 4.1. Relationship to SCHEMA.md and PROTOCOLS.md (Normative)

This specification describes the architecture, semantics, and illustrative shapes of the Corpus Protocol Suite.

- **SCHEMA.md** is the authoritative source of truth for JSON wire-format shapes and validation (including required fields and `additionalProperties` behavior).
- **PROTOCOLS.md** describes operation-level semantics, streaming behavior, and error-handling rules consistent with SCHEMA.md.
- When SCHEMA.md and this specification disagree on a JSON field name or type, **SCHEMA.md is authoritative** and this document MUST be updated to match.
- When PROTOCOLS.md and this specification disagree on operation semantics, **PROTOCOLS.md is authoritative** for protocol behavior.

### 4.2. Wire-First Canonical Form (Normative)

The canonical interface is defined at the wire level using JSON documents and streaming frames.

#### 4.2.1. Envelopes and Content Types (MUST)

Non-streaming operations:

* `Content-Type: application/json`
* UTF-8 encoded JSON.

**Request envelope:**

```json
{
  "op": "<component>.<operation>",
  "ctx": {},
  "args": {}
}
```

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

* `code` MUST be a normalized code.
* `error` MUST be a normalized error class name when `ok=false`.

**Closed Response Envelopes (Normative)**

Core response envelopes are **closed objects** at the wire level:

- Unary success envelopes MUST NOT contain any top-level keys other than `{ "ok", "code", "ms", "result" }`.
- Error envelopes MUST NOT contain any top-level keys other than `{ "ok", "code", "error", "message", "retry_after_ms", "details", "ms" }`.
- Streaming success frames use the `{ "ok", "code", "ms", "chunk" }` envelope defined in PROTOCOLS.md and MUST NOT contain any top-level keys beyond those fields.

These constraints are enforced in SCHEMA.md via `additionalProperties: false` on the corresponding envelope schemas.

#### 4.2.2. Version Identification (MUST)

* Each protocol instance declares `"{component}/v1.0"`. Example: `"protocol": "vector/v1.0"`.
* Clients MAY send `X-Adapter-Protocol: {component}/v1.0`.
* All `v1.x` revisions are wire-compatible with `{component}/v1.0`.
* Breaking changes require `"{component}/v2.0"`.

## 4.2.3. Streaming Frames (MUST where applicable)

The canonical streaming interface is defined at the wire level using **JSON envelopes** validated by **SCHEMA.md**. For streaming operations (e.g., `llm.stream`, `graph.stream_query`, `embedding.stream_embed`), adapters MUST emit a sequence of **streaming success frames** and MAY terminate with an **error envelope**.

### 4.2.3.1 Canonical Streaming Success Frame (MUST)

Each streaming success frame MUST be a JSON object with the following top-level shape:

```json
{ "ok": true, "code": "STREAMING", "ms": 45.2, "chunk": { ... } }
```

Rules:

* `ok` MUST be `true`.
* `code` MUST be `"STREAMING"` for all streaming success frames.
* `ms` MUST be a non-negative number representing time elapsed in milliseconds (as defined in SCHEMA.md).
* `chunk` MUST be an operation-specific payload validated by the operation's streaming chunk schema (e.g., `llm.types.chunk`, `graph.types.chunk`, `embedding.types.chunk`).
* Streaming success frames are **closed objects** at the top level: adapters MUST NOT include top-level keys beyond `{ "ok", "code", "ms", "chunk" }` (enforced by SCHEMA.md).

### 4.2.3.2 Terminal Conditions (MUST)

Exactly one terminal condition MUST occur per stream:

**A) Terminal Success (MUST)**
A stream completes successfully when a streaming success frame is emitted whose `chunk.is_final` is `true`.

* A well-formed success stream MUST include exactly one frame with `chunk.is_final: true`.
* No streaming frames (success or error) may follow the terminal success frame.

**B) Terminal Error (MUST)**
A stream terminates with failure if an **error envelope** is emitted:

```json
{
  "ok": false,
  "code": "UNAVAILABLE",
  "error": "Unavailable",
  "message": "...",
  "retry_after_ms": null,
  "details": null,
  "ms": 123.4
}
```

Rules:

* Error frames MUST use the canonical error envelope shape (validated by SCHEMA.md).
* Error envelopes are **closed objects** at the top level: adapters MUST NOT include top-level keys beyond those permitted by the error envelope schema.
* No frames may follow a terminal error envelope.

### 4.2.3.3 Ordering and Integrity (MUST)

* Streams MUST preserve semantic ordering: chunks MUST be delivered in the order defined by the operation (e.g., token order for LLM, record order for graph).
* Chunks MUST be self-contained JSON objects valid under the appropriate chunk schema.
* Clients MUST NOT rely on JSON field ordering within frames.

### 4.2.3.4 Frame Size and Keepalive (MUST/SHOULD)

* Each serialized frame MUST be ≤ 1 MiB.
* For streams idle for more than 15 seconds, adapters SHOULD emit a keepalive **as a standard streaming success frame** whose chunk is a no-op per the relevant chunk schema and/or operation guidance.

> Note: Keepalive messages MUST still conform to the schema-defined streaming success envelope and MUST NOT violate terminal conditions.

### 4.2.3.5 Optional Gateway Event Overlay (Informative)

Gateways or routers MAY present an alternate client-facing event model (e.g., SSE `event:` + `data:` wrappers) for convenience. Such overlays are **not** part of the base adapter wire protocol and MUST NOT replace the canonical envelope+chunk frames emitted by adapters.

If a gateway chooses to expose an event overlay, it SHOULD preserve the semantics and terminal conditions described above.

## 4.2.4. Transport Bindings for Streaming (Normative)

This section defines how the canonical streaming frame model (§4.2.3) is carried over common transports. The base unit across all transports is a **single JSON frame** that is either:

* a streaming success envelope: `{ "ok": true, "code": "STREAMING", "ms": ..., "chunk": ... }`, or
* an error envelope: `{ "ok": false, ... }`.

Adapters MAY support one or more transports and SHOULD advertise supported transports via capabilities.

### 4.2.4.1 HTTP/1.1 + NDJSON (MUST if supported)

* `Content-Type: application/x-ndjson`
* `Transfer-Encoding: chunked`
* Each line MUST be one complete JSON frame (streaming success envelope or error envelope).
* The final line MUST contain the terminal condition: either a frame with `chunk.is_final: true`, or an error envelope.

### 4.2.4.2 Server-Sent Events (SSE) (MUST if supported)

* `Content-Type: text/event-stream`
* Each SSE `data:` payload MUST be a complete JSON frame (streaming success envelope or error envelope).
* Gateways MAY set SSE `event:` fields for client convenience, but clients MUST be able to recover the canonical JSON frame from `data:`.

### 4.2.4.3 WebSocket (MUST if supported)

* Each WebSocket message MUST contain exactly one complete JSON frame (streaming success envelope or error envelope).
* Terminal conditions follow §4.2.3.2.

### 4.2.4.4 gRPC / HTTP/2 Streaming (MUST if supported)

* Each streamed message MUST correspond to exactly one logical JSON frame (streaming success envelope or error envelope), encoded in a transport-appropriate way.
* Terminal conditions follow §4.2.3.2.

### 4.2.4.5 Advertising Transport Support (SHOULD)

Adapters SHOULD advertise supported streaming transports via capabilities:

```json
"extensions": {
  "streaming_transports": ["ndjson", "sse", "websocket"]
}
```

Unknown keys in `extensions` MUST be ignored by clients.

#### 4.2.5. Compatibility and Unknown Fields (MUST)

* Unknown keys in **permissive objects** (e.g., `ctx`, capability objects, health results, and arguments whose schemas allow `additionalProperties: true`) MUST be ignored by clients and servers.
* Core response envelopes (success, error, streaming) and other **strict objects** follow SCHEMA.md: their schemas may reject unknown keys via `additionalProperties: false`.
* Clients MUST NOT rely on JSON field ordering.
* Numeric fields MUST follow §4.1 numeric rules.

#### 4.2.6. Operation Registry (Normative)

The following values of `op` are reserved for V1.0:

| Operation Name | op String | Protocol | Description |
|---|---|---|---|
| capabilities | graph.capabilities | Graph | Discover supported graph features and limits |
| upsert_nodes | graph.upsert_nodes | Graph | Create or update multiple nodes |
| upsert_edges | graph.upsert_edges | Graph | Create or update multiple edges |
| delete_nodes | graph.delete_nodes | Graph | Remove nodes by ID |
| delete_edges | graph.delete_edges | Graph | Remove edges by ID |
| query | graph.query | Graph | Execute a graph query and return results |
| stream_query | graph.stream_query | Graph | Execute a graph query with streaming results |
| bulk_vertices | graph.bulk_vertices | Graph | Bulk operations on vertices (import/export) |
| batch | graph.batch | Graph | Execute multiple operations in batch |
| get_schema | graph.get_schema | Graph | Retrieve graph schema information |
| health | graph.health | Graph | Check adapter and provider health status |
| **transaction** | **graph.transaction** | **Graph** | **Execute operations in an atomic transaction** |
| **traversal** | **graph.traversal** | **Graph** | **Traverse graph relationships from starting nodes** |
| capabilities | llm.capabilities | LLM | Discover supported LLM features and models |
| complete | llm.complete | LLM | Generate LLM completion for given messages |
| stream | llm.stream | LLM | Stream LLM completion incrementally |
| count_tokens | llm.count_tokens | LLM | Count tokens in text for a specific model |
| health | llm.health | LLM | Check LLM provider health and model availability |
| capabilities | vector.capabilities | Vector | Discover supported vector features and limits |
| query | vector.query | Vector | Find similar vectors using approximate nearest neighbor search |
| **batch_query** | **vector.batch_query** | **Vector** | **Execute multiple vector queries in batch** |
| upsert | vector.upsert | Vector | Insert or update vectors in a namespace |
| delete | vector.delete | Vector | Remove vectors by ID |
| create_namespace | vector.create_namespace | Vector | Create a new vector namespace/collection |
| delete_namespace | vector.delete_namespace | Vector | Remove a vector namespace and all its vectors |
| health | vector.health | Vector | Check vector store health and namespace status |
| capabilities | embedding.capabilities | Embedding | Discover supported embedding features and models |
| embed | embedding.embed | Embedding | Generate embedding vector for a single text |
| embed_batch | embedding.embed_batch | Embedding | Generate embeddings for multiple texts in batch |
| **stream_embed** | **embedding.stream_embed** | **Embedding** | **Stream embeddings incrementally for a single text** |
| count_tokens | embedding.count_tokens | Embedding | Count tokens in text for embedding model |
| **get_stats** | **embedding.get_stats** | **Embedding** | **Retrieve embedding service statistics and usage metrics** |
| health | embedding.health | Embedding | Check embedding provider health and model status |

**Note:** Filter-based deletion operations (e.g., delete by metadata filter) MAY be supported via extensions (e.g., `graph.delete_nodes_by_filter`, `vector.delete_by_filter`).

Adapters MAY expose additional `op` values via namespaced extensions (e.g. `vendorX.llm.complete_raw`) but MUST NOT alter semantics of reserved `op` strings.

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

Profiles define behavior but MUST NOT change wire contracts.

* **Thin (default):** No-op caches, limiters, breakers; propagate deadlines downstream.
* **Standalone:** Local deadlines, circuit breaker, token-bucket limiter, short-TTL in-memory caches for Embedding + LLM `complete`.

---

## 6. Common Foundation

### 6.1. Operation Context

The following Python dataclass is an SDK-level representation of the operation context. The **canonical wire shape** of `ctx` is defined by SCHEMA.md; this dataclass is illustrative and may include convenience fields or defaults not present on the wire.

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

    # SDK-level cache hints (MAY be passed in attrs at wire level)
    cache_scope: Optional[str] = None        # "tenant" | "global" | "session"; default "tenant"
    cache_tags: Optional[List[str]] = None   # advisory tags for cache keying/invalidation
```

**Note:** `cache_scope` and `cache_tags` are advisory SDK-level fields. At the wire level, they MAY be passed in `ctx.attrs` per PROTOCOLS.md §6.1.

**Normative behavior:**

* `request_id` SHOULD uniquely identify the operation end-to-end.
* `idempotency_key` MAY be supplied for mutating operations.

**Idempotency Key Scope and Lifetime**

* Scope is `(tenant, op, args-hash)`.
* Servers MUST deduplicate replays with the same scoped key for at least 24 hours (configurable).
* Servers MUST NOT apply side-effects twice for the same scoped key.
* On replay, servers MUST respond with `code: "OK"` and a result semantically identical to the originally committed response.

**Deadline Semantics**

* On receipt, servers MUST compute:
  `remaining_ms = deadline_ms - now()`.
* If `deadline_ms` is present and `remaining_ms <= 0`, servers MUST fail fast with `DeadlineExceeded` without invoking backends.
* For multi-hop calls, adapters MUST propagate the absolute `deadline_ms` and each hop MUST recompute remaining time using the same rule.

**Trace and Tenant**

* `traceparent` MUST be forwarded unchanged.
* `tenant` MUST be used for isolation. Raw tenant identifiers MUST NOT be logged.

**Cache Coordination (Advisory)**

* If `cache_scope` is unset, treat as `"tenant"`.
* `cache_scope` MUST NOT weaken tenant isolation; cross-tenant sharing MUST NOT occur unless explicitly configured outside this spec.
* `cache_tags` MUST NOT contain secrets or raw tenant IDs.
* `cache_scope` and `cache_tags` are advisory: implementations MAY ignore them without violating this spec.

**Design Note (Cache Fields):**

Cache coordination fields are advisory to enable:

1. Future cache invalidation patterns.
2. Router-level cache optimization where supported.
3. Tenant isolation hints without mandating specific cache implementations.

### 6.2. Capability Discovery

`capabilities()` MUST return a JSON object similar to:

```json
{
  "server": "example-backend",
  "version": "5.17.1",
  "protocol": "graph/v1.0",  // REQUIRED field
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

**Note:** The `protocol` field is REQUIRED in all `*Capabilities` results per PROTOCOLS.md §2.12 and SCHEMA.md capability schemas.

Rules:

* Unknown `features`, `limits`, `cache`, `extensions` keys MUST be ignored.
* `extensions` keys MUST be namespaced (e.g. `vendor:foo.bar`).

**Frozen Keys (v1.x)**

* Keys `server`, `version`, `protocol`, and all capability keys named in this document are frozen for v1.x.
* Adapters MUST NOT repurpose or change semantics of these keys in v1.x.
* New capability keys MUST be added under `extensions`.

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

Errors MUST use normalized `code` values and SHOULD include hints, e.g.:

```json
{
  "ok": false,
  "code": "RESOURCE_EXHAUSTED",
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

  * No raw tenant IDs, prompts, vectors, or source texts.
* `extra` SHOULD use low-cardinality keys such as `tenant_hash`, `deadline_bucket`, `cache_hit`, `rows`, `matches_returned`, `batch_size`, and `model` where allowed.

---

## 7. Graph Protocol V1.0 Specification

### 7.1. Overview

Vendor-neutral interface for graph databases (Cypher, OpenCypher, Gremlin, GQL), covering CRUD, queries (sync/streaming), batch operations, and optional schema management.

### 7.2. Data Types

The Python dataclasses in this section describe typical SDK-level representations. The **canonical JSON wire shapes** are defined in SCHEMA.md and reflected in the JSON examples in this specification. Where there is any discrepancy, SCHEMA.md is authoritative.

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
    rate_limit_unit: str = "requests_per_second"
    max_qps: Optional[int] = None
    idempotent_writes: bool = False
    supports_multi_tenant: bool = False
    supports_streaming: bool = False
    supports_bulk_ops: bool = False
    max_query_length: Optional[int] = None
```

### 7.3. Operations

#### 7.3.1. Node/Edge CRUD

> **Note:** The Python signatures shown here are illustrative SDK-level representations. The **canonical wire format** is defined in PROTOCOLS.md §7 and SCHEMA.md §4.4, which includes additional fields such as `labels`, `namespace`, `created_at`, and `updated_at`.

```python
async def upsert_nodes(
    nodes: Iterable[Tuple[str, Mapping[str, Any]]],
    *,
    ctx: Optional[OperationContext] = None
) -> List[GraphID]

async def delete_nodes(
    node_ids: List[GraphID],
    *,
    ctx: Optional[OperationContext] = None
) -> None

async def upsert_edges(
    edges: Iterable[Tuple[str, GraphID, GraphID, Mapping[str, Any]]],
    *,
    ctx: Optional[OperationContext] = None
) -> List[GraphID]

async def delete_edges(
    edge_ids: List[GraphID],
    *,
    ctx: Optional[OperationContext] = None
) -> None
```

Semantics:

* `props` keys MUST be strings; values MUST be JSON-serializable.
* Create operations SHOULD accept `idempotency_key` and MUST be idempotent when provided.
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

Rules:

* `dialect` MUST be one of the advertised dialects.
* `params` MUST be safely bound; no string interpolation.
* Deadline behavior follows §6.1.

##### Streaming Finalization (Normative)

When exposed over the wire:

1. Emit zero or more `{"ok": true, "code": "STREAMING", "ms": ..., "chunk": {...}}` frames.
2. Emit exactly one terminal frame:

   * A frame with `chunk.is_final: true` on success, or
   * An error envelope on failure.

As async iterator:

* `StopAsyncIteration` MUST release resources.
* Implementation MUST emit one final `observe` reflecting success/failure.

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

* Enforce `max_batch_ops` if provided.
* Partial failures MUST follow §12.5 semantics.
* Supported batch op codes MUST be advertised via `capabilities.extensions`.

### 7.4. Dialects

Supported dialects (e.g., `cypher`, `opencypher`, `gremlin`, `gql`) MUST be listed in capabilities. Unknown dialect → `NotSupported`.

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

Sample values MUST NOT be logged.

### 7.6. Health

`graph.health` SHOULD:

* Return:

  ```json
  {
    "ok": true,
    "status": "ok",
    "server": "...",
    "version": "..."
  }
  ```

  when healthy.
* On degraded service:

  ```json
  {
    "ok": true,
    "status": "degraded",
    "reason": "low-capacity"
  }
  ```
* On down:

  ```json
  {
    "ok": false,
    "status": "down",
    "reason": "unreachable"
  }
  ```

Gateways:

* `status:"ok"` and `status:"degraded"` → HTTP 200.
* `status:"down"` → HTTP 503.

---

## 8. LLM Protocol V1.0 Specification

### 8.1. Overview

Standardized interface for chat-style completions, streaming, token accounting, and model discovery, with unified telemetry and errors.

### 8.2. Data Types

The Python dataclasses in this section describe SDK-level representations of LLM types. The **canonical JSON wire shapes** are defined in SCHEMA.md and reflected in the JSON examples in this specification. Where there is any discrepancy, SCHEMA.md is authoritative.

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
    model_family: str
    usage: TokenUsage
    finish_reason: str  # "stop" | "length" | "tool_call" | "content_filter"

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

* `role` MUST be one of `system`, `user`, `assistant`, `tool`.
* `temperature` in `[0,2]`, `top_p` in `(0,1]`.
* `frequency_penalty`, `presence_penalty` in `[-2,2]`.

Streaming:

* Uses §4.2.3 frame semantics.
* Last chunk MUST have `is_final=true`.
* One final `observe` metric per stream.

### 8.4. Model Discovery

`llm.capabilities` MUST include:

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
    "supports_system_message": true,
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

### 8.5. LLM-Specific Errors

* `ModelOverloaded` → `Unavailable` (retryable).
* `ContentFiltered` → `BadRequest` (non-retryable without input change).
* `DeadlineExceeded` as per §12.1.

---

## 9. Vector Protocol V1.0 Specification

### 9.1. Overview

Standardized vector storage and similarity search with namespaces, filters, and consistent metrics.

### 9.2. Data Types

The Python dataclasses in this section are SDK-level representations of vector types. The **canonical JSON wire shapes** are defined in SCHEMA.md and reflected in the JSON examples. Where there is any discrepancy, SCHEMA.md is authoritative.

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

Helper specs:

```python
@dataclass(frozen=True)
class QuerySpec:
    vector: List[float]
    top_k: int
    namespace: str = "default"
    filter: Optional[Dict[str, Any]] = None
    include_metadata: bool = True
    include_vectors: bool = False

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
* `top_k` MUST be > 0 and ≤ `limits.max_top_k` when published.
* Filters applied pre-search when supported; post-filtering MUST be documented.
* If `include_vectors=false`, raw vectors MAY be omitted.

### 9.4. Distance Metrics

Adapters MUST advertise supported metrics (`cosine`, `euclidean`, `dot`) and behavior.

Requirements:

* For `cosine` and `dot`, adapters MUST return scores where higher is better.
* For `euclidean`, adapters MUST return distances where lower is better.
* If an adapter cannot natively comply, it MUST:

  * Return a normalized score in `[0,1]` where higher is better.
  * Include the raw distance as an additional, clearly-labeled field.

### 9.5. Vector-Specific Errors

* `DimensionMismatch` → HTTP 400, non-retryable.
* `IndexNotReady` → HTTP 503, retryable; SHOULD include `retry_after_ms`.

---

## 10. Embedding Protocol V1.0 Specification

### 10.1. Overview

Vendor-neutral interface for generating embeddings (single/batch), counting tokens, and health checking, including partial-failure reporting.

### 10.2. Data Types (Formal)

The Python dataclasses in this section describe SDK-level representations of embedding types. The **canonical JSON wire shapes** are defined in SCHEMA.md and reflected in the JSON examples. Where there is any discrepancy, SCHEMA.md is authoritative.

> **Note:** The types shown here are SDK-level representations. PROTOCOLS.md uses distinct types for single vs batch operations:
> - `EmbedResult` (singular `embedding`) for `embedding.embed`
> - `EmbedBatchResult` (plural `embeddings`) for `embedding.embed_batch`
> 
> See PROTOCOLS.md §18-§20 for canonical wire format definitions.

```python
from dataclasses import dataclass
from typing import List, Optional, Mapping, Any

@dataclass(frozen=True)
class EmbeddingVector:
    vector: List[float]
    text: Optional[str]
    model: str
    dimensions: int

@dataclass(frozen=True)
class EmbeddingFailure:
    index: int
    error: str
    message: Optional[str] = None
    details: Optional[Mapping[str, Any]] = None

@dataclass(frozen=True)
class EmbeddingResult:
    embeddings: List[EmbeddingVector]
    model: str
    total_tokens: Optional[int] = None
    processing_time_ms: Optional[float] = None
    failures: Optional[List[EmbeddingFailure]] = None
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
    idempotent_writes: bool = True  # Wire key: idempotent_writes
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
) -> Mapping[str, Any]
```

`embedding.health` SHOULD mirror `graph.health` graded states:

* `"status": "ok" | "degraded" | "down"`, with `"reason"` for degraded/down.

### 10.4. Errors (Embedding-Specific)

* `TextTooLong` (`BadRequest`) when `truncate=false` and length exceeds `max_text_length`.
* `ModelNotAvailable` → `NotSupported` or `Unavailable`.
* `DeadlineExceeded` follows §12.1.

### 10.5. Capabilities

Implementations MUST declare:

* `server`, `version`, `supported_models`.
* Optional: `max_batch_size`, `max_text_length`, `max_dimensions`.
* Boolean flags as defined in `EmbeddingCapabilities`.

**Note:** Both the wire-level JSON key and the SDK field share the same name: `idempotent_writes`. This field and its semantics are defined in PROTOCOLS.md and SCHEMA.md.

### 10.6. Semantics

* `model` MUST be in `supported_models`.
* `normalize=true` MUST fail with `NotSupported` if unsupported.
* `embed_batch` MUST:

  * Enforce `max_batch_size` if present.
  * Validate each text.
  * Use partial-failure reporting per §12.5.
* If `truncate=false` and too long → `TextTooLong`.
* Deadline semantics as in §6.1.

---

## 11. Cross-Protocol Patterns

### 11.1. Unified Error Handling

* All adapters MUST map provider errors into §6.3 taxonomy.
* Error envelopes MUST include `code` and `message`.
* SHOULD include `retry_after_ms`, `throttle_scope`, and `details.provider_error_id` when applicable.

### 11.2. Consistent Observability

* Use `MetricsSink` for all components.
* Exactly one final `observe` per operation; streams included.
* MUST NOT log raw content or tenant identifiers.

### 11.3. Context Propagation

* A single `OperationContext` flows across components.
* `traceparent` forwarded unchanged.
* `deadline_ms` propagated as absolute; each hop recomputes remaining time.

### 11.4. Idempotency and Exactly-Once

* Operations that accept `idempotency_key` MUST treat duplicate keys as safe replays within the defined scope.
* MUST ensure no duplicate side-effects.
* SHOULD audit idempotency behavior in SIEM-safe form.

### 11.5. Pagination and Streaming

* For streaming, follow §4.2.3.
* When pagination is supported (Graph or Vector or others):

  * Responses MUST include an opaque `next_page_token` when more results exist.
  * Missing or empty `next_page_token` means no further pages.
  * Clients MUST NOT infer semantics from token contents.

### 11.6. Caching (Implementation Guidance)

* Embedding and deterministic LLM calls are strong candidates for caching.
* Caches MUST:

  * Use content hashes (e.g., `sha256`) instead of raw texts.
  * Include `model`, parameters, and `tenant_hash` when `cache_scope="tenant"`.
* If `cache.supports_tags=true`, `cache_tags` SHOULD be used for group invalidation.
* Tenant isolation MUST always be preserved.

### 11.7. Best-Effort Distributed Transactions

A `TransactionCoordinator` MAY provide best-effort multi-component workflows without assuming global ACID. Failures MUST respect tenant isolation and surface composite status.

---

## 12. Error Handling and Resilience

### 12.1. Retry Semantics

**Retryable:**

* `ResourceExhausted`
* `TransientNetwork`
* `Unavailable`
* `IndexNotReady`

**Conditionally Retryable:**

* `DeadlineExceeded` — retry only if:

  * Extending `deadline_ms`, and/or
  * Reducing requested work (e.g., fewer tokens, fewer items).

**Non-Retryable:**

* `BadRequest`, `AuthError`, `NotSupported`
* `DimensionMismatch`, `ContentFiltered`, `TextTooLong` (unless request is changed accordingly)
* Other validation errors.

**Machine-Actionable Hints**

* For retryable errors:

  * If `retry_after_ms` is absent, servers SHOULD include `details.suggested_backoff_ms` (integer).
* For `ResourceExhausted`:

  * Servers SHOULD include `throttle_scope`.
  * MAY include `details.suggested_concurrency`.

### 12.2. Backoff and Jitter (RECOMMENDED)

* Exponential backoff base 100–500 ms, factor ×2, cap 10–30 s.
* Use full jitter.
* Honor `retry_after_ms` or `suggested_backoff_ms` when present.

### 12.3. Circuit Breaking

* On repeated retryable failures, implementations MAY open a circuit.
* While open, fail fast with `Unavailable` (e.g. `"circuit_open"`).
* Include `retry_after_ms` guidance when possible.
* Prefer per-tenant, per-operation circuits.

### 12.4. Error Mapping Table (Normative)

**Canonical Success Codes:**
- `OK` - Operation completed successfully

**Canonical Error Codes (Illustrative, Not Exhaustive):**
- `BAD_REQUEST` - Invalid parameters, malformed requests
- `AUTH_ERROR` - Invalid credentials, permissions
- `RESOURCE_EXHAUSTED` - Rate limits, quotas exceeded
- `TRANSIENT_NETWORK` - Network timeouts, connection issues
- `UNAVAILABLE` - Service temporarily down
- `NOT_SUPPORTED` - Unsupported operation or parameter
- `DEADLINE_EXCEEDED` - Operation timeout
- `MODEL_OVERLOADED` - Model capacity exceeded
- `CONTENT_FILTERED` - Input violates content policy
- `TEXT_TOO_LONG` - Input exceeds context window
- `DIMENSION_MISMATCH` - Vector dimensions don't match
- `INDEX_NOT_READY` - Vector index not ready for queries
- `MODEL_NOT_AVAILABLE` - Requested model not available
- `QUERY_PARSE_ERROR` - Invalid query syntax
- `VERTEX_NOT_FOUND` - Graph node not found
- `EDGE_NOT_FOUND` - Graph edge not found
- `SCHEMA_VALIDATION_ERROR` - Schema constraint violation

| Error Class        | HTTP    | Retryable   | Client Guidance                                       |
| ------------------ | ------- | ----------- | ----------------------------------------------------- |
| BadRequest         | 400     | No          | Fix request; do not retry                             |
| AuthError          | 401/403 | No          | Refresh credentials; fix scopes                       |
| ResourceExhausted  | 429     | Yes         | Back off; honor hints; reduce concurrency/batch       |
| TransientNetwork   | 502/504 | Yes         | Retry with backoff + jitter; consider failover        |
| Unavailable        | 503     | Yes         | Retry; use circuit breakers/failover                  |
| NotSupported       | 400/501 | No          | Use `capabilities()`; change feature/model            |
| DimensionMismatch* | 400     | No          | Fix vector dimension                                  |
| IndexNotReady*     | 503     | Yes         | Retry after `retry_after_ms` or suggested backoff     |
| ModelOverloaded**  | 503     | Yes         | Retry; backoff; consider alternate model              |
| ContentFiltered**  | 400     | No          | Sanitize or change input                              |
| TextTooLong***     | 400     | No          | Truncate or chunk, or enable truncation               |
| DeadlineExceeded   | 504     | Conditional | Retry only with extended deadline or reduced workload |

* Vector-specific
** LLM-specific
*** Embedding-specific

### 12.5. Partial Failure Contracts

For non-atomic batch operations (e.g. `embed_batch`, vector `upsert`, graph `batch`):

**Option B Selected:** Keep `code: "OK"` and encode partial vs full success inside result only (matching the existing BatchResult convention in PROTOCOLS.md).

> **Note:** Batch result field names vary by protocol (see PROTOCOLS.md §27.2):
> - Embedding: `embeddings`, `total_texts`, `failed_texts`
> - Vector: `upserted_count`/`deleted_count`, `failed_count`, `failures`
> - Graph: `GraphBatchResult` with `results[]`, `success`, `error?`
>
> The examples below use generic field names; refer to protocol-specific sections for exact shapes.

The transport envelope for batch operations MUST be:

```json
{
  "ok": true,
  "code": "OK",
  "ms": 38.4,
  "result": {
    "processed_count": 2,
    "failed_count": 1,
    "failures": [
      {
        "index": 2,
        "error": "TEXT_TOO_LONG",
        "detail": "Input exceeds max_text_length"
      }
    ],
    "results": [
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
    ]
  }
}
```

**Note:** Protocol-specific batch operations use protocol-specific result field names (e.g., `embeddings` for embedding operations, `vectors` for vector operations) within the generic `BatchResult` wrapper.

Requirements:

* Use `ok:true` and `code:"OK"` for all batch operations regardless of partial failures.
* Each failed item MUST include `index` and `error`. `detail` is OPTIONAL.
* Success items MUST preserve the input order across `results`.
* MUST NOT drop failed items silently.

### 12.6. Backpressure Integration

Implementations SHOULD integrate cooperative backpressure. On saturation, surface `ResourceExhausted` or `Unavailable` following this spec's hints.

---

## 13. Observability and Monitoring

### 13.1. Metrics Taxonomy (MUST)

Adapters MUST expose:

* `ops_total{component,op,code}`
* `latency_ms{component,op,code,quantile}`
* When applicable:

  * `tokens_total{component,model}`
  * `matches_returned_total{component,op}`

Constraints:

* No labels containing free-text prompts, queries, or raw tenant IDs.
* Cardinality MUST be controlled.

### 13.2. Structured Logging (MUST)

Logs SHOULD be structured JSON with low-cardinality fields. Examples:

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

### 13.3. Distributed Tracing (SHOULD)

* Propagate `traceparent`.
* Use spans with attributes for `component`, `op`, `tenant_hash`, `model`, counts.
* For streams, final span status MUST match terminal frame outcome.

---

## 14. Security Considerations

### 14.1. Tenant Isolation (MUST)

* Strong isolation for data and control planes.
* No cross-tenant data leakage in caches, logs, or training.

### 14.2. Authentication and Authorization (MUST)

* Credentials provisioned out-of-band.
* Credentials MUST NOT be logged.
* Enforce least privilege and tenant-aware RBAC.

### 14.3. Threat Model (SHOULD)

Implementations SHOULD address:

* Prompt/query injection.
* Vector/embedding poisoning.
* Idempotency-key spoofing.
* Resource exhaustion and DoS.
* Cross-tenant leakage.
* Tool/model misuse.

### 14.4. Mitigation Matrix (Normative)

| Threat Category                | Component         | MUST                                                   | SHOULD                                                  |
| ------------------------------ | ----------------- | ------------------------------------------------------ | ------------------------------------------------------- |
| Injection (prompt/query)       | LLM, Graph        | Parameterized bindings; strict role/dialect validation | Allowlist dialects/models; escape/templatize user input |
| Data exfiltration via logs     | All               | SIEM-safe logs; no prompts/vectors/raw tenants         | Field redaction; log retention ≤30 days                 |
| Idempotency-key spoofing       | All mutating      | Treat duplicates as replays; exactly-once semantics    | Use HMAC/nonce-scoped keys with TTL                     |
| Poisoning (vectors/embeddings) | Vector, Embedding | Validate dimensions; reject NaN/Inf; enforce ACLs      | Outlier detection; quarantine suspicious batches        |
| Resource exhaustion / DoS      | All               | Enforce deadlines; per-tenant rate limits              | Adaptive backoff; tuned circuit breakers                |
| Cross-tenant leakage           | All               | Strict tenant isolation; tenant in cache keys          | Isolation tests; policy-as-code in CI/CD                |
| Tool/model misuse              | LLM               | Enforce param ranges; map filters to `ContentFiltered` | Guardrails; controlled tool/model registries            |
| Unbounded traversals           | Graph             | Depth/row limits; timeouts; schema/RBAC                | Configurable caps; anomaly alerts                       |

---

## 15. Privacy Considerations

* MUST NOT log raw prompts, source texts, vectors, or tenant IDs.
* SHOULD hash tenant identifiers using SHA-256 with per-deployment salt.
* SHOULD limit log retention (≤30 days RECOMMENDED).
* Training/analytics retention MUST be explicit and access-controlled.

**Structured Log Redaction**

* If structured logs contain operation arguments:

  * Any string value longer than 64 bytes MUST be replaced with:

    * `"content_hash": "sha256:<hash>"`,
    * `"len": <original_length>`.
  * Raw content MUST NOT be preserved in logs.

---

## 16. Performance Characteristics

### 16.1. Latency Targets (Indicative)

**Note:** Ranges are indicative for typical enterprise deployments:

| Operation Category | Typical Range | Notes |
|-------------------|---------------|-------|
| Graph CRUD | 1–10 ms | Single node/edge operations |
| Graph queries (simple) | 10–100 ms | Small result sets, simple patterns |
| Graph queries (complex) | 100–1000 ms | Large traversals, aggregations, joins |
| Graph batch | 100–5000 ms | Bulk operations, size-dependent |
| LLM token counting | 1–5 ms | Local tokenizer operations |
| LLM completion (small) | 100–1000 ms | <100 tokens, fast models |
| LLM completion (large) | 1000–30000 ms | Large contexts, many tokens |
| Vector search | 1–100 ms | ANN search, filter-dependent |
| Vector batch upsert | 10–1000 ms | Size and dimension dependent |
| Embedding single | 5–50 ms | Typical embedding models |
| Embedding batch | 10–1000 ms | Batch size and model dependent |

### 16.2. Concurrency Limits

* `capabilities.limits` SHOULD include:

  * `concurrency`
  * `rate_limit_qps`
  * `max_batch_ops`
  * `max_top_k`

### 16.3. Caching Strategies

* Embeddings: cache by `(model, normalize, sha256(text))`.
* LLM: cache only deterministic invocations (e.g., `temperature=0`).
* Vector: prefer caching at router; key by query spec hash.
* Graph: cache read-only queries by `(dialect, text, params)` hash.
* All caches MUST respect tenant isolation.

---

## 17. Implementation Guidelines

### 17.1. Adapter Pattern

* Centralize:

  * Validation
  * Error normalization
  * Metrics and tracing
* Provider-specific shims translate to backend SDKs.

### 17.2. Validation (MUST)

Adapters MUST:

* Reject:

  * Empty labels where disallowed.
  * Negative `top_k`.
  * NaN/Inf vectors or out-of-range numeric values.
* Enforce:

  * JSON-serializable `props`/`metadata`.
  * Valid message roles.
  * Parameter ranges.
  * Context limits.
  * `max_text_length` / `max_batch_size`.
  * Vector dimension matching.

### 17.3. Testing

#### 17.3.1. One-Command Conformance Testing

**Recommended: Make targets (from repo root)**
```bash
# Test ALL protocols at once (LLM + Vector + Graph + Embedding)
make test-all-conformance

# Test specific protocols
make test-llm-conformance
make test-vector-conformance
make test-graph-conformance
make test-embedding-conformance
```

**Alternative: Corpus SDK CLI**
Available when installed with the entrypoint: `[project.scripts] corpus-sdk = "corpus_sdk.cli:main"`

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

**Direct: pytest (no wrappers)**
```bash
# Run everything with coverage
pytest tests/ -v --cov=corpus_sdk --cov-report=html

# Run specific protocol suites
pytest tests/llm/ -v
pytest tests/vector/ -v
pytest tests/graph/ -v
pytest tests/embedding/ -v
```

#### 17.3.2. Conformance Test Coverage

The test suite validates:

* **Wire format compliance** - All request/response envelopes match specification
* **Error normalization** - Provider errors mapped to canonical taxonomy
* **Streaming semantics** - Proper frame sequencing and terminal events
* **Batch operations** - Partial failure handling and atomicity
* **Capability discovery** - Truthful feature reporting
* **Observability** - SIEM-safe metrics and logging
* **Security** - Tenant isolation and content redaction

#### 17.3.3. Corpus Protocol Suite Badge

Implementations passing the full conformance test suite may display:

![LLM Protocol](https://img.shields.io/badge/CorpusLLM%20Protocol-100%25%20Conformant-brightgreen)
![Vector Protocol](https://img.shields.io/badge/CorpusVector%20Protocol-100%25%20Conformant-brightgreen)
![Graph Protocol](https://img.shields.io/badge/CorpusGraph%20Protocol-100%25%20Conformant-brightgreen)
![Embedding Protocol](https://img.shields.io/badge/CorpusEmbedding%20Protocol-100%25%20Conformant-brightgreen)


**Requirements for badge display:**
- Pass 100% of protocol-specific conformance tests
- Implement all required operations for claimed protocols
- Maintain backward compatibility within major version
- Follow SIEM-safe observability requirements

---

## 18. Versioning and Compatibility

### 18.1. Semantic Versioning (MUST)

`MAJOR.MINOR.PATCH`:

* MAJOR: breaking.
* MINOR: additive, backward compatible.
* PATCH: fixes/clarifications only.

### 18.2. Version Identification and Negotiation

* Clients MAY send `X-Adapter-Protocol: {component}/v1.0`.
* Adapters MUST reject unsupported major versions with `NotSupported`.
* Supported versions MUST be advertised via `capabilities.protocol`.

### 18.3. Backward Compatibility

* Additive changes in `v1.x` MUST be backward compatible.
* Semantic changes requiring client changes MUST wait for `v2.0`.

### 18.4. Deprecation Policy

* Deprecations MUST be documented.
* Deprecated features SHOULD remain for at least one minor version before removal in the next MAJOR.

---

## 19. IANA Considerations

No IANA actions are required.

---

## 20. References

### 20.1. Normative References

* [RFC2119]
* [RFC8174]
* W3C Trace Context
* OpenTelemetry Specification
* Semantic Versioning 2.0.0

### 20.2. Informative References

* Corpus GitHub repositories as referenced in appendices.

---

## 21. Author's Address

Corpus Working Group
Email: [standards@corpusos.com](mailto:standards@corpusos.com)
GitHub: [https://github.com/corpus-sdk/standards](https://github.com/corpus-sdk/standards)

---

## Appendix A — End-to-End Example (Normative)

```python
import time, random, asyncio
from typing import Any, Mapping

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

summary = await llm_adapter.complete(
    messages=[
        {"role": "system", "content": "Summarize tersely."},
        {"role": "user", "content": f"Summarize docs: {doc_ids}"}
    ],
    max_tokens=256,
    temperature=0.2,
    model="gpt-4.1-mini",
    ctx=ctx
)

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
        delay_ms = getattr(e, "retry_after_ms", None) or \
                   (getattr(e, "details", {}).get("suggested_backoff_ms", 500))
        await asyncio.sleep(delay_ms / 1000)
    except (TransientNetwork, Unavailable):
        sleep = min((2 ** attempt) * 0.2, 5.0) * random.random()
        await asyncio.sleep(sleep)
    except DeadlineExceeded:
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
  "idempotent_writes": true,
  "supports_multi_tenant": true,
  "cache": {
    "supports_tags": true,
    "max_ttl_ms": 3600000
  }
}
```

---

## Appendix C — Wire-Level Envelopes

Core envelopes in this appendix follow the closed-envelope rules from §4.2.1 and SCHEMA.md: adapters MUST NOT add extra top-level fields beyond those shown for success, error, and streaming envelopes.

### Embedding Batch Request

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

### Embedding Batch Partial-Success Response

```json
{
  "ok": true,
  "code": "OK",
  "ms": 38.4,
  "result": {
    "processed_count": 2,
    "failed_count": 1,
    "failures": [
      {
        "index": 2,
        "error": "TEXT_TOO_LONG",
        "detail": "Input exceeds max_text_length"
      }
    ],
    "embeddings": [
      {
        "vector": [0.01, 0.02],
        "text": "a",
        "model": "example-embed-1",
        "dimensions": 2
      },
      {
        "vector": [0.03, 0.04],
        "text": "b", 
        "model": "example-embed-1",
        "dimensions": 2
      }
    ]
  }
}
```

### Streaming LLM over NDJSON

**Note:** The base adapter protocol uses canonical streaming success envelopes per §4.2.3. The event-stream format shown below is an optional gateway overlay per PROTOCOLS.md §2.5.

**Base Protocol Format (adapters MUST emit):**
```json
{"ok": true, "code": "STREAMING", "ms": 12.3, "chunk": {"text": "Hello", "is_final": false}}
{"ok": true, "code": "STREAMING", "ms": 15.7, "chunk": {"text": " world", "is_final": false}}
{"ok": true, "code": "STREAMING", "ms": 18.2, "chunk": {"text": "!", "is_final": true}}
```

**Gateway Event Overlay (optional, client-facing):**
```json
{"event":"data","data":{"text":"Hello","is_final":false}}
{"event":"data","data":{"text":" world","is_final":false}}
{"event":"end","code":"OK"}
```

### Error Envelope Example

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

* Replace user/tenant identifiers with irreversible hashes (SHA-256 with per-deployment salt).
* Replace prompts, graph queries, vectors with hashes or structural metadata in logs.
* For vectors/embeddings, log only aggregate statistics (e.g. norms, dimensions).
* Apply the 64-byte truncation+hash rule from §15 to any logged arguments.

---

## Appendix E — Implementation Status (Non-Normative)

### Official Implementations

1. **Corpus SDK (Python)**
   Repository: `https://github.com/corpus/corpus-sdk`
   Protocols: Graph, LLM, Vector, Embedding
   Conformance: 100% to this specification
   Status: Stable (`v1.0.0`)

### Third-Party Implementations

Implementations claiming Corpus-compatible status SHOULD:

* Implement required operations for claimed protocols.
* Pass the interoperability test suite.
* Submit a registration entry to the Known Implementations Registry.

### Interoperability Test Suite

Coverage:

* 400+ conformance tests
* Cross-protocol flows
* Error normalization and partial-failure handling
* Streaming semantics verification

---

## Appendix F — Change Log / Revision History (Non-Normative)

* **v1.0.0 — 2026-02-10**
  * Initial specification publication
  * Establishes complete Corpus Protocol Suite covering Graph, LLM, Vector, and Embedding protocols
  * Defines wire-first canonical form with envelopes and streaming frames
  * Implements unified error taxonomy and resilience patterns
  * Includes production-grade observability, security, and privacy requirements
  * Provides comprehensive implementation guidelines and testing framework

---

**Note:** Since this is the first published version, there are no previous versions to list. Future updates will be added chronologically above this initial entry or in a designated CHANGELOG.md.



---

## Appendix G — Migration from Existing APIs (Informative)

### From OpenAI-style APIs to Corpus Protocol

| OpenAI API                        | Corpus Protocol                                  |
| --------------------------------- | ------------------------------------------------ |
| `chat.completions.create()`       | `op: "llm.complete"`                             |
| `chat.completions.create(stream)` | `op: "llm.stream"`                               |
| `embeddings.create()`             | `op: "embedding.embed"` / `embed_batch`          |
| `responses.create()` (tools)      | `op: "llm.complete"` with tools (via extensions) |
| `RateLimitError`                  | `code: "RESOURCE_EXHAUSTED"`                     |
| `AuthenticationError`             | `code: "AUTH_ERROR"`                             |
| `BadRequestError`                 | `code: "BAD_REQUEST"`                            |

Migration notes:

* Map models via `llm.capabilities.models`.
* Normalize error handling to the Corpus taxonomy.
* Use `OperationContext` for deadlines, idempotency, and tracing.

### From Pinecone-style APIs to Corpus Protocol

| Pinecone API      | Corpus Protocol                                                       |
| ----------------- | --------------------------------------------------------------------- |
| `index.query()`   | `op: "vector.query"`                                                  |
| `index.upsert()`  | `op: "vector.upsert"`                                                 |
| `index.delete()`  | `op: "vector.delete"`                                                 |
| `namespace` field | `QuerySpec.namespace`, `UpsertSpec.namespace`, `DeleteSpec.namespace` |

Migration notes:

* Ensure vector dimensions match `capabilities.limits.dimension`.
* Map Pinecone-specific errors into `DimensionMismatch`, `IndexNotReady`, `ResourceExhausted`, etc.
* Adopt Corpus pagination semantics if applicable.

---
