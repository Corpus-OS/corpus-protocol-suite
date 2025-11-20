# CORPUS PROTOCOLS

**Table of Contents**
- [0. Document Metadata](#0-document-metadata)
- [1. Introduction](#1-introduction)
- [2. Common Foundation (All Adapters)](#2-common-foundation-all-adapters)
- [3. Shared Types](#3-shared-types)
- [4. Protocol Overview](#4-protocol-overview)
- [PART I — GRAPH PROTOCOL (v1.0)](#part-i--graph-protocol-v10)
- [5. Graph Capabilities](#5-graph-capabilities)
- [6. Graph Types](#6-graph-types)
- [7. Graph Operations](#7-graph-operations)
- [8. Graph Semantics (Normative)](#8-graph-semantics-normative)
- [PART II — LLM PROTOCOL (v1.0)](#part-ii--llm-protocol-v10)
- [9. LLM Capabilities](#9-llm-capabilities)
- [10. LLM Types](#10-llm-types)
- [11. LLM Operations](#11-llm-operations)
- [12. LLM Semantics (Normative)](#12-llm-semantics-normative)
- [PART III — VECTOR PROTOCOL (v1.0)](#part-iii--vector-protocol-v10)
- [13. Vector Capabilities](#13-vector-capabilities)
- [14. Vector Types](#14-vector-types)
- [15. Vector Operations](#15-vector-operations)
- [16. Vector Semantics (Normative)](#16-vector-semantics-normative)
- [PART IV — EMBEDDING PROTOCOL (v1.0)](#part-iv--embedding-protocol-v10)
- [17. Embedding Capabilities](#17-embedding-capabilities)
- [18. Embedding Types](#18-embedding-types)
- [19. Embedding Operations](#19-embedding-operations)
- [20. Embedding Semantics (Normative)](#20-embedding-semantics-normative)
- [PART V — CROSS-PROTOCOL SECTIONS](#part-v--cross-protocol-sections)
- [21. Observability Integration](#21-observability-integration)
- [22. Normalized Error Integration](#22-normalized-error-integration)
- [23. Testing & Conformance](#23-testing--conformance)
- [24. Security Requirements](#24-security-requirements)
- [25. Versioning, Deprecation & Evolution](#25-versioning-deprecation--evolution)
- [26. Compliance Matrices](#26-compliance-matrices)
- [27. Cross-Protocol Standardization Tables](#27-cross-protocol-standardization-tables)
- [28. Glossary](#28-glossary)
- [29. Appendix](#29-appendix)
- [30. Index](#30-index)

---

**Corpus Protocol Suite — Unified Specification for Graph, LLM, Vector, and Embedding Adapters**  
**protocols_version:** `1.0`

> This document defines the unified protocol specification for all Corpus adapters. It establishes the normative behavior, types, and semantics for Graph, LLM, Vector, and Embedding protocols while maintaining cross-protocol consistency.

> **Document Precedence:** When PROTOCOLS.md and SPECIFICATION.md disagree on **wire format or field names**, PROTOCOLS.md is authoritative. SPECIFICATION.md is descriptive and may contain language-specific reference bindings.

> **Type Definition Convention:** Type definitions use a TypeScript-like pseudo-notation as a language-neutral IDL for JSON structures. They are normative for the wire format. Python, Go, etc. bindings are reference implementations.

## 0. Document Metadata

### 0.1 protocols_version: "1.0"
- **Status:** Stable / Normative
- **Effective Date:** 2025-01-01
- **Replaces:** None (initial version)

### 0.2 Canonical Protocol IDs
- **LLM Protocol:** `"llm/v1.0"`
- **Graph Protocol:** `"graph/v1.0"`
- **Vector Protocol:** `"vector/v1.0"`
- **Embedding Protocol:** `"embedding/v1.0"`

### 0.3 Relationship to Companion Documents
- **SPECIFICATION.md:** Defines high-level architecture, design philosophy, and normative requirements referenced throughout this document
- **METRICS.md:** Defines observability requirements referenced in §21
- **ERRORS.md:** Defines error taxonomy referenced in §22
- **IMPLEMENTATION.md:** Provides implementation guidance and patterns

### 0.4 Intended Audience
- Adapter developers implementing Corpus protocol support
- Platform engineers building routing and orchestration systems
- SRE/Operations teams managing production deployments
- Client library developers consuming adapter services

### 0.5 Non-Goals
- Provider-specific implementation details
- Transport-layer specifics (HTTP/gRPC/WebSocket wire formats)
- Business logic or application-level semantics
- UI/UX considerations

### 0.6 Terminology & Conventions
- **Adapter:** Protocol implementation bridging Corpus APIs to provider backends
- **Provider:** Underlying service (OpenAI, Pinecone, Neo4j, etc.)
- **MUST/SHOULD/MAY:** RFC 2119 normative language
- **Tenant:** Logical isolation boundary for multi-tenant deployments
- **Namespace:** Protocol-specific isolation scope (Vector collections, Graph databases, etc.)

## 1. Introduction

### 1.1 Purpose of the Protocols
The Corpus Protocol Suite provides a unified interface for AI and data services, enabling:
- Consistent developer experience across different AI providers
- Standardized observability and error handling
- Portable applications across infrastructure providers
- Reliable production operation with clear semantics

### 1.2 Adapter Model Overview
Adapters act as translation layers between Corpus protocols and provider-specific APIs:
```
Corpus Protocol → Adapter → Provider API
     ↑                    ↓
   Client ←----------- Response
```

### 1.3 Core Guarantees
- **Consistency:** Same inputs produce same outputs (where applicable)
- **Observability:** All operations emit standardized metrics
- **Resilience:** Clear error semantics and retry guidance
- **Security:** SIEM-safe by design; no content leakage

### 1.4 Supported Transports
- HTTP/REST with JSON envelopes
- gRPC with protobuf (future)
- Async message queues (future)

### 1.5 High-Level Responsibilities of an Adapter
- Protocol compliance and capability reporting
- Error normalization and mapping
- Deadline propagation and enforcement
- Tenant isolation and security
- Metrics emission and observability

## 2. Common Foundation (All Adapters)

### 2.1 OperationContext
```typescript
interface OperationContext {
  request_id?: string;           // Unique request identifier
  idempotency_key?: string;      // Idempotency guarantee scope
  deadline_ms?: number;          // Absolute epoch milliseconds
  traceparent?: string;          // W3C trace context
  tenant?: string;               // Tenant identifier (hashed in metrics)
  attrs?: Map<string, any>;      // Opaque extension attributes
}
```

### 2.2 Tenant Isolation & Hashing
- **Tenant MUST NOT** appear in raw form in metrics, logs, or error messages
- **Tenant hash:** First 12 characters of `SHA256(tenant)` for telemetry
- **Isolation:** Adapters enforce tenant boundaries in provider API calls

### 2.3 Deadlines & Budgets
- **Propagation:** Adapters MUST propagate `deadline_ms` to provider APIs
- **Safety buffer:** Subtract 50-100ms from remaining time for network overhead
- **Expiration:** Reject operations where context deadline has expired with `DeadlineExceeded("deadline already exceeded")`
- **Enforcement:** Adapters use `DeadlinePolicy` to normalize `asyncio.TimeoutError` into `DEADLINE_EXCEEDED`

### 2.4 Wire-Level Envelope Standardization

**Canonical Request Envelope:**
```json
{
  "op": "protocol.operation",
  "ctx": {
    "request_id": "string|null",
    "idempotency_key": "string|null", 
    "deadline_ms": "int|null",
    "traceparent": "string|null",
    "tenant": "string|null",
    "attrs": { "...": "..." }
  },
  "args": { ... }
}
```

**Success Response Envelope:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 45.2,
  "result": { ... }
}
```

**Error Response Envelope:**
```json
{
  "ok": false,
  "code": "BAD_REQUEST",
  "error": "BadRequest",
  "message": "human readable",
  "retry_after_ms": 5000,
  "details": { ... },
  "ms": 45.2
}
```

**Streaming Frame Envelope (Adapter-level):**
```json
{
  "ok": true,
  "code": "OK", 
  "ms": 45.2,
  "chunk": { ... }
}
```

### 2.5 Event Stream vs Base Protocol
- **Base Adapter Protocol:** Uses `{ok, code, ms, chunk}` envelope for streaming
- **Event Stream Layer:** `{type: "data", data: { ... }}` is an optional higher-level overlay for routers/gateways
- **Clarification:** Event-stream shape is a v2 overlay, not part of the base adapter protocol

### 2.6 Mode Strategy: Thin vs Standalone
- **Thin mode:** For composition under external control planes. All policies are no-op: no caching, no rate limiting, no circuit breaker, no deadline enforcement.
- **Standalone mode:** For direct use. Enables basic deadline enforcement, circuit breaker, in-memory TTL cache, and token-bucket rate limiter.
- **Implementation detail:** These are recommended infra patterns, not part of the wire contract.

### 2.7 Stream Semantics
- **Single terminal:** Exactly one terminal event (success or error)
- **No content after terminal:** Stream MUST end after final event
- **Heartbeats:** Optional keep-alive messages allowed but not required
- **Backpressure:** Clients control consumption rate; adapters MUST implement bounded buffering and flow control

### 2.8 Thread Safety & In-Memory Infrastructure
- **Adapter instances are not thread-safe** by default
- **In-memory implementations are not distributed** - state is local to process
- **Concurrent access** requires external synchronization by callers
- **Standalone mode** assumes single-process deployment constraints

### 2.9 Cache Serialization Constraints
- **All cached data MUST be JSON-serializable**
- **Metadata maps** MUST contain only JSON-serializable values
- **Filter expressions** MUST be serializable for cache key composition
- **Vector data** may require custom serialization for performance

### 2.10 Normalized Error Model
All adapters MUST use protocol-specific base errors:
- **LLM:** `LLMAdapterError` and subclasses
- **Graph:** `GraphAdapterError` and subclasses  
- **Vector:** `VectorAdapterError` and subclasses
- **Embedding:** `EmbeddingAdapterError` and subclasses

**Canonical Error Codes:** `BAD_REQUEST`, `AUTH_ERROR`, `RESOURCE_EXHAUSTED`, `TRANSIENT_NETWORK`, `UNAVAILABLE`, `NOT_SUPPORTED`, `DEADLINE_EXCEEDED`

**Wire Error Mapping:** Error envelope `code` comes from `e.code` if set, otherwise `type(e).__name__.upper()`

> **Error Naming Convention:** 
> - **Wire codes:** `ALL_CAPS_SNAKE` (e.g., `DIMENSION_MISMATCH`)
> - **Exception/type names:** `PascalCase` (e.g., `DimensionMismatch`)
> - Wire `code` MUST use the canonical ALL_CAPS identifier.

### 2.11 Observability & Metrics
- **All operations** emit standardized metrics with tenant hashing
- **SIEM-safe:** No raw content (prompts, vectors, embeddings) in logs, metrics, or errors
- **Low cardinality:** Bounded label sets in all telemetry

### 2.12 Capability Probing
- **Dynamic discovery:** Clients probe `capabilities()` to determine supported features
- **Truthful reporting:** Adapters MUST accurately report actual capabilities
- **Protocol field:** All capabilities objects MUST expose a `protocol` field with the canonical protocol ID
- **Caching:** Capabilities may be cached with appropriate TTL (typically 5-60 minutes)

### 2.13 Idempotency Expectations
| Operation Type | Idempotent? | Notes |
|---------------|-------------|--------|
| Read operations | Yes | Same inputs → same outputs |
| Create with ID | Yes | Duplicate ID returns existing |
| Create without ID | No | Generates new ID each time |
| Update | Yes | Last write wins |
| Delete | Yes | Multiple deletes return success |

### 2.14 Global Invariants (MUST/MUST NOT)

**All adapters MUST:**
- Emit metrics for every operation with tenant hashing
- Propagate `deadline_ms` to provider APIs with safety buffer
- Map provider errors to canonical error taxonomy
- Report capabilities truthfully including `protocol` field
- Enforce SIEM-safe requirements for all telemetry
- Use canonical wire envelopes for all responses

**All adapters MUST NOT:**
- Log raw prompts, vectors, embeddings, or tenant IDs
- Exceed provider batch size limits without client consent
- Return unnormalized errors with provider-specific details
- Cache capabilities beyond reasonable TTL (max 1 hour)
- Process requests after context deadline expiration

## 3. Shared Types

### 3.1 Numeric Types
- **IDs:** `string` type, provider-specific format
- **Vectors:** `number[]` with consistent precision
- **Floats:** IEEE 754 double-precision
- **Integers:** 64-bit signed integers where applicable

### 3.2 Metadata Maps
```typescript
type Metadata = {
  [key: string]: string | number | boolean | null | string[] | number[];
};
```
- **Constraints:** JSON-serializable values only
- **Cardinality:** Bounded key sets recommended for filter performance

### 3.3 Paging / Streaming Tokens
- **Opaque:** Clients treat as black strings
- **Stateless:** Adapters may encode state but should avoid large payloads
- **Expiration:** Tokens should be valid for reasonable periods (hours/days)

### 3.4 Batches & Partial Failure Envelope
```typescript
interface BatchResult<T> {
  processed_count: number;
  failed_count: number;
  failures: Array<{
    id?: string;           // Item identifier when available
    index?: number;        // Batch position when no ID
    error: string;         // Normalized error type
    detail: string;        // Human-readable details
  }>;
  results?: T[];          // Successful results when applicable
}
```

### 3.5 Common Validation Rules
- **Required fields:** Presence validated before provider calls
- **Type checking:** JSON schema validation where applicable
- **Range validation:** Numeric bounds enforcement
- **Size limits:** Payload size constraints per capabilities

### 3.6 JSON Schema Requirements
- **Additional properties:** MUST be `false` unless explicitly allowed
- **Nullability:** Fields explicitly marked optional vs required
- **String formats:** UUID, ISO8601, email where semantically meaningful
- **Array bounds:** Minimum/maximum lengths specified per capability

### 3.7 Shared Token Usage Type
```typescript
interface TokenUsage {
  prompt_tokens: number;
  completion_tokens?: number;  // Optional for non-generation operations
  total_tokens: number;
}
```

## 4. Protocol Overview

### 4.1 Graph Protocol
- **Purpose:** Property graph storage and querying
- **Key operations:** CRUD on nodes/edges, graph queries, batch operations
- **Primary use cases:** Knowledge graphs, recommendation systems, fraud detection

### 4.2 LLM Protocol
- **Purpose:** Large language model inference
- **Key operations:** Completion, streaming, token counting
- **Primary use cases:** Chat applications, content generation, summarization

### 4.3 Vector Protocol
- **Purpose:** Vector similarity search and storage
- **Key operations:** Query, upsert, delete, namespace management
- **Primary use cases:** Semantic search, RAG, recommendation, clustering

### 4.4 Embedding Protocol
- **Purpose:** Text-to-vector embedding generation
- **Key operations:** Embed, batch embed, token counting
- **Primary use cases:** Text vectorization for search and ML pipelines

### 4.5 Cross-Protocol Consistency Requirements
- **Error handling:** Unified taxonomy across all protocols
- **Observability:** Consistent metrics and logging patterns
- **Tenant isolation:** Uniform multi-tenant support
- **Deadline propagation:** Standard timeout handling

### 4.6 Adapter Lifecycle Responsibilities
- **Initialization:** Configuration validation, provider connectivity
- **Health checking:** Continuous provider availability monitoring
- **Capability reporting:** Dynamic feature availability
- **Cleanup:** Resource release on shutdown

---

## PART I — GRAPH PROTOCOL (v1.0)

## 5. Graph Capabilities

### 5.1 Required Fields
```typescript
interface GraphCapabilities {
  protocol: string;               // "graph/v1.0"
  server: string;                 // Adapter identifier
  version: string;                // Protocol version
  supported_query_dialects: string[]; // e.g., ["cypher", "gremlin"]
  supports_stream_query: boolean;
  supports_bulk_vertices: boolean;
  supports_batch: boolean;
  supports_schema: boolean;
  idempotent_writes: boolean;
  supports_deadline: boolean;
  supports_namespaces: boolean;
  supports_property_filters: boolean;
  supports_multi_tenant: boolean;
  max_batch_ops?: number;
  max_query_complexity?: number;  // Optional: maximum query complexity score
  max_nodes_per_transaction?: number; // Optional: transaction size limits
  supports_analytics_queries?: boolean; // Optional: analytical query support
}
```

### 5.2 Validation Rules
- **Truthfulness:** Reported capabilities MUST match actual provider support
- **Consistency:** Capabilities SHOULD remain stable between health checks
- **Discovery:** Clients SHOULD probe capabilities before using advanced features

### 5.3 Query Dialect Negotiation
- **Client preference:** Clients specify dialect in query requests
- **Adapter fallback:** Adapters MAY translate between dialects if supported
- **Error on unsupported:** `NOT_SUPPORTED` error for unknown dialects
- **Parameter binding:** All dialects MUST support named parameter binding

## 6. Graph Types

### 6.1 Node
```typescript
interface Node {
  id: string;
  labels: string[];               // Multiple labels supported
  properties: Metadata;
  namespace?: string;             // Optional namespace isolation
}
```

### 6.2 Edge
```typescript
interface Edge {
  id: string;
  src: string;                    // Source node ID
  dst: string;                    // Target node ID  
  label: string;
  properties: Metadata;
  namespace?: string;             // Optional namespace isolation
}
```

### 6.3 QuerySpec
```typescript
interface QuerySpec {
  text: string;                   // Query in supported dialect
  params?: Metadata;              // Named parameters for query
  timeout_ms?: number;            // Query-specific timeout
  dialect?: string;               // Preferred query dialect
  namespace?: string;             // Namespace context
  stream?: boolean;               // Stream results
}
```

### 6.4 QueryResult
```typescript
interface QueryResult {
  records: any[];                 // Query result data
  summary: {                      // Execution summary (REQUIRED)
    query_time_ms: number;
    results_count: number;
    has_more: boolean;
    dialect_used: string;         // Actual dialect used
  };
  dialect?: string;               // Optional dialect used
  namespace?: string;             // Optional namespace context
}
```

### 6.5 QueryChunk
```typescript
interface QueryChunk {
  records: any[];                 // Incremental result data
  is_final: boolean;              // True for final chunk
  summary?: {                     // Final execution summary
    query_time_ms: number;
    results_count: number;
    has_more: boolean;
    dialect_used: string;
  };
}
```

## 7. Graph Operations

### 7.1 capabilities
**Purpose:** Discover supported graph features and limits

**Operation:** `graph.capabilities`

**Request Body:**
```json
{
  "op": "graph.capabilities",
  "ctx": {
    "request_id": "req-graph-cap-001",
    "tenant": "acme-corp"
  },
  "args": {}
}
```

**Output:** `GraphCapabilities`

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 1.2,
  "result": {
    "protocol": "graph/v1.0",
    "server": "neo4j-adapter",
    "version": "1.0.0",
    "supported_query_dialects": ["cypher"],
    "supports_stream_query": true,
    "supports_bulk_vertices": true,
    "supports_batch": true,
    "supports_schema": true,
    "idempotent_writes": true,
    "supports_deadline": true,
    "supports_namespaces": true,
    "supports_property_filters": true,
    "supports_multi_tenant": true,
    "max_batch_ops": 1000,
    "max_query_complexity": 10000,
    "max_nodes_per_transaction": 50000,
    "supports_analytics_queries": true
  }
}
```

### 7.2 upsert_nodes
**Purpose:** Create or update multiple nodes

**Operation:** `graph.upsert_nodes`

**Input:**
```typescript
interface UpsertNodesSpec {
  nodes: Node[];
  namespace?: string;
}
```

**Validation:**
- `nodes` MUST be non-empty
- Each `Node.properties` MUST be JSON-serializable

**Request Body:**
```json
{
  "op": "graph.upsert_nodes",
  "ctx": {
    "request_id": "req-graph-upsert-nodes-001",
    "tenant": "acme-corp",
    "idempotency_key": "batch-user-import-20250115"
  },
  "args": {
    "nodes": [
      {
        "id": "user-alice-123",
        "labels": ["User", "Premium"],
        "properties": {
          "name": "Alice Smith",
          "email": "alice@example.com",
          "age": 30,
          "department": "Engineering"
        },
        "namespace": "production"
      }
    ],
    "namespace": "production"
  }
}
```

**Output:** `BatchResult<Node>`

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 45.7,
  "result": {
    "processed_count": 1,
    "failed_count": 0,
    "failures": [],
    "results": [
      {
        "id": "user-alice-123",
        "labels": ["User", "Premium"],
        "properties": {
          "name": "Alice Smith",
          "email": "alice@example.com",
          "age": 30,
          "department": "Engineering"
        },
        "namespace": "production"
      }
    ]
  }
}
```

### 7.3 upsert_edges
**Purpose:** Create or update multiple edges

**Operation:** `graph.upsert_edges`

**Input:**
```typescript
interface UpsertEdgesSpec {
  edges: Edge[];
  namespace?: string;
}
```

**Validation:**
- `edges` MUST be non-empty
- `edge.label` MUST be non-empty string
- Each `Edge.properties` MUST be JSON-serializable

**Request Body:**
```json
{
  "op": "graph.upsert_edges",
  "ctx": {
    "request_id": "req-graph-upsert-edges-001",
    "tenant": "acme-corp"
  },
  "args": {
    "edges": [
      {
        "id": "works-with-001",
        "src": "user-alice-123",
        "dst": "user-bob-456",
        "label": "WORKS_WITH",
        "properties": {
          "since": "2024-06-01",
          "project": "Phoenix"
        },
        "namespace": "production"
      }
    ],
    "namespace": "production"
  }
}
```

**Output:** `BatchResult<Edge>`

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 32.1,
  "result": {
    "processed_count": 1,
    "failed_count": 0,
    "failures": [],
    "results": [
      {
        "id": "works-with-001",
        "src": "user-alice-123",
        "dst": "user-bob-456",
        "label": "WORKS_WITH",
        "properties": {
          "since": "2024-06-01",
          "project": "Phoenix"
        },
        "namespace": "production"
      }
    ]
  }
}
```

### 7.4 delete_nodes
**Purpose:** Remove nodes by ID or filter

**Operation:** `graph.delete_nodes`

**Input:**
```typescript
interface DeleteNodesSpec {
  ids: string[];
  namespace?: string;
  filter?: Metadata;
}
```

**Validation:**
- MUST have either non-empty `ids` or `filter` (not both empty)
- `filter`, if present, MUST be JSON-serializable

**Request Body:**
```json
{
  "op": "graph.delete_nodes",
  "ctx": {
    "request_id": "req-graph-delete-nodes-001",
    "tenant": "acme-corp"
  },
  "args": {
    "ids": ["user-alice-123"],
    "namespace": "production"
  }
}
```

**Output:** `{ deleted_count: number }`

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 18.3,
  "result": {
    "deleted_count": 1
  }
}
```

### 7.5 delete_edges
**Purpose:** Remove edges by ID or filter

**Operation:** `graph.delete_edges`

**Input:**
```typescript
interface DeleteEdgesSpec {
  ids: string[];
  namespace?: string;
  filter?: Metadata;
}
```

**Validation:**
- MUST have either non-empty `ids` or `filter` (not both empty)
- `filter`, if present, MUST be JSON-serializable

**Request Body:**
```json
{
  "op": "graph.delete_edges",
  "ctx": {
    "request_id": "req-graph-delete-edges-001",
    "tenant": "acme-corp"
  },
  "args": {
    "ids": ["works-with-001"],
    "namespace": "production"
  }
}
```

**Output:** `{ deleted_count: number }`

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 15.7,
  "result": {
    "deleted_count": 1
  }
}
```

### 7.6 query
**Purpose:** Execute a graph query and return results

**Operation:** `graph.query`

**Input:** `QuerySpec`

**Request Body:**
```json
{
  "op": "graph.query",
  "ctx": {
    "request_id": "req-graph-query-001",
    "tenant": "acme-corp",
    "deadline_ms": 1736929200000
  },
  "args": {
    "text": "MATCH (u:User) WHERE u.department = $dept RETURN u.name, u.email",
    "params": {
      "dept": "Engineering"
    },
    "dialect": "cypher",
    "timeout_ms": 5000
  }
}
```

**Output:** `QueryResult`

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 45.2,
  "result": {
    "records": [
      {
        "u.name": "Alice Smith",
        "u.email": "alice@example.com"
      }
    ],
    "summary": {
      "query_time_ms": 42.1,
      "results_count": 1,
      "has_more": false,
      "dialect_used": "cypher"
    },
    "dialect": "cypher"
  }
}
```

### 7.7 stream_query
**Purpose:** Execute a graph query with streaming results

**Operation:** `graph.stream_query`

**Input:** `QuerySpec`

**Request Body:**
```json
{
  "op": "graph.stream_query",
  "ctx": {
    "request_id": "req-graph-stream-001",
    "tenant": "acme-corp"
  },
  "args": {
    "text": "MATCH (u:User) RETURN u.id, u.name, u.department ORDER BY u.name",
    "dialect": "cypher"
  }
}
```

**Output:** `AsyncIterable<QueryChunk>`

**Stream Response Frames:**
```json
{"ok": true, "code": "OK", "ms": 12.3, "chunk": {"records": [{"u.id": "user-alice-123", "u.name": "Alice Smith", "u.department": "Engineering"}], "is_final": false}}
{"ok": true, "code": "OK", "ms": 15.7, "chunk": {"records": [{"u.id": "user-bob-456", "u.name": "Bob Johnson", "u.department": "Sales"}], "is_final": false}}
{"ok": true, "code": "OK", "ms": 18.2, "chunk": {"summary": {"query_time_ms": 125.4, "results_count": 2, "dialect_used": "cypher", "has_more": false}, "is_final": true}}
```

### 7.8 bulk_vertices
**Purpose:** Bulk operations on vertices (import/export)

**Operation:** `graph.bulk_vertices`

**Availability:** Only if `capabilities.supports_bulk_vertices` is true

**Request Body:**
```json
{
  "op": "graph.bulk_vertices",
  "ctx": {
    "request_id": "req-graph-bulk-001",
    "tenant": "acme-corp"
  },
  "args": {
    "operation": "export",
    "format": "json",
    "namespace": "production"
  }
}
```

**Output:** `{ nodes: Node[] }`

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 89.5,
  "result": {
    "nodes": [
      {
        "id": "user-alice-123",
        "labels": ["User", "Premium"],
        "properties": {
          "name": "Alice Smith",
          "email": "alice@example.com",
          "age": 30,
          "department": "Engineering"
        },
        "namespace": "production"
      }
    ]
  }
}
```

### 7.9 batch
**Purpose:** Execute multiple operations in batch

**Operation:** `graph.batch`

**Availability:** Only if `capabilities.supports_batch` is true

**Validation:**
- `ops` MUST be non-empty
- If `max_batch_ops` is non-null, `len(ops) <= max_batch_ops`

**Request Body:**
```json
{
  "op": "graph.batch",
  "ctx": {
    "request_id": "req-graph-batch-001",
    "tenant": "acme-corp"
  },
  "args": {
    "ops": [
      {
        "op": "upsert_nodes",
        "args": {
          "nodes": [
            {
              "id": "user-charlie-789",
              "labels": ["User"],
              "properties": {
                "name": "Charlie Brown",
                "email": "charlie@example.com"
              }
            }
          ]
        }
      }
    ]
  }
}
```

**Output:** `BatchResult<any>`

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 23.4,
  "result": {
    "processed_count": 1,
    "failed_count": 0,
    "failures": [],
    "results": [
      {
        "processed_count": 1,
        "failed_count": 0,
        "failures": []
      }
    ]
  }
}
```

### 7.10 get_schema
**Purpose:** Retrieve graph schema information

**Operation:** `graph.get_schema`

**Request Body:**
```json
{
  "op": "graph.get_schema",
  "ctx": {
    "request_id": "req-graph-schema-001",
    "tenant": "acme-corp"
  },
  "args": {
    "namespace": "production"
  }
}
```

**Output:** 
```typescript
interface GraphSchema {
  node_labels: string[];
  edge_types: string[];
  property_keys: string[];
  constraints: any[];
  indexes: any[];
}
```

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 12.3,
  "result": {
    "node_labels": ["User", "Premium"],
    "edge_types": ["WORKS_WITH"],
    "property_keys": ["name", "email", "age", "department", "since", "project"],
    "constraints": [
      {
        "type": "UNIQUE",
        "label": "User",
        "property": "email"
      }
    ],
    "indexes": [
      {
        "type": "BTREE",
        "label": "User",
        "property": "department"
      }
    ]
  }
}
```

### 7.11 health
**Purpose:** Check adapter and provider health status

**Operation:** `graph.health`

**Request Body:**
```json
{
  "op": "graph.health",
  "ctx": {
    "request_id": "req-graph-health-001",
    "tenant": "acme-corp"
  },
  "args": {}
}
```

**Output:**
```typescript
interface GraphHealthStatus {
  ok: boolean;
  status: "ok" | "degraded";
  server: string;
  version: string;
  namespaces?: { 
    [namespace: string]: {
      node_count: number;
      edge_count: number;
      ready: boolean;
      read_only?: boolean;
    }
  };
  read_only?: boolean;
  degraded?: boolean;
  message?: string;  // Optional status message
  last_check?: string; // Optional timestamp of last health check
}
```

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 2.1,
  "result": {
    "ok": true,
    "status": "ok",
    "server": "neo4j-adapter",
    "version": "1.0.0",
    "namespaces": {
      "production": {
        "node_count": 1500,
        "edge_count": 3200,
        "ready": true,
        "read_only": false
      }
    },
    "read_only": false,
    "degraded": false,
    "message": "All systems operational",
    "last_check": "2025-01-15T10:20:00Z"
  }
}
```

## 8. Graph Semantics (Normative)

### 8.1 Consistency Requirements
- **Read-after-write:** Updates MUST be visible to subsequent queries
- **Causal consistency:** Operations from same client MUST be observed in order
- **Cross-tenant isolation:** No data leakage between tenants

### 8.2 Referential Integrity Rules
- **Node deletion:** Connected edges MUST be automatically deleted when nodes are deleted
- **Edge validation:** Source and target nodes MUST exist
- **ID uniqueness:** Node/edge IDs MUST be unique within their namespace

### 8.3 Streaming Guarantees
- **Exactly-once delivery:** Each row MUST be delivered exactly once
- **Order preservation:** Results MUST be delivered in query result order
- **Terminal event:** Stream MUST always end with final chunk (`is_final: true`)
- **Cardinality bounds:** Streams SHOULD support at least 1M rows

### 8.4 Batch Operation Semantics
- **Order preservation:** Operations MUST be executed in specified order
- **Atomic batches:** When `atomic=true`, all operations MUST succeed or fail together
- **Partial visibility:** Non-atomic batch results MAY be visible as they complete
- **Failure isolation:** Individual operation failures MUST NOT affect others in non-atomic mode

### 8.5 Error Mappings
| Provider Error | Normalized Error | Wire Code | Details |
|----------------|------------------|-----------|---------|
| Syntax error | `QueryParseError` | `QUERY_PARSE_ERROR` | `{"dialect": "cypher", "position": 45}` |
| Unknown node | `VertexNotFound` | `VERTEX_NOT_FOUND` | `{"node_id": "v123"}` |
| Unknown edge | `EdgeNotFound` | `EDGE_NOT_FOUND` | `{"edge_id": "e456"}` |
| Schema violation | `SchemaValidationError` | `SCHEMA_VALIDATION_ERROR` | `{"constraint": "label_missing"}` |
| Timeout | `DeadlineExceeded` | `DEADLINE_EXCEEDED` | `{"query_time_ms": 5000}` |

### 8.6 Deadlines
- **Query timeout:** `timeout_ms` in QuerySpec overrides context deadline
- **Stream duration:** Deadline applies to entire stream execution
- **Batch operations:** Deadline applies to entire batch execution

---

## PART II — LLM PROTOCOL (v1.0)

## 9. LLM Capabilities

### 9.1 Required Fields
```typescript
interface LLMCapabilities {
  protocol: string;               // "llm/v1.0"
  server: string;
  version: string;
  model_family: string;           // e.g., "openai", "anthropic", "cohere"
  supported_models: string[];
  max_context_length: number;
  supports_streaming: boolean;
  supports_roles: boolean;
  supports_system_message: boolean;
  supports_json_output: boolean;
  supports_parallel_tool_calls: boolean;
  supports_deadline: boolean;
  supports_count_tokens: boolean;
  idempotent_writes: boolean;
  supports_multi_tenant: boolean;
  max_tokens_per_minute?: number; // Optional: rate limiting info
  max_requests_per_minute?: number; // Optional: request rate limits
  supports_function_calling?: boolean; // Optional: function calling support
  supports_vision?: boolean;      // Optional: multimodal vision support
  supports_logprobs?: boolean;    // Optional: token probabilities
}
```

### 9.2 Model Family & Supported Models
- **Model listing:** Accurate list of available provider models
- **Context lengths:** Per-model context limits where they differ
- **Feature support:** Model-specific capabilities (JSON, tools, etc.)

## 10. LLM Types

### 10.1 Message Schema
```typescript
interface Message {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  name?: string;                  // Tool call function name
  tool_call_id?: string;          // Associate tool calls with responses
  tool_calls?: ToolCall[];        // Function calls from model
}

interface ToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;           // JSON string
  };
}
```

### 10.2 CompletionSpec
```typescript
interface CompletionSpec {
  model: string;                  // Target model - SINGLE DEFINITION
  messages: Message[];
  max_tokens?: number;
  temperature?: number;          // Range: [0.0, 2.0]
  top_p?: number;                // Range: (0.0, 1.0]
  frequency_penalty?: number;    // Range: [-2.0, 2.0]
  presence_penalty?: number;     // Range: [-2.0, 2.0]
  stop_sequences?: string[];
  system_message?: string;       // System message override
}
```

### 10.3 LLMCompletion
```typescript
interface LLMCompletion {
  text: string;                  // Generated completion text
  model: string;
  usage: TokenUsage;
  finish_reason: string;         // Reason generation stopped
}
```

### 10.4 LLMChunk
```typescript
interface LLMChunk {
  text: string;                  // Incremental text content
  is_final: boolean;             // True for final chunk
  model?: string;                // Optional model identifier
  usage_so_far?: TokenUsage;     // Cumulative token usage
}
```

## 11. LLM Operations

### 11.1 capabilities
**Purpose:** Discover supported LLM features and models

**Operation:** `llm.capabilities`

**Request Body:**
```json
{
  "op": "llm.capabilities",
  "ctx": {
    "request_id": "req-llm-cap-001",
    "tenant": "acme-corp"
  },
  "args": {}
}
```

**Output:** `LLMCapabilities`

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 1.5,
  "result": {
    "protocol": "llm/v1.0",
    "server": "openai-adapter",
    "version": "1.0.0",
    "model_family": "openai",
    "supported_models": ["gpt-4.1-mini", "gpt-4.1-max"],
    "max_context_length": 128000,
    "supports_streaming": true,
    "supports_roles": true,
    "supports_system_message": true,
    "supports_json_output": true,
    "supports_parallel_tool_calls": true,
    "supports_deadline": true,
    "supports_count_tokens": true,
    "idempotent_writes": true,
    "supports_multi_tenant": true,
    "max_tokens_per_minute": 100000,
    "max_requests_per_minute": 1000,
    "supports_function_calling": true,
    "supports_vision": false,
    "supports_logprobs": true
  }
}
```

### 11.2 complete
**Purpose:** Generate LLM completion for given messages

**Operation:** `llm.complete`

**Input:** `CompletionSpec`

**Validation:**
- `messages` MUST be non-empty list of `{role, content}` objects
- `messages` MUST be JSON-serializable
- Parameter ranges enforced per `BaseLLMAdapter._validate_sampling_params`:
  - `temperature ∈ [0.0, 2.0]`
  - `top_p ∈ (0.0, 1.0]`
  - `frequency_penalty ∈ [-2.0, 2.0]`
  - `presence_penalty ∈ [-2.0, 2.0]`

**Request Body:**
```json
{
  "op": "llm.complete",
  "ctx": {
    "request_id": "req-llm-comp-001",
    "tenant": "acme-corp",
    "deadline_ms": 1736929300000
  },
  "args": {
    "model": "gpt-4.1-mini",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant that provides concise answers."
      },
      {
        "role": "user", 
        "content": "What are the main benefits of renewable energy?"
      }
    ],
    "max_tokens": 150,
    "temperature": 0.7,
    "top_p": 0.9
  }
}
```

**Output:** `LLMCompletion`

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 1250.8,
  "result": {
    "text": "Renewable energy offers several key benefits:\n\n1. Environmental sustainability - Reduces greenhouse gas emissions and air pollution\n2. Energy security - Decreases dependence on fossil fuel imports\n3. Cost stability - Renewable sources have predictable long-term costs\n4. Job creation - Growing industry creates new employment opportunities\n5. Public health - Cleaner air leads to better health outcomes",
    "model": "gpt-4.1-mini",
    "usage": {
      "prompt_tokens": 28,
      "completion_tokens": 87,
      "total_tokens": 115
    },
    "finish_reason": "stop"
  }
}
```

### 11.3 stream
**Purpose:** Stream LLM completion incrementally

**Operation:** `llm.stream`

**Input:** `CompletionSpec` (same parameters as `llm.complete`)

**Request Body:**
```json
{
  "op": "llm.stream",
  "ctx": {
    "request_id": "req-llm-stream-001",
    "tenant": "acme-corp"
  },
  "args": {
    "model": "gpt-4.1-mini",
    "messages": [
      {
        "role": "user",
        "content": "Explain quantum computing in simple terms."
      }
    ],
    "max_tokens": 200,
    "temperature": 0.8
  }
}
```

**Output:** `AsyncIterable<LLMChunk>`

**Stream Response Frames:**
```json
{"ok": true, "code": "OK", "ms": 12.3, "chunk": {"text": "Quantum", "is_final": false, "model": "gpt-4.1-mini"}}
{"ok": true, "code": "OK", "ms": 15.7, "chunk": {"text": " computing", "is_final": false, "model": "gpt-4.1-mini"}}
{"ok": true, "code": "OK", "ms": 18.2, "chunk": {"text": " is", "is_final": false, "model": "gpt-4.1-mini"}}
{"ok": true, "code": "OK", "ms": 21.4, "chunk": {"text": " a", "is_final": false, "model": "gpt-4.1-mini"}}
{"ok": true, "code": "OK", "ms": 24.8, "chunk": {"text": " new", "is_final": false, "model": "gpt-4.1-mini"}}
{"ok": true, "code": "OK", "ms": 28.1, "chunk": {"text": " type", "is_final": false, "model": "gpt-4.1-mini"}}
{"ok": true, "code": "OK", "ms": 31.5, "chunk": {"text": " of", "is_final": false, "model": "gpt-4.1-mini"}}
{"ok": true, "code": "OK", "ms": 34.9, "chunk": {"text": " computing", "is_final": false, "model": "gpt-4.1-mini"}}
{"ok": true, "code": "OK", "ms": 38.2, "chunk": {"text": " that", "is_final": false, "model": "gpt-4.1-mini"}}
{"ok": true, "code": "OK", "ms": 41.6, "chunk": {"text": " uses", "is_final": false, "model": "gpt-4.1-mini"}}
{"ok": true, "code": "OK", "ms": 45.0, "chunk": {"text": " quantum", "is_final": false, "model": "gpt-4.1-mini"}}
{"ok": true, "code": "OK", "ms": 48.3, "chunk": {"text": " bits", "is_final": false, "model": "gpt-4.1-mini"}}
{"ok": true, "code": "OK", "ms": 51.7, "chunk": {"text": " (qubits)", "is_final": true, "model": "gpt-4.1-mini", "usage_so_far": {"prompt_tokens": 12, "completion_tokens": 18, "total_tokens": 30}}}
```

### 11.4 count_tokens
**Purpose:** Count tokens in text for a specific model

**Operation:** `llm.count_tokens`

**Input:** 
```typescript
interface CountTokensSpec {
  text: string;
  model: string;
}
```

**Request Body:**
```json
{
  "op": "llm.count_tokens",
  "ctx": {
    "request_id": "req-llm-tokens-001",
    "tenant": "acme-corp"
  },
  "args": {
    "text": "The quick brown fox jumps over the lazy dog",
    "model": "gpt-4.1-mini"
  }
}
```

**Output:** `number` (bare integer)

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 3.2,
  "result": 9
}
```

### 11.5 health
**Purpose:** Check LLM provider health and model availability

**Operation:** `llm.health`

**Request Body:**
```json
{
  "op": "llm.health",
  "ctx": {
    "request_id": "req-llm-health-001",
    "tenant": "acme-corp"
  },
  "args": {}
}
```

**Output:**
```typescript
interface LLMHealthStatus {
  ok: boolean;
  server: string;
  version: string;
  status?: "ok" | "degraded";  // Optional status indicator
  models?: {                   // Optional model status information
    [model: string]: {
      available: boolean;
      rate_limited?: boolean;
      degraded?: boolean;
    };
  };
  rate_limits?: {             // Optional rate limit information
    tokens_per_minute?: number;
    requests_per_minute?: number;
  };
  message?: string;           // Optional status message
}
```

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 5.1,
  "result": {
    "ok": true,
    "server": "openai-adapter",
    "version": "1.0.0",
    "status": "ok",
    "models": {
      "gpt-4.1-mini": {
        "available": true,
        "rate_limited": false,
        "degraded": false
      },
      "gpt-4.1-max": {
        "available": true,
        "rate_limited": false,
        "degraded": false
      }
    },
    "rate_limits": {
      "tokens_per_minute": 100000,
      "requests_per_minute": 1000
    },
    "message": "All models operational"
  }
}
```

## 12. LLM Semantics (Normative)

### 12.1 Message Ordering & Role Rules
- **Role sequence:** Valid transitions between message roles
- **System message:** Only one system message at conversation start
- **Tool calls:** Assistant messages may contain tool calls
- **Tool results:** Tool role messages follow corresponding tool calls

### 12.2 Determinism Requirements
- **Same inputs:** Identical messages + parameters → identical outputs when `temperature=0`
- **Stream equivalence:** Concatenated stream chunks MUST match non-streamed completion
- **Seed behavior:** When `seed` provided, identical outputs across requests

### 12.3 Stop Sequence Handling
- **Sequence matching:** Stop at first occurrence of any stop sequence
- **Partial matches:** Do not stop on partial sequence matches
- **Exclusion:** Stop sequences excluded from final output
- **Precedence:** Character-level matching takes precedence over token boundaries

### 12.4 Context Window Preflight (Optional v1 Feature)
- **Best-effort:** Context window preflight is optional and best-effort in v1
- **Error swallowing:** Errors from `count_tokens` during preflight are swallowed
- **Caller responsibility:** Callers must still handle backend "too long" errors
- **Capability gated:** Only performed if `supports_count_tokens` and `max_context_length > 0`

### 12.5 JSON Mode Strictness
- **Guaranteed JSON:** When `response_format: {type: "json_object"}`, output MUST be valid JSON
- **Schema adherence:** JSON structure follows model training, no schema enforcement
- **Error on violation:** Return `BAD_REQUEST` if JSON mode requested but prompt doesn't specify JSON

### 12.6 Tool Call Semantics
- **Deterministic ordering:** Tool calls returned in consistent order
- **Parallel execution:** When `supports_parallel_tool_calls=true`, tools may be called concurrently
- **Exhaustion behavior:** Stop generation after tool calls if no further text content

### 12.7 Streaming Guarantees
- **Chunk integrity:** Complete tokens delivered in each chunk
- **Order preservation:** Chunks delivered in correct sequence
- **Final indication:** Clear termination with `is_final: true`

### 12.8 Error Mappings
| Provider Error | Normalized Error | Wire Code | Details |
|----------------|------------------|-----------|---------|
| Rate limit exceeded | `ResourceExhausted` | `RESOURCE_EXHAUSTED` | `{"resource_scope": "model", "retry_after_ms": 5000}` |
| Model overloaded | `ResourceExhausted` | `RESOURCE_EXHAUSTED` | `{"resource_scope": "model", "retry_after_ms": 10000}` |
| Context length exceeded | `BadRequest` | `BAD_REQUEST` | `{"max_context_length": 8192, "provided_tokens": 8500}` |
| Content filtered | `BadRequest` | `BAD_REQUEST` | `{"filtered_reason": "violates_policy"}` |

### 12.9 Deadline Rules
- **Generation timeouts:** Deadline applies to entire generation process
- **Partial streams:** Stream may be terminated early if deadline exceeded
- **Token counting:** Fast token counting with minimal timeout impact

---

## PART III — VECTOR PROTOCOL (v1.0)

## 13. Vector Capabilities

### 13.1 Required Fields
```typescript
interface VectorCapabilities {
  protocol: string;               // "vector/v1.0"
  server: string;
  version: string;
  max_dimensions: number;
  supported_metrics: string[];    // e.g., ["cosine", "euclidean", "dotproduct"]
  supports_namespaces: boolean;
  supports_metadata_filtering: boolean;
  supports_batch_operations: boolean;
  max_batch_size?: number;
  supports_index_management: boolean;
  supports_deadline: boolean;
  idempotent_writes: boolean;
  supports_multi_tenant: boolean;
  max_top_k?: number;
  max_filter_terms?: number;
  supports_hybrid_search?: boolean; // Optional: hybrid vector+keyword search
  supports_sparse_vectors?: boolean; // Optional: sparse vector support
  max_namespaces?: number;        // Optional: namespace count limit
  supports_vector_compression?: boolean; // Optional: vector compression
}
```

### 13.2 Supported Metrics
- **cosine:** Cosine similarity (1 - cosine distance)
- **euclidean:** Euclidean distance (inverted for similarity)
- **dotproduct:** Dot product similarity

## 14. Vector Types

### 14.1 Vector
```typescript
interface Vector {
  id: string;
  vector: number[];
  metadata?: Metadata;
  namespace?: string;
}
```

### 14.2 VectorMatch
```typescript
interface VectorMatch {
  vector: Vector;
  score: number;                  // Similarity score (higher = more similar)
  distance: number;               // Raw distance metric (lower = more similar) - REQUIRED
}
```

### 14.3 QuerySpec
```typescript
interface QuerySpec {
  vector: number[];
  top_k: number;                  // Default: 10
  namespace?: string;             // Default: "default"
  filter?: Metadata;              // Metadata filter conditions
  include_metadata?: boolean;     // Default: true
  include_vectors?: boolean;      // Default: false
}
```

### 14.4 QueryResult
```typescript
interface QueryResult {
  matches: VectorMatch[];
  query_vector: number[];         // May be normalized
  namespace: string;
  total_matches: number;          // Total matches before top_k - REQUIRED
}
```

### 14.5 UpsertSpec
```typescript
interface UpsertSpec {
  vectors: Vector[];
  namespace?: string;
}
```

### 14.6 UpsertResult
```typescript
interface UpsertResult {
  upserted_count: number;
  failed_count: number;
  failures: Array<{
    id: string;
    error: string;
    detail: string;
  }>;
}
```

### 14.7 DeleteSpec
```typescript
interface DeleteSpec {
  ids: string[];
  namespace?: string;
  filter?: Metadata;              // Delete by metadata filter
}
```

### 14.8 DeleteResult
```typescript
interface DeleteResult {
  deleted_count: number;
  failed_count: number;
  failures: Array<{
    id: string;
    error: string;
    detail: string;
  }>;
}
```

### 14.9 NamespaceSpec
```typescript
interface NamespaceSpec {
  namespace: string;
  dimensions: number;
  distance_metric: string;        // e.g., "cosine", "euclidean"
}
```

### 14.10 NamespaceResult
```typescript
interface NamespaceResult {
  success: boolean;
  namespace: string;
  details: Metadata;
  vector_count?: number;          // Optional: count of vectors in namespace
  index_status?: string;          // Optional: index build status
  ready?: boolean;                // Optional: namespace readiness
}
```

## 15. Vector Operations

### 15.1 capabilities
**Purpose:** Discover supported vector features and limits

**Operation:** `vector.capabilities`

**Request Body:**
```json
{
  "op": "vector.capabilities",
  "ctx": {
    "request_id": "req-vector-cap-001",
    "tenant": "acme-corp"
  },
  "args": {}
}
```

**Output:** `VectorCapabilities`

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 1.8,
  "result": {
    "protocol": "vector/v1.0",
    "server": "pinecone-adapter",
    "version": "1.0.0",
    "max_dimensions": 2000,
    "supported_metrics": ["cosine", "euclidean", "dotproduct"],
    "supports_namespaces": true,
    "supports_metadata_filtering": true,
    "supports_batch_operations": true,
    "max_batch_size": 100,
    "supports_index_management": true,
    "supports_deadline": true,
    "idempotent_writes": true,
    "supports_multi_tenant": true,
    "max_top_k": 10000,
    "max_filter_terms": 10,
    "supports_hybrid_search": true,
    "supports_sparse_vectors": false,
    "max_namespaces": 1000,
    "supports_vector_compression": true
  }
}
```

### 15.2 query
**Purpose:** Find similar vectors using approximate nearest neighbor search

**Operation:** `vector.query`

**Input:** `QuerySpec`

**Validation:**
- `vector` MUST be non-empty list of numeric values
- `top_k` MUST be positive integer
- `namespace` MUST be non-empty string
- `filter`, if present, MUST be JSON-serializable mapping

**Capabilities Enforcement:**
- If `max_dimensions > 0` and `len(vector) > max_dimensions` → `DIMENSION_MISMATCH`
- If `max_top_k` is not None and `top_k > max_top_k` → `BAD_REQUEST`
- If `filter` is set but `supports_metadata_filtering` is False → `NOT_SUPPORTED`

**Request Body:**
```json
{
  "op": "vector.query",
  "ctx": {
    "request_id": "req-vector-query-001",
    "tenant": "acme-corp",
    "deadline_ms": 1736929400000
  },
  "args": {
    "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5],
    "top_k": 5,
    "namespace": "documents",
    "filter": {
      "category": "technology",
      "language": "en"
    },
    "include_metadata": true,
    "include_vectors": false
  }
}
```

**Output:** `QueryResult`

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 23.4,
  "result": {
    "matches": [
      {
        "vector": {
          "id": "doc-123",
          "vector": [0.12, 0.22, 0.32, 0.42, 0.52, 0.12, 0.22, 0.32, 0.42, 0.52],
          "metadata": {
            "title": "AI Research Paper",
            "category": "technology",
            "language": "en"
          },
          "namespace": "documents"
        },
        "score": 0.95,
        "distance": 0.05
      }
    ],
    "query_vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5],
    "namespace": "documents",
    "total_matches": 1
  }
}
```

### 15.3 upsert
**Purpose:** Insert or update vectors in a namespace

**Operation:** `vector.upsert`

**Input:** `UpsertSpec`

**Validation:**
- `namespace` MUST be non-empty
- `vectors` MUST be non-empty
- Each vector `id` MUST be non-empty string
- Each `vector` MUST be non-empty numeric list
- Each `metadata`, if present, MUST be JSON-serializable

**Capabilities Enforcement:**
- If `max_batch_size` is set and `len(vectors) > max_batch_size` → `BAD_REQUEST`
- If `max_dimensions` is set and any `len(v.vector) > max_dimensions` → `DIMENSION_MISMATCH`

**Request Body:**
```json
{
  "op": "vector.upsert",
  "ctx": {
    "request_id": "req-vector-upsert-001",
    "tenant": "acme-corp",
    "idempotency_key": "vector-import-20250115"
  },
  "args": {
    "vectors": [
      {
        "id": "doc-789",
        "vector": [0.15, 0.25, 0.35, 0.45, 0.55, 0.15, 0.25, 0.35, 0.45, 0.55],
        "metadata": {
          "title": "New Research on Neural Networks",
          "category": "science",
          "language": "en"
        },
        "namespace": "documents"
      }
    ],
    "namespace": "documents"
  }
}
```

**Output:** `UpsertResult`

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 15.8,
  "result": {
    "upserted_count": 1,
    "failed_count": 0,
    "failures": []
  }
}
```

### 15.4 delete
**Purpose:** Remove vectors by ID or metadata filter

**Operation:** `vector.delete`

**Input:** `DeleteSpec`

**Validation:**
- `namespace` MUST be non-empty
- MUST provide at least one of: non-empty `ids` or `filter`
- `filter`, if present, MUST be JSON-serializable
- Each `id` MUST be non-empty string

**Capabilities Enforcement:**
- If `max_batch_size` is set and `ids` non-empty and `len(ids) > max_batch_size` → `BAD_REQUEST`
- If `filter` set but `supports_metadata_filtering` is False → `NOT_SUPPORTED`

**Request Body:**
```json
{
  "op": "vector.delete",
  "ctx": {
    "request_id": "req-vector-delete-001",
    "tenant": "acme-corp"
  },
  "args": {
    "ids": ["doc-123"],
    "namespace": "documents"
  }
}
```

**Output:** `DeleteResult`

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 12.3,
  "result": {
    "deleted_count": 1,
    "failed_count": 0,
    "failures": []
  }
}
```

### 15.5 create_namespace
**Purpose:** Create a new vector namespace/collection

**Operation:** `vector.create_namespace`

**Input:** `NamespaceSpec`

**Validation:**
- `namespace` MUST be non-empty string
- `dimensions` MUST be positive integer
- `distance_metric` MUST be one of supported metrics

**Capabilities Enforcement:**
- If `max_dimensions` and `dimensions > max_dimensions` → `BAD_REQUEST`
- If `distance_metric` not in `supported_metrics` → `NOT_SUPPORTED`

**Request Body:**
```json
{
  "op": "vector.create_namespace",
  "ctx": {
    "request_id": "req-vector-create-ns-001",
    "tenant": "acme-corp"
  },
  "args": {
    "namespace": "images",
    "dimensions": 512,
    "distance_metric": "cosine"
  }
}
```

**Output:** `NamespaceResult`

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 45.2,
  "result": {
    "success": true,
    "namespace": "images",
    "details": {
      "index_status": "building",
      "estimated_completion_ms": 120000
    },
    "vector_count": 0,
    "index_status": "building",
    "ready": false
  }
}
```

### 15.6 delete_namespace
**Purpose:** Remove a vector namespace and all its vectors

**Operation:** `vector.delete_namespace`

**Input:** `{ namespace: string }`

**Validation:**
- `namespace` MUST be non-empty string

**Request Body:**
```json
{
  "op": "vector.delete_namespace",
  "ctx": {
    "request_id": "req-vector-delete-ns-001",
    "tenant": "acme-corp"
  },
  "args": {
    "namespace": "old-data"
  }
}
```

**Output:** `NamespaceResult`

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 28.7,
  "result": {
    "success": true,
    "namespace": "old-data",
    "details": {
      "vectors_deleted": 15000
    },
    "vector_count": 15000,
    "index_status": "deleted",
    "ready": false
  }
}
```

### 15.7 health
**Purpose:** Check vector store health and namespace status

**Operation:** `vector.health`

**Request Body:**
```json
{
  "op": "vector.health",
  "ctx": {
    "request_id": "req-vector-health-001",
    "tenant": "acme-corp"
  },
  "args": {}
}
```

**Output:**
```typescript
interface VectorHealthStatus {
  ok: boolean;
  server: string;
  version: string;
  status?: "ok" | "degraded";  // Optional status indicator
  namespaces: {
    [namespace: string]: {
      ready: boolean;
      vector_count: number;
      dimensions: number;
      index_status?: string;    // Optional index status
      read_only?: boolean;      // Optional read-only status
    };
  };
  total_vector_count?: number;  // Optional total vectors across all namespaces
  message?: string;             // Optional status message
}
```

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 3.8,
  "result": {
    "ok": true,
    "server": "pinecone-adapter",
    "version": "1.0.0",
    "status": "ok",
    "namespaces": {
      "documents": {
        "ready": true,
        "vector_count": 15000,
        "dimensions": 1536,
        "index_status": "ready",
        "read_only": false
      },
      "images": {
        "ready": false,
        "vector_count": 0,
        "dimensions": 512,
        "index_status": "building",
        "read_only": false
      }
    },
    "total_vector_count": 15000,
    "message": "Vector store operational"
  }
}
```

## 16. Vector Semantics (Normative)

### 16.1 Dimension Enforcement
- **Strict matching:** All vectors in namespace MUST have same dimensions
- **Query validation:** Query vectors MUST match namespace dimensions
- **Error reporting:** Clear dimension mismatch errors with expected/actual values

### 16.2 Filter Semantics
- **Equality filters:** `{ field: value }` for exact matching
- **Set membership:** `{ field: [value1, value2] }` for IN queries
- **Range queries:** `{ field: { gt: value } }` for greater than
- **Combination:** Multiple conditions combined with AND

### 16.3 Metrics & Scoring Rules
- **Normalization:** Scores normalized to [0, 1] range where possible
- **Consistency:** Same metric always produces same score range
- **Documentation:** Clear explanation of score meaning for each metric
- **Cosine normalization:** For cosine similarity, vectors SHOULD be L2-normalized

### 16.4 Search Behavior
- **Approximate vs exact:** Adapters SHOULD document search accuracy guarantees
- **Result ordering:** Matches ordered by descending score (highest similarity first)

### 16.5 Batch Semantics
- **Partial success:** Batch operations continue despite individual failures
- **Order preservation:** Results maintain input order where applicable
- **Size limits:** Batch size constrained by provider capabilities
- **Empty batch validation:** Reject empty batches with `BAD_REQUEST`

### 16.6 Error Mappings
| Provider Error | Normalized Error | Wire Code | Details |
|----------------|------------------|-----------|---------|
| Dimension mismatch | `DimensionMismatch` | `DIMENSION_MISMATCH` | `{"provided": 1537, "expected": 1536}` |
| Namespace not found | `NamespaceNotFound` | `NAMESPACE_NOT_FOUND` | `{"namespace": "unknown"}` |
| Index not ready | `IndexNotReady` | `INDEX_NOT_READY` | `{"namespace": "building", "estimated_ms": 120000}` |
| Batch size exceeded | `BadRequest` | `BAD_REQUEST` | `{"max_batch_size": 100, "provided": 150}` |

### 16.7 Deadline Rules
- **Query timeouts:** Deadline applies to entire search operation
- **Batch operations:** Deadline applies to entire batch processing
- **Namespace operations:** Deadline applies to management operations

---

## PART IV — EMBEDDING PROTOCOL (v1.0)

## 17. Embedding Capabilities

### 17.1 Required Fields
```typescript
interface EmbeddingCapabilities {
  protocol: string;               // "embedding/v1.0"
  server: string;
  version: string;
  supported_models: string[];
  max_batch_size?: number;
  max_text_length?: number;
  max_dimensions?: number;
  supports_normalization: boolean;
  supports_truncation: boolean;
  supports_token_counting: boolean;
  supports_multi_tenant: boolean;
  supports_deadline: boolean;     // ADDED MISSING FIELD
  normalizes_at_source: boolean;
  idempotent_operations: boolean;
  truncation_mode: string;        // "base" or "adapter"
  max_batch_tokens?: number;      // Optional: token-based batch limits
  supports_multilingual?: boolean; // Optional: multilingual support
  supports_custom_dimensions?: boolean; // Optional: dimension customization
}
```

### 17.2 Model Support
- **Model listing:** Available embedding models with dimensions
- **Batch capabilities:** Maximum batch sizes per model
- **Text limits:** Maximum input lengths per model

## 18. Embedding Types

### 18.1 EmbedSpec
```typescript
interface EmbedSpec {
  text: string;
  model: string;
  truncate?: boolean;             // Allow automatic truncation
  normalize?: boolean;            // Return normalized vectors
}
```

### 18.2 EmbedBatchSpec
```typescript
interface EmbedBatchSpec {
  texts: string[];
  model: string;
  truncate?: boolean;
  normalize?: boolean;
}
```

### 18.3 EmbedResult
```typescript
interface EmbedResult {
  embedding: EmbeddingVector;
  model: string;
  text: string;                   // Possibly truncated
  tokens_used?: number;
  truncated?: boolean;            // True if text was truncated
}

interface EmbeddingVector {
  vector: number[];
  text: string;
  model: string;
  dimensions: number;
}
```

### 18.4 EmbedBatchResult
```typescript
interface EmbedBatchResult {
  embeddings: EmbeddingVector[];
  model: string;
  total_texts: number;
  total_tokens?: number;
  failed_texts: Array<{
    index: number;                // Index in original batch
    text: string;
    error: string;
    message: string;
  }>;
}
```

## 19. Embedding Operations

### 19.1 capabilities
**Purpose:** Discover supported embedding features and models

**Operation:** `embedding.capabilities`

**Request Body:**
```json
{
  "op": "embedding.capabilities",
  "ctx": {
    "request_id": "req-embed-cap-001",
    "tenant": "acme-corp"
  },
  "args": {}
}
```

**Output:** `EmbeddingCapabilities`

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 1.3,
  "result": {
    "protocol": "embedding/v1.0",
    "server": "openai-embedding-adapter",
    "version": "1.0.0",
    "supported_models": ["text-embedding-3-large", "text-embedding-3-small"],
    "max_batch_size": 2048,
    "max_text_length": 8192,
    "max_dimensions": 3072,
    "supports_normalization": true,
    "supports_truncation": true,
    "supports_token_counting": true,
    "supports_multi_tenant": true,
    "supports_deadline": true,
    "normalizes_at_source": true,
    "idempotent_operations": true,
    "truncation_mode": "base",
    "max_batch_tokens": 8000000,
    "supports_multilingual": true,
    "supports_custom_dimensions": true
  }
}
```

### 19.2 embed
**Purpose:** Generate embedding vector for a single text

**Operation:** `embedding.embed`

**Input:** `EmbedSpec`

**Validation:**
- `text` MUST be non-empty string
- `model` MUST be non-empty string

**Request Body:**
```json
{
  "op": "embedding.embed",
  "ctx": {
    "request_id": "req-embed-single-001",
    "tenant": "acme-corp",
    "deadline_ms": 1736929500000
  },
  "args": {
    "text": "The quick brown fox jumps over the lazy dog",
    "model": "text-embedding-3-large",
    "truncate": true,
    "normalize": true
  }
}
```

**Output:** `EmbedResult`

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 45.2,
  "result": {
    "embedding": {
      "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5],
      "text": "The quick brown fox jumps over the lazy dog",
      "model": "text-embedding-3-large",
      "dimensions": 3072
    },
    "model": "text-embedding-3-large",
    "text": "The quick brown fox jumps over the lazy dog",
    "tokens_used": 9,
    "truncated": false
  }
}
```

### 19.3 embed_batch
**Purpose:** Generate embeddings for multiple texts in batch

**Operation:** `embedding.embed_batch`

**Input:** `EmbedBatchSpec`

**Validation:**
- `model` MUST be non-empty string
- `texts` MUST be non-empty list

**Request Body:**
```json
{
  "op": "embedding.embed_batch",
  "ctx": {
    "request_id": "req-embed-batch-001",
    "tenant": "acme-corp"
  },
  "args": {
    "texts": [
      "Machine learning is a subset of artificial intelligence.",
      "Deep learning uses neural networks with multiple layers.",
      "Natural language processing helps computers understand human language."
    ],
    "model": "text-embedding-3-large",
    "truncate": true,
    "normalize": false
  }
}
```

**Output:** `EmbedBatchResult`

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 89.7,
  "result": {
    "embeddings": [
      {
        "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5],
        "text": "Machine learning is a subset of artificial intelligence.",
        "model": "text-embedding-3-large",
        "dimensions": 3072
      },
      {
        "vector": [0.15, 0.25, 0.35, 0.45, 0.55, 0.15, 0.25, 0.35, 0.45, 0.55],
        "text": "Deep learning uses neural networks with multiple layers.",
        "model": "text-embedding-3-large",
        "dimensions": 3072
      },
      {
        "vector": [0.12, 0.22, 0.32, 0.42, 0.52, 0.12, 0.22, 0.32, 0.42, 0.52],
        "text": "Natural language processing helps computers understand human language.",
        "model": "text-embedding-3-large",
        "dimensions": 3072
      }
    ],
    "model": "text-embedding-3-large",
    "total_texts": 3,
    "total_tokens": 42,
    "failed_texts": []
  }
}
```

### 19.4 count_tokens
**Purpose:** Count tokens in text for embedding model

**Operation:** `embedding.count_tokens`

**Input:** 
```typescript
interface CountTokensSpec {
  text: string;
  model: string;
}
```

**Request Body:**
```json
{
  "op": "embedding.count_tokens",
  "ctx": {
    "request_id": "req-embed-tokens-001",
    "tenant": "acme-corp"
  },
  "args": {
    "text": "The quick brown fox jumps over the lazy dog",
    "model": "text-embedding-3-large"
  }
}
```

**Output:** `number` (bare integer)

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 2.8,
  "result": 9
}
```

### 19.5 health
**Purpose:** Check embedding provider health and model status

**Operation:** `embedding.health`

**Request Body:**
```json
{
  "op": "embedding.health",
  "ctx": {
    "request_id": "req-embed-health-001",
    "tenant": "acme-corp"
  },
  "args": {}
}
```

**Output:**
```typescript
interface EmbeddingHealthStatus {
  ok: boolean;
  server: string;
  version: string;
  status?: "ok" | "degraded";  // Optional status indicator
  models?: {                   // Optional model status information
    [model: string]: {
      available: boolean;
      rate_limited?: boolean;
      degraded?: boolean;
    };
  };
  rate_limits?: {             // Optional rate limit information
    tokens_per_minute?: number;
    requests_per_minute?: number;
  };
  message?: string;           // Optional status message
}
```

**Response Body:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 4.2,
  "result": {
    "ok": true,
    "server": "openai-embedding-adapter",
    "version": "1.0.0",
    "status": "ok",
    "models": {
      "text-embedding-3-large": {
        "available": true,
        "rate_limited": false,
        "degraded": false
      },
      "text-embedding-3-small": {
        "available": true,
        "rate_limited": false,
        "degraded": false
      }
    },
    "rate_limits": {
      "tokens_per_minute": 1000000,
      "requests_per_minute": 5000
    },
    "message": "All embedding models operational"
  }
}
```

## 20. Embedding Semantics (Normative)

### 20.1 Truncation Rules
- **Explicit consent:** Truncation only when `truncate=true`
- **Clear indication:** `truncated` flag indicates when truncation occurred
- **Consistent behavior:** Same truncation method for single and batch
- **Mode behavior:** When `truncation_mode` specified, follow provider semantics

### 20.2 Normalization Rules
- **L2 normalization:** When `normalize=true`, vectors MUST be L2-normalized
- **Source normalization:** When `normalizes_at_source=true`, provider handles normalization
- **Determinism:** Same text + normalization settings → identical vector

### 20.3 Dimension Consistency
- **Model adherence:** Embedding dimensions MUST match model specifications
- **Batch consistency:** All embeddings in batch have same dimensions
- **Error on mismatch:** Return error if dimensions don't match expectations

### 20.4 Empty Input Handling
- **Zero-length texts:** Reject empty inputs with `BAD_REQUEST` error
- **Whitespace-only:** Treat as normal text, not empty
- **Error preference:** Prefer error over zero vector for empty inputs

### 20.5 Batch Semantics
- **Order preservation:** Embeddings maintain input text order
- **Partial success:** Continue processing despite individual failures
- **Fallback behavior:** Automatic fallback to single embedding if batch unsupported

### 20.6 Determinism
- **Same inputs:** Identical text + model → identical embedding vector
- **Batch consistency:** Batch embedding identical to individual embeddings
- **Normalization:** Consistent normalization when requested

### 20.7 Error Mappings
| Provider Error | Normalized Error | Wire Code | Details |
|----------------|------------------|-----------|---------|
| Text too long | `TextTooLong` | `TEXT_TOO_LONG` | `{"max_text_length": 8192, "provided_length": 8500}` |
| Model not available | `ModelNotAvailable` | `MODEL_NOT_AVAILABLE` | `{"requested_model": "unknown-model"}` |
| Normalization not supported | `NotSupported` | `NOT_SUPPORTED` | `{"feature": "normalization"}` |

### 20.8 Deadline Rules
- **Batch timeouts:** Deadline applies to entire batch processing
- **Token counting:** Fast operation with minimal timeout impact
- **Provider limits:** Respect provider-specific timeout constraints

---

## PART V — CROSS-PROTOCOL SECTIONS

## 21. Observability Integration

### 21.1 Mapping to METRICS.md
- **Operation metrics:** All operations emit `observe_operation` and `count_operation`
- **Stream metrics:** Streaming operations emit `count_stream_final_outcome`
- **Error metrics:** Errors tracked via `code` label using normalized error types

### 21.2 Required Per-Op Emission Rules
- **One observation per operation:** Latency and outcome recording
- **Consistent labeling:** Same labels for observe and count metrics
- **Deadline buckets:** Categorized timeout budgets per METRICS.md §3

### 21.3 Deadline Bucket Consistency
All adapters MUST use the same deadline bucket categorization:
- `"<1s"`, `"<5s"`, `"<15s"`, `"<60s"`, `">=60s"`
- Consistent across all protocols and operations

### 21.4 Streaming Final Outcome Rules
- **Exactly one terminal metric:** Per stream regardless of chunk count
- **Error classification:** Final error determines metrics code
- **Duration measurement:** Total stream duration from start to terminal event

## 22. Normalized Error Integration

### 22.1 Canonical Error Mapping
All adapters MUST map provider errors to the canonical taxonomy defined in **ERRORS.md**:
- **Base errors:** `BadRequest`, `AuthError`, `ResourceExhausted`, etc.
- **Protocol subtypes:** Protocol-specific error refinements
- **Consistent retryability:** Uniform retry semantics across protocols

### 22.2 Per-Protocol Subtype Usage
- **LLM:** `ModelOverloaded`, `PromptTooLong`, `ContentFiltered`
- **Embedding:** `TextTooLong`, `ModelNotAvailable`
- **Vector:** `DimensionMismatch`, `IndexNotReady`, `NamespaceNotFound`
- **Graph:** `QueryParseError`, `VertexNotFound`, `SchemaValidationError`

### 22.3 Partial Failure Envelopes
All batch operations use consistent partial failure reporting:
```typescript
{
  processed_count: number;
  failed_count: number;
  failures: Array<{
    id?: string;           // Item identifier when available
    index?: number;        // Batch position when no ID
    error: string;         // Normalized error type
    detail: string;        // Human-readable details
  }>;
}
```

### 22.4 Retry Semantics
- **Automatic retry:** `TRANSIENT_NETWORK`, `UNAVAILABLE`, `RESOURCE_EXHAUSTED`
- **Conditional retry:** `DEADLINE_EXCEEDED` only with increased deadline
- **No retry:** `BAD_REQUEST`, `AUTH_ERROR`, `NOT_SUPPORTED`

## 23. Testing & Conformance

### 23.1 Adapter Conformance Suite
- **Schema validation:** All request/response shapes validated
- **Behavioral tests:** Protocol semantics and edge cases
- **Error mapping:** Provider error → canonical error validation
- **Metrics emission:** Observability compliance checking

### 23.2 Required Mock Behaviors
- **Deterministic responses:** Same inputs → same outputs
- **Error injection:** Configurable error scenarios
- **Performance simulation:** Realistic latency and throughput
- **Partial failure:** Batch operation failure scenarios

### 23.3 Stream Determinism Tests
- **Chunk equivalence:** Stream chunks concatenate to match single response
- **Terminal events:** Exactly one terminal event per stream
- **Error propagation:** Stream errors match non-stream errors
- **Backpressure handling:** Respects client consumption rate

### 23.4 Partial-Failure Tests
- **Batch continuation:** Continues processing after individual failures
- **Error reporting:** Accurate failure counts and details
- **Order preservation:** Maintains input order in results
- **Atomic batches:** All-or-nothing behavior when atomic=true

### 23.5 Cross-Protocol Validation
- **Consistent context handling:** OperationContext usage across protocols
- **Uniform error handling:** Same error patterns and retry behavior
- **Standard metrics:** Consistent observability across all operations
- **Security compliance:** SIEM-safe operation across protocols

## 24. Security Requirements

### 24.1 SIEM-Safe Rules
- **No content logging:** Prompts, vectors, embeddings excluded from logs
- **Structured errors:** Error messages contain no sensitive data
- **Tenant hashing:** Tenant identifiers hashed in telemetry
- **Cardinality control:** Bounded label sets in metrics

### 24.2 Required Redaction List
**MUST NOT log or include in telemetry:**
- Raw prompt text or message content
- Vector values or embedding coordinates  
- Tenant identifiers (use hashes only)
- API keys, tokens, or credentials
- Personal identifying information (PII)

**MUST hash before inclusion:**
- Tenant IDs (SHA256, first 12 chars)
- Document IDs (when high cardinality)
- User IDs (when high cardinality)

### 24.3 PII & Content Handling
- **Input sanitization:** No PII in error messages or details
- **Content filtering:** Provider content policies respected and enforced
- **Data minimization:** Only necessary data processed and stored

### 24.4 Logging Constraints
- **Structured logging:** JSON format with bounded field sets
- **Sensitive field exclusion:** No credentials, tokens, or raw content
- **Audit trails:** Sufficient context for debugging without content exposure

### 24.5 Transport Security Requirements
- **Encryption in transit:** TLS 1.2+ for all external communications
- **Authentication:** Secure credential management and rotation
- **Authorization:** Tenant isolation and access control enforcement

## 25. Versioning, Deprecation & Evolution

### 25.1 protocols_version Rules
- **Major version:** Breaking changes to protocol semantics
- **Minor version:** Additive changes with backward compatibility
- **Patch version:** Bug fixes and non-breaking improvements

### 25.2 Backwards Compatibility Guarantees
- **Stable envelopes:** Wire format compatibility within major version
- **Additive changes:** New fields optional, existing fields unchanged
- **Behavior preservation:** Existing operation semantics unchanged

### 25.3 Deprecation Approach
- **Advance notice:** Features deprecated for ≥1 minor release before removal
- **Gradual migration:** Clear migration paths and alternatives
- **Tooling support:** Validation tools flag deprecated usage

### 25.4 Cross-Document Version Alignment
- **METRICS.md:** metrics_version aligned with protocol capabilities
- **ERRORS.md:** errors_version aligned with error taxonomy
- **IMPLEMENTATION.md:** Updated with protocol changes

## 26. Compliance Matrices

### 26.1 Global Compliance Checklist
- [ ] All operations emit required metrics
- [ ] All errors mapped to canonical taxonomy
- [ ] Tenant hashing applied in all telemetry
- [ ] Deadline propagation with safety buffer
- [ ] SIEM-safe requirements enforced
- [ ] Capabilities reported truthfully including protocol field
- [ ] Wire envelopes follow standardization
- [ ] Streaming semantics followed

### 26.2 Protocol-Specific Compliance
**Graph:**
- [ ] Query dialect negotiation supported
- [ ] Parameter binding implemented
- [ ] Streaming cardinality bounds respected
- [ ] Batch partial failure reporting
- [ ] Validation rules enforced (non-empty nodes/edges, etc.)

**LLM:**
- [ ] Stop sequence precedence followed
- [ ] JSON mode strictness enforced
- [ ] Tool call determinism maintained
- [ ] Stream equivalence guaranteed
- [ ] Parameter range validation enforced
- [ ] Context window preflight implemented

**Vector:**
- [ ] Dimension enforcement strict
- [ ] Filter semantics consistent
- [ ] Metric scoring normalized
- [ ] Batch size limits enforced
- [ ] Namespace validation implemented

**Embedding:**
- [ ] Truncation rules followed
- [ ] Normalization consistent
- [ ] Dimension matching enforced
- [ ] Empty input handling defined
- [ ] Batch fallback behavior implemented

## 27. Cross-Protocol Standardization Tables

### 27.1 Health Check Standardization
| Field | All Protocols | Graph | LLM | Vector | Embedding |
|-------|---------------|-------|-----|--------|-----------|
| `ok` | REQUIRED | ✓ | ✓ | ✓ | ✓ |
| `server` | REQUIRED | ✓ | ✓ | ✓ | ✓ |
| `version` | REQUIRED | ✓ | ✓ | ✓ | ✓ |
| `status` | OPTIONAL | ✓ | ✓ | ✓ | ✓ |
| `namespaces` | PROTOCOL | ✓ | ✗ | ✓ | ✗ |
| `models` | PROTOCOL | ✗ | ✓ | ✗ | ✓ |
| `read_only` | OPTIONAL | ✓ | ✗ | ✗ | ✗ |
| `degraded` | OPTIONAL | ✓ | ✗ | ✗ | ✗ |
| `message` | OPTIONAL | ✓ | ✓ | ✓ | ✓ |

### 27.2 Batch Operation Standardization
| Field | All Protocols | Purpose |
|-------|---------------|---------|
| `processed_count` | REQUIRED | Successfully processed items |
| `failed_count` | REQUIRED | Failed items count |
| `failures[]` | REQUIRED | Detailed failure information |
| `results[]` | OPTIONAL | Successful operation results |
| Atomic support | PROTOCOL | Protocol-specific availability |

### 27.3 Streaming Standardization
| Aspect | All Protocols | Requirements |
|--------|---------------|--------------|
| Terminal events | REQUIRED | Exactly one per stream |
| Error delivery | REQUIRED | Single error event terminates |
| Order preservation | REQUIRED | Maintains semantic order |
| Backpressure | RECOMMENDED | Respect client consumption |
| Heartbeats | OPTIONAL | Keep-alive messages |

### 27.4 Error Code Standardization
| Error Code | Retryable | Protocols | Typical Cause |
|------------|-----------|-----------|---------------|
| `BAD_REQUEST` | No | All | Invalid parameters, malformed requests |
| `AUTH_ERROR` | No | All | Invalid credentials, permissions |
| `RESOURCE_EXHAUSTED` | Yes* | All | Rate limits, quotas exceeded |
| `TRANSIENT_NETWORK` | Yes | All | Network timeouts, connection issues |
| `UNAVAILABLE` | Yes | All | Service temporarily down |
| `NOT_SUPPORTED` | No | All | Unsupported operation or parameter |
| `DEADLINE_EXCEEDED` | Conditional | All | Operation timeout |
| `MODEL_OVERLOADED` | Yes | LLM | Model capacity exceeded |
| `TEXT_TOO_LONG` | No | LLM, Embedding | Input exceeds context window |
| `DIMENSION_MISMATCH` | No | Vector | Vector dimensions don't match |
| `QUERY_PARSE_ERROR` | No | Graph | Invalid query syntax |

*\*Retry after suggested delay*

## 28. Glossary

### 28.1 Common Terminology
- **Adapter:** Protocol implementation bridging Corpus APIs to providers
- **Provider:** Underlying AI/data service (OpenAI, Pinecone, Neo4j, etc.)
- **Tenant:** Logical isolation boundary for multi-tenant deployments
- **Namespace:** Protocol-specific isolation scope (collections, graphs, etc.)
- **OperationContext:** Request-scoped context for deadlines, tracing, etc.
- **Thin mode:** Adapter composition under external control plane (no-op policies)
- **Standalone mode:** Direct use with built-in resiliency patterns
- **Wire envelope:** Canonical JSON format for all protocol communications
- **SIEM-safe:** Telemetry design that prevents sensitive data exposure

### 28.2 Protocol-Specific Terms
- **Graph:** Property graph with nodes, edges, and graph queries
- **LLM:** Large language model for text generation and completion
- **Vector:** High-dimensional vectors for similarity search
- **Embedding:** Text-to-vector transformation models
- **Streaming:** Incremental response delivery for long-running operations
- **Batch operation:** Multiple items processed in single request
- **Partial failure:** Batch operations where some items succeed, others fail
- **Capabilities:** Dynamic feature discovery and limits reporting

### 28.3 Error & Observability Terms
- **Normalized error:** Provider errors mapped to canonical taxonomy
- **Retry-after:** Suggested delay before retrying exhausted operations
- **Tenant hash:** Privacy-preserving tenant identifier for telemetry
- **Deadline bucket:** Categorized timeout budget for metrics
- **Cardinality control:** Limiting unique values in metric labels

## 29. Appendix

### 29.1 Example Capability Responses
```json
{
  "LLM": {
    "protocol": "llm/v1.0",
    "server": "openai-adapter",
    "version": "1.0.0",
    "model_family": "openai",
    "supported_models": ["gpt-4", "gpt-3.5-turbo"],
    "max_context_length": 8192,
    "supports_streaming": true,
    "supports_count_tokens": true
  },
  "Vector": {
    "protocol": "vector/v1.0",
    "server": "pinecone-adapter", 
    "version": "1.0.0",
    "max_dimensions": 2000,
    "supported_metrics": ["cosine", "euclidean"],
    "supports_namespaces": true,
    "max_batch_size": 100
  }
}
```

### 29.2 Example Error Envelopes
```json
{
  "ResourceExhausted (retryable)": {
    "ok": false,
    "error": "ResourceExhausted",
    "message": "Rate limit exceeded for model gpt-4",
    "code": "RESOURCE_EXHAUSTED",
    "retry_after_ms": 5000,
    "resource_scope": "model",
    "ms": 12.3
  },
  "BadRequest (non-retryable)": {
    "ok": false,
    "error": "BadRequest", 
    "message": "Invalid parameter: temperature must be between 0.0 and 2.0",
    "code": "BAD_REQUEST",
    "retry_after_ms": null,
    "details": {"parameter": "temperature", "min": 0.0, "max": 2.0},
    "ms": 3.2
  }
}
```

### 29.3 Example Batch Responses
```json
{
  "Vector Upsert (partial success)": {
    "upserted_count": 8,
    "failed_count": 2,
    "failures": [
      {
        "id": "doc-17",
        "error": "DimensionMismatch",
        "detail": "expected 1536, got 1537"
      },
      {
        "id": "doc-23", 
        "error": "BadRequest",
        "detail": "metadata not JSON-serializable"
      }
    ]
  },
  "Graph Batch (all success)": {
    "processed_count": 3,
    "failed_count": 0,
    "failures": [],
    "results": [
      {"deleted_count": 1},
      {"upserted_count": 2}, 
      {"processed_count": 1}
    ]
  }
}
```

### 29.4 Example Streaming Traces
**Graph Stream Frames:**
```json
{"ok": true, "code": "OK", "ms": 12.3, "chunk": {"records": [{"user": {"id": "u1", "name": "Alice"}}], "is_final": false}}
{"ok": true, "code": "OK", "ms": 15.7, "chunk": {"records": [{"user": {"id": "u2", "name": "Bob"}}], "is_final": false}}
{"ok": true, "code": "OK", "ms": 18.2, "chunk": {"summary": {"query_time_ms": 125.4, "results_count": 2, "has_more": false}, "is_final": true}}
```

**LLM Stream Frames:**
```json
{"ok": true, "code": "OK", "ms": 12.3, "chunk": {"text": "Hello", "is_final": false, "model": "gpt-4"}}
{"ok": true, "code": "OK", "ms": 15.7, "chunk": {"text": " world", "is_final": false, "model": "gpt-4"}}
{"ok": true, "code": "OK", "ms": 18.2, "chunk": {"text": "!", "is_final": true, "model": "gpt-4", "usage_so_far": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}}
```

### 29.5 Context Deadline Examples
```json
{
  "Valid deadline (future)": {
    "deadline_ms": 1736929200000,  // 2025-01-15T10:20:00Z
    "remaining_ms": 45000          // 45 seconds remaining
  },
  "Expired deadline": {
    "deadline_ms": 1736929155000,  // 2025-01-15T10:19:15Z  
    "remaining_ms": 0              // Immediately expired
  },
  "No deadline": {
    "deadline_ms": null,
    "remaining_ms": null
  }
}
```

### 29.6 Mode Configuration Examples
```python
# Thin mode (external control plane)
adapter = BaseLLMAdapter(mode="thin")
# Result: NoopDeadline, NoopBreaker, NoopCache, NoopLimiter

# Standalone mode (direct use)  
adapter = BaseLLMAdapter(mode="standalone")
# Result: EnforcingDeadline, SimpleCircuitBreaker, InMemoryTTLCache, TokenBucketLimiter
```

### 29.7 JSON Schema References
- **Strict validation:** All schemas prohibit additional properties unless specified
- **Required fields:** Clearly marked in type definitions
- **Default values:** Specified where semantically meaningful
- **Range validation:** Numeric bounds enforced per capabilities
- **Serialization:** All data MUST be JSON-serializable for caching and wire transfer

## 30. Index

### 30.1 Operations Index
- **Graph Operations:** capabilities, upsert_nodes, upsert_edges, delete_nodes, delete_edges, query, stream_query, bulk_vertices, batch, get_schema, health
- **LLM Operations:** capabilities, complete, stream, count_tokens, health  
- **Vector Operations:** capabilities, query, upsert, delete, create_namespace, delete_namespace, health
- **Embedding Operations:** capabilities, embed, embed_batch, count_tokens, health

### 30.2 Types Index
- **Common Types:** OperationContext, Metadata, BatchResult, TokenUsage
- **Graph Types:** Node, Edge, QuerySpec, QueryResult, QueryChunk, GraphSchema, GraphHealthStatus
- **LLM Types:** Message, ToolCall, CompletionSpec, LLMCompletion, LLMChunk, LLMHealthStatus
- **Vector Types:** Vector, VectorMatch, QuerySpec, QueryResult, UpsertSpec, UpsertResult, DeleteSpec, DeleteResult, NamespaceSpec, NamespaceResult, VectorHealthStatus
- **Embedding Types:** EmbedSpec, EmbedBatchSpec, EmbedResult, EmbeddingVector, EmbedBatchResult, EmbeddingHealthStatus

### 30.3 Capabilities Index
- **Graph Capabilities:** protocol, server, version, supported_query_dialects, supports_stream_query, supports_bulk_vertices, supports_batch, supports_schema, idempotent_writes, supports_deadline, supports_namespaces, supports_property_filters, supports_multi_tenant, max_batch_ops, max_query_complexity, max_nodes_per_transaction, supports_analytics_queries
- **LLM Capabilities:** protocol, server, version, model_family, supported_models, max_context_length, supports_streaming, supports_roles, supports_system_message, supports_json_output, supports_parallel_tool_calls, supports_deadline, supports_count_tokens, idempotent_writes, supports_multi_tenant, max_tokens_per_minute, max_requests_per_minute, supports_function_calling, supports_vision, supports_logprobs
- **Vector Capabilities:** protocol, server, version, max_dimensions, supported_metrics, supports_namespaces, supports_metadata_filtering, supports_batch_operations, max_batch_size, supports_index_management, supports_deadline, idempotent_writes, supports_multi_tenant, max_top_k, max_filter_terms, supports_hybrid_search, supports_sparse_vectors, max_namespaces, supports_vector_compression
- **Embedding Capabilities:** protocol, server, version, supported_models, max_batch_size, max_text_length, max_dimensions, supports_normalization, supports_truncation, supports_token_counting, supports_multi_tenant, supports_deadline, normalizes_at_source, idempotent_operations, truncation_mode, max_batch_tokens, supports_multilingual, supports_custom_dimensions

### 30.4 Error Codes Index
- BAD_REQUEST, AUTH_ERROR, RESOURCE_EXHAUSTED, TRANSIENT_NETWORK, UNAVAILABLE, NOT_SUPPORTED, DEADLINE_EXCEEDED, MODEL_OVERLOADED, TEXT_TOO_LONG, DIMENSION_MISMATCH, QUERY_PARSE_ERROR

### 30.5 Cross-Protocol Concepts
- Tenant Isolation, Deadline Propagation, Streaming Semantics, Batch Operations, Partial Failures, Health Checking, Capability Discovery, Error Normalization, Observability, Security Requirements

---

*End of PROTOCOLS.md (protocols_version 1.0)*
```
