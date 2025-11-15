# PROTOCOLS

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

---

**Corpus Protocol Suite — Unified Specification for Graph, LLM, Vector, and Embedding Adapters**  
**protocols_version:** `1.0`

> This document defines the unified protocol specification for all Corpus adapters. It establishes the normative behavior, types, and semantics for Graph, LLM, Vector, and Embedding protocols while maintaining cross-protocol consistency.

## 0. Document Metadata

### 0.1 protocols_version: "1.0"
- **Status:** Stable / Normative
- **Effective Date:** 2025-01-01
- **Replaces:** None (initial version)

### 0.2 Relationship to Companion Documents
- **SPECIFICATION.md:** Defines high-level architecture, design philosophy, and normative requirements referenced throughout this document
- **METRICS.md:** Defines observability requirements referenced in §21
- **ERRORS.md:** Defines error taxonomy referenced in §22
- **IMPLEMENTATION.md:** Provides implementation guidance and patterns

### 0.3 Intended Audience
- Adapter developers implementing Corpus protocol support
- Platform engineers building routing and orchestration systems
- SRE/Operations teams managing production deployments
- Client library developers consuming adapter services

### 0.4 Non-Goals
- Provider-specific implementation details
- Transport-layer specifics (HTTP/gRPC/WebSocket wire formats)
- Business logic or application-level semantics
- UI/UX considerations

### 0.5 Terminology & Conventions
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
  attrs: Map<string, any>;       // Opaque extension attributes
}
```

### 2.2 Tenant Isolation & Hashing
- **Tenant MUST NOT** appear in raw form in metrics, logs, or error messages
- **Tenant hash:** First 12 characters of `SHA256(tenant)` for telemetry
- **Isolation:** Adapters enforce tenant boundaries in provider API calls

### 2.3 Deadlines & Budgets
- **Propagation:** Adapters MUST propagate `deadline_ms` to provider APIs
- **Safety buffer:** Subtract 50-100ms from remaining time for network overhead
- **Expiration:** Reject operations where context deadline has expired

### 2.4 Stream Semantics
- **Single terminal:** Exactly one terminal event (success or error)
- **No content after terminal:** Stream MUST end after final event
- **Heartbeats:** Optional keep-alive messages allowed but not required
- **Backpressure:** Clients control consumption rate; adapters MUST implement bounded buffering and flow control

### 2.5 Thread Safety & In-Memory Infrastructure
- **Adapter instances are not thread-safe** by default
- **In-memory implementations are not distributed** - state is local to process
- **Concurrent access** requires external synchronization by callers
- **Standalone mode** assumes single-process deployment constraints

### 2.6 Configuration Management
- **Thin mode:** Adapters rely on external configuration for policies, limits, and routing
- **Standalone mode:** Limited built-in configuration for development/testing
- **Policy configuration** (rate limits, quotas, etc.) is outside wire protocol scope
- **Adapter initialization** accepts configuration but runtime changes require restart

### 2.7 Cache Serialization Constraints
- **All cached data MUST be JSON-serializable**
- **Metadata maps** MUST contain only JSON-serializable values
- **Filter expressions** MUST be serializable for cache key composition
- **Vector data** may require custom serialization for performance

### 2.8 Cache Key Composition
- **Cache keys MUST include:** tenant, namespace, operation, and critical parameters
- **Filter expressions** MUST be normalized for consistent key generation
- **Vector queries** SHOULD use fingerprinting for high-dimensional data
- **Collision risk:** Adapters MUST ensure key uniqueness across tenants and namespaces

### 2.9 Normalized Error Model
See **ERRORS.md** for complete error taxonomy and mapping requirements.

### 2.10 Observability & Metrics
See **METRICS.md** for complete metrics specification and emission rules.

### 2.11 Capability Probing
- **Dynamic discovery:** Clients probe `capabilities()` to determine supported features
- **Truthful reporting:** Adapters MUST accurately report actual capabilities
- **Caching:** Capabilities may be cached with appropriate TTL (typically 5-60 minutes)

### 2.12 SIEM-Safe Requirements
- **No raw content** in logs, metrics, or errors (prompts, vectors, embeddings)
- **Structured data only** in telemetry fields
- **Tenant hashing** as described in §2.2
- **Cardinality control** in labels and dimensions

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
- Emit `observe_operation` and `count_operation` for every operation
- Use tenant hashing in all telemetry contexts
- Propagate `deadline_ms` to provider APIs with safety buffer
- Map provider errors to canonical error taxonomy
- Report capabilities truthfully and accurately
- Enforce SIEM-safe requirements for all telemetry

**All adapters MUST NOT:**
- Log raw prompts, vectors, embeddings, or tenant IDs
- Exceed provider batch size limits without client consent
- Return unnormalized errors with provider-specific details
- Cache capabilities beyond reasonable TTL (max 1 hour)
- Process requests after context deadline expiration

### 2.15 Wire-Level Envelope Standardization

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
  "error": "ResourceExhausted",
  "message": "Rate limit exceeded",
  "code": "RATE_LIMIT",
  "retry_after_ms": 5000,
  "details": { ... }
}
```

**Streaming Chunk Envelope:**
```json
{
  "type": "data|error|end",
  "data": { ... },
  "error": { ... }
}
```

### 2.16 Transport Envelope Specification

**HTTP/REST Transport Binding:**
```typescript
// Request Headers (MUST propagate)
interface RequestHeaders {
  'x-corpus-request-id'?: string;
  'x-corpus-deadline-ms'?: string;  // Epoch milliseconds
  'x-corpus-tenant'?: string;       // Hashed in telemetry
  'x-corpus-idempotency-key'?: string;
  'traceparent'?: string;           // W3C trace context
}

// Response Headers (MUST include)
interface ResponseHeaders {
  'x-corpus-request-id': string;
  'x-corpus-server': string;
  'x-corpus-version': string;
  'x-corpus-ms': string;           // Processing time in milliseconds
}
```

**Request Envelope:**
```json
{
  "op": "llm.complete|vector.query|graph.query|embedding.embed",
  "ctx": {
    "request_id": "req-123",
    "deadline_ms": 1731456000000,
    "tenant": "tenant-a",
    "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
  },
  "args": { ... }
}
```

**Response Envelope:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 45.2,
  "result": { ... }
}
```

**Streaming Framing:**
```typescript
// Each chunk is a complete JSON object followed by newline
{"type": "data", "data": { ... }}\n
{"type": "data", "data": { ... }}\n
{"type": "end"}\n
```

### 2.17 Type Serialization and Wire Format

**Field Name Stability:**
- Protocol type definitions use exact field names from reference implementations
- WireHandlers perform serialization using `dataclasses.asdict()` or equivalent
- Field names in JSON wire format match Python dataclass field names exactly
- No field name translation is performed at the wire layer

**Type Name Conventions:**
- LLM chunks use `LLMChunk` (not `StreamChunk`)
- Graph query results use `QueryResult` with `records` field
- All protocols use consistent naming patterns for similar concepts

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
    id: string;
    error: string;      // Normalized error type
    detail: string;     // Human-readable detail
    code?: string;      // Provider-specific code (optional)
  }>;
  results?: T[];        // Successful results when applicable
}
```

### 3.5 Normalized Error Envelope
See **ERRORS.md** §2 for complete error envelope specification.

### 3.6 Common Validation Rules
- **Required fields:** Presence validated before provider calls
- **Type checking:** JSON schema validation where applicable
- **Range validation:** Numeric bounds enforcement
- **Size limits:** Payload size constraints per capabilities

### 3.7 JSON Schema Requirements
- **Additional properties:** MUST be `false` unless explicitly allowed
- **Nullability:** Fields explicitly marked optional vs required
- **String formats:** UUID, ISO8601, email where semantically meaningful
- **Array bounds:** Minimum/maximum lengths specified per capability

### 3.8 Shared Token Usage Type
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
  protocol: string;               // "graph"
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
}
```

### 5.2 Optional Fields & Extensions
```typescript
// Optional capabilities
max_batch_size?: number;
max_query_complexity?: number;
supports_transactions?: boolean;
supports_index_management?: boolean;
supported_property_types?: string[];
query_timeout_ms?: number;          // Maximum query execution time
max_result_set_size?: number;       // Maximum rows returned
supports_explain?: boolean;         // Query explanation
```

### 5.3 Validation Rules
- **Truthfulness:** Reported capabilities MUST match actual provider support
- **Consistency:** Capabilities SHOULD remain stable between health checks
- **Discovery:** Clients SHOULD probe capabilities before using advanced features

### 5.4 Query Dialect Negotiation
- **Client preference:** Clients specify dialect in query requests
- **Adapter fallback:** Adapters MAY translate between dialects if supported
- **Error on unsupported:** `NotSupported` error for unknown dialects
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

**Note on Timestamps:**
The `created_at` and `updated_at` fields shown in operation response examples 
are added by backend implementations and are not part of the protocol's base 
Node/Edge type contracts. Adapters MAY include these fields in responses, but 
clients MUST NOT depend on their presence.

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

**Note on Timestamps:**
The `created_at` and `updated_at` fields shown in operation response examples 
are added by backend implementations and are not part of the protocol's base 
Node/Edge type contracts. Adapters MAY include these fields in responses, but 
clients MUST NOT depend on their presence.

### 6.3 QuerySpec
```typescript
interface QuerySpec {
  text: string;                   // Query in supported dialect
  params?: Metadata;              // Named parameters for query
  timeout_ms?: number;            // Query-specific timeout
  dialect?: string;               // Preferred query dialect
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

### 6.6 BatchSpec
```typescript
interface BatchSpec {
  operations: any[];              // Opaque operation objects
  atomic?: boolean;               // All-or-nothing execution
}
```

### 6.7 BatchResult
```typescript
interface BatchResult {
  processed_count: number;
  failed_count: number;
  failures: BatchFailure[];
  results?: any[];                // Opaque operation results
}

interface BatchFailure {
  operation_index: number;        // Index in original batch
  operation_type: string;         // Type of failed operation
  error: string;                  // Normalized error type
  detail: string;                 // Failure details
}
```

## 7. Graph Operations

### 7.1 capabilities
**Purpose:** Discover supported graph features and limits

**Input:** None (uses OperationContext)

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
    "protocol": "graph",
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
    "max_batch_ops": 1000
  }
}
```

### 7.2 upsert_nodes
**Purpose:** Create or update multiple nodes

**Input:**
```typescript
interface UpsertNodesSpec {
  nodes: Node[];
  namespace?: string;
}
```

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
      },
      {
        "id": "user-bob-456",
        "labels": ["User", "Standard"],
        "properties": {
          "name": "Bob Johnson",
          "email": "bob@example.com",
          "age": 25,
          "department": "Sales"
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
    "processed_count": 2,
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
      },
      {
        "id": "user-bob-456",
        "labels": ["User", "Standard"],
        "properties": {
          "name": "Bob Johnson",
          "email": "bob@example.com",
          "age": 25,
          "department": "Sales"
        },
        "namespace": "production"
      }
    ]
  }
}
```

### 7.3 upsert_edges
**Purpose:** Create or update multiple edges

**Input:**
```typescript
interface UpsertEdgesSpec {
  edges: Edge[];
  namespace?: string;
}
```

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
      },
      {
        "id": "reports-to-001",
        "src": "user-bob-456",
        "dst": "user-alice-123",
        "label": "REPORTS_TO",
        "properties": {
          "since": "2024-01-15",
          "role": "Team Lead"
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
    "processed_count": 2,
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
      },
      {
        "id": "reports-to-001",
        "src": "user-bob-456",
        "dst": "user-alice-123",
        "label": "REPORTS_TO",
        "properties": {
          "since": "2024-01-15",
          "role": "Team Lead"
        },
        "namespace": "production"
      }
    ]
  }
}
```

### 7.4 delete_nodes
**Purpose:** Remove nodes by ID or filter

**Input:**
```typescript
interface DeleteNodesSpec {
  ids: string[];
  namespace?: string;
  filter?: Metadata;
}
```

**Request Body:**
```json
{
  "op": "graph.delete_nodes",
  "ctx": {
    "request_id": "req-graph-delete-nodes-001",
    "tenant": "acme-corp"
  },
  "args": {
    "ids": ["user-alice-123", "user-bob-456"],
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
    "deleted_count": 2
  }
}
```

### 7.5 delete_edges
**Purpose:** Remove edges by ID or filter

**Input:**
```typescript
interface DeleteEdgesSpec {
  ids: string[];
  namespace?: string;
  filter?: Metadata;
}
```

**Request Body:**
```json
{
  "op": "graph.delete_edges",
  "ctx": {
    "request_id": "req-graph-delete-edges-001",
    "tenant": "acme-corp"
  },
  "args": {
    "ids": ["works-with-001", "reports-to-001"],
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
    "deleted_count": 2
  }
}
```

### 7.6 query
**Purpose:** Execute a graph query and return results

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
    "text": "MATCH (u:User)-[r:WORKS_WITH]->(c:User) WHERE u.department = $dept RETURN u.name, c.name, r.project",
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
        "c.name": "Bob Johnson",
        "r.project": "Phoenix"
      }
    ],
    "summary": {
      "query_time_ms": 42.1,
      "results_count": 1,
      "has_more": false,
      "dialect_used": "cypher"
    },
    "dialect": "cypher",
    "namespace": "production"
  }
}
```

### 7.7 stream_query
**Purpose:** Execute a graph query with streaming results

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
{"type": "data", "data": {"records": {"u.id": "user-alice-123", "u.name": "Alice Smith", "u.department": "Engineering"}, "is_final": false}}
{"type": "data", "data": {"records": {"u.id": "user-bob-456", "u.name": "Bob Johnson", "u.department": "Sales"}, "is_final": false}}
{"type": "data", "data": {"summary": {"query_time_ms": 125.4, "results_count": 2, "dialect_used": "cypher", "has_more": false}, "is_final": true}}
```

### 7.8 bulk_vertices
**Purpose:** Bulk operations on vertices (import/export)

**Input:** 
```typescript
interface BulkVerticesSpec {
  operation: 'import' | 'export';
  nodes?: Node[];
  format?: string;
  namespace?: string;
}
```

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
      },
      {
        "id": "user-bob-456",
        "labels": ["User", "Standard"],
        "properties": {
          "name": "Bob Johnson",
          "email": "bob@example.com",
          "age": 25,
          "department": "Sales"
        },
        "namespace": "production"
      }
    ]
  }
}
```

### 7.9 get_schema
**Purpose:** Retrieve graph schema information

**Input:** `{ namespace?: string }`

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
    "node_labels": ["User", "Premium", "Standard"],
    "edge_types": ["WORKS_WITH", "REPORTS_TO"],
    "property_keys": ["name", "email", "age", "department", "since", "project", "role"],
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

### 7.10 health
**Purpose:** Check adapter and provider health status

**Input:** None (uses OperationContext)

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
interface HealthStatus {
  ok: boolean;
  server: string;
  version: string;
  checks: {
    [check_name: string]: {
      ok: boolean;
      message?: string;
      details?: Metadata;
    };
  };
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
    "server": "neo4j-adapter",
    "version": "1.0.0",
    "checks": {
      "database_connectivity": {
        "ok": true,
        "message": "Connected to Neo4j cluster"
      },
      "query_execution": {
        "ok": true,
        "message": "Test query executed successfully"
      }
    }
  }
}
```

## 8. Graph Semantics (Normative)

### 8.1 Consistency Requirements
- **Read-after-write:** Updates visible to subsequent queries
- **Causal consistency:** Operations from same client observed in order
- **Cross-tenant isolation:** No data leakage between tenants

### 8.2 Referential Integrity Rules
- **Node deletion:** Connected edges automatically deleted when nodes are deleted
- **Edge validation:** Source and target nodes must exist
- **ID uniqueness:** Node/edge IDs unique within their namespace

### 8.3 Streaming Guarantees
- **Exactly-once delivery:** Each row delivered exactly once
- **Order preservation:** Results delivered in query result order
- **Terminal event:** Stream always ends with final chunk (`is_final: true`)
- **Cardinality bounds:** Streams SHOULD support at least 1M rows

### 8.4 Query Dialect Semantics
- **Parameter binding:** All dialects MUST support `$param` syntax
- **Query planning:** Adapters MAY optimize queries but MUST preserve semantics
- **Cost limits:** Queries exceeding `max_query_complexity` return `ResourceExhausted`

### 8.5 Batch Operation Semantics
- **Order preservation:** Operations executed in specified order
- **Atomic batches:** When `atomic=true`, all operations succeed or fail together
- **Partial visibility:** Non-atomic batch results visible as they complete
- **Failure isolation:** Individual operation failures don't affect others in non-atomic mode

### 8.6 Error Mappings
| Provider Error | Normalized Error | Details |
|----------------|------------------|---------|
| Syntax error | `QueryParseError` | `{"dialect": "cypher", "position": 45}` |
| Unknown node | `VertexNotFound` | `{"node_id": "v123"}` |
| Unknown edge | `EdgeNotFound` | `{"edge_id": "e456"}` |
| Schema violation | `SchemaValidationError` | `{"constraint": "label_missing"}` |
| Constraint violation | `BadRequest` | `{"constraint": "unique_property"}` |
| Timeout | `DeadlineExceeded` | `{"query_time_ms": 5000}` |

### 8.7 Deadlines
- **Query timeout:** `timeout_ms` in QuerySpec overrides context deadline
- **Stream duration:** Deadline applies to entire stream execution
- **Batch operations:** Deadline applies to entire batch execution

### 8.8 Idempotency Table
| Operation | Idempotent | Conditions |
|-----------|------------|------------|
| upsert_nodes | Yes | With same ID |
| delete_nodes | Yes | Always |
| upsert_edges | Yes | With same ID |
| delete_edges | Yes | Always |
| query | Yes | Always |
| batch | Conditional | Depends on individual operations |

---

## PART II — LLM PROTOCOL (v1.0)

## 9. LLM Capabilities

### 9.1 Required Flags
```typescript
interface LLMCapabilities {
  protocol: string;               // "llm"
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
}
```

### 9.2 Model Family & Supported Models
- **Model listing:** Accurate list of available provider models
- **Context lengths:** Per-model context limits where they differ
- **Feature support:** Model-specific capabilities (JSON, tools, etc.)

### 9.3 Optional Extensions
```typescript
// Advanced capabilities
max_tokens_per_minute?: number;
supports_vision?: boolean;
supports_audio?: boolean;
supports_function_calling?: boolean;
supported_sampling_parameters?: string[];
```

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
  model: string;
  messages: Message[];
  max_tokens?: number;
  temperature?: number;          // Range: [0.0, 2.0]
  top_p?: number;                // Range: [0.0, 1.0]
  stop_sequences?: string[];
  tools?: ToolDefinition[];
  tool_choice?: 'auto' | 'none' | { type: 'function'; function: { name: string } };
  response_format?: { type: 'text' } | { type: 'json_object' };
  seed?: number;                 // For deterministic sampling
}
```

### 10.3 LLMCompletion
```typescript
interface LLMCompletion {
  text: string;                  // Generated completion text
  model: string;
  model_family: string;          // Model family identifier
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

**Input:** None (uses OperationContext)

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
    "protocol": "llm",
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
    "supports_multi_tenant": true
  }
}
```

### 11.2 complete
**Purpose:** Generate LLM completion for given messages

**Input:** `CompletionSpec`

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
    "model_family": "openai",
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

**Input:** `CompletionSpec`

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
{"type": "data", "data": {"text": "Quantum", "is_final": false, "model": "gpt-4.1-mini"}}
{"type": "data", "data": {"text": " computing", "is_final": false, "model": "gpt-4.1-mini"}}
{"type": "data", "data": {"text": " is", "is_final": false, "model": "gpt-4.1-mini"}}
{"type": "data", "data": {"text": " a", "is_final": false, "model": "gpt-4.1-mini"}}
{"type": "data", "data": {"text": " new", "is_final": false, "model": "gpt-4.1-mini"}}
{"type": "data", "data": {"text": " type", "is_final": false, "model": "gpt-4.1-mini"}}
{"type": "data", "data": {"text": " of", "is_final": false, "model": "gpt-4.1-mini"}}
{"type": "data", "data": {"text": " computing", "is_final": false, "model": "gpt-4.1-mini"}}
{"type": "data", "data": {"text": " that", "is_final": false, "model": "gpt-4.1-mini"}}
{"type": "data", "data": {"text": " uses", "is_final": false, "model": "gpt-4.1-mini"}}
{"type": "data", "data": {"text": " quantum", "is_final": false, "model": "gpt-4.1-mini"}}
{"type": "data", "data": {"text": " bits", "is_final": false, "model": "gpt-4.1-mini"}}
{"type": "data", "data": {"text": " (qubits)", "is_final": false, "model": "gpt-4.1-mini"}}
{"type": "data", "data": {"text": " instead", "is_final": false, "model": "gpt-4.1-mini"}}
{"type": "data", "data": {"text": " of", "is_final": false, "model": "gpt-4.1-mini"}}
{"type": "data", "data": {"text": " classical", "is_final": false, "model": "gpt-4.1-mini"}}
{"type": "data", "data": {"text": " bits.", "is_final": true, "model": "gpt-4.1-mini", "usage_so_far": {"prompt_tokens": 12, "completion_tokens": 18, "total_tokens": 30}}}
{"type": "end"}
```

### 11.4 count_tokens
**Purpose:** Count tokens in text for a specific model

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
  models: {
    [model_name: string]: {
      status: 'ready' | 'loading' | 'error';
      message?: string;
    };
  };
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
    "models": {
      "gpt-4.1-mini": {
        "status": "ready",
        "message": "Model available"
      },
      "gpt-4.1-max": {
        "status": "loading", 
        "message": "Model initializing"
      }
    }
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

### 12.4 Max Token Handling
- **Context limits:** Automatic rejection when prompt + max_tokens > context window
- **Token counting:** Use provider tokenizer for accurate counting
- **Truncation:** Never truncate prompts automatically

### 12.5 JSON Mode Strictness
- **Guaranteed JSON:** When `response_format: {type: "json_object"}`, output MUST be valid JSON
- **Schema adherence:** JSON structure follows model training, no schema enforcement
- **Error on violation:** Return `BadRequest` if JSON mode requested but prompt doesn't specify JSON

### 12.6 Tool Call Semantics
- **Deterministic ordering:** Tool calls returned in consistent order
- **Parallel execution:** When `supports_parallel_tool_calls=true`, tools may be called concurrently
- **Exhaustion behavior:** Stop generation after tool calls if no further text content

### 12.7 Streaming Guarantees
- **Chunk integrity:** Complete tokens delivered in each chunk
- **Order preservation:** Chunks delivered in correct sequence
- **Final indication:** Clear termination with `is_final: true`

### 12.8 Error Mappings
See **ERRORS.md** §6.1 for complete LLM error mapping specifications.

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
  protocol: string;               // "vector"
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
}
```

### 13.2 Supported Metrics
- **cosine:** Cosine similarity (1 - cosine distance)
- **euclidean:** Euclidean distance (inverted for similarity)
- **dotproduct:** Dot product similarity
- **Manhattan:** L1 distance (inverted for similarity)

### 13.3 Optional Extensions
```typescript
supports_hybrid_search?: boolean;
supports_vector_compression?: boolean;
supported_index_types?: string[];
approximate_search_accuracy?: number; // 0.0 to 1.0
```

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
}
```

## 15. Vector Operations

### 15.1 capabilities
**Purpose:** Discover supported vector features and limits

**Input:** None (uses OperationContext)

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
    "protocol": "vector",
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
    "max_filter_terms": 10
  }
}
```

### 15.2 query
**Purpose:** Find similar vectors using approximate nearest neighbor search

**Input:** `QuerySpec`

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
      },
      {
        "vector": {
          "id": "doc-456",
          "vector": [0.08, 0.18, 0.28, 0.38, 0.48, 0.08, 0.18, 0.28, 0.38, 0.48],
          "metadata": {
            "title": "Machine Learning Guide",
            "category": "technology", 
            "language": "en"
          },
          "namespace": "documents"
        },
        "score": 0.92,
        "distance": 0.08
      }
    ],
    "query_vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5],
    "namespace": "documents",
    "total_matches": 2
  }
}
```

### 15.3 upsert
**Purpose:** Insert or update vectors in a namespace

**Input:** `UpsertSpec`

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
      },
      {
        "id": "doc-999",
        "vector": [0.09, 0.19, 0.29, 0.39, 0.49, 0.09, 0.19, 0.29, 0.39, 0.49],
        "metadata": {
          "title": "Updated Programming Guide",
          "category": "technology"
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
    "upserted_count": 2,
    "failed_count": 0,
    "failures": []
  }
}
```

### 15.4 delete
**Purpose:** Remove vectors by ID or metadata filter

**Input:** `DeleteSpec`

**Request Body:**
```json
{
  "op": "vector.delete",
  "ctx": {
    "request_id": "req-vector-delete-001",
    "tenant": "acme-corp"
  },
  "args": {
    "ids": ["doc-123", "doc-456"],
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
    "deleted_count": 2,
    "failed_count": 0,
    "failures": []
  }
}
```

### 15.5 create_namespace
**Purpose:** Create a new vector namespace/collection

**Input:** `NamespaceSpec`

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
    }
  }
}
```

### 15.6 delete_namespace
**Purpose:** Remove a vector namespace and all its vectors

**Input:** `{ namespace: string }`

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
    }
  }
}
```

### 15.7 health
**Purpose:** Check vector store health and namespace status

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
  namespaces: {
    [namespace: string]: {
      ready: boolean;
      vector_count: number;
      dimensions: number;
      [key: string]: any;  // Provider-specific metadata
    };
  };
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
    "namespaces": {
      "documents": {
        "ready": true,
        "vector_count": 15000,
        "dimensions": 1536,
        "index_size_mb": 245
      },
      "images": {
        "ready": false,
        "vector_count": 0,
        "dimensions": 512,
        "index_status": "initializing"
      }
    }
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
- **Precedence:** AND takes precedence over OR in complex filters

### 16.3 Metrics & Scoring Rules
- **Normalization:** Scores normalized to [0, 1] range where possible
- **Consistency:** Same metric always produces same score range
- **Documentation:** Clear explanation of score meaning for each metric
- **Cosine normalization:** For cosine similarity, vectors SHOULD be L2-normalized

### 16.4 Search Behavior
- **Approximate vs exact:** Adapters SHOULD document search accuracy guarantees
- **Hybrid search:** When supported, balance vector and keyword search
- **Result ordering:** Matches ordered by descending score (highest similarity first)

### 16.5 Batch Semantics
- **Partial success:** Batch operations continue despite individual failures
- **Order preservation:** Results maintain input order where applicable
- **Size limits:** Batch size constrained by provider capabilities
- **Empty batch validation:** Reject empty batches with `BadRequest`

### 16.6 Error Mappings
See **ERRORS.md** §6.3 for complete Vector error mapping specifications.

### 16.7 Deadline Rules
- **Query timeouts:** Deadline applies to entire search operation
- **Batch operations:** Deadline applies to entire batch processing
- **Namespace operations:** Deadline applies to management operations

---

## PART IV — EMBEDDING PROTOCOL (v1.0)

## 17. Embedding Capabilities

### 17.1 Required Flags
```typescript
interface EmbeddingCapabilities {
  protocol: string;               // "embedding"
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
  supports_deadline: boolean;
  normalizes_at_source: boolean;
  idempotent_operations: boolean;
}
```

### 17.2 Model Support
- **Model listing:** Available embedding models with dimensions
- **Batch capabilities:** Maximum batch sizes per model
- **Text limits:** Maximum input lengths per model

### 17.3 Optional Extensions
```typescript
truncation_mode?: 'start' | 'end' | 'auto';
supported_languages?: string[];
supports_async_embedding?: boolean;
max_requests_per_minute?: number;
```

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

**Input:** None (uses OperationContext)

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
    "protocol": "embedding",
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
    "idempotent_operations": true
  }
}
```

### 19.2 embed
**Purpose:** Generate embedding vector for a single text

**Input:** `EmbedSpec`

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

**Input:** `EmbedBatchSpec`

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
  models: {
    [model_name: string]: {
      status: 'ready' | 'loading' | 'error';
      dimensions: number;
      max_text_length: number;
    };
  };
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
    "models": {
      "text-embedding-3-large": {
        "status": "ready",
        "dimensions": 3072,
        "max_text_length": 8192
      },
      "text-embedding-3-small": {
        "status": "ready",
        "dimensions": 1536,
        "max_text_length": 8192
      }
    }
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
- **Zero-length texts:** Reject empty inputs with `BadRequest` error
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
See **ERRORS.md** §6.2 for complete Embedding error mapping specifications.

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
- **Embedding:** `TextTooLong`, `EmbeddingDimensionMismatch`
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
- **Automatic retry:** `TransientNetwork`, `Unavailable`, `ResourceExhausted`
- **Conditional retry:** `DeadlineExceeded` only with increased deadline
- **No retry:** `BadRequest`, `AuthError`, `NotSupported`

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
- [ ] Capabilities reported truthfully
- [ ] Wire envelopes follow standardization
- [ ] Streaming semantics followed

### 26.2 Protocol-Specific Compliance
**Graph:**
- [ ] Query dialect negotiation supported
- [ ] Parameter binding implemented
- [ ] Streaming cardinality bounds respected
- [ ] Batch partial failure reporting
- [ ] Result schema support (when available)

**LLM:**
- [ ] Stop sequence precedence followed
- [ ] JSON mode strictness enforced
- [ ] Tool call determinism maintained
- [ ] Stream equivalence guaranteed

**Vector:**
- [ ] Dimension enforcement strict
- [ ] Filter semantics consistent
- [ ] Metric scoring normalized
- [ ] Hybrid search balanced

**Embedding:**
- [ ] Truncation rules followed
- [ ] Normalization consistent
- [ ] Dimension matching enforced
- [ ] Empty input handling defined

## 27. Cross-Protocol Standardization Tables

### 27.1 Health Check Standardization
| Field | All Protocols | Graph | LLM | Vector | Embedding |
|-------|---------------|-------|-----|--------|-----------|
| `ok` | REQUIRED | ✓ | ✓ | ✓ | ✓ |
| `server` | REQUIRED | ✓ | ✓ | ✓ | ✓ |
| `version` | REQUIRED | ✓ | ✓ | ✓ | ✓ |
| `checks` | OPTIONAL | ✓ | ✓ | ✓ | ✓ |
| `models` | PROTOCOL | ✗ | ✓ | ✗ | ✓ |
| `namespaces` | PROTOCOL | ✗ | ✗ | ✓ | ✗ |

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

## 28. Glossary

### 28.1 Common Terminology
- **Adapter:** Protocol implementation bridging Corpus APIs to providers
- **Provider:** Underlying AI/data service (OpenAI, Pinecone, Neo4j, etc.)
- **Tenant:** Logical isolation boundary for multi-tenant deployments
- **Namespace:** Protocol-specific isolation scope (collections, graphs, etc.)
- **OperationContext:** Request-scoped context for deadlines, tracing, etc.

### 28.2 Protocol-Specific Terms
- **Graph:** Property graph with nodes, edges, and graph queries
- **LLM:** Large language model for text generation and completion
- **Vector:** High-dimensional vectors for similarity search
- **Embedding:** Text-to-vector transformation models
- **Streaming:** Incremental response delivery for long-running operations

## 29. Appendix

### 29.1 Example Capability Responses
```json
{
  "LLM": {
    "protocol": "llm",
    "server": "openai-adapter",
    "version": "1.0.0",
    "model_family": "openai",
    "supported_models": ["gpt-4", "gpt-3.5-turbo"],
    "max_context_length": 8192,
    "supports_streaming": true,
    "supports_count_tokens": true
  }
}
```

### 29.2 Example Error Envelopes
```json
{
  "ok": false,
  "error": "ResourceExhausted",
  "message": "Rate limit exceeded for model gpt-4",
  "code": "RATE_LIMIT",
  "retry_after_ms": 5000,
  "resource_scope": "model"
}
```

### 29.3 Example Batch Responses
```json
{
  "processed_count": 95,
  "failed_count": 5,
  "failures": [
    {
      "id": "vec-123",
      "error": "DimensionMismatch", 
      "detail": "Expected 1536 dimensions, got 1537"
    }
  ]
}
```

### 29.4 Example Streaming Traces
**Graph Stream:**
```json
{"type": "data", "data": {"records": {"user": {"id": "u1", "labels": ["User"], "properties": {"name": "Alice"}}}, "is_final": false}}
{"type": "data", "data": {"records": {"user": {"id": "u2", "labels": ["User"], "properties": {"name": "Bob"}}}, "is_final": false}}
{"type": "data", "data": {"summary": {"query_time_ms": 45, "results_count": 2, "dialect_used": "cypher", "has_more": false}, "is_final": true}}
```

**LLM Stream:**
```json
{"type": "data", "data": {"text": "Hello", "is_final": false, "model": "gpt-4"}}
{"type": "data", "data": {"text": " world", "is_final": false, "model": "gpt-4"}}
{"type": "data", "data": {"text": "!", "is_final": true, "model": "gpt-4", "usage_so_far": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}}
```

**Vector Batch Failure:**
```json
{
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
}
```

### 29.5 JSON Schema References
- **Strict validation:** All schemas prohibit additional properties unless specified
- **Required fields:** Clearly marked in type definitions
- **Default values:** Specified where semantically meaningful
- **Range validation:** Numeric bounds enforced per capabilities

---

*End of PROTOCOLS.md (protocols_version 1.0)*