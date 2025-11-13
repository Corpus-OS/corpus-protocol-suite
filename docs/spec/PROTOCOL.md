# PROTOCOLS.md

**Corpus Protocol Suite — Unified Specification for Graph, LLM, Vector, and Embedding Adapters**  
**protocols_version:** `1.0`

> This document defines the unified protocol specification for all Corpus adapters. It establishes the normative behavior, types, and semantics for Graph, LLM, Vector, and Embedding protocols while maintaining cross-protocol consistency.

---

## 0. Document Metadata

### 0.1 protocols_version: "1.0"
- **Status:** Stable / Normative
- **Effective Date:** 2025-01-01
- **Replaces:** None (initial version)

### 0.2 Relationship to Companion Documents
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

---

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

---

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
  
  // Derived methods
  remaining_ms(): number | null; // Time until deadline
  is_expired(): boolean;         // Deadline check
}
```

### 2.2 Tenant Isolation & Hashing
- **Tenant MUST NOT** appear in raw form in metrics, logs, or error messages
- **Tenant hash:** First 12 characters of `SHA256(tenant)` for telemetry
- **Isolation:** Adapters enforce tenant boundaries in provider API calls

### 2.3 Deadlines & Budgets
- **Propagation:** Adapters MUST propagate `deadline_ms` to provider APIs
- **Safety buffer:** Subtract 50-100ms from remaining time for network overhead
- **Expiration:** Reject operations where `ctx.is_expired()` returns true

### 2.4 Stream Semantics
- **Single terminal:** Exactly one terminal event (success or error)
- **No content after terminal:** Stream MUST end after final event
- **Heartbeats:** Optional keep-alive messages allowed but not required
- **Backpressure:** Clients control consumption rate; adapters respect flow control

### 2.5 Normalized Error Model
See **ERRORS.md** for complete error taxonomy and mapping requirements.

### 2.6 Observability & Metrics
See **METRICS.md** for complete metrics specification and emission rules.

### 2.7 Capability Probing
- **Dynamic discovery:** Clients probe `capabilities()` to determine supported features
- **Truthful reporting:** Adapters MUST accurately report actual capabilities
- **Caching:** Capabilities may be cached with appropriate TTL (typically 5-60 minutes)

### 2.8 SIEM-Safe Requirements
- **No raw content** in logs, metrics, or errors (prompts, vectors, embeddings)
- **Structured data only** in telemetry fields
- **Tenant hashing** as described in §2.2
- **Cardinality control** in labels and dimensions

### 2.9 Idempotency Expectations
| Operation Type | Idempotent? | Notes |
|---------------|-------------|--------|
| Read operations | Yes | Same inputs → same outputs |
| Create with ID | Yes | Duplicate ID returns existing |
| Create without ID | No | Generates new ID each time |
| Update | Yes | Last write wins |
| Delete | Yes | Multiple deletes return success |

### 2.10 Global Invariants (MUST/MUST NOT)

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

### 2.11 Wire-Level Envelope Standardization

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
  "http_status": 429,
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

### 2.12 Transport Envelope Specification

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

---

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

---

## 4. Protocol Overview

### 4.1 Graph Protocol
- **Purpose:** Property graph storage and querying
- **Key operations:** CRUD on vertices/edges, graph queries, batch operations
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
  server: string;                   // Adapter identifier
  version: string;                  // Protocol version
  supported_query_dialects: string[]; // e.g., ["cypher", "gremlin"]
  supports_stream_query: boolean;
  supports_bulk_vertices: boolean;
  supports_batch: boolean;
  supports_schema: boolean;
  idempotent_writes: boolean;
  supports_deadline: boolean;
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

---

## 6. Graph Types

### 6.1 Vertex
```typescript
interface Vertex {
  id: string;
  label: string;
  properties: Metadata;
  created_at?: string;    // ISO 8601 timestamp
  updated_at?: string;    // ISO 8601 timestamp
}
```

### 6.2 Edge
```typescript
interface Edge {
  id: string;
  label: string;
  from_vertex: string;    // Source vertex ID
  to_vertex: string;      // Target vertex ID  
  properties: Metadata;
  created_at?: string;
  updated_at?: string;
}
```

### 6.3 QuerySpec
```typescript
interface QuerySpec {
  query: string;                   // Query in supported dialect
  parameters?: Metadata;          // Named parameters for query
  timeout_ms?: number;            // Query-specific timeout
  include_metadata?: boolean;     // Return vertex/edge properties
  limit?: number;                 // Maximum results to return
  dialect?: string;               // Preferred query dialect
  explain?: boolean;              // Return query execution plan
}
```

### 6.4 QueryResult
```typescript
interface QueryResult {
  rows: any[];                    // Query result rows
  columns?: string[];             // Column names for tabular results
  schema?: ResultSchema;          // Optional result schema
  summary: {                      // Execution summary (REQUIRED)
    query_time_ms: number;
    results_count: number;
    has_more: boolean;
    dialect_used: string;         // Actual dialect used
    plan?: QueryPlan;             // Execution plan if explain=true
  };
}

interface ResultSchema {
  columns: Array<{
    name: string;
    type: 'string' | 'number' | 'boolean' | 'vertex' | 'edge' | 'path';
    nullable: boolean;
  }>;
}

interface QueryPlan {
  steps: Array<{
    operation: string;
    estimated_rows?: number;
    cost?: number;
  }>;
}
```

### 6.5 StreamQueryResult
```typescript
// Async iterator yielding stream events
type StreamQueryResult = AsyncIterable<StreamEvent>;

interface StreamEvent {
  type: 'row' | 'summary' | 'error' | 'end' | 'schema';
  data?: any;
  schema?: ResultSchema;
  error?: NormalizedError;
}
```

### 6.6 BatchSpec
```typescript
interface BatchSpec {
  operations: BatchOperation[];
  atomic?: boolean;               // All-or-nothing execution
}

type BatchOperation = 
  | { type: 'create_vertex'; vertex: Vertex }
  | { type: 'delete_vertex'; id: string }
  | { type: 'create_edge'; edge: Edge }
  | { type: 'delete_edge'; id: string }
  | { type: 'update_vertex'; vertex: Vertex }
  | { type: 'update_edge'; edge: Edge };
```

### 6.7 BatchResult
```typescript
interface BatchResult {
  processed_count: number;
  failed_count: number;
  failures: BatchFailure[];
  results?: BatchOperationResult[]; // For successful operations
}

interface BatchFailure {
  operation_index: number;        // Index in original batch
  operation_type: string;         // Type of failed operation
  error: string;                  // Normalized error type
  detail: string;                 // Failure details
}
```

---

## 7. Graph Operations

### 7.1 create_vertex
**Purpose:** Create a new vertex in the graph

**Input:**
```typescript
interface CreateVertexSpec {
  vertex: Vertex;
  if_not_exists?: boolean;        // Skip if vertex already exists
}
```

**Output:** `Vertex` (created vertex with server-generated timestamps)

**Errors:**
- `BadRequest`: Invalid vertex data, missing required fields
- `AuthError`: Insufficient permissions
- `SchemaValidationError`: Vertex violates schema constraints

### 7.2 delete_vertex
**Purpose:** Remove a vertex and its edges from the graph

**Input:**
```typescript
interface DeleteVertexSpec {
  id: string;
  cascade?: boolean;              // Also delete connected edges
}
```

**Output:** `{ deleted: boolean }` (true if vertex existed and was deleted)

**Errors:**
- `BadRequest`: Invalid ID format
- `VertexNotFound`: Vertex does not exist

### 7.3 create_edge
**Purpose:** Create a new relationship between vertices

**Input:**
```typescript
interface CreateEdgeSpec {
  edge: Edge;
  if_not_exists?: boolean;
}
```

**Output:** `Edge` (created edge with server-generated timestamps)

**Errors:**
- `BadRequest`: Invalid edge data, missing vertices
- `VertexNotFound`: Source or target vertex does not exist
- `SchemaValidationError`: Edge violates schema constraints

### 7.4 delete_edge
**Purpose:** Remove an edge from the graph

**Input:**
```typescript
interface DeleteEdgeSpec {
  id: string;
}
```

**Output:** `{ deleted: boolean }` (true if edge existed and was deleted)

**Errors:**
- `BadRequest`: Invalid ID format
- `EdgeNotFound`: Edge does not exist

### 7.5 query
**Purpose:** Execute a graph query and return results

**Input:** `QuerySpec`

**Output:** `QueryResult`

**Errors:**
- `BadRequest`: Invalid query syntax, missing parameters
- `QueryParseError`: Query cannot be parsed
- `NotSupported`: Query dialect or feature not supported
- `ResourceExhausted`: Query too complex or timeout

### 7.6 stream_query
**Purpose:** Execute a graph query with streaming results

**Input:** `QuerySpec`

**Output:** `StreamQueryResult` (async iterator)

**Stream Semantics:**
- Optional `schema` event with result structure
- Zero or more `row` events with result data
- Exactly one `summary` event with query statistics
- Exactly one `end` event to terminate stream
- Or one `error` event if query fails

**Errors:**
- Same as `query` but delivered via stream error event

### 7.7 batch
**Purpose:** Execute multiple graph operations in a single request

**Input:** `BatchSpec`

**Output:** `BatchResult`

**Partial Failure:**
- Individual operation failures reported in `failures[]`
- Successful operations executed and reported in `results[]`
- If `atomic=true`, entire batch fails on first error

**Errors:**
- `BadRequest`: Invalid batch specification
- `ResourceExhausted`: Batch too large

### 7.8 health
**Purpose:** Check adapter and provider health status

**Input:** None (uses OperationContext)

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

---

## 8. Graph Semantics (Normative)

### 8.1 Consistency Requirements
- **Read-after-write:** Updates visible to subsequent queries
- **Causal consistency:** Operations from same client observed in order
- **Cross-tenant isolation:** No data leakage between tenants

### 8.2 Referential Integrity Rules
- **Vertex deletion:** With `cascade=true`, connected edges automatically deleted
- **Edge validation:** Source and target vertices must exist
- **ID uniqueness:** Vertex/edge IDs unique within their namespace

### 8.3 Streaming Guarantees
- **Exactly-once delivery:** Each row delivered exactly once
- **Order preservation:** Results delivered in query result order
- **Terminal event:** Stream always ends with `end` or `error` event
- **Cardinality bounds:** Streams SHOULD support at least 1M rows
- **Schema early delivery:** Schema events SHOULD precede row events when available

### 8.4 Query Dialect Semantics
- **Parameter binding:** All dialects MUST support `$param` or `{param}` syntax
- **Query planning:** Adapters MAY optimize queries but MUST preserve semantics
- **Cost limits:** Queries exceeding `max_query_complexity` return `ResourceExhausted`
- **Explain output:** When `explain=true`, return execution plan in summary

### 8.5 Result Schema Semantics
- **Optional schema:** Adapters MAY provide result schemas for structured queries
- **Type consistency:** Schema types SHOULD match actual result types
- **Nullability:** Schema indicates which columns may contain null values
- **Dynamic schemas:** Schemas MAY vary based on query parameters

### 8.6 Batch Operation Semantics
- **Order preservation:** Operations executed in specified order
- **Atomic batches:** When `atomic=true`, all operations succeed or fail together
- **Partial visibility:** Non-atomic batch results visible as they complete
- **Failure isolation:** Individual operation failures don't affect others in non-atomic mode

### 8.7 Error Mappings
| Provider Error | Normalized Error | Details |
|----------------|------------------|---------|
| Syntax error | `QueryParseError` | `{"dialect": "cypher", "position": 45}` |
| Unknown vertex | `VertexNotFound` | `{"vertex_id": "v123"}` |
| Unknown edge | `EdgeNotFound` | `{"edge_id": "e456"}` |
| Schema violation | `SchemaValidationError` | `{"constraint": "label_missing"}` |
| Constraint violation | `BadRequest` | `{"constraint": "unique_property"}` |
| Timeout | `DeadlineExceeded` | `{"query_time_ms": 5000}` |

### 8.8 Deadlines
- **Query timeout:** `timeout_ms` in QuerySpec overrides context deadline
- **Stream duration:** Deadline applies to entire stream execution
- **Batch operations:** Deadline applies to entire batch execution

### 8.9 Idempotency Table
| Operation | Idempotent | Conditions |
|-----------|------------|------------|
| create_vertex | Yes | With `if_not_exists=true` or same ID |
| delete_vertex | Yes | Always |
| create_edge | Yes | With `if_not_exists=true` or same ID |
| delete_edge | Yes | Always |
| query | Yes | Always |
| batch | Conditional | Depends on individual operations |

---

## PART II — LLM PROTOCOL (v1.0)

## 9. LLM Capabilities

### 9.1 Required Flags
```typescript
interface LLMCapabilities {
  server: string;
  version: string;
  supported_models: string[];
  max_context_length: number;
  supports_streaming: boolean;
  supports_roles: boolean;
  supports_system_message: boolean;
  supports_json_output: boolean;
  supports_parallel_tool_calls: boolean;
  supports_deadline: boolean;
  supports_count_tokens: boolean;
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

---

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

### 10.3 CompletionResult
```typescript
interface CompletionResult {
  id: string;                    // Provider-generated completion ID
  model: string;
  choices: Array<{
    index: number;
    message: Message;
    finish_reason: 'stop' | 'length' | 'tool_calls' | 'content_filter';
  }>;
  usage: TokenUsage;
}
```

### 10.4 StreamChunk
```typescript
interface StreamChunk {
  id: string;
  model: string;
  choices: Array<{
    index: number;
    delta: Partial<Message>;     // Incremental message content
    finish_reason?: string;
  }>;
  usage?: TokenUsage;
}
```

---

## 11. LLM Operations

### 11.1 complete
**Purpose:** Generate LLM completion for given messages

**Input:** `CompletionSpec`

**Output:** `CompletionResult`

**Errors:**
- `BadRequest`: Invalid messages, unsupported parameters
- `ModelNotFound`: Requested model not available
- `PromptTooLong`: Input exceeds context window
- `ContentFiltered`: Content violates safety policies

### 11.2 stream
**Purpose:** Stream LLM completion incrementally

**Input:** `CompletionSpec`

**Output:** `AsyncIterable<StreamChunk>`

**Stream Semantics:**
- Multiple chunks with incremental content
- Exactly one chunk with `finish_reason` set
- Final chunk may include token usage

**Errors:** Same as `complete`, delivered as final error chunk

### 11.3 count_tokens
**Purpose:** Count tokens in text for a specific model

**Input:** 
```typescript
interface CountTokensSpec {
  text: string;
  model: string;
}
```

**Output:** `{ tokens: number }`

**Errors:**
- `NotSupported`: Token counting not available for model
- `ModelNotFound`: Model not available

### 11.4 health
**Purpose:** Check LLM provider health and model availability

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

---

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
- **Final indication:** Clear termination with finish reason

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
  server: string;
  version: string;
  max_dimensions: number;
  supported_metrics: string[];    // e.g., ["cosine", "euclidean", "dotproduct"]
  supports_namespaces: boolean;
  supports_metadata_filtering: boolean;
  supports_batch_operations: boolean;
  max_batch_size: number;
  supports_index_management: boolean;
  supports_deadline: boolean;
}
```

### 13.2 Supported Metrics
- **cosine:** Cosine similarity (1 - cosine distance)
- **euclidean:** Euclidean distance (inverted for similarity)
- **dotproduct:** Dot product similarity
- **Manhattan:** L1 distance (inverted for similarity)

### 13.3 Optional Extensions
```typescript
max_top_k?: number;
max_filter_terms?: number;
supports_hybrid_search?: boolean;
supports_vector_compression?: boolean;
supported_index_types?: string[];
approximate_search_accuracy?: number; // 0.0 to 1.0
```

---

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
  distance?: number;              // Raw distance metric (lower = more similar)
}
```

### 14.3 QuerySpec
```typescript
interface QuerySpec {
  vector: number[];
  top_k: number;
  namespace?: string;
  filter?: Metadata;              // Metadata filter conditions
  include_metadata?: boolean;
  include_vectors?: boolean;
  hybrid_alpha?: number;          // 0.0 = pure vector, 1.0 = pure keyword
}
```

### 14.4 QueryResult
```typescript
interface QueryResult {
  matches: VectorMatch[];
  query_vector: number[];         // May be normalized
  namespace: string;
  total_matches?: number;         // Total matches before top_k
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
  metric: string;                 // e.g., "cosine", "euclidean"
  config?: Metadata;              // Provider-specific configuration
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

---

## 15. Vector Operations

### 15.1 query
**Purpose:** Find similar vectors using approximate nearest neighbor search

**Input:** `QuerySpec`

**Output:** `QueryResult`

**Errors:**
- `BadRequest`: Invalid vector dimensions, unsupported metric
- `DimensionMismatch`: Query vector dimensions don't match namespace
- `NamespaceNotFound`: Specified namespace doesn't exist
- `IndexNotReady`: Namespace exists but index not ready

### 15.2 upsert
**Purpose:** Insert or update vectors in a namespace

**Input:** `UpsertSpec`

**Output:** `UpsertResult`

**Partial Failure:**
- Individual vector failures reported in `failures[]`
- Successful vectors upserted and counted in `upserted_count`

**Errors:**
- `BadRequest`: Invalid vectors, batch too large
- `DimensionMismatch`: Vector dimensions don't match namespace
- `NamespaceNotFound`: Namespace doesn't exist

### 15.3 delete
**Purpose:** Remove vectors by ID or metadata filter

**Input:** `DeleteSpec`

**Output:** `DeleteResult`

**Partial Failure:**
- Individual deletion failures reported in `failures[]`
- `deleted_count` reflects successful deletions

**Errors:**
- `BadRequest`: Invalid filter syntax
- `NamespaceNotFound`: Namespace doesn't exist

### 15.4 create_namespace
**Purpose:** Create a new vector namespace/collection

**Input:** `NamespaceSpec`

**Output:** `NamespaceResult`

**Errors:**
- `BadRequest`: Invalid namespace configuration
- `NotSupported`: Namespace management not supported
- `ResourceExhausted`: Too many namespaces

### 15.5 delete_namespace
**Purpose:** Remove a vector namespace and all its vectors

**Input:** `{ namespace: string }`

**Output:** `NamespaceResult`

**Errors:**
- `BadRequest`: Invalid namespace name
- `NamespaceNotFound`: Namespace doesn't exist
- `NotSupported`: Namespace management not supported

### 15.6 health
**Purpose:** Check vector store health and namespace status

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
    };
  };
}
```

---

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
- **Hybrid search:** When `hybrid_alpha` provided, balance vector and keyword search
- **Result ordering:** Matches ordered by descending score (highest similarity first)

### 16.5 Batch Semantics
- **Partial success:** Batch operations continue despite individual failures
- **Order preservation:** Results maintain input order where applicable
- **Size limits:** Batch size constrained by provider capabilities

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
  server: string;
  version: string;
  supported_models: string[];
  max_batch_size: number;
  max_text_length: number;
  max_dimensions: number;
  supports_normalization: boolean;
  supports_truncation: boolean;
  supports_token_counting: boolean;
  supports_multi_tenant: boolean;
  supports_deadline: boolean;
  normalizes_at_source: boolean;
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

---

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

---

## 19. Embedding Operations

### 19.1 embed
**Purpose:** Generate embedding vector for a single text

**Input:** `EmbedSpec`

**Output:** `EmbedResult`

**Errors:**
- `BadRequest`: Invalid text, unsupported model
- `TextTooLong`: Text exceeds limits and truncation disabled
- `ModelNotFound`: Requested model not available
- `ContentFiltered`: Text violates content policies

### 19.2 embed_batch
**Purpose:** Generate embeddings for multiple texts in batch

**Input:** `EmbedBatchSpec`

**Output:** `EmbedBatchResult`

**Partial Failure:**
- Individual text failures reported in `failed_texts[]`
- Successful embeddings returned in `embeddings[]`

**Errors:**
- `BadRequest`: Batch too large, invalid texts
- `NotSupported`: Batch embedding not available (fallback to single)

### 19.3 count_tokens
**Purpose:** Count tokens in text for embedding model

**Input:** 
```typescript
interface CountTokensSpec {
  text: string;
  model: string;
}
```

**Output:** `{ tokens: number }`

**Errors:**
- `NotSupported`: Token counting not available
- `ModelNotFound`: Model not available

### 19.4 health
**Purpose:** Check embedding provider health and model status

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

---

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
- **Model adherence:** Embedding dimensions MUST match `capabilities.max_dimensions` for model
- **Batch consistency:** All embeddings in batch have same dimensions
- **Error on mismatch:** `EmbeddingDimensionMismatch` if dimensions don't match expectations

### 20.4 Empty Input Handling
- **Zero-length texts:** Return zero vector or error based on provider capability
- **Whitespace-only:** Treat as normal text, not empty
- **Error preference:** Prefer zero vector over error for empty inputs

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
- **Graph:** Property graph with vertices, edges, and graph queries
- **LLM:** Large language model for text generation and completion
- **Vector:** High-dimensional vectors for similarity search
- **Embedding:** Text-to-vector transformation models
- **Streaming:** Incremental response delivery for long-running operations

## 29. Appendix

### 29.1 Example Capability Responses
```json
{
  "LLM": {
    "server": "openai-adapter",
    "version": "1.0.0",
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
  "http_status": 429,
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
{"type": "schema", "schema": {"columns": [{"name": "user", "type": "vertex"}]}}
{"type": "row", "data": {"user": {"id": "u1", "label": "User", "properties": {"name": "Alice"}}}}
{"type": "row", "data": {"user": {"id": "u2", "label": "User", "properties": {"name": "Bob"}}}}
{"type": "summary", "data": {"query_time_ms": 45, "results_count": 2, "dialect_used": "cypher"}}
{"type": "end"}
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
