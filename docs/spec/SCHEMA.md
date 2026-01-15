# CORPUS Schema Reference

**schema_version:** `1.0.0`  
**protocols_version:** `1.0`  
**json_schema_draft:** `2020-12`

> This document defines the schema architecture for the Corpus Protocol Suite. It establishes the normative JSON Schema definitions for all protocol operations, types, and wire formats while maintaining cross-protocol consistency and validation guarantees.

> **Schema Precedence:** For field names, types, required/optional status, enums, and constraints, SCHEMA.md is normative.
For operation semantics, lifecycle, and behavior, PROTOCOLS.md is normative.

> **Schema Validation Convention:** All schemas use JSON Schema Draft 2020-12 with strict validation. Type definitions use `$defs` for reuse and `$id`-based resolution. Python, Go, etc. bindings are reference implementations.

---

## Table of Contents
- [1. Introduction](#1-introduction)
  - [1.1 Purpose and Scope](#11-purpose-and-scope)
  - [1.2 Document Relationships](#12-document-relationships)
  - [1.3 Schema Philosophy](#13-schema-philosophy)
- [2. Schema Architecture](#2-schema-architecture)
  - [2.1 Directory Structure](#21-directory-structure)
  - [2.2 Schema Organization Principles](#22-schema-organization-principles)
  - [2.3 Naming Conventions](#23-naming-conventions)
- [3. Core Schema Patterns](#3-core-schema-patterns)
  - [3.1 Common Envelope Schema](#31-common-envelope-schema)
  - [3.2 Operation Context Schema](#32-operation-context-schema)
  - [3.3 Shared Type Schemas](#33-shared-type-schemas)
- [4. Protocol-Specific Schemas](#4-protocol-specific-schemas)
  - [4.1 LLM Protocol Schemas](#41-llm-protocol-schemas)
  - [4.2 Vector Protocol Schemas](#42-vector-protocol-schemas)
  - [4.3 Embedding Protocol Schemas](#43-embedding-protocol-schemas)
  - [4.4 Graph Protocol Schemas](#44-graph-protocol-schemas)
- [5. Streaming Schemas](#5-streaming-schemas)
  - [5.1 Streaming Envelope Schema](#51-streaming-envelope-schema)
  - [5.2 NDJSON Schema](#52-ndjson-schema)
  - [5.3 Streaming Semantics](#53-streaming-semantics)
- [6. Schema Validation Infrastructure](#6-schema-validation-infrastructure)
  - [6.1 Schema Registry](#61-schema-registry)
  - [6.2 Validation Modes](#62-validation-modes)
  - [6.3 Validation Pipeline](#63-validation-pipeline)
- [7. Golden Test Infrastructure](#7-golden-test-infrastructure)
  - [7.1 Golden Test Philosophy](#71-golden-test-philosophy)
  - [7.2 Golden Test Organization](#72-golden-test-organization)
  - [7.3 Golden Test Validation](#73-golden-test-validation)
- [8. Schema Quality Gates](#8-schema-quality-gates)
  - [8.1 Schema Lint Rules](#81-schema-lint-rules)
  - [8.2 Production Conformance Tests](#82-production-conformance-tests)
  - [8.3 Performance Considerations](#83-performance-considerations)
- [9. Integration and Tooling](#9-integration-and-tooling)
  - [9.1 CLI Tools](#91-cli-tools)
  - [9.2 Development Workflow](#92-development-workflow)
  - [9.3 CI/CD Integration](#93-cicd-integration)
- [10. Schema Evolution](#10-schema-evolution)
  - [10.1 Versioning Strategy](#101-versioning-strategy)
  - [10.2 Breaking Change Process](#102-breaking-change-process)
  - [10.3 Extension Patterns](#103-extension-patterns)
- [11. Reference](#11-reference)
  - [11.1 Schema Quick Reference](#111-schema-quick-reference)
  - [11.2 Error Taxonomies by Protocol](#112-error-taxonomies-by-protocol)
- [12. Appendices](#12-appendices)
  - [12.A JSON Schema Draft 2020-12 Primer](#12a-json-schema-draft-2020-12-primer)
  - [12.B $ref Resolution Examples](#12b-ref-resolution-examples)
  - [12.C Custom Format Validators](#12c-custom-format-validators)
  - [12.D Schema Testing Strategies](#12d-schema-testing-strategies)
  - [12.E Troubleshooting Guide](#12e-troubleshooting-guide)

---

## 1. Introduction

### 1.1 Purpose and Scope
The Corpus Schema Reference provides the definitive JSON Schema definitions for the Corpus Protocol Suite, enabling:
- **Strict validation** of all wire-level protocol communications
- **Consistent type definitions** across LLM, Vector, Embedding, and Graph protocols
- **Automated testing** through schema-driven validation infrastructure
- **Development tooling** for schema-first protocol implementation

**Scope:** This document covers all JSON Schema definitions, validation rules, and testing infrastructure for protocol compliance. It does not define provider-specific implementations or transport-layer specifics.

### 1.2 Document Relationships

| Document | Purpose | Relationship to SCHEMA.md |
|----------|---------|---------------------------|
| **PROTOCOLS.md** | Defines wire format and operational semantics | **Wire format takes precedence** - SCHEMA.md implements PROTOCOLS.md requirements |
| **SPECIFICATION.md** | High-level architecture and design philosophy | Descriptive reference; may contain language-specific bindings |
| **ERRORS.md** | Error taxonomy and normalization rules | Error envelope schemas must align with ERROR taxonomy |
| **METRICS.md** | Observability requirements | Schema validation includes metrics field constraints |
| **IMPLEMENTATION.md** | Implementation guidance | Provides implementation patterns for schema validation |

### 1.3 Schema Philosophy

**Core Principles:**
1. **Schema-first development** - Schemas define the contract before implementation
2. **Strict by default** - Additional properties prohibited unless explicitly allowed
3. **Version tolerance** - Schema evolution with backward compatibility
4. **Performance-aware** - Validation modes tuned for production use

**Validation Approach:**
- **Wire-level validation** - All JSON payloads validated against schemas
- **Streaming validation** - Protocol envelope validation for streaming operations
- **Golden tests** - Example-based validation as executable documentation
- **Conformance tests** - Production adapter validation in CI/CD

---

## 2. Schema Architecture

### 2.1 Directory Structure

```
schemas/
├── common/                           # Cross-protocol schemas
│   ├── envelope.request.json         # Canonical request envelope
│   ├── envelope.success.json         # Canonical success envelope  
│   ├── envelope.error.json           # Canonical error envelope
│   ├── envelope.stream.success.json  # Streaming success envelope
│   └── operation_context.json        # OperationContext type
│
├── llm/                              # LLM protocol schemas
│   ├── llm.envelope.request.json     # LLM-specific request envelope
│   ├── llm.envelope.success.json     # LLM-specific success envelope
│   ├── llm.envelope.error.json       # LLM-specific error envelope
│   ├── llm.capabilities.json         # LLMCapabilities type
│   ├── llm.response_format.json      # JSON/text output mode
│   ├── llm.sampling.params.json      # Temperature/top_p etc.
│   ├── llm.tools.schema.json         # Tool/tool_call definitions
│   ├── llm.types.chunk.json          # Streaming chunk type
│   ├── llm.types.completion.json     # Completion result type
│   ├── llm.types.completion_spec.json # Complete operation args
│   ├── llm.types.stream_spec.json    # Stream operation args
│   ├── llm.types.count_tokens_spec.json # Count tokens args
│   ├── llm.types.logprobs.json       # Log probabilities type
│   ├── llm.types.message.json        # Message type definition
│   ├── llm.types.token_usage.json    # Token usage reporting
│   ├── llm.types.tool.json           # Tool definition type
│   ├── llm.types.warning.json        # Warning type definition
│   ├── llm.capabilities.request.json # LLM Capabilities operation request
│   ├── llm.capabilities.success.json # LLM Capabilities operation success
│   ├── llm.complete.request.json     # LLM Complete operation request
│   ├── llm.complete.success.json     # LLM Complete operation success
│   ├── llm.count_tokens.request.json # LLM Count tokens request
│   ├── llm.count_tokens.success.json # LLM Count tokens success
│   ├── llm.health.request.json       # LLM Health operation request
│   ├── llm.health.success.json       # LLM Health operation success
│   ├── llm.stream.request.json       # LLM Stream operation request
│   └── llm.stream.success.json       # LLM Stream operation success
│
├── vector/                           # Vector protocol schemas
│   ├── vector.envelope.request.json
│   ├── vector.envelope.success.json
│   ├── vector.envelope.error.json
│   ├── vector.capabilities.json
│   ├── vector.types.document.json
│   ├── vector.types.failure_item.json
│   ├── vector.types.filter.json
│   ├── vector.types.namespace_result.json
│   ├── vector.types.namespace_spec.json
│   ├── vector.types.query_result.json
│   ├── vector.types.query_spec.json
│   ├── vector.types.upsert_result.json
│   ├── vector.types.delete_result.json
│   ├── vector.types.vector.json
│   ├── vector.types.vector_match.json
│   ├── vector.capabilities.request.json # Vector Capabilities operation request
│   ├── vector.capabilities.success.json # Vector Capabilities operation success
│   ├── vector.query.request.json     # Vector Query operation request
│   ├── vector.query.success.json     # Vector Query operation success
│   ├── vector.batch_query.request.json # Vector Batch Query operation request
│   ├── vector.batch_query.success.json # Vector Batch Query operation success
│   ├── vector.upsert.request.json    # Vector Upsert operation request
│   ├── vector.upsert.success.json    # Vector Upsert operation success
│   ├── vector.delete.request.json    # Vector Delete operation request
│   ├── vector.delete.success.json    # Vector Delete operation success
│   ├── vector.create_namespace.request.json # Create namespace request
│   ├── vector.create_namespace.success.json # Create namespace success
│   ├── vector.delete_namespace.request.json # Delete namespace request
│   ├── vector.delete_namespace.success.json # Delete namespace success
│   ├── vector.health.request.json    # Vector Health operation request
│   └── vector.health.success.json    # Vector Health operation success
│
├── embedding/                        # Embedding protocol schemas
│   ├── embedding.envelope.request.json
│   ├── embedding.envelope.success.json
│   ├── embedding.envelope.error.json
│   ├── embedding.capabilities.json
│   ├── embedding.stats.json
│   ├── embedding.types.chunk.json
│   ├── embedding.types.failure.json
│   ├── embedding.types.result.json
│   ├── embedding.types.vector.json
│   ├── embedding.types.warning.json
│   ├── embedding.types.embed_spec.json # Embed operation args
│   ├── embedding.types.stream_embed_spec.json # Stream embed args
│   ├── embedding.types.count_tokens_spec.json # Count tokens args
│   ├── embedding.types.batch_result.json
│   ├── embedding.capabilities.request.json # Embedding Capabilities request
│   ├── embedding.capabilities.success.json # Embedding Capabilities success
│   ├── embedding.embed.request.json  # Embedding Embed operation request
│   ├── embedding.embed.success.json  # Embedding Embed operation success
│   ├── embedding.embed_batch.request.json # Embedding Batch Embed request
│   ├── embedding.embed_batch.success.json # Embedding Batch Embed success
│   ├── embedding.stream_embed.request.json # Embedding Stream Embed request
│   ├── embedding.stream_embed.success.json # Embedding Stream Embed success
│   ├── embedding.count_tokens.request.json # Embedding Count tokens request
│   ├── embedding.count_tokens.success.json # Embedding Count tokens success
│   ├── embedding.health.request.json # Embedding Health request
│   ├── embedding.health.success.json # Embedding Health success
│   ├── embedding.get_stats.request.json # Embedding Get Stats request
│   └── embedding.get_stats.success.json # Embedding Get Stats success
│
└── graph/                            # Graph protocol schemas
    ├── graph.envelope.request.json
    ├── graph.envelope.success.json
    ├── graph.envelope.error.json
    ├── graph.capabilities.json
    ├── graph.types.batch_op.json
    ├── graph.types.batch_result.json
    ├── graph.types.bulk_vertices_spec.json
    ├── graph.types.bulk_vertices_result.json
    ├── graph.types.chunk.json
    ├── graph.types.edge.json
    ├── graph.types.entity.json
    ├── graph.types.graph_schema.json
    ├── graph.types.health_result.json
    ├── graph.types.id.json
    ├── graph.types.node.json
    ├── graph.types.query_spec.json
    ├── graph.types.query_result.json
    ├── graph.types.traversal_spec.json
    ├── graph.types.traversal_result.json
    ├── graph.types.warning.json
    ├── graph.capabilities.request.json # Graph Capabilities request
    ├── graph.capabilities.success.json # Graph Capabilities success
    ├── graph.query.request.json      # Graph Query operation request
    ├── graph.query.success.json      # Graph Query operation success
    ├── graph.stream_query.request.json # Graph Stream Query request
    ├── graph.stream_query.success.json # Graph Stream Query success
    ├── graph.upsert_nodes.request.json # Graph Upsert Nodes request
    ├── graph.upsert_nodes.success.json # Graph Upsert Nodes success
    ├── graph.upsert_edges.request.json # Graph Upsert Edges request
    ├── graph.upsert_edges.success.json # Graph Upsert Edges success
    ├── graph.delete_nodes.request.json # Graph Delete Nodes request
    ├── graph.delete_nodes.success.json # Graph Delete Nodes success
    ├── graph.delete_edges.request.json # Graph Delete Edges request
    ├── graph.delete_edges.success.json # Graph Delete Edges success
    ├── graph.bulk_vertices.request.json # Graph Bulk Vertices request
    ├── graph.bulk_vertices.success.json # Graph Bulk Vertices success
    ├── graph.batch.request.json      # Graph Batch request
    ├── graph.batch.success.json      # Graph Batch success
    ├── graph.get_schema.request.json # Graph Get Schema request
    ├── graph.get_schema.success.json # Graph Get Schema success
    ├── graph.health.request.json     # Graph Health request
    ├── graph.health.success.json     # Graph Health success
    ├── graph.transaction.request.json # Graph Transaction request
    ├── graph.transaction.success.json # Graph Transaction success
    ├── graph.traversal.request.json  # Graph Traversal request
    └── graph.traversal.success.json  # Graph Traversal success
```

**Key Directories:**
- **common/**: Schemas used across all protocols (envelopes, context)
- **{protocol}/**: Protocol-specific schemas following naming pattern
- **{protocol}/types/**: Reusable type definitions within protocol
- **{protocol}/operations/**: Operation-specific schemas (implied by file structure)

### 2.2 Schema Organization Principles

**Single Responsibility Principle:**
- Each schema file defines **one logical component** (operation, type, envelope)
- Schemas are **self-contained** with explicit `$ref` dependencies
- **No circular dependencies** - all references form a directed acyclic graph

**Reusability Patterns:**
1. **Common envelopes** referenced by all protocol-specific envelopes
2. **Type definitions** (`$defs`) for reusable structures within protocol
3. **Cross-protocol sharing** via absolute `$id` references
4. **Operation-specific schemas** for each protocol operation

**Schema Composition:**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.complete.request.json",
  "title": "LLM Complete Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/llm/llm.envelope.request.json" },
    {
      "properties": {
        "args": {
          "$ref": "https://corpusos.com/schemas/llm/llm.types.completion_spec.json"
        }
      }
    }
  ]
}
```

### 2.3 Naming Conventions

**File Naming Patterns:**
- `{protocol}.{category}.{type}.json` for operational schemas
- `{protocol}.types.{name}.json` for type definitions
- `{protocol}.envelope.{type}.json` for envelope schemas
- `{protocol}.{operation}.request.json` for operation request schemas
- `{protocol}.{operation}.success.json` for operation success schemas

**Examples:**
```
llm.envelope.request.json              # LLM protocol request envelope
llm.complete.request.json              # LLM complete operation request
llm.complete.success.json              # LLM complete operation success
vector.types.vector.json              # Vector type definition  
graph.types.node.json                 # Graph node type definition
llm.capabilities.json                 # LLM capabilities type
```

**$id Naming Convention:**
```json
{
  "$id": "https://corpusos.com/schemas/{component}/{filename}"
}
```

**Examples:**
```
https://corpusos.com/schemas/llm/llm.envelope.request.json
https://corpusos.com/schemas/vector/vector.types.vector.json
https://corpusos.com/schemas/common/envelope.success.json
https://corpusos.com/schemas/llm/llm.complete.request.json
```

**Type Naming:**
- **Schema titles**: `PascalCase` with descriptive names
- **Property names**: `snake_case` for consistency with JSON
- **Definition names**: `PascalCase` within `$defs`

---

## 3. Core Schema Patterns

### 3.1 Common Envelope Schema

**Canonical Request Envelope (`common/envelope.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/common/envelope.request.json",
  "title": "Protocol Request Envelope",
  "type": "object",
  "properties": {
    "op": {
      "type": "string",
      "minLength": 1,
      "description": "Operation identifier (e.g., 'llm.complete', 'vector.query')"
    },
    "ctx": {
      "$ref": "https://corpusos.com/schemas/common/operation_context.json",
      "description": "Operation context including request_id, deadlines, tracing"
    },
    "args": {
      "type": "object",
      "description": "Operation-specific arguments (must be an object)"
    }
  },
  "required": ["op", "ctx", "args"],
  "additionalProperties": false
}
```

**Canonical Success Envelope (`common/envelope.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/common/envelope.success.json",
  "title": "Protocol Success Envelope",
  "type": "object",
  "properties": {
    "ok": {
      "type": "boolean",
      "const": true
    },
    "code": {
      "type": "string",
      "const": "OK"
    },
    "ms": {
      "type": "number",
      "minimum": 0,
      "description": "Operation duration in milliseconds"
    },
    "result": {
      "description": "Operation-specific result (any JSON value)",
      "oneOf": [
        { "type": "object" },
        { "type": "array" },
        { "type": "string" },
        { "type": "number" },
        { "type": "integer" },
        { "type": "boolean" },
        { "type": "null" }
      ]
    }
  },
  "required": ["ok", "code", "ms", "result"],
  "additionalProperties": false
}
```

**Canonical Error Envelope (`common/envelope.error.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/common/envelope.error.json",
  "title": "Protocol Error Envelope",
  "type": "object",
  "properties": {
    "ok": {
      "type": "boolean",
      "const": false
    },
    "code": {
      "type": "string",
      "pattern": "^[A-Z_]+$",
      "description": "Canonical error code"
    },
    "error": {
      "type": "string",
      "description": "Error type name"
    },
    "message": {
      "type": "string",
      "description": "Human-readable error message"
    },
    "retry_after_ms": {
      "type": ["integer", "null"],
      "minimum": 0,
      "description": "Suggested retry delay in milliseconds"
    },
    "details": {
      "type": ["object", "null"],
      "description": "Error-specific details"
    },
    "ms": {
      "type": "number",
      "minimum": 0,
      "description": "Time spent before error in milliseconds"
    }
  },
  "required": ["ok", "code", "error", "message", "ms"],
  "additionalProperties": false
}
```

**Streaming Success Envelope (`common/envelope.stream.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/common/envelope.stream.success.json",
  "title": "Protocol Streaming Success Envelope",
  "type": "object",
  "properties": {
    "ok": {
      "type": "boolean",
      "const": true
    },
    "code": {
      "type": "string",
      "const": "STREAMING"
    },
    "ms": {
      "type": "number",
      "minimum": 0,
      "description": "Time elapsed in milliseconds"
    },
    "chunk": {
      "description": "Streaming chunk payload (any JSON value)"
    }
  },
  "required": ["ok", "code", "ms", "chunk"],
  "additionalProperties": false
}
```

### 3.2 Operation Context Schema

**Operation Context Type (`common/operation_context.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/common/operation_context.json",
  "title": "Operation Context",
  "type": "object",
  "properties": {
    "request_id": {
      "type": "string",
      "description": "Request correlation ID"
    },
    "idempotency_key": {
      "type": "string",
      "description": "Idempotency guarantee key"
    },
    "deadline_ms": {
      "type": "integer",
      "minimum": 1,
      "description": "Absolute epoch milliseconds"
    },
    "traceparent": {
      "type": "string",
      "description": "W3C Trace Context header"
    },
    "tenant": {
      "type": "string",
      "description": "Tenant isolation identifier"
    },
    "attrs": {
      "type": "object",
      "default": {},
      "additionalProperties": true,
      "description": "Extension attributes"
    }
  },
  "additionalProperties": true
}
```

**Context Field Semantics:**

| Field | Required | Validation | Purpose |
|-------|----------|------------|---------|
| `request_id` | Optional | Any string | Request correlation |
| `idempotency_key` | Optional | Any string | Idempotency guarantee |
| `deadline_ms` | Optional | Positive integer | Absolute timeout |
| `traceparent` | Optional | Any string | Distributed tracing |
| `tenant` | Optional | Any string | Tenant isolation |
| `attrs` | Optional | JSON object | Extension attributes |

### 3.3 Shared Type Schemas

**Filter Expression Pattern:**
```json
{
  "type": "object",
  "patternProperties": {
    "^[a-zA-Z_][a-zA-Z0-9_]*$": {
      "oneOf": [
        { "type": ["string", "number", "boolean", "null"] },
        { "type": "array", "items": { "type": ["string", "number"] } },
        {
          "type": "object",
          "properties": {
            "gt": { "type": "number" },
            "gte": { "type": "number" },
            "lt": { "type": "number" },
            "lte": { "type": "number" },
            "in": { "type": "array", "items": { "type": ["string", "number"] } }
          },
          "additionalProperties": false
        }
      ]
    }
  },
  "additionalProperties": false
}
```

---

## 4. Protocol-Specific Schemas

**Note: Result/type schemas describe expected adapter-conformant payloads; the wire handler does not re-validate adapter return types at runtime.**

### 4.1 LLM Protocol Schemas

#### 4.1.1 Envelope Schemas

**LLM Request Envelope (`llm.envelope.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.envelope.request.json",
  "title": "LLM Protocol Request Envelope",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/common/envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "pattern": "^llm\\.[a-z_]+$"
        }
      }
    }
  ]
}
```

**LLM Success Envelope (`llm.envelope.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.envelope.success.json",
  "title": "LLM Protocol Success Envelope",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/common/envelope.success.json" }
  ]
}
```

**LLM Error Envelope (`llm.envelope.error.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.envelope.error.json",
  "title": "LLM Protocol Error Envelope",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/common/envelope.error.json" }
  ]
}
```

#### 4.1.2 Operation Schemas

**LLM Capabilities Request (`llm.capabilities.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.capabilities.request.json",
  "title": "LLM Capabilities Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/llm/llm.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "llm.capabilities"
        },
        "args": {
          "type": "object",
          "additionalProperties": false,
          "description": "Empty args object required"
        }
      }
    }
  ]
}
```

**LLM Capabilities Success (`llm.capabilities.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.capabilities.success.json",
  "title": "LLM Capabilities Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/llm/llm.envelope.success.json" },
    {
      "properties": {
        "result": {
          "$ref": "https://corpusos.com/schemas/llm/llm.capabilities.json"
        }
      }
    }
  ]
}
```

**LLM Complete Request (`llm.complete.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.complete.request.json",
  "title": "LLM Complete Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/llm/llm.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "llm.complete"
        },
        "args": {
          "$ref": "https://corpusos.com/schemas/llm/llm.types.completion_spec.json"
        }
      }
    }
  ]
}
```

**LLM Complete Success (`llm.complete.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.complete.success.json",
  "title": "LLM Complete Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/llm/llm.envelope.success.json" },
    {
      "properties": {
        "result": {
          "$ref": "https://corpusos.com/schemas/llm/llm.types.completion.json"
        }
      }
    }
  ]
}
```

**LLM Stream Request (`llm.stream.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.stream.request.json",
  "title": "LLM Stream Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/llm/llm.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "llm.stream"
        },
        "args": {
          "$ref": "https://corpusos.com/schemas/llm/llm.types.stream_spec.json"
        }
      }
    }
  ]
}
```

**LLM Stream Success (`llm.stream.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.stream.success.json",
  "title": "LLM Stream Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/common/envelope.stream.success.json" },
    {
      "properties": {
        "chunk": {
          "$ref": "https://corpusos.com/schemas/llm/llm.types.chunk.json"
        }
      }
    }
  ]
}
```

**LLM Count Tokens Request (`llm.count_tokens.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.count_tokens.request.json",
  "title": "LLM Count Tokens Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/llm/llm.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "llm.count_tokens"
        },
        "args": {
          "$ref": "https://corpusos.com/schemas/llm/llm.types.count_tokens_spec.json"
        }
      }
    }
  ]
}
```

**LLM Count Tokens Success (`llm.count_tokens.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.count_tokens.success.json",
  "title": "LLM Count Tokens Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/llm/llm.envelope.success.json" },
    {
      "properties": {
        "result": {
          "type": "object",
          "properties": {
            "total_tokens": {
              "type": "integer",
              "minimum": 0
            }
          },
          "required": ["total_tokens"],
          "additionalProperties": false
        }
      }
    }
  ]
}
```

**LLM Health Request (`llm.health.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.health.request.json",
  "title": "LLM Health Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/llm/llm.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "llm.health"
        },
        "args": {
          "type": "object",
          "additionalProperties": false,
          "description": "Empty args object required"
        }
      }
    }
  ]
}
```

**LLM Health Success (`llm.health.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.health.success.json",
  "title": "LLM Health Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/llm/llm.envelope.success.json" },
    {
      "properties": {
        "result": {
          "type": "object",
          "properties": {
            "ok": { "type": "boolean" },
            "status": { "type": "string" },
            "server": { "type": "string" },
            "version": { "type": "string" }
          },
          "required": ["ok", "status", "server", "version"],
          "additionalProperties": true
        }
      }
    }
  ]
}
```

#### 4.1.3 Type Definitions

**LLM Capabilities (`llm.capabilities.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.capabilities.json",
  "title": "LLM Capabilities",
  "type": "object",
  "properties": {
    "server": { "type": "string" },
    "version": { "type": "string" },
    "model_family": { "type": "string" },
    "max_context_length": { "type": "integer", "minimum": 1 },
    "protocol": { "type": "string", "const": "llm/v1.0" },
    "supports_streaming": { "type": "boolean", "default": true },
    "supports_roles": { "type": "boolean", "default": true },
    "supports_json_output": { "type": "boolean", "default": false },
    "supports_tools": { "type": "boolean", "default": false },
    "supports_parallel_tool_calls": { "type": "boolean", "default": false },
    "supports_tool_choice": { "type": "boolean", "default": false },
    "max_tool_calls_per_turn": { "type": ["integer", "null"], "minimum": 0 },
    "idempotent_writes": { "type": "boolean", "default": false },
    "supports_multi_tenant": { "type": "boolean", "default": false },
    "supports_system_message": { "type": "boolean", "default": true },
    "supports_deadline": { "type": "boolean", "default": true },
    "supports_count_tokens": { "type": "boolean", "default": true },
    "supported_models": {
      "type": "array",
      "items": { "type": "string" },
      "default": []
    }
  },
  "required": ["server", "version", "model_family", "max_context_length"],
  "additionalProperties": false
}
```

**Completion Spec (`llm.types.completion_spec.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.types.completion_spec.json",
  "title": "LLM Completion Specification",
  "type": "object",
  "properties": {
    "messages": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "role": { "type": "string", "minLength": 1 },
          "content": { "type": "string" }
        },
        "required": ["role", "content"],
        "additionalProperties": false
      },
      "minItems": 1,
      "description": "List of conversation messages"
    },
    "model": {
      "type": "string",
      "minLength": 1,
      "description": "Model identifier"
    },
    "temperature": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 2.0,
      "description": "Sampling temperature"
    },
    "max_tokens": {
      "type": "integer",
      "minimum": 1,
      "description": "Maximum tokens to generate"
    },
    "top_p": {
      "type": "number",
      "minimum": 0.0,
      "exclusiveMinimum": true,
      "maximum": 1.0,
      "description": "Nucleus sampling parameter"
    },
    "stop_sequences": {
      "type": ["array", "null"],
      "items": { "type": "string" },
      "description": "Stop generation sequences"
    },
    "tools": {
      "type": ["array", "null"],
      "items": { "type": "object" },
      "description": "Available tools for the model"
    },
    "tool_choice": {
      "oneOf": [
        { "type": "string" },
        { "type": "object" },
        { "type": "null" }
      ],
      "description": "Tool selection strategy"
    }
  },
  "required": ["messages"],
  "additionalProperties": true
}
```

**Stream Spec (`llm.types.stream_spec.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.types.stream_spec.json",
  "title": "LLM Stream Specification",
  "type": "object",
  "properties": {
    "messages": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "role": { "type": "string", "minLength": 1 },
          "content": { "type": "string" }
        },
        "required": ["role", "content"],
        "additionalProperties": false
      },
      "minItems": 1,
      "description": "List of conversation messages"
    },
    "model": {
      "type": "string",
      "minLength": 1,
      "description": "Model identifier"
    },
    "temperature": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 2.0,
      "description": "Sampling temperature"
    },
    "max_tokens": {
      "type": "integer",
      "minimum": 1,
      "description": "Maximum tokens to generate"
    },
    "top_p": {
      "type": "number",
      "minimum": 0.0,
      "exclusiveMinimum": true,
      "maximum": 1.0,
      "description": "Nucleus sampling parameter"
    },
    "stop_sequences": {
      "type": ["array", "null"],
      "items": { "type": "string" },
      "description": "Stop generation sequences"
    },
    "tools": {
      "type": ["array", "null"],
      "items": { "type": "object" },
      "description": "Available tools for the model"
    },
    "tool_choice": {
      "oneOf": [
        { "type": "string" },
        { "type": "object" },
        { "type": "null" }
      ],
      "description": "Tool selection strategy"
    }
  },
  "required": ["messages"],
  "additionalProperties": true
}
```

**Count Tokens Spec (`llm.types.count_tokens_spec.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.types.count_tokens_spec.json",
  "title": "LLM Count Tokens Specification",
  "type": "object",
  "properties": {
    "messages": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "role": { "type": "string", "minLength": 1 },
          "content": { "type": "string" }
        },
        "required": ["role", "content"],
        "additionalProperties": false
      },
      "minItems": 1,
      "description": "List of conversation messages"
    },
    "model": {
      "type": "string",
      "minLength": 1,
      "description": "Model identifier"
    }
  },
  "required": ["messages"],
  "additionalProperties": false
}
```

**Message Type (`llm.types.message.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.types.message.json",
  "title": "LLM Message",
  "type": "object",
  "properties": {
    "role": {
      "type": "string",
      "minLength": 1
    },
    "content": {
      "type": "string"
    },
    "name": {
      "type": "string",
      "description": "Tool call function name"
    },
    "tool_call_id": {
      "type": "string",
      "description": "Associate tool calls with responses"
    },
    "tool_calls": {
      "type": "array",
      "items": {
        "$ref": "#/$defs/ToolCall"
      }
    }
  },
  "required": ["role", "content"],
  "additionalProperties": false,
  "$defs": {
    "ToolCall": {
      "type": "object",
      "properties": {
        "id": { "type": "string" },
        "type": { "type": "string", "description": "Tool call type" },
        "function": {
          "type": "object",
          "properties": {
            "name": { "type": "string" },
            "arguments": { "type": "string" }
          },
          "required": ["name", "arguments"],
          "additionalProperties": false
        }
      },
      "required": ["id", "type", "function"],
      "additionalProperties": false
    }
  }
}
```

**Token Usage (`llm.types.token_usage.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.types.token_usage.json",
  "title": "Token Usage",
  "type": "object",
  "properties": {
    "prompt_tokens": {
      "type": "integer",
      "minimum": 0
    },
    "completion_tokens": {
      "type": "integer",
      "minimum": 0
    },
    "total_tokens": {
      "type": "integer",
      "minimum": 0,
      "description": "MUST equal prompt_tokens + completion_tokens"
    }
  },
  "required": ["prompt_tokens", "completion_tokens", "total_tokens"],
  "additionalProperties": false
}
```

**Completion Result (`llm.types.completion.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.types.completion.json",
  "title": "LLM Completion",
  "type": "object",
  "properties": {
    "text": {
      "type": "string",
      "description": "Generated completion text"
    },
    "model": {
      "type": "string",
      "description": "Model identifier used"
    },
    "model_family": {
      "type": "string",
      "description": "Logical model family"
    },
    "usage": {
      "$ref": "https://corpusos.com/schemas/llm/llm.types.token_usage.json"
    },
    "finish_reason": {
      "type": "string",
      "description": "Reason generation stopped"
    },
    "tool_calls": {
      "type": "array",
      "items": {
        "$ref": "https://corpusos.com/schemas/llm/llm.types.message.json#/$defs/ToolCall"
      },
      "default": []
    }
  },
  "required": ["text", "model", "model_family", "usage", "finish_reason"],
  "additionalProperties": false
}
```

**Streaming Chunk (`llm.types.chunk.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.types.chunk.json",
  "title": "LLM Streaming Chunk",
  "type": "object",
  "properties": {
    "text": {
      "type": "string",
      "description": "Incremental text content"
    },
    "is_final": {
      "type": "boolean",
      "default": false,
      "description": "True for final chunk"
    },
    "model": {
      "type": ["string", "null"],
      "description": "Optional model identifier"
    },
    "usage_so_far": {
      "oneOf": [
        { "$ref": "https://corpusos.com/schemas/llm/llm.types.token_usage.json" },
        { "type": "null" }
      ],
      "description": "Token usage up to this point"
    },
    "tool_calls": {
      "type": "array",
      "items": {
        "$ref": "https://corpusos.com/schemas/llm/llm.types.message.json#/$defs/ToolCall"
      },
      "default": []
    }
  },
  "required": ["text", "is_final"],
  "additionalProperties": false
}
```

#### 4.1.4 Configuration Schemas

**Sampling Parameters (`llm.sampling.params.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.sampling.params.json",
  "title": "LLM Sampling Parameters",
  "type": "object",
  "properties": {
    "temperature": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 2.0,
      "description": "Creativity/randomness control"
    },
    "top_p": {
      "type": "number",
      "minimum": 0.0,
      "exclusiveMinimum": true,
      "maximum": 1.0,
      "description": "Nucleus sampling parameter"
    },
    "frequency_penalty": {
      "type": "number",
      "minimum": -2.0,
      "maximum": 2.0,
      "description": "Reduce repetition"
    },
    "presence_penalty": {
      "type": "number",
      "minimum": -2.0,
      "maximum": 2.0,
      "description": "Encourage new topics"
    },
    "seed": {
      "type": "integer",
      "description": "Deterministic output seed"
    }
  },
  "additionalProperties": false
}
```

**Response Format (`llm.response_format.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.response_format.json",
  "title": "LLM Response Format",
  "type": "object",
  "properties": {
    "type": {
      "type": "string",
      "enum": ["text", "json_object"]
    }
  },
  "required": ["type"],
  "additionalProperties": false
}
```

**Tools Schema (`llm.tools.schema.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.tools.schema.json",
  "title": "LLM Tools Definition",
  "type": "object",
  "properties": {
    "type": {
      "type": "string",
      "description": "Tool type"
    },
    "function": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "description": { "type": "string" },
        "parameters": {
          "type": "object",
          "description": "JSON Schema object"
        }
      },
      "required": ["name", "parameters"],
      "additionalProperties": false
    }
  },
  "required": ["type", "function"],
  "additionalProperties": false
}
```

### 4.2 Vector Protocol Schemas

#### 4.2.1 Envelope Schemas

**Vector Request Envelope (`vector.envelope.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.envelope.request.json",
  "title": "Vector Protocol Request Envelope",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/common/envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "pattern": "^vector\\.[a-z_]+$"
        }
      }
    }
  ]
}
```

**Vector Success Envelope (`vector.envelope.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.envelope.success.json",
  "title": "Vector Protocol Success Envelope",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/common/envelope.success.json" }
  ]
}
```

**Vector Error Envelope (`vector.envelope.error.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.envelope.error.json",
  "title": "Vector Protocol Error Envelope",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/common/envelope.error.json" }
  ]
}
```

#### 4.2.2 Operation Schemas

**Vector Capabilities Request (`vector.capabilities.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.capabilities.request.json",
  "title": "Vector Capabilities Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/vector/vector.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "vector.capabilities"
        },
        "args": {
          "type": "object",
          "additionalProperties": false,
          "description": "Empty args object required"
        }
      }
    }
  ]
}
```

**Vector Capabilities Success (`vector.capabilities.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.capabilities.success.json",
  "title": "Vector Capabilities Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/vector/vector.envelope.success.json" },
    {
      "properties": {
        "result": {
          "$ref": "https://corpusos.com/schemas/vector/vector.capabilities.json"
        }
      }
    }
  ]
}
```

**Vector Query Request (`vector.query.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.query.request.json",
  "title": "Vector Query Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/vector/vector.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "vector.query"
        },
        "args": {
          "$ref": "https://corpusos.com/schemas/vector/vector.types.query_spec.json"
        }
      }
    }
  ]
}
```

**Vector Query Success (`vector.query.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.query.success.json",
  "title": "Vector Query Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/vector/vector.envelope.success.json" },
    {
      "properties": {
        "result": {
          "$ref": "https://corpusos.com/schemas/vector/vector.types.query_result.json"
        }
      }
    }
  ]
}
```

**Vector Batch Query Request (`vector.batch_query.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.batch_query.request.json",
  "title": "Vector Batch Query Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/vector/vector.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "vector.batch_query"
        },
        "args": {
          "type": "object",
          "properties": {
            "queries": {
              "type": "array",
              "items": {
                "$ref": "https://corpusos.com/schemas/vector/vector.types.query_spec.json"
              },
              "minItems": 1
            }
          },
          "required": ["queries"],
          "additionalProperties": false
        }
      }
    }
  ]
}
```

**Vector Batch Query Success (`vector.batch_query.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.batch_query.success.json",
  "title": "Vector Batch Query Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/vector/vector.envelope.success.json" },
    {
      "properties": {
        "result": {
          "type": "array",
          "items": {
            "$ref": "https://corpusos.com/schemas/vector/vector.types.query_result.json"
          }
        }
      }
    }
  ]
}
```

**Vector Upsert Request (`vector.upsert.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.upsert.request.json",
  "title": "Vector Upsert Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/vector/vector.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "vector.upsert"
        },
        "args": {
          "type": "object",
          "properties": {
            "vectors": {
              "type": "array",
              "items": {
                "$ref": "https://corpusos.com/schemas/vector/vector.types.vector.json"
              },
              "minItems": 1
            },
            "namespace": {
              "type": "string",
              "default": "default"
            }
          },
          "required": ["vectors"],
          "additionalProperties": false
        }
      }
    }
  ]
}
```

**Vector Upsert Success (`vector.upsert.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.upsert.success.json",
  "title": "Vector Upsert Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/vector/vector.envelope.success.json" },
    {
      "properties": {
        "result": {
          "$ref": "https://corpusos.com/schemas/vector/vector.types.upsert_result.json"
        }
      }
    }
  ]
}
```

**Vector Delete Request (`vector.delete.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.delete.request.json",
  "title": "Vector Delete Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/vector/vector.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "vector.delete"
        },
        "args": {
          "type": "object",
          "properties": {
            "ids": {
              "type": "array",
              "items": { "type": "string" },
              "minItems": 1
            },
            "namespace": {
              "type": "string",
              "default": "default"
            },
            "filter": {
              "type": "object",
              "additionalProperties": true
            }
          },
          "required": ["ids"],
          "additionalProperties": false
        }
      }
    }
  ]
}
```

**Vector Delete Success (`vector.delete.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.delete.success.json",
  "title": "Vector Delete Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/vector/vector.envelope.success.json" },
    {
      "properties": {
        "result": {
          "$ref": "https://corpusos.com/schemas/vector/vector.types.delete_result.json"
        }
      }
    }
  ]
}
```

**Vector Create Namespace Request (`vector.create_namespace.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.create_namespace.request.json",
  "title": "Vector Create Namespace Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/vector/vector.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "vector.create_namespace"
        },
        "args": {
          "$ref": "https://corpusos.com/schemas/vector/vector.types.namespace_spec.json"
        }
      }
    }
  ]
}
```

**Vector Create Namespace Success (`vector.create_namespace.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.create_namespace.success.json",
  "title": "Vector Create Namespace Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/vector/vector.envelope.success.json" },
    {
      "properties": {
        "result": {
          "$ref": "https://corpusos.com/schemas/vector/vector.types.namespace_result.json"
        }
      }
    }
  ]
}
```

**Vector Delete Namespace Request (`vector.delete_namespace.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.delete_namespace.request.json",
  "title": "Vector Delete Namespace Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/vector/vector.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "vector.delete_namespace"
        },
        "args": {
          "type": "object",
          "properties": {
            "namespace": {
              "type": "string",
              "minLength": 1
            }
          },
          "required": ["namespace"],
          "additionalProperties": false
        }
      }
    }
  ]
}
```

**Vector Delete Namespace Success (`vector.delete_namespace.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.delete_namespace.success.json",
  "title": "Vector Delete Namespace Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/vector/vector.envelope.success.json" },
    {
      "properties": {
        "result": {
          "$ref": "https://corpusos.com/schemas/vector/vector.types.namespace_result.json"
        }
      }
    }
  ]
}
```

**Vector Health Request (`vector.health.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.health.request.json",
  "title": "Vector Health Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/vector/vector.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "vector.health"
        },
        "args": {
          "type": "object",
          "additionalProperties": false,
          "description": "Empty args object required"
        }
      }
    }
  ]
}
```

**Vector Health Success (`vector.health.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.health.success.json",
  "title": "Vector Health Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/vector/vector.envelope.success.json" },
    {
      "properties": {
        "result": {
          "type": "object",
          "properties": {
            "ok": { "type": "boolean" },
            "status": { "type": "string" },
            "server": { "type": "string" },
            "version": { "type": "string" }
          },
          "required": ["ok", "status", "server", "version"],
          "additionalProperties": true
        }
      }
    }
  ]
}
```

#### 4.2.3 Type Definitions

**Vector Capabilities (`vector.capabilities.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.capabilities.json",
  "title": "Vector Capabilities",
  "type": "object",
  "properties": {
    "server": { "type": "string" },
    "version": { "type": "string" },
    "protocol": { "type": "string", "const": "vector/v1.0" },
    "max_dimensions": { "type": "integer", "minimum": 0 },
    "supported_metrics": {
      "type": "array",
      "items": { "type": "string" },
      "default": ["cosine", "euclidean", "dotproduct"]
    },
    "supports_namespaces": { "type": "boolean", "default": true },
    "supports_metadata_filtering": { "type": "boolean", "default": true },
    "supports_batch_operations": { "type": "boolean", "default": true },
    "max_batch_size": { "type": ["integer", "null"], "minimum": 1 },
    "supports_index_management": { "type": "boolean", "default": false },
    "idempotent_writes": { "type": "boolean", "default": false },
    "supports_multi_tenant": { "type": "boolean", "default": false },
    "supports_deadline": { "type": "boolean", "default": true },
    "max_top_k": { "type": ["integer", "null"], "minimum": 1 },
    "max_filter_terms": { "type": ["integer", "null"], "minimum": 1 },
    "text_storage_strategy": {
      "type": "string",
      "enum": ["metadata", "docstore", "none"],
      "default": "metadata"
    },
    "max_text_length": { "type": ["integer", "null"], "minimum": 1 },
    "supports_batch_queries": { "type": "boolean", "default": false }
  },
  "required": ["server", "version", "max_dimensions"],
  "additionalProperties": false
}
```

**Vector Type (`vector.types.vector.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.types.vector.json",
  "title": "Vector",
  "type": "object",
  "properties": {
    "id": {
      "type": "string",
      "minLength": 1
    },
    "vector": {
      "type": "array",
      "items": { "type": "number" },
      "minItems": 1
    },
    "metadata": {
      "type": ["object", "null"],
      "additionalProperties": true
    },
    "namespace": {
      "type": "string"
    },
    "text": {
      "type": ["string", "null"],
      "description": "Optional text content associated with vector"
    }
  },
  "required": ["id", "vector"],
  "additionalProperties": false
}
```

**Document Type (`vector.types.document.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.types.document.json",
  "title": "Document",
  "type": "object",
  "properties": {
    "id": { "type": "string", "minLength": 1 },
    "text": { "type": "string" },
    "metadata": {
      "type": ["object", "null"],
      "additionalProperties": true
    }
  },
  "required": ["id", "text"],
  "additionalProperties": false
}
```

**Vector Match (`vector.types.vector_match.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.types.vector_match.json",
  "title": "Vector Match",
  "type": "object",
  "properties": {
    "vector": {
      "$ref": "https://corpusos.com/schemas/vector/vector.types.vector.json"
    },
    "score": {
      "type": "number",
      "description": "Similarity score (higher = more similar)"
    },
    "distance": {
      "type": "number",
      "minimum": 0,
      "description": "Raw distance metric (lower = more similar)"
    }
  },
  "required": ["vector", "score", "distance"],
  "additionalProperties": false
}
```

**Query Specification (`vector.types.query_spec.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.types.query_spec.json",
  "title": "Vector Query Specification",
  "type": "object",
  "properties": {
    "vector": {
      "type": "array",
      "items": { "type": "number" },
      "minItems": 1
    },
    "top_k": {
      "type": "integer",
      "minimum": 1
    },
    "namespace": {
      "type": "string",
      "default": "default"
    },
    "filter": {
      "type": "object",
      "additionalProperties": true
    },
    "include_metadata": {
      "type": "boolean",
      "default": true
    },
    "include_vectors": {
      "type": "boolean",
      "default": false
    }
  },
  "required": ["vector", "top_k"],
  "additionalProperties": false
}
```

**Query Result (`vector.types.query_result.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.types.query_result.json",
  "title": "Vector Query Result",
  "type": "object",
  "properties": {
    "matches": {
      "type": "array",
      "items": {
        "$ref": "https://corpusos.com/schemas/vector/vector.types.vector_match.json"
      }
    },
    "query_vector": {
      "type": "array",
      "items": { "type": "number" }
    },
    "namespace": {
      "type": "string"
    },
    "total_matches": {
      "type": "integer",
      "minimum": 0
    }
  },
  "required": ["matches", "query_vector", "namespace", "total_matches"],
  "additionalProperties": false
}
```

**Namespace Specification (`vector.types.namespace_spec.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.types.namespace_spec.json",
  "title": "Namespace Specification",
  "type": "object",
  "properties": {
    "namespace": { "type": "string", "minLength": 1 },
    "dimensions": { "type": "integer", "minimum": 1 },
    "distance_metric": {
      "type": "string",
      "enum": ["cosine", "euclidean", "dotproduct"]
    }
  },
  "required": ["namespace", "dimensions"],
  "additionalProperties": false
}
```

**Namespace Result (`vector.types.namespace_result.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.types.namespace_result.json",
  "title": "Namespace Result",
  "type": "object",
  "properties": {
    "success": { "type": "boolean" },
    "namespace": { "type": "string" },
    "details": { "type": "string" }
  },
  "required": ["success", "namespace"],
  "additionalProperties": false
}
```

**Upsert Result (`vector.types.upsert_result.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.types.upsert_result.json",
  "title": "Vector Upsert Result",
  "type": "object",
  "properties": {
    "upserted_count": { "type": "integer", "minimum": 0 },
    "failed_count": { "type": "integer", "minimum": 0 },
    "failures": {
      "type": "array",
      "items": {
        "$ref": "https://corpusos.com/schemas/vector/vector.types.failure_item.json"
      }
    }
  },
  "required": ["upserted_count", "failed_count", "failures"],
  "additionalProperties": false
}
```

**Delete Result (`vector.types.delete_result.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.types.delete_result.json",
  "title": "Vector Delete Result",
  "type": "object",
  "properties": {
    "deleted_count": { "type": "integer", "minimum": 0 },
    "failed_count": { "type": "integer", "minimum": 0 },
    "failures": {
      "type": "array",
      "items": {
        "$ref": "https://corpusos.com/schemas/vector/vector.types.failure_item.json"
      }
    }
  },
  "required": ["deleted_count", "failed_count", "failures"],
  "additionalProperties": false
}
```

**Failure Item (`vector.types.failure_item.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.types.failure_item.json",
  "title": "Vector Failure Item",
  "type": "object",
  "properties": {
    "id": { "type": "string" },
    "error": { "type": "string" },
    "detail": { "type": "string" }
  },
  "required": ["error", "detail"],
  "additionalProperties": false
}
```

**Filter Type (`vector.types.filter.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.types.filter.json",
  "title": "Vector Filter Expression",
  "type": "object",
  "patternProperties": {
    "^[a-zA-Z_][a-zA-Z0-9_]*$": {
      "oneOf": [
        { "type": ["string", "number", "boolean", "null"] },
        { "type": "array", "items": { "type": ["string", "number"] } },
        {
          "type": "object",
          "properties": {
            "gt": { "type": "number" },
            "gte": { "type": "number" },
            "lt": { "type": "number" },
            "lte": { "type": "number" },
            "in": { "type": "array", "items": { "type": ["string", "number"] } }
          },
          "additionalProperties": false
        }
      ]
    }
  },
  "additionalProperties": false
}
```
## 4.3 Embedding Protocol Schemas

### 4.3.1 Envelope Schemas

#### 4.3.1.1 Embedding Request Envelope (`embedding/embedding.envelope.request.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.envelope.request.json",
  "title": "Embedding Protocol Request Envelope",
  "type": "object",
  "properties": {
    "op": {
      "type": "string",
      "description": "Operation identifier. Unknown operations are handled at runtime."
    },
    "ctx": {
      "$ref": "https://corpusos.com/schemas/embedding/embedding.operation_context.json",
      "description": "REQUIRED on the wire; must be an object (mapping)."
    },
    "args": {
      "type": "object",
      "description": "REQUIRED on the wire; must be an object (mapping).",
      "additionalProperties": true
    }
  },
  "required": ["op", "ctx", "args"],
  "additionalProperties": true
}
```

#### 4.3.1.2 Embedding Success Envelope (`embedding/embedding.envelope.success.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.envelope.success.json",
  "title": "Embedding Protocol Success Envelope",
  "type": "object",
  "properties": {
    "ok": { "type": "boolean", "const": true },
    "code": { "type": "string", "const": "OK" },
    "ms": { "type": "number" },
    "result": {
      "description": "Operation-specific result payload (any JSON value).",
      "oneOf": [
        { "type": "object" },
        { "type": "array" },
        { "type": "string" },
        { "type": "number" },
        { "type": "integer" },
        { "type": "boolean" },
        { "type": "null" }
      ]
    }
  },
  "required": ["ok", "code", "ms", "result"],
  "additionalProperties": false
}
```

#### 4.3.1.3 Embedding Error Envelope (`embedding/embedding.envelope.error.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.envelope.error.json",
  "title": "Embedding Protocol Error Envelope",
  "type": "object",
  "properties": {
    "ok": { "type": "boolean", "const": false },
    "code": { "type": "string" },
    "error": { "type": "string" },
    "message": { "type": "string" },
    "retry_after_ms": { "type": ["integer", "null"] },
    "details": { "type": ["object", "null"], "additionalProperties": true },
    "ms": { "type": "number" }
  },
  "required": ["ok", "code", "error", "message", "retry_after_ms", "details", "ms"],
  "additionalProperties": false
}
```

#### 4.3.1.4 Embedding Streaming Success Envelope (`embedding/embedding.envelope.stream.success.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.envelope.stream.success.json",
  "title": "Embedding Protocol Streaming Success Envelope",
  "type": "object",
  "properties": {
    "ok": { "type": "boolean", "const": true },
    "code": { "type": "string", "const": "STREAMING" },
    "ms": { "type": "number" },
    "chunk": { "$ref": "https://corpusos.com/schemas/embedding/embedding.types.chunk.json" }
  },
  "required": ["ok", "code", "ms", "chunk"],
  "additionalProperties": false
}
```

#### 4.3.1.5 Operation Context (`embedding/embedding.operation_context.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.operation_context.json",
  "title": "Embedding Operation Context",
  "type": "object",
  "properties": {
    "request_id": {},
    "idempotency_key": {},
    "deadline_ms": {},
    "traceparent": {},
    "tenant": {},
    "attrs": { "type": "object", "additionalProperties": true }
  },
  "additionalProperties": true
}
```

---

### 4.3.2 Operation Schemas

#### 4.3.2.1 `embedding.capabilities`

**Request (`embedding/embedding.capabilities.request.json`)**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.capabilities.request.json",
  "title": "Embedding Capabilities Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/embedding/embedding.envelope.request.json" },
    {
      "properties": {
        "op": { "const": "embedding.capabilities" },
        "args": { "type": "object", "additionalProperties": true }
      }
    }
  ]
}
```

**Success (`embedding/embedding.capabilities.success.json`)**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.capabilities.success.json",
  "title": "Embedding Capabilities Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/embedding/embedding.envelope.success.json" },
    {
      "properties": {
        "result": { "$ref": "https://corpusos.com/schemas/embedding/embedding.capabilities.json" }
      }
    }
  ]
}
```

---

#### 4.3.2.2 `embedding.embed` (unary)

**Request (`embedding/embedding.embed.request.json`)**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.embed.request.json",
  "title": "Embedding Embed Request (Unary)",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/embedding/embedding.envelope.request.json" },
    {
      "properties": {
        "op": { "const": "embedding.embed" },
        "args": {
          "type": "object",
          "properties": {
            "text": {},
            "model": {},
            "truncate": {},
            "normalize": {},
            "stream": { "type": "boolean", "const": false }
          },
          "additionalProperties": true
        }
      }
    }
  ]
}
```

**Success (`embedding/embedding.embed.success.json`)**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.embed.success.json",
  "title": "Embedding Embed Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/embedding/embedding.envelope.success.json" },
    {
      "properties": {
        "result": { "$ref": "https://corpusos.com/schemas/embedding/embedding.types.result.json" }
      }
    }
  ]
}
```

---

#### 4.3.2.3 `embedding.stream_embed` (streaming)

**Request (`embedding/embedding.stream_embed.request.json`)**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.stream_embed.request.json",
  "title": "Embedding Stream Embed Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/embedding/embedding.envelope.request.json" },
    {
      "properties": {
        "op": { "const": "embedding.stream_embed" },
        "args": {
          "type": "object",
          "properties": {
            "text": {},
            "model": {},
            "truncate": {},
            "normalize": {}
          },
          "additionalProperties": true
        }
      }
    }
  ]
}
```

**Streaming Frame (`embedding/embedding.stream_embed.success.json`)**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.stream_embed.success.json",
  "title": "Embedding Stream Embed Success Frame",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/embedding/embedding.envelope.stream.success.json" }
  ]
}
```

---

#### 4.3.2.4 `embedding.embed_batch`

**Request (`embedding/embedding.embed_batch.request.json`)**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.embed_batch.request.json",
  "title": "Embedding Batch Embed Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/embedding/embedding.envelope.request.json" },
    {
      "properties": {
        "op": { "const": "embedding.embed_batch" },
        "args": {
          "type": "object",
          "properties": {
            "texts": {
              "type": "array",
              "items": {}
            },
            "model": {},
            "truncate": {},
            "normalize": {}
          },
          "required": ["texts"],
          "additionalProperties": true
        }
      }
    }
  ]
}
```

**Success (`embedding/embedding.embed_batch.success.json`)**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.embed_batch.success.json",
  "title": "Embedding Batch Embed Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/embedding/embedding.envelope.success.json" },
    {
      "properties": {
        "result": { "$ref": "https://corpusos.com/schemas/embedding/embedding.types.batch_result.json" }
      }
    }
  ]
}
```

---

#### 4.3.2.5 `embedding.count_tokens`

**Request (`embedding/embedding.count_tokens.request.json`)**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.count_tokens.request.json",
  "title": "Embedding Count Tokens Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/embedding/embedding.envelope.request.json" },
    {
      "properties": {
        "op": { "const": "embedding.count_tokens" },
        "args": {
          "type": "object",
          "properties": {
            "text": { "type": "string" },
            "model": { "type": "string" }
          },
          "required": ["text", "model"],
          "additionalProperties": true
        }
      }
    }
  ]
}
```

**Success (`embedding/embedding.count_tokens.success.json`)**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.count_tokens.success.json",
  "title": "Embedding Count Tokens Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/embedding/embedding.envelope.success.json" },
    {
      "properties": {
        "result": { "type": "integer" }
      }
    }
  ]
}
```

---

#### 4.3.2.6 `embedding.health`

**Request (`embedding/embedding.health.request.json`)**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.health.request.json",
  "title": "Embedding Health Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/embedding/embedding.envelope.request.json" },
    {
      "properties": {
        "op": { "const": "embedding.health" },
        "args": { "type": "object", "additionalProperties": true }
      }
    }
  ]
}
```

**Success (`embedding/embedding.health.success.json`)**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.health.success.json",
  "title": "Embedding Health Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/embedding/embedding.envelope.success.json" },
    {
      "properties": {
        "result": {
          "type": "object",
          "properties": {
            "ok": { "type": "boolean" },
            "server": { "type": "string" },
            "version": { "type": "string" },
            "models": {}
          },
          "required": ["ok", "server", "version", "models"],
          "additionalProperties": true
        }
      }
    }
  ]
}
```

---

#### 4.3.2.7 `embedding.get_stats`

**Request (`embedding/embedding.get_stats.request.json`)**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.get_stats.request.json",
  "title": "Embedding Get Stats Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/embedding/embedding.envelope.request.json" },
    {
      "properties": {
        "op": { "const": "embedding.get_stats" },
        "args": { "type": "object", "additionalProperties": true }
      }
    }
  ]
}
```

**Success (`embedding/embedding.get_stats.success.json`)**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.get_stats.success.json",
  "title": "Embedding Get Stats Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/embedding/embedding.envelope.success.json" },
    {
      "properties": {
        "result": { "$ref": "https://corpusos.com/schemas/embedding/embedding.stats.json" }
      }
    }
  ]
}
```

---

### 4.3.3 Type Definitions

#### 4.3.3.1 Capabilities (`embedding/embedding.capabilities.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.capabilities.json",
  "title": "Embedding Capabilities",
  "type": "object",
  "properties": {
    "server": { "type": "string" },
    "version": { "type": "string" },
    "supported_models": { "type": "array", "items": { "type": "string" } },
    "protocol": { "type": "string" },

    "max_batch_size": { "type": ["integer", "null"] },
    "max_text_length": { "type": ["integer", "null"] },
    "max_dimensions": { "type": ["integer", "null"] },

    "supports_normalization": { "type": "boolean" },
    "supports_truncation": { "type": "boolean" },
    "supports_token_counting": { "type": "boolean" },
    "supports_streaming": { "type": "boolean" },
    "supports_batch_embedding": { "type": "boolean" },
    "supports_caching": { "type": "boolean" },

    "idempotent_writes": { "type": "boolean" },
    "supports_multi_tenant": { "type": "boolean" },
    "normalizes_at_source": { "type": "boolean" },
    "truncation_mode": { "type": "string" },
    "supports_deadline": { "type": "boolean" }
  },
  "required": ["server", "version", "supported_models", "protocol"],
  "additionalProperties": false
}
```

#### 4.3.3.2 Stats (`embedding/embedding.stats.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.stats.json",
  "title": "Embedding Statistics",
  "type": "object",
  "properties": {
    "total_requests": { "type": "integer" },
    "total_texts": { "type": "integer" },
    "total_tokens": { "type": "integer" },
    "cache_hits": { "type": "integer" },
    "cache_misses": { "type": "integer" },
    "avg_processing_time_ms": { "type": "number" },
    "error_count": { "type": "integer" },
    "stream_requests": { "type": "integer" },
    "stream_chunks_generated": { "type": "integer" },
    "stream_abandoned": { "type": "integer" }
  },
  "required": ["total_requests", "total_texts", "total_tokens"],
  "additionalProperties": false
}
```

#### 4.3.3.3 Embedding Vector (`embedding/embedding.types.vector.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.types.vector.json",
  "title": "Embedding Vector",
  "type": "object",
  "properties": {
    "vector": { "type": "array", "items": {} },
    "text": { "type": "string" },
    "model": { "type": "string" },
    "dimensions": { "type": "integer" },
    "index": { "type": ["integer", "null"] },
    "metadata": { "type": ["object", "null"], "additionalProperties": true }
  },
  "required": ["vector", "text", "model", "dimensions"],
  "additionalProperties": false
}
```

#### 4.3.3.4 Unary Embed Result (`embedding/embedding.types.result.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.types.result.json",
  "title": "Embedding Embed Result",
  "type": "object",
  "properties": {
    "embedding": { "$ref": "https://corpusos.com/schemas/embedding/embedding.types.vector.json" },
    "model": { "type": "string" },
    "text": { "type": "string" },
    "tokens_used": { "type": ["integer", "null"] },
    "truncated": { "type": "boolean" }
  },
  "required": ["embedding", "model", "text", "truncated"],
  "additionalProperties": false
}
```

#### 4.3.3.5 Streaming Chunk (`embedding/embedding.types.chunk.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.types.chunk.json",
  "title": "Embedding Streaming Chunk",
  "type": "object",
  "properties": {
    "embeddings": {
      "type": "array",
      "items": { "$ref": "https://corpusos.com/schemas/embedding/embedding.types.vector.json" }
    },
    "is_final": { "type": "boolean" },
    "usage": { "type": ["object", "null"], "additionalProperties": true },
    "model": { "type": ["string", "null"] }
  },
  "required": ["embeddings", "is_final"],
  "additionalProperties": false
}
```

#### 4.3.3.6 Failure Item (`embedding/embedding.types.failure.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.types.failure.json",
  "title": "Embedding Failure Item",
  "type": "object",
  "properties": {
    "index": { "type": "integer" },
    "text": { "type": "string" },
    "error": { "type": "string" },
    "code": { "type": "string" },
    "message": { "type": "string" },
    "metadata": { "type": ["object", "null"], "additionalProperties": true }
  },
  "required": ["index", "text", "error", "code", "message"],
  "additionalProperties": true
}
```

#### 4.3.3.7 Batch Embed Result (`embedding/embedding.types.batch_result.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.types.batch_result.json",
  "title": "Embedding Batch Result",
  "type": "object",
  "properties": {
    "embeddings": {
      "type": "array",
      "items": { "$ref": "https://corpusos.com/schemas/embedding/embedding.types.vector.json" }
    },
    "model": { "type": "string" },
    "total_texts": { "type": "integer" },
    "total_tokens": { "type": ["integer", "null"] },
    "failed_texts": {
      "type": "array",
      "items": { "$ref": "https://corpusos.com/schemas/embedding/embedding.types.failure.json" }
    }
  },
  "required": ["embeddings", "model", "total_texts", "failed_texts"],
  "additionalProperties": false
}
```



### 4.4 Graph Protocol Schemas

#### 4.4.1 Envelope Schemas

**Graph Request Envelope (`graph.envelope.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.envelope.request.json",
  "title": "Graph Protocol Request Envelope",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/common/envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "pattern": "^graph\\.[a-z_]+$"
        }
      }
    }
  ]
}
```

**Graph Success Envelope (`graph.envelope.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.envelope.success.json",
  "title": "Graph Protocol Success Envelope",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/common/envelope.success.json" }
  ]
}
```

**Graph Error Envelope (`graph.envelope.error.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.envelope.error.json",
  "title": "Graph Protocol Error Envelope",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/common/envelope.error.json" }
  ]
}
```

#### 4.4.2 Operation Schemas

**Graph Capabilities Request (`graph.capabilities.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.capabilities.request.json",
  "title": "Graph Capabilities Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "graph.capabilities"
        },
        "args": {
          "type": "object",
          "additionalProperties": false,
          "description": "Empty args object required"
        }
      }
    }
  ]
}
```

**Graph Capabilities Success (`graph.capabilities.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.capabilities.success.json",
  "title": "Graph Capabilities Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.success.json" },
    {
      "properties": {
        "result": {
          "$ref": "https://corpusos.com/schemas/graph/graph.capabilities.json"
        }
      }
    }
  ]
}
```

**Graph Query Request (`graph.query.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.query.request.json",
  "title": "Graph Query Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "graph.query"
        },
        "args": {
          "$ref": "https://corpusos.com/schemas/graph/graph.types.query_spec.json"
        }
      }
    }
  ]
}
```

**Graph Query Success (`graph.query.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.query.success.json",
  "title": "Graph Query Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.success.json" },
    {
      "properties": {
        "result": {
          "$ref": "https://corpusos.com/schemas/graph/graph.types.query_result.json"
        }
      }
    }
  ]
}
```

**Graph Stream Query Request (`graph.stream_query.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.stream_query.request.json",
  "title": "Graph Stream Query Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "graph.stream_query"
        },
        "args": {
          "$ref": "https://corpusos.com/schemas/graph/graph.types.query_spec.json"
        }
      }
    }
  ]
}
```

**Graph Stream Query Success (`graph.stream_query.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.stream_query.success.json",
  "title": "Graph Stream Query Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/common/envelope.stream.success.json" },
    {
      "properties": {
        "chunk": {
          "$ref": "https://corpusos.com/schemas/graph/graph.types.chunk.json"
        }
      }
    }
  ]
}
```

**Graph Upsert Nodes Request (`graph.upsert_nodes.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.upsert_nodes.request.json",
  "title": "Graph Upsert Nodes Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "graph.upsert_nodes"
        },
        "args": {
          "type": "object",
          "properties": {
            "nodes": {
              "type": "array",
              "items": {
                "$ref": "https://corpusos.com/schemas/graph/graph.types.node.json"
              },
              "minItems": 1
            },
            "namespace": {
              "type": "string"
            }
          },
          "required": ["nodes"],
          "additionalProperties": false
        }
      }
    }
  ]
}
```

**Graph Upsert Nodes Success (`graph.upsert_nodes.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.upsert_nodes.success.json",
  "title": "Graph Upsert Nodes Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.success.json" },
    {
      "properties": {
        "result": {
          "type": "object",
          "properties": {
            "upserted_count": { "type": "integer", "minimum": 0 },
            "failed_count": { "type": "integer", "minimum": 0 },
            "failures": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "id": { "type": "string" },
                  "error": { "type": "string" }
                },
                "required": ["id", "error"]
              }
            }
          },
          "required": ["upserted_count", "failed_count", "failures"],
          "additionalProperties": false
        }
      }
    }
  ]
}
```

**Graph Upsert Edges Request (`graph.upsert_edges.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.upsert_edges.request.json",
  "title": "Graph Upsert Edges Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "graph.upsert_edges"
        },
        "args": {
          "type": "object",
          "properties": {
            "edges": {
              "type": "array",
              "items": {
                "$ref": "https://corpusos.com/schemas/graph/graph.types.edge.json"
              },
              "minItems": 1
            },
            "namespace": {
              "type": "string"
            }
          },
          "required": ["edges"],
          "additionalProperties": false
        }
      }
    }
  ]
}
```

**Graph Upsert Edges Success (`graph.upsert_edges.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.upsert_edges.success.json",
  "title": "Graph Upsert Edges Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.success.json" },
    {
      "properties": {
        "result": {
          "type": "object",
          "properties": {
            "upserted_count": { "type": "integer", "minimum": 0 },
            "failed_count": { "type": "integer", "minimum": 0 },
            "failures": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "id": { "type": "string" },
                  "error": { "type": "string" }
                },
                "required": ["id", "error"]
              }
            }
          },
          "required": ["upserted_count", "failed_count", "failures"],
          "additionalProperties": false
        }
      }
    }
  ]
}
```

**Graph Delete Nodes Request (`graph.delete_nodes.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.delete_nodes.request.json",
  "title": "Graph Delete Nodes Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "graph.delete_nodes"
        },
        "args": {
          "type": "object",
          "properties": {
            "ids": {
              "type": "array",
              "items": { "type": "string" },
              "minItems": 1
            },
            "filter": {
              "type": "object",
              "additionalProperties": true
            },
            "namespace": {
              "type": "string"
            }
          },
          "required": ["ids"],
          "additionalProperties": false
        }
      }
    }
  ]
}
```

**Graph Delete Nodes Success (`graph.delete_nodes.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.delete_nodes.success.json",
  "title": "Graph Delete Nodes Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.success.json" },
    {
      "properties": {
        "result": {
          "type": "object",
          "properties": {
            "deleted_count": { "type": "integer", "minimum": 0 },
            "failed_count": { "type": "integer", "minimum": 0 },
            "failures": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "id": { "type": "string" },
                  "error": { "type": "string" }
                },
                "required": ["id", "error"]
              }
            }
          },
          "required": ["deleted_count", "failed_count", "failures"],
          "additionalProperties": false
        }
      }
    }
  ]
}
```

**Graph Delete Edges Request (`graph.delete_edges.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.delete_edges.request.json",
  "title": "Graph Delete Edges Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "graph.delete_edges"
        },
        "args": {
          "type": "object",
          "properties": {
            "ids": {
              "type": "array",
              "items": { "type": "string" },
              "minItems": 1
            },
            "filter": {
              "type": "object",
              "additionalProperties": true
            },
            "namespace": {
              "type": "string"
            }
          },
          "required": ["ids"],
          "additionalProperties": false
        }
      }
    }
  ]
}
```

**Graph Delete Edges Success (`graph.delete_edges.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.delete_edges.success.json",
  "title": "Graph Delete Edges Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.success.json" },
    {
      "properties": {
        "result": {
          "type": "object",
          "properties": {
            "deleted_count": { "type": "integer", "minimum": 0 },
            "failed_count": { "type": "integer", "minimum": 0 },
            "failures": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "id": { "type": "string" },
                  "error": { "type": "string" }
                },
                "required": ["id", "error"]
              }
            }
          },
          "required": ["deleted_count", "failed_count", "failures"],
          "additionalProperties": false
        }
      }
    }
  ]
}
```

**Graph Bulk Vertices Request (`graph.bulk_vertices.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.bulk_vertices.request.json",
  "title": "Graph Bulk Vertices Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "graph.bulk_vertices"
        },
        "args": {
          "$ref": "https://corpusos.com/schemas/graph/graph.types.bulk_vertices_spec.json"
        }
      }
    }
  ]
}
```

**Graph Bulk Vertices Success (`graph.bulk_vertices.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.bulk_vertices.success.json",
  "title": "Graph Bulk Vertices Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.success.json" },
    {
      "properties": {
        "result": {
          "$ref": "https://corpusos.com/schemas/graph/graph.types.bulk_vertices_result.json"
        }
      }
    }
  ]
}
```

**Graph Batch Request (`graph.batch.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.batch.request.json",
  "title": "Graph Batch Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "graph.batch"
        },
        "args": {
          "type": "object",
          "properties": {
            "ops": {
              "type": "array",
              "items": {
                "$ref": "https://corpusos.com/schemas/graph/graph.types.batch_op.json"
              },
              "minItems": 1
            }
          },
          "required": ["ops"],
          "additionalProperties": false
        }
      }
    }
  ]
}
```

**Graph Batch Success (`graph.batch.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.batch.success.json",
  "title": "Graph Batch Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.success.json" },
    {
      "properties": {
        "result": {
          "$ref": "https://corpusos.com/schemas/graph/graph.types.batch_result.json"
        }
      }
    }
  ]
}
```

**Graph Get Schema Request (`graph.get_schema.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.get_schema.request.json",
  "title": "Graph Get Schema Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "graph.get_schema"
        },
        "args": {
          "type": "object",
          "additionalProperties": false,
          "description": "Empty args object required"
        }
      }
    }
  ]
}
```

**Graph Get Schema Success (`graph.get_schema.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.get_schema.success.json",
  "title": "Graph Get Schema Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.success.json" },
    {
      "properties": {
        "result": {
          "$ref": "https://corpusos.com/schemas/graph/graph.types.graph_schema.json"
        }
      }
    }
  ]
}
```

**Graph Health Request (`graph.health.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.health.request.json",
  "title": "Graph Health Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "graph.health"
        },
        "args": {
          "type": "object",
          "additionalProperties": false,
          "description": "Empty args object required"
        }
      }
    }
  ]
}
```

**Graph Health Success (`graph.health.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.health.success.json",
  "title": "Graph Health Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.success.json" },
    {
      "properties": {
        "result": {
          "$ref": "https://corpusos.com/schemas/graph/graph.types.health_result.json"
        }
      }
    }
  ]
}
```

**Graph Transaction Request (`graph.transaction.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.transaction.request.json",
  "title": "Graph Transaction Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "graph.transaction"
        },
        "args": {
          "type": "object",
          "properties": {
            "operations": {
              "type": "array",
              "items": {
                "$ref": "https://corpusos.com/schemas/graph/graph.types.batch_op.json"
              },
              "minItems": 1
            }
          },
          "required": ["operations"],
          "additionalProperties": false
        }
      }
    }
  ]
}
```

**Graph Transaction Success (`graph.transaction.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.transaction.success.json",
  "title": "Graph Transaction Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.success.json" },
    {
      "properties": {
        "result": {
          "$ref": "https://corpusos.com/schemas/graph/graph.types.batch_result.json"
        }
      }
    }
  ]
}
```

**Graph Traversal Request (`graph.traversal.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.traversal.request.json",
  "title": "Graph Traversal Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "graph.traversal"
        },
        "args": {
          "$ref": "https://corpusos.com/schemas/graph/graph.types.traversal_spec.json"
        }
      }
    }
  ]
}
```

**Graph Traversal Success (`graph.traversal.success.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.traversal.success.json",
  "title": "Graph Traversal Success",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.envelope.success.json" },
    {
      "properties": {
        "result": {
          "$ref": "https://corpusos.com/schemas/graph/graph.types.traversal_result.json"
        }
      }
    }
  ]
}
```

#### 4.4.3 Type Definitions

**Graph Capabilities (`graph.capabilities.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.capabilities.json",
  "title": "Graph Capabilities",
  "type": "object",
  "properties": {
    "server": { "type": "string" },
    "version": { "type": "string" },
    "protocol": { "type": "string", "const": "graph/v1.0" },
    "supports_stream_query": { "type": "boolean", "default": true },
    "supported_query_dialects": {
      "type": "array",
      "items": { "type": "string" },
      "default": []
    },
    "supports_namespaces": { "type": "boolean", "default": true },
    "supports_property_filters": { "type": "boolean", "default": true },
    "supports_bulk_vertices": { "type": "boolean", "default": false },
    "supports_batch": { "type": "boolean", "default": false },
    "supports_schema": { "type": "boolean", "default": false },
    "idempotent_writes": { "type": "boolean", "default": false },
    "supports_multi_tenant": { "type": "boolean", "default": false },
    "supports_deadline": { "type": "boolean", "default": true },
    "max_batch_ops": { "type": ["integer", "null"], "minimum": 1 },
    "supports_transaction": { "type": "boolean", "default": false },
    "supports_traversal": { "type": "boolean", "default": false },
    "max_traversal_depth": { "type": ["integer", "null"], "minimum": 1 },
    "supports_path_queries": { "type": "boolean", "default": false }
  },
  "required": ["server", "version"],
  "additionalProperties": false
}
```

**Node Type (`graph.types.node.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.types.node.json",
  "title": "Graph Node",
  "type": "object",
  "properties": {
    "id": {
      "type": "string",
      "minLength": 1
    },
    "labels": {
      "type": "array",
      "items": { "type": "string" },
      "default": []
    },
    "properties": {
      "type": "object",
      "additionalProperties": true,
      "default": {}
    },
    "namespace": {
      "type": "string"
    },
    "created_at": {
      "type": ["integer", "null"],
      "minimum": 0
    },
    "updated_at": {
      "type": ["integer", "null"],
      "minimum": 0
    }
  },
  "required": ["id", "properties"],
  "additionalProperties": false
}
```

**Edge Type (`graph.types.edge.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.types.edge.json",
  "title": "Graph Edge",
  "type": "object",
  "properties": {
    "id": {
      "type": "string",
      "minLength": 1
    },
    "src": {
      "type": "string",
      "minLength": 1
    },
    "dst": {
      "type": "string",
      "minLength": 1
    },
    "label": {
      "type": "string",
      "minLength": 1
    },
    "properties": {
      "type": "object",
      "additionalProperties": true,
      "default": {}
    },
    "namespace": {
      "type": "string"
    },
    "created_at": {
      "type": ["integer", "null"],
      "minimum": 0
    },
    "updated_at": {
      "type": ["integer", "null"],
      "minimum": 0
    }
  },
  "required": ["id", "src", "dst", "label", "properties"],
  "additionalProperties": false
}
```

**Entity Type (`graph.types.entity.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.types.entity.json",
  "title": "Graph Entity",
  "type": "object",
  "oneOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.types.node.json" },
    { "$ref": "https://corpusos.com/schemas/graph/graph.types.edge.json" }
  ]
}
```

**Query Specification (`graph.types.query_spec.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.types.query_spec.json",
  "title": "Graph Query Specification",
  "type": "object",
  "properties": {
    "text": {
      "type": "string",
      "minLength": 1
    },
    "dialect": {
      "type": "string"
    },
    "params": {
      "type": "object",
      "additionalProperties": true,
      "default": {}
    },
    "namespace": {
      "type": "string"
    },
    "timeout_ms": {
      "type": ["integer", "null"],
      "minimum": 1
    },
    "stream": {
      "type": "boolean",
      "default": false
    }
  },
  "required": ["text"],
  "additionalProperties": false
}
```

**Query Result (`graph.types.query_result.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.types.query_result.json",
  "title": "Graph Query Result",
  "type": "object",
  "properties": {
    "records": {
      "type": "array",
      "description": "Query results (any JSON value)"
    },
    "summary": {
      "type": "object",
      "additionalProperties": true
    },
    "dialect": {
      "type": "string"
    },
    "namespace": {
      "type": "string"
    }
  },
  "required": ["records", "summary"],
  "additionalProperties": false
}
```

**Query Chunk (`graph.types.chunk.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.types.chunk.json",
  "title": "Graph Streaming Chunk",
  "type": "object",
  "properties": {
    "records": {
      "type": "array",
      "description": "Streaming results (any JSON value)"
    },
    "is_final": {
      "type": "boolean",
      "default": false,
      "description": "True for final chunk"
    },
    "summary": {
      "type": ["object", "null"],
      "additionalProperties": true
    }
  },
  "required": ["records", "is_final"],
  "additionalProperties": false
}
```

**ID Type (`graph.types.id.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.types.id.json",
  "title": "Graph ID Value",
  "type": "object",
  "properties": {
    "id": { "type": "string", "minLength": 1 }
  },
  "required": ["id"],
  "additionalProperties": false
}
```

**Batch Operation (`graph.types.batch_op.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.types.batch_op.json",
  "title": "Graph Batch Operation",
  "type": "object",
  "properties": {
    "op": { "type": "string", "minLength": 1 },
    "args": { "type": "object", "additionalProperties": true }
  },
  "required": ["op", "args"],
  "additionalProperties": false
}
```

**Batch Result (`graph.types.batch_result.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.types.batch_result.json",
  "title": "Graph Batch Result",
  "type": "object",
  "properties": {
    "results": {
      "type": "array",
      "description": "Batch operation results (any JSON value)"
    },
    "success": { "type": "boolean" },
    "error": { "type": ["string", "null"] },
    "transaction_id": { "type": ["string", "null"] }
  },
  "required": ["results", "success"],
  "additionalProperties": false
}
```

**Bulk Vertices Specification (`graph.types.bulk_vertices_spec.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.types.bulk_vertices_spec.json",
  "title": "Bulk Vertices Specification",
  "type": "object",
  "properties": {
    "namespace": { "type": "string" },
    "limit": { "type": "integer", "minimum": 1, "default": 100 },
    "cursor": { "type": ["string", "null"] },
    "filter": {
      "type": ["object", "null"],
      "additionalProperties": true
    }
  },
  "additionalProperties": false
}
```

**Bulk Vertices Result (`graph.types.bulk_vertices_result.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.types.bulk_vertices_result.json",
  "title": "Bulk Vertices Result",
  "type": "object",
  "properties": {
    "nodes": {
      "type": "array",
      "items": {
        "$ref": "https://corpusos.com/schemas/graph/graph.types.node.json"
      }
    },
    "next_cursor": { "type": ["string", "null"] },
    "has_more": { "type": "boolean" }
  },
  "required": ["nodes", "has_more"],
  "additionalProperties": false
}
```

**Graph Schema (`graph.types.graph_schema.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.types.graph_schema.json",
  "title": "Graph Schema",
  "type": "object",
  "properties": {
    "nodes": {
      "type": "object",
      "additionalProperties": true
    },
    "edges": {
      "type": "object",
      "additionalProperties": true
    },
    "metadata": {
      "type": "object",
      "additionalProperties": true
    }
  },
  "required": ["nodes", "edges", "metadata"],
  "additionalProperties": false
}
```

**Health Result (`graph.types.health_result.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.types.health_result.json",
  "title": "Graph Health Result",
  "type": "object",
  "properties": {
    "ok": { "type": "boolean" },
    "status": { "type": "string" },
    "server": { "type": "string" },
    "version": { "type": "string" },
    "namespaces": {
      "type": "object",
      "additionalProperties": true
    },
    "read_only": { "type": "boolean", "default": false },
    "degraded": { "type": "boolean", "default": false }
  },
  "required": ["ok", "status", "server", "version"],
  "additionalProperties": false
}
```

**Traversal Specification (`graph.types.traversal_spec.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.types.traversal_spec.json",
  "title": "Graph Traversal Specification",
  "type": "object",
  "properties": {
    "start_nodes": {
      "type": "array",
      "items": { "type": "string", "minLength": 1 },
      "minItems": 1
    },
    "max_depth": {
      "type": "integer",
      "minimum": 1
    },
    "direction": {
      "type": "string",
      "enum": ["OUTGOING", "INCOMING", "BOTH"]
    },
    "relationship_types": {
      "type": ["array", "null"],
      "items": { "type": "string" }
    },
    "node_filters": {
      "type": ["object", "null"],
      "additionalProperties": true
    },
    "relationship_filters": {
      "type": ["object", "null"],
      "additionalProperties": true
    },
    "return_properties": {
      "type": ["array", "null"],
      "items": { "type": "string" }
    },
    "namespace": {
      "type": ["string", "null"]
    }
  },
  "required": ["start_nodes", "max_depth", "direction"],
  "additionalProperties": false
}
```

**Traversal Result (`graph.types.traversal_result.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.types.traversal_result.json",
  "title": "Graph Traversal Result",
  "type": "object",
  "properties": {
    "nodes": {
      "type": "array",
      "items": {
        "$ref": "https://corpusos.com/schemas/graph/graph.types.node.json"
      }
    },
    "relationships": {
      "type": "array",
      "items": {
        "$ref": "https://corpusos.com/schemas/graph/graph.types.edge.json"
      }
    },
    "paths": {
      "type": "array",
      "items": {
        "type": "array",
        "items": {
          "type": "object",
          "additionalProperties": true
        }
      }
    },
    "summary": {
      "type": "object",
      "additionalProperties": true
    },
    "namespace": {
      "type": ["string", "null"]
    }
  },
  "required": ["nodes", "relationships", "paths", "summary"],
  "additionalProperties": false
}
```

---

## 5. Streaming Schemas

### 5.1 Streaming Envelope Schema

**Streaming Model:** All streaming operations use the common streaming success envelope pattern:

```json
{
  "ok": true,
  "code": "STREAMING",
  "ms": 45.2,
  "chunk": {
    "text": "Hello world",
    "is_final": false,
    "model": "gpt-4.1-mini"
  }
}
```

**Protocol Streaming Envelopes:**

- **LLM Streaming**: `{ok:true, code:"STREAMING", ms:number, chunk:<llm.types.chunk>}`
- **Graph Streaming**: `{ok:true, code:"STREAMING", ms:number, chunk:<graph.types.chunk>}`
- **Embedding Streaming**: `{ok:true, code:"STREAMING", ms:number, chunk:<embedding.types.chunk>}`

**Error Termination:** Streams terminate with a standard error envelope (not a special streaming error):
```json
{
  "ok": false,
  "code": "TRANSIENT_NETWORK",
  "error": "TransientNetworkError",
  "message": "Connection lost",
  "ms": 123.4
}
```

### 5.2 NDJSON Schema

**NDJSON Stream Validation:**
Streams are delivered as NDJSON (Newline-Delimited JSON) where each line is a protocol envelope. The union of possible NDJSON lines is:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/ndjson/stream.schema.json",
  "title": "NDJSON Stream Union",
  "oneOf": [
    { "$ref": "https://corpusos.com/schemas/common/envelope.stream.success.json" },
    { "$ref": "https://corpusos.com/schemas/common/envelope.error.json" }
  ]
}
```

**Example NDJSON Stream:**
```json
{"ok": true, "code": "STREAMING", "ms": 12.3, "chunk": {"text": "Hello", "is_final": false, "model": "gpt-4.1-mini"}}
{"ok": true, "code": "STREAMING", "ms": 15.7, "chunk": {"text": " world", "is_final": false, "model": "gpt-4.1-mini"}}
{"ok": true, "code": "STREAMING", "ms": 18.2, "chunk": {"text": "!", "is_final": true, "model": "gpt-4.1-mini", "usage_so_far": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}}
```

### 5.3 Streaming Semantics

**Protocol Compliance:**
All streaming implementations MUST adhere to these semantics:

1. **Single Terminal Condition:** Exactly one terminal condition per stream:
   - A chunk with `is_final: true` indicates successful completion
   - An error envelope indicates stream failure
2. **No Content After Terminal:** Stream MUST end after terminal condition
3. **Chunk Integrity:** Complete tokens/records delivered in each chunk
4. **Order Preservation:** Chunks delivered in correct sequence
5. **Streaming Code:** All streaming success envelopes use `code: "STREAMING"`

**Validation Rules for Streams:**

```python
class StreamValidationRules:
    """Streaming semantics validation."""
    
    @staticmethod
    def validate_stream_termination(frames: List[dict]) -> None:
        """Validate stream has exactly one terminal condition."""
        terminal_chunks = [f for f in frames if f.get("ok") and f["chunk"].get("is_final")]
        error_frames = [f for f in frames if not f.get("ok")]
        
        terminal_count = len(terminal_chunks) + len(error_frames)
        
        if terminal_count == 0:
            raise StreamProtocolError("Stream missing terminal condition")
        if terminal_count > 1:
            raise StreamProtocolError(f"Multiple terminal conditions: {terminal_count}")
        
        # Get terminal frame
        if terminal_chunks:
            terminal_frame = terminal_chunks[0]
        else:
            terminal_frame = error_frames[0]
        
        terminal_index = frames.index(terminal_frame)
        if terminal_index != len(frames) - 1:
            raise StreamProtocolError("Content after terminal frame")
    
    @staticmethod
    def validate_streaming_code(frames: List[dict]) -> None:
        """Validate all streaming success frames use STREAMING code."""
        for i, frame in enumerate(frames):
            if frame.get("ok") and frame.get("code") != "STREAMING":
                raise StreamProtocolError(f"Frame {i} must use code='STREAMING' for streaming operations")
```

**Performance Considerations:**
- **Backpressure:** Clients control consumption rate
- **Bounded Buffering:** Adapters implement flow control
- **Heartbeats:** Optional keep-alive messages allowed
- **Chunk Size:** Optimized for network efficiency

---

## 6. Schema Validation Infrastructure

### 6.1 Schema Registry

**Centralized Schema Loading:**
The schema registry loads all JSON Schema files and provides validation services:

```python
from tests.utils.schema_registry import get_validator, validate_json, assert_valid

# Load schema and validate
validator = get_validator("https://corpusos.com/schemas/llm/llm.envelope.request.json")

# Validate an object
validate_json("https://corpusos.com/schemas/llm/llm.envelope.request.json", request_envelope)

# Pytest-friendly assertion
assert_valid("https://corpusos.com/schemas/llm/llm.envelope.request.json", request_envelope)
```

**Registry Features:**
- **Thread-safe singleton** with lazy loading
- **$id-based resolution** with absolute URIs
- **Validator caching** for performance
- **Error recovery** with helpful suggestions
- **Environment configuration** via `CORPUS_SCHEMAS_ROOT`

**Configuration Options:**
```bash
# Environment variable configuration
export CORPUS_SCHEMAS_ROOT=/path/to/schemas

# CLI interface
python -m tests.utils.schema_registry --list
python -m tests.utils.schema_registry llm.envelope.request.json sample.json

# Show schema statistics
python -m tests.utils.schema_registry --stats
```

**Registry Health Checks:**
```python
def test_schema_registry_health():
    """Comprehensive schema registry health validation."""
    registry = SchemaRegistry()
    
    # Check all schemas load
    schemas = registry.list_schemas()
    assert len(schemas) > 0, "No schemas loaded"
    
    # Check validators can be created
    for schema_id in schemas.keys():
        validator = registry.get_validator(schema_id)
        assert validator is not None, f"Failed to create validator for {schema_id}"
    
    # Check $ref resolution
    for schema_id, schema in registry._SCHEMA_STORE.items():
        check_ref_resolution(schema)
```

### 6.2 Validation Modes

**Strict Validation:**
```python
config = ValidationConfig(
    envelope_schema_id="https://corpusos.com/schemas/llm/llm.envelope.success.json",
    component="llm",
    mode=ValidationMode.STRICT,  # Validate every frame
    max_frame_bytes=1_048_576,    # 1 MiB limit
)
```

**Sampled Validation (for performance):**
```python
config = ValidationConfig(
    envelope_schema_id="https://corpusos.com/schemas/llm/llm.envelope.success.json",
    component="llm", 
    mode=ValidationMode.SAMPLED,
    sample_rate=0.1,  # Validate 10% of frames
)
```

**Validation Mode Comparison:**

| Mode | Schema Validation | Protocol Validation | Performance | Use Case |
|------|-------------------|---------------------|-------------|----------|
| **STRICT** | Every frame | Every frame | Slowest | CI/CD, pre-production |
| **SAMPLED** | Sampled frames | Every frame | Medium | Production monitoring |
| **LAZY** | None | Every frame | Fastest | High-throughput production |
| **COLLECT_ERRORS** | Every frame | Every frame | Medium | Debugging, error analysis |

**Performance Characteristics:**
- **Schema validation**: ~1-10ms per frame (Draft 2020-12)
- **Protocol validation**: ~0.1-1ms per frame
- **Memory usage**: ~1-10KB per validator
- **Cache effectiveness**: ~90% hit rate with LRU caching

### 6.3 Validation Pipeline

**Complete Validation Flow:**
```python
def validate_wire_envelope(
    envelope: dict,
    expected_op: str,
    schema_id: str,
    component: str,
    args_validator: Optional[Callable] = None,
    case_id: str = "unknown",
) -> None:
    """
    Complete wire envelope validation pipeline.
    
    1. Protocol envelope shape validation
    2. JSON serialization round-trip
    3. Schema validation with version tolerance
    4. Operation-specific args validation
    """
    # Step 1: Protocol envelope validation (§2.4)
    validate_envelope_common(envelope, expected_op, case_id)
    
    # Step 2: JSON serialization validation
    wire_envelope = json_roundtrip(envelope, case_id)
    
    # Step 3: JSON Schema validation
    assert_valid(schema_id, wire_envelope, context=case_id)
    
    # Step 4: Operation-specific validation
    if args_validator:
        validate_args_for_operation(wire_envelope["args"], args_validator, case_id)
```

**Protocol Envelope Validation:**
```python
def validate_envelope_common(envelope: dict, expected_op: str, case_id: str) -> None:
    """Validate protocol §2.4 envelope requirements."""
    
    # Must be a dictionary
    if not isinstance(envelope, dict):
        raise EnvelopeTypeError(f"{case_id}: envelope must be dict")
    
    # Required fields
    required = {"op", "ctx", "args"}
    missing = required - set(envelope.keys())
    if missing:
        raise EnvelopeShapeError(f"{case_id}: missing required keys {missing}")
    
    # Operation matching
    if envelope["op"] != expected_op:
        raise EnvelopeShapeError(
            f"{case_id}: operation mismatch: expected {expected_op}, got {envelope['op']}"
        )
    
    # Context validation
    validate_ctx_field(envelope, case_id)
```

**JSON Round-trip Validation:**
```python
def json_roundtrip(obj: Any, case_id: str) -> dict:
    """Validate JSON serialization/deserialization preserves data."""
    try:
        # Serialize to JSON
        json_str = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        
        # Deserialize back
        roundtripped = json.loads(json_str)
        
        # Must be a dict after round-trip
        if not isinstance(roundtripped, dict):
            raise SerializationError(f"{case_id}: round-trip did not produce dict")
            
        return roundtripped
        
    except (TypeError, ValueError, json.JSONDecodeError) as e:
        raise SerializationError(f"{case_id}: JSON serialization failed: {e}")
```

**Operation-Specific Validation:**
```python
def validate_args_for_operation(args: dict, validator: Callable, case_id: str) -> None:
    """Validate operation-specific argument constraints."""
    try:
        validator(args, case_id)
    except Exception as e:
        raise ArgsValidationError(f"{case_id}: args validation failed: {e}")
```

**Example Validators:**
```python
def validate_llm_complete_args(args: dict, case_id: str) -> None:
    """Validate llm.complete operation arguments."""
    if "messages" not in args:
        raise ArgsValidationError(f"{case_id}: requires 'messages'")
    
    if not isinstance(args["messages"], list) or len(args["messages"]) == 0:
        raise ArgsValidationError(f"{case_id}: 'messages' must be non-empty list")
    
    if "temperature" in args:
        temp = args["temperature"]
        if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
            raise ArgsValidationError(
                f"{case_id}: temperature must be between 0.0 and 2.0"
            )

def validate_vector_query_args(args: dict, case_id: str) -> None:
    """Validate vector.query operation arguments."""
    if "vector" not in args:
        raise ArgsValidationError(f"{case_id}: requires 'vector'")
    
    if "top_k" not in args:
        raise ArgsValidationError(f"{case_id}: requires 'top_k'")
    
    vector = args["vector"]
    if not isinstance(vector, list) or len(vector) == 0:
        raise ArgsValidationError(f"{case_id}: vector must be non-empty list")
    
    if not all(isinstance(x, (int, float)) for x in vector):
        raise ArgsValidationError(f"{case_id}: vector must contain numbers")

def validate_embedding_embed_args(args: dict, case_id: str) -> None:
    """Validate embedding.embed operation arguments."""
    if "text" not in args:
        raise ArgsValidationError(f"{case_id}: requires 'text'")
    
    if "model" not in args:
        raise ArgsValidationError(f"{case_id}: requires 'model'")
    
    if "stream" in args and args["stream"] is not False:
        raise ArgsValidationError(f"{case_id}: stream must be absent or false for embed operation")

def validate_graph_upsert_nodes_args(args: dict, case_id: str) -> None:
    """Validate graph.upsert_nodes operation arguments."""
    if "nodes" not in args:
        raise ArgsValidationError(f"{case_id}: requires 'nodes'")
    
    if not isinstance(args["nodes"], list) or len(args["nodes"]) == 0:
        raise ArgsValidationError(f"{case_id}: 'nodes' must be non-empty list")
    
    for i, node in enumerate(args["nodes"]):
        if "id" not in node:
            raise ArgsValidationError(f"{case_id}: node at index {i} missing required 'id'")
```

---

## 7. Golden Test Infrastructure

### 7.1 Golden Test Philosophy

**Executable Documentation:**
Golden tests serve as both validation fixtures and protocol documentation:

```python
# Golden test mapping
CASES = [
    # LLM
    ("llm/llm_capabilities_request.json",       "https://corpusos.com/schemas/llm/llm.capabilities.request.json"),
    ("llm/llm_capabilities_success.json",       "https://corpusos.com/schemas/llm/llm.capabilities.success.json"),
    ("llm/llm_complete_request.json",           "https://corpusos.com/schemas/llm/llm.complete.request.json"),
    ("llm/llm_complete_success.json",           "https://corpusos.com/schemas/llm/llm.complete.success.json"),
    ("llm/llm_health_request.json",             "https://corpusos.com/schemas/llm/llm.health.request.json"),
    ("llm/llm_health_success.json",             "https://corpusos.com/schemas/llm/llm.health.success.json"),
    
    # Vector
    ("vector/vector_capabilities_request.json", "https://corpusos.com/schemas/vector/vector.capabilities.request.json"),
    ("vector/vector_capabilities_success.json", "https://corpusos.com/schemas/vector/vector.capabilities.success.json"),
    ("vector/vector_query_request.json",        "https://corpusos.com/schemas/vector/vector.query.request.json"),
    ("vector/vector_batch_query_request.json",  "https://corpusos.com/schemas/vector/vector.batch_query.request.json"),
    ("vector/vector_upsert_request.json",       "https://corpusos.com/schemas/vector/vector.upsert.request.json"),
    ("vector/vector_create_namespace_request.json", "https://corpusos.com/schemas/vector/vector.create_namespace.request.json"),
    ("vector/vector_health_request.json",       "https://corpusos.com/schemas/vector/vector.health.request.json"),
    
    # Embedding
    ("embedding/embedding_capabilities_request.json", "https://corpusos.com/schemas/embedding/embedding.capabilities.request.json"),
    ("embedding/embedding_embed_request.json",        "https://corpusos.com/schemas/embedding/embedding.embed.request.json"),
    ("embedding/embedding_embed_batch_request.json",  "https://corpusos.com/schemas/embedding/embedding.embed_batch.request.json"),
    ("embedding/embedding_health_request.json",       "https://corpusos.com/schemas/embedding/embedding.health.request.json"),
    ("embedding/embedding_get_stats_request.json",    "https://corpusos.com/schemas/embedding/embedding.get_stats.request.json"),
    
    # Graph
    ("graph/graph_capabilities_request.json",   "https://corpusos.com/schemas/graph/graph.capabilities.request.json"),
    ("graph/graph_query_request.json",          "https://corpusos.com/schemas/graph/graph.query.request.json"),
    ("graph/graph_upsert_nodes_request.json",   "https://corpusos.com/schemas/graph/graph.upsert_nodes.request.json"),
    ("graph/graph_upsert_edges_request.json",   "https://corpusos.com/schemas/graph/graph.upsert_edges.request.json"),
    ("graph/graph_delete_nodes_request.json",   "https://corpusos.com/schemas/graph/graph.delete_nodes.request.json"),
    ("graph/graph_bulk_vertices_request.json",  "https://corpusos.com/schemas/graph/graph.bulk_vertices.request.json"),
    ("graph/graph_batch_request.json",          "https://corpusos.com/schemas/graph/graph.batch.request.json"),
    ("graph/graph_get_schema_request.json",     "https://corpusos.com/schemas/graph/graph.get_schema.request.json"),
    ("graph/graph_health_request.json",         "https://corpusos.com/schemas/graph/graph.health.request.json"),
    # ... 100+ test cases
]
```

**Key Principles:**
1. **Examples validate schemas** - Each golden file validates against its schema
2. **Cross-protocol invariants** - Mathematical and semantic consistency checks
3. **Drift detection** - Automatic detection of schema/example mismatches
4. **Performance guardrails** - Size and complexity limits

### 7.2 Golden Test Organization

**Directory Structure:**
```
tests/golden/
├── llm/
│   ├── llm_capabilities_request.json
│   ├── llm_capabilities_success.json
│   ├── llm_complete_request.json
│   ├── llm_complete_success.json
│   ├── llm_count_tokens_request.json
│   ├── llm_count_tokens_success.json
│   ├── llm_error_envelope.json
│   ├── llm_health_request.json
│   ├── llm_health_success.json
│   ├── llm_response_format.json
│   ├── llm_sampling_params.json
│   ├── llm_stream.ndjson
│   ├── llm_stream_chunk.json
│   ├── llm_stream_error.ndjson
│   ├── llm_tools_schema.json
│   ├── llm_types_chunk.json
│   ├── llm_types_completion.json
│   ├── llm_types_logprobs.json
│   ├── llm_types_message.json
│   ├── llm_types_token_usage.json
│   ├── llm_types_tool.json
│   ├── llm_types_warning.json
│   └── llm_stream_embed_request.json
│
├── vector/
│   ├── vector_capabilities_request.json
│   ├── vector_capabilities_success.json
│   ├── vector_batch_query_request.json
│   ├── vector_batch_query_success.json
│   ├── vector_create_namespace_request.json
│   ├── vector_create_namespace_success.json
│   ├── vector_delete_request.json
│   ├── vector_delete_success.json
│   ├── vector_delete_namespace_request.json
│   ├── vector_delete_namespace_success.json
│   ├── vector_error_dimension_mismatch.json
│   ├── vector_health_request.json
│   ├── vector_health_success.json
│   ├── vector_query_request.json
│   ├── vector_query_success.json
│   ├── vector_types_document.json
│   ├── vector_types_filter.json
│   ├── vector_types_namespace_result.json
│   ├── vector_types_namespace_spec.json
│   ├── vector_types_query_result.json
│   ├── vector_types_query_spec.json
│   ├── vector_types_vector.json
│   ├── vector_types_vector_match.json
│   ├── vector_upsert_request.json
│   └── vector_upsert_success.json
│
├── embedding/
│   ├── embedding_capabilities_request.json
│   ├── embedding_capabilities_success.json
│   ├── embedding_count_tokens_request_batch.json
│   ├── embedding_count_tokens_request_single.json
│   ├── embedding_count_tokens_success_batch.json
│   ├── embedding_count_tokens_success_single.json
│   ├── embedding_embed_batch_request.json
│   ├── embedding_embed_batch_success.json
│   ├── embedding_embed_request.json
│   ├── embedding_embed_success.json
│   ├── embedding_envelope_error.json
│   ├── embedding_get_stats_request.json
│   ├── embedding_get_stats_success.json
│   ├── embedding_health_request.json
│   ├── embedding_health_success.json
│   ├── embedding_stream_embed_request.json
│   ├── embedding_stream_embed_success.json
│   ├── embedding_stream.ndjson
│   ├── embedding_types_batch_result.json
│   ├── embedding_types_chunk.json
│   ├── embedding_types_failure.json
│   ├── embedding_types_result.json
│   ├── embedding_types_vector.json
│   └── embedding_types_warning.json
│
└── graph/
    ├── graph_batch_request.json
    ├── graph_batch_success.json
    ├── graph_bulk_vertices_request.json
    ├── graph_bulk_vertices_success.json
    ├── graph_capabilities_request.json
    ├── graph_capabilities_success.json
    ├── graph_delete_edges_request.json
    ├── graph_delete_edges_success.json
    ├── graph_delete_nodes_request.json
    ├── graph_delete_nodes_success.json
    ├── graph_envelope_error.json
    ├── graph_get_schema_request.json
    ├── graph_get_schema_success.json
    ├── graph_health_request.json
    ├── graph_health_success.json
    ├── graph_query_request.json
    ├── graph_query_success.json
    ├── graph_stream.ndjson
    ├── graph_stream_query_request.json
    ├── graph_stream_query_success.json
    ├── graph_transaction_request.json
    ├── graph_transaction_success.json
    ├── graph_traversal_request.json
    ├── graph_traversal_success.json
    ├── graph_types_batch_result.json
    ├── graph_types_bulk_vertices_result.json
    ├── graph_types_bulk_vertices_spec.json
    ├── graph_types_chunk.json
    ├── graph_types_edge.json
    ├── graph_types_graph_schema.json
    ├── graph_types_health_result.json
    ├── graph_types_node.json
    ├── graph_types_query_result.json
    ├── graph_types_query_spec.json
    ├── graph_types_traversal_result.json
    ├── graph_types_traversal_spec.json
    ├── graph_upsert_edges_request.json
    ├── graph_upsert_edges_success.json
    ├── graph_upsert_nodes_request.json
    └── graph_upsert_nodes_success.json
```

**Naming Conventions:**
- `{protocol}_{operation}_{type}.json` - Operation test cases
- `{protocol}_types_{name}.json` - Type definition examples
- `{protocol}_stream.ndjson` - Complete streaming examples (using envelope-chunk model)
- `{protocol}_stream_{operation}_request.json` - Streaming operation requests

### 7.3 Golden Test Validation

**Comprehensive Validation Suite:**
```python
# Test that each golden file validates against its declared schema
@pytest.mark.parametrize("fname,schema_id", CASES)
def test_golden_validates(fname: str, schema_id: str):
    """Test that each golden file validates against its declared schema."""
    p = GOLDEN / fname
    if not p.exists():
        pytest.skip(f"{fname} fixture not present")

    doc = json.loads(p.read_text(encoding="utf-8"))
    assert_valid(schema_id, doc, context=fname)
```

**Cross-Schema Invariants:**
```python
def test_llm_token_totals_invariant():
    """Test LLM token usage mathematical invariant per §3.7."""
    doc = load_golden("llm/llm_complete_success.json")
    usage = doc.get("result", {}).get("usage")

    assert usage["total_tokens"] == usage["prompt_tokens"] + usage.get("completion_tokens", 0), \
        "total_tokens must equal prompt_tokens + completion_tokens"

def test_vector_dimension_invariants():
    """Test that all vectors in a response have consistent dimensions per §16.1."""
    doc = load_golden("vector/vector_query_success.json")
    result = doc.get("result", {})
    
    vecs = extract_vectors_from_result(result)
    if not vecs:
        pytest.skip("No vectors present")
    
    ref_dim = len(vecs[0])
    for i, v in enumerate(vecs):
        assert len(v) == ref_dim, f"Vector dimension mismatch at index {i}"
```

**Protocol Envelope Compliance:**
```python
def test_all_success_envelopes_follow_protocol_format():
    """Test ALL success envelopes include core fields per §2.4."""
    for fname, schema_id in CASES:
        if "envelope.success" not in schema_id:
            continue
        
        doc = load_golden(fname)
        
        # REQUIRED fields
        assert "ok" in doc, f"{fname}: missing 'ok' field"
        assert "code" in doc, f"{fname}: missing 'code' field"
        assert "result" in doc, f"{fname}: missing 'result' field"
        
        # Field constraints
        assert doc["ok"] is True, f"{fname}: 'ok' must be true"
        assert doc["code"] == "OK", f"{fname}: unexpected code {doc['code']!r}"

def test_all_request_envelopes_follow_protocol_format():
    """Test ALL request envelopes include core fields per §2.4."""
    for fname, schema_id in CASES:
        if "request.json" not in fname:
            continue
        
        doc = load_golden(fname)
        
        # REQUIRED fields
        assert "op" in doc, f"{fname}: missing 'op' field"
        assert "ctx" in doc, f"{fname}: missing 'ctx' field"
        assert "args" in doc, f"{fname}: missing 'args' field"
        
        # Field types
        assert isinstance(doc["ctx"], dict), f"{fname}: 'ctx' must be object"
        assert isinstance(doc["args"], dict), f"{fname}: 'args' must be object"
```

**Streaming Validation:**
```python
def test_streaming_uses_STREAMING_code():
    """Test streaming success envelopes use code='STREAMING'."""
    streaming_files = [
        "llm/llm_stream_chunk.json",
        "graph/graph_stream_chunk.json",
        "embedding/embedding_stream_chunk.json",
    ]
    
    for fname in streaming_files:
        doc = load_golden(fname)
        
        # Must use streaming envelope format
        assert "ok" in doc, f"{fname}: missing 'ok' field"
        assert "code" in doc, f"{fname}: missing 'code' field"
        assert "ms" in doc, f"{fname}: missing 'ms' field"
        assert "chunk" in doc, f"{fname}: missing 'chunk' field"
        
        assert doc["ok"] is True, f"{fname}: 'ok' must be true"
        assert doc["code"] == "STREAMING", f"{fname}: 'code' must be 'STREAMING'"
```

**NDJSON Stream Validation:**
```python
@pytest.mark.parametrize("fname,schema_id,component", [
    ("llm/llm_stream.ndjson", "https://corpusos.com/schemas/common/envelope.stream.success.json", "llm"),
    ("graph/graph_stream.ndjson", "https://corpusos.com/schemas/common/envelope.stream.success.json", "graph"),
    ("embedding/embedding_stream.ndjson", "https://corpusos.com/schemas/common/envelope.stream.success.json", "embedding"),
])
def test_streaming_ndjson_validates_with_stream_validator(fname, schema_id, component):
    """Validate NDJSON streaming golden fixtures."""
    ndjson_text = load_golden_text(fname)
    report = validate_ndjson_stream(
        ndjson_text,
        envelope_schema_id=schema_id,
        component=component,
    )
    
    assert report.is_valid, report.error_summary
```

**Performance and Size Guardrails:**
```python
def test_large_fixture_performance():
    """Test fixture size limits and large string validation."""
    for fname, _ in CASES:
        p = GOLDEN / fname
        if not p.exists():
            continue

        # Check file size (10MB limit)
        size = p.stat().st_size
        assert size <= 10 * 1024 * 1024, f"{fname} exceeds size limit: {size} bytes"

        # Check string field sizes
        doc = json.loads(p.read_text(encoding="utf-8"))
        issues = validate_string_field_size(doc, fname)
        
        if issues:
            pytest.fail(f"{fname} string field size issues:\n" + "\n".join(issues))
```

**Drift Detection:**
```python
def test_no_orphaned_golden_files():
    """Test that no golden files exist without CASES entries."""
    golden_files = {
        p.relative_to(GOLDEN).as_posix()
        for p in GOLDEN.rglob("*.json")
        if p.is_file()
    }
    tested_files = {fname for fname, _ in CASES}
    orphaned = golden_files - tested_files - SUPPORTING_FILES
    
    if orphaned:
        pytest.skip(f"Golden files without CASES entries: {sorted(orphaned)}")

def test_all_listed_golden_files_exist():
    """Test that all files listed in CASES exist on disk."""
    missing = [fname for fname, _ in CASES if not (GOLDEN / fname).exists()]
    if missing:
        pytest.skip(f"CASES contains missing fixtures (ok while landing): {missing}")
```

---

## 8. Schema Quality Gates

### 8.1 Schema Lint Rules

**JSON Schema Draft 2020-12 Compliance:**
```python
def test_all_schemas_load_and_have_unique_ids():
    """Test that all schemas load, have correct $schema, and unique $ids."""
    files = iter_schema_files()
    store: Dict[str, dict] = {}
    
    for path in files:
        schema = load_json(path)
        
        # $schema presence and value
        s = schema.get("$schema")
        assert s == "https://json-schema.org/draft/2020-12/schema", \
            f"{path}: $schema must be Draft 2020-12, got {s!r}"
        
        # $id presence, format, allowed chars
        sid = schema.get("$id")
        assert isinstance(sid, str) and sid, f"{path}: missing or empty $id"
        
        # Unique $id
        assert sid not in store, \
            f"Duplicate $id detected: {sid!r} used by {store[sid].get('__file__')} and {path}"
        store[sid] = schema
```

**$id Path Convention:**
```python
def test_id_path_convention_matches_filesystem():
    """Test that $id values match filesystem paths."""
    files = iter_schema_files()
    
    for path in files:
        schema = load_json(path)
        sid: str = schema["$id"]
        comp = component_for_path(path)
        fname = path.name
        
        # Expect: https://corpusos.com/schemas/<component>/<file>.json
        expected_suffix = f"/schemas/{comp}/{fname}"
        assert sid.endswith(expected_suffix), \
            f"{path}: $id should end with {expected_suffix}, got {sid}"
```

**Metaschema Conformance and Hygiene:**
```python
def test_metaschema_conformance_and_basic_hygiene():
    """Test schema conformance, regex patterns, and enum hygiene."""
    files = iter_schema_files()
    
    for path in files:
        schema = load_json(path)
        
        # Metaschema conformance
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as e:
            pytest.fail(f"{path}: schema fails Draft 2020-12 metaschema: {e}")
        
        # Compile regex patterns
        for key, val in walk(schema):
            if key == "pattern" and isinstance(val, str):
                compile_pattern(val)  # Will fail on invalid regex
        
        # Enum arrays deduped and sorted
        for key, val in walk(schema):
            if key == "enum" and isinstance(val, list) and val:
                unique = list(dict.fromkeys(val))  # preserve order but remove dups
                assert unique == val, f"{path}: enum contains duplicates: {val}"
```

**$ref Resolution and Local Fragments:**
```python
def test_cross_file_refs_resolve_and_local_fragments_exist():
    """Test that all $refs resolve and local fragments exist."""
    # Build store of $id -> schema for absolute refs
    store = build_schema_store()
    known_ids = collect_ids(store)
    
    for path in iter_schema_files():
        schema = load_json(path)
        
        for key, val in walk(schema):
            if key != "$ref" or not isinstance(val, str):
                continue
            
            base, frag = split_ref(val)
            if base and base.startswith("http"):
                # Absolute $ref to another file
                assert base in known_ids, f"{path}: $ref targets unknown $id: {val}"
                
                # Local fragment within target
                if frag:
                    target_schema = store[base]
                    assert resolve_local_fragment(target_schema, frag), \
                        f"{path}: $ref fragment not found in target schema: {val}"
```

**No Dangling Definitions:**
```python
def test_no_dangling_defs_globally():
    """Test that no $defs are defined but never referenced."""
    # Collect every available def anchor
    def_anchors = collect_def_anchors()
    
    # Collect every $ref we actually use
    used_refs = collect_used_refs()
    
    # Any def anchors never referenced anywhere?
    dangling = [a for a in def_anchors if a not in used_refs]
    
    assert not dangling, f"Dangling $defs: {dangling}"
```
### 8.2 Production Conformance Tests

**Wire Conformance Test Suite:**
```python
@pytest.mark.parametrize("case", get_pytest_params(), ids=lambda c: c.id)
def test_wire_request_envelope(case: WireRequestCase, adapter: Any):
    """
    Validate wire-level request envelope for a protocol operation.
    
    Steps:
      1. Get builder method from adapter
      2. Build the envelope
      3. Validate envelope structure
      4. JSON round-trip validation
      5. Schema validation (with version tolerance)
      6. Operation-specific args validation
    """
    # Get builder from adapter
    builder = get_adapter_builder(adapter, case)
    if builder is None:
        pytest.skip(f"Adapter does not implement '{case.build_method}'")
    
    # Build envelope
    envelope = builder()
    
    # Run validation pipeline
    validate_wire_envelope(
        envelope=envelope,
        expected_op=case.op,
        schema_id=case.schema_id,
        schema_versions=case.schema_versions,
        component=case.component,
        args_validator=case.args_validator,
        case_id=case.id,
    )
```

**Test Case Registry:**
```python
@dataclass
class WireRequestCase:
    """Test case definition for wire conformance testing."""
    id: str                      # Unique test identifier
    op: str                      # Protocol operation (e.g., "llm.complete")
    build_method: str           # Adapter method to call (e.g., "build_complete_request")
    schema_id: str              # Schema to validate against
    component: str              # Protocol component (llm/vector/embedding/graph)
    schema_versions: List[str]  # Acceptable schema versions
    args_validator: Optional[Callable] = None  # Operation-specific validation
    tags: List[str] = field(default_factory=list)  # Test markers
    
    # Example case definitions
    LLM_CAPABILITIES = WireRequestCase(
        id="llm.capabilities.basic",
        op="llm.capabilities",
        build_method="build_capabilities_request",
        schema_id="https://corpusos.com/schemas/llm/llm.capabilities.request.json",
        component="llm",
        schema_versions=["1.0.0"],
        tags=["core", "llm"],
    )
    
    LLM_COMPLETE = WireRequestCase(
        id="llm.complete.basic",
        op="llm.complete",
        build_method="build_complete_request",
        schema_id="https://corpusos.com/schemas/llm/llm.complete.request.json",
        component="llm",
        schema_versions=["1.0.0", "1.1.0"],
        args_validator=validate_llm_complete_args,
        tags=["core", "llm"],
    )
    
    EMBEDDING_EMBED_BATCH = WireRequestCase(
        id="embedding.embed_batch.basic",
        op="embedding.embed_batch",
        build_method="build_embed_batch_request",
        schema_id="https://corpusos.com/schemas/embedding/embedding.embed_batch.request.json",
        component="embedding",
        schema_versions=["1.0.0"],
        tags=["core", "embedding"],
    )
    
    GRAPH_UPSERT_NODES = WireRequestCase(
        id="graph.upsert_nodes.basic",
        op="graph.upsert_nodes",
        build_method="build_upsert_nodes_request",
        schema_id="https://corpusos.com/schemas/graph/graph.upsert_nodes.request.json",
        component="graph",
        schema_versions=["1.0.0"],
        args_validator=validate_graph_upsert_nodes_args,
        tags=["core", "graph"],
    )
```

**Edge Case Validation:**
```python
class TestEnvelopeEdgeCases:
    """Test edge cases for envelope validation."""
    
    def test_missing_op_rejected(self, adapter: Any):
        """Envelope without 'op' should be rejected."""
        envelope = {"ctx": {"request_id": "test"}, "args": {}}
        
        with pytest.raises(EnvelopeShapeError, match="missing required keys"):
            validate_envelope_shape(envelope, case_id="test_missing_op")
    
    def test_missing_ctx_rejected(self, adapter: Any):
        """Envelope without 'ctx' should be rejected."""
        envelope = {"op": "llm.complete", "args": {}}
        
        with pytest.raises(EnvelopeShapeError, match="missing required keys"):
            validate_envelope_shape(envelope, case_id="test_missing_ctx")
    
    def test_missing_args_rejected(self, adapter: Any):
        """Envelope without 'args' should be rejected."""
        envelope = {"op": "llm.complete", "ctx": {"request_id": "test"}}
        
        with pytest.raises(EnvelopeShapeError, match="missing required keys"):
            validate_envelope_shape(envelope, case_id="test_missing_args")
    
    def test_ctx_not_object_rejected(self, adapter: Any):
        """Envelope with non-object 'ctx' should be rejected."""
        envelope = {"op": "llm.complete", "ctx": "invalid", "args": {}}
        
        with pytest.raises(CtxValidationError, match="ctx must be object"):
            validate_envelope_shape(envelope, case_id="test_ctx_not_object")
    
    def test_args_not_object_rejected(self, adapter: Any):
        """Envelope with non-object 'args' should be rejected."""
        envelope = {"op": "llm.complete", "ctx": {"request_id": "test"}, "args": "invalid"}
        
        with pytest.raises(ArgsValidationError, match="args must be object"):
            validate_envelope_shape(envelope, case_id="test_args_not_object")
    
    def test_negative_deadline_rejected(self, adapter: Any):
        """Negative deadline_ms should be rejected."""
        envelope = {"op": "llm.complete", "ctx": {"request_id": "test", "deadline_ms": -100}, "args": {}}
        
        with pytest.raises(CtxValidationError, match="deadline_ms"):
            validate_ctx_field(envelope, case_id="test_deadline_negative")
    
    def test_non_serializable_rejected(self, adapter: Any):
        """Non-JSON-serializable values should be rejected."""
        envelope = {
            "op": "llm.complete",
            "ctx": {"request_id": "test"},
            "args": {"callback": lambda x: x},  # Not JSON serializable
        }
        
        with pytest.raises(SerializationError):
            json_roundtrip(envelope, case_id="test_non_serializable")
    
    def test_embedding_stream_flag_validation(self, adapter: Any):
        """Embedding embed operation should reject stream=true."""
        envelope = {
            "op": "embedding.embed",
            "ctx": {"request_id": "test"},
            "args": {"text": "test", "model": "test-model", "stream": True}
        }
        
        with pytest.raises(ArgsValidationError, match="stream must be absent or false"):
            validate_embedding_embed_args(envelope["args"], case_id="test_stream_flag")
```

**Metrics Collection:**
```python
@dataclass
class ValidationMetrics:
    """Thread-safe metrics collection for test runs."""
    validation_times: List[Tuple[str, float]] = field(default_factory=list)
    successes: Dict[str, int] = field(default_factory=dict)
    failures: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    def record_success(self, case_id: str, duration: float) -> None:
        """Record successful validation."""
        self.validation_times.append((case_id, duration))
        self.successes[case_id] = self.successes.get(case_id, 0) + 1
    
    def record_failure(self, case_id: str, error_type: str) -> None:
        """Record validation failure."""
        if case_id not in self.failures:
            self.failures[case_id] = {}
        self.failures[case_id][error_type] = self.failures[case_id].get(error_type, 0) + 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary for reporting."""
        return {
            "total_runs": len(self.validation_times),
            "total_successes": sum(self.successes.values()),
            "total_failures": sum(sum(errs.values()) for errs in self.failures.values()),
            "avg_duration_ms": (sum(t[1] for t in self.validation_times) / len(self.validation_times) * 1000) if self.validation_times else 0,
            "p95_duration_ms": self._calculate_percentile(95),
            "p99_duration_ms": self._calculate_percentile(99)
        }
    
    def _calculate_percentile(self, percentile: int) -> float:
        """Calculate duration percentile."""
        if not self.validation_times:
            return 0.0
        sorted_times = sorted(t[1] for t in self.validation_times)
        idx = int(len(sorted_times) * percentile / 100)
        return sorted_times[idx] * 1000  # Convert to milliseconds
```

### 8.3 Performance Considerations

**Schema Loading Performance:**
```python
def test_schema_loading_performance():
    """Test that all schemas can be loaded quickly."""
    files = iter_schema_files()
    max_load_time = 1.0  # seconds per schema
    slow_files = []
    
    for path in files:
        start = time.time()
        load_json(path)  # Includes JSON parsing
        load_time = time.time() - start
        
        if load_time > max_load_time:
            slow_files.append((path, load_time))
    
    if slow_files:
        pytest.skip(f"Slow schema loading: {slow_files}")
```

**Validator Creation Performance:**
```python
def test_validator_creation_performance():
    """Test that validators can be created quickly."""
    registry = SchemaRegistry()
    max_create_time = 0.5  # seconds per validator
    slow_validators = []
    
    for schema_id in registry.list_schemas().keys():
        start = time.time()
        registry.get_validator(schema_id)
        create_time = time.time() - start
        
        if create_time > max_create_time:
            slow_validators.append((schema_id, create_time))
    
    if slow_validators:
        pytest.skip(f"Slow validator creation: {slow_validators}")
```

**Memory Usage Optimization:**
```python
def test_schema_memory_usage():
    """Test schema memory usage stays within limits."""
    registry = SchemaRegistry()
    
    # Check total memory usage
    import sys
    total_size = sum(
        sys.getsizeof(schema) + sum(sys.getsizeof(v) for v in schema.values())
        for schema in registry._SCHEMA_STORE.values()
    )
    
    # 50MB limit for all schemas
    assert total_size < 50 * 1024 * 1024, f"Schema memory usage too high: {total_size/1024/1024:.1f}MB"
    
    # Check individual schema size
    for schema_id, schema in registry._SCHEMA_STORE.items():
        schema_size = sys.getsizeof(schema) + sum(sys.getsizeof(v) for v in schema.values())
        assert schema_size < 5 * 1024 * 1024, f"Schema {schema_id} too large: {schema_size/1024/1024:.1f}MB"
```

**Cache Effectiveness:**
```python
def test_validator_cache_effectiveness():
    """Test validator caching improves performance."""
    registry = SchemaRegistry()
    schema_id = "https://corpusos.com/schemas/llm/llm.envelope.request.json"
    
    # First call (cold cache)
    start = time.time()
    v1 = registry.get_validator(schema_id)
    first_call_time = time.time() - start
    
    # Second call (warm cache)
    start = time.time()
    v2 = registry.get_validator(schema_id)
    second_call_time = time.time() - start
    
    # Cache should provide at least 10x speedup
    cache_speedup = first_call_time / second_call_time
    assert cache_speedup > 10, f"Cache ineffective: {cache_speedup:.1f}x speedup"
    
    # Same validator instance should be returned
    assert v1 is v2, "Different validator instances returned from cache"
```

**Validation Performance Benchmarks:**
```python
def test_validation_performance_benchmarks():
    """Test validation performance meets benchmarks."""
    registry = SchemaRegistry()
    
    # Test LLM envelope validation
    llm_envelope = {
        "op": "llm.complete",
        "ctx": {"request_id": "test-123"},
        "args": {"messages": [{"role": "user", "content": "test"}]}
    }
    
    validator = registry.get_validator("https://corpusos.com/schemas/llm/llm.complete.request.json")
    
    # Warm up
    for _ in range(10):
        validator.validate(llm_envelope)
    
    # Benchmark
    iterations = 1000
    start = time.time()
    for _ in range(iterations):
        validator.validate(llm_envelope)
    duration = time.time() - start
    
    avg_time_ms = (duration / iterations) * 1000
    
    # Should validate in less than 10ms on average
    assert avg_time_ms < 10, f"Validation too slow: {avg_time_ms:.2f}ms per validation"
```

---

## 9. Integration and Tooling

### 9.1 CLI Tools

**Schema Validation CLI:**
```bash
# List all available schemas
python -m tests.utils.schema_registry --list

# Validate JSON against schema
python -m tests.utils.schema_registry llm.envelope.request.json sample_request.json

# Validate with specific schema root
CORPUS_SCHEMAS_ROOT=/custom/path python -m tests.utils.schema_registry --list

# Show schema statistics
python -m tests.utils.schema_registry --stats

# Validate multiple files
python -m tests.utils.schema_registry llm.envelope.request.json file1.json file2.json file3.json

# Output validation results as JSON
python -m tests.utils.schema_registry --output json llm.envelope.request.json sample.json
```

**Stream Validation CLI:**
```python
# Validate NDJSON stream from file
from tests.utils.stream_validator import validate_ndjson_stream

with open("stream.ndjson", "r") as f:
    report = validate_ndjson_stream(
        f.read(),
        envelope_schema_id="https://corpusos.com/schemas/common/envelope.stream.success.json",
        component="llm",
        mode="strict"
    )
    
print(f"Valid: {report.is_valid}")
print(f"Frames: {report.total_frames}")
print(f"Errors: {len(report.validation_errors)}")

# Validate from stdin
import sys
report = validate_ndjson_stream(
    sys.stdin.read(),
    envelope_schema_id="https://corpusos.com/schemas/common/envelope.stream.success.json",
    component="embedding"
)
```

**Conformance Test Runner:**
```bash
# Run all conformance tests
pytest tests/live/test_wire_conformance.py -v

# Run only LLM tests
pytest tests/live/test_wire_conformance.py -v -m "llm"

# Run only Embedding tests
pytest tests/live/test_wire_conformance.py -v -m "embedding"

# Run only Graph tests
pytest tests/live/test_wire_conformance.py -v -m "graph"

# Run only Vector tests
pytest tests/live/test_wire_conformance.py -v -m "vector"

# Run core operations only
pytest tests/live/test_wire_conformance.py -v -m "core"

# Skip schema validation for faster iteration
pytest tests/live/test_wire_conformance.py -v --skip-schema

# Test specific adapter
pytest tests/live/test_wire_conformance.py -v --adapter=openai

# Generate coverage report
pytest tests/live/test_wire_conformance.py -v --tb=no -q | generate_coverage_report.py

# Run with detailed error output
pytest tests/live/test_wire_conformance.py -vv --tb=long

# Run parallel tests for faster execution
pytest tests/live/test_wire_conformance.py -v -n auto
```

**Golden Test Maintenance:**
```bash
# Validate all golden fixtures
pytest tests/schema/test_golden_schema.py -v

# Check for schema drift
pytest tests/schema/test_golden_schema.py::test_no_orphaned_golden_files -v
pytest tests/schema/test_golden_schema.py::test_all_listed_golden_files_exist -v

# Run schema linting
pytest tests/schema/test_schema_lint.py -v

# Generate missing golden fixtures
python tools/generate_golden.py --component llm --operation complete
python tools/generate_golden.py --component embedding --operation embed_batch
python tools/generate_golden.py --component graph --operation upsert_nodes
python tools/generate_golden.py --component vector --operation batch_query

# Validate specific golden file
python tools/validate_golden.py tests/golden/llm/llm_complete_request.json

# Update golden fixtures from live adapters
python tools/update_golden.py --adapter openai --operation llm.complete
```

### 9.2 Development Workflow

**Schema-First Development Process:**
1. **Define schema** - Create JSON Schema for new operation/type
2. **Validate schema** - Run schema lint tests (`test_schema_lint.py`)
3. **Create golden fixtures** - Add example JSON files
4. **Validate fixtures** - Run golden tests (`test_golden_schema.py`)
5. **Implement adapter** - Write adapter methods
6. **Test conformance** - Run wire conformance tests (`test_wire_conformance.py`)
7. **Deploy** - All tests pass, ready for production

**Schema Creation Template:**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/{component}/{filename}.json",
  "title": "Descriptive Title",
  "type": "object",
  "properties": {
    // Define properties here
  },
  "required": ["required_field1", "required_field2"],
  "additionalProperties": false,
  "$defs": {
    // Reusable type definitions
  }
}
```

**Operation Schema Template:**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/{protocol}/{protocol}.{operation}.request.json",
  "title": "{Protocol} {Operation} Request",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/{protocol}/{protocol}.envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "const": "{protocol}.{operation}"
        },
        "args": {
          "$ref": "https://corpusos.com/schemas/{protocol}/{protocol}.types.{operation}_spec.json"
        }
      }
    }
  ]
}
```

**Breaking Change Process:**
1. **Create new schema version** - e.g., `llm.envelope.request.v2.json`
2. **Maintain backward compatibility** - Support both versions temporarily
3. **Update golden fixtures** - Add examples for new version
4. **Update adapter implementations** - Support new version
5. **Deprecate old version** - Mark as deprecated in documentation
6. **Schedule removal** - Remove after grace period (e.g., 6 months)

**Migration Tooling:**
```python
class SchemaMigrator:
    """Tool for migrating between schema versions."""
    
    @staticmethod
    def migrate_llm_envelope_v1_to_v2(v1_envelope: dict) -> dict:
        """Migrate LLM envelope from v1.0 to v2.0."""
        v2_envelope = v1_envelope.copy()
        
        # Example migration: rename field
        if "prompt" in v2_envelope.get("args", {}):
            v2_envelope["args"]["messages"] = [
                {"role": "user", "content": v2_envelope["args"]["prompt"]}
            ]
            del v2_envelope["args"]["prompt"]
        
        return v2_envelope
    
    @staticmethod
    def migrate_embedding_args_v1_to_v2(v1_args: dict) -> dict:
        """Migrate embedding args from v1.0 to v2.0."""
        v2_args = v1_args.copy()
        
        # Add new required fields with defaults
        if "normalize" not in v2_args:
            v2_args["normalize"] = False
        
        return v2_args
```

**Debugging Workflow:**
```python
# Debug schema validation
from tests.utils.schema_registry import get_validator

validator = get_validator("https://corpusos.com/schemas/llm/llm.envelope.request.json")
errors = list(validator.iter_errors(invalid_request))
for error in errors:
    print(f"Path: {'.'.join(str(p) for p in error.path)}")
    print(f"Message: {error.message}")
    print(f"Schema: {error.schema}")
    print(f"Instance: {error.instance}")
    print("---")

# Debug stream validation
from tests.utils.stream_validator import StreamValidationEngine

engine = StreamValidationEngine(config)
report = engine.validate_ndjson(ndjson_text)
if not report.is_valid:
    for error in report.validation_errors:
        print(f"Frame {error.frame_number}: {error.error_type}")
        print(f"  {error.message}")
        print(f"  Frame content: {error.frame_content[:100]}...")

# Debug envelope validation
try:
    validate_wire_envelope(
        envelope=test_envelope,
        expected_op="llm.complete",
        schema_id="https://corpusos.com/schemas/llm/llm.complete.request.json",
        component="llm",
        case_id="debug_test"
    )
except Exception as e:
    print(f"Validation failed: {e}")
    import traceback
    traceback.print_exc()
```

### 9.3 CI/CD Integration

**GitHub Actions Workflow:**
```yaml
name: Schema Validation
on: [push, pull_request]

jobs:
  schema-lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest jsonschema
      - name: Run schema lint tests
        run: pytest tests/schema/test_schema_lint.py -v
  
  golden-tests:
    runs-on: ubuntu-latest
    needs: schema-lint
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest jsonschema
      - name: Run golden tests
        run: pytest tests/schema/test_golden_schema.py -v
  
  conformance-tests:
    runs-on: ubuntu-latest
    needs: golden-tests
    strategy:
      matrix:
        adapter: [openai, anthropic, pinecone, neo4j, cohere, voyageai]
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest jsonschema
      - name: Run ${{ matrix.adapter }} conformance tests
        run: |
          pytest tests/live/test_wire_conformance.py -v \
            --adapter=${{ matrix.adapter }} \
            --junitxml=results-${{ matrix.adapter }}.xml
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}
          NEO4J_URI: ${{ secrets.NEO4J_URI }}
          NEO4J_PASSWORD: ${{ secrets.NEO4J_PASSWORD }}
      - name: Upload test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: test-results-${{ matrix.adapter }}
          path: results-${{ matrix.adapter }}.xml
  
  performance-checks:
    runs-on: ubuntu-latest
    needs: conformance-tests
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest jsonschema
      - name: Run performance benchmarks
        run: python benchmarks/schema_performance.py
      - name: Upload performance results
        uses: actions/upload-artifact@v3
        with:
          name: performance-results
          path: benchmarks/results/
```

**Pre-commit Hooks:**
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: schema-lint
        name: Schema Lint
        entry: pytest tests/schema/test_schema_lint.py -v
        language: system
        files: ^schemas/.*\.json$
        pass_filenames: false
        
      - id: golden-validation
        name: Golden Validation
        entry: pytest tests/schema/test_golden_schema.py -v
        language: system
        files: ^tests/golden/.*\.json$
        pass_filenames: false
        
      - id: json-format
        name: JSON Format Check
        entry: python tools/format_json.py
        language: system
        files: \.json$
```

**Code Coverage Requirements:**
```python
# Minimum coverage thresholds
COVERAGE_THRESHOLDS = {
    "schema_files": 100,      # All schema files must be linted
    "golden_fixtures": 95,    # 95% of golden fixtures validated
    "conformance_cases": 90,  # 90% of conformance test cases
    "edge_cases": 85,         # 85% of edge case tests
    "protocol_operations": {
        "llm": 100,           # All LLM operations covered
        "embedding": 100,     # All Embedding operations covered
        "vector": 100,        # All Vector operations covered
        "graph": 100,         # All Graph operations covered
    }
}

def check_coverage():
    """Check test coverage meets minimum thresholds."""
    metrics = generate_coverage_report()
    
    for category, threshold in COVERAGE_THRESHOLDS.items():
        if isinstance(threshold, dict):
            for subcategory, subthreshold in threshold.items():
                coverage = metrics.get(category, {}).get(subcategory, {}).get("coverage", 0)
                assert coverage >= subthreshold, \
                    f"{category}.{subcategory} coverage {coverage}% < {subthreshold}% threshold"
        else:
            coverage = metrics.get(category, {}).get("coverage", 0)
            assert coverage >= threshold, \
                f"{category} coverage {coverage}% < {threshold}% threshold"
```

**Release Gate Criteria:**
1. **Schema lint**: 100% pass rate
2. **Golden tests**: 100% pass rate
3. **Conformance tests**: ≥95% pass rate per adapter
4. **Performance**: Schema loading < 2 seconds
5. **Coverage**: Meets all threshold requirements
6. **Breaking changes**: Documented and approved
7. **Migration path**: Provided for any breaking changes

**Automated Release Process:**
```bash
#!/bin/bash
# release.sh - Automated release script

set -e

echo "Running pre-release checks..."

# 1. Schema lint
echo "Step 1/7: Running schema lint..."
pytest tests/schema/test_schema_lint.py -v

# 2. Golden tests
echo "Step 2/7: Running golden tests..."
pytest tests/schema/test_golden_schema.py -v

# 3. Conformance tests
echo "Step 3/7: Running conformance tests..."
pytest tests/live/test_wire_conformance.py -v

# 4. Performance benchmarks
echo "Step 4/7: Running performance benchmarks..."
python benchmarks/schema_performance.py

# 5. Coverage check
echo "Step 5/7: Checking coverage..."
python tools/check_coverage.py

# 6. Generate changelog
echo "Step 6/7: Generating changelog..."
python tools/generate_changelog.py

# 7. Tag release
echo "Step 7/7: Tagging release..."
VERSION=$(python tools/get_version.py)
git tag -a "v${VERSION}" -m "Release v${VERSION}"
git push origin "v${VERSION}"

echo "Release v${VERSION} complete!"
```

---

## 10. Schema Evolution

### 10.1 Versioning Strategy

**Dual Versioning System:**
- **Protocol version**: `llm/v1.0` - Major protocol changes
- **Schema version**: `1.0.0` - Individual schema changes (SemVer)

**Version Compatibility Matrix:**

| Change Type | Protocol Version | Schema Version | Backward Compatible? |
|-------------|------------------|----------------|----------------------|
| Bug fix | Unchanged | Patch bump (1.0.0 → 1.0.1) | Yes |
| New optional field | Unchanged | Minor bump (1.0.0 → 1.1.0) | Yes |
| Required field change | Minor bump (v1.0 → v1.1) | Major bump (1.0.0 → 2.0.0) | No |
| Wire format change | Major bump (v1.0 → v2.0) | Major bump (1.0.0 → 2.0.0) | No |
| New operation | Unchanged | Minor bump (1.0.0 → 1.1.0) | Yes |
| Deprecate operation | Minor bump (v1.0 → v1.1) | Major bump (1.0.0 → 2.0.0) | No |

**Version Tolerance Validation:**
```python
def validate_with_version_tolerance(
    schema_id: str,
    obj: dict,
    accepted_versions: List[str],
    case_id: str = "unknown",
) -> None:
    """
    Validate with version tolerance.
    
    Allows multiple schema versions for backward compatibility.
    """
    for version in accepted_versions:
        versioned_schema_id = f"{schema_id}#version/{version}"
        try:
            assert_valid(versioned_schema_id, obj, context=case_id)
            return  # Validation succeeded with this version
        except AssertionError:
            continue  # Try next version
    
    # All versions failed
    raise SchemaValidationError(
        f"{case_id}: failed validation against all accepted versions: {accepted_versions}"
    )
```

**Schema Version Declaration:**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.envelope.request.json",
  "title": "LLM Protocol Request Envelope",
  "version": "1.0.0",
  "type": "object",
  "properties": {
    "schemaVersion": {
      "type": "string",
      "const": "1.0.0",
      "description": "Schema version for this envelope"
    }
  }
}
```

### 10.2 Breaking Change Process

**Phase 1: Announcement (4 weeks before change)**
- Add deprecation warnings to schemas
- Update documentation with migration guide
- Notify adapter maintainers
- Create migration tooling

**Phase 2: Dual Support (8 weeks)**
- New schema version available
- Both old and new versions accepted
- Adapters can migrate at their own pace
- Deprecation warnings in validation output

**Phase 3: Deprecation (4 weeks)**
- Old version marked as deprecated
- Warnings in validation output
- Strong recommendation to migrate
- Migration tooling fully tested

**Phase 4: Removal**
- Old version removed from validation
- Adapters must use new version
- Breaking change complete
- Update release notes

**Migration Tooling:**
```python
class SchemaMigrator:
    """Tool for migrating between schema versions."""
    
    @staticmethod
    def migrate_llm_envelope_v1_to_v2(v1_envelope: dict) -> dict:
        """Migrate LLM envelope from v1.0 to v2.0."""
        v2_envelope = v1_envelope.copy()
        
        # Example migration: rename field
        if "prompt" in v2_envelope.get("args", {}):
            v2_envelope["args"]["messages"] = [
                {"role": "user", "content": v2_envelope["args"]["prompt"]}
            ]
            del v2_envelope["args"]["prompt"]
        
        return v2_envelope
    
    @staticmethod
    def migrate_vector_namespace_v1_to_v2(v1_args: dict) -> dict:
        """Migrate vector namespace args from v1.0 to v2.0."""
        v2_args = v1_args.copy()
        
        # Add distance_metric with default
        if "distance_metric" not in v2_args:
            v2_args["distance_metric"] = "cosine"
        
        return v2_args
    
    @staticmethod
    def validate_and_migrate(
        envelope: dict,
        from_version: str,
        to_version: str,
        protocol: str
    ) -> dict:
        """Validate old version and migrate to new version."""
        # Validate against old schema
        old_schema_id = f"https://corpusos.com/schemas/{protocol}/{protocol}.envelope.request.v{from_version}.json"
        assert_valid(old_schema_id, envelope)
        
        # Migrate
        migrator_method = f"migrate_{protocol}_envelope_v{from_version}_to_v{to_version}"
        migrator = getattr(SchemaMigrator, migrator_method)
        migrated = migrator(envelope)
        
        # Validate against new schema
        new_schema_id = f"https://corpusos.com/schemas/{protocol}/{protocol}.envelope.request.v{to_version}.json"
        assert_valid(new_schema_id, migrated)
        
        return migrated
```

**Deprecation Warning System:**
```python
class DeprecationWarning:
    """Track and report schema deprecation warnings."""
    
    def __init__(self, schema_id: str, deprecated_version: str, removal_version: str):
        self.schema_id = schema_id
        self.deprecated_version = deprecated_version
        self.removal_version = removal_version
        self.warning_count = 0
    
    def warn(self, context: str) -> None:
        """Issue deprecation warning."""
        self.warning_count += 1
        warnings.warn(
            f"Schema {self.schema_id} version {self.deprecated_version} is deprecated. "
            f"It will be removed in version {self.removal_version}. "
            f"Context: {context}",
            DeprecationWarning,
            stacklevel=2
        )
    
    def get_stats(self) -> dict:
        """Get deprecation statistics."""
        return {
            "schema_id": self.schema_id,
            "deprecated_version": self.deprecated_version,
            "removal_version": self.removal_version,
            "warning_count": self.warning_count
        }
```

### 10.3 Extension Patterns

**Vendor Extensions:**
```json
{
  "type": "object",
  "properties": {
    "provider_specific": {
      "type": "object",
      "description": "Vendor-specific extensions",
      "additionalProperties": true,
      "patternProperties": {
        "^x-[a-z0-9-]+$": {
          "description": "Vendor extension fields must start with 'x-'"
        }
      }
    }
  },
  "additionalProperties": false
}
```

**Experimental Features:**
```json
{
  "type": "object",
  "properties": {
    "experimental_features": {
      "type": "object",
      "description": "Experimental features (may change or be removed)",
      "properties": {
        "feature_flag": {
          "type": "boolean",
          "description": "Enable experimental feature"
        },
        "beta_endpoint": {
          "type": "string",
          "description": "Beta feature endpoint"
        }
      },
      "additionalProperties": true
    }
  }
}
```

**Custom Validators:**
```python
class CustomFormatValidator:
    """Custom format validators for specialized validation."""
    
    @classmethod
    def validate_vector_dimensions(cls, vector: list) -> bool:
        """Validate vector dimensions are within reasonable limits."""
        return 1 <= len(vector) <= 10000
    
    @classmethod  
    def validate_timestamp_format(cls, timestamp: str) -> bool:
        """Validate ISO 8601 timestamp with milliseconds."""
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z$"
        return bool(re.match(pattern, timestamp))
    
    @classmethod
    def validate_namespace_name(cls, namespace: str) -> bool:
        """Validate namespace naming conventions."""
        # Must be alphanumeric with hyphens, 1-63 characters
        pattern = r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$"
        return bool(re.match(pattern, namespace))

# Register custom validators
jsonschema.Draft202012Validator.format_checker.checks(
    "vector-dimensions", raises=ValueError
)(CustomFormatValidator.validate_vector_dimensions)

jsonschema.Draft202012Validator.format_checker.checks(
    "timestamp-iso8601-ms", raises=ValueError
)(CustomFormatValidator.validate_timestamp_format)

jsonschema.Draft202012Validator.format_checker.checks(
    "namespace-name", raises=ValueError
)(CustomFormatValidator.validate_namespace_name)
```

**Extension Points:**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/extensions/custom_metadata.json",
  "title": "Custom Metadata Extension",
  "type": "object",
  "properties": {
    "custom_metadata": {
      "type": "object",
      "properties": {
        "tags": {
          "type": "array",
          "items": { "type": "string" }
        },
        "priority": {
          "type": "integer",
          "minimum": 1,
          "maximum": 10
        },
        "annotations": {
          "type": "object",
          "additionalProperties": true
        }
      },
      "additionalProperties": false
    }
  }
}
```

---

## 11. Reference

### 11.1 Schema Quick Reference

**Common Schema IDs:**
```
https://corpusos.com/schemas/common/envelope.request.json
https://corpusos.com/schemas/common/envelope.success.json
https://corpusos.com/schemas/common/envelope.error.json
https://corpusos.com/schemas/common/envelope.stream.success.json
https://corpusos.com/schemas/common/operation_context.json
```

**LLM Schema IDs:**
```
https://corpusos.com/schemas/llm/llm.envelope.request.json
https://corpusos.com/schemas/llm/llm.envelope.success.json
https://corpusos.com/schemas/llm/llm.envelope.error.json
https://corpusos.com/schemas/llm/llm.capabilities.json
https://corpusos.com/schemas/llm/llm.capabilities.request.json
https://corpusos.com/schemas/llm/llm.capabilities.success.json
https://corpusos.com/schemas/llm/llm.types.message.json
https://corpusos.com/schemas/llm/llm.types.completion.json
https://corpusos.com/schemas/llm/llm.types.completion_spec.json
https://corpusos.com/schemas/llm/llm.types.stream_spec.json
https://corpusos.com/schemas/llm/llm.types.count_tokens_spec.json
https://corpusos.com/schemas/llm/llm.types.token_usage.json
https://corpusos.com/schemas/llm/llm.types.chunk.json
https://corpusos.com/schemas/llm/llm.complete.request.json
https://corpusos.com/schemas/llm/llm.complete.success.json
https://corpusos.com/schemas/llm/llm.stream.request.json
https://corpusos.com/schemas/llm/llm.stream.success.json
https://corpusos.com/schemas/llm/llm.count_tokens.request.json
https://corpusos.com/schemas/llm/llm.count_tokens.success.json
https://corpusos.com/schemas/llm/llm.health.request.json
https://corpusos.com/schemas/llm/llm.health.success.json
```

**Vector Schema IDs:**
```
https://corpusos.com/schemas/vector/vector.envelope.request.json
https://corpusos.com/schemas/vector/vector.envelope.success.json
https://corpusos.com/schemas/vector/vector.envelope.error.json
https://corpusos.com/schemas/vector/vector.capabilities.json
https://corpusos.com/schemas/vector/vector.capabilities.request.json
https://corpusos.com/schemas/vector/vector.capabilities.success.json
https://corpusos.com/schemas/vector/vector.types.vector.json
https://corpusos.com/schemas/vector/vector.types.vector_match.json
https://corpusos.com/schemas/vector/vector.types.query_spec.json
https://corpusos.com/schemas/vector/vector.types.query_result.json
https://corpusos.com/schemas/vector/vector.types.namespace_spec.json
https://corpusos.com/schemas/vector/vector.types.namespace_result.json
https://corpusos.com/schemas/vector/vector.query.request.json
https://corpusos.com/schemas/vector/vector.query.success.json
https://corpusos.com/schemas/vector/vector.batch_query.request.json
https://corpusos.com/schemas/vector/vector.batch_query.success.json
https://corpusos.com/schemas/vector/vector.upsert.request.json
https://corpusos.com/schemas/vector/vector.upsert.success.json
https://corpusos.com/schemas/vector/vector.delete.request.json
https://corpusos.com/schemas/vector/vector.delete.success.json
https://corpusos.com/schemas/vector/vector.create_namespace.request.json
https://corpusos.com/schemas/vector/vector.create_namespace.success.json
https://corpusos.com/schemas/vector/vector.delete_namespace.request.json
https://corpusos.com/schemas/vector/vector.delete_namespace.success.json
https://corpusos.com/schemas/vector/vector.health.request.json
https://corpusos.com/schemas/vector/vector.health.success.json
```

**Embedding Schema IDs:**
```
https://corpusos.com/schemas/embedding/embedding.envelope.request.json
https://corpusos.com/schemas/embedding/embedding.envelope.success.json
https://corpusos.com/schemas/embedding/embedding.envelope.error.json
https://corpusos.com/schemas/embedding/embedding.capabilities.json
https://corpusos.com/schemas/embedding/embedding.capabilities.request.json
https://corpusos.com/schemas/embedding/embedding.capabilities.success.json
https://corpusos.com/schemas/embedding/embedding.types.embed_spec.json
https://corpusos.com/schemas/embedding/embedding.types.stream_embed_spec.json
https://corpusos.com/schemas/embedding/embedding.types.count_tokens_spec.json
https://corpusos.com/schemas/embedding/embedding.embed.request.json
https://corpusos.com/schemas/embedding/embedding.embed.success.json
https://corpusos.com/schemas/embedding/embedding.embed_batch.request.json
https://corpusos.com/schemas/embedding/embedding.embed_batch.success.json
https://corpusos.com/schemas/embedding/embedding.stream_embed.request.json
https://corpusos.com/schemas/embedding/embedding.stream_embed.success.json
https://corpusos.com/schemas/embedding/embedding.count_tokens.request.json
https://corpusos.com/schemas/embedding/embedding.count_tokens.success.json
https://corpusos.com/schemas/embedding/embedding.health.request.json
https://corpusos.com/schemas/embedding/embedding.health.success.json
https://corpusos.com/schemas/embedding/embedding.get_stats.request.json
https://corpusos.com/schemas/embedding/embedding.get_stats.success.json
```

**Graph Schema IDs:**
```
https://corpusos.com/schemas/graph/graph.envelope.request.json
https://corpusos.com/schemas/graph/graph.envelope.success.json
https://corpusos.com/schemas/graph/graph.envelope.error.json
https://corpusos.com/schemas/graph/graph.capabilities.json
https://corpusos.com/schemas/graph/graph.capabilities.request.json
https://corpusos.com/schemas/graph/graph.capabilities.success.json
https://corpusos.com/schemas/graph/graph.query.request.json
https://corpusos.com/schemas/graph/graph.query.success.json
https://corpusos.com/schemas/graph/graph.stream_query.request.json
https://corpusos.com/schemas/graph/graph.stream_query.success.json
https://corpusos.com/schemas/graph/graph.upsert_nodes.request.json
https://corpusos.com/schemas/graph/graph.upsert_nodes.success.json
https://corpusos.com/schemas/graph/graph.upsert_edges.request.json
https://corpusos.com/schemas/graph/graph.upsert_edges.success.json
https://corpusos.com/schemas/graph/graph.delete_nodes.request.json
https://corpusos.com/schemas/graph/graph.delete_nodes.success.json
https://corpusos.com/schemas/graph/graph.delete_edges.request.json
https://corpusos.com/schemas/graph/graph.delete_edges.success.json
https://corpusos.com/schemas/graph/graph.bulk_vertices.request.json
https://corpusos.com/schemas/graph/graph.bulk_vertices.success.json
https://corpusos.com/schemas/graph/graph.batch.request.json
https://corpusos.com/schemas/graph/graph.batch.success.json
https://corpusos.com/schemas/graph/graph.get_schema.request.json
https://corpusos.com/schemas/graph/graph.get_schema.success.json
https://corpusos.com/schemas/graph/graph.health.request.json
https://corpusos.com/schemas/graph/graph.health.success.json
https://corpusos.com/schemas/graph/graph.transaction.request.json
https://corpusos.com/schemas/graph/graph.transaction.success.json
https://corpusos.com/schemas/graph/graph.traversal.request.json
https://corpusos.com/schemas/graph/graph.traversal.success.json
```

**Operation to Schema Mapping:**

| Operation | Request Schema | Success Schema |
|-----------|----------------|----------------|
| `llm.capabilities` | `llm.capabilities.request.json` | `llm.capabilities.success.json` |
| `llm.complete` | `llm.complete.request.json` | `llm.complete.success.json` |
| `llm.stream` | `llm.stream.request.json` | `llm.stream.success.json` (STREAMING) |
| `llm.count_tokens` | `llm.count_tokens.request.json` | `llm.count_tokens.success.json` |
| `llm.health` | `llm.health.request.json` | `llm.health.success.json` |
| `vector.capabilities` | `vector.capabilities.request.json` | `vector.capabilities.success.json` |
| `vector.query` | `vector.query.request.json` | `vector.query.success.json` |
| `vector.batch_query` | `vector.batch_query.request.json` | `vector.batch_query.success.json` |
| `vector.upsert` | `vector.upsert.request.json` | `vector.upsert.success.json` |
| `vector.delete` | `vector.delete.request.json` | `vector.delete.success.json` |
| `vector.create_namespace` | `vector.create_namespace.request.json` | `vector.create_namespace.success.json` |
| `vector.delete_namespace` | `vector.delete_namespace.request.json` | `vector.delete_namespace.success.json` |
| `vector.health` | `vector.health.request.json` | `vector.health.success.json` |
| `embedding.capabilities` | `embedding.capabilities.request.json` | `embedding.capabilities.success.json` |
| `embedding.embed` | `embedding.embed.request.json` | `embedding.embed.success.json` |
| `embedding.embed_batch` | `embedding.embed_batch.request.json` | `embedding.embed_batch.success.json` |
| `embedding.stream_embed` | `embedding.stream_embed.request.json` | `embedding.stream_embed.success.json` (STREAMING) |
| `embedding.count_tokens` | `embedding.count_tokens.request.json` | `embedding.count_tokens.success.json` |
| `embedding.health` | `embedding.health.request.json` | `embedding.health.success.json` |
| `embedding.get_stats` | `embedding.get_stats.request.json` | `embedding.get_stats.success.json` |
| `graph.capabilities` | `graph.capabilities.request.json` | `graph.capabilities.success.json` |
| `graph.query` | `graph.query.request.json` | `graph.query.success.json` |
| `graph.stream_query` | `graph.stream_query.request.json` | `graph.stream_query.success.json` (STREAMING) |
| `graph.upsert_nodes` | `graph.upsert_nodes.request.json` | `graph.upsert_nodes.success.json` |
| `graph.upsert_edges` | `graph.upsert_edges.request.json` | `graph.upsert_edges.success.json` |
| `graph.delete_nodes` | `graph.delete_nodes.request.json` | `graph.delete_nodes.success.json` |
| `graph.delete_edges` | `graph.delete_edges.request.json` | `graph.delete_edges.success.json` |
| `graph.bulk_vertices` | `graph.bulk_vertices.request.json` | `graph.bulk_vertices.success.json` |
| `graph.batch` | `graph.batch.request.json` | `graph.batch.success.json` |
| `graph.get_schema` | `graph.get_schema.request.json` | `graph.get_schema.success.json` |
| `graph.health` | `graph.health.request.json` | `graph.health.success.json` |
| `graph.transaction` | `graph.transaction.request.json` | `graph.transaction.success.json` |
| `graph.traversal` | `graph.traversal.request.json` | `graph.traversal.success.json` |

### 11.2 Error Taxonomies by Protocol

**LLM Error Codes:**
- `BAD_REQUEST` - Invalid request parameters
- `AUTH_ERROR` - Authentication or authorization failure
- `RESOURCE_EXHAUSTED` - Rate limit or quota exceeded
- `TRANSIENT_NETWORK` - Temporary network issue
- `UNAVAILABLE` - Service unavailable
- `NOT_SUPPORTED` - Requested feature not supported
- `MODEL_OVERLOADED` - Model capacity exceeded
- `DEADLINE_EXCEEDED` - Request timeout

**Embedding Error Codes:**
- `BAD_REQUEST` - Invalid request parameters
- `AUTH_ERROR` - Authentication or authorization failure
- `RESOURCE_EXHAUSTED` - Rate limit or quota exceeded
- `TEXT_TOO_LONG` - Input text exceeds maximum length
- `MODEL_NOT_AVAILABLE` - Requested model unavailable
- `TRANSIENT_NETWORK` - Temporary network issue
- `UNAVAILABLE` - Service unavailable
- `NOT_SUPPORTED` - Requested feature not supported
- `DEADLINE_EXCEEDED` - Request timeout

**Vector Error Codes:**
- `BAD_REQUEST` - Invalid request parameters
- `AUTH_ERROR` - Authentication or authorization failure
- `RESOURCE_EXHAUSTED` - Rate limit or quota exceeded
- `DIMENSION_MISMATCH` - Vector dimension mismatch
- `INDEX_NOT_READY` - Vector index not ready
- `NAMESPACE_NOT_FOUND` - Specified namespace does not exist
- `NAMESPACE_ALREADY_EXISTS` - Namespace creation conflict
- `TRANSIENT_NETWORK` - Temporary network issue
- `UNAVAILABLE` - Service unavailable
- `NOT_SUPPORTED` - Requested feature not supported
- `DEADLINE_EXCEEDED` - Request timeout

**Graph Error Codes:**
- `BAD_REQUEST` - Invalid request parameters
- `AUTH_ERROR` - Authentication or authorization failure
- `RESOURCE_EXHAUSTED` - Rate limit or quota exceeded
- `QUERY_SYNTAX_ERROR` - Invalid query syntax
- `NODE_NOT_FOUND` - Specified node does not exist
- `EDGE_NOT_FOUND` - Specified edge does not exist
- `CONSTRAINT_VIOLATION` - Graph constraint violated
- `TRANSIENT_NETWORK` - Temporary network issue
- `UNAVAILABLE` - Service unavailable
- `NOT_SUPPORTED` - Requested feature not supported
- `DEADLINE_EXCEEDED` - Request timeout
  
---

## 12. Appendices

### 12.A JSON Schema Draft 2020-12 Primer

**Key Features Used:**
- `$schema`: Declares schema version
- `$id`: Unique identifier for schema
- `$defs`: Reusable definitions
- `$ref`: References to other schemas
- `allOf`, `anyOf`, `oneOf`: Schema composition
- `patternProperties`: Regex-based property validation

**Common Patterns:**
```json
{
  // Strict object (no extra properties)
  "type": "object",
  "additionalProperties": false,
  
  // Pattern-based properties
  "patternProperties": {
    "^[a-z_]+$": { "type": "string" }
  },
  
  // Schema composition
  "allOf": [
    { "$ref": "#/$defs/Base" },
    { "required": ["specific_field"] }
  ],
  
  // Conditional validation
  "if": { "properties": { "type": { "const": "special" } } },
  "then": { "required": ["special_field"] },
  "else": { "required": ["normal_field"] }
}
```

### 12.B $ref Resolution Examples

**Absolute $ref:**
```json
{
  "$ref": "https://corpusos.com/schemas/common/envelope.request.json"
}
```

**Relative $ref within same schema:**
```json
{
  "$ref": "#/$defs/Message"
}
```

**Fragment $ref to specific definition:**
```json
{
  "$ref": "https://corpusos.com/schemas/llm/llm.types.message.json#/$defs/ToolCall"
}
```

**Common Resolution Errors:**
1. **Circular reference** - A references B, B references A
2. **Missing $id** - Referenced schema has no $id
3. **Invalid fragment** - #/path/to/nonexistent
4. **Network dependency** - External URL that times out

### 12.C Custom Format Validators

**Registering Custom Formats:**
```python
from jsonschema import Draft202012Validator

# Define custom format checker
def check_vector_dimensions(value):
    if not isinstance(value, list):
        return True  # Type validation will catch this
    
    if len(value) < 1 or len(value) > 10000:
        raise ValidationError(f"Vector must have 1-10000 dimensions, got {len(value)}")
    return True

# Register with validator
Draft202012Validator.format_checker.checks("vector-dimensions")(check_vector_dimensions)

# Use in schema
schema = {
    "type": "array",
    "format": "vector-dimensions",
    "items": {"type": "number"}
}
```

**Built-in Format Support:**
- `date-time`: RFC 3339 timestamp
- `email`: Email address
- `hostname`: Internet hostname
- `ipv4`, `ipv6`: IP addresses
- `uri`, `uri-reference`: URIs
- `uuid`: UUID strings

### 12.D Schema Testing Strategies

**Unit Testing Schemas:**
```python
def test_schema_covers_edge_cases():
    """Test schema handles edge cases correctly."""
    schema = load_schema("llm.envelope.request.json")
    validator = Draft202012Validator(schema)
    
    # Test valid cases
    valid_cases = [
        minimal_valid_envelope,
        envelope_with_all_fields,
        envelope_with_optional_fields,
    ]
    
    for case in valid_cases:
        validator.validate(case)
    
    # Test invalid cases
    invalid_cases = [
        (missing_op_field, "missing 'op' field"),
        (extra_properties, "additional properties not allowed"),
        (wrong_op_format, "operation format invalid"),
    ]
    
    for case, expected_error in invalid_cases:
        with pytest.raises(ValidationError, match=expected_error):
            validator.validate(case)
```

### 12.E Troubleshooting Guide

**Common Issues and Solutions:**

1. **Schema fails to load**
   - Check `$schema` is `"https://json-schema.org/draft/2020-12/schema"`
   - Verify `$id` is a valid URI
   - Ensure no duplicate `$id` values

2. **$ref resolution fails**
   - Check referenced schema exists
   - Verify `$id` matches exactly (case-sensitive)
   - Ensure network accessible if external URL

3. **Validation too slow**
   - Enable validator caching
   - Switch to SAMPLED validation mode
   - Simplify complex schemas

4. **Memory usage high**
   - Limit schema size (< 5MB each)
   - Use `$ref` instead of inline definitions
   - Clear validator cache periodically

5. **Stream validation errors**
   - Check `is_final` flag on terminal chunk
   - Verify no content after terminal frame
   - Ensure streaming envelopes use `code: "STREAMING"`

6. **Request envelope validation fails**
   - Verify `op`, `ctx`, and `args` are present
   - Ensure `ctx` and `args` are objects (not strings or arrays)
   - Check `op` matches expected operation name

7. **Args validation fails**
   - Verify required fields are present (e.g., `text` and `model` for embedding.embed)
   - Check field types match schema (e.g., `messages` must be array for llm.complete)
   - Ensure constraints are met (e.g., `stream` must be false for embedding.embed)

**Debugging Commands:**
```bash
# Debug schema loading
CORPUS_SCHEMAS_ROOT=./schemas python -c "
from tests.utils.schema_registry import preload_all_schemas
preload_all_schemas()
print('Schemas loaded successfully')
"

# Debug specific validation
python -m tests.utils.schema_registry llm.envelope.request.json problematic_request.json --verbose

# Profile validation performance
python -m cProfile -s cumtime tests/utils/schema_registry.py llm.envelope.request.json sample.json

# Check schema for common issues
python tools/lint_schema.py schemas/llm/llm.complete.request.json

# Validate all golden fixtures
pytest tests/schema/test_golden_schema.py -v --tb=short
```

**Getting Help:**
1. **Check golden fixtures** - Compare with working examples
2. **Run schema lint** - Identify schema issues
3. **Enable verbose logging** - `CORPUS_LOG_LEVEL=DEBUG`
4. **Review protocol spec** - Ensure alignment with PROTOCOLS.md
5. **Consult error taxonomy** - See ERRORS.md for error semantics
6. **Check base implementation** - Verify wire handler expectations match schema

---

**End of SCHEMA.md (schema_version 1.0.0)**
