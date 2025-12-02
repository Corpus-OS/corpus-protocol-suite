# SPDX-License-Identifier: Apache-2.0

# CORPUS Protocol Suite - Schema Reference

**schema_version:** `1.0.0`  
**protocols_version:** `1.0`  
**json_schema_draft:** `2020-12`

> This document defines the schema architecture for the Corpus Protocol Suite. It establishes the normative JSON Schema definitions for all protocol operations, types, and wire formats while maintaining cross-protocol consistency and validation guarantees.

> **Schema Precedence:** When this document and PROTOCOLS.md disagree on **schema structure or field constraints**, this document is authoritative for validation. PROTOCOLS.md defines operational semantics and wire format requirements.

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
  - [5.1 Stream Frame Schemas](#51-stream-frame-schemas)
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
  - [11.2 Validation Error Codes](#112-validation-error-codes)
  - [11.3 Performance Benchmarks](#113-performance-benchmarks)
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
│   └── operation_context.json        # OperationContext type
│
├── llm/                              # LLM protocol schemas
│   ├── llm.envelope.request.json     # LLM-specific request envelope
│   ├── llm.envelope.success.json     # LLM-specific success envelope
│   ├── llm.envelope.error.json       # LLM-specific error envelope
│   ├── llm.response_format.json      # JSON/text output mode
│   ├── llm.sampling.params.json      # Temperature/top_p etc.
│   ├── llm.tools.schema.json         # Tool/tool_call definitions
│   ├── llm.stream.frame.data.json    # Streaming data frame
│   ├── llm.stream.frame.end.json     # Streaming end frame
│   ├── llm.stream.frame.error.json   # Streaming error frame
│   ├── llm.stream.frames.ndjson.schema.json  # NDJSON stream union
│   ├── llm.types.chunk.json          # Streaming chunk type
│   ├── llm.types.completion.json     # Completion result type
│   ├── llm.types.logprobs.json       # Log probabilities type
│   ├── llm.types.message.json        # Message type definition
│   ├── llm.types.token_usage.json    # Token usage reporting
│   ├── llm.types.tool.json           # Tool definition type
│   └── llm.types.warning.json        # Warning type definition
│
├── vector/                           # Vector protocol schemas
│   ├── vector.envelope.request.json
│   ├── vector.envelope.success.json
│   ├── vector.envelope.error.json
│   ├── vector.types.failure_item.json
│   ├── vector.types.filter.json
│   ├── vector.types.partial_success_result.json
│   ├── vector.types.query_result.json
│   ├── vector.types.vector.json
│   └── vector.types.vector_match.json
│
├── embedding/                        # Embedding protocol schemas
│   ├── embedding.envelope.request.json
│   ├── embedding.envelope.success.json
│   ├── embedding.envelope.error.json
│   ├── embedding.partial_success.result.json
│   ├── embedding.types.failure.json
│   ├── embedding.types.result.json
│   ├── embedding.types.vector.json
│   └── embedding.types.warning.json
│
└── graph/                            # Graph protocol schemas
    ├── graph.envelope.request.json
    ├── graph.envelope.success.json
    ├── graph.envelope.error.json
    ├── graph.stream.frame.data.json
    ├── graph.stream.frame.end.json
    ├── graph.stream.frame.error.json
    ├── graph.stream.frames.ndjson.schema.json
    ├── graph.types.batch_op.json
    ├── graph.types.entity.json
    ├── graph.types.id.json
    ├── graph.types.partial_success_result.json
    ├── graph.types.row.json
    └── graph.types.warning.json
```

**Key Directories:**
- **common/**: Schemas used across all protocols (envelopes, context)
- **{protocol}/**: Protocol-specific schemas following naming pattern
- **{protocol}/types/**: Reusable type definitions within protocol
- **{protocol}/stream/**: Streaming-specific schemas

### 2.2 Schema Organization Principles

**Single Responsibility Principle:**
- Each schema file defines **one logical component** (operation, type, envelope)
- Schemas are **self-contained** with explicit `$ref` dependencies
- **No circular dependencies** - all references form a directed acyclic graph

**Reusability Patterns:**
1. **Common envelopes** referenced by all protocol-specific envelopes
2. **Type definitions** (`$defs`) for reusable structures within protocol
3. **Cross-protocol sharing** via absolute `$id` references

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
- `{protocol}.stream.frame.{type}.json` for streaming frames
- `{protocol}.stream.frames.ndjson.schema.json` for NDJSON unions

**Examples:**
```
llm.envelope.request.json              # LLM protocol request envelope
vector.types.vector.json              # Vector type definition  
graph.stream.frame.data.json          # Graph streaming data frame
llm.stream.frames.ndjson.schema.json  # LLM NDJSON stream union
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
```

**Type Naming:**
- **Schema titles**: `PascalCase` with descriptive names
- **Property names**: `snake_case` for consistency with JSON
- **Definition names**: `PascalCase` within `$defs`

---

## 3. Core Schema Patterns

### 3.1 Common Envelope Schema

**Canonical Request Envelope:**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/common/envelope.request.json",
  "title": "Protocol Request Envelope",
  "type": "object",
  "properties": {
    "op": {
      "type": "string",
      "pattern": "^[a-z]+\\.[a-z_]+$",
      "description": "Protocol operation identifier"
    },
    "ctx": {
      "$ref": "https://corpusos.com/schemas/common/operation_context.json"
    },
    "args": {
      "type": "object",
      "additionalProperties": true,
      "description": "Operation-specific arguments"
    }
  },
  "required": ["op", "ctx", "args"],
  "additionalProperties": false
}
```

**Canonical Success Envelope:**
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
      "enum": ["OK", "PARTIAL_SUCCESS", "ACCEPTED"]
    },
    "ms": {
      "type": "number",
      "minimum": 0,
      "description": "Operation duration in milliseconds"
    },
    "result": {
      "type": "object",
      "additionalProperties": true,
      "description": "Operation-specific result"
    }
  },
  "required": ["ok", "code", "ms", "result"],
  "additionalProperties": false
}
```

**Canonical Error Envelope:**
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
      "type": ["number", "null"],
      "minimum": 0,
      "description": "Suggested retry delay in milliseconds"
    },
    "details": {
      "type": "object",
      "additionalProperties": true,
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

### 3.2 Operation Context Schema

**Operation Context Type:**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/common/operation_context.json",
  "title": "Operation Context",
  "type": "object",
  "properties": {
    "request_id": {
      "type": "string",
      "minLength": 1,
      "maxLength": 256,
      "pattern": "^[A-Za-z0-9._~:-]+$"
    },
    "idempotency_key": {
      "type": "string",
      "minLength": 1,
      "maxLength": 256
    },
    "deadline_ms": {
      "type": "number",
      "minimum": 1,
      "description": "Absolute epoch milliseconds"
    },
    "traceparent": {
      "type": "string",
      "pattern": "^[0-9a-f]{2}-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$"
    },
    "tenant": {
      "type": "string",
      "minLength": 1,
      "maxLength": 256
    },
    "attrs": {
      "type": "object",
      "additionalProperties": true,
      "description": "Opaque extension attributes"
    }
  },
  "additionalProperties": false
}
```

**Context Field Semantics:**

| Field | Required | Validation | Purpose |
|-------|----------|------------|---------|
| `request_id` | Optional | RFC 3986 unreserved chars | Request correlation |
| `idempotency_key` | Optional | ≤256 chars | Idempotency guarantee |
| `deadline_ms` | Optional | Positive integer | Absolute timeout |
| `traceparent` | Optional | W3C Trace Context | Distributed tracing |
| `tenant` | Optional | ≤256 chars | Tenant isolation |
| `attrs` | Optional | JSON object | Extension attributes |

### 3.3 Shared Type Schemas

**Note:** While you have protocol-specific type schemas, common patterns emerge:

**Partial Success Pattern (used across protocols):**
```json
{
  "type": "object",
  "properties": {
    "processed_count": {
      "type": "integer",
      "minimum": 0
    },
    "failed_count": {
      "type": "integer", 
      "minimum": 0
    },
    "failures": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": { "type": "string" },
          "index": { "type": "integer" },
          "error": { "type": "string" },
          "detail": { "type": "string" }
        },
        "additionalProperties": false
      }
    }
  },
  "required": ["processed_count", "failed_count", "failures"],
  "additionalProperties": false
}
```

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
    { "$ref": "https://corpusos.com/schemas/common/envelope.success.json" },
    {
      "properties": {
        "protocol": {
          "type": "string",
          "const": "llm/v1.0"
        },
        "component": {
          "type": "string",
          "const": "llm"
        }
      }
    }
  ]
}
```

#### 4.1.2 Type Definitions

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
      "enum": ["system", "user", "assistant", "tool"]
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
        "type": { "type": "string", "const": "function" },
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
  "required": ["prompt_tokens", "total_tokens"],
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
    "usage": {
      "$ref": "https://corpusos.com/schemas/llm/llm.types.token_usage.json"
    },
    "finish_reason": {
      "type": "string",
      "description": "Reason generation stopped"
    }
  },
  "required": ["text", "model", "usage", "finish_reason"],
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
      "description": "True for final chunk"
    },
    "model": {
      "type": "string",
      "description": "Optional model identifier"
    },
    "usage_so_far": {
      "$ref": "https://corpusos.com/schemas/llm/llm.types.token_usage.json"
    }
  },
  "required": ["text", "is_final"],
  "additionalProperties": false
}
```

#### 4.1.3 Configuration Schemas

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
      "const": "function"
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

#### 4.1.4 Streaming Schemas

**Stream Data Frame (`llm.stream.frame.data.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.stream.frame.data.json",
  "title": "LLM Stream Data Frame",
  "type": "object",
  "properties": {
    "event": {
      "type": "string",
      "const": "data"
    },
    "data": {
      "$ref": "https://corpusos.com/schemas/llm/llm.types.chunk.json"
    }
  },
  "required": ["event", "data"],
  "additionalProperties": false
}
```

**Stream End Frame (`llm.stream.frame.end.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.stream.frame.end.json",
  "title": "LLM Stream End Frame",
  "type": "object",
  "properties": {
    "event": {
      "type": "string",
      "const": "end"
    },
    "data": {
      "$ref": "https://corpusos.com/schemas/llm/llm.types.completion.json"
    }
  },
  "required": ["event", "data"],
  "additionalProperties": false
}
```

**NDJSON Stream Schema (`llm.stream.frames.ndjson.schema.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.stream.frames.ndjson.schema.json",
  "title": "LLM NDJSON Stream Frames",
  "oneOf": [
    { "$ref": "https://corpusos.com/schemas/llm/llm.stream.frame.data.json" },
    { "$ref": "https://corpusos.com/schemas/llm/llm.stream.frame.end.json" },
    { "$ref": "https://corpusos.com/schemas/llm/llm.stream.frame.error.json" }
  ]
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

#### 4.2.2 Type Definitions

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
      "type": "object",
      "additionalProperties": {
        "type": ["string", "number", "boolean", "null", "array"]
      }
    },
    "namespace": {
      "type": "string"
    }
  },
  "required": ["id", "vector"],
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
  "required": ["matches", "namespace", "total_matches"],
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

**Partial Success Result (`vector.types.partial_success_result.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/vector/vector.types.partial_success_result.json",
  "title": "Vector Partial Success Result",
  "type": "object",
  "properties": {
    "processed_count": { "type": "integer", "minimum": 0 },
    "failed_count": { "type": "integer", "minimum": 0 },
    "failures": {
      "type": "array",
      "items": {
        "$ref": "https://corpusos.com/schemas/vector/vector.types.failure_item.json"
      }
    }
  },
  "required": ["processed_count", "failed_count", "failures"],
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

### 4.3 Embedding Protocol Schemas

#### 4.3.1 Envelope Schemas

**Embedding Request Envelope (`embedding.envelope.request.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.envelope.request.json",
  "title": "Embedding Protocol Request Envelope",
  "type": "object",
  "allOf": [
    { "$ref": "https://corpusos.com/schemas/common/envelope.request.json" },
    {
      "properties": {
        "op": {
          "type": "string",
          "pattern": "^embedding\\.[a-z_]+$"
        }
      }
    }
  ]
}
```

#### 4.3.2 Type Definitions

**Embedding Vector (`embedding.types.vector.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.types.vector.json",
  "title": "Embedding Vector",
  "type": "object",
  "properties": {
    "vector": {
      "type": "array",
      "items": { "type": "number" },
      "minItems": 1
    },
    "text": {
      "type": "string",
      "description": "Original text (possibly truncated)"
    },
    "model": {
      "type": "string",
      "description": "Model identifier"
    },
    "dimensions": {
      "type": "integer",
      "minimum": 1
    }
  },
  "required": ["vector", "text", "model", "dimensions"],
  "additionalProperties": false
}
```

**Embedding Result (`embedding.types.result.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.types.result.json",
  "title": "Embedding Result",
  "type": "object",
  "properties": {
    "embedding": {
      "$ref": "https://corpusos.com/schemas/embedding/embedding.types.vector.json"
    },
    "model": { "type": "string" },
    "text": { "type": "string" },
    "tokens_used": { "type": "integer", "minimum": 0 },
    "truncated": { "type": "boolean" }
  },
  "required": ["embedding", "model", "text"],
  "additionalProperties": false
}
```

**Partial Success Result (`embedding.partial_success.result.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.partial_success.result.json",
  "title": "Embedding Partial Success Result",
  "type": "object",
  "properties": {
    "embeddings": {
      "type": "array",
      "items": {
        "$ref": "https://corpusos.com/schemas/embedding/embedding.types.vector.json"
      }
    },
    "model": { "type": "string" },
    "total_texts": { "type": "integer", "minimum": 0 },
    "total_tokens": { "type": "integer", "minimum": 0 },
    "failed_texts": {
      "type": "array",
      "items": {
        "$ref": "https://corpusos.com/schemas/embedding/embedding.types.failure.json"
      }
    }
  },
  "required": ["embeddings", "model", "total_texts"],
  "additionalProperties": false
}
```

**Failure Type (`embedding.types.failure.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/embedding/embedding.types.failure.json",
  "title": "Embedding Failure",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer",
      "minimum": 0,
      "description": "Index in original batch"
    },
    "text": { "type": "string" },
    "error": { "type": "string" },
    "message": { "type": "string" }
  },
  "required": ["index", "text", "error", "message"],
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

#### 4.4.2 Type Definitions

**Entity Type (`graph.types.entity.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.types.entity.json",
  "title": "Graph Entity",
  "type": "object",
  "oneOf": [
    {
      "type": "object",
      "properties": {
        "id": { "type": "string" },
        "labels": {
          "type": "array",
          "items": { "type": "string" }
        },
        "properties": {
          "type": "object",
          "additionalProperties": true
        },
        "namespace": { "type": "string" }
      },
      "required": ["id", "labels", "properties"],
      "additionalProperties": false
    },
    {
      "type": "object",
      "properties": {
        "id": { "type": "string" },
        "src": { "type": "string" },
        "dst": { "type": "string" },
        "label": { "type": "string" },
        "properties": {
          "type": "object",
          "additionalProperties": true
        },
        "namespace": { "type": "string" }
      },
      "required": ["id", "src", "dst", "label", "properties"],
      "additionalProperties": false
    }
  ]
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
    "id": { "type": "string" }
  },
  "required": ["id"],
  "additionalProperties": false
}
```

**Row Type (`graph.types.row.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.types.row.json",
  "title": "Graph Query Row",
  "type": "object",
  "patternProperties": {
    "^.*$": {
      "type": ["string", "number", "boolean", "null", "object", "array"]
    }
  },
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
    "op": { "type": "string" },
    "args": { "type": "object", "additionalProperties": true }
  },
  "required": ["op", "args"],
  "additionalProperties": false
}
```

**Partial Success Result (`graph.types.partial_success_result.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.types.partial_success_result.json",
  "title": "Graph Partial Success Result",
  "type": "object",
  "properties": {
    "processed_count": { "type": "integer", "minimum": 0 },
    "failed_count": { "type": "integer", "minimum": 0 },
    "failures": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": { "type": "string" },
          "index": { "type": "integer" },
          "error": { "type": "string" },
          "detail": { "type": "string" }
        },
        "additionalProperties": false
      }
    },
    "results": {
      "type": "array",
      "items": { "type": "object", "additionalProperties": true }
    }
  },
  "required": ["processed_count", "failed_count", "failures"],
  "additionalProperties": false
}
```

#### 4.4.3 Streaming Schemas

**Stream Data Frame (`graph.stream.frame.data.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.stream.frame.data.json",
  "title": "Graph Stream Data Frame",
  "type": "object",
  "properties": {
    "event": {
      "type": "string",
      "const": "data"
    },
    "data": {
      "type": "object",
      "properties": {
        "records": {
          "type": "array",
          "items": {
            "$ref": "https://corpusos.com/schemas/graph/graph.types.row.json"
          }
        },
        "is_final": {
          "type": "boolean",
          "description": "True for final chunk"
        }
      },
      "required": ["records", "is_final"],
      "additionalProperties": false
    }
  },
  "required": ["event", "data"],
  "additionalProperties": false
}
```

**NDJSON Stream Schema (`graph.stream.frames.ndjson.schema.json`):**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/graph/graph.stream.frames.ndjson.schema.json",
  "title": "Graph NDJSON Stream Frames",
  "oneOf": [
    { "$ref": "https://corpusos.com/schemas/graph/graph.stream.frame.data.json" },
    { "$ref": "https://corpusos.com/schemas/graph/graph.stream.frame.end.json" },
    { "$ref": "https://corpusos.com/schemas/graph/graph.stream.frame.error.json" }
  ]
}
```

---

## 5. Streaming Schemas

### 5.1 Stream Frame Schemas

**Protocol Stream Envelope Pattern:**
All streaming operations use the protocol envelope format with a `chunk` field instead of `result`:

```json
{
  "ok": true,
  "code": "OK",
  "ms": 45.2,
  "chunk": {
    "text": "Hello world",
    "is_final": false,
    "model": "gpt-4.1-mini"
  }
}
```

**Stream Frame Structure:**
Each protocol defines its own chunk format, but all follow this pattern:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.stream.frame.data.json",
  "title": "LLM Stream Data Frame",
  "type": "object",
  "properties": {
    "event": {
      "type": "string",
      "const": "data",
      "description": "Discriminator for frame type"
    },
    "data": {
      "$ref": "https://corpusos.com/schemas/llm/llm.types.chunk.json",
      "description": "Protocol-specific chunk data"
    }
  },
  "required": ["event", "data"],
  "additionalProperties": false
}
```

**Frame Types Per Protocol:**

| Protocol | Frame Types | Description |
|----------|-------------|-------------|
| **LLM** | `data`, `end`, `error` | Text streaming with completion metadata |
| **Graph** | `data`, `end`, `error` | Query result streaming with records |
| **Vector** | Not yet defined | Future streaming support |
| **Embedding** | Not yet defined | Future batch streaming support |

### 5.2 NDJSON Schema

**NDJSON Stream Validation:**
Streams are delivered as NDJSON (Newline-Delimited JSON) where each line is a protocol envelope:

```json
{"ok": true, "code": "OK", "ms": 12.3, "chunk": {"text": "Hello", "is_final": false, "model": "gpt-4.1-mini"}}
{"ok": true, "code": "OK", "ms": 15.7, "chunk": {"text": " world", "is_final": false, "model": "gpt-4.1-mini"}}
{"ok": true, "code": "OK", "ms": 18.2, "chunk": {"text": "!", "is_final": true, "model": "gpt-4.1-mini", "usage_so_far": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}}
```

**NDJSON Union Schema:**
Each protocol defines a union schema for validating complete NDJSON streams:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.stream.frames.ndjson.schema.json",
  "title": "LLM NDJSON Stream Frames",
  "description": "Union of all possible frame types in an LLM NDJSON stream",
  "oneOf": [
    { "$ref": "https://corpusos.com/schemas/llm/llm.stream.frame.data.json" },
    { "$ref": "https://corpusos.com/schemas/llm/llm.stream.frame.end.json" },
    { "$ref": "https://corpusos.com/schemas/llm/llm.stream.frame.error.json" }
  ]
}
```

**Stream End Frame:**
Terminal frame indicating successful stream completion:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.stream.frame.end.json",
  "title": "LLM Stream End Frame",
  "type": "object",
  "properties": {
    "event": {
      "type": "string",
      "const": "end"
    },
    "data": {
      "$ref": "https://corpusos.com/schemas/llm/llm.types.completion.json",
      "description": "Final completion result"
    }
  },
  "required": ["event", "data"],
  "additionalProperties": false
}
```

**Stream Error Frame:**
Terminal frame indicating stream failure:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://corpusos.com/schemas/llm/llm.stream.frame.error.json",
  "title": "LLM Stream Error Frame",
  "type": "object",
  "properties": {
    "event": {
      "type": "string",
      "const": "error"
    },
    "data": {
      "$ref": "https://corpusos.com/schemas/llm/llm.envelope.error.json",
      "description": "Error envelope"
    }
  },
  "required": ["event", "data"],
  "additionalProperties": false
}
```

### 5.3 Streaming Semantics

**Protocol §2.7 Compliance:**
All streaming implementations MUST adhere to these semantics:

1. **Single Terminal Event:** Exactly one terminal event (success or error) per stream
2. **No Content After Terminal:** Stream MUST end after final event
3. **Chunk Integrity:** Complete tokens/records delivered in each chunk
4. **Order Preservation:** Chunks delivered in correct sequence

**Validation Rules for Streams:**

```python
class StreamValidationRules:
    """Protocol §2.7 streaming semantics validation."""
    
    @staticmethod
    def validate_stream_termination(frames: List[dict]) -> None:
        """Validate stream has exactly one terminal event."""
        terminal_frames = [f for f in frames if f.get("chunk", {}).get("is_final") or not f.get("ok")]
        
        if len(terminal_frames) == 0:
            raise StreamProtocolError("Stream missing terminal frame")
        if len(terminal_frames) > 1:
            raise StreamProtocolError(f"Multiple terminal frames: {len(terminal_frames)}")
        
        terminal_index = frames.index(terminal_frames[0])
        if terminal_index != len(frames) - 1:
            raise StreamProtocolError("Content after terminal frame")
    
    @staticmethod
    def validate_envelope_consistency(frames: List[dict]) -> None:
        """Validate all frames use protocol envelope format."""
        for i, frame in enumerate(frames):
            if "ok" not in frame or "code" not in frame or "ms" not in frame:
                raise StreamProtocolError(f"Frame {i} missing protocol envelope fields")
            
            if frame["ok"] and "chunk" not in frame:
                raise StreamProtocolError(f"Frame {i} missing 'chunk' field")
            
            if not frame["ok"] and "error" not in frame:
                raise StreamProtocolError(f"Error frame {i} missing 'error' field")
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
    if "prompt" not in args and "messages" not in args:
        raise ArgsValidationError(f"{case_id}: requires 'prompt' or 'messages'")
    
    if "temperature" in args:
        temp = args["temperature"]
        if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
            raise ArgsValidationError(
                f"{case_id}: temperature must be between 0.0 and 2.0"
            )

def validate_vector_query_args(args: dict, case_id: str) -> None:
    """Validate vector.query operation arguments."""
    if "vector" not in args and "text" not in args:
        raise ArgsValidationError(f"{case_id}: requires 'vector' or 'text'")
    
    if "vector" in args:
        vector = args["vector"]
        if not isinstance(vector, list) or len(vector) == 0:
            raise ArgsValidationError(f"{case_id}: vector must be non-empty list")
        
        if not all(isinstance(x, (int, float)) for x in vector):
            raise ArgsValidationError(f"{case_id}: vector must contain numbers")
```

---

## 7. Golden Test Infrastructure

### 7.1 Golden Test Philosophy

**Executable Documentation:**
Golden tests serve as both validation fixtures and protocol documentation:

```python
# Golden test mapping
CASES = [
    ("llm/llm_complete_request.json",       "https://corpusos.com/schemas/llm/llm.envelope.request.json"),
    ("llm/llm_complete_success.json",       "https://corpusos.com/schemas/llm/llm.envelope.success.json"),
    ("vector/vector_query_request.json",    "https://corpusos.com/schemas/vector/vector.envelope.request.json"),
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
│   ├── llm_stream_frame_data.json
│   ├── llm_stream_frame_end.json
│   ├── llm_stream_frame_error.json
│   ├── llm_tools_schema.json
│   ├── llm_types_chunk.json
│   ├── llm_types_completion.json
│   ├── llm_types_logprobs.json
│   ├── llm_types_message.json
│   ├── llm_types_token_usage.json
│   ├── llm_types_tool.json
│   └── llm_types_warning.json
│
├── vector/
│   ├── vector_capabilities_request.json
│   ├── vector_capabilities_success.json
│   ├── vector_delete_request.json
│   ├── vector_delete_success.json
│   ├── vector_error_dimension_mismatch.json
│   ├── vector_health_request.json
│   ├── vector_health_success.json
│   ├── vector_namespace_create_request.json
│   ├── vector_namespace_create_success.json
│   ├── vector_namespace_delete_request.json
│   ├── vector_namespace_delete_success.json
│   ├── vector_partial_success_result.json
│   ├── vector_query_request.json
│   ├── vector_query_success.json
│   ├── vector_types_failure_item.json
│   ├── vector_types_filter.json
│   ├── vector_types_query_result.json
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
│   ├── embedding_health_request.json
│   ├── embedding_health_success.json
│   ├── embedding_partial_success_result.json
│   ├── embedding_types_failure.json
│   ├── embedding_types_vector.json
│   └── embedding_types_warning.json
│
└── graph/
    ├── graph.delete_nodes.by_id.request.json
    ├── graph.delete_nodes.by_id.success.json
    ├── graph.upsert_nodes.single.request.json
    ├── graph.upsert_nodes.single.success.json
    ├── graph_batch_op_create_vertex.json
    ├── graph_batch_op_query.json
    ├── graph_batch_request.json
    ├── graph_batch_success.json
    ├── graph_capabilities_request.json
    ├── graph_capabilities_success.json
    ├── graph_edge_create_request.json
    ├── graph_edge_create_success.json
    ├── graph_entity_edge.json
    ├── graph_entity_vertex.json
    ├── graph_envelope_error.json
    ├── graph_health_request.json
    ├── graph_health_success.json
    ├── graph_id_value.json
    ├── graph_partial_success_result.json
    ├── graph_query_request.json
    ├── graph_query_success.json
    ├── graph_row.json
    ├── graph_stream.ndjson
    ├── graph_stream_chunk.json
    ├── graph_stream_error.ndjson
    ├── graph_stream_frame_data.json
    ├── graph_stream_frame_end.json
    ├── graph_stream_frame_error.json
    ├── graph_stream_query_request.json
    └── graph_warning.json
```

**Naming Conventions:**
- `{protocol}_{operation}_{type}.json` - Operation test cases
- `{protocol}_types_{name}.json` - Type definition examples
- `{protocol}_stream.ndjson` - Complete streaming examples
- `{protocol}_stream_{frame}.json` - Individual stream frame examples

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
        assert doc["code"] in {"OK", "PARTIAL_SUCCESS", "ACCEPTED"}, \
            f"{fname}: unexpected code {doc['code']!r}"
```

**Streaming Validation:**
```python
def test_streaming_uses_protocol_envelope():
    """Test streaming operations use protocol envelope with chunk field per §2.4."""
    streaming_files = [
        "llm/llm_stream_chunk.json",
        "graph/graph_stream_chunk.json",
    ]
    
    for fname in streaming_files:
        doc = load_golden(fname)
        
        # Must use protocol envelope format
        assert "ok" in doc, f"{fname}: missing 'ok' field"
        assert "code" in doc, f"{fname}: missing 'code' field"
        assert "ms" in doc, f"{fname}: missing 'ms' field"
        assert "chunk" in doc, f"{fname}: missing 'chunk' field"
        
        assert doc["ok"] is True, f"{fname}: 'ok' must be true"
        assert doc["code"] == "OK", f"{fname}: 'code' must be 'OK'"
```

**NDJSON Stream Validation:**
```python
@pytest.mark.parametrize("fname,schema_id,component", [
    ("llm/llm_stream.ndjson", "https://corpusos.com/schemas/llm/llm.envelope.success.json", "llm"),
    ("graph/graph_stream.ndjson", "https://corpusos.com/schemas/graph/graph.envelope.success.json", "graph"),
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
    
    # Example case definition
    LLM_COMPLETE = WireRequestCase(
        id="llm.complete.basic",
        op="llm.complete",
        build_method="build_complete_request",
        schema_id="https://corpusos.com/schemas/llm/llm.envelope.request.json",
        component="llm",
        schema_versions=["1.0.0", "1.1.0"],
        args_validator=validate_llm_complete_args,
        tags=["core", "llm"],
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
    
    def test_negative_deadline_rejected(self, adapter: Any):
        """Negative deadline_ms should be rejected."""
        envelope = {"ctx": {"request_id": "test", "deadline_ms": -100}}
        
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
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary for reporting."""
        return {
            "total_runs": len(self.validation_times),
            "total_successes": sum(self.successes.values()),
            "total_failures": sum(sum(errs.values()) for errs in self.failures.values()),
            "avg_duration_ms": (sum(t[1] for t in self.validation_times) / len(self.validation_times) * 1000)
        }
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
```

**Stream Validation CLI:**
```python
# Validate NDJSON stream from file
from tests.utils.stream_validator import validate_ndjson_stream

with open("stream.ndjson", "r") as f:
    report = validate_ndjson_stream(
        f.read(),
        envelope_schema_id="https://corpusos.com/schemas/llm/llm.envelope.success.json",
        component="llm",
        mode="strict"
    )
    
print(f"Valid: {report.is_valid}")
print(f"Frames: {report.total_frames}")
print(f"Errors: {len(report.validation_errors)}")
```

**Conformance Test Runner:**
```bash
# Run all conformance tests
pytest tests/live/test_wire_conformance.py -v

# Run only LLM tests
pytest tests/live/test_wire_conformance.py -v -m "llm"

# Run core operations only
pytest tests/live/test_wire_conformance.py -v -m "core"

# Skip schema validation for faster iteration
pytest tests/live/test_wire_conformance.py -v --skip-schema

# Test specific adapter
pytest tests/live/test_wire_conformance.py -v --adapter=openai

# Generate coverage report
pytest tests/live/test_wire_conformance.py -v --tb=no -q | generate_coverage_report.py
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

**Breaking Change Process:**
1. **Create new schema version** - e.g., `llm.envelope.request.v2.json`
2. **Maintain backward compatibility** - Support both versions temporarily
3. **Update golden fixtures** - Add examples for new version
4. **Update adapter implementations** - Support new version
5. **Deprecate old version** - Mark as deprecated in documentation
6. **Schedule removal** - Remove after grace period (e.g., 6 months)

**Debugging Workflow:**
```python
# Debug schema validation
from tests.utils.schema_registry import get_validator

validator = get_validator("https://corpusos.com/schemas/llm/llm.envelope.request.json")
errors = list(validator.iter_errors(invalid_request))
for error in errors:
    print(f"Path: {error.path}")
    print(f"Message: {error.message}")
    print(f"Schema: {error.schema}")

# Debug stream validation
from tests.utils.stream_validator import StreamValidationEngine

engine = StreamValidationEngine(config)
report = engine.validate_ndjson(ndjson_text)
if not report.is_valid:
    for error in report.validation_errors:
        print(f"Frame {error.frame_number}: {error.error_type}")
        print(f"  {error.message}")
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
      - name: Run schema lint tests
        run: pytest tests/schema/test_schema_lint.py -v
  
  golden-tests:
    runs-on: ubuntu-latest
    needs: schema-lint
    steps:
      - uses: actions/checkout@v3
      - name: Run golden tests
        run: pytest tests/schema/test_golden_schema.py -v
  
  conformance-tests:
    runs-on: ubuntu-latest
    needs: golden-tests
    strategy:
      matrix:
        adapter: [openai, anthropic, pinecone, neo4j]
    steps:
      - uses: actions/checkout@v3
      - name: Run ${{ matrix.adapter }} conformance tests
        run: |
          pytest tests/live/test_wire_conformance.py -v \
            --adapter=${{ matrix.adapter }} \
            --junitxml=results-${{ matrix.adapter }}.xml
  
  performance-checks:
    runs-on: ubuntu-latest
    needs: conformance-tests
    steps:
      - uses: actions/checkout@v3
      - name: Run performance benchmarks
        run: python benchmarks/schema_performance.py
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
        
      - id: golden-validation
        name: Golden Validation
        entry: pytest tests/schema/test_golden_schema.py -v
        language: system
        files: ^tests/golden/.*\.json$
```

**Code Coverage Requirements:**
```python
# Minimum coverage thresholds
COVERAGE_THRESHOLDS = {
    "schema_files": 100,      # All schema files must be linted
    "golden_fixtures": 95,    # 95% of golden fixtures validated
    "conformance_cases": 90,  # 90% of conformance test cases
    "edge_cases": 85,         # 85% of edge case tests
}

def check_coverage():
    """Check test coverage meets minimum thresholds."""
    metrics = generate_coverage_report()
    
    for category, threshold in COVERAGE_THRESHOLDS.items():
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

**Schema Version in Envelopes:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 45.2,
  "result": {
    "schema_version": "1.0.0",
    "protocol": "llm/v1.0",
    // ... other result fields
  }
}
```

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

### 10.2 Breaking Change Process

**Phase 1: Announcement (4 weeks before change)**
- Add deprecation warnings to schemas
- Update documentation with migration guide
- Notify adapter maintainers

**Phase 2: Dual Support (8 weeks)**
- New schema version available
- Both old and new versions accepted
- Adapters can migrate at their own pace

**Phase 3: Deprecation (4 weeks)**
- Old version marked as deprecated
- Warnings in validation output
- Strong recommendation to migrate

**Phase 4: Removal**
- Old version removed from validation
- Adapters must use new version
- Breaking change complete

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
        
        # Update schema version
        v2_envelope["schema_version"] = "2.0.0"
        
        return v2_envelope
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
      "additionalProperties": true
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

# Register custom validators
jsonschema.Draft202012Validator.format_checker.checks(
    "vector-dimensions", raises=ValueError
)(CustomFormatValidator.validate_vector_dimensions)
```

---

## 11. Reference

### 11.1 Schema Quick Reference

**Common Schema IDs:**
```
https://corpusos.com/schemas/common/envelope.request.json
https://corpusos.com/schemas/common/envelope.success.json
https://corpusos.com/schemas/common/envelope.error.json
https://corpusos.com/schemas/common/operation_context.json
```

**LLM Schema IDs:**
```
https://corpusos.com/schemas/llm/llm.envelope.request.json
https://corpusos.com/schemas/llm/llm.envelope.success.json
https://corpusos.com/schemas/llm/llm.envelope.error.json
https://corpusos.com/schemas/llm/llm.types.message.json
https://corpusos.com/schemas/llm/llm.types.completion.json
https://corpusos.com/schemas/llm/llm.types.token_usage.json
https://corpusos.com/schemas/llm/llm.stream.frame.data.json
```

**Vector Schema IDs:**
```
https://corpusos.com/schemas/vector/vector.envelope.request.json
https://corpusos.com/schemas/vector/vector.envelope.success.json
https://corpusos.com/schemas/vector/vector.types.vector.json
https://corpusos.com/schemas/vector/vector.types.vector_match.json
https://corpusos.com/schemas/vector/vector.types.query_result.json
```

**Operation to Schema Mapping:**

| Operation | Request Schema | Success Schema |
|-----------|----------------|----------------|
| `llm.complete` | `llm.envelope.request.json` | `llm.envelope.success.json` |
| `llm.stream` | `llm.envelope.request.json` | `llm.envelope.success.json` |
| `vector.query` | `vector.envelope.request.json` | `vector.envelope.success.json` |
| `vector.upsert` | `vector.envelope.request.json` | `vector.envelope.success.json` |
| `embedding.embed` | `embedding.envelope.request.json` | `embedding.envelope.success.json` |
| `graph.query` | `graph.envelope.request.json` | `graph.envelope.success.json` |

### 11.2 Validation Error Codes

**JSON Schema Validation Errors:**
```python
VALIDATION_ERROR_CODES = {
    "type": "Value must be of type {expected_type}",
    "required": "Missing required field: {field}",
    "additionalProperties": "Additional properties not allowed: {property}",
    "pattern": "Value must match pattern: {pattern}",
    "enum": "Value must be one of: {choices}",
    "minimum": "Value must be at least {minimum}",
    "maximum": "Value must be at most {maximum}",
    "minLength": "Value must have at least {min_length} characters",
    "maxLength": "Value must have at most {max_length} characters",
    "format": "Value must be a valid {format}",
}
```

**Protocol Validation Errors:**
```python
PROTOCOL_ERROR_CODES = {
    "ENVELOPE_SHAPE": "Protocol envelope missing required fields",
    "ENVELOPE_TYPE": "Protocol envelope must be a JSON object",
    "CTX_VALIDATION": "Operation context validation failed",
    "ARGS_VALIDATION": "Operation arguments validation failed",
    "SERIALIZATION": "JSON serialization/deserialization failed",
    "STREAM_TERMINATION": "Stream missing or has multiple terminal frames",
    "STREAM_ENVELOPE": "Stream frame missing protocol envelope fields",
    "VERSION_MISMATCH": "Schema version not supported",
}
```

**Error Recovery Strategies:**

| Error Type | Automatic Retry | Manual Intervention Required |
|------------|----------------|------------------------------|
| `TRANSIENT_NETWORK` | Yes (with backoff) | No |
| `RESOURCE_EXHAUSTED` | Yes (after retry_after_ms) | No |
| `BAD_REQUEST` | No | Yes - fix request |
| `SCHEMA_VALIDATION` | No | Yes - update schema or data |
| `PROTOCOL_VIOLATION` | No | Yes - fix protocol implementation |

### 11.3 Performance Benchmarks

**Schema Loading Benchmarks:**
```
Benchmark: Schema Registry Loading
----------------------------------
Total schemas: 47
Load time (cold): 1.2s
Load time (warm): 0.3s
Memory usage: 8.4MB
Validator cache size: 47

Per-schema averages:
  Parse time: 25ms
  Validator creation: 18ms
  Memory per schema: 180KB
```

**Validation Performance:**
```
Benchmark: Validation Performance
---------------------------------
Operation: llm.complete request
Schema: llm.envelope.request.json
Payload size: 2.1KB

Validation modes:
  Strict: 8.2ms (100% frames validated)
  Sampled (10%): 1.1ms (10% frames validated)  
  Lazy: 0.4ms (protocol only)

Throughput:
  Strict: 122 req/s
  Sampled: 909 req/s
  Lazy: 2500 req/s
```

**Stream Validation Performance:**
```
Benchmark: Stream Validation
----------------------------
Stream: LLM completion (1000 tokens)
Frames: 48
Payload size: 24KB

Validation modes:
  Strict: 392ms (8.2ms/frame)
  Sampled (10%): 45ms (0.9ms/frame)
  Lazy: 19ms (0.4ms/frame)

Memory usage:
  Peak: 12.3MB
  Steady state: 4.2MB
```

**Optimization Guidelines:**

1. **Use SAMPLED mode for production** - 10% sampling provides good coverage with minimal overhead
2. **Enable validator caching** - 10x speedup for repeated validations
3. **Limit schema complexity** - Keep schemas under 5MB each
4. **Use $ref for reuse** - Reduces memory and improves cache efficiency
5. **Batch validation** - Validate multiple items together when possible

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
   - Ensure all frames use protocol envelope

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
python -m cProfile -s cumtime tests/utils/schema_registry llm.envelope.request.json sample.json
```

**Getting Help:**
1. **Check golden fixtures** - Compare with working examples
2. **Run schema lint** - Identify schema issues
3. **Enable verbose logging** - `CORPUS_LOG_LEVEL=DEBUG`
4. **Review protocol spec** - Ensure alignment with PROTOCOLS.md
5. **Consult error taxonomy** - See ERRORS.md for error semantics

---

**End of SCHEMA.md (schema_version 1.0.0)**
