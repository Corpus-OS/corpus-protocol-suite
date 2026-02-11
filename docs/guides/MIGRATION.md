# CORPUS Protocol Suite - Migration Reference Guide

**Version:** 1.0  
**Protocol Compatibility:** v1.0  
**Last Updated:** January 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [How to Use This Guide](#2-how-to-use-this-guide)
3. [Core Migration Patterns](#3-core-migration-patterns)
    - 3.1 [The CORPUS Envelope Pattern](#31-the-corpus-envelope-pattern)
    - 3.2 [Response Transformation](#32-response-transformation)
    - 3.3 [Streaming Pattern](#33-streaming-pattern)
    - 3.4 [Filter Expression Standard](#34-filter-expression-standard)
    - 3.5 [Vendor Extension Handling](#35-vendor-extension-handling)
4. [LLM Protocol Migrations](#4-llm-protocol-migrations)
    - 4.1 [LLM Operations Reference](#41-llm-operations-reference)
    - 4.2 [OpenAI to CORPUS](#42-openai-to-corpus)
    - 4.3 [Anthropic to CORPUS](#43-anthropic-to-corpus)
    - 4.4 [Cohere to CORPUS](#44-cohere-to-corpus)
    - 4.5 [Google AI to CORPUS](#45-google-ai-to-corpus)
    - 4.6 [Azure OpenAI to CORPUS](#46-azure-openai-to-corpus)
5. [Vector Protocol Migrations](#5-vector-protocol-migrations)
    - 5.1 [Vector Operations Reference](#51-vector-operations-reference)
    - 5.2 [Pinecone to CORPUS](#52-pinecone-to-corpus)
    - 5.3 [Qdrant to CORPUS](#53-qdrant-to-corpus)
    - 5.4 [Weaviate to CORPUS](#54-weaviate-to-corpus)
    - 5.5 [Milvus to CORPUS](#55-milvus-to-corpus)
    - 5.6 [Chroma to CORPUS](#56-chroma-to-corpus)
6. [Embedding Protocol Migrations](#6-embedding-protocol-migrations)
    - 6.1 [Embedding Operations Reference](#61-embedding-operations-reference)
    - 6.2 [OpenAI Embeddings to CORPUS](#62-openai-embeddings-to-corpus)
    - 6.3 [Cohere Embed to CORPUS](#63-cohere-embed-to-corpus)
    - 6.4 [HuggingFace to CORPUS](#64-huggingface-to-corpus)
    - 6.5 [Google Vertex AI to CORPUS](#65-google-vertex-ai-to-corpus)
    - 6.6 [AWS Bedrock to CORPUS](#66-aws-bedrock-to-corpus)
7. [Graph Protocol Migrations](#7-graph-protocol-migrations)
    - 7.1 [Graph Operations Reference](#71-graph-operations-reference)
    - 7.2 [Neo4j to CORPUS](#72-neo4j-to-corpus)
    - 7.3 [Amazon Neptune to CORPUS](#73-amazon-neptune-to-corpus)
    - 7.4 [JanusGraph to CORPUS](#74-janusgraph-to-corpus)
    - 7.5 [TigerGraph to CORPUS](#75-tigergraph-to-corpus)
    - 7.6 [ArangoDB to CORPUS](#76-arangodb-to-corpus)
8. [Error Code Mapping](#8-error-code-mapping)
9. [Context Propagation](#9-context-propagation)
10. [Migration Validation Checklist](#10-migration-validation-checklist)
11. [References](#11-references)
12. [Migration Guide Implementation Notes](#12-migration-guide-implementation-notes)

---

## 1. Executive Summary

### What This Guide Provides
This document provides **mapping guidance** for migrating existing AI service APIs to the CORPUS Protocol Suite. It offers **wire-level mapping tables** showing how to transform requests and responses between vendor-specific formats and the standardized CORPUS protocol defined in SCHEMA.md and PROTOCOLS.md.

### Key Migration Benefits
- **Unified Interface**: Use one consistent protocol across multiple providers
- **Vendor Flexibility**: Switch between providers with minimal code changes
- **Production Features**: Leverage built-in observability, error handling, and security patterns
- **Protocol Evolution**: Benefit from protocol improvements independent of provider changes

### Scope Boundaries
| What's Included | What's Excluded |
|----------------|-----------------|
| Wire format translation examples | SDK implementation details |
| Parameter mapping guidance | Operational deployment guides |
| Error code normalization patterns | Business logic guidance |
| Context propagation patterns | Performance optimization |
| Schema-compatible examples | Provider-specific features not in CORPUS |

### Document Authority
- **SCHEMA.md** is authoritative for field names, types, requiredness, and envelope closure
- **PROTOCOLS.md** is authoritative for operation semantics and behavior
- **This guide** provides non-normative mapping examples and migration patterns

---

## 2. How to Use This Guide

### For Adapter Developers
1. **Find your provider** in the relevant protocol section (LLM, Vector, etc.)
2. **Follow the mapping tables** to transform requests/responses according to CORPUS schema
3. **Validate** using the checklist in Section 10
4. **Test** with the provided example transformations

### Quick Reference Patterns
Each migration follows this pattern:

```python
# Pattern: Provider → CORPUS transformation
def transform_provider_to_corpus(provider_request):
    return {
        "op": "protocol.operation",
        "ctx": extract_context(provider_request),
        "args": transform_arguments(provider_request)
    }
```

### Document Conventions
- **Bold terms**: CORPUS-specific concepts
- `Inline code`: Field names and values
- **Tables**: Wire-level mappings
- **Notes**: Important migration considerations

---

## 3. Core Migration Patterns

### 3.1 The CORPUS Envelope Pattern
**Every CORPUS request follows this structure (per SCHEMA.md):**

```json
{
  "op": "protocol.operation",    // What operation to perform
  "ctx": {                       // Request context metadata
    "request_id": "req-123",
    "deadline_ms": 1730312345000,
    "tenant": "acme-corp",
    "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
    "attrs": {}                  // Optional vendor extensions
  },
  "args": {                      // Operation-specific parameters
    // ... varies by operation
  }
}
```

### 3.2 Response Transformation
**Provider responses transform to CORPUS envelopes (closed per SCHEMA.md):**

```json
// CORPUS Success Response
{
  "ok": true,
  "code": "OK",
  "ms": 125.4,
  "result": {
    // Operation-specific result data
  }
}

// CORPUS Error Response  
{
  "ok": false,
  "code": "RESOURCE_EXHAUSTED",
  "error": "ResourceExhausted",
  "message": "Rate limit exceeded",
  "retry_after_ms": 5000,
  "details": {
    "provider_error_id": "rate_limit_exceeded_123"
  },
  "ms": 12.3
}
```

### 3.3 Streaming Pattern
**Streaming operations use chunked envelopes with strict closure rules:**

```json
// Each streaming success frame (closed envelope)
{
  "ok": true,
  "code": "STREAMING",
  "ms": 15.7,
  "chunk": {
    "text": "Hello",
    "is_final": false
  }
}

// Final chunk
{
  "ok": true,
  "code": "STREAMING",
  "ms": 156.2,
  "chunk": {
    "text": " world.",
    "is_final": true
  }
}

// Stream terminated by error
{
  "ok": false,
  "code": "RESOURCE_EXHAUSTED",
  "error": "ResourceExhausted",
  "message": "Rate limit exceeded",
  "retry_after_ms": 5000,
  "ms": 89.3
}
```

**Streaming Rules (per SCHEMA.md and PROTOCOLS.md):**
- All success frames use `"code": "STREAMING"` (never `"OK"`)
- Streams end with either a final chunk (`"is_final": true`) or an error envelope
- No frames may appear after terminal condition
- All streaming envelopes are closed to `{ok, code, ms, chunk}` only

### 3.4 Filter Expression Standard
**CORPUS uses a unified operator-object form for filter expressions (per PROTOCOLS.md):**

```json
// Equality filter
{
  "category": "books"
}

// Range filter (using operator objects)
{
  "price": {"gte": 20, "lt": 100}
}

// Membership filter
{
  "tags": {"in": ["fiction", "scifi"]}
}

// Combined filters
{
  "category": "books",
  "price": {"gte": 20},
  "tags": {"in": ["fiction", "scifi"]}
}
```

**Important:** Do not use field__gte or field__in conventions. Always use the operator-object form shown above.

### 3.5 Vendor Extension Handling
**Request-side extensions follow these patterns:**

```json
// Using ctx.attrs for vendor-specific metadata
{
  "op": "llm.complete",
  "ctx": {
    "request_id": "req-123",
    "attrs": {
      "x-openai-organization": "org-123",
      "x-azure-api-version": "2023-12-01-preview"
    }
  },
  "args": {
    "model": "gpt-4",
    "messages": [...]
  }
}
```

**Key Rules (per SCHEMA.md):**
- **Never** add top-level fields to success/error/streaming envelopes (they are closed)
- Use `ctx.attrs` for vendor-specific request metadata
- Only place vendor-specific keys in `args` when the operation's schema explicitly allows `additionalProperties: true`
- Namespace extension keys with `x-<vendor>-` prefix (e.g., `x-openai-`, `x-azure-`)
- Response envelopes must not contain vendor extensions in the success/error structure

---

## 4. LLM Protocol Migrations

### 4.1 LLM Operations Reference
CORPUS v1.0 defines these LLM operations (per PROTOCOLS.md):

| Operation | Description | Provider Example |
|-----------|-------------|------------------|
| `llm.capabilities` | Get provider/model capabilities | OpenAI `/models` endpoint |
| `llm.complete` | Standard completion (non-streaming) | OpenAI `/chat/completions` |
| `llm.stream` | Streaming completion | OpenAI `/chat/completions` with `stream: true` |
| `llm.count_tokens` | Count tokens in text | Anthropic token counting |
| `llm.health` | Check LLM service health | Provider health endpoint |

**Provider-Agnostic Mapping Template:**
```json
// General pattern for provider → CORPUS transformation
{
  "op": "llm.operation",
  "ctx": {
    "request_id": "generated-or-from-header",
    "deadline_ms": "calculated-deadline",
    "attrs": {
      // Vendor-specific headers go here with x- prefix
    }
  },
  "args": {
    // Operation-specific arguments per SCHEMA.md
  }
}
```

### 4.2 OpenAI to CORPUS

#### Request Mapping (Chat Completions API)
**Primary Source:** [OpenAI Chat Completions API Reference](https://platform.openai.com/docs/api-reference/chat)

| OpenAI Field | CORPUS Field | Type | Transformation |
|--------------|--------------|------|----------------|
| `model` | `args.model` | string | Direct mapping |
| `messages` | `args.messages` | array | Same format, roles unchanged |
| `max_tokens` | `args.max_tokens` | integer | Direct mapping |
| `temperature` | `args.temperature` | float | Same range [0.0, 2.0] |
| `top_p` | `args.top_p` | float | Same range (0.0, 1.0] |
| `frequency_penalty` | `args.frequency_penalty` | float | Same range [-2.0, 2.0] |
| `presence_penalty` | `args.presence_penalty` | float | Same range [-2.0, 2.0] |
| `stream` | `op: "llm.stream"` | boolean | Different operation |
| `tools` | `args.tools` | array | Same format |
| `tool_choice` | `args.tool_choice` | string/object | Same format |
| `response_format` | `args.response_format` | object | Same format |
| `seed` | `ctx.attrs.x-openai-seed` | integer | Namespaced in attrs |
| `user` | `ctx.attrs.x-openai-user` | string | Namespaced in attrs |
| `stop` | `args.stop` | string/array | Direct mapping |
| `n` | `ctx.attrs.x-openai-n` | integer | Namespaced in attrs |
| `logit_bias` | `ctx.attrs.x-openai-logit_bias` | object | Namespaced in attrs |

**Wire Envelope Example:**
```json
// OpenAI Request
POST /v1/chat/completions
{
  "model": "gpt-4",
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello"}
  ],
  "temperature": 0.7,
  "max_tokens": 100,
  "seed": 42,
  "user": "user-123"
}

// CORPUS Equivalent
POST /v1/operations
{
  "op": "llm.complete",
  "ctx": {
    "request_id": "req-123",
    "deadline_ms": 1730312345000,
    "attrs": {
      "x-openai-user": "user-123",
      "x-openai-seed": 42
    }
  },
  "args": {
    "model": "gpt-4",
    "messages": [
      {"role": "system", "content": "You are helpful"},
      {"role": "user", "content": "Hello"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }
}
```

**Streaming Example:**
```json
// OpenAI streaming request
POST /v1/chat/completions
{
  "model": "gpt-4",
  "messages": [...],
  "stream": true
}

// CORPUS streaming equivalent
{
  "op": "llm.stream",
  "ctx": {
    "request_id": "req-456",
    "deadline_ms": 1730312345000
  },
  "args": {
    "model": "gpt-4",
    "messages": [...]
  }
}

// CORPUS streaming chunk
{"ok": true, "code": "STREAMING", "ms": 15.7, "chunk": {"text": "Hello", "is_final": false}}
```

#### Response Mapping
| OpenAI Response Field | CORPUS Response Field | Transformation |
|----------------------|----------------------|----------------|
| `choices[0].message.content` | `result.text` | Direct mapping |
| `model` | `result.model` | Direct mapping |
| `usage.prompt_tokens` | `result.usage.prompt_tokens` | Direct mapping |
| `usage.completion_tokens` | `result.usage.completion_tokens` | Direct mapping |
| `usage.total_tokens` | `result.usage.total_tokens` | Direct mapping |
| `finish_reason` | `result.finish_reason` | Direct mapping |
| `id` | `ctx.request_id` (if provided) | Context mapping |
| `created` | Not included | Timestamp in `ms` field |
| `object` | Not included | Type indicator |

#### Other LLM Operations
```json
// llm.capabilities example
{
  "op": "llm.capabilities",
  "ctx": {
    "request_id": "cap-req-123"
  },
  "args": {}
}

// llm.count_tokens example (per SCHEMA.md: text required)
{
  "op": "llm.count_tokens",
  "ctx": {
    "request_id": "count-req-456"
  },
  "args": {
    "model": "gpt-4",
    "text": "Hello world"
  }
}

// llm.health example
{
  "op": "llm.health",
  "ctx": {
    "request_id": "health-req-789"
  },
  "args": {}
}
```

#### Error Mapping
| OpenAI Error | CORPUS Error Code | Retryable | Notes |
|--------------|------------------|-----------|-------|
| `RateLimitError` | `RESOURCE_EXHAUSTED` | Yes | Add `retry_after_ms` |
| `AuthenticationError` | `AUTH_ERROR` | No | Direct mapping |
| `BadRequestError` | `BAD_REQUEST` | No | Direct mapping |
| `APIConnectionError` | `TRANSIENT_NETWORK` | Yes | Network issues |
| `APIError` (5xx) | `UNAVAILABLE` | Yes | Provider issue |
| `TimeoutError` | `DEADLINE_EXCEEDED` | Conditional | Check deadline |
| `ContentFilterError` | `BAD_REQUEST` | No | Add `details.filtered_reason` |

### 4.3 Anthropic to CORPUS

#### Request Mapping (Messages API)
**Primary Source:** Anthropic Messages API Reference

| Anthropic Field | CORPUS Field | Type | Transformation |
|----------------|--------------|------|----------------|
| `model` | `args.model` | string | Map: `claude-3-opus-20240229` → `claude-3-opus` |
| `messages` | `args.messages` | array | Convert Anthropic format: `{role, content}` → same |
| `max_tokens` | `args.max_tokens` | integer | Direct mapping |
| `temperature` | `args.temperature` | float | Same range [0.0, 1.0] → [0.0, 2.0] |
| `top_p` | `args.top_p` | float | Same range (0.0, 1.0] |
| `top_k` | `ctx.attrs.x-anthropic-top_k` | integer | Namespaced in attrs |
| `stream` | `op: "llm.stream"` | boolean | Different operation |
| `system` | `args.system_message` | string | Move from messages array |
| `tools` | `args.tools` | array | Convert Anthropic tool format |
| `tool_choice` | `args.tool_choice` | object | Convert format |
| `stop_sequences` | `args.stop` | array | Direct mapping |
| `metadata` | `ctx.attrs.x-anthropic-metadata` | object | Namespaced in attrs |

**Tool Call Conversion:**
```json
// Anthropic tool call
{
  "type": "tool_use",
  "id": "toolu_123",
  "name": "get_weather",
  "input": {"city": "San Francisco"}
}

// CORPUS tool call
{
  "id": "toolu_123",
  "type": "function",
  "function": {
    "name": "get_weather",
    "arguments": "{\"city\": \"San Francisco\"}"
  }
}
```

#### Response Mapping
| Anthropic Response Field | CORPUS Response Field | Transformation |
|-------------------------|----------------------|----------------|
| `content[0].text` | `result.text` | Join content blocks if multiple |
| `model` | `result.model` | Map to normalized model name |
| `usage.input_tokens` | `result.usage.prompt_tokens` | Rename field |
| `usage.output_tokens` | `result.usage.completion_tokens` | Rename field |
| `stop_reason` | `result.finish_reason` | Map values: `end_turn` → `stop`, `max_tokens` → `length` |
| `id` | `ctx.request_id` (if provided) | Context mapping |
| `type` | Not included | Type indicator |

### 4.4 Cohere to CORPUS

#### Request Mapping (Chat API)
**Needs verification against current Cohere Chat API reference**

| Cohere Field | CORPUS Field | Type | Transformation |
|--------------|--------------|------|----------------|
| `model` | `args.model` | string | Map: `command-r-plus` → `cohere-command-r-plus` |
| `message` | `args.messages` | string | Convert to messages array |
| `chat_history` | `args.messages` | array | Merge with current message |
| `max_tokens` | `args.max_tokens` | integer | Direct mapping |
| `temperature` | `args.temperature` | float | Same range [0.0, 1.0] → [0.0, 2.0] |
| `p` | `args.top_p` | float | Rename field |
| `k` | `ctx.attrs.x-cohere-top_k` | integer | Namespaced in attrs |
| `stream` | `op: "llm.stream"` | boolean | Different operation |
| `tools` | `args.tools` | array | Convert Cohere tool format |
| `tool_results` | `args.messages` | array | Add as tool role messages |
| `connectors` | `ctx.attrs.x-cohere-connectors` | array | Namespaced in attrs |

**Message Format Conversion:**
```python
# Cohere chat → CORPUS messages
if "chat_history" in cohere_request:
    corpus_messages = []
    for item in cohere_request["chat_history"]:
        corpus_messages.append({
            "role": "user" if item["role"] == "USER" else "assistant",
            "content": item["message"]
        })
    
    # Add current message
    corpus_messages.append({
        "role": "user",
        "content": cohere_request["message"]
    })
else:
    corpus_messages = [{
        "role": "user",
        "content": cohere_request["message"]
    }]
```

#### Response Mapping
| Cohere Response Field | CORPUS Response Field | Transformation |
|----------------------|----------------------|----------------|
| `text` | `result.text` | Direct mapping |
| `generation_id` | `result.id` | Store in result |
| `token_count.prompt_tokens` | `result.usage.prompt_tokens` | Direct mapping |
| `token_count.response_tokens` | `result.usage.completion_tokens` | Direct mapping |
| `token_count.total_tokens` | `result.usage.total_tokens` | Direct mapping |
| `finish_reason` | `result.finish_reason` | Map values |
| `tool_calls` | `result.tool_calls` | Convert format |

### 4.5 Google AI to CORPUS

#### Request Mapping (Gemini API)
**Status: Needs verification against official Google Gemini API reference**

| Google AI Field | CORPUS Field | Type | Transformation |
|-----------------|--------------|------|----------------|
| `model` | `args.model` | string | Map: `gemini-pro` → `google-gemini-pro` |
| `contents` | `args.messages` | array | Convert parts format |
| `generationConfig.maxOutputTokens` | `args.max_tokens` | integer | Rename and restructure |
| `generationConfig.temperature` | `args.temperature` | float | Same range [0.0, 1.0] → [0.0, 2.0] |
| `generationConfig.topP` | `args.top_p` | float | Same range (0.0, 1.0] |
| `generationConfig.topK` | `ctx.attrs.x-google-top_k` | integer | Namespaced in attrs |
| `safetySettings` | `ctx.attrs.x-google-safety_settings` | array | Namespaced in attrs |
| `tools` | `args.tools` | array | Convert Google tool format |
| `toolConfig` | `args.tool_choice` | object | Convert format |
| `systemInstruction` | `args.system_message` | string | Direct mapping |
| `stopSequences` | `args.stop` | array | Direct mapping |

**Message Format Conversion:**
```python
# Google AI contents → CORPUS messages
corpus_messages = []
for content in google_request.get("contents", []):
    role = "user" if content.get("role") == "user" else "assistant"
    
    # Join parts into single content string
    parts = content.get("parts", [])
    text_parts = [p["text"] for p in parts if "text" in p]
    content_text = " ".join(text_parts)
    
    if content_text:
        corpus_messages.append({
            "role": role,
            "content": content_text
        })
```

#### Response Mapping
| Google AI Response Field | CORPUS Response Field | Transformation |
|-------------------------|----------------------|----------------|
| `candidates[0].content.parts[0].text` | `result.text` | Extract text from parts |
| `modelVersion` | `result.model` | Include version info |
| `usageMetadata.promptTokenCount` | `result.usage.prompt_tokens` | Direct mapping |
| `usageMetadata.candidatesTokenCount` | `result.usage.completion_tokens` | Direct mapping |
| `usageMetadata.totalTokenCount` | `result.usage.total_tokens` | Direct mapping |
| `finishReason` | `result.finish_reason` | Map values: `STOP` → `stop`, `MAX_TOKENS` → `length` |
| `safetyRatings` | Not in result | Store in ctx.attrs |

### 4.6 Azure OpenAI to CORPUS

#### Request Mapping (Chat Completions)
**Primary Source:** [Azure OpenAI Chat Completions Reference](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/reference?view=foundry-classic)

| Azure OpenAI Field | CORPUS Field | Type | Transformation |
|-------------------|--------------|------|----------------|
| `model` | `args.model` | string | Map deployment name to model: `gpt-4-deployment` → `gpt-4` |
| `messages` | `args.messages` | array | Same format as OpenAI |
| `max_tokens` | `args.max_tokens` | integer | Direct mapping |
| `temperature` | `args.temperature` | float | Same range [0.0, 2.0] |
| `top_p` | `args.top_p` | float | Same range (0.0, 1.0] |
| `frequency_penalty` | `args.frequency_penalty` | float | Same range [-2.0, 2.0] |
| `presence_penalty` | `args.presence_penalty` | float | Same range [-2.0, 2.0] |
| `stream` | `op: "llm.stream"` | boolean | Different operation |
| `tools` | `args.tools` | array | Same format as OpenAI |
| `dataSources` | `ctx.attrs.x-azure-data_sources` | array | Namespaced in attrs |
| `enhancements` | `ctx.attrs.x-azure-enhancements` | object | Namespaced in attrs |
| API version in URL | `ctx.attrs.x-azure-api_version` | string | In context attrs |

**Deployment to Model Mapping:**
```python
# Azure deployment names → standard model names
deployment_mapping = {
    "gpt-4-deployment": "gpt-4",
    "gpt-35-turbo-deployment": "gpt-3.5-turbo",
    "gpt-4-32k-deployment": "gpt-4-32k",
    # Add custom mappings as needed
}

azure_deployment = request.get("model")  # e.g., "gpt-4-deployment"
corpus_model = deployment_mapping.get(azure_deployment, azure_deployment)
```

#### Response Mapping
| Azure OpenAI Response Field | CORPUS Response Field | Transformation |
|----------------------------|----------------------|----------------|
| `choices[0].message.content` | `result.text` | Direct mapping |
| `model` | `result.model` | Map deployment back to model name |
| `usage.prompt_tokens` | `result.usage.prompt_tokens` | Direct mapping |
| `usage.completion_tokens` | `result.usage.completion_tokens` | Direct mapping |
| `usage.total_tokens` | `result.usage.total_tokens` | Direct mapping |
| `finish_reason` | `result.finish_reason` | Direct mapping |
| `id` | `ctx.request_id` (if provided) | Context mapping |

---

## 5. Vector Protocol Migrations

### 5.1 Vector Operations Reference
CORPUS v1.0 defines these vector operations (per PROTOCOLS.md):

| Operation | Description | Provider Example |
|-----------|-------------|------------------|
| `vector.capabilities` | Get vector store capabilities | Pinecone `/describe_index_stats` |
| `vector.query` | Single vector similarity search | Pinecone `/query` |
| `vector.batch_query` | Multiple vector searches | Pinecone batch query |
| `vector.upsert` | Insert/update vectors | Pinecone `/vectors/upsert` |
| `vector.delete` | Delete vectors by ID/filter | Pinecone `/vectors/delete` |
| `vector.create_namespace` | Create namespace/collection | Pinecone namespace creation |
| `vector.delete_namespace` | Delete namespace/collection | Pinecone namespace deletion |
| `vector.health` | Check vector service health | Provider health endpoint |

**Provider-Agnostic Mapping Template:**
```json
// General pattern for vector operations
{
  "op": "vector.operation",
  "ctx": {...},
  "args": {
    // Operation-specific arguments per SCHEMA.md
  }
}
```

### 5.2 Pinecone to CORPUS

#### Request Mapping (Query Operation)
**Primary Source:** Pinecone API Reference (metadata filtering guide)

| Pinecone Field | CORPUS Field | Type | Transformation |
|----------------|--------------|------|----------------|
| `vector` | `args.vector` | array[float] | Direct mapping |
| `topK` | `args.top_k` | integer | Rename field |
| `namespace` | `args.namespace` | string | Direct mapping |
| `filter` | `args.filter` | object | Convert Pinecone filter syntax |
| `includeMetadata` | `args.include_metadata` | boolean | Rename field |
| `includeValues` | `args.include_vectors` | boolean | Rename field |
| `sparseVector` | `args.sparse_vector` | object | Direct mapping |
| `id` (for query by ID) | `ctx.attrs.x-pinecone-query_by_id` | string | Namespaced in attrs |
| `id` (for upsert) | `vectors[].id` | string | Restructure for batch |

**Filter Conversion (CORPUS Standard Form):**
```json
// Pinecone filter (using $ operators)
{
  "category": {"$eq": "books"},
  "price": {"$gte": 20, "$lt": 100},
  "tags": {"$in": ["fiction", "scifi"]}
}

// CORPUS filter (operator-object form per SCHEMA.md)
{
  "category": "books",
  "price": {"gte": 20, "lt": 100},
  "tags": {"in": ["fiction", "scifi"]}
}
```

**Wire Envelope Example:**
```json
// Pinecone Query Request
POST /query
{
  "vector": [0.1, 0.2, 0.3],
  "topK": 10,
  "namespace": "documents",
  "filter": {
    "category": {"$eq": "technology"},
    "year": {"$gte": 2020}
  },
  "includeMetadata": true,
  "includeValues": false
}

// CORPUS Equivalent
POST /v1/operations
{
  "op": "vector.query",
  "ctx": {
    "request_id": "vec-123",
    "deadline_ms": 1730312345000
  },
  "args": {
    "vector": [0.1, 0.2, 0.3],
    "top_k": 10,
    "namespace": "documents",
    "filter": {
      "category": "technology",
      "year": {"gte": 2020}
    },
    "include_metadata": true,
    "include_vectors": false
  }
}
```

#### Other Vector Operations
```json
// vector.capabilities example
{
  "op": "vector.capabilities",
  "ctx": {
    "request_id": "cap-req-123"
  },
  "args": {}
}

// vector.batch_query example
{
  "op": "vector.batch_query",
  "ctx": {
    "request_id": "batch-req-456"
  },
  "args": {
    "namespace": "documents",
    "queries": [
      {
        "vector": [0.1, 0.2, 0.3],
        "top_k": 10,
        "filter": {"category": "books"}
      },
      {
        "vector": [0.4, 0.5, 0.6],
        "top_k": 5,
        "filter": {"category": "articles"}
      }
    ]
  }
}

// vector.upsert example
{
  "op": "vector.upsert",
  "ctx": {
    "request_id": "upsert-req-789"
  },
  "args": {
    "namespace": "documents",
    "vectors": [
      {
        "id": "vec-1",
        "vector": [0.1, 0.2, 0.3],
        "metadata": {"category": "book", "author": "Author Name"},
        "sparse_vector": {"indices": [1, 3], "values": [0.5, 0.7]}
      }
    ]
  }
}

// vector.delete example
{
  "op": "vector.delete",
  "ctx": {
    "request_id": "delete-req-101"
  },
  "args": {
    "namespace": "documents",
    "filter": {"category": "obsolete"}  // Delete by filter
    // or "ids": ["vec-1", "vec-2"] for ID-based deletion
  }
}

// vector.create_namespace example (per SCHEMA.md)
{
  "op": "vector.create_namespace",
  "ctx": {
    "request_id": "create-req-112"
  },
  "args": {
    "namespace": "new-collection",
    "dimensions": 768,
    "distance_metric": "cosine"
  }
}

// vector.delete_namespace example
{
  "op": "vector.delete_namespace",
  "ctx": {
    "request_id": "delete-ns-req-113"
  },
  "args": {
    "namespace": "old-collection"
  }
}

// vector.health example
{
  "op": "vector.health",
  "ctx": {
    "request_id": "health-req-114"
  },
  "args": {}
}
```

#### Response Mapping
| Pinecone Response Field | CORPUS Response Field | Transformation |
|------------------------|----------------------|----------------|
| `matches[]` | `result.matches[]` | Array of matches |
| `matches[].id` | `matches[].vector.id` | Nest under vector |
| `matches[].score` | `matches[].score` | Direct mapping |
| `matches[].values` | `matches[].vector.vector` | Nest under vector |
| `matches[].metadata` | `matches[].vector.metadata` | Nest under vector |
| `matches[].sparseValues` | `matches[].vector.sparse_vector` | Nest under vector |
| `namespace` | `result.namespace` | Direct mapping |
| `usage` | `result.usage` | Map to usage object |

#### Error Mapping
| Pinecone Error | CORPUS Error Code | Retryable | Notes |
|----------------|------------------|-----------|-------|
| `429 Too Many Requests` | `RESOURCE_EXHAUSTED` | Yes | Rate limit |
| `400 Bad Request` | `BAD_REQUEST` | No | Invalid parameters |
| `403 Forbidden` | `AUTH_ERROR` | No | Authentication |
| `404 Not Found` | `NAMESPACE_NOT_FOUND` | No | Namespace doesn't exist |
| `422 Unprocessable Entity` | `DIMENSION_MISMATCH` | No | Vector dimension mismatch |
| `503 Service Unavailable` | `UNAVAILABLE` | Yes | Pinecone service down |
| `504 Gateway Timeout` | `DEADLINE_EXCEEDED` | Conditional | Check deadline |

### 5.3 Qdrant to CORPUS

#### Request Mapping (Search Points)
**Primary Source:** Qdrant Search Points Documentation

| Qdrant Field | CORPUS Field | Type | Transformation |
|--------------|--------------|------|----------------|
| `vector` | `args.vector` | array/dict | Qdrant supports named vectors |
| `limit` | `args.top_k` | integer | Rename field |
| `with_payload` | `args.include_metadata` | boolean/array | Rename and handle array case |
| `with_vector` | `args.include_vectors` | boolean | Rename field |
| `filter` | `args.filter` | object | Convert Qdrant filter syntax |
| `score_threshold` | `args.score_threshold` | float | Direct mapping |
| `offset` | `args.offset` | integer | Direct mapping |
| `collection_name` | `args.namespace` | string | Rename field |
| `params` | `ctx.attrs.x-qdrant-params` | object | Namespaced in attrs |

**Filter Conversion:**
```json
// Qdrant filter
{
  "must": [
    {"key": "category", "match": {"value": "books"}},
    {"key": "price", "range": {"gte": 20}}
  ]
}

// CORPUS filter (operator-object form)
{
  "category": "books",
  "price": {"gte": 20}
}
```

### 5.4 Weaviate to CORPUS

#### Request Mapping (GraphQL Get)
**Primary Source:** Weaviate GraphQL API Reference

| Weaviate Field | CORPUS Field | Type | Transformation |
|----------------|--------------|------|----------------|
| `vector` | `args.vector` | array | Direct mapping |
| `limit` | `args.top_k` | integer | Rename field |
| `nearVector` | `args.vector` | object | Extract vector from object |
| `nearText` | `args.text` | string | Text-based search |
| `where` | `args.filter` | object | Convert GraphQL-like filter |
| `className` | `args.namespace` | string | Rename field |
| `_additional` | Selection | object | Map to include flags |
| `autocut` | `ctx.attrs.x-weaviate-autocut` | integer | Namespaced in attrs |

**Filter Conversion:**
```json
// Weaviate where filter
{
  "operator": "And",
  "operands": [
    {
      "path": ["category"],
      "operator": "Equal",
      "valueString": "books"
    },
    {
      "path": ["price"],
      "operator": "GreaterThanEqual",
      "valueNumber": 20
    }
  ]
}

// CORPUS filter (operator-object form)
{
  "category": "books",
  "price": {"gte": 20}
}
```

### 5.5 Milvus to CORPUS

#### Request Mapping
**Status: Needs verification against Milvus REST/SDK reference**

| Milvus Field | CORPUS Field | Type | Transformation |
|--------------|--------------|------|----------------|
| `vector` | `args.vector` | array | Direct mapping |
| `limit` | `args.top_k` | integer | Rename field |
| `output_fields` | Selection | array | Map to include flags |
| `filter` | `args.filter` | string | Keep as string or parse |
| `expr` | `args.filter` | string | Boolean expression |
| `collection_name` | `args.namespace` | string | Rename field |
| `anns_field` | `ctx.attrs.x-milvus-anns_field` | string | Namespaced in attrs |
| `metric_type` | `args.metric` | string | Distance metric |
| `params` | `ctx.attrs.x-milvus-search_params` | object | Namespaced in attrs |

**Note:** Milvus filter expressions as strings can be kept as-is in `args.filter` or parsed into CORPUS operator-object form if possible.

### 5.6 Chroma to CORPUS

#### Request Mapping
**Status: Needs verification against Chroma API documentation**

| Chroma Field | CORPUS Field | Type | Transformation |
|--------------|--------------|------|----------------|
| `query_embeddings` | `args.vector` | array | Can be batch of vectors |
| `n_results` | `args.top_k` | integer | Rename field |
| `where` | `args.filter` | object | Direct mapping (similar syntax) |
| `where_document` | `ctx.attrs.x-chroma-document_filter` | object | Namespaced in attrs |
| `include` | Selection | array/object | Map to include flags |
| `collection_name` | `args.namespace` | string | Direct mapping |
| `query_texts` | `args.text` | array | Text-based search |

**Filter Conversion:**
```json
// Chroma where filter
{
  "where": {
    "category": {"$eq": "books"},
    "price": {"$gte": 20}
  }
}

// CORPUS filter (operator-object form)
{
  "category": "books",
  "price": {"gte": 20}
}
```

---

## 6. Embedding Protocol Migrations

### 6.1 Embedding Operations Reference
CORPUS v1.0 defines these embedding operations (per PROTOCOLS.md):

| Operation | Description | Provider Example |
|-----------|-------------|------------------|
| `embedding.capabilities` | Get embedding model capabilities | OpenAI `/models` endpoint |
| `embedding.embed` | Single text embedding | OpenAI `/embeddings` |
| `embedding.embed_batch` | Batch text embeddings | OpenAI `/embeddings` with array input |
| `embedding.stream_embed` | Streaming embeddings | Provider-specific streaming |
| `embedding.count_tokens` | Count tokens in text | Provider token counting |
| `embedding.get_stats` | Get embedding statistics | Provider usage statistics |
| `embedding.health` | Check embedding service health | Provider health endpoint |

**Provider-Agnostic Mapping Template:**
```json
// General pattern for embedding operations
{
  "op": "embedding.operation",
  "ctx": {...},
  "args": {
    // Operation-specific arguments per SCHEMA.md
  }
}
```

### 6.2 OpenAI Embeddings to CORPUS

#### Request Mapping (Embeddings API)
**Primary Source:** [OpenAI Embeddings API Reference](https://platform.openai.com/docs/api-reference/embeddings)

| OpenAI Field | CORPUS Field | Type | Transformation |
|--------------|--------------|------|----------------|
| `model` | `args.model` | string | Map: `text-embedding-ada-002` → `ada-002` |
| `input` (single) | `args.text` | string | Single embedding |
| `input` (batch) | `args.texts` | array | Batch embedding |
| `encoding_format` | `ctx.attrs.x-openai-encoding_format` | string | Namespaced in attrs |
| `user` | `ctx.attrs.x-openai-user` | string | In context attrs |
| `dimensions` | `args.dimensions` | integer | Reduce dimensions |

**Single vs Batch Operations:**
```json
// Single embedding - OpenAI
POST /v1/embeddings
{
  "model": "text-embedding-ada-002",
  "input": "The quick brown fox",
  "encoding_format": "float"
}

// Single embedding - CORPUS
{
  "op": "embedding.embed",
  "ctx": {
    "request_id": "embed-req-123",
    "attrs": {
      "x-openai-encoding_format": "float"
    }
  },
  "args": {
    "text": "The quick brown fox",
    "model": "ada-002"
  }
}

// Batch embedding - OpenAI
{
  "model": "text-embedding-ada-002",
  "input": ["Text 1", "Text 2", "Text 3"]
}

// Batch embedding - CORPUS
{
  "op": "embedding.embed_batch",
  "ctx": {
    "request_id": "batch-embed-req-456"
  },
  "args": {
    "texts": ["Text 1", "Text 2", "Text 3"],
    "model": "ada-002"
  }
}
```

#### Other Embedding Operations
```json
// embedding.capabilities example
{
  "op": "embedding.capabilities",
  "ctx": {
    "request_id": "cap-req-123"
  },
  "args": {}
}

// embedding.count_tokens example (per SCHEMA.md: text and model required)
{
  "op": "embedding.count_tokens",
  "ctx": {
    "request_id": "count-req-456"
  },
  "args": {
    "model": "ada-002",
    "text": "Hello world"
  }
}

// embedding.get_stats example
{
  "op": "embedding.get_stats",
  "ctx": {
    "request_id": "stats-req-789"
  },
  "args": {}
}

// embedding.health example
{
  "op": "embedding.health",
  "ctx": {
    "request_id": "health-req-890"
  },
  "args": {}
}
```

#### Response Mapping
| OpenAI Response Field | CORPUS Response Field | Transformation |
|----------------------|----------------------|----------------|
| `data[].embedding` | `result.embedding.vector` | Single embedding |
| `data[].embedding` | `result.embeddings[].vector` | Batch embeddings |
| `model` | `result.model` | Normalized model name |
| `usage.prompt_tokens` | `result.total_tokens` | Token usage |
| `object` | Not included | Type indicator |
| `data[].index` | `result.embeddings[].index` | Preserve order |

### 6.3 Cohere Embed to CORPUS

#### Request Mapping
**Status: Needs verification against Cohere Embed API reference**

| Cohere Field | CORPUS Field | Type | Transformation |
|--------------|--------------|------|----------------|
| `model` | `args.model` | string | Map: `embed-english-v3.0` → `cohere-v3-en` |
| `texts` | `args.texts` | array | Always batch in Cohere |
| `input_type` | `ctx.attrs.x-cohere-input_type` | string | Namespaced in attrs |
| `embedding_types` | `ctx.attrs.x-cohere-embedding_types` | array | Namespaced in attrs |
| `truncate` | `args.truncate` | boolean/string | Map to boolean or string |
| `compress` | `ctx.attrs.x-cohere-compress` | boolean | Namespaced in attrs |

### 6.4 HuggingFace to CORPUS

#### Request Mapping
**Status: Needs verification against Hugging Face Inference API docs**

| HuggingFace Field | CORPUS Field | Type | Transformation |
|-------------------|--------------|------|----------------|
| `model` | `args.model` | string | Map: `sentence-transformers/all-MiniLM-L6-v2` → `miniLM-L6-v2` |
| `inputs` | `args.texts` | array/string | Can be single or batch |
| `normalize` | `args.normalize` | boolean | Direct mapping |
| `truncate` | `args.truncate` | boolean | Direct mapping |
| `options` | `ctx.attrs.x-huggingface-options` | object | Namespaced in attrs |
| `parameters` | `ctx.attrs.x-huggingface-parameters` | object | Namespaced in attrs |

### 6.5 Google Vertex AI to CORPUS

#### Request Mapping
**Status: Needs verification against Vertex AI embeddings reference**

| Vertex AI Field | CORPUS Field | Type | Transformation |
|-----------------|--------------|------|----------------|
| `instances[]` | `args.texts` | array | Extract from instances |
| `parameters` | `args` fields | object | Map to CORPUS args |
| `endpoint` | `args.model` | string | Extract model from endpoint |

**Instances Structure:**
```json
// Vertex AI request
{
  "instances": [
    {"content": "The quick brown fox"},
    {"content": "AI is amazing"}
  ],
  "parameters": {
    "autoTruncate": true,
    "outputDimensionality": 256
  }
}

// CORPUS equivalent
{
  "op": "embedding.embed_batch",
  "ctx": {...},
  "args": {
    "texts": ["The quick brown fox", "AI is amazing"],
    "model": "google-textembedding-gecko",
    "truncate": true,
    "dimensions": 256
  }
}
```

### 6.6 AWS Bedrock to CORPUS

#### Request Mapping
**Primary Source:** AWS Bedrock Titan Embeddings Documentation

| Bedrock Field | CORPUS Field | Type | Transformation |
|---------------|--------------|------|----------------|
| `modelId` | `args.model` | string | Map: `amazon.titan-embed-text-v1` → `titan-embed-text-v1` |
| `inputText` | `args.text` | string | Single text |
| `inputTexts` | `args.texts` | array | Batch texts |
| `dimensions` | `args.dimensions` | integer | Output dimensions |
| `normalize` | `args.normalize` | boolean | Direct mapping |
| `embeddingTypes` | `ctx.attrs.x-bedrock-embedding_types` | array | Namespaced in attrs |

---

## 7. Graph Protocol Migrations

### 7.1 Graph Operations Reference
CORPUS v1.0 defines these graph operations (per PROTOCOLS.md):

| Operation | Description | Provider Example |
|-----------|-------------|------------------|
| `graph.capabilities` | Get graph database capabilities | Neo4j endpoint info |
| `graph.query` | Execute query (Cypher/Gremlin/etc.) | Neo4j transaction endpoint |
| `graph.stream_query` | Stream query results | Neo4j streaming endpoint |
| `graph.upsert_nodes` | Insert/update nodes | Neo4j CREATE/MERGE |
| `graph.upsert_edges` | Insert/update edges/relationships | Neo4j CREATE/MERGE |
| `graph.delete_nodes` | Delete nodes | Neo4j DELETE |
| `graph.delete_edges` | Delete edges/relationships | Neo4j DELETE |
| `graph.bulk_vertices` | Bulk vertex operations | Neo4j LOAD CSV |
| `graph.batch` | Batch operations | Neo4j transaction with multiple statements |
| `graph.transaction` | Transaction management | Neo4j transaction endpoints |
| `graph.traversal` | Graph traversal | Gremlin traversal |
| `graph.get_schema` | Get graph schema | Neo4j `CALL db.schema.visualization()` |
| `graph.health` | Check graph service health | Provider health endpoint |

**Provider-Agnostic Mapping Template:**
```json
// General pattern for graph operations
{
  "op": "graph.operation",
  "ctx": {...},
  "args": {
    // Operation-specific arguments per SCHEMA.md
  }
}
```

### 7.2 Neo4j to CORPUS

#### Request Mapping (Transactional HTTP API)
**Status: Needs verification against Neo4j HTTP API reference**

| Neo4j Field | CORPUS Field | Type | Transformation |
|-------------|--------------|------|----------------|
| `cypher` | `args.text` | string | Cypher query text |
| `params` | `args.params` | object | Query parameters |
| `database` | `args.namespace` | string | Database name |
| `resultDataContents` | `ctx.attrs.x-neo4j-result_data_contents` | array | Namespaced in attrs |
| `includeStats` | `args.include_stats` | boolean | Rename field |

**Wire Envelope Examples:**
```json
// Neo4j Cypher Query
POST /db/neo4j/tx/commit
{
  "statements": [
    {
      "statement": "MATCH (n:Person) WHERE n.name = $name RETURN n",
      "parameters": {"name": "Alice"},
      "resultDataContents": ["row", "graph"]
    }
  ]
}

// CORPUS Equivalent
{
  "op": "graph.query",
  "ctx": {
    "request_id": "graph-123",
    "deadline_ms": 1730312345000,
    "attrs": {
      "x-neo4j-result_data_contents": ["row", "graph"]
    }
  },
  "args": {
    "dialect": "cypher",
    "text": "MATCH (n:Person) WHERE n.name = $name RETURN n",
    "params": {"name": "Alice"},
    "namespace": "neo4j",
    "include_stats": true
  }
}

// graph.stream_query example
{
  "op": "graph.stream_query",
  "ctx": {
    "request_id": "stream-query-456"
  },
  "args": {
    "dialect": "cypher",
    "text": "MATCH (n:Person) RETURN n LIMIT 1000",
    "namespace": "neo4j"
  }
}

// graph.upsert_nodes example (per SCHEMA.md Node shape)
{
  "op": "graph.upsert_nodes",
  "ctx": {
    "request_id": "upsert-nodes-789"
  },
  "args": {
    "dialect": "cypher",
    "namespace": "neo4j",
    "nodes": [
      {
        "id": "person-1",
        "labels": ["Person"],
        "properties": {
          "name": "Alice",
          "age": 30
        }
      }
    ]
  }
}

// graph.upsert_edges example (per SCHEMA.md Edge shape)
{
  "op": "graph.upsert_edges",
  "ctx": {
    "request_id": "upsert-edges-890"
  },
  "args": {
    "dialect": "cypher",
    "namespace": "neo4j",
    "edges": [
      {
        "id": "edge-1",
        "src": "person-1",
        "dst": "person-2",
        "label": "KNOWS",
        "properties": {"since": 2020}
      }
    ]
  }
}

// graph.get_schema example
{
  "op": "graph.get_schema",
  "ctx": {
    "request_id": "schema-req-901"
  },
  "args": {}
}

// graph.batch example (per SCHEMA.md GraphBatchSpec)
{
  "op": "graph.batch",
  "ctx": {
    "request_id": "batch-req-902"
  },
  "args": {
    "namespace": "neo4j",
    "ops": [
      {
        "op": "graph.query",
        "args": {
          "dialect": "cypher",
          "text": "CREATE (n:Person {name: $name})",
          "params": {"name": "Alice"}
        }
      },
      {
        "op": "graph.query",
        "args": {
          "dialect": "cypher",
          "text": "CREATE (n:Person {name: $name})",
          "params": {"name": "Bob"}
        }
      }
    ]
  }
}

// graph.transaction example (per SCHEMA.md GraphTransactionSpec)
{
  "op": "graph.transaction",
  "ctx": {
    "request_id": "tx-req-903"
  },
  "args": {
    "namespace": "neo4j",
    "operations": [
      {
        "op": "graph.upsert_nodes",
        "args": {
          "dialect": "cypher",
          "nodes": [{
            "id": "person-3",
            "labels": ["Person"],
            "properties": {"name": "Charlie"}
          }]
        }
      },
      {
        "op": "graph.upsert_edges",
        "args": {
          "dialect": "cypher",
          "edges": [{
            "id": "edge-2",
            "src": "person-1",
            "dst": "person-3",
            "label": "KNOWS",
            "properties": {"since": 2021}
          }]
        }
      }
    ]
  }
}

// graph.health example
{
  "op": "graph.health",
  "ctx": {
    "request_id": "health-req-904"
  },
  "args": {}
}
```

#### Response Mapping
| Neo4j Response Field | CORPUS Response Field | Transformation |
|---------------------|----------------------|----------------|
| `results[]` | `result.records[]` | Multiple statements |
| `results[].columns` | Column names | Implicit in records |
| `results[].data[]` | `records[]` | Row data |
| `results[].stats` | `result.summary.stats` | Query statistics |
| `errors` | Error envelope | Convert to CORPUS error |

### 7.3 Amazon Neptune to CORPUS

#### Request Mapping (Gremlin API)
**Primary Source:** Amazon Neptune Gremlin REST API Documentation

| Neptune Field | CORPUS Field | Type | Transformation |
|--------------|--------------|------|----------------|
| `gremlin` | `args.text` | string | Gremlin query |
| `sparql` | `args.text` | string | SPARQL query |
| `opencypher` | `args.text` | string | OpenCypher query |
| Query type | `args.dialect` | string | `gremlin|sparql|opencypher` |
| `profile` | `ctx.attrs.x-neptune-profile` | boolean | Namespaced in attrs |

### 7.4 JanusGraph to CORPUS

#### Request Mapping (Gremlin Server)
**Status: Needs verification against JanusGraph Gremlin Server reference**

| JanusGraph Field | CORPUS Field | Type | Transformation |
|------------------|--------------|------|----------------|
| `gremlin` | `args.text` | string | Gremlin query |
| `bindings` | `args.params` | object | Query parameters |
| Graph name | `args.namespace` | string | Graph namespace |
| `language` | `ctx.attrs.x-janusgraph-language` | string | Namespaced in attrs |
| `aliases` | `ctx.attrs.x-janusgraph-aliases` | object | Namespaced in attrs |
| `session` | `ctx.attrs.x-janusgraph-session` | string | Namespaced in attrs |
| `timeout` | `ctx.attrs.x-janusgraph-timeout` | integer | Namespaced in attrs |

### 7.5 TigerGraph to CORPUS

#### Request Mapping (RESTPP API)
**Status: Needs verification against TigerGraph RESTPP documentation**

| TigerGraph Field | CORPUS Field | Type | Transformation |
|------------------|--------------|------|----------------|
| `query` | `args.text` | string | GSQL query |
| Graph name | `args.namespace` | string | Graph name |
| `params` | `args.params` | object | Query parameters |
| Query name in URL | `ctx.attrs.x-tigergraph-query_name` | string | Namespaced in attrs |

### 7.6 ArangoDB to CORPUS

#### Request Mapping (AQL Cursor API)
**Status: Needs verification against ArangoDB HTTP API cursor reference**

| ArangoDB Field | CORPUS Field | Type | Transformation |
|----------------|--------------|------|----------------|
| `query` | `args.text` | string | AQL query |
| `bindVars` | `args.params` | object | Query parameters |
| Database name | `args.namespace` | string | Database name |
| `count` | `args.include_count` | boolean | Include count in result |
| `batchSize` | `ctx.attrs.x-arangodb-batch_size` | integer | Namespaced in attrs |
| `ttl` | `ctx.attrs.x-arangodb-ttl` | number | Namespaced in attrs |
| `options` | `ctx.attrs.x-arangodb-options` | object | Namespaced in attrs |

---

## 8. Error Code Mapping

### Cross-Provider Error Translation
| Provider Error Pattern | CORPUS Error Code | HTTP Code | Retry Strategy |
|------------------------|------------------|-----------|----------------|
| Rate limiting | `RESOURCE_EXHAUSTED` | 429 | Exponential backoff with jitter |
| Authentication failure | `AUTH_ERROR` | 401/403 | Fix credentials, no retry |
| Invalid parameters | `BAD_REQUEST` | 400 | Fix request, no retry |
| Network timeout | `TRANSIENT_NETWORK` | 502/504 | Retry with backoff |
| Service unavailable | `UNAVAILABLE` | 503 | Circuit breaker pattern |
| Not implemented | `NOT_SUPPORTED` | 501 | No retry |
| Deadline exceeded | `DEADLINE_EXCEEDED` | 504 | Extend deadline or reduce load |
| Content filtered | `BAD_REQUEST` | 400 | Modify content, no retry |
| Quota exceeded | `RESOURCE_EXHAUSTED` | 429 | Wait for quota reset |

### Provider-Specific Error Details
**OpenAI/Anthropic/Cohere**: Map specific error types to CORPUS codes
**Vector DBs**: Handle dimension mismatches, index not ready
**Embedding**: Model not available, text too long
**Graph DBs**: Query syntax errors, constraint violations

---

## 9. Context Propagation

### Standard Context Fields
| Provider Context | CORPUS `ctx` Field | Example |
|-----------------|-------------------|---------|
| `X-Request-ID` | `request_id` | `req-123` |
| `X-Idempotency-Key` | `idempotency_key` | `idem-456` |
| Deadline header | `deadline_ms` | Absolute epoch ms |
| `traceparent` | `traceparent` | W3C Trace Context |
| Tenant header | `tenant` | Hashed in metrics |
| User context | `ctx.attrs` | Namespaced vendor attrs |

### Provider-Specific Context Mapping
**AWS**: `X-Amzn-Trace-Id` → `traceparent`
**Google Cloud**: `X-Cloud-Trace-Context` → `traceparent`
**Azure**: `traceparent` header (W3C standard)
**Custom headers**: Map to `ctx.attrs` with `x-` prefix

---

## 10. Migration Validation Checklist

### Pre-Migration Checks
- [ ] Provider API documentation reviewed
- [ ] CORPUS protocol specification (PROTOCOLS.md) understood
- [ ] CORPUS schema (SCHEMA.md) reviewed for field requirements
- [ ] Test environment configured

### Wire Transformation Tests
- [ ] Request envelope transformation validated (exactly {op, ctx, args})
- [ ] Response envelope transformation validated (closed envelopes)
- [ ] Error mapping tested with all provider errors
- [ ] Context propagation verified
- [ ] Streaming support validated with `code: "STREAMING"`

### Schema Compliance
- [ ] All requests validate against CORPUS request schema
- [ ] All responses validate against CORPUS response/error schemas
- [ ] Streaming frames use closed envelope with `chunk` field only
- [ ] Filter expressions use operator-object form (not field__gte)
- [ ] Vendor extensions use `x-` prefix in `ctx.attrs`

### Production Readiness
- [ ] Observability integrated (metrics, logs, traces)
- [ ] Security requirements met (tenant isolation, etc.)
- [ ] Documentation updated
- [ ] Rollback plan prepared

---

## 11. References

### CORPUS Documentation
- **PROTOCOLS.md**: Wire format specification and operation semantics
- **SCHEMA.md**: JSON Schema definitions and envelope closure rules
- **ERRORS.md**: Error taxonomy and handling guidelines

### Provider Documentation
- **OpenAI API Reference**: https://platform.openai.com/docs/api-reference/chat
- **OpenAI Embeddings API**: https://platform.openai.com/docs/api-reference/embeddings
- **Azure OpenAI Reference**: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/reference?view=foundry-classic
- **Pinecone API**: https://docs.pinecone.io/reference/api/overview
- **Qdrant Search Points**: Qdrant documentation
- **Weaviate GraphQL**: Weaviate documentation
- **Anthropic Messages API**: Anthropic documentation

### Standards
- **W3C Trace Context**: https://www.w3.org/TR/trace-context/
- **JSON Schema**: https://json-schema.org/

---

### Testing Considerations
When implementing migrations based on this guide:
1. Validate against SCHEMA.md for all envelope structures
2. Test streaming termination conditions (final chunk or error)
3. Verify filter expressions use correct operator-object syntax
4. Ensure vendor extensions don't leak into response envelopes
5. Verify operation arguments match PROTOCOLS.md specifications
6. For sections marked "Needs verification", confirm with current provider documentation

---

*CORPUS Protocol Suite Migration Reference Guide v1.0*
