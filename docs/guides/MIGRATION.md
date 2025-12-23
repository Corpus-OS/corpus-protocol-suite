# CORPUS Protocol Suite - Migration Reference Guide

**Version:** 1.0 • **Protocol Compatibility:** v1.0 • **Last Updated:** January 2025

---

## 📋 Table of Contents

### [1. 🎯 Executive Summary](#1--executive-summary)
### [2. 📖 How to Use This Guide](#2--how-to-use-this-guide)
### [3. 🔧 Core Migration Patterns](#3--core-migration-patterns)
### [4. 🤖 LLM Protocol Migrations](#4--llm-protocol-migrations)
  - [4.1 OpenAI to CORPUS](#41-openai-to-corpus)
  - [4.2 Anthropic to CORPUS](#42-anthropic-to-corpus)
  - [4.3 Cohere to CORPUS](#43-cohere-to-corpus)
  - [4.4 Google AI to CORPUS](#44-google-ai-to-corpus)
  - [4.5 Azure OpenAI to CORPUS](#45-azure-openai-to-corpus)
### [5. 🔍 Vector Protocol Migrations](#5--vector-protocol-migrations)
  - [5.1 Pinecone to CORPUS](#51-pinecone-to-corpus)
  - [5.2 Qdrant to CORPUS](#52-qdrant-to-corpus)
  - [5.3 Weaviate to CORPUS](#53-weaviate-to-corpus)
  - [5.4 Milvus to CORPUS](#54-milvus-to-corpus)
  - [5.5 Chroma to CORPUS](#55-chroma-to-corpus)
### [6. 📊 Embedding Protocol Migrations](#6--embedding-protocol-migrations)
  - [6.1 OpenAI Embeddings to CORPUS](#61-openai-embeddings-to-corpus)
  - [6.2 Cohere Embed to CORPUS](#62-cohere-embed-to-corpus)
  - [6.3 HuggingFace to CORPUS](#63-huggingface-to-corpus)
  - [6.4 Google Vertex AI to CORPUS](#64-google-vertex-ai-to-corpus)
  - [6.5 AWS Bedrock to CORPUS](#65-aws-bedrock-to-corpus)
### [7. 🌐 Graph Protocol Migrations](#7--graph-protocol-migrations)
  - [7.1 Neo4j to CORPUS](#71-neo4j-to-corpus)
  - [7.2 Amazon Neptune to CORPUS](#72-amazon-neptune-to-corpus)
  - [7.3 JanusGraph to CORPUS](#73-janusgraph-to-corpus)
  - [7.4 TigerGraph to CORPUS](#74-tigergraph-to-corpus)
  - [7.5 ArangoDB to CORPUS](#75-arangodb-to-corpus)
### [8. ⚡ Error Code Mapping](#8--error-code-mapping)
### [9. 🔗 Context Propagation](#9--context-propagation)
### [10. ✅ Migration Validation Checklist](#10--migration-validation-checklist)
### [11. 📚 References](#11--references)

---

## 1. 🎯 Executive Summary

### What This Guide Provides
This document is your **definitive reference** for migrating existing AI service APIs to the CORPUS Protocol Suite. It provides **wire-level mapping tables** showing exactly how to transform requests and responses between vendor-specific formats and the standardized CORPUS protocol.

### Key Migration Benefits
- **Unified Interface**: Replace multiple provider APIs with one consistent protocol
- **Vendor Agnostic**: Switch between providers without code changes
- **Production Ready**: Built-in observability, error handling, and security
- **Future Proof**: Protocol evolves independently of provider changes

### Scope Boundaries
| What's Included | What's Excluded |
|----------------|-----------------|
| Wire format translations | SDK implementation details |
| Parameter mapping tables | Operational deployment guides |
| Error code normalization | Business logic guidance |
| Context propagation | Performance optimization |
| Schema validation rules | Provider-specific features |

---

## 2. 📖 How to Use This Guide

### For Adapter Developers
1. **Find your provider** in the relevant protocol section (LLM, Vector, etc.)
2. **Follow the mapping tables** to transform requests/responses
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
- 📊 **Tables**: Wire-level mappings
- ⚠️ **Notes**: Important migration considerations

---

## 3. 🔧 Core Migration Patterns

### 3.1 The CORPUS Envelope Pattern
**Every CORPUS request follows this structure:**

```json
{
  "op": "protocol.operation",    // What operation to perform
  "ctx": {                       // Request context metadata
    "request_id": "req-123",
    "deadline_ms": 1730312345000,
    "tenant": "acme-corp",
    "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
  },
  "args": {                      // Operation-specific parameters
    // ... varies by operation
  }
}
```

### 3.2 Response Transformation
**Provider responses transform to CORPUS envelopes:**

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
**Streaming operations use chunked envelopes:**

```json
// Each stream frame
{
  "ok": true,
  "code": "OK",
  "ms": 15.7,
  "chunk": {
    "text": "Hello",
    "is_final": false
  }
}
```

---

## 4. 🤖 LLM Protocol Migrations

### 4.1 OpenAI to CORPUS

#### Request Mapping
| OpenAI Field | CORPUS Field | Type | Transformation Required |
|--------------|--------------|------|-------------------------|
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
| `seed` | `args.seed` | integer | Direct mapping |

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
  "max_tokens": 100
}

// CORPUS Equivalent
POST /v1/operations
{
  "op": "llm.complete",
  "ctx": {
    "request_id": "req-123",
    "deadline_ms": 1730312345000
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

**Streaming Differences:**
```json
// OpenAI streaming chunk
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}

// CORPUS streaming chunk
{"ok": true, "code": "OK", "ms": 15.7, "chunk": {"text": "Hello", "is_final": false}}
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

---

### 4.2 Anthropic to CORPUS

#### Request Mapping
| Anthropic Field | CORPUS Field | Type | Transformation Required |
|----------------|--------------|------|-------------------------|
| `model` | `args.model` | string | Map: `claude-3-opus-20240229` → `claude-3-opus` |
| `messages` | `args.messages` | array | Convert Anthropic format: `{role, content}` → same |
| `max_tokens` | `args.max_tokens` | integer | Direct mapping |
| `temperature` | `args.temperature` | float | Same range [0.0, 1.0] → [0.0, 2.0] |
| `top_p` | `args.top_p` | float | Same range (0.0, 1.0] |
| `top_k` | `args.top_k` | integer | Via extensions: `extensions.anthropic:top_k` |
| `stream` | `op: "llm.stream"` | boolean | Different operation |
| `system` | `args.system_message` | string | Move from messages array |
| `tools` | `args.tools` | array | Convert Anthropic tool format |
| `tool_choice` | `args.tool_choice` | object | Convert format |

**Message Format Conversion:**
```python
# Anthropic messages → CORPUS messages
anthropic_messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]

# CORPUS messages (same format)
corpus_messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]

# System message handling
if anthropic_request.get("system"):
    corpus_args["system_message"] = anthropic_request["system"]
```

**Wire Envelope Example:**
```json
// Anthropic Request
POST /v1/messages
{
  "model": "claude-3-opus-20240229",
  "max_tokens": 1024,
  "messages": [{"role": "user", "content": "Hello"}],
  "system": "You are a helpful assistant"
}

// CORPUS Equivalent
POST /v1/operations
{
  "op": "llm.complete",
  "ctx": {
    "request_id": "req-456",
    "deadline_ms": 1730312345000
  },
  "args": {
    "model": "claude-3-opus",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 1024,
    "system_message": "You are a helpful assistant"
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

#### Error Mapping
| Anthropic Error | CORPUS Error Code | Retryable | Notes |
|----------------|------------------|-----------|-------|
| `rate_limit_error` | `RESOURCE_EXHAUSTED` | Yes | Add `retry_after_ms` |
| `authentication_error` | `AUTH_ERROR` | No | Direct mapping |
| `invalid_request_error` | `BAD_REQUEST` | No | Direct mapping |
| `overloaded_error` | `UNAVAILABLE` | Yes | Provider overloaded |
| `api_error` | `UNAVAILABLE` | Yes | Provider issue |
| `timeout_error` | `DEADLINE_EXCEEDED` | Conditional | Check deadline |

---

### 4.3 Cohere to CORPUS

#### Request Mapping
| Cohere Field | CORPUS Field | Type | Transformation Required |
|--------------|--------------|------|-------------------------|
| `model` | `args.model` | string | Map: `command-r-plus` → `cohere-command-r-plus` |
| `message` | `args.messages` | string | Convert to messages array |
| `chat_history` | `args.messages` | array | Merge with current message |
| `max_tokens` | `args.max_tokens` | integer | Direct mapping |
| `temperature` | `args.temperature` | float | Same range [0.0, 1.0] → [0.0, 2.0] |
| `p` | `args.top_p` | float | Rename field |
| `k` | `args.top_k` | integer | Via extensions: `extensions.cohere:top_k` |
| `stream` | `op: "llm.stream"` | boolean | Different operation |
| `tools` | `args.tools` | array | Convert Cohere tool format |
| `tool_results` | `args.messages` | array | Add as tool role messages |

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

**Wire Envelope Example:**
```json
// Cohere Request
POST /v1/chat
{
  "model": "command-r-plus",
  "message": "What is AI?",
  "chat_history": [
    {"role": "USER", "message": "Hello"},
    {"role": "CHATBOT", "message": "Hi there!"}
  ],
  "temperature": 0.3,
  "max_tokens": 200
}

// CORPUS Equivalent
POST /v1/operations
{
  "op": "llm.complete",
  "ctx": {
    "request_id": "req-789",
    "deadline_ms": 1730312345000
  },
  "args": {
    "model": "cohere-command-r-plus",
    "messages": [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hi there!"},
      {"role": "user", "content": "What is AI?"}
    ],
    "temperature": 0.3,
    "max_tokens": 200
  }
}
```

#### Response Mapping
| Cohere Response Field | CORPUS Response Field | Transformation |
|----------------------|----------------------|----------------|
| `text` | `result.text` | Direct mapping |
| `generation_id` | `result.model` (or context) | Store in `details` |
| `token_count.prompt_tokens` | `result.usage.prompt_tokens` | Direct mapping |
| `token_count.response_tokens` | `result.usage.completion_tokens` | Direct mapping |
| `token_count.total_tokens` | `result.usage.total_tokens` | Direct mapping |
| `finish_reason` | `result.finish_reason` | Map values |
| `tool_calls` | `result.tool_calls` | Convert format |

**Tool Call Conversion:**
```json
// Cohere tool call
{
  "name": "get_weather",
  "parameters": {"city": "San Francisco"}
}

// CORPUS tool call
{
  "id": "call_123",  // Generate if not provided
  "type": "function",
  "function": {
    "name": "get_weather",
    "arguments": "{\"city\": \"San Francisco\"}"
  }
}
```

#### Error Mapping
| Cohere Error | CORPUS Error Code | Retryable | Notes |
|--------------|------------------|-----------|-------|
| `rate_limit_error` | `RESOURCE_EXHAUSTED` | Yes | Add `retry_after_ms` |
| `authentication_error` | `AUTH_ERROR` | No | Direct mapping |
| `invalid_request_error` | `BAD_REQUEST` | No | Direct mapping |
| `internal_server_error` | `UNAVAILABLE` | Yes | Provider issue |
| `service_unavailable` | `UNAVAILABLE` | Yes | Provider down |
| `timeout_error` | `DEADLINE_EXCEEDED` | Conditional | Check deadline |

---

### 4.4 Google AI (Gemini) to CORPUS

#### Request Mapping
| Google AI Field | CORPUS Field | Type | Transformation Required |
|-----------------|--------------|------|-------------------------|
| `model` | `args.model` | string | Map: `gemini-pro` → `google-gemini-pro` |
| `contents` | `args.messages` | array | Convert parts format |
| `generationConfig.maxOutputTokens` | `args.max_tokens` | integer | Rename and restructure |
| `generationConfig.temperature` | `args.temperature` | float | Same range [0.0, 1.0] → [0.0, 2.0] |
| `generationConfig.topP` | `args.top_p` | float | Same range (0.0, 1.0] |
| `generationConfig.topK` | `args.top_k` | integer | Via extensions |
| `safetySettings` | Not directly mapped | array | Handle via extensions |
| `tools` | `args.tools` | array | Convert Google tool format |
| `toolConfig` | `args.tool_choice` | object | Convert format |

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

**Wire Envelope Example:**
```json
// Google AI Request
POST /v1/models/gemini-pro:generateContent
{
  "contents": [
    {
      "role": "user",
      "parts": [{"text": "Explain quantum computing"}]
    }
  ],
  "generationConfig": {
    "temperature": 0.7,
    "maxOutputTokens": 200
  },
  "safetySettings": [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
  ]
}

// CORPUS Equivalent
POST /v1/operations
{
  "op": "llm.complete",
  "ctx": {
    "request_id": "req-901",
    "deadline_ms": 1730312345000
  },
  "args": {
    "model": "google-gemini-pro",
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "temperature": 0.7,
    "max_tokens": 200
  },
  "extensions": {
    "google:safety_settings": [
      {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    ]
  }
}
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
| `safetyRatings` | Not in result | Store in `extensions` |

**Safety Handling:**
```json
{
  "ok": true,
  "code": "OK",
  "ms": 145.2,
  "result": {
    "text": "Quantum computing is...",
    "model": "google-gemini-pro",
    "usage": {...}
  },
  "extensions": {
    "google:safety_ratings": [
      {
        "category": "HARM_CATEGORY_HARASSMENT",
        "probability": "LOW",
        "blocked": false
      }
    ]
  }
}
```

#### Error Mapping
| Google AI Error | CORPUS Error Code | Retryable | Notes |
|-----------------|------------------|-----------|-------|
| `RESOURCE_EXHAUSTED` | `RESOURCE_EXHAUSTED` | Yes | Direct mapping |
| `PERMISSION_DENIED` | `AUTH_ERROR` | No | Direct mapping |
| `INVALID_ARGUMENT` | `BAD_REQUEST` | No | Direct mapping |
| `UNAVAILABLE` | `UNAVAILABLE` | Yes | Direct mapping |
| `DEADLINE_EXCEEDED` | `DEADLINE_EXCEEDED` | Conditional | Check deadline |
| `FAILED_PRECONDITION` | `BAD_REQUEST` | No | Check preconditions |

---

### 4.5 Azure OpenAI to CORPUS

#### Request Mapping
| Azure OpenAI Field | CORPUS Field | Type | Transformation Required |
|-------------------|--------------|------|-------------------------|
| `model` | `args.model` | string | Map deployment name to model: `gpt-4-deployment` → `gpt-4` |
| `messages` | `args.messages` | array | Same format as OpenAI |
| `max_tokens` | `args.max_tokens` | integer | Direct mapping |
| `temperature` | `args.temperature` | float | Same range [0.0, 2.0] |
| `top_p` | `args.top_p` | float | Same range (0.0, 1.0] |
| `frequency_penalty` | `args.frequency_penalty` | float | Same range [-2.0, 2.0] |
| `presence_penalty` | `args.presence_penalty` | float | Same range [-2.0, 2.0] |
| `stream` | `op: "llm.stream"` | boolean | Different operation |
| `tools` | `args.tools` | array | Same format as OpenAI |
| `dataSources` | `extensions.azure:data_sources` | array | Azure-specific feature |
| `enhancements` | `extensions.azure:enhancements` | object | Azure-specific feature |

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

**Wire Envelope Example:**
```json
// Azure OpenAI Request
POST /openai/deployments/gpt-4-deployment/chat/completions?api-version=2023-12-01-preview
{
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello"}
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "dataSources": [
    {
      "type": "AzureCognitiveSearch",
      "parameters": {...}
    }
  ]
}

// CORPUS Equivalent
POST /v1/operations
{
  "op": "llm.complete",
  "ctx": {
    "request_id": "req-234",
    "deadline_ms": 1730312345000
  },
  "args": {
    "model": "gpt-4",
    "messages": [
      {"role": "system", "content": "You are helpful"},
      {"role": "user", "content": "Hello"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  },
  "extensions": {
    "azure:data_sources": [
      {
        "type": "AzureCognitiveSearch",
        "parameters": {...}
      }
    ],
    "azure:api_version": "2023-12-01-preview"
  }
}
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

**Azure-Specific Extensions:**
```json
{
  "ok": true,
  "code": "OK", 
  "ms": 167.8,
  "result": {
    "text": "The answer is...",
    "model": "gpt-4",
    "usage": {...}
  },
  "extensions": {
    "azure:system_fingerprint": "fp_1234567890",
    "azure:content_filter_results": {
      "hate": {"filtered": false, "severity": "safe"},
      "self_harm": {"filtered": false, "severity": "safe"}
    }
  }
}
```

#### Error Mapping
| Azure OpenAI Error | CORPUS Error Code | Retryable | Notes |
|-------------------|------------------|-----------|-------|
| `429 Too Many Requests` | `RESOURCE_EXHAUSTED` | Yes | Add `retry_after_ms` |
| `401 Unauthorized` | `AUTH_ERROR` | No | Direct mapping |
| `403 Forbidden` | `AUTH_ERROR` | No | Permission issue |
| `400 Bad Request` | `BAD_REQUEST` | No | Direct mapping |
| `503 Service Unavailable` | `UNAVAILABLE` | Yes | Azure service down |
| `504 Gateway Timeout` | `DEADLINE_EXCEEDED` | Conditional | Check deadline |
| `429 Resource Exhausted` | `RESOURCE_EXHAUSTED` | Yes | Quota exceeded |

---

## 5. 🔍 Vector Protocol Migrations

### 5.1 Pinecone to CORPUS

#### Request Mapping
| Pinecone Field | CORPUS Field | Type | Transformation Required |
|----------------|--------------|------|-------------------------|
| `vector` | `args.vector` | array[float] | Direct mapping |
| `topK` | `args.top_k` | integer | Rename: `topK` → `top_k` |
| `namespace` | `args.namespace` | string | Direct mapping |
| `filter` | `args.filter` | object | Convert Pinecone filter syntax |
| `includeMetadata` | `args.include_metadata` | boolean | Rename field |
| `includeValues` | `args.include_vectors` | boolean | Rename: `includeValues` → `include_vectors` |
| `sparseVector` | `args.sparse_vector` | object | Direct mapping |
| `id` (for upsert) | `vectors[].id` | string | Restructure for batch |

**Filter Syntax Conversion:**
```python
# Pinecone filter → CORPUS filter
pinecone_filter = {
    "category": {"$eq": "books"},
    "price": {"$gte": 20, "$lt": 100},
    "tags": {"$in": ["fiction", "scifi"]}
}

# CORPUS filter
corpus_filter = {
    "category": "books",
    "price__gte": 20,
    "price__lt": 100,
    "tags__in": ["fiction", "scifi"]
}

# OR using extensions for complex operators
corpus_filter = {
    "category": "books",
    "price": {"gte": 20, "lt": 100},
    "tags": ["fiction", "scifi"]
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
      "year__gte": 2020
    },
    "include_metadata": true,
    "include_vectors": false
  }
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

**Upsert/Delete Operations:**
```json
// Pinecone upsert
POST /vectors/upsert
{
  "vectors": [
    {
      "id": "vec-1",
      "values": [0.1, 0.2, 0.3],
      "metadata": {"category": "book"},
      "sparseValues": {"indices": [1, 3], "values": [0.5, 0.7]}
    }
  ],
  "namespace": "documents"
}

// CORPUS upsert
{
  "op": "vector.upsert",
  "ctx": {...},
  "args": {
    "vectors": [
      {
        "id": "vec-1",
        "vector": [0.1, 0.2, 0.3],
        "metadata": {"category": "book"},
        "sparse_vector": {"indices": [1, 3], "values": [0.5, 0.7]},
        "namespace": "documents"
      }
    ],
    "namespace": "documents"
  }
}
```

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

---

### 5.2 Qdrant to CORPUS

#### Request Mapping
| Qdrant Field | CORPUS Field | Type | Transformation Required |
|--------------|--------------|------|-------------------------|
| `vector` | `args.vector` | array/dict | Qdrant supports named vectors |
| `limit` | `args.top_k` | integer | Rename: `limit` → `top_k` |
| `with_payload` | `args.include_metadata` | boolean/array | Rename and handle array case |
| `with_vector` | `args.include_vectors` | boolean | Rename field |
| `filter` | `args.filter` | object | Convert Qdrant filter syntax |
| `score_threshold` | `args.score_threshold` | float | Direct mapping |
| `offset` | `args.offset` | integer | Direct mapping |
| `collection_name` | `args.namespace` | string | Rename: collection → namespace |

**Filter Syntax Conversion:**
```python
# Qdrant filter → CORPUS filter
qdrant_filter = {
    "must": [
        {"key": "category", "match": {"value": "books"}},
        {"key": "price", "range": {"gte": 20}}
    ],
    "should": [
        {"key": "tags", "match": {"any": ["fiction", "scifi"]}}
    ]
}

# CORPUS filter (simplified)
corpus_filter = {
    "category": "books",
    "price__gte": 20
}
# Complex filters via extensions
corpus_filter = {
    "category": "books",
    "price": {"gte": 20}
}
```

**Named Vectors Handling:**
```json
// Qdrant with named vectors
{
  "vector": {
    "name": "text_embedding",
    "vector": [0.1, 0.2, 0.3]
  }
}

// CORPUS equivalent
{
  "vector": [0.1, 0.2, 0.3],
  "extensions": {
    "qdrant:vector_name": "text_embedding"
  }
}
```

**Wire Envelope Example:**
```json
// Qdrant Search Request
POST /collections/documents/points/search
{
  "vector": [0.1, 0.2, 0.3],
  "limit": 10,
  "with_payload": true,
  "with_vector": false,
  "filter": {
    "must": [
      {"key": "category", "match": {"value": "technology"}}
    ]
  }
}

// CORPUS Equivalent
POST /v1/operations
{
  "op": "vector.query",
  "ctx": {
    "request_id": "vec-456",
    "deadline_ms": 1730312345000
  },
  "args": {
    "vector": [0.1, 0.2, 0.3],
    "top_k": 10,
    "namespace": "documents",
    "filter": {
      "category": "technology"
    },
    "include_metadata": true,
    "include_vectors": false
  }
}
```

#### Response Mapping
| Qdrant Response Field | CORPUS Response Field | Transformation |
|----------------------|----------------------|----------------|
| `result[]` | `result.matches[]` | Array of matches |
| `result[].id` | `matches[].vector.id` | Qdrant uses integer or UUID |
| `result[].score` | `matches[].score` | Direct mapping |
| `result[].version` | `matches[].vector.version` | Store in vector metadata |
| `result[].payload` | `matches[].vector.metadata` | Rename: payload → metadata |
| `result[].vector` | `matches[].vector.vector` | Direct mapping |
| `time` | `result.query_time_ms` | Convert to milliseconds |

**Batch Operations:**
```json
// Qdrant batch upsert
POST /collections/{collection}/points
{
  "points": [
    {
      "id": 1,
      "vector": [0.1, 0.2, 0.3],
      "payload": {"category": "book"}
    }
  ]
}

// CORPUS upsert
{
  "op": "vector.upsert",
  "ctx": {...},
  "args": {
    "vectors": [
      {
        "id": "1",  // Convert to string
        "vector": [0.1, 0.2, 0.3],
        "metadata": {"category": "book"},
        "namespace": "documents"
      }
    ],
    "namespace": "documents"
  }
}
```

#### Error Mapping
| Qdrant Error | CORPUS Error Code | Retryable | Notes |
|--------------|------------------|-----------|-------|
| `429 Too Many Requests` | `RESOURCE_EXHAUSTED` | Yes | Rate limit |
| `400 Bad Request` | `BAD_REQUEST` | No | Invalid parameters |
| `401 Unauthorized` | `AUTH_ERROR` | No | Authentication |
| `404 Not Found` | `NAMESPACE_NOT_FOUND` | No | Collection doesn't exist |
| `500 Internal Server Error` | `UNAVAILABLE` | Yes | Qdrant internal error |
| `503 Service Unavailable` | `UNAVAILABLE` | Yes | Service down |

---

### 5.3 Weaviate to CORPUS

#### Request Mapping
| Weaviate Field | CORPUS Field | Type | Transformation Required |
|----------------|--------------|------|-------------------------|
| `vector` | `args.vector` | array | Direct mapping |
| `limit` | `args.top_k` | integer | Rename: `limit` → `top_k` |
| `nearVector` | `args.vector` | object | Extract vector from object |
| `nearText` | `args.text` | object | Text-based search |
| `where` | `args.filter` | object | Convert GraphQL-like filter |
| `additional` | Selection of fields | object | Map to include flags |
| `className` | `args.namespace` | string | Rename: class → namespace |
| `autocut` | `args.autocut` | integer | Direct mapping |

**Filter Syntax Conversion:**
```python
# Weaviate where filter → CORPUS filter
weaviate_filter = {
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

# CORPUS filter
corpus_filter = {
    "category": "books",
    "price__gte": 20
}

# Complex operator mapping
operator_map = {
    "Equal": "eq",
    "NotEqual": "neq",
    "GreaterThan": "gt",
    "GreaterThanEqual": "gte",
    "LessThan": "lt",
    "LessThanEqual": "lte",
    "Like": "like",
    "ContainsAny": "in"
}
```

**NearText Handling:**
```json
// Weaviate nearText search
{
  "nearText": {
    "concepts": ["quantum physics"],
    "certainty": 0.8,
    "moveAwayFrom": {
      "concepts": ["classical physics"],
      "force": 0.5
    }
  }
}

// CORPUS equivalent with extensions
{
  "text": "quantum physics",
  "extensions": {
    "weaviate:certainty": 0.8,
    "weaviate:move_away_from": {
      "concepts": ["classical physics"],
      "force": 0.5
    }
  }
}
```

**Wire Envelope Example:**
```json
// Weaviate GraphQL Query
{
  "query": """
  {
    Get {
      Article(
        nearVector: {
          vector: [0.1, 0.2, 0.3]
        }
        limit: 10
        where: {
          path: ["category"]
          operator: Equal
          valueString: "technology"
        }
      ) {
        _additional {
          id
          vector
        }
        title
        content
      }
    }
  }
  """
}

// CORPUS Equivalent (simplified)
POST /v1/operations
{
  "op": "vector.query",
  "ctx": {...},
  "args": {
    "vector": [0.1, 0.2, 0.3],
    "top_k": 10,
    "namespace": "Article",
    "filter": {
      "category": "technology"
    },
    "include_metadata": true,
    "include_vectors": true
  }
}
```

#### Response Mapping
| Weaviate Response Field | CORPUS Response Field | Transformation |
|------------------------|----------------------|----------------|
| `data.Get.{Class}[]` | `result.matches[]` | Extract from nested structure |
| `_additional.id` | `matches[].vector.id` | Direct mapping |
| `_additional.vector` | `matches[].vector.vector` | Direct mapping |
| `_additional.certainty` | `matches[].score` | Convert certainty to score |
| `_additional.distance` | `matches[].distance` | Direct mapping |
| Properties (title, etc.) | `matches[].vector.metadata` | Flatten into metadata |
| `errors` | Error envelope | Convert to CORPUS error |

**Certainty to Score Conversion:**
```python
# Weaviate certainty [0, 1] → CORPUS score [0, 1]
def certainty_to_score(certainty: float) -> float:
    return certainty  # Direct mapping for cosine similarity
    
# For distance metrics, need conversion
def distance_to_score(distance: float, metric: str) -> float:
    if metric == "cosine":
        return 1 - distance
    elif metric == "l2-squared":
        return 1 / (1 + distance)  # Normalize
    else:
        return distance  # Use as-is
```

#### Error Mapping
| Weaviate Error | CORPUS Error Code | Retryable | Notes |
|----------------|------------------|-----------|-------|
| `429 Too Many Requests` | `RESOURCE_EXHAUSTED` | Yes | Rate limit |
| `400 Bad Request` | `BAD_REQUEST` | No | Invalid GraphQL |
| `401 Unauthorized` | `AUTH_ERROR` | No | Authentication |
| `404 Not Found` | `NAMESPACE_NOT_FOUND` | No | Class doesn't exist |
| `422 Unprocessable Entity` | `DIMENSION_MISMATCH` | No | Vector dimension issue |
| `500 Internal Server Error` | `UNAVAILABLE` | Yes | Weaviate internal error |
| `503 Service Unavailable` | `UNAVAILABLE` | Yes | Service down |

---

### 5.4 Milvus to CORPUS

#### Request Mapping
| Milvus Field | CORPUS Field | Type | Transformation Required |
|--------------|--------------|------|-------------------------|
| `vector` | `args.vector` | array | Direct mapping |
| `limit` | `args.top_k` | integer | Rename: `limit` → `top_k` |
| `output_fields` | Selection | array | Map to include flags |
| `filter` | `args.filter` | string | Convert boolean expression |
| `expr` | `args.filter` | string | Boolean expression |
| `collection_name` | `args.namespace` | string | Rename: collection → namespace |
| `anns_field` | `args.field` | string | Vector field name |
| `metric_type` | `args.metric` | string | Distance metric |
| `params` | `args.search_params` | object | Search parameters |

**Filter Expression Conversion:**
```python
# Milvus boolean expression → CORPUS filter
milvius_expr = 'category == "books" and price >= 20 and tags in ["fiction", "scifi"]'

# Parse and convert to CORPUS filter
# This requires expression parsing
corpus_filter = {
    "category": "books",
    "price__gte": 20,
    "tags__in": ["fiction", "scifi"]
}

# Complex expressions remain as string
corpus_filter = milvus_expr  # Keep as string for complex cases
```

**Search Parameters:**
```json
// Milvus search params
{
  "params": {
    "metric_type": "IP",  # or "L2", "COSINE"
    "params": {
      "nprobe": 10,
      "ef": 64
    }
  }
}

// CORPUS with extensions
{
  "metric": "inner_product",  # or "l2", "cosine"
  "extensions": {
    "milvus:nprobe": 10,
    "milvus:ef": 64
  }
}
```

**Wire Envelope Example:**
```json
// Milvus Search Request
POST /v1/vector/search
{
  "collection_name": "documents",
  "vector": [0.1, 0.2, 0.3],
  "limit": 10,
  "output_fields": ["title", "category"],
  "expr": "category == 'technology' and year >= 2020",
  "search_params": {
    "metric_type": "COSINE",
    "params": {"nprobe": 10}
  }
}

// CORPUS Equivalent
POST /v1/operations
{
  "op": "vector.query",
  "ctx": {...},
  "args": {
    "vector": [0.1, 0.2, 0.3],
    "top_k": 10,
    "namespace": "documents",
    "filter": "category == 'technology' and year >= 2020",
    "include_metadata": true,
    "metric": "cosine"
  },
  "extensions": {
    "milvus:nprobe": 10
  }
}
```

#### Response Mapping
| Milvus Response Field | CORPUS Response Field | Transformation |
|----------------------|----------------------|----------------|
| `results[]` | `result.matches[]` | Array of matches |
| `results[].id` | `matches[].vector.id` | Direct mapping |
| `results[].score` | `matches[].score` | Direct mapping |
| `results[].distance` | `matches[].distance` | Direct mapping |
| Output fields | `matches[].vector.metadata` | Flatten into metadata |
| `status.error_code` | Error code | Map to CORPUS error |
| `status.reason` | Error message | Include in details |

**ID Field Handling:**
```python
# Milvus IDs can be integers or strings
milvus_id = result["id"]
if isinstance(milvus_id, int):
    corpus_id = str(milvus_id)
else:
    corpus_id = milvus_id
```

#### Error Mapping
| Milvus Error | CORPUS Error Code | Retryable | Notes |
|--------------|------------------|-----------|-------|
| `429 Too Many Requests` | `RESOURCE_EXHAUSTED` | Yes | Rate limit |
| `400 Bad Request` | `BAD_REQUEST` | No | Invalid parameters |
| `401 Unauthorized` | `AUTH_ERROR` | No | Authentication |
| `404 Not Found` | `NAMESPACE_NOT_FOUND` | No | Collection doesn't exist |
| `500 Internal Server Error` | `UNAVAILABLE` | Yes | Milvus internal error |
| `503 Service Unavailable` | `UNAVAILABLE` | Yes | Service down |
| `Code 1` (Success) | `OK` | N/A | Success code |

---

### 5.5 Chroma to CORPUS

#### Request Mapping
| Chroma Field | CORPUS Field | Type | Transformation Required |
|--------------|--------------|------|-------------------------|
| `query_embeddings` | `args.vector` | array | Can be batch of vectors |
| `n_results` | `args.top_k` | integer | Rename: `n_results` → `top_k` |
| `where` | `args.filter` | object | Direct mapping |
| `where_document` | `args.document_filter` | object | Filter on document content |
| `include` | Selection | array/object | Map to include flags |
| `collection_name` | `args.namespace` | string | Direct mapping |
| `query_texts` | `args.text` | array | Text-based search |

**Filter Syntax:**
```json
// Chroma where filter
{
  "where": {
    "category": {"$eq": "books"},
    "price": {"$gte": 20}
  },
  "where_document": {
    "$contains": "quantum"
  }
}

// CORPUS equivalent
{
  "filter": {
    "category": "books",
    "price__gte": 20
  },
  "extensions": {
    "chroma:document_contains": "quantum"
  }
}
```

**Include Field Mapping:**
```json
// Chroma include
{
  "include": ["metadatas", "documents", "distances"]
}

// CORPUS equivalent
{
  "include_metadata": true,
  "include_documents": true,
  "include_distances": true
}
```

**Wire Envelope Example:**
```json
// Chroma Query Request
POST /api/v1/collections/documents/query
{
  "query_embeddings": [[0.1, 0.2, 0.3]],
  "n_results": 10,
  "where": {
    "category": {"$eq": "technology"}
  },
  "include": ["metadatas", "distances"]
}

// CORPUS Equivalent
POST /v1/operations
{
  "op": "vector.query",
  "ctx": {...},
  "args": {
    "vector": [0.1, 0.2, 0.3],
    "top_k": 10,
    "namespace": "documents",
    "filter": {
      "category": "technology"
    },
    "include_metadata": true,
    "include_distances": true
  }
}
```

#### Response Mapping
| Chroma Response Field | CORPUS Response Field | Transformation |
|----------------------|----------------------|----------------|
| `ids[0]` | `matches[].vector.id` | Batch results |
| `distances[0]` | `matches[].distance` | Direct mapping |
| `metadatas[0]` | `matches[].vector.metadata` | Direct mapping |
| `documents[0]` | `matches[].vector.document` | Store in metadata |
| `embeddings[0]` | `matches[].vector.vector` | Direct mapping |
| `uris[0]` | `matches[].vector.uri` | Store in metadata |

**Batch Results Handling:**
```python
# Chroma returns batch results
chroma_response = {
    "ids": [["id1", "id2"], ["id3", "id4"]],  # Per query
    "distances": [[0.1, 0.2], [0.3, 0.4]],
    "metadatas": [[{"cat": "a"}, {"cat": "b"}], [{"cat": "c"}, {"cat": "d"}]]
}

# CORPUS format for first query
corpus_matches = [
    {
        "vector": {
            "id": "id1",
            "vector": None,  # Not included in this example
            "metadata": {"cat": "a"}
        },
        "distance": 0.1,
        "score": 1 - 0.1  # Convert distance to score
    },
    # ... more matches
]
```

#### Error Mapping
| Chroma Error | CORPUS Error Code | Retryable | Notes |
|--------------|------------------|-----------|-------|
| `429 Too Many Requests` | `RESOURCE_EXHAUSTED` | Yes | Rate limit |
| `400 Bad Request` | `BAD_REQUEST` | No | Invalid parameters |
| `401 Unauthorized` | `AUTH_ERROR` | No | Authentication |
| `404 Not Found` | `NAMESPACE_NOT_FOUND` | No | Collection doesn't exist |
| `422 Unprocessable Entity` | `DIMENSION_MISMATCH` | No | Embedding dimension issue |
| `500 Internal Server Error` | `UNAVAILABLE` | Yes | Chroma internal error |
| `503 Service Unavailable` | `UNAVAILABLE` | Yes | Service down |

---

## 6. 📊 Embedding Protocol Migrations

### 6.1 OpenAI Embeddings to CORPUS

#### Request Mapping
| OpenAI Field | CORPUS Field | Type | Transformation Required |
|--------------|--------------|------|-------------------------|
| `model` | `args.model` | string | Map: `text-embedding-ada-002` → `ada-002` |
| `input` (single) | `args.text` | string | Single embedding |
| `input` (batch) | `args.texts` | array | Batch embedding |
| `encoding_format` | `args.encoding_format` | string | Via extensions |
| `user` | `ctx.attrs.user` | string | Move to context |
| `dimensions` | `args.dimensions` | integer | Reduce dimensions |

**Model Name Normalization:**
```python
# OpenAI model → CORPUS model name
model_mapping = {
    "text-embedding-ada-002": "ada-002",
    "text-embedding-3-small": "text-embedding-3-small",
    "text-embedding-3-large": "text-embedding-3-large",
    # Add other mappings
}

openai_model = request.get("model")
corpus_model = model_mapping.get(openai_model, openai_model)
```

**Single vs Batch Operations:**
```json
// OpenAI single embedding
POST /v1/embeddings
{
  "model": "text-embedding-ada-002",
  "input": "The quick brown fox",
  "encoding_format": "float"
}

// CORPUS equivalent
POST /v1/operations
{
  "op": "embedding.embed",
  "ctx": {...},
  "args": {
    "text": "The quick brown fox",
    "model": "ada-002"
  },
  "extensions": {
    "openai:encoding_format": "float"
  }
}

// OpenAI batch embedding
{
  "model": "text-embedding-ada-002",
  "input": ["Text 1", "Text 2", "Text 3"]
}

// CORPUS batch
{
  "op": "embedding.embed_batch",
  "ctx": {...},
  "args": {
    "texts": ["Text 1", "Text 2", "Text 3"],
    "model": "ada-002"
  }
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

**Batch Response Structure:**
```json
// OpenAI batch response
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, 0.3],
      "index": 0
    },
    {
      "object": "embedding", 
      "embedding": [0.4, 0.5, 0.6],
      "index": 1
    }
  ],
  "model": "text-embedding-ada-002",
  "usage": {"prompt_tokens": 20, "total_tokens": 20}
}

// CORPUS batch response
{
  "ok": true,
  "code": "OK",
  "ms": 45.2,
  "result": {
    "embeddings": [
      {
        "vector": [0.1, 0.2, 0.3],
        "text": "Text 1",
        "model": "ada-002",
        "dimensions": 1536
      },
      {
        "vector": [0.4, 0.5, 0.6],
        "text": "Text 2", 
        "model": "ada-002",
        "dimensions": 1536
      }
    ],
    "model": "ada-002",
    "total_texts": 2,
    "total_tokens": 20
  }
}
```

#### Error Mapping
| OpenAI Error | CORPUS Error Code | Retryable | Notes |
|--------------|------------------|-----------|-------|
| `RateLimitError` | `RESOURCE_EXHAUSTED` | Yes | Rate limit |
| `AuthenticationError` | `AUTH_ERROR` | No | Authentication |
| `BadRequestError` | `BAD_REQUEST` | No | Invalid input |
| `APIConnectionError` | `TRANSIENT_NETWORK` | Yes | Network issues |
| `APIError` | `UNAVAILABLE` | Yes | Provider issue |
| `TimeoutError` | `DEADLINE_EXCEEDED` | Conditional | Check deadline |

---

### 6.2 Cohere Embed to CORPUS

#### Request Mapping
| Cohere Field | CORPUS Field | Type | Transformation Required |
|--------------|--------------|------|-------------------------|
| `model` | `args.model` | string | Map: `embed-english-v3.0` → `cohere-v3-en` |
| `texts` | `args.texts` | array | Always batch in Cohere |
| `input_type` | `args.input_type` | string | Via extensions |
| `embedding_types` | `args.embedding_types` | array | Via extensions |
| `truncate` | `args.truncate` | string | Map to boolean |
| `compress` | `args.compress` | boolean | Via extensions |

**Model and Input Type Mapping:**
```python
# Cohere model names
cohere_models = {
    "embed-english-v3.0": "cohere-v3-en",
    "embed-multilingual-v3.0": "cohere-v3-multi",
    "embed-english-light-v3.0": "cohere-v3-en-light",
    # Add other models
}

# Input types
input_types = {
    "search_document": "document",
    "search_query": "query",
    "classification": "classification",
    "clustering": "clustering"
}
```

**Wire Envelope Example:**
```json
// Cohere Embed Request
POST /v1/embed
{
  "model": "embed-english-v3.0",
  "texts": ["Hello world", "AI is amazing"],
  "input_type": "search_document",
  "embedding_types": ["float"],
  "truncate": "END"
}

// CORPUS Equivalent
POST /v1/operations
{
  "op": "embedding.embed_batch",
  "ctx": {...},
  "args": {
    "texts": ["Hello world", "AI is amazing"],
    "model": "cohere-v3-en",
    "truncate": true
  },
  "extensions": {
    "cohere:input_type": "search_document",
    "cohere:embedding_types": ["float"],
    "cohere:truncate_mode": "END"
  }
}
```

#### Response Mapping
| Cohere Response Field | CORPUS Response Field | Transformation |
|----------------------|----------------------|----------------|
| `embeddings.float` | `result.embeddings[].vector` | Float embeddings |
| `embeddings.int8` | Not directly mapped | Store in extensions |
| `embeddings.ubinary` | Not directly mapped | Store in extensions |
| `id` | `ctx.request_id` | Request correlation |
| `texts` | `result.embeddings[].text` | Original texts |
| `meta` | `result.meta` | Via extensions |

**Multiple Embedding Types:**
```json
// Cohere response with multiple types
{
  "id": "embed-123",
  "texts": ["Hello world"],
  "embeddings": {
    "float": [[0.1, 0.2, 0.3]],
    "int8": [[1, 2, 3]],
    "ubinary": [[1, 0, 1]]
  },
  "meta": {
    "api_version": {"version": "1"}
  }
}

// CORPUS response (primary type only)
{
  "ok": true,
  "code": "OK",
  "ms": 32.7,
  "result": {
    "embeddings": [
      {
        "vector": [0.1, 0.2, 0.3],
        "text": "Hello world",
        "model": "cohere-v3-en",
        "dimensions": 1024
      }
    ],
    "model": "cohere-v3-en",
    "total_texts": 1
  },
  "extensions": {
    "cohere:int8_embeddings": [[1, 2, 3]],
    "cohere:ubinary_embeddings": [[1, 0, 1]],
    "cohere:api_version": "1"
  }
}
```

#### Error Mapping
| Cohere Error | CORPUS Error Code | Retryable | Notes |
|--------------|------------------|-----------|-------|
| `rate_limit_error` | `RESOURCE_EXHAUSTED` | Yes | Rate limit |
| `authentication_error` | `AUTH_ERROR` | No | Authentication |
| `invalid_request_error` | `BAD_REQUEST` | No | Invalid input |
| `internal_server_error` | `UNAVAILABLE` | Yes | Provider issue |
| `service_unavailable` | `UNAVAILABLE` | Yes | Service down |
| `timeout_error` | `DEADLINE_EXCEEDED` | Conditional | Check deadline |

---

### 6.3 HuggingFace to CORPUS

#### Request Mapping
| HuggingFace Field | CORPUS Field | Type | Transformation Required |
|-------------------|--------------|------|-------------------------|
| `model` | `args.model` | string | Map: `sentence-transformers/all-MiniLM-L6-v2` → `miniLM-L6-v2` |
| `inputs` | `args.texts` | array/string | Can be single or batch |
| `normalize` | `args.normalize` | boolean | Direct mapping |
| `truncate` | `args.truncate` | boolean | Direct mapping |
| `options` | `args.options` | object | Via extensions |
| `parameters` | `args.parameters` | object | Model parameters |

**Model Name Normalization:**
```python
# Common HuggingFace models
model_aliases = {
    "sentence-transformers/all-MiniLM-L6-v2": "miniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2": "mpnet-base-v2",
    "intfloat/e5-large-v2": "e5-large-v2",
    "BAAI/bge-large-en-v1.5": "bge-large-en-v1.5",
    # Add more as needed
}

hf_model = request.get("model")
corpus_model = model_aliases.get(hf_model, hf_model.split("/")[-1])
```

**Wire Envelope Example:**
```json
// HuggingFace Inference Request
POST /models/sentence-transformers/all-MiniLM-L6-v2
{
  "inputs": "The quick brown fox jumps over the lazy dog",
  "options": {
    "wait_for_model": true,
    "use_cache": false
  },
  "parameters": {
    "normalize": true
  }
}

// CORPUS Equivalent
POST /v1/operations
{
  "op": "embedding.embed",
  "ctx": {...},
  "args": {
    "text": "The quick brown fox jumps over the lazy dog",
    "model": "miniLM-L6-v2",
    "normalize": true
  },
  "extensions": {
    "huggingface:wait_for_model": true,
    "huggingface:use_cache": false
  }
}
```

#### Response Mapping
| HuggingFace Response | CORPUS Response Field | Transformation |
|---------------------|----------------------|----------------|
| Single array | `result.embedding.vector` | Single embedding |
| Array of arrays | `result.embeddings[].vector` | Batch embeddings |
| Inference time | `result.processing_time_ms` | Convert to ms |
| Model info | `result.model_details` | Via extensions |
| Error array | `result.failed_texts` | Partial failures |

**Response Formats:**
```json
// HuggingFace single embedding
[[0.1, 0.2, 0.3, 0.4, 0.5]]

// CORPUS single embedding
{
  "ok": true,
  "code": "OK",
  "ms": 28.3,
  "result": {
    "embedding": {
      "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
      "text": "The quick brown fox...",
      "model": "miniLM-L6-v2",
      "dimensions": 384
    },
    "model": "miniLM-L6-v2"
  }
}

// HuggingFace batch embeddings
[
  [0.1, 0.2, 0.3],
  [0.4, 0.5, 0.6],
  [0.7, 0.8, 0.9]
]

// CORPUS batch embeddings
{
  "ok": true,
  "code": "OK",
  "ms": 56.7,
  "result": {
    "embeddings": [
      {
        "vector": [0.1, 0.2, 0.3],
        "text": "Text 1",
        "model": "miniLM-L6-v2",
        "dimensions": 384
      },
      // ... more embeddings
    ],
    "model": "miniLM-L6-v2",
    "total_texts": 3
  }
}
```

#### Error Mapping
| HuggingFace Error | CORPUS Error Code | Retryable | Notes |
|-------------------|------------------|-----------|-------|
| `429 Too Many Requests` | `RESOURCE_EXHAUSTED` | Yes | Rate limit |
| `400 Bad Request` | `BAD_REQUEST` | No | Invalid input |
| `401 Unauthorized` | `AUTH_ERROR` | No | Authentication |
| `503 Service Unavailable` | `UNAVAILABLE` | Yes | Model loading |
| `504 Gateway Timeout` | `DEADLINE_EXCEEDED` | Conditional | Check deadline |
| `422 Unprocessable Entity` | `BAD_REQUEST` | No | Input too long |

---

### 6.4 Google Vertex AI to CORPUS

#### Request Mapping
| Vertex AI Field | CORPUS Field | Type | Transformation Required |
|-----------------|--------------|------|-------------------------|
| `instances[]` | `args.texts` | array | Extract from instances |
| `parameters` | `args.parameters` | object | Model parameters |
| `endpoint` | `args.model` | string | Extract model from endpoint |
| `task_type` | `args.task_type` | string | Via extensions |
| `title` | `args.title` | string | Via extensions |

**Endpoint to Model Mapping:**
```python
# Vertex AI endpoint pattern
# projects/{project}/locations/{location}/publishers/google/models/{model}
import re

def extract_model_from_endpoint(endpoint: str) -> str:
    pattern = r"/models/([^/]+)$"
    match = re.search(pattern, endpoint)
    if match:
        return f"google-{match.group(1)}"
    return endpoint
```

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

#### Response Mapping
| Vertex AI Response Field | CORPUS Response Field | Transformation |
|-------------------------|----------------------|----------------|
| `predictions[]` | `result.embeddings[]` | Array of predictions |
| `predictions[].embeddings` | `embeddings[].vector` | Extract embeddings |
| `predictions[].statistics` | `embeddings[].statistics` | Via extensions |
| `metadata` | `result.metadata` | Via extensions |
| `model_version_id` | `result.model_version` | Via extensions |

**Response Structure:**
```json
// Vertex AI response
{
  "predictions": [
    {
      "embeddings": {
        "values": [0.1, 0.2, 0.3],
        "statistics": {
          "token_count": 9,
          "truncated": false
        }
      }
    }
  ],
  "metadata": {
    "billableCharacterCount": 27,
    "model_version_id": "001"
  }
}

// CORPUS response
{
  "ok": true,
  "code": "OK",
  "ms": 67.8,
  "result": {
    "embeddings": [
      {
        "vector": [0.1, 0.2, 0.3],
        "text": "The quick brown fox",
        "model": "google-textembedding-gecko",
        "dimensions": 768
      }
    ],
    "model": "google-textembedding-gecko",
    "total_texts": 1,
    "total_tokens": 9
  },
  "extensions": {
    "vertex:model_version": "001",
    "vertex:billable_character_count": 27
  }
}
```

#### Error Mapping
| Vertex AI Error | CORPUS Error Code | Retryable | Notes |
|-----------------|------------------|-----------|-------|
| `RESOURCE_EXHAUSTED` | `RESOURCE_EXHAUSTED` | Yes | Quota exceeded |
| `PERMISSION_DENIED` | `AUTH_ERROR` | No | Authentication |
| `INVALID_ARGUMENT` | `BAD_REQUEST` | No | Invalid input |
| `NOT_FOUND` | `MODEL_NOT_AVAILABLE` | No | Model not found |
| `UNAVAILABLE` | `UNAVAILABLE` | Yes | Service down |
| `DEADLINE_EXCEEDED` | `DEADLINE_EXCEEDED` | Conditional | Check deadline |

---

### 6.5 AWS Bedrock to CORPUS

#### Request Mapping
| Bedrock Field | CORPUS Field | Type | Transformation Required |
|---------------|--------------|------|-------------------------|
| `modelId` | `args.model` | string | Map: `amazon.titan-embed-text-v1` → `titan-embed-text-v1` |
| `inputText` | `args.text` | string | Single text |
| `inputTexts` | `args.texts` | array | Batch texts |
| `dimensions` | `args.dimensions` | integer | Output dimensions |
| `normalize` | `args.normalize` | boolean | Direct mapping |
| `embeddingTypes` | `args.embedding_types` | array | Via extensions |

**Model ID Mapping:**
```python
# Bedrock model IDs
bedrock_models = {
    "amazon.titan-embed-text-v1": "titan-embed-text-v1",
    "amazon.titan-embed-text-v2:0": "titan-embed-text-v2",
    "cohere.embed-english-v3": "cohere-v3-en",
    "cohere.embed-multilingual-v3": "cohere-v3-multi"
}
```

**Request Body Structure:**
```json
// Bedrock request (Titan model)
{
  "inputText": "The quick brown fox",
  "dimensions": 256,
  "normalize": true
}

// CORPUS equivalent
{
  "op": "embedding.embed",
  "ctx": {...},
  "args": {
    "text": "The quick brown fox",
    "model": "titan-embed-text-v1",
    "dimensions": 256,
    "normalize": true
  }
}

// Bedrock request (Cohere model)
{
  "texts": ["Hello", "World"],
  "input_type": "search_document",
  "truncate": "END"
}

// CORPUS equivalent
{
  "op": "embedding.embed_batch",
  "ctx": {...},
  "args": {
    "texts": ["Hello", "World"],
    "model": "cohere-v3-en",
    "truncate": true
  },
  "extensions": {
    "cohere:input_type": "search_document",
    "cohere:truncate_mode": "END"
  }
}
```

#### Response Mapping
| Bedrock Response Field | CORPUS Response Field | Transformation |
|-----------------------|----------------------|----------------|
| `embedding` | `result.embedding.vector` | Titan model |
| `embeddings[]` | `result.embeddings[].vector` | Cohere model |
| `inputTextTokenCount` | `result.total_tokens` | Token count |
| `message` | Error message | For errors |
| `type` | Error type | For errors |

**Model-Specific Responses:**
```json
// Titan response
{
  "embedding": [0.1, 0.2, 0.3],
  "inputTextTokenCount": 9
}

// CORPUS response for Titan
{
  "ok": true,
  "code": "OK",
  "ms": 89.1,
  "result": {
    "embedding": {
      "vector": [0.1, 0.2, 0.3],
      "text": "The quick brown fox",
      "model": "titan-embed-text-v1",
      "dimensions": 1536
    },
    "model": "titan-embed-text-v1",
    "total_tokens": 9
  }
}

// Cohere via Bedrock response
{
  "embeddings": [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6]
  ],
  "id": "embed-123",
  "texts": ["Hello", "World"]
}

// CORPUS response for Cohere via Bedrock
{
  "ok": true,
  "code": "OK",
  "ms": 45.6,
  "result": {
    "embeddings": [
      {
        "vector": [0.1, 0.2, 0.3],
        "text": "Hello",
        "model": "cohere-v3-en",
        "dimensions": 1024
      },
      {
        "vector": [0.4, 0.5, 0.6],
        "text": "World",
        "model": "cohere-v3-en",
        "dimensions": 1024
      }
    ],
    "model": "cohere-v3-en",
    "total_texts": 2
  }
}
```

#### Error Mapping
| Bedrock Error | CORPUS Error Code | Retryable | Notes |
|---------------|------------------|-----------|-------|
| `ThrottlingException` | `RESOURCE_EXHAUSTED` | Yes | AWS throttling |
| `AccessDeniedException` | `AUTH_ERROR` | No | Permissions |
| `ValidationException` | `BAD_REQUEST` | No | Invalid request |
| `ModelNotReadyException` | `MODEL_NOT_AVAILABLE` | Yes | Model loading |
| `ServiceQuotaExceededException` | `RESOURCE_EXHAUSTED` | No | Quota exceeded |
| `InternalServerException` | `UNAVAILABLE` | Yes | AWS internal error |

---

## 7. 🌐 Graph Protocol Migrations

*Note: Due to context limits, I'll provide the first graph migration in detail and summarize the others. Press "Continue" to get the complete Graph Protocol section.*

### 7.1 Neo4j to CORPUS

#### Request Mapping
| Neo4j Field | CORPUS Field | Type | Transformation Required |
|-------------|--------------|------|-------------------------|
| `cypher` | `args.text` | string | Direct mapping |
| `params` | `args.params` | object | Direct mapping |
| `database` | `args.namespace` | string | Rename: database → namespace |
| `resultDataContents` | `args.result_format` | array | Via extensions |
| `includeStats` | `args.include_stats` | boolean | Via extensions |

**Wire Envelope Example:**
```json
// Neo4j Cypher Request
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
POST /v1/operations
{
  "op": "graph.query",
  "ctx": {
    "request_id": "graph-123",
    "deadline_ms": 1730312345000
  },
  "args": {
    "dialect": "cypher",
    "text": "MATCH (n:Person) WHERE n.name = $name RETURN n",
    "params": {"name": "Alice"},
    "namespace": "neo4j"
  },
  "extensions": {
    "neo4j:result_data_contents": ["row", "graph"]
  }
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

**Transaction Support:**
```json
// Neo4j transaction
{
  "statements": [...],
  "commit": true
}

// CORPUS with transaction hint
{
  "op": "graph.query",
  "ctx": {...},
  "args": {...},
  "extensions": {
    "neo4j:transaction": {"commit": true}
  }
}
```

---

### Quick Summary of Remaining Graph Migrations:

**7.2 Amazon Neptune:**
- **Gremlin queries**: `args.dialect = "gremlin"`
- **SPARQL queries**: `args.dialect = "sparql"`
- **OpenCypher**: `args.dialect = "opencypher"`
- **Batch operations**: Neptune batch → CORPUS `graph.batch`

**7.3 JanusGraph:**
- **Gremlin focus**: Similar to Neptune
- **Schema operations**: `graph.get_schema`
- **Transaction management**: Via extensions
- **Index management**: Via `extensions.janusgraph:index_ops`

**7.4 TigerGraph:**
- **GSQL queries**: `args.dialect = "gsql"`
- **Built-in algorithms**: Via extensions
- **Graph analytics**: `extensions.tigergraph:algorithm`
- **REST endpoints**: Map to CORPUS operations

**7.5 ArangoDB:**
- **AQL queries**: `args.dialect = "aql"`
- **Graph traversals**: Special AQL functions
- **Multi-model**: Document + graph support
- **Foxx services**: Map to custom operations

---

## 8. ⚡ Error Code Mapping

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

## 9. 🔗 Context Propagation

### Standard Context Fields
| Provider Context | CORPUS `ctx` Field | Example |
|-----------------|-------------------|---------|
| `X-Request-ID` | `request_id` | `req-123` |
| `X-Idempotency-Key` | `idempotency_key` | `idem-456` |
| Deadline header | `deadline_ms` | Absolute epoch ms |
| `traceparent` | `traceparent` | W3C Trace Context |
| Tenant header | `tenant` | Hashed in metrics |
| User context | `attrs.user` | User ID or context |

### Provider-Specific Context Mapping
**AWS**: `X-Amzn-Trace-Id` → `traceparent`
**Google Cloud**: `X-Cloud-Trace-Context` → `traceparent`
**Azure**: `traceparent` header (W3C standard)
**Custom headers**: Map to `ctx.attrs`

---

## 10. ✅ Migration Validation Checklist

### Pre-Migration Checks
- [ ] Provider API documentation reviewed
- [ ] CORPUS protocol specification understood
- [ ] Schema validation infrastructure ready
- [ ] Test environment configured

### Wire Transformation Tests
- [ ] Request envelope transformation validated
- [ ] Response envelope transformation validated
- [ ] Error mapping tested with all provider errors
- [ ] Context propagation verified
- [ ] Streaming support validated (if applicable)

### Schema Compliance
- [ ] All requests validate against CORPUS schemas
- [ ] All responses validate against CORPUS schemas
- [ ] Golden test fixtures created and validated
- [ ] Edge cases tested (empty arrays, null values, etc.)

### Performance Validation
- [ ] Latency within acceptable bounds
- [ ] Memory usage monitored
- [ ] Throughput meets requirements
- [ ] Error recovery tested

### Production Readiness
- [ ] Observability integrated (metrics, logs, traces)
- [ ] Security requirements met (tenant isolation, etc.)
- [ ] Documentation updated
- [ ] Rollback plan prepared

---

## 11. 📚 References

### CORPUS Documentation
- **PROTOCOLS.md**: Wire format specification
- **SCHEMA.md**: JSON Schema definitions
- **SPECIFICATION.md**: Architecture and design
- **ERRORS.md**: Error taxonomy and handling
- **METRICS.md**: Observability requirements

### Provider Documentation
- **OpenAI API Reference**: https://platform.openai.com/docs/api-reference
- **Anthropic Messages API**: https://docs.anthropic.com/claude/reference/messages_post
- **Pinecone API**: https://docs.pinecone.io/reference/api/overview
- **Qdrant API**: https://qdrant.tech/documentation/
- **Neo4j HTTP API**: https://neo4j.com/docs/http-api/current/

### Tools & Libraries
- **JSON Schema Validator**: For schema validation
- **OpenAPI/Swagger**: For provider API documentation
- **Golden Test Framework**: For migration validation
- **Protocol Buffers**: For future gRPC support

---

*Migration Reference Guide v1.0 - Complete*