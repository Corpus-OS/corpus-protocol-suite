# Corpus OS Adapters Guide

### From Implementation to Production: Adapter Patterns & Decisions

---

**Table of Contents**
- [0. How to Use This Guide](#0-how-to-use-this-guide)
- [1. Before You Write Code: Provider Assessment](#1-before-you-write-code-provider-assessment)
- [2. Batch Failure Mode: The First Critical Choice](#2-batch-failure-mode-the-first-critical-choice)
- [3. Streaming Pattern: Matching Provider Reality](#3-streaming-pattern-matching-provider-reality)
- [4. Error Mapping Strategies by Provider Type](#4-error-mapping-strategies-by-provider-type)
- [5. Provider-Specific Migration Paths](#5-provider-specific-migration-paths)
- [6. Multi-Tenancy Deep Dive](#6-multi-tenancy-deep-dive)
- [7. Deadline Propagation Patterns](#7-deadline-propagation-patterns)
- [8. Cache Ownership Explained](#8-cache-ownership-explained)
- [9. Idempotency Implementation Patterns](#9-idempotency-implementation-patterns)
- [10. Provider SDK Integration Patterns](#10-provider-sdk-integration-patterns)
- [11. Rate Limit Handling Deep Dive](#11-rate-limit-handling-deep-dive)
- [12. Operational Patterns](#12-operational-patterns)
- [13. Troubleshooting by Symptom](#13-troubleshooting-by-symptom)
- [14. Decision Matrix Quick Reference](#14-decision-matrix-quick-reference)
- [Appendix A: Provider Assessment Worksheet](#appendix-a-provider-assessment-worksheet)

---

> **Goal:** Help you make the right implementation choices for YOUR specific provider.  
> **Audience:** Developers who have read the [Quick Start](./QUICK_START.md) and [Implementation Guide](./IMPLEMENTATION.md) and now need to build production adapters.

**This is a decision guide, not a reference.** For exhaustive rules and requirements, see:
- [Quick Start](./QUICK_START.md) — Hello world, certification process, reference implementations
- [Implementation Guide](./IMPLEMENTATION.md) — All rules, all requirements, all edge cases

---

## 0. How to Use This Guide

Each section helps you **choose** between valid patterns based on your provider's actual behavior. You'll leave with:

- A clear set of decisions about your adapter's architecture
- Provider-specific migration guidance
- Operational patterns for running in production
- A troubleshooting reference for common issues

---

## 1. Before You Write Code: Provider Assessment

**Stop.** Before writing a single line of code, answer these questions about your provider. Your answers will determine every subsequent choice.

> 💡 **Pro Tip:** Use the [Provider Assessment Worksheet](#appendix-a-provider-assessment-worksheet) in Appendix A to document your answers. This becomes your adapter's architecture decision record.

### 1.1 Capabilities Inventory Checklist

Copy this table and fill it out for your provider:

| Capability | Your Provider's Behavior | Corpus Impact |
|------------|-------------------------|----------------|
| **Batch operations** | Does the API accept multiple items? | Determines batch failure mode (§2) |
| **Per-item status** | On batch, does it return success/failure per item? | Collection vs fail-fast choice |
| **Streaming** | Does it support streaming responses? How? (SSE, chunks, etc.) | Streaming pattern choice (§3) |
| **Token counting** | Does it provide token counts? API or local tokenizer? | `supports_token_counting` flag |
| **Normalization** | Are returned vectors already normalized? | `normalizes_at_source` flag |
| **Idempotency** | Does it support idempotency keys? | `idempotent_writes` flag must reflect reality |
| **Error details** | What fields appear in error responses? | Error detail mapping (§4) |
| **Rate limits** | How are they communicated? Headers? Response body? | ResourceExhausted mapping |
| **Authentication** | API key? OAuth? mTLS? | AuthError mapping |
| **Timeouts** | Does the SDK/client support timeouts? | Deadline propagation |
| **Tenancy** | Multi-tenant? How is tenant identified? | Tenant hashing requirements |

# Complete Error Taxonomy Reference

## 1.2 Error Taxonomy Mapping Worksheet

Before implementing error mapping, map your provider's errors using this complete reference:

| Provider Error (Example) | HTTP Status | Corpus Error | Required Details | Defined In |
|--------------------------|-------------|--------------|------------------|------------|
| `rate_limit_exceeded` | 429 | **ResourceExhausted** | `retry_after_ms`, `resource_scope` | [Impl Guide §4.2](./IMPLEMENTATION.md#42-provider-error-mapping-mandatory) |
| `invalid_api_key` | 401 | **AuthError** | none | [Impl Guide §4.2](./IMPLEMENTATION.md#42-provider-error-mapping-mandatory) |
| `model_not_found` | 404 | **ModelNotAvailable** | `requested_model`, `supported_models` | [Impl Guide §4.3](./IMPLEMENTATION.md#43-error-detail-schemas-per-error-type-mandatory) |
| `text_too_long` | 400 | **TextTooLong** | `max_length`, `actual_length` | [Impl Guide §4.3](./IMPLEMENTATION.md#43-error-detail-schemas-per-error-type-mandatory) |
| `dimension_mismatch` | 400 | **DimensionMismatch** | `expected`, `actual`, `namespace`, `vector_id`, `index` | [Impl Guide §4.3](./IMPLEMENTATION.md#43-error-detail-schemas-per-error-type-mandatory) |
| `index_not_ready` | 503 | **IndexNotReady** | `retry_after_ms`, `namespace` | [Impl Guide §4.3](./IMPLEMENTATION.md#43-error-detail-schemas-per-error-type-mandatory) |
| `namespace_mismatch` | 400 | **BadRequest** | `spec_namespace`, `vector_namespace`, `vector_id`, `index` | [Impl Guide §9.12](./IMPLEMENTATION.md#912-namespace-mismatch-error-details-canonical-shape-mandatory) |
| `filter_validation` | 400 | **BadRequest** | `operator`, `field`, `supported`, `namespace` | [Impl Guide §9.5](./IMPLEMENTATION.md#95-filter-operator-error-details-canonical-shape-mandatory) |
| `unsupported_operation` | 400 | **NotSupported** | none | [Impl Guide §4.2](./IMPLEMENTATION.md#42-provider-error-mapping-mandatory) |
| `timeout` | 504 | **TransientNetwork** | none | [Impl Guide §4.2](./IMPLEMENTATION.md#42-provider-error-mapping-mandatory) |
| `service_unavailable` | 503 | **Unavailable** | none | [Impl Guide §4.2](./IMPLEMENTATION.md#42-provider-error-mapping-mandatory) |
| `deadline_exceeded` | 408 | **DeadlineExceeded** | none | [Impl Guide §4.2](./IMPLEMENTATION.md#42-provider-error-mapping-mandatory) |

## Domain-Specific Errors

### LLM-Only Errors
| Provider Error | Corpus Error | Required Details |
|----------------|--------------|------------------|
| `model_overloaded` | **ModelOverloaded** | `retry_after_ms` (optional) |

### Embedding-Only Errors
| Provider Error | Corpus Error | Required Details |
|----------------|--------------|------------------|
| `text_too_long` | **TextTooLong** | `max_length`, `actual_length` |
| `model_not_available` | **ModelNotAvailable** | `requested_model`, `supported_models` |

### Vector-Only Errors
| Provider Error | Corpus Error | Required Details |
|----------------|--------------|------------------|
| `dimension_mismatch` | **DimensionMismatch** | `expected`, `actual`, `namespace`, `vector_id`, `index` |
| `index_not_ready` | **IndexNotReady** | `retry_after_ms`, `namespace` |

## Canonical Error Hierarchy

All errors inherit from domain-specific base classes:
- **LLM**: `LLMAdapterError` (import from `llm_base`)
- **Embedding**: `EmbeddingAdapterError` (import from `embedding_base`)
- **Vector**: `VectorAdapterError` (import from `vector_base`)
- **Graph**: `GraphAdapterError` (import from `graph_base`)

**See also:** [Implementation Guide §4.1 - Canonical Error Hierarchy](./IMPLEMENTATION.md#41-canonical-error-hierarchy)

### 1.3 Batch Operation Semantics

| Question | If Yes → | If No → |
|----------|----------|---------|
| Does the API return per-item status? | Collection pattern possible | Fail-fast only |
| Can you submit 100 items and get 98 successes? | Use collection pattern | Use fail-fast |
| What happens on validation error? | Batch rejected entirely → fail-fast | Per-item errors → collection |

---

## 2. Batch Failure Mode: The First Critical Choice

**This is the most important decision you'll make.** Your choice determines how your adapter behaves when some items in a batch succeed and others fail.

### 2.1 Decision Tree

```
Start: Does your provider have a batch API?
├─ NO → Use single-item operations (base handles batching via loop)
│
└─ YES → Does the provider return per-item status?
    ├─ NO → You MUST use FAIL-FAST mode
    │   (any error fails the entire batch)
    │
    └─ YES → Can you submit 100 items and get 98 successes + 2 errors?
        ├─ NO (provider fails whole batch on any error) → FAIL-FAST
        │
        └─ YES → You can choose, but COLLECTION is recommended
            (better user experience, partial success reporting)
```

### 2.2 Provider Patterns by Category

| Provider Type | Typical Behavior | Recommended Mode | Rationale |
|--------------|------------------|------------------|-----------|
| **Embedding APIs** (OpenAI, Cohere) | Per-item status with partial success | **Collection** | Users expect partial results when some texts fail |
| **Vector DBs** (Pinecone, Qdrant) | Batch rejected on dimension mismatch | **Fail-fast** | Schema violations are request-level, not per-item |
| **Graph DBs with transactions** | Atomic batches | **Fail-fast** | Transaction semantics require all-or-nothing |
| **Graph DBs without transactions** | Per-op status | **Collection** | No atomicity guarantee, report per-op failures |

### 2.3 🔴 CRITICAL: The One Rule - NEVER Make It Configurable

```python
# ❌ WRONG
class MyAdapter(BaseAdapter):
    def __init__(self, collect_failures=True):  # NO
        self.collect_failures = collect_failures

# ✅ CORRECT
class MyAdapter(BaseAdapter):
    """BATCH FAILURE MODE: Collection pattern (provider returns per-item status)"""
    # Hardcoded, documented, not configurable
```

**Why:** Conformance tests expect deterministic behavior. If your adapter behaves differently based on a flag, it will fail certification in non-deterministic ways. This single rule prevents 80% of certification-related support tickets.

### 2.4 Documenting Your Choice

```python
class MyEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    BATCH FAILURE MODE: Collect per-item failures.
    
    This adapter processes all items in the batch and collects individual failures.
    The batch operation always succeeds (HTTP 200) with partial results.
    Failures are reported in the `failed_texts` array with:
    - index: position in original batch
    - text: truncated preview (first 100 chars)
    - error: exception class name
    - code: uppercase error code
    - message: truncated error message
    - metadata: original metadata if provided
    
    Rationale: Provider's batch API returns per-item status and supports
    partial success. This provides the best user experience.
    """
```

**See also:** [Implementation Guide §8.3 - Batch Failure Mode (CHOOSE ONE)](./IMPLEMENTATION.md#83-batch-failure-mode-choose-one-mandatory), [§12 - Batch Failure Mode Decision Matrix](./IMPLEMENTATION.md#12-batch-failure-mode-decision-matrix)

---

## 3. Streaming Pattern: Matching Provider Reality

**Streaming means different things for different components.** Choose the pattern that matches your provider's actual behavior.

### 3.1 Streaming by Component

| Component | What Streaming Typically Means | Common Patterns |
|-----------|-------------------------------|-----------------|
| **LLM** | Token-by-token generation | Progressive (token stream) |
| **Embedding** | Rare - most return complete vectors | Single-chunk (emulated) or Multi-vector (batch streaming) |
| **Vector** | Not applicable | N/A (no streaming in protocol) |
| **Graph** | Row-by-row query results | Multi-vector (row stream) |

### 3.2 The Three Patterns

| Pattern | Description | When To Use |
|---------|-------------|-------------|
| **Single-chunk** | One final chunk with complete result | Provider returns entire response at once; used to emulate streaming when provider doesn't support it |
| **Progressive** | Partial results that build to completion | LLM token streaming, progressive inference |
| **Multi-vector** | Multiple complete results per stream | Batch results returning one by one, graph query row streaming |

### 3.3 Decision Tree: Which Pattern Fits Your Provider?

```
Start: Does your provider have a streaming API?
├─ NO → You MUST use SINGLE-CHUNK (emulate streaming with one chunk)
│
└─ YES → What component are you implementing?
    ├─ LLM → PROGRESSIVE (token stream)
    │   Example: "Hello", " world", "!" 
    │
    ├─ Graph Query → MULTI-VECTOR (row stream)
    │   Example: row 1, row 2, row 3, end
    │
    └─ Embedding → (Rare) What does the stream contain?
        ├─ Complete vectors, one at a time? → MULTI-VECTOR
        │   Example: "embedding-1", then "embedding-2"
        │
        └─ Partial vectors building to completion? → PROGRESSIVE
            (Very rare - confirm your provider actually does this)
```

### 3.4 Pattern Examples by Component

| Component | Pattern | Example Providers |
|-----------|---------|-------------------|
| **LLM** | Progressive | OpenAI stream=True, Anthropic streaming |
| **Graph** | Multi-vector | Neo4j streaming results, SQL row streaming |
| **Embedding** | Single-chunk (emulated) | OpenAI embeddings (non-streaming) |
| **Embedding** | Multi-vector (rare) | Batch embedding APIs that return as ready |

### 3.5 Special Case: LLM Tool Call Streaming

For LLM adapters, tool calls have specific rules regardless of pattern:

```python
# Rule 1: Tool calls appear ONLY in final chunk
if tool_calls:
    # Non-final chunks are empty
    yield LLMChunk(text="", is_final=False)
    
    # Final chunk has tool_calls
    yield LLMChunk(
        text="",
        is_final=True,
        tool_calls=tool_calls,
        usage_so_far=final_usage
    )
```

**See also:** [Implementation Guide §7.6 - Streaming Rules (Tool Calls)](./IMPLEMENTATION.md#76-streaming-rules-tool-calls-mandatory)

### 3.6 Documenting Your Choice

```python
class MyLLMAdapter(BaseLLMAdapter):
    """
    STREAMING PATTERN: Progressive (token stream).
    
    This adapter emits chunks as tokens are generated by the provider.
    Tool calls appear only in the final chunk with empty text.
    
    Rationale: Provider streams tokens via server-sent events.
    """

class MyEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    STREAMING PATTERN: Single-chunk (emulated).
    
    This adapter emits exactly one chunk with is_final=True.
    Provider does not support true streaming; this is for protocol compliance.
    """
```

---

## 4. Error Mapping Strategies by Provider Type

**Different provider types need different error mapping strategies.** This section helps you choose the right approach.

### 4.1 REST API Providers (OpenAI, Cohere, Pinecone)

**Strategy:** Map HTTP status codes first, then response body fields.

| Status Code | Typical Meaning | Corpus Error | Details to Extract |
|-------------|-----------------|--------------|-------------------|
| 429 | Rate limited | ResourceExhausted | Retry-After header, rate limit scope |
| 401 | Unauthorized | AuthError | Error message |
| 403 | Forbidden | AuthError | Error message |
| 400 | Bad request | BadRequest | Validation details from body |
| 404 | Not found | BadRequest | What wasn't found |
| 500+ | Server error | Unavailable | None |

### 4.2 SDK-Based Providers (Neo4j Driver, pgvector)

**Strategy:** Map exception types directly.

| SDK Exception | Corpus Error | Notes |
|--------------|--------------|-------|
| `TransientError` | TransientNetwork | Connection issues, timeouts |
| `ClientError` | BadRequest | Invalid queries, syntax errors |
| `AuthError` | AuthError | Authentication failures |
| `DatabaseError` | Unavailable | Server-side issues |

### 4.3 gRPC Providers

**Strategy:** Map gRPC status codes, check trailing metadata.

| gRPC Status Code | Corpus Error | Notes |
|------------------|--------------|-------|
| `RESOURCE_EXHAUSTED` | ResourceExhausted | Check trailing metadata for retry info |
| `UNAUTHENTICATED` | AuthError | - |
| `INVALID_ARGUMENT` | BadRequest | - |
| `UNAVAILABLE` | TransientNetwork | - |
| `DEADLINE_EXCEEDED` | DeadlineExceeded | - |

### 4.4 The Detail Schema Requirements (By Error Type)

**Every error MUST include specific fields.** Here's the complete reference:

| Error Type | Required Fields | Example |
|------------|-----------------|---------|
| **DimensionMismatch** | `expected`, `actual`, `namespace`, `vector_id`, `index` | `{"expected": 384, "actual": 512, "namespace": "docs", "vector_id": "vec_123", "index": 3}` |
| **NamespaceMismatch** | `spec_namespace`, `vector_namespace`, `vector_id`, `index` | `{"spec_namespace": "default", "vector_namespace": "other", "vector_id": "vec_123", "index": 2}` |
| **FilterValidation** | `operator`, `field`, `supported`, `namespace` | `{"operator": "$regex", "field": "title", "supported": ["$in"], "namespace": "docs"}` |
| **ModelNotAvailable** | `requested_model`, `supported_models` | `{"requested_model": "gpt-5", "supported_models": ["gpt-4", "gpt-3.5"]}` |
| **IndexNotReady** | `namespace` | `{"namespace": "docs"}` |
| **TextTooLong** | `max_length`, `actual_length` | `{"max_length": 8192, "actual_length": 15000}` |
| **ResourceExhausted** | `resource_scope` | `{"resource_scope": "rate_limit"}` or `"quota"` or `"concurrency"` |

**See also:** [Implementation Guide §4.3 - Error Detail Schemas (PER ERROR TYPE)](./IMPLEMENTATION.md#43-error-detail-schemas-per-error-type-mandatory)

---

## 5. Provider-Specific Migration Paths

### 5.1 Coming from OpenAI

**What changes when you move from OpenAI SDK to Corpus adapter?**

| OpenAI Concept | Corpus Concept | What You Must Change |
|----------------|----------------|---------------------|
| **Idempotency optional** | `idempotent_writes` must reflect reality | Set flag based on whether you implement it |
| **Raw tenant IDs** | Tenant hashing | Hash all tenant IDs before logging/storing |
| **Batch errors** | `failed_texts` array | Map per-item errors with `index` field |
| **No deadline propagation** | `ctx.remaining_ms()` | Add timeout to all HTTP calls |
| **Streaming chunks** | Tool call rules | Tool calls only in final chunk |

**Error Mapping: OpenAI → Corpus**

| OpenAI Error | Corpus Error | Required Details |
|--------------|--------------|------------------|
| `rate_limit_error` | `ResourceExhausted` | `retry_after_ms` from headers, `resource_scope="rate_limit"` |
| `invalid_api_key` | `AuthError` | None |
| `model_not_found` | `ModelNotAvailable` | `requested_model`, `supported_models` |
| `context_length_exceeded` | `BadRequest` | None |
| `server_error` (500) | `Unavailable` | None |

### 5.2 Coming from Pinecone

**What changes when you move from Pinecone SDK to Corpus adapter?**

| Pinecone Concept | Corpus Concept | Why Your Code Probably Breaks |
|------------------|----------------|------------------------------|
| **Per-vector namespace** | **Single namespace per call** | Pinecone allows per-vector namespaces; Corpus requires ALL vectors in a batch to have the same namespace |
| **$eq is default** | **Only $in is supported** | Your filters using `{"genre": "doc"}` (implicit $eq) must become `{"genre": {"$in": ["doc"]}}` |
| **No delete validation** | **IDs XOR Filter required** | Your delete calls must provide EITHER ids OR filter, never both, never neither |
| **Delete counts attempts** | **Delete counts actual deletions** | Your code probably returns `len(ids)`; must return actual count of deleted vectors |
| **Index status** | **IndexNotReady with retry_after** | Must add retry hints and namespace details |

**Filter Translation Guide:**

```python
# Pinecone filter -> Corpus filter
pinecone_filter = {"genre": "doc"}
# becomes
corpus_filter = {"genre": {"$in": ["doc"]}}

pinecone_filter = {"genre": {"$in": ["doc", "article"]}}
# becomes (same, $in is supported)
corpus_filter = {"genre": {"$in": ["doc", "article"]}}

# UNSUPPORTED: $ne, $gt, $lt, $exists, etc.
# Must raise BadRequest with supported list
```

**See also:** [Implementation Guide §9.4 - Filter Dialect Validation](./IMPLEMENTATION.md#94-filter-dialect-validation-strict-no-silent-ignore-mandatory), [§9.5 - Filter Operator Error Details](./IMPLEMENTATION.md#95-filter-operator-error-details-canonical-shape-mandatory)

### 5.3 Coming from Neo4j

**What changes when you move from Neo4j driver to Corpus adapter?**

| Neo4j Concept | Corpus Concept | What You Must Change |
|----------------|----------------|---------------------|
| **Raw session results** | **QueryResult wrapper** | Wrap records in QueryResult object |
| **No dialect validation** | **Must validate dialects** | Check `spec.dialect` against `supported_query_dialects` |
| **Batch operations** | **{ok, result} envelope** | Wrap each batch result in `{"ok": True/False, "result": ...}` |
| **Transactions** | **Atomic transactions** | ALL operations must succeed or none |
| **Delete on missing** | **Idempotent delete** | No error when node/edge doesn't exist |

**Batch Operation Envelope Pattern:**

```python
# In your shared op executor
results.append({
    "ok": True,
    "result": {
        "nodes_created": 1,
        "properties_set": 3
    }
})

# Not
results.append({
    "nodes_created": 1,
    "properties_set": 3
})
```

**See also:** [Implementation Guide §10.2 - Batch/Transaction Result Envelope](./IMPLEMENTATION.md#102-batchtransaction-result-envelope-ok-result-mandatory), [§10.4 - Dialect Validation](./IMPLEMENTATION.md#104-dialect-validation-two-layers-mandatory)

---

## 6. Multi-Tenancy Deep Dive

**Getting tenant isolation right is critical. Most production incidents stem from getting this wrong.**

### 6.1 The Three Rules of Tenant Handling

```python
# Rule 1: NEVER log or emit raw tenant IDs
tenant_id = ctx.tenant if ctx else None
if tenant_id:
    logger.info(f"Request for tenant {tenant_id}")  # ❌ PII LEAK
    metrics.increment("requests", tags={"tenant": tenant_id})  # ❌ PII LEAK

# ✅ CORRECT: Always hash
tenant_hash = self._tenant_hash(tenant_id)
logger.info(f"Request for tenant {tenant_hash}")
metrics.increment("requests", tags={"tenant": tenant_hash})

# Rule 2: Use hashed tenant in cache keys
cache_key = f"embed:{tenant_hash}:{model}:{text_hash}"  # ✅

# Rule 3: Use raw tenant ONLY for provider calls that need it
if tenant_id:
    provider_request["tenant"] = tenant_id  # ✅ Provider needs raw ID
```

### 6.2 The Tenant Hashing Function

```python
def _tenant_hash(self, tenant: Optional[str]) -> Optional[str]:
    """Create privacy-preserving hash of tenant identifier."""
    if not tenant:
        return None
    # Use SHA256, truncate to reasonable length
    return hashlib.sha256(tenant.encode()).hexdigest()[:16]
```

### 6.3 When to Use Raw vs Hashed Tenant

| Use Case | Use Raw Tenant | Use Hashed Tenant |
|----------|---------------|-------------------|
| Provider API calls | ✅ Provider needs it | ❌ |
| Logging | ❌ Never | ✅ Always |
| Metrics tags | ❌ Never | ✅ Always |
| Cache keys | ❌ PII exposure | ✅ Safe |
| Error details | ❌ Never | ✅ Safe |
| Namespace names | ✅ Often needed | ❌ |

### 6.4 Common Multi-Tenancy Mistakes

```python
# Mistake 1: Raw tenant in logs
logger.info(f"Processing request for {ctx.tenant}")

# Mistake 2: Raw tenant in cache keys
cache_key = f"embedding:{ctx.tenant}:{model}"

# Mistake 3: Assuming tenant is always present
if ctx.tenant:  # May be None for global/anon requests
    # Handle tenant-specific logic
else:
    # Must handle global case

# Mistake 4: Forgetting tenant in cache isolation
# Different tenants should NOT share cache keys
# Base handles this automatically when you use tenant_hash
```

### 6.5 Tenant-Aware Rate Limiting

```python
# The base class handles tenant-aware rate limiting automatically
# You don't need to implement this - just ensure ctx.tenant is passed

# BUT: If your provider has per-tenant rate limits, you need to track them
def _map_provider_error(self, e):
    if "rate limit" in str(e).lower():
        # Extract tenant from context for error details
        tenant_hash = self._tenant_hash(ctx.tenant) if ctx else None
        return ResourceExhausted(
            "Rate limit exceeded",
            retry_after_ms=5000,
            resource_scope="rate_limit",
            details={
                "tenant_hash": tenant_hash,  # ✅ Safe to include
                "provider_error": str(e)
            }
        )
```

**See also:** [Implementation Guide §3.3 - Tenant Hashing (MANDATORY)](./IMPLEMENTATION.md#33-tenant-hashing-mandatory)

---

## 7. Deadline Propagation Patterns

**Different provider SDKs handle timeouts differently. Here's how to map `ctx.remaining_ms()` to each.**

### 7.1 HTTPX / Requests (Most REST APIs)

```python
def _timeout_from_ctx(self, ctx):
    if ctx is None:
        return None
    rem = ctx.remaining_ms()
    if rem is None or rem <= 0:
        return None
    return rem / 1000.0  # Convert to seconds

async def _do_embed(self, spec, *, ctx=None):
    timeout = self._timeout_from_ctx(ctx)
    
    response = await self._client.post(
        url,
        json=payload,
        timeout=timeout  # ✅ Pass seconds
    )
```

### 7.2 gRPC (Deadline Pattern)

```python
import datetime

def _deadline_from_ctx(self, ctx):
    if ctx is None or ctx.deadline_ms is None:
        return None
    # gRPC expects absolute datetime
    return datetime.datetime.fromtimestamp(ctx.deadline_ms / 1000)

async def _do_query(self, spec, *, ctx=None):
    deadline = self._deadline_from_ctx(ctx)
    
    response = await self._stub.Query(
        request,
        deadline=deadline  # ✅ gRPC deadline
    )
```

### 7.3 AWS SDK (Boto3)

```python
def _timeout_from_ctx(self, ctx):
    if ctx is None:
        return None
    rem = ctx.remaining_ms()
    if rem is None or rem <= 0:
        return None
    return rem / 1000.0

async def _do_embed(self, spec, *, ctx=None):
    timeout = self._timeout_from_ctx(ctx)
    
    # Boto3 uses config objects
    config = Config(
        connect_timeout=timeout,
        read_timeout=timeout
    )
    
    response = await self._client.embed(
        Text=spec.text,
        Config=config
    )
```

### 7.4 Custom SDK Without Timeout Support

```python
async def _do_embed(self, spec, *, ctx=None):
    if ctx and ctx.deadline_ms:
        remaining = ctx.remaining_ms()
        if remaining <= 0:
            raise DeadlineExceeded("deadline expired")
        
        # Use asyncio.timeout as a safety net
        try:
            async with asyncio.timeout(remaining / 1000):
                response = await self._client.embed(spec.text)
        except asyncio.TimeoutError:
            raise DeadlineExceeded("provider timeout")
    else:
        response = await self._client.embed(spec.text)
```

### 7.5 🔴 CRITICAL: The Critical Rule - Always Propagate

```python
# ❌ WRONG: Ignoring deadline
async def _do_embed(self, spec, *, ctx=None):
    return await self._client.embed(spec.text)  # No timeout!

# ✅ CORRECT: Always propagate
async def _do_embed(self, spec, *, ctx=None):
    timeout = self._timeout_from_ctx(ctx)
    return await self._client.embed(spec.text, timeout=timeout)
```

**Why:** If you don't propagate deadlines, requests can hang forever, exhausting resources and violating SLOs.

**See also:** [Implementation Guide §3.2 - Deadline Propagation (MANDATORY)](./IMPLEMENTATION.md#32-deadline-propagation-mandatory), [§6 - Deadlines & Cancellation](./IMPLEMENTATION.md#6-deadlines--cancellation)

---

## 8. Cache Ownership Explained

**This is the most misunderstood boundary in the entire system.**

### 8.1 What the Base Class Owns

```python
# The base class automatically provides:
# - In-memory TTL cache (in standalone mode)
# - Cache key generation (includes tenant hash)
# - Cache hit/miss counting
# - Cache invalidation for vector/graph after writes
# - TTL enforcement

# You DO NOT:
# - Implement your own cache
# - Report cache metrics in _do_get_stats
# - Invalidate cache manually (base handles it)
```

### 8.2 What You (The Adapter) Own

```python
# You own:
# - Provider response generation
# - Error mapping
# - Provider-side metrics (latency, error rates)
# - Everything NOT related to caching

# In _do_get_stats, you report ONLY adapter metrics:
async def _do_get_stats(self, *, ctx=None):
    return EmbeddingStats(
        total_requests=self._stats["total_ops"],
        total_texts=self._stats["total_texts"],
        total_tokens=self._stats["total_tokens"],
        avg_processing_time_ms=self._stats["avg_ms"],
        error_count=self._stats["errors"]
        # ❌ NO cache_hits
        # ❌ NO cache_misses
        # ❌ NO stream_stats
    )
```

### 8.3 The Cache Invalidation Contract (Vector/Graph)

```python
async def _do_upsert(self, spec: UpsertSpec, *, ctx=None) -> UpsertResult:
    # 1. Perform the upsert
    result = await self._client.upsert(vectors)
    
    # 2. ✅ AFTER successful write, base automatically invalidates
    #    You do NOT call _invalidate_namespace_cache manually
    #    Base detects writes via UpsertResult
    
    return UpsertResult(
        upserted_count=result.count,
        failed_count=0,
        failures=[]
    )

# ❌ WRONG: Manual invalidation
await self._invalidate_namespace_cache(spec.namespace)  # NO - base handles it
```

### 8.4 Cache Key Generation (For Reference)

```python
# You don't need to implement this, but understand how it works:
# Base generates cache keys like:
cache_key = f"{component}:{tenant_hash}:{operation}:{content_hash}"

# Example:
# "embedding:1a2b3c4d:embed:5e6f7g8h"
```

### 8.5 🔴 CRITICAL: Common Cache Ownership Mistakes

```python
# Mistake 1: Including cache stats in _do_get_stats
async def _do_get_stats(self):
    return {
        "total_requests": 1000,
        "cache_hits": 800,  # ❌ Base owns this
        "cache_misses": 200,  # ❌ Base owns this
    }

# Mistake 2: Implementing your own cache
self._my_cache = {}  # ❌ Use base cache or none

# Mistake 3: Manual invalidation
await self._invalidate_namespace_cache(ns)  # ❌ Base handles it

# Mistake 4: Assuming cache exists in 'thin' mode
# In thin mode, cache is a no-op. Your adapter should work either way.
```

**See also:** [Implementation Guide §8.6 - Cache Stats Ownership (CRITICAL BOUNDARY)](./IMPLEMENTATION.md#86-cache-stats-ownership-critical-boundary-mandatory), [§11 - Cache Ownership Boundary (CRITICAL)](./IMPLEMENTATION.md#11-cache-ownership-boundary-critical)

---

## 9. Idempotency Implementation Patterns

**Required ONLY if your provider supports idempotency AND you set `idempotent_writes=True`.**

### 9.1 What Idempotency Means in Corpus

```python
# If you set idempotent_writes=True, you MUST implement:
# - Given the same idempotency key within 24 hours:
#   - First call → process and store result
#   - Subsequent calls → return stored result without processing
# - Different keys → treated as different requests

# If your provider DOES NOT support idempotency, set idempotent_writes=False
```

### 9.2 Pattern A: In-Memory Cache (Development/Testing)

```python
class MyEmbeddingAdapter(BaseEmbeddingAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._idempotency_cache = {}  # {key: (timestamp, result)}
    
    async def _do_embed(self, spec, *, ctx=None):
        # Check idempotency
        if ctx and ctx.idempotency_key and ctx.tenant:
            key = f"idem:{ctx.tenant}:{ctx.idempotency_key}"
            cached = self._idempotency_cache.get(key)
            if cached:
                timestamp, result = cached
                if time.time() - timestamp < 86400:  # 24 hours
                    return result
        
        # Process normally
        result = await self._process_embed(spec, ctx)
        
        # Store for idempotency
        if ctx and ctx.idempotency_key and ctx.tenant:
            self._idempotency_cache[key] = (time.time(), result)
        
        return result
    
    # Optional: Cleanup old entries periodically
    def _cleanup_cache(self):
        now = time.time()
        self._idempotency_cache = {
            k: v for k, v in self._idempotency_cache.items()
            if now - v[0] < 86400
        }
```

### 9.3 Pattern B: Redis (Production)

```python
import redis.asyncio as redis
import json

class MyEmbeddingAdapter(BaseEmbeddingAdapter):
    def __init__(self, redis_url=None, **kwargs):
        super().__init__(**kwargs)
        self._redis = redis.from_url(redis_url) if redis_url else None
    
    async def _do_embed(self, spec, *, ctx=None):
        # Check idempotency
        if ctx and ctx.idempotency_key and ctx.tenant and self._redis:
            key = f"idem:v1:{ctx.tenant}:{ctx.idempotency_key}"
            cached = await self._redis.get(key)
            if cached:
                data = json.loads(cached)
                return self._deserialize_result(data)
        
        # Process normally
        result = await self._process_embed(spec, ctx)
        
        # Store for idempotency with 24h TTL
        if ctx and ctx.idempotency_key and ctx.tenant and self._redis:
            data = self._serialize_result(result)
            data["_schema_version"] = "1.0"
            await self._redis.setex(
                key,
                86400,  # 24 hours
                json.dumps(data)
            )
        
        return result
```

### 9.4 Pattern C: No Storage (When Idempotency Not Required)

```python
class MyLLMAdapter(BaseLLMAdapter):
    async def _do_capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            # ... other fields
            idempotent_writes=False,  # ✅ Honest - not implementing
        )
    
    # No idempotency implementation needed
```

### 9.5 Key Requirements (If You Implement Idempotency)

```python
# 1. Key MUST include tenant
key = f"idem:{ctx.tenant}:{ctx.idempotency_key}"  # ✅
key = f"idem:{ctx.idempotency_key}"  # ❌ No tenant isolation

# 2. TTL MUST be 24 hours minimum
await self._redis.setex(key, 86400, data)  # ✅ 24h
await self._redis.setex(key, 3600, data)   # ❌ Too short

# 3. Storage SHOULD use versioned schema
# 4. Cache MUST survive adapter restarts in production
```

**See also:** [Quick Start §7.1 - Embedding Protocol](./QUICK_START.md#71-embedding-protocol), [Implementation Guide §8 - Embedding Adapter Implementation Requirements](./IMPLEMENTATION.md#8-embedding-adapter-implementation-requirements)

---

## 10. Provider SDK Integration Patterns

**How to integrate with existing provider SDKs.**

### 10.1 Pattern 1: Wrap Existing Client

```python
class MyAdapter(BaseAdapter):
    def __init__(self, provider_client, **kwargs):
        """
        Args:
            provider_client: Existing client from provider SDK
                            (OpenAI client, Pinecone index, etc.)
        """
        super().__init__(**kwargs)
        self._client = provider_client  # Use existing client
    
    async def _do_embed(self, spec, *, ctx=None):
        # Just adapt the interface
        timeout = self._timeout_from_ctx(ctx)
        
        # Provider SDK may have different timeout mechanisms
        response = await self._client.embeddings.create(
            model=spec.model,
            input=spec.text,
            timeout=timeout
        )
        
        # Map to Corpus format
        return EmbedResult(...)
```

### 10.2 Pattern 2: Build New Client

```python
import httpx

class MyAdapter(BaseAdapter):
    def __init__(self, api_key, base_url=None, **kwargs):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._base_url = base_url or "https://api.provider.com/v1"
        self._client = httpx.AsyncClient(timeout=30.0)
    
    async def _do_embed(self, spec, *, ctx=None):
        timeout = self._timeout_from_ctx(ctx)
        
        response = await self._client.post(
            f"{self._base_url}/embeddings",
            json={
                "model": spec.model,
                "input": spec.text
            },
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        
        return EmbedResult(...)
    
    async def close(self):
        await self._client.aclose()
```

### 10.3 Pattern 3: Sync SDK + asyncio

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class MyAdapter(BaseAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = ProviderSyncClient()  # Sync SDK
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    async def _do_embed(self, spec, *, ctx=None):
        # Run blocking call in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self._executor,
            lambda: self._client.embed(
                model=spec.model,
                text=spec.text
            )
        )
        return EmbedResult(...)
```

### 10.4 Pattern 4: Multi-Protocol Provider

```python
# adapters/acme/__init__.py
from .client import AcmeClient
from .llm import AcmeLLMAdapter
from .embedding import AcmeEmbeddingAdapter
from .vector import AcmeVectorAdapter

# All adapters share the same client
client = AcmeClient(api_key="...")

llm_adapter = AcmeLLMAdapter(client)
embedding_adapter = AcmeEmbeddingAdapter(client)
vector_adapter = AcmeVectorAdapter(client)
```

### 10.5 Client Lifecycle Management

```python
class MyAdapter(BaseAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = None  # Lazy init
    
    async def _ensure_client(self):
        if self._client is None:
            self._client = await self._create_client()
    
    async def _do_embed(self, spec, *, ctx=None):
        await self._ensure_client()
        # Use client...
    
    async def close(self):
        if self._client:
            await self._client.aclose()
```

---

## 11. Rate Limit Handling Deep Dive

**Different providers surface rate limits differently. Here's how to handle each pattern.**

### 11.1 Pattern A: Headers-Based (Most REST APIs)

```python
def _map_provider_error(self, e: httpx.HTTPStatusError):
    if e.response.status_code == 429:
        # Check multiple header formats
        retry_after = None
        
        # Standard Retry-After (seconds)
        if "Retry-After" in e.response.headers:
            retry_after = int(e.response.headers["Retry-After"]) * 1000
        
        # X-RateLimit-* headers
        elif "X-RateLimit-Reset" in e.response.headers:
            # Unix timestamp
            reset = int(e.response.headers["X-RateLimit-Reset"])
            now = time.time()
            retry_after = max(0, (reset - now)) * 1000
        
        # Determine scope
        scope = "api"
        if "X-RateLimit-Scope" in e.response.headers:
            scope = e.response.headers["X-RateLimit-Scope"]
        
        return ResourceExhausted(
            "Rate limit exceeded",
            retry_after_ms=retry_after or 5000,
            resource_scope=scope,
            details={
                "limit": e.response.headers.get("X-RateLimit-Limit"),
                "remaining": e.response.headers.get("X-RateLimit-Remaining")
            }
        )
```

### 11.2 Pattern B: Response Body

```python
def _map_provider_error(self, e: Exception):
    if isinstance(e, ProviderRateLimitError):
        # Extract from error object
        data = e.response.json()
        
        retry_after = None
        if "retry_after" in data:
            retry_after = data["retry_after"]  # May be seconds or ms
            if retry_after < 100:  # Probably seconds
                retry_after *= 1000
        
        scope = data.get("scope", "api")
        
        return ResourceExhausted(
            data.get("message", "Rate limit exceeded"),
            retry_after_ms=retry_after or 5000,
            resource_scope=scope,
            details={"provider_response": data}
        )
```

### 11.3 Pattern C: No Retry Information

```python
def _map_provider_error(self, e: Exception):
    if self._is_rate_limit(e):
        # No retry info provided - use defaults
        return ResourceExhausted(
            "Rate limit exceeded",
            retry_after_ms=5000,  # Default
            resource_scope="unknown",
            details={"provider_error": str(e)}
        )
```

### 11.4 Pattern D: Multiple Scopes

```python
def _map_provider_error(self, e: Exception):
    if e.response.status_code == 429:
        data = e.response.json()
        
        # Determine which scope was hit
        if "model" in data.get("scope", ""):
            scope = f"model:{data.get('model', 'unknown')}"
        elif "namespace" in data.get("scope", ""):
            scope = f"namespace:{data.get('namespace', 'unknown')}"
        else:
            scope = "api"
        
        return ResourceExhausted(
            "Rate limit exceeded",
            retry_after_ms=data.get("retry_after_ms", 5000),
            resource_scope=scope,
            details={
                "limit_type": data.get("type"),
                "limit": data.get("limit")
            }
        )
```

### 11.5 Rate Limit Scope Values

| Scope | Meaning | When Used |
|-------|---------|-----------|
| `api` | Overall API rate limit | Default |
| `model` | Per-model rate limit | LLM/Embedding providers |
| `namespace` | Per-namespace rate limit | Vector/Graph databases |
| `user` | Per-user rate limit | Multi-tenant services |
| `concurrency` | Concurrent request limit | Streaming/async APIs |

### 11.6 Implementing Retry with Rate Limits

```python
async def _call_with_retry(self, fn, ctx=None):
    for attempt in range(3):
        try:
            return await fn()
        except ResourceExhausted as e:
            if attempt == 2:  # Last attempt
                raise
            
            # Use retry_after from error
            delay_ms = e.retry_after_ms or (5000 * (2 ** attempt))
            
            # Check if we have time to retry
            if ctx:
                remaining = ctx.remaining_ms()
                if remaining and delay_ms > remaining:
                    raise DeadlineExceeded("deadline too short for retry")
            
            await asyncio.sleep(delay_ms / 1000)
```

**See also:** [Implementation Guide §4.4 - Retry Semantics (MANDATORY)](./IMPLEMENTATION.md#44-retry-semantics-mandatory)

---

## 12. Operational Patterns

### 12.1 Monitoring: What Metrics Matter

| Metric | Source | What to Watch For |
|--------|--------|-------------------|
| **Provider latency** | Your HTTP/SDK client | Spikes indicate provider issues |
| **Error rate by type** | Your error mapping | Unexpected error types mean mapping gaps |
| **Rate limit hits** | ResourceExhausted count | May need to adjust retry strategy |
| **Deadline exceeded** | DeadlineExceeded count | Timeouts too aggressive? |
| **Cache hit rate** | Base class metrics | Low hit rate? Adjust TTL |

### 12.2 🔴 CRITICAL: Logging - What's Safe, What's Not

**NEVER log:**
- Raw tenant IDs (use tenant hash)
- Full text payloads (PII risk)
- Embedding vectors (can be reverse-engineered)
- API keys or secrets

**ALWAYS log:**
- Tenant hash (for debugging multi-tenant issues)
- Operation type (embed, complete, query)
- Error type and code (not full stack traces in production)
- Request IDs for correlation

**Example safe log:**
```json
{
  "tenant_hash": "a1b2c3d4e5f6",
  "operation": "embed",
  "model": "text-embedding-ada-002",
  "duration_ms": 245,
  "success": true,
  "request_id": "req_123"
}
```

### 12.3 Health Check Best Practices

```python
async def _do_health(self, *, ctx=None) -> Dict[str, Any]:
    """
    Health check should:
    - Be lightweight (don't call expensive operations)
    - Return graded status (ok/degraded/down)
    - Include version information
    - Include component-specific status
    """
    try:
        # Simple ping to provider
        await self._client.ping(timeout=2.0)
        return {
            "ok": True,
            "status": "ok",
            "server": "my-adapter",
            "version": "1.0.0"
        }
    except Exception:
        return {
            "ok": False,
            "status": "down",
            "server": "my-adapter",
            "version": "1.0.0"
        }
```

### 12.4 Shutdown and Cleanup

```python
class MyAdapter(BaseAdapter):
    async def close(self):
        """Clean up resources when adapter is shut down."""
        if hasattr(self, "_client"):
            await self._client.aclose()
        
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True)
```

---

## 13. Troubleshooting by Symptom

### 13.1 Conformance Failures During Certification

| Error Message | Likely Cause | Fix |
|--------------|--------------|-----|
| `capabilities missing required field: protocol` | Forgot protocol field | Add `protocol="llm/v1.0"` (or your domain) |
| `capabilities missing required field: model_family` (LLM) | Missing model_family | Add model_family to LLM capabilities |
| `capabilities missing required field: idempotent_writes` (Embedding) | Missing idempotency flag | Set based on your implementation (true/false) |
| `Batch result missing field: failures` | Wrong field name | Use `failed_texts` for embedding, `failures` for others |
| `Batch success missing index` | No index on embedding vectors | Add `index=idx` to EmbeddingVector |
| `Namespace mismatch` | Vector.namespace != spec.namespace | Enforce namespace authority |
| `Missing index in failure` | Failure object missing required field | Add `index` to all failure objects |
| `Missing metadata in failure` | Failure object missing metadata | Add `metadata` field (may be null) |
| `include_vectors=False returned vectors` | Returning full vectors | Return `[]` when `include_vectors=False` |
| `Delete returned attempted count` | Counting attempts not actuals | Count actual deletions only |
| `Batch query not atomic` | Continuing after query failure | Validate all first, then execute |
| `Missing retry_after_ms` | IndexNotReady missing retry hint | Always add `retry_after_ms` |
| `Missing namespace in error details` | Error missing required field | Add namespace to all namespace-related errors |
| `Tool calls in non-final chunk` | Streaming pattern wrong | Tool calls only in final chunk |
| `Stop sequence cut at last occurrence` | Wrong stop logic | Cut at FIRST occurrence |
| `Zero completion tokens for tool calls` | Missing token synthesis | Synthesize from tool call payload |

### 13.2 Production Errors

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `ResourceExhausted` with no retry_after | Incomplete error mapping | Add retry_after_ms from provider headers |
| `DeadlineExceeded` but provider fast | Not propagating deadlines | Add `ctx.remaining_ms()` to client timeouts |
| Tenant IDs in logs/metrics | Missing tenant hashing | Add `_tenant_hash()` method |
| Cache stats appearing twice | _do_get_stats() includes cache metrics | Remove cache_hits/cache_misses |
| Batch operations failing randomly | Configurable batch mode | Hardcode one mode, remove flag |
| Streaming behaves differently between calls | Configurable streaming pattern | Hardcode one pattern, remove flag |

---

## 14. Decision Matrix Quick Reference

| Decision | Options | How to Choose | Must Not Do |
|----------|---------|---------------|-------------|
| **Batch failure mode** | Collect vs Fail-fast | Does provider return per-item status? | Make configurable |
| **Streaming pattern** | Single/Progressive/Multi | Component + provider capabilities | Make configurable |
| **Normalization** | `normalizes_at_source=True/False` | Does provider normalize vectors? | Normalize twice |
| **Idempotency** | True/False | Set based on implementation, not wishful thinking | Claim True without implementing |
| **Delete semantics** | Count actual deletions | Always | Count attempts |
| **Namespace handling** | Enforce authority | Always | Silently correct |
| **Filter validation** | Strict with `supported` list | Always | Silently ignore |
| **Include vectors** | `[]` when False | Always | Return `null` or omit |
| **Token counting** | Accurate (tiktoken/provider) | If `supports_token_counting=True` | Use approximations |
| **Tenant IDs** | Hash before logging | Always | Log raw tenant IDs |
| **Deadlines** | Propagate to provider | Always | Ignore `ctx.remaining_ms()` |
| **Cache stats** | Adapter metrics only | Always | Include cache_hits/cache_misses |

---

## Appendix A: Provider Assessment Worksheet

Use this worksheet to document your provider's capabilities before writing your adapter. This becomes your adapter's architecture decision record.

### Provider Information

| Field | Your Answer |
|-------|-------------|
| Provider Name | |
| API Version | |
| Base URL | |
| Authentication Method | |
| SDK Available? (Language) | |

### Capabilities Checklist

| Capability | Provider Support | Notes |
|------------|------------------|-------|
| **Batch API?** | ☐ Yes ☐ No | Max batch size: _____ |
| **Per-item status in batch?** | ☐ Yes ☐ No | |
| **Streaming API?** | ☐ Yes ☐ No | Protocol: _____ |
| **Token counting API?** | ☐ Yes ☐ No | |
| **Tokenizer available locally?** | ☐ Yes ☐ No | |
| **Returns normalized vectors?** | ☐ Yes ☐ No | |
| **Supports idempotency keys?** | ☐ Yes ☐ No | |
| **Timeout support in SDK?** | ☐ Yes ☐ No | |
| **Multi-tenant?** | ☐ Yes ☐ No | Tenant identifier: _____ |

### Error Mapping Worksheet

| Provider Error | HTTP Status | Corpus Error | Required Details | Source (Header/Body) |
|----------------|-------------|--------------|------------------|---------------------|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

### Rate Limit Pattern

| Question | Answer |
|----------|--------|
| How are rate limits communicated? | ☐ Headers ☐ Response body ☐ Both |
| Header names (if applicable) | |
| Retry-After format | ☐ Seconds ☐ Milliseconds ☐ Timestamp |
| Multiple scopes? (api/model/namespace) | |
| Sample rate limit response | |

### Batch Semantics

| Question | Answer |
|----------|--------|
| Does batch API return per-item status? | ☐ Yes ☐ No |
| Can you get 98 successes + 2 errors? | ☐ Yes ☐ No |
| What happens on validation error? | ☐ Whole batch fails ☐ Per-item errors |
| Does order matter? | ☐ Yes ☐ No |

### Tenant Handling

| Question | Answer |
|----------|--------|
| How is tenant identified in requests? | |
| Does provider need raw tenant ID? | ☐ Yes ☐ No |
| Any PII concerns with tenant IDs? | ☐ Yes ☐ No |
| Default tenant for anonymous requests? | |

### Deadline Support

| Question | Answer |
|----------|--------|
| Does SDK support timeouts? | ☐ Yes ☐ No |
| Timeout format | ☐ Seconds ☐ Milliseconds ☐ Deadline |
| Connection timeout separate? | ☐ Yes ☐ No |

### Idempotency Decision

| Question | Answer |
|----------|--------|
| Does provider support idempotency keys? | ☐ Yes ☐ No |
| Will you implement idempotency? | ☐ Yes ☐ No |
| If yes, storage backend | ☐ Redis ☐ DynamoDB ☐ In-memory (dev only) |

### Completed By

| Field | |
|-------|-|
| Name | |
| Date | |
| Adapter Type | ☐ LLM ☐ Embedding ☐ Vector ☐ Graph |

---

**Maintainers:** Corpus SDK Team  
**See also:** [Quick Start](./QUICK_START.md) | [Implementation Guide](./IMPLEMENTATION.md)