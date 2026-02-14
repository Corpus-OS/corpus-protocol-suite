# Corpus OS Implementation Guide

## ⚠️ Implementation Ground Rules (READ FIRST)

### Document role
- This guide is **non-normative implementation guidance** for adapter authors.
- **Spec / Protocols / Schema are canonical.** If any conflict exists, those documents take precedence.
- This guide explains **how to implement behavior inside `_do_*` hooks**, not protocol rules.

### Modes & deadlines

| Mode        | Base enforces deadlines | Adapter should propagate `ctx.remaining_ms()` to provider |
|-------------|-------------------------|-----------------------------------------------------------|
| thin        | No                      | Yes (recommended best practice)                           |
| standalone  | Yes                     | Yes (required)                                            |

- Always pass remaining budget to upstream/provider timeouts when available.

### Streaming invariants (LLM & Graph)
- Emit **exactly one** terminal (final/end or error)
- **No data after terminal**
- Tool calls **only in final**
- Do not emit multiple finals or partial terminal states

### Adapter determinism
- Adapters must not introduce additional randomness or non-deterministic behavior beyond the underlying provider/model
- Do not add RNG, probabilistic failures, artificial jitter, or test-only randomness inside adapters
- Given the same inputs and provider behavior, the adapter should behave consistently

### Schema fidelity
- All responses and examples must **match canonical Schema exactly**
- Use correct field names/types
- Use **`[]` not `null`** for empty collections
- Do not invent ad-hoc dict shapes when typed objects are defined (e.g., failures)

### Capabilities caching
- Capabilities may be cached with a short TTL
- Cached values are acceptable while fresh
- Refresh periodically; avoid long-lived or permanently stale capability data

### Consistency rule
- Do not duplicate or reinterpret protocol rules here; when unsure, defer to the canonical docs

---

**TABLE OF CONTENTS**

[1. PURPOSE & SCOPE](#1-purpose--scope)

[2. SYSTEM LAYOUT](#2-system-layout)

&nbsp;&nbsp;&nbsp;&nbsp;[2.1 Base Classes (Provided)](#21-base-classes-provided)

&nbsp;&nbsp;&nbsp;&nbsp;[2.2 What You Implement (_do_* hooks)](#22-what-you-implement-_do_-hooks)

&nbsp;&nbsp;&nbsp;&nbsp;[2.3 What You NEVER Implement (Public Methods)](#23-what-you-never-implement-public-methods)

&nbsp;&nbsp;&nbsp;&nbsp;[2.4 What You NEVER Call](#24-what-you-never-call)

[3. CONTEXT & IDENTITY (PRODUCTION RULES)](#3-context--identity-production-rules)

&nbsp;&nbsp;&nbsp;&nbsp;[3.1 OperationContext Structure](#31-operationcontext-structure)

&nbsp;&nbsp;&nbsp;&nbsp;[3.2 Deadline Propagation (MANDATORY)](#32-deadline-propagation-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[3.3 Tenant Hashing (MANDATORY)](#33-tenant-hashing-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[3.4 CRITICAL: Context Attributes Are TEST ONLY](#34-critical-context-attributes-are-test-only)

[4. ERROR TAXONOMY & MAPPING](#4-error-taxonomy--mapping)

&nbsp;&nbsp;&nbsp;&nbsp;[4.1 Canonical Error Hierarchy](#41-canonical-error-hierarchy)

&nbsp;&nbsp;&nbsp;&nbsp;[4.2 Provider Error Mapping (MANDATORY)](#42-provider-error-mapping-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[4.3 Error Detail Schemas (PER ERROR TYPE) (MANDATORY)](#43-error-detail-schemas-per-error-type-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[4.4 Retry Semantics (MANDATORY)](#44-retry-semantics-mandatory)

[5. MODES: THIN vs STANDALONE](#5-modes-thin-vs-standalone)

[6. DEADLINES & CANCELLATION](#6-deadlines--cancellation)

[7. LLM ADAPTER IMPLEMENTATION REQUIREMENTS](#7-llm-adapter-implementation-requirements)

&nbsp;&nbsp;&nbsp;&nbsp;[7.1 Required Methods](#71-required-methods)

&nbsp;&nbsp;&nbsp;&nbsp;[7.2 Shared Planning Path (COMPLETE and STREAM) (MANDATORY)](#72-shared-planning-path-complete-and-stream-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[7.3 Tool Calling (Validation and Accounting) (MANDATORY)](#73-tool-calling-validation-and-accounting-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[7.4 Token Counting and Usage Accounting (TOOL CALLS) (MANDATORY)](#74-token-counting-and-usage-accounting-tool-calls-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[7.5 Stop Sequences (FIRST Occurrence Rule) (MANDATORY)](#75-stop-sequences-first-occurrence-rule-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[7.6 Streaming Rules (Tool Calls) (MANDATORY)](#76-streaming-rules-tool-calls-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[7.7 Capabilities Enforcement (Operation Coupling) (MANDATORY)](#77-capabilities-enforcement-operation-coupling-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[7.8 Role Validation (Permissive, Not Restrictive) (MANDATORY)](#78-role-validation-permissive-not-restrictive-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[7.9 Complete LLM Example (Production Ready)](#79-complete-llm-example-production-ready)

[8. EMBEDDING ADAPTER IMPLEMENTATION REQUIREMENTS](#8-embedding-adapter-implementation-requirements)

&nbsp;&nbsp;&nbsp;&nbsp;[8.1 Required Methods](#81-required-methods)

&nbsp;&nbsp;&nbsp;&nbsp;[8.2 Validation Placement (_do_embed MUST Validate) (MANDATORY)](#82-validation-placement-_do_embed-must-validate-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[8.3 Batch Failure Mode (CHOOSE ONE) (MANDATORY)](#83-batch-failure-mode-choose-one-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[8.4 Batch Failure Mode (CONFIGURABILITY FORBIDDEN) (MANDATORY)](#84-batch-failure-mode-configurability-forbidden-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[8.5 Truncation and Normalization (Base Owns, You Report)](#85-truncation-and-normalization-base-owns-you-report)

&nbsp;&nbsp;&nbsp;&nbsp;[8.6 Cache Stats Ownership (CRITICAL BOUNDARY) (MANDATORY)](#86-cache-stats-ownership-critical-boundary-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[8.7 Streaming Pattern (CHOOSE ONE) (MANDATORY)](#87-streaming-pattern-choose-one-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[8.8 Capabilities (NO RUNTIME CONFIGURATION) (MANDATORY)](#88-capabilities-no-runtime-configuration-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[8.9 Token Counting (NO APPROXIMATIONS) (MANDATORY)](#89-token-counting-no-approximations-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[8.10 Complete Embedding Example (Production Ready)](#810-complete-embedding-example-production-ready)

[9. VECTOR ADAPTER IMPLEMENTATION REQUIREMENTS](#9-vector-adapter-implementation-requirements)

&nbsp;&nbsp;&nbsp;&nbsp;[9.1 Required Methods](#91-required-methods)

&nbsp;&nbsp;&nbsp;&nbsp;[9.2 Namespace Authority (Spec.namespace is Authoritative) (MANDATORY)](#92-namespace-authority-specnamespace-is-authoritative-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[9.3 Include Vectors Contract ([] NOT null) (MANDATORY)](#93-include-vectors-contract--not-null-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[9.4 Filter Dialect Validation (Strict, No Silent Ignore) (MANDATORY)](#94-filter-dialect-validation-strict-no-silent-ignore-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[9.5 Filter Operator Error Details (CANONICAL SHAPE) (MANDATORY)](#95-filter-operator-error-details-canonical-shape-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[9.6 Batch Query Atomicity (All or Nothing) (MANDATORY)](#96-batch-query-atomicity-all-or-nothing-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[9.7 Delete Idempotency (No Error on Missing) (MANDATORY)](#97-delete-idempotency-no-error-on-missing-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[9.8 Delete Parameter Rule (IDs XOR Filter) (MANDATORY)](#98-delete-parameter-rule-ids-xor-filter-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[9.9 Distance Metric Strings (EXACT VALUES) (MANDATORY)](#99-distance-metric-strings-exact-values-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[9.10 Suggested Batch Reduction (Percentage Semantics) (MANDATORY)](#910-suggested-batch-reduction-percentage-semantics-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[9.11 IndexNotReady (Retry Semantics) (MANDATORY)](#911-indexnotready-retry-semantics-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[9.12 Namespace Mismatch Error Details (CANONICAL SHAPE) (MANDATORY)](#912-namespace-mismatch-error-details-canonical-shape-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[9.13 Dimension Mismatch Error Details (CANONICAL SHAPE) (MANDATORY)](#913-dimension-mismatch-error-details-canonical-shape-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[9.14 Health Response (Namespace Status) (MANDATORY)](#914-health-response-namespace-status-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[9.15 Complete Vector Example (Production Ready)](#915-complete-vector-example-production-ready)

&nbsp;&nbsp;&nbsp;&nbsp;[9.16 Technical Reference: Similarity & Normalization (MANDATORY)](#916-technical-reference-similarity--normalization-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[9.16.1 Cosine Similarity Canonical Form](#9161-cosine-similarity-canonical-form)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[9.16.2 L2 Normalization Contract](#9162-l2-normalization-contract)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[9.16.3 Euclidean Distance Canonical Form](#9163-euclidean-distance-canonical-form)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[9.16.4 Dot Product Canonical Form](#9164-dot-product-canonical-form)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[9.16.5 Precision Requirements](#9165-precision-requirements)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[9.16.6 Conformance Test Vectors](#9166-conformance-test-vectors)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[9.16.7 Common Failure Patterns](#9167-common-failure-patterns)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[9.16.8 Zero Vector Edge Cases (Production Safety)](#9168-zero-vector-edge-cases-production-safety)

[10. GRAPH ADAPTER IMPLEMENTATION REQUIREMENTS](#10-graph-adapter-implementation-requirements)

&nbsp;&nbsp;&nbsp;&nbsp;[10.1 Required Methods](#101-required-methods)

&nbsp;&nbsp;&nbsp;&nbsp;[10.2 Batch/Transaction Result Envelope ({ok, result}) (MANDATORY)](#102-batchtransaction-result-envelope-ok-result-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[10.3 Shared Op Executor (Single Kernel for Batch + Transaction) (MANDATORY)](#103-shared-op-executor-single-kernel-for-batch--transaction-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[10.4 Dialect Validation (TWO Layers) (MANDATORY)](#104-dialect-validation-two-layers-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[10.5 Delete Idempotency (No Error on Missing) (MANDATORY)](#105-delete-idempotency-no-error-on-missing-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[10.6 Bulk Vertices Pagination (Cursor Contract) (MANDATORY)](#106-bulk-vertices-pagination-cursor-contract-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[10.7 Traversal Result Shape (Nodes, Edges, Paths) (MANDATORY)](#107-traversal-result-shape-nodes-edges-paths-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[10.8 Capabilities Enforcement (Operation Coupling) (MANDATORY)](#108-capabilities-enforcement-operation-coupling-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[10.9 Capabilities (NO RUNTIME CONFIGURATION) (MANDATORY)](#109-capabilities-no-runtime-configuration-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[10.10 Complete Graph Example (Production Ready)](#1010-complete-graph-example-production-ready)

[11. CACHE OWNERSHIP BOUNDARY (CRITICAL)](#11-cache-ownership-boundary-critical)

&nbsp;&nbsp;&nbsp;&nbsp;[11.1 Embedding Stats (NO Cache Metrics) (MANDATORY)](#111-embedding-stats-no-cache-metrics-mandatory)

&nbsp;&nbsp;&nbsp;&nbsp;[11.2 Capabilities Caching (Allowed, With Rules) (MANDATORY)](#112-capabilities-caching-allowed-with-rules-mandatory)

[12. BATCH FAILURE MODE (DECISION MATRIX)](#12-batch-failure-mode-decision-matrix)

[13. STREAMING PATTERN (DECISION MATRIX)](#13-streaming-pattern-decision-matrix)

[14. PRODUCTION HARDENING (REMOVE ALL MOCK ONLY CODE)](#14-production-hardening-remove-all-mock-only-code)

&nbsp;&nbsp;&nbsp;&nbsp;[14.1 Patterns to DELETE Entirely](#141-patterns-to-delete-entirely)

&nbsp;&nbsp;&nbsp;&nbsp;[14.2 Patterns to TRANSFORM](#142-patterns-to-transform)

[15. PER DOMAIN IMPLEMENTATION CHECKLISTS](#15-per-domain-implementation-checklists)

&nbsp;&nbsp;&nbsp;&nbsp;[15.1 LLM Adapter Checklist](#151-llm-adapter-checklist)

&nbsp;&nbsp;&nbsp;&nbsp;[15.2 Embedding Adapter Checklist](#152-embedding-adapter-checklist)

&nbsp;&nbsp;&nbsp;&nbsp;[15.3 Vector Adapter Checklist](#153-vector-adapter-checklist)

&nbsp;&nbsp;&nbsp;&nbsp;[15.4 Graph Adapter Checklist](#154-graph-adapter-checklist)

&nbsp;&nbsp;&nbsp;&nbsp;[16. COMMON PITFALLS: CONFORMANCE FAILURES](#16-common-pitfalls-conformance-failures)

---

## 1. PURPOSE & SCOPE

This document specifies the mandatory implementation requirements for building production adapters that pass Corpus Protocol conformance suites.

You are a Corpus OS Implementer. Your job is to write `_do_*` methods that map your provider's API to the Corpus base classes.

These requirements are DERIVED FROM WORKING MOCKS that pass 100% conformance. Every rule below exists because a mock implements it and conformance tests enforce it.

If you skip any requirement below, YOUR ADAPTER WILL FAIL CONFORMANCE.

**33 OPERATIONS ACROSS 4 DOMAINS:**
- **Graph (13 ops):** capabilities, upsert_nodes, upsert_edges, delete_nodes, delete_edges, query, stream_query, bulk_vertices, batch, transaction, traversal, get_schema, health
- **LLM (5 ops):** capabilities, complete, stream, count_tokens, health
- **Vector (8 ops):** capabilities, query, batch_query, upsert, delete, create_namespace, delete_namespace, health
- **Embedding (7 ops):** capabilities, embed, embed_batch, stream_embed, count_tokens, get_stats, health

---

## 2. SYSTEM LAYOUT

### 2.1 Base Classes (Provided)

You receive these base classes. DO NOT modify them:

```python
from corpus_sdk.llm.llm_base import BaseLLMAdapter
from corpus_sdk.embedding.embedding_base import BaseEmbeddingAdapter
from corpus_sdk.vector.vector_base import BaseVectorAdapter
from corpus_sdk.graph.graph_base import BaseGraphAdapter
```

**What base classes do FOR you:**
- Public methods (`capabilities()`, `complete()`, `query()`, etc.)
- Wire envelope parsing and validation
- Deadline enforcement via `_apply_deadline()`
- Circuit breaker integration
- Rate limiting
- Cache management and key generation
- Metrics observation
- Error → canonical envelope conversion
- Streaming gate with deadline checks
- Truncation (Embedding)
- Normalization (Embedding)
- Per-item fallback for batch when `NotSupported` raised (Embedding)

**What base classes do NOT do for you:**
- Validate provider-specific constraints (model exists, dimensions match, filter operators supported)
- Enforce namespace authority (Vector/Graph)
- Implement idempotent delete semantics (no error on missing)
- Choose your batch failure mode (collect vs fail-fast)
- Define your streaming pattern (single/progressive/multi-vector)
- Remove mock-only code (`ctx.attrs`, RNG, latency simulation)
- Enforce capability-operation coupling
- Handle tool call token accounting
- Define error detail schemas per error type
- Define response shape contracts per operation
- Enforce mathematical precision for similarity metrics

**YOU implement these. This document tells you how.**

---

### 2.2 What You Implement (_do_* hooks)

**In your subclass, you implement the `_do_*` hook methods. The base class public methods call these hooks.**

| Component | Required `_do_*` Methods (YOU IMPLEMENT THESE) |
|-----------|------------------------------------------------|
| **LLM** | `_do_capabilities()`, `_do_complete()`, `_do_stream()`, `_do_count_tokens()`, `_do_health()` |
| **Embedding** | `_do_capabilities()`, `_do_embed()`, `_do_embed_batch()`, `_do_stream_embed()`, `_do_count_tokens()` (if supported), `_do_get_stats()`, `_do_health()` |
| **Vector** | `_do_capabilities()`, `_do_query()`, `_do_batch_query()`, `_do_upsert()`, `_do_delete()`, `_do_create_namespace()`, `_do_delete_namespace()`, `_do_health()` |
| **Graph** | `_do_capabilities()`, `_do_query()`, `_do_stream_query()`, `_do_bulk_vertices()`, `_do_batch()`, `_do_transaction()` (if supported), `_do_traversal()` (if supported), `_do_get_schema()` (if supported), `_do_upsert_nodes()`, `_do_upsert_edges()`, `_do_delete_nodes()`, `_do_delete_edges()`, `_do_health()` |

**Example pattern:**

```python
from corpus_sdk.llm.llm_base import BaseLLMAdapter

class MyLLMAdapter(BaseLLMAdapter):
    async def _do_capabilities(self):    # ✅ YOU implement this
        ...
    
    async def _do_complete(self, ...):   # ✅ YOU implement this
        ...
    
    # ❌ NEVER implement the public method
    # async def capabilities(self): ...  # WRONG - base owns this
```

---

### 2.3 What You NEVER Implement (Public Methods)

```python
# ❌ NEVER IMPLEMENT THESE PUBLIC METHODS: BASE OWNS THEM
def capabilities(self): ...  # Base implements, calls your _do_capabilities
def complete(self): ...      # Base implements, calls your _do_complete
def query(self): ...         # Base implements, calls your _do_query
def upsert(self): ...        # Base implements, calls your _do_upsert
def embed(self): ...         # Base implements, calls your _do_embed
def stream(self): ...        # Base implements, calls your _do_stream

# ✅ CORRECT: Implement the _do_* hooks instead
async def _do_capabilities(self): ...
async def _do_complete(self, ...): ...
async def _do_query(self, ...): ...
async def _do_upsert(self, ...): ...
async def _do_embed(self, ...): ...
async def _do_stream(self, ...): ...
```

**RULE:** The base classes implement the public protocol methods. You implement the `_do_*` hooks that the base calls. Never override the public methods.

---

### 2.4 What You NEVER Call

```python
# ❌ NEVER CALL PUBLIC METHODS FROM _do_* METHODS
await self.capabilities()    # WRONG: causes deadlock/recursion
await self.complete()        # WRONG: calls back into your _do_complete
await self.query()           # WRONG: calls back into your _do_query

# ✅ CORRECT: Call the _do_* hooks directly
await self._do_capabilities()  # CORRECT: direct hook call
await self._do_complete(...)   # CORRECT: direct hook call
await self._do_query(...)      # CORRECT: direct hook call

# ❌ NEVER IMPLEMENT YOUR OWN CACHE
self._my_cache = {}  # WRONG: base owns caching

# ❌ NEVER OVERRIDE BASE STATS WITH CACHE METRICS
def _do_get_stats(self, ctx=None):
    return {
        "total_requests": ...,
        "cache_hits": ...,  # ❌ WRONG: base owns cache stats
        "cache_misses": ... # ❌ WRONG: base owns cache stats
    }
```

---

## 3. CONTEXT & IDENTITY: PRODUCTION RULES

### 3.1 OperationContext Structure

Every `_do_*` method receives `ctx: Optional[OperationContext]`. **Import from domain base classes:**

```python
from corpus_sdk.llm.llm_base import OperationContext as LLMOperationContext
from corpus_sdk.embedding.embedding_base import OperationContext as EmbeddingOperationContext
from corpus_sdk.vector.vector_base import OperationContext as VectorOperationContext
from corpus_sdk.graph.graph_base import OperationContext as GraphOperationContext
```

**Note:** These are identical dataclasses; alias them to avoid name collisions.

```python
@dataclass(frozen=True)
class OperationContext:
    request_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    deadline_ms: Optional[int] = None  # epoch milliseconds
    traceparent: Optional[str] = None  # W3C trace context
    tenant: Optional[str] = None
    attrs: Mapping[str, Any] = field(default_factory=dict)
    
    def remaining_ms(self) -> Optional[int]:
        """Return remaining milliseconds until deadline, or None if no deadline."""
        if self.deadline_ms is None:
            return None
        now_ms = int(time.time() * 1000)
        return max(0, self.deadline_ms - now_ms)
```

### 3.2 Deadline Propagation: MANDATORY

```python
async def _do_embed(self, spec, *, ctx=None):
    timeout_s = None
    if ctx is not None:
        remaining_ms = ctx.remaining_ms()
        if remaining_ms is not None and remaining_ms > 0:
            timeout_s = remaining_ms / 1000.0
        elif remaining_ms is not None and remaining_ms <= 0:
            # Base already raises DeadlineExceeded preflight, but protect defensively
            from corpus_sdk.embedding.embedding_base import DeadlineExceeded
            raise DeadlineExceeded("deadline already exceeded")
    
    # Pass timeout_s to provider SDK
    response = await self._client.call(timeout=timeout_s)
```

**RULE:** Always convert `ctx.remaining_ms()` to provider timeouts. Never call provider if deadline already expired.

### 3.3 Tenant Hashing: MANDATORY

```python
def _tenant_hash(self, tenant: Optional[str]) -> Optional[str]:
    """Create privacy-preserving hash of tenant identifier."""
    if not tenant:
        return None
    return hashlib.sha256(tenant.encode()).hexdigest()[:12]

# NEVER log or emit raw tenant IDs
tenant_id = ctx.tenant if ctx else None
metric_tenant = self._tenant_hash(tenant_id) if tenant_id else "global"

# Use hashed tenant in metrics, cache keys, logs
cache_key = f"embedding:{metric_tenant}:{model}:{text_hash}"
```

**RULE:** Tenant IDs in logs, metrics, and cache keys MUST be hashed via `_tenant_hash()`. Raw tenant IDs are PII and forbidden.

### 3.4 ⚠️ CRITICAL: Context Attributes Are TEST ONLY

**THE MOCKS USE THIS PATTERN. YOU MUST NOT. REMOVE ALL OF THESE:**

```python
# ❌ NEVER DO THIS IN PRODUCTION
ctx.attrs.get("simulate_error")  # Mock error injection
ctx.attrs.get("fail")            # Mock forced failures  
ctx.attrs.get("sleep_ms")        # Mock artificial latency
ctx.attrs.get("health") == "degraded"  # Mock health forcing

# ❌ NEVER HAVE THESE FIELDS IN YOUR ADAPTER
self.failure_rate = 0.0  # No probabilistic failures
self._rng = random.Random()  # No RNG in production adapters
self.simulate_latency = False  # No latency simulation flags

# ✅ ALLOWED: Client-provided extensions
ctx.attrs.get("client_version")  # OK
ctx.attrs.get("source")          # OK
ctx.attrs.get("workflow_id")     # OK

# ❌ FORBIDDEN: Mock error injection
ctx.attrs.get("simulate_error")  # NO
ctx.attrs.get("fail")            # NO
ctx.attrs.get("sleep_ms")        # NO
ctx.attrs.get("health")          # NO - use real health checks
```

**PRODUCTION ADAPTERS MUST:**
- Zero references to `ctx.attrs` for operational logic
- Zero RNG-based failure injection
- Zero artificial sleep/simulation code
- Zero probabilistic behavior (`failure_rate`, `random_failure()`)
- Zero test-only knobs in constructor

**CONFORMANCE FAILURE:** Any production adapter containing these patterns will FAIL conformance.

---

## 4. ERROR TAXONOMY & MAPPING

### 4.1 Canonical Error Hierarchy

**Import errors directly from their respective domain base classes:**

```python
# LLM errors - import from llm_base
from corpus_sdk.llm.llm_base import (
    LLMAdapterError, BadRequest, AuthError, ResourceExhausted,
    TransientNetwork, Unavailable, NotSupported, DeadlineExceeded,
    ModelOverloaded
)

# Embedding errors - import from embedding_base  
from corpus_sdk.embedding.embedding_base import (
    EmbeddingAdapterError, BadRequest, AuthError, ResourceExhausted,
    TransientNetwork, Unavailable, NotSupported, DeadlineExceeded,
    TextTooLong, ModelNotAvailable
)

# Vector errors - import from vector_base
from corpus_sdk.vector.vector_base import (
    VectorAdapterError, BadRequest, AuthError, ResourceExhausted,
    TransientNetwork, Unavailable, NotSupported, DeadlineExceeded,
    DimensionMismatch, IndexNotReady
)

# Graph errors - import from graph_base
from corpus_sdk.graph.graph_base import (
    GraphAdapterError, BadRequest, AuthError, ResourceExhausted,
    TransientNetwork, Unavailable, NotSupported, DeadlineExceeded
)
```

**Note:** There is no shared `corpus_sdk.exceptions` module. Each domain defines its own error hierarchy in its base file. This is intentional — domains evolve independently.

### 4.2 Provider Error Mapping: MANDATORY

**Implement error mapping as a method on your adapter class:**

```python
from corpus_sdk.llm.llm_base import (
    BadRequest, AuthError, ResourceExhausted,
    TransientNetwork, Unavailable, ModelOverloaded,
    DeadlineExceeded, NotSupported
)

class MyLLMAdapter(BaseLLMAdapter):
    def _map_provider_error(self, e: Exception) -> LLMAdapterError:
        """Map provider-specific errors to canonical Corpus errors."""
        
        # Rate limits
        if isinstance(e, ProviderRateLimitError):
            return ResourceExhausted(
                "Rate limit exceeded",
                retry_after_ms=e.retry_after or 5000,
                resource_scope="rate_limit",
                details={"provider_error": str(e)}
            )
        
        # Authentication
        if isinstance(e, ProviderAuthError):
            return AuthError("Invalid credentials")
        
        # Invalid requests
        if isinstance(e, ProviderInvalidRequest):
            return BadRequest(str(e))
        
        # Model availability
        if isinstance(e, ProviderModelNotFound):
            return ModelNotAvailable(
                f"Model '{e.model}' not available",
                details={"requested_model": e.model}
            )
        
        # Text too long (embedding)
        if isinstance(e, ProviderTextTooLong):
            return TextTooLong(
                f"Text exceeds maximum length of {e.max_length}",
                details={
                    "max_length": e.max_length,
                    "actual_length": e.actual_length
                }
            )
        
        # Dimension mismatch (vector)
        if isinstance(e, ProviderDimensionError):
            return DimensionMismatch(
                f"Vector dimension {e.actual} does not match index dimension {e.expected}",
                details={
                    "expected": e.expected,
                    "actual": e.actual,
                    "namespace": e.namespace
                }
            )
        
        # Index not ready (vector)
        if isinstance(e, ProviderIndexNotReady):
            return IndexNotReady(
                "Index not ready for queries",
                retry_after_ms=e.retry_after or 500,
                details={"namespace": e.namespace}
            )
        
        # Timeouts
        if isinstance(e, ProviderTimeout):
            return TransientNetwork("Upstream timeout")
        
        # Service unavailable
        if isinstance(e, ProviderServerError):
            return Unavailable("Provider unavailable")
        
        # Catch-all
        return Unavailable(f"Unknown provider error: {type(e).__name__}")
```

**RULE:** Every provider error MUST map to the same canonical error type every time. No conditional mapping based on context. Import error classes directly from the domain base.

### 4.3 Error Detail Schemas: PER ERROR TYPE: MANDATORY

**DimensionMismatch: REQUIRED fields:**

```python
from corpus_sdk.vector.vector_base import DimensionMismatch

raise DimensionMismatch(
    f"Vector dimension {actual} does not match namespace {expected}",
    details={
        "expected": 384,           # REQUIRED
        "actual": 512,            # REQUIRED
        "namespace": "docs",      # REQUIRED
        "vector_id": "vec_123",   # REQUIRED if available
        "index": 3               # REQUIRED for batch operations
    }
)
```

**NamespaceMismatch: REQUIRED fields:**

```python
from corpus_sdk.vector.vector_base import BadRequest

raise BadRequest(
    "vector.namespace must match UpsertSpec.namespace",
    details={
        "spec_namespace": "default",      # REQUIRED
        "vector_namespace": "other",      # REQUIRED
        "vector_id": "vec_123",          # REQUIRED
        "index": 2                       # REQUIRED
    }
)
```

**FilterValidation: REQUIRED fields:**

```python
raise BadRequest(
    "unsupported filter operator",
    details={
        "operator": "$regex",        # REQUIRED
        "field": "title",           # REQUIRED
        "supported": ["$in"],       # REQUIRED: array of supported operators
        "namespace": "docs"         # REQUIRED
    }
)
```

**ModelNotAvailable: REQUIRED fields:**

```python
from corpus_sdk.embedding.embedding_base import ModelNotAvailable

raise ModelNotAvailable(
    f"Model '{requested}' is not supported",
    details={
        "requested_model": "gpt-5",           # REQUIRED
        "supported_models": ["gpt-4", "gpt-3.5"]  # REQUIRED
    }
)
```

**IndexNotReady: REQUIRED fields:**

```python
from corpus_sdk.vector.vector_base import IndexNotReady

raise IndexNotReady(
    "index not ready (no data in namespace)",
    retry_after_ms=500,  # REQUIRED: MUST provide retry hint
    details={
        "namespace": "docs"  # REQUIRED
    }
)
```

**TextTooLong: REQUIRED fields:**

```python
from corpus_sdk.embedding.embedding_base import TextTooLong

raise TextTooLong(
    f"Text length {actual} exceeds maximum of {max_len}",
    details={
        "max_length": 8192,    # REQUIRED
        "actual_length": 15000  # REQUIRED
    }
)
```

**RULE:** Every error MUST include ALL required detail fields shown above. Conformance tests verify these exact field names and presence.

### 4.4 Retry Semantics: MANDATORY

| Error Class | Retryable | Condition |
|-------------|-----------|-----------|
| ResourceExhausted | YES | Honor `retry_after_ms` |
| TransientNetwork | YES | Exponential backoff + jitter |
| Unavailable | YES | Circuit breaker recommended |
| IndexNotReady | YES | Honor `retry_after_ms` (default 500) |
| DeadlineExceeded | CONDITIONAL | Only with extended deadline or reduced workload |
| BadRequest | NO | Fix request |
| AuthError | NO | Refresh credentials |
| NotSupported | NO | Use different feature/model |
| DimensionMismatch | NO | Fix vector dimension |
| ModelNotAvailable | NO | Use different model |
| TextTooLong | NO | Truncate or chunk |

```python
# RETRY IMPLEMENTATION PATTERN
for attempt in range(max_retries):
    try:
        return await self._client.call()
    except ResourceExhausted as e:
        wait_ms = e.retry_after_ms or (backoff * (2 ** attempt) * 100)
        await asyncio.sleep(wait_ms / 1000)
    except TransientNetwork:
        wait_ms = min((2 ** attempt) * 100, 10000) * random.uniform(0.5, 1.5)
        await asyncio.sleep(wait_ms / 1000)
```

---

## 5. MODES: THIN vs STANDALONE

Every base adapter supports two modes:

- **`mode="thin"` (default)**
  - For use under an external control plane
  - Deadline policy = no-op
  - Circuit breaker = no-op
  - Rate limiter = no-op
  - Cache = no-op

- **`mode="standalone"`**
  - For direct use, demos, and light production
  - Enforces deadlines
  - Uses per-process circuit breaker
  - Uses token-bucket rate limiter
  - Uses in-memory TTL cache
  - Logs warning if running standalone with `NoopMetrics`

```python
adapter = MyRealAdapter(
    mode="standalone",
    metrics=my_metrics_sink,
    cache=InMemoryTTLCache(default_ttl_s=60),
    limiter=TokenBucketLimiter(rate=50, burst=100)
)
```

---

## 6. DEADLINES & CANCELLATION

### 6.1 Preflight Check

Base classes call `_fail_if_expired(ctx)` before your `_do_*` method:
- If `ctx.deadline_ms` is set and `ctx.remaining_ms() <= 0`
- → raises `DeadlineExceeded("deadline already exceeded")`

**You do not need to implement this. Base does it.**

### 6.2 Deadline Propagation: YOU MUST DO THIS

```python
async def _do_complete(self, request, *, ctx=None):
    timeout_s = None
    if ctx is not None:
        rem = ctx.remaining_ms()
        if rem is not None and rem > 0:
            timeout_s = rem / 1000.0
    
    # Pass to provider SDK
    response = await self._client.complete(
        messages=request.messages,
        timeout=timeout_s  # CRITICAL
    )
```

**RULE:** Always pass `remaining_ms()` to provider timeouts. Never call provider without deadline propagation.

---

## 7. LLM ADAPTER: IMPLEMENTATION REQUIREMENTS

### 7.1 Required Methods

```python
from corpus_sdk.llm.llm_base import BaseLLMAdapter
from corpus_sdk.llm.llm_base import (
    LLMCapabilities, LLMCompletion, LLMChunk, TokenUsage,
    ToolCall, ToolCallFunction
)
from corpus_sdk.llm.llm_base import (
    BadRequest, NotSupported, ResourceExhausted, 
    ModelOverloaded, DeadlineExceeded, Unavailable
)

class MyLLMAdapter(BaseLLMAdapter):
    async def _do_capabilities(self) -> LLMCapabilities:
        """REQUIRED: Describe your models and features."""
        
    async def _do_complete(
        self, 
        messages: List[Mapping[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> LLMCompletion:
        """REQUIRED: Unary completion."""
        
    async def _do_stream(
        self,
        messages: List[Mapping[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[LLMChunk]:
        """REQUIRED: Streaming completion."""
        
    async def _do_count_tokens(
        self,
        text: str,
        model: Optional[str] = None,
        ctx: Optional[OperationContext] = None,
    ) -> int:
        """REQUIRED: Accurate token counting."""
        
    async def _do_health(self, *, ctx=None) -> Dict[str, Any]:
        """REQUIRED: Health check."""
```

### 7.2 Shared Planning Path: COMPLETE and STREAM: MANDATORY

```python
# ❌ WRONG: Separate logic causes mismatched output
async def _do_complete(self, request, *, ctx=None):
    return "Hello"  # Stream returns different text ❌

async def _do_stream(self, request, *, ctx=None):
    yield "H"
    yield "e"
    yield "llo"  # Different from complete ❌

# ✅ CORRECT: Single planning function for both
def _plan_response(self, request, ctx=None) -> Tuple[str, str, str, List[ToolCall]]:
    """Single source of truth for both complete() and stream()."""
    # Deterministic logic based ONLY on request + ctx
    # NO random, NO RNG, NO temperature simulation
    
    # Validate model
    if request.model not in self._supported_models:
        from corpus_sdk.llm.llm_base import ModelNotAvailable
        raise ModelNotAvailable(
            f"Model '{request.model}' is not supported",
            details={
                "requested_model": request.model,
                "supported_models": list(self._supported_models)
            }
        )
    
    # Build prompt
    prompt = self._build_prompt(request)
    
    # Generate completion (call provider)
    completion = self._call_provider_completion(request, ctx)
    
    # Apply stop sequences (FIRST occurrence rule)
    text = self._apply_stop_sequences(completion.text, request.stop_sequences)
    
    finish_reason = "length" if completion.finish_reason == "length" else "stop"
    
    return prompt, text, finish_reason, completion.tool_calls

async def _do_complete(self, request, *, ctx=None):
    prompt, text, finish_reason, tool_calls = self._plan_response(request, ctx)
    
    # Calculate usage
    usage = self._calculate_usage(prompt, text, tool_calls, request.model)
    
    return LLMCompletion(
        text=text,
        model=request.model,
        model_family=self._get_model_family(request.model),
        usage=usage,
        finish_reason=finish_reason,
        tool_calls=tool_calls
    )

async def _do_stream(self, request, *, ctx=None):
    prompt, text, finish_reason, tool_calls = self._plan_response(request, ctx)
    
    # Stream tokens from the SAME text
    if tool_calls:
        # Tool call streaming pattern
        yield LLMChunk(
            text="",
            is_final=False,
            model=request.model,
            usage_so_far=self._calculate_usage_so_far(prompt, "", None)
        )
        
        final_usage = self._calculate_usage(prompt, text, tool_calls, request.model)
        yield LLMChunk(
            text="",
            is_final=True,
            model=request.model,
            usage_so_far=final_usage,
            tool_calls=tool_calls
        )
    else:
        # Normal text streaming
        tokens = text.split()
        emitted = []
        for i, token in enumerate(tokens):
            emitted.append(token)
            partial = " ".join(emitted)
            usage = self._calculate_usage_so_far(prompt, partial, None)
            
            yield LLMChunk(
                text=token + (" " if i < len(tokens) - 1 else ""),
                is_final=False,
                model=request.model,
                usage_so_far=usage
            )
        
        final_usage = self._calculate_usage(prompt, text, None, request.model)
        yield LLMChunk(
            text="",
            is_final=True,
            model=request.model,
            usage_so_far=final_usage
        )
```

**RULE:** The text returned by `_do_complete()` MUST be identical to the concatenated text streamed by `_do_stream()` for the same request. Conformance tests verify this.

### 7.3 Tool Calling: Validation and Accounting: MANDATORY

```python
def _validate_tool_choice(self, tool_choice, tools):
    """Validate tool_choice against available tools."""
    from corpus_sdk.llm.llm_base import BadRequest
    
    if not tools:
        if tool_choice not in (None, "none", "auto"):
            raise BadRequest("tool_choice provided but no tools")
        return
    
    # Extract requested tool name
    requested = None
    if isinstance(tool_choice, dict):
        if "function" in tool_choice:
            requested = tool_choice["function"].get("name")
        elif "name" in tool_choice:
            requested = tool_choice["name"]
    elif isinstance(tool_choice, str):
        if tool_choice not in ("none", "auto", "required"):
            requested = tool_choice
    
    # Validate if specific tool requested
    if requested:
        tool_names = []
        for t in tools:
            if isinstance(t, dict):
                if "function" in t:
                    tool_names.append(t["function"].get("name"))
                elif "name" in t:
                    tool_names.append(t["name"])
        
        if requested not in tool_names:
            raise BadRequest(
                f"tool_choice requested unknown tool: {requested}",
                details={
                    "requested": requested,
                    "available": tool_names
                }
            )
```

**RULE:** You MUST validate `tool_choice` against available tools. Do not rely on provider to reject. Do not make strictness configurable: choose one behavior and document it.

### 7.4 Token Counting & Usage Accounting: TOOL CALLS: MANDATORY

```python
def _calculate_usage(self, prompt_text: str, completion_text: str, 
                    tool_calls: Optional[List[ToolCall]], model: str) -> TokenUsage:
    """Calculate token usage with tool call accounting."""
    from corpus_sdk.llm.llm_base import TokenUsage
    
    # Prompt tokens
    prompt_tokens = self._count_tokens_sync(prompt_text, model)
    
    # Completion tokens
    if tool_calls:
        # SYNTHESIZE tokens from tool call payload if provider returns 0
        tool_payload = json.dumps([
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            }
            for tc in tool_calls
        ], separators=(",", ":"))
        
        completion_tokens = self._count_tokens_sync(tool_payload, model)
        
        # ENSURE non-zero completion tokens for tool-calling turns
        if completion_tokens == 0:
            completion_tokens = 10  # Minimum viable
    else:
        completion_tokens = self._count_tokens_sync(completion_text, model)
    
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens
    )
```

**RULE:** If your provider returns tool calls with zero completion tokens, you MUST synthesize usage from the tool call payload. Tool-calling turns MUST report non-zero `completion_tokens`.

### 7.5 Stop Sequences: FIRST Occurrence Rule: MANDATORY

```python
def _apply_stop_sequences(self, text: str, stop_sequences: Optional[List[str]]) -> str:
    """Apply stop sequences: cut at FIRST occurrence of ANY stop sequence."""
    if not stop_sequences or not text:
        return text
    
    cut_position = len(text)
    for stop in stop_sequences:
        if not stop:
            continue
        pos = text.find(stop)
        if pos != -1 and pos < cut_position:
            cut_position = pos
    
    if cut_position < len(text):
        return text[:cut_position].rstrip()
    return text
```

**RULE:** Stop sequences MUST cut at the FIRST occurrence of ANY stop sequence. Do not cut at the last occurrence. Do not include the stop sequence itself in the output.

### 7.6 Streaming Rules: Tool Calls: MANDATORY

```python
async def _do_stream(self, request, *, ctx=None):
    """Stream with proper tool call semantics."""
    
    prompt_text, completion_text, finish_reason, tool_calls = self._plan_response(request, ctx)
    
    # TOOL CALL STREAMING PATTERN
    if tool_calls:
        # 1. At least one non-final chunk (empty text, no tool calls)
        yield LLMChunk(
            text="",
            is_final=False,
            model=request.model,
            usage_so_far=self._calculate_usage_so_far(prompt_text, "", None)
        )
        
        # 2. Final chunk with tool_calls populated
        final_usage = self._calculate_usage(prompt_text, completion_text, tool_calls, request.model)
        yield LLMChunk(
            text="",
            is_final=True,
            model=request.model,
            usage_so_far=final_usage,
            tool_calls=tool_calls  # REQUIRED in final chunk
        )
        return
    
    # NORMAL TEXT STREAMING
    tokens = completion_text.split()
    emitted = []
    
    for i, token in enumerate(tokens):
        emitted.append(token)
        partial = " ".join(emitted)
        usage = self._calculate_usage_so_far(prompt_text, partial, None)
        
        yield LLMChunk(
            text=token + (" " if i < len(tokens) - 1 else ""),
            is_final=False,
            model=request.model,
            usage_so_far=usage
        )
    
    # Final chunk with no text, is_final=True
    final_usage = self._calculate_usage(prompt_text, completion_text, None, request.model)
    yield LLMChunk(
        text="",
        is_final=True,
        model=request.model,
        usage_so_far=final_usage
    )
```

**RULE FOR TOOL CALL STREAMING:**
- MUST emit at least one non-final chunk (text="", tool_calls=[])
- MUST emit exactly one final chunk with `tool_calls` populated
- MUST NOT emit tool calls incrementally
- MUST NOT emit tool calls in non-final chunks

### 7.7 Capabilities Enforcement: Operation Coupling: MANDATORY

```python
async def _do_stream(self, request, *, ctx=None):
    """Enforce capabilities before proceeding."""
    from corpus_sdk.llm.llm_base import NotSupported
    
    caps = await self._do_capabilities()
    
    if not caps.supports_streaming:
        raise NotSupported(
            "streaming is not supported by this adapter",
            details={"capability": "supports_streaming"}
        )
    
    # Proceed with streaming implementation...
```

**RULE:** If `caps.supports_X = False`, `_do_X` MUST raise `NotSupported`. Do not silently no-op. Do not implement operations advertised as unsupported.

### 7.8 Role Validation: Permissive, Not Restrictive: MANDATORY

```python
# ✅ CORRECT: Permissive, allows future roles
ALLOWED_ROLES = {"system", "user", "assistant", "tool", "function", "developer"}

def _validate_roles(self, messages):
    """Validate roles: be permissive, not restrictive."""
    from corpus_sdk.llm.llm_base import BadRequest
    
    for msg in messages:
        role = msg.get("role")
        if role and role not in ALLOWED_ROLES:
            # Only reject roles the provider ACTUALLY doesn't support
            if not self._provider_supports_role(role):
                raise BadRequest(f"role '{role}' not supported by provider")
        # Otherwise, allow it: don't block protocol evolution
```

**RULE:** Do not hard-block roles not in your allowlist. Only reject roles your provider explicitly cannot handle. Be permissive to support future protocol extensions.

### 7.9 Complete LLM Example (Production Ready)

```python
from typing import AsyncIterator, Dict, Any, List, Optional, Tuple, Union
import asyncio
import json
import time
import hashlib
from corpus_sdk.llm.llm_base import BaseLLMAdapter
from corpus_sdk.llm.llm_base import (
    LLMCapabilities, LLMCompletion, LLMChunk, TokenUsage,
    ToolCall, ToolCallFunction, OperationContext
)
from corpus_sdk.llm.llm_base import (
    BadRequest, AuthError, ResourceExhausted, TransientNetwork,
    Unavailable, NotSupported, ModelOverloaded, DeadlineExceeded
)

class ProductionLLMAdapter(BaseLLMAdapter):
    """
    Production-ready LLM adapter with 100% conformance.
    
    SHARED PLANNING: Single _plan_response used by both complete and stream.
    TOOL CALLS: Validates tool_choice, synthesizes token usage.
    STOP SEQUENCES: Cuts at FIRST occurrence.
    STREAMING: Tool calls only in final chunk, empty non-final chunks.
    CAPABILITIES: Hardcoded, not configurable.
    """
    
    def __init__(self, client, supported_models, model_families=None, **kwargs):
        super().__init__(**kwargs)
        self._client = client
        self._supported_models = tuple(supported_models)
        self._model_families = model_families or {}
        self._max_context_length = 128000
        self._max_tool_calls_per_turn = 5
        
        # Stats tracking (adapter-owned only)
        self._stats = {
            "complete_calls": 0,
            "stream_calls": 0,
            "count_tokens_calls": 0,
            "total_completion_tokens": 0,
            "total_prompt_tokens": 0,
            "total_processing_time_ms": 0.0,
            "error_count": 0
        }
    
    # ----------------------------------------------------------------------
    # CAPABILITIES (Hardcoded - NOT configurable)
    # ----------------------------------------------------------------------
    
    async def _do_capabilities(self) -> LLMCapabilities:
        """Advertise true provider capabilities - NEVER configurable."""
        return LLMCapabilities(
            server="my-llm-provider",
            version="1.0.0",
            protocol="llm/v1.0",
            model_family="gpt",
            max_context_length=self._max_context_length,
            supports_streaming=True,
            supports_roles=True,
            supports_json_output=False,
            supports_tools=True,
            supports_parallel_tool_calls=False,
            supports_tool_choice=True,
            max_tool_calls_per_turn=self._max_tool_calls_per_turn,
            idempotent_writes=False,
            supports_multi_tenant=True,
            supports_system_message=True,
            supports_deadline=True,
            supports_count_tokens=True,
            supported_models=self._supported_models
        )
    
    # ----------------------------------------------------------------------
    # SHARED PLANNING PATH (Single source of truth)
    # ----------------------------------------------------------------------
    
    def _plan_response(self, request, ctx=None):
        """
        Single planning function for BOTH complete() and stream().
        
        Returns: (prompt_text, completion_text, finish_reason, tool_calls)
        """
        # Validate model
        if request.model not in self._supported_models:
            raise ModelNotAvailable(
                f"Model '{request.model}' is not supported",
                details={
                    "requested_model": request.model,
                    "supported_models": list(self._supported_models)
                }
            )
        
        # Validate tool_choice
        self._validate_tool_choice(request.tool_choice, request.tools)
        
        # Build prompt
        prompt = self._build_prompt(request)
        
        # Get timeout from context
        timeout = self._get_timeout(ctx)
        
        # Call provider
        try:
            response = self._client.complete(
                model=request.model,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                stop=request.stop_sequences,
                system=request.system_message,
                tools=request.tools,
                tool_choice=request.tool_choice,
                timeout=timeout
            )
        except Exception as e:
            raise self._map_provider_error(e)
        
        # Apply stop sequences (FIRST occurrence rule)
        text = self._apply_stop_sequences(response.text, request.stop_sequences)
        
        finish_reason = response.finish_reason
        if finish_reason not in ("stop", "length", "tool_calls", "error"):
            finish_reason = "stop"
        
        return prompt, text, finish_reason, response.tool_calls
    
    def _validate_tool_choice(self, tool_choice, tools):
        """Validate tool_choice against available tools."""
        if not tools:
            if tool_choice not in (None, "none", "auto"):
                raise BadRequest("tool_choice provided but no tools")
            return
        
        requested = None
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                fn = tool_choice.get("function", {})
                requested = fn.get("name")
            elif "name" in tool_choice:
                requested = tool_choice["name"]
        elif isinstance(tool_choice, str):
            if tool_choice not in ("none", "auto", "required"):
                requested = tool_choice
        
        if requested:
            tool_names = []
            for t in tools:
                if isinstance(t, dict):
                    if "function" in t:
                        tool_names.append(t["function"].get("name"))
                    elif "name" in t:
                        tool_names.append(t["name"])
            
            if requested not in tool_names:
                raise BadRequest(
                    f"tool_choice requested unknown tool: {requested}",
                    details={
                        "requested": requested,
                        "available": tool_names
                    }
                )
    
    def _apply_stop_sequences(self, text: str, stops: Optional[List[str]]) -> str:
        """Stop at FIRST occurrence of ANY stop sequence."""
        if not stops or not text:
            return text
        
        cut_pos = len(text)
        for stop in stops:
            if not stop:
                continue
            pos = text.find(stop)
            if pos != -1 and pos < cut_pos:
                cut_pos = pos
        
        if cut_pos < len(text):
            return text[:cut_pos].rstrip()
        return text
    
    def _build_prompt(self, request) -> str:
        """Build prompt string for token counting."""
        parts = []
        if request.system_message:
            parts.append(f"system: {request.system_message}")
        for msg in request.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)
    
    # ----------------------------------------------------------------------
    # TOKEN COUNTING (Accurate - no approximations)
    # ----------------------------------------------------------------------
    
    async def _do_count_tokens(self, text: str, model: Optional[str], *, ctx=None) -> int:
        """Accurate token counting - NEVER approximate."""
        self._stats["count_tokens_calls"] += 1
        t0 = time.monotonic()
        
        if model and model not in self._supported_models:
            raise ModelNotAvailable(f"Model '{model}' is not supported")
        
        timeout = self._get_timeout(ctx)
        
        try:
            count = await self._client.count_tokens(
                text=text,
                model=model or self._supported_models[0],
                timeout=timeout
            )
        except Exception as e:
            self._stats["error_count"] += 1
            raise self._map_provider_error(e)
        
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
        return count
    
    def _count_tokens_sync(self, text: str, model: str) -> int:
        """Synchronous token counting for internal use."""
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    
    def _calculate_usage(self, prompt: str, completion: str, 
                        tool_calls: Optional[List[ToolCall]], 
                        model: str) -> TokenUsage:
        """Calculate token usage with tool call accounting."""
        prompt_tokens = self._count_tokens_sync(prompt, model)
        
        if tool_calls:
            # Synthesize tokens from tool call payload
            tool_payload = json.dumps([
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in tool_calls
            ], separators=(",", ":"))
            
            completion_tokens = self._count_tokens_sync(tool_payload, model)
            if completion_tokens == 0:
                completion_tokens = 10  # Minimum viable
        else:
            completion_tokens = self._count_tokens_sync(completion, model)
        
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    
    def _calculate_usage_so_far(self, prompt: str, partial: str, 
                               tool_calls: Optional[List[ToolCall]]) -> TokenUsage:
        """Calculate usage for partial stream."""
        prompt_tokens = self._count_tokens_sync(prompt, self._get_default_model())
        
        if tool_calls:
            return TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                total_tokens=prompt_tokens
            )
        
        completion_tokens = self._count_tokens_sync(partial, self._get_default_model())
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    
    def _get_default_model(self) -> str:
        return self._supported_models[0] if self._supported_models else "gpt-4"
    
    # ----------------------------------------------------------------------
    # COMPLETE (Unary)
    # ----------------------------------------------------------------------
    
    async def _do_complete(self, messages, max_tokens=None, temperature=None,
                          top_p=None, frequency_penalty=None, presence_penalty=None,
                          stop_sequences=None, model=None, system_message=None,
                          tools=None, tool_choice=None, ctx=None) -> LLMCompletion:
        
        self._stats["complete_calls"] += 1
        t0 = time.monotonic()
        
        # Create request object
        request = _Request(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop_sequences,
            model=model or self._supported_models[0],
            system_message=system_message,
            tools=tools,
            tool_choice=tool_choice
        )
        
        # Plan response
        prompt, text, finish_reason, tool_calls = self._plan_response(request, ctx)
        
        # Calculate usage
        usage = self._calculate_usage(prompt, text, tool_calls, request.model)
        
        # Update stats
        self._stats["total_prompt_tokens"] += usage.prompt_tokens
        self._stats["total_completion_tokens"] += usage.completion_tokens
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
        
        return LLMCompletion(
            text=text,
            model=request.model,
            model_family=self._get_model_family(request.model),
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=tool_calls or []
        )
    
    def _get_model_family(self, model: str) -> str:
        """Extract model family from model name."""
        if model in self._model_families:
            return self._model_families[model]
        if "gpt" in model:
            return "gpt"
        if "claude" in model:
            return "claude"
        if "gemini" in model:
            return "gemini"
        return "custom"
    
    # ----------------------------------------------------------------------
    # STREAM
    # ----------------------------------------------------------------------
    
    async def _do_stream(self, messages, max_tokens=None, temperature=None,
                        top_p=None, frequency_penalty=None, presence_penalty=None,
                        stop_sequences=None, model=None, system_message=None,
                        tools=None, tool_choice=None, ctx=None) -> AsyncIterator[LLMChunk]:
        
        self._stats["stream_calls"] += 1
        t0 = time.monotonic()
        
        # Enforce capabilities
        caps = await self._do_capabilities()
        if not caps.supports_streaming:
            raise NotSupported("streaming is not supported")
        
        # Create request object
        request = _Request(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop_sequences,
            model=model or self._supported_models[0],
            system_message=system_message,
            tools=tools,
            tool_choice=tool_choice
        )
        
        # Plan response (SAME planning function as complete)
        prompt, text, finish_reason, tool_calls = self._plan_response(request, ctx)
        
        # TOOL CALL STREAMING
        if tool_calls:
            # Non-final chunk (empty)
            yield LLMChunk(
                text="",
                is_final=False,
                model=request.model,
                usage_so_far=self._calculate_usage_so_far(prompt, "", None)
            )
            
            # Final chunk with tool_calls
            final_usage = self._calculate_usage(prompt, text, tool_calls, request.model)
            yield LLMChunk(
                text="",
                is_final=True,
                model=request.model,
                usage_so_far=final_usage,
                tool_calls=tool_calls
            )
            
            # Update stats
            self._stats["total_prompt_tokens"] += final_usage.prompt_tokens
            self._stats["total_completion_tokens"] += final_usage.completion_tokens
            self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
            return
        
        # NORMAL TEXT STREAMING
        tokens = text.split()
        emitted = []
        
        for i, token in enumerate(tokens):
            emitted.append(token)
            partial = " ".join(emitted)
            usage = self._calculate_usage_so_far(prompt, partial, None)
            
            yield LLMChunk(
                text=token + (" " if i < len(tokens) - 1 else ""),
                is_final=False,
                model=request.model,
                usage_so_far=usage
            )
        
        # Final chunk
        final_usage = self._calculate_usage(prompt, text, None, request.model)
        yield LLMChunk(
            text="",
            is_final=True,
            model=request.model,
            usage_so_far=final_usage
        )
        
        # Update stats
        self._stats["total_prompt_tokens"] += final_usage.prompt_tokens
        self._stats["total_completion_tokens"] += final_usage.completion_tokens
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
    
    # ----------------------------------------------------------------------
    # HEALTH
    # ----------------------------------------------------------------------
    
    async def _do_health(self, *, ctx=None) -> Dict[str, Any]:
        """Health check - NO ctx.attrs-driven forcing."""
        try:
            healthy = await self._client.health_check()
            return {
                "ok": healthy,
                "status": "ok" if healthy else "degraded",
                "server": "my-llm-provider",
                "version": "1.0.0",
                "models": {
                    m: {"status": "ready"} 
                    for m in self._supported_models
                }
            }
        except Exception:
            return {
                "ok": False,
                "status": "down",
                "server": "my-llm-provider",
                "version": "1.0.0"
            }
    
    # ----------------------------------------------------------------------
    # UTILITIES
    # ----------------------------------------------------------------------
    
    def _get_timeout(self, ctx):
        """Convert deadline to timeout."""
        if ctx is None:
            return None
        rem = ctx.remaining_ms()
        if rem is None or rem <= 0:
            return None
        return rem / 1000.0
    
    def _map_provider_error(self, e: Exception):
        """Map provider errors to canonical Corpus errors."""
        # Import error classes directly from llm_base
        from corpus_sdk.llm.llm_base import (
            BadRequest, AuthError, ResourceExhausted,
            TransientNetwork, Unavailable, ModelOverloaded,
            DeadlineExceeded, NotSupported
        )
        
        # Mapping logic - customize for your provider
        if "rate limit" in str(e).lower():
            return ResourceExhausted("Rate limit exceeded", retry_after_ms=5000)
        if "auth" in str(e).lower() or "key" in str(e).lower():
            return AuthError("Authentication failed")
        if "timeout" in str(e).lower():
            return TransientNetwork("Request timeout")
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            return BadRequest(str(e))
        
        return Unavailable(f"Provider error: {type(e).__name__}")


@dataclass
class _Request:
    """Internal request container."""
    messages: List[Mapping[str, str]]
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
    frequency_penalty: Optional[float]
    presence_penalty: Optional[float]
    stop_sequences: Optional[List[str]]
    model: str
    system_message: Optional[str]
    tools: Optional[List[Dict[str, Any]]]
    tool_choice: Optional[Union[str, Dict[str, Any]]]
```

---

## 8. EMBEDDING ADAPTER: IMPLEMENTATION REQUIREMENTS

### 8.1 Required Methods

```python
from corpus_sdk.embedding.embedding_base import BaseEmbeddingAdapter
from corpus_sdk.embedding.embedding_base import (
    EmbeddingCapabilities, EmbedSpec, BatchEmbedSpec,
    EmbedResult, BatchEmbedResult, EmbeddingVector,
    EmbedChunk, EmbeddingStats, OperationContext
)
from corpus_sdk.embedding.embedding_base import (
    BadRequest, AuthError, ResourceExhausted,
    TransientNetwork, Unavailable, NotSupported,
    DeadlineExceeded, TextTooLong, ModelNotAvailable
)

class MyEmbeddingAdapter(BaseEmbeddingAdapter):
    async def _do_capabilities(self) -> EmbeddingCapabilities:
        """REQUIRED: Describe your models, limits, normalization behavior."""
        
    async def _do_embed(
        self,
        spec: EmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> EmbedResult:
        """REQUIRED: Single text embedding with validation."""
        
    async def _do_embed_batch(
        self,
        spec: BatchEmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchEmbedResult:
        """REQUIRED: Batch embedding with chosen failure mode."""
        
    async def _do_stream_embed(
        self,
        spec: EmbedSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[EmbedChunk]:
        """REQUIRED: Streaming embedding with chosen pattern."""
        
    async def _do_count_tokens(
        self,
        text: str,
        model: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> int:
        """REQUIRED if supports_token_counting=True: Accurate token counting."""
        
    async def _do_get_stats(
        self,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> EmbeddingStats:
        """REQUIRED: Adapter-owned stats only. NO cache metrics."""
        
    async def _do_health(
        self,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> Dict[str, Any]:
        """REQUIRED: Health check."""
```

### 8.2 Validation Placement: _do_embed MUST Validate: MANDATORY

```python
async def _do_embed(self, spec: EmbedSpec, *, ctx=None) -> EmbedResult:
    """MANDATORY: Validate in _do_embed, do NOT assume base validated."""
    
    # ✅ CORRECT: Validate in _do_embed
    if not isinstance(spec.text, str) or not spec.text.strip():
        raise BadRequest("text must be a non-empty string")
    
    if spec.model not in self._supported_models:
        raise ModelNotAvailable(
            f"Model '{spec.model}' is not supported",
            details={
                "requested_model": spec.model,
                "supported_models": list(self._supported_models)
            }
        )
    
    # Proceed with embedding generation...
```

**RULE:** Every `_do_embed` MUST validate its own inputs. Do not assume base validation happened. Base validates wire format, NOT provider-specific constraints.

### 8.3 Batch Failure Mode: CHOOSE ONE: MANDATORY

**YOU MUST CHOOSE ONE of these two patterns. Document your choice. NEVER make it configurable.**

---

**OPTION A: Collect per-item failures (Partial Success)**

```python
async def _do_embed_batch(self, spec: BatchEmbedSpec, *, ctx=None) -> BatchEmbedResult:
    """Batch with per-item failure collection."""
    
    embeddings = []
    failures = []
    
    for i, text in enumerate(spec.texts):
        try:
            # Validate each item
            if not isinstance(text, str) or not text.strip():
                raise BadRequest("text must be non-empty")
            
            # Generate embedding
            vec = await self._generate_embedding(text, spec.model)
            tokens = self._count_tokens_sync(text, spec.model)
            
            embeddings.append(EmbeddingVector(
                vector=vec,
                text=text,
                model=spec.model,
                dimensions=len(vec)
            ))
            
        except Exception as e:
            # COLLECT failure, continue processing
            failures.append({
                "index": i,
                "text": text[:50],
                "error": type(e).__name__,
                "code": getattr(e, "code", None) or type(e).__name__.upper(),
                "message": str(e)[:200],
                "metadata": spec.metadatas[i] if spec.metadatas else None
            })
    
    total_tokens = sum(
        self._count_tokens_sync(ev.text, ev.model) 
        for ev in embeddings
    )
    
    return BatchEmbedResult(
        embeddings=embeddings,
        model=spec.model,
        total_texts=len(spec.texts),
        total_tokens=total_tokens,
        failed_texts=failures  # REQUIRED: partial failure reporting
    )
```

---

**OPTION B: Fail-fast (All or nothing)**

```python
async def _do_embed_batch(self, spec: BatchEmbedSpec, *, ctx=None) -> BatchEmbedResult:
    """Batch with fail-fast on first error."""
    
    # Validate ALL items FIRST
    for i, text in enumerate(spec.texts):
        if not isinstance(text, str) or not text.strip():
            raise BadRequest(
                f"text at index {i} must be non-empty",
                details={"index": i, "text_preview": text[:50]}
            )
    
    # If validation passes, process entire batch
    embeddings = []
    total_tokens = 0
    
    for text in spec.texts:
        vec = await self._generate_embedding(text, spec.model)
        tokens = self._count_tokens_sync(text, spec.model)
        total_tokens += tokens
        
        embeddings.append(EmbeddingVector(
            vector=vec,
            text=text,
            model=spec.model,
            dimensions=len(vec)
        ))
    
    return BatchEmbedResult(
        embeddings=embeddings,
        model=spec.model,
        total_texts=len(spec.texts),
        total_tokens=total_tokens,
        failed_texts=[]  # REQUIRED: empty array, not None
    )
```

### 8.4 Batch Failure Mode: CONFIGURABILITY FORBIDDEN: MANDATORY

```python
# ❌ WRONG: NEVER make batch failure mode configurable
class MyEmbeddingAdapter(BaseEmbeddingAdapter):
    def __init__(self, collect_failures=True, **kwargs):  # ❌ NO
        super().__init__(**kwargs)
        self._collect_failures = collect_failures  # ❌ NO
        # Conformance tests expect deterministic behavior

# ✅ CORRECT: Choose ONE, document it, remove config
class MyEmbeddingAdapter(BaseEmbeddingAdapter):
    """Production embedding adapter.
    
    BATCH FAILURE MODE: Collect per-item failures with partial success reporting.
    This adapter never fails the entire batch due to individual item errors.
    """
    
    async def _do_embed_batch(self, spec, *, ctx=None):
        # Implementation with collection pattern
        pass
```

**RULE:** Your adapter MUST have exactly ONE batch failure behavior. It MUST NOT be configurable at runtime. Document your choice in the class docstring.

### 8.5 Truncation and Normalization: Base Owns, You Report

```python
async def _do_capabilities(self) -> EmbeddingCapabilities:
    return EmbeddingCapabilities(
        # ... other fields ...
        max_text_length=8192,  # Set your actual limit
        supports_truncation=True,  # Base will truncate
        supports_normalization=True,  # Base can normalize
        normalizes_at_source=False,  # Set to True ONLY if provider normalizes
    )

# ✅ CORRECT: Let base handle truncation
# You do NOT truncate in _do_embed
async def _do_embed(self, spec, *, ctx=None):
    # spec.text is ALREADY truncated by base if needed
    # You just generate embedding from whatever text arrives
    
# ✅ CORRECT: Let base handle normalization
# Return raw vectors, base normalizes if spec.normalize=True and normalizes_at_source=False
```

**RULE:** Do not implement truncation or normalization in `_do_embed`. Base handles both. You only report whether provider normalizes at source.

### 8.6 Cache Stats Ownership: CRITICAL BOUNDARY: MANDATORY

```python
async def _do_get_stats(self, *, ctx=None) -> EmbeddingStats:
    """MANDATORY: Adapter-owned stats ONLY. NO cache metrics."""
    
    # ✅ CORRECT: Only adapter-owned metrics
    total_ops = (
        self._stats["embed_calls"] +
        self._stats["embed_batch_calls"] +
        self._stats["stream_embed_calls"] +
        self._stats["count_tokens_calls"]
    )
    
    avg_ms = (self._stats["total_processing_time_ms"] / total_ops) if total_ops else 0
    
    return EmbeddingStats(
        total_requests=total_ops,
        total_texts=self._stats["total_texts_embedded"],
        total_tokens=self._stats["total_tokens_processed"],
        avg_processing_time_ms=avg_ms,
        error_count=self._stats["error_count"]
        # ❌ NEVER include these: base owns them
        # cache_hits=...  WRONG
        # cache_misses=... WRONG
        # stream_stats=... WRONG: base owns
    )
```

**RULE:** `_do_get_stats()` MUST NOT include `cache_hits`, `cache_misses`, or any stream stats aggregated by base. Base owns these counters. Duplicating them causes double-counting in metrics and conformance failures.

### 8.7 Streaming Pattern: CHOOSE ONE: MANDATORY

**YOU MUST CHOOSE ONE of these three streaming patterns. Document your choice. NEVER make it configurable.**

---

**PATTERN 1: Single-chunk**

```python
async def _do_stream_embed(self, spec: EmbedSpec, *, ctx=None):
    """STREAMING PATTERN: Single chunk with one vector."""
    
    # Generate full vector
    vec = await self._generate_embedding(spec.text, spec.model)
    tokens = self._count_tokens_sync(spec.text, spec.model)
    
    ev = EmbeddingVector(
        vector=vec,
        text=spec.text,
        model=spec.model,
        dimensions=len(vec)
    )
    
    # Single chunk, is_final=True
    yield EmbedChunk(
        embeddings=[ev],
        is_final=True,
        usage={"tokens": tokens},
        model=spec.model
    )
```

---

**PATTERN 2: Progressive (partial vectors)**

```python
async def _do_stream_embed(self, spec: EmbedSpec, *, ctx=None):
    """STREAMING PATTERN: Progressive: partial vectors growing to full dimension."""
    
    dim = self._get_dimension(spec.model)
    chunk_size = max(1, dim // 3)
    accumulated = []
    tokens = self._count_tokens_sync(spec.text, spec.model)
    
    for i in range(0, dim, chunk_size):
        # Generate next chunk
        chunk_vec = await self._generate_partial_embedding(
            spec.text, spec.model,
            start=i, length=min(chunk_size, dim - i)
        )
        accumulated.extend(chunk_vec)
        is_final = len(accumulated) >= dim
        
        ev = EmbeddingVector(
            vector=accumulated.copy(),
            text=spec.text if i == 0 else "",
            model=spec.model,
            dimensions=len(accumulated),
            index=i // chunk_size,
            metadata={"partial": True, "total_dimensions": dim} if not is_final else None
        )
        
        yield EmbedChunk(
            embeddings=[ev],
            is_final=is_final,
            usage={"tokens": tokens} if is_final else None,
            model=spec.model
        )
```

---

**PATTERN 3: Multi-vector (multiple complete vectors)**

```python
async def _do_stream_embed(self, spec: EmbedSpec, *, ctx=None):
    """STREAMING PATTERN: Multi-vector: multiple complete vectors per chunk."""
    
    num_vectors = 3
    tokens = self._count_tokens_sync(spec.text, spec.model)
    
    for i in range(num_vectors):
        vec = await self._generate_embedding_variation(spec.text, spec.model, i)
        
        ev = EmbeddingVector(
            vector=vec,
            text=f"{spec.text} [variation {i}]",
            model=spec.model,
            dimensions=len(vec),
            index=i
        )
        
        is_final = i == num_vectors - 1
        
        yield EmbedChunk(
            embeddings=[ev],
            is_final=is_final,
            usage={"tokens": tokens} if is_final else None,
            model=spec.model
        )
```

**RULE:** Your adapter MUST implement EXACTLY ONE streaming pattern. Document which pattern in your class docstring. Do NOT make it configurable at runtime.

### 8.8 Capabilities: NO RUNTIME CONFIGURATION: MANDATORY

```python
# ❌ WRONG: NEVER make core capabilities configurable
class MyEmbeddingAdapter(BaseEmbeddingAdapter):
    def __init__(
        self,
        supports_batch=True,      # ❌ NO
        supports_streaming=True, # ❌ NO
        **kwargs
    ):
        super().__init__(**kwargs)
        self.supports_batch = supports_batch  # ❌ NO
        self.supports_streaming = supports_streaming  # ❌ NO

# ✅ CORRECT: Capabilities are HARDCODED based on provider
class MyEmbeddingAdapter(BaseEmbeddingAdapter):
    async def _do_capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            # ... other fields ...
            supports_batch=True,      # HARDCODED: provider supports batch
            supports_streaming=True,  # HARDCODED: provider supports streaming
        )
```

**RULE:** Core capabilities (`supports_batch`, `supports_streaming`, etc.) MUST be hardcoded based on your provider's actual capabilities. Do NOT make them configurable constructor parameters.

### 8.9 Token Counting: NO APPROXIMATIONS: MANDATORY

```python
# ❌ WRONG: NEVER use approximations
def _approx_tokens(self, text: str) -> int:
    words = len(text.split())  # ❌ NOT ACCURATE
    return words + 2  # ❌ CONFORMANCE FAILURE

# ✅ CORRECT: Use real tokenizer or provider API
async def _do_count_tokens(self, text: str, model: str, *, ctx=None) -> int:
    """MANDATORY: Accurate token counting."""
    
    if model not in self._supported_models:
        raise ModelNotAvailable(f"Model '{model}' is not supported")
    
    timeout = self._get_timeout(ctx)
    
    try:
        return await self._client.count_tokens(
            model=model,
            text=text,
            timeout=timeout
        )
    except Exception as e:
        raise self._map_provider_error(e)

def _count_tokens_sync(self, text: str, model: str) -> int:
    """Synchronous token count for internal use."""
    import tiktoken
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
```

**RULE:** Token counting MUST be accurate. Word-count approximations are FOR TESTING ONLY and MUST NOT appear in production adapters.

### 8.10 Complete Embedding Example (Production Ready)

```python
from typing import AsyncIterator, Dict, Any, List, Optional
import asyncio
import time
import hashlib
from corpus_sdk.embedding.embedding_base import BaseEmbeddingAdapter
from corpus_sdk.embedding.embedding_base import (
    EmbeddingCapabilities, EmbedSpec, BatchEmbedSpec,
    EmbedResult, BatchEmbedResult, EmbeddingVector,
    EmbedChunk, EmbeddingStats, OperationContext
)
from corpus_sdk.embedding.embedding_base import (
    BadRequest, AuthError, ResourceExhausted, TransientNetwork,
    Unavailable, NotSupported, DeadlineExceeded,
    TextTooLong, ModelNotAvailable
)

class ProductionEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Production-ready embedding adapter with 100% conformance.
    
    BATCH FAILURE MODE: Collect per-item failures with partial success reporting.
    STREAMING PATTERN: Single-chunk (one final chunk with complete vector).
    CAPABILITIES: Hardcoded based on provider capabilities.
    """
    
    def __init__(self, client, supported_models, model_dimensions, **kwargs):
        super().__init__(**kwargs)
        self._client = client
        self._supported_models = tuple(supported_models)
        self._dimensions = model_dimensions
        self._max_batch_size = 256
        self._max_text_length = 8192
        
        # Stats: NO CACHE METRICS (base owns those)
        self._stats = {
            "embed_calls": 0,
            "embed_batch_calls": 0,
            "stream_embed_calls": 0,
            "count_tokens_calls": 0,
            "total_texts_embedded": 0,
            "total_tokens_processed": 0,
            "total_processing_time_ms": 0.0,
            "error_count": 0
        }
    
    # ----------------------------------------------------------------------
    # CAPABILITIES (Hardcoded - NOT configurable)
    # ----------------------------------------------------------------------
    
    async def _do_capabilities(self) -> EmbeddingCapabilities:
        """Advertise true provider capabilities - NEVER configurable."""
        return EmbeddingCapabilities(
            server="my-embed-provider",
            version="1.0.0",
            protocol="embedding/v1.0",
            supported_models=self._supported_models,
            max_batch_size=self._max_batch_size,
            max_text_length=self._max_text_length,
            max_dimensions=max(self._dimensions.values()),
            supports_normalization=True,
            supports_truncation=True,
            supports_token_counting=True,
            normalizes_at_source=False,  # Provider returns raw vectors
            truncation_mode="base",
            supports_deadline=True,
            idempotent_writes=True,
            supports_multi_tenant=True,
            supports_streaming=True,
            supports_batch_embedding=True
        )
    
    # ----------------------------------------------------------------------
    # SINGLE EMBED (with validation)
    # ----------------------------------------------------------------------
    
    async def _do_embed(self, spec: EmbedSpec, *, ctx=None) -> EmbedResult:
        """MANDATORY: Single embedding with validation."""
        self._stats["embed_calls"] += 1
        t0 = time.monotonic()
        
        # ✅ VALIDATE in _do_embed
        if not isinstance(spec.text, str) or not spec.text.strip():
            raise BadRequest("text must be a non-empty string")
        
        if spec.model not in self._supported_models:
            raise ModelNotAvailable(
                f"Model '{spec.model}' is not supported",
                details={
                    "requested_model": spec.model,
                    "supported_models": list(self._supported_models)
                }
            )
        
        # Get timeout from deadline
        timeout = self._get_timeout(ctx)
        
        try:
            response = await self._client.embed(
                model=spec.model,
                text=spec.text,
                timeout=timeout
            )
        except Exception as e:
            self._stats["error_count"] += 1
            raise self._map_provider_error(e)
        
        vec = response["vector"]
        tokens = response.get("tokens", self._count_tokens_sync(spec.text, spec.model))
        
        ev = EmbeddingVector(
            vector=vec,
            text=spec.text,
            model=spec.model,
            dimensions=len(vec)
        )
        
        self._stats["total_texts_embedded"] += 1
        self._stats["total_tokens_processed"] += tokens
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
        
        return EmbedResult(
            embedding=ev,
            model=spec.model,
            text=spec.text,
            tokens_used=tokens,
            truncated=False  # Base sets this if truncation occurred
        )
    
    # ----------------------------------------------------------------------
    # BATCH EMBED (Collection pattern - chosen, not configurable)
    # ----------------------------------------------------------------------
    
    async def _do_embed_batch(self, spec: BatchEmbedSpec, *, ctx=None) -> BatchEmbedResult:
        """
        BATCH FAILURE MODE: Collect per-item failures, continue processing.
        
        This adapter never fails the entire batch due to individual item errors.
        Failures are reported in failed_texts with full error details.
        """
        self._stats["embed_batch_calls"] += 1
        t0 = time.monotonic()
        
        if spec.model not in self._supported_models:
            raise ModelNotAvailable(f"Model '{spec.model}' is not supported")
        
        if len(spec.texts) > self._max_batch_size:
            reduction_pct = self._suggested_batch_reduction_percent(
                len(spec.texts), self._max_batch_size
            )
            raise BadRequest(
                f"Batch size {len(spec.texts)} exceeds maximum of {self._max_batch_size}",
                details={
                    "max_batch_size": self._max_batch_size,
                    "actual": len(spec.texts)
                },
                suggested_batch_reduction=reduction_pct
            )
        
        timeout = self._get_timeout(ctx)
        
        embeddings = []
        failures = []
        total_tokens = 0
        
        for i, text in enumerate(spec.texts):
            try:
                # ✅ Validate each item
                if not isinstance(text, str) or not text.strip():
                    raise BadRequest("text must be non-empty")
                
                if len(text) > self._max_text_length:
                    raise TextTooLong(
                        f"Text length {len(text)} exceeds maximum of {self._max_text_length}",
                        details={
                            "max_length": self._max_text_length,
                            "actual_length": len(text)
                        }
                    )
                
                # Call provider
                response = await self._client.embed(
                    model=spec.model,
                    text=text,
                    timeout=timeout
                )
                
                vec = response["vector"]
                tokens = response.get("tokens", self._count_tokens_sync(text, spec.model))
                
                embeddings.append(EmbeddingVector(
                    vector=vec,
                    text=text,
                    model=spec.model,
                    dimensions=len(vec)
                ))
                
                total_tokens += tokens
                self._stats["total_texts_embedded"] += 1
                self._stats["total_tokens_processed"] += tokens
                
            except Exception as e:
                # COLLECT failure, continue processing
                failures.append({
                    "index": i,
                    "text": text[:100],  # Truncate for safety
                    "error": type(e).__name__,
                    "code": getattr(e, "code", None) or type(e).__name__.upper(),
                    "message": str(e)[:200],  # Truncate for safety
                    "metadata": spec.metadatas[i] if spec.metadatas else None
                })
                self._stats["error_count"] += 1
        
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
        
        return BatchEmbedResult(
            embeddings=embeddings,
            model=spec.model,
            total_texts=len(spec.texts),
            total_tokens=total_tokens,
            failed_texts=failures  # REQUIRED: partial failure reporting
        )
    
    def _suggested_batch_reduction_percent(self, requested: int, maximum: int) -> Optional[int]:
        """PERCENTAGE reduction hint, not absolute."""
        if requested <= 0 or maximum < 0 or requested <= maximum:
            return None
        return int(100 * (requested - maximum) / requested)
    
    # ----------------------------------------------------------------------
    # STREAM EMBED (Single-chunk pattern - chosen, not configurable)
    # ----------------------------------------------------------------------
    
    async def _do_stream_embed(
        self, spec: EmbedSpec, *, ctx=None
    ) -> AsyncIterator[EmbedChunk]:
        """
        STREAMING PATTERN: Single chunk with one complete vector.
        
        This adapter emits exactly one chunk with is_final=True containing
        the complete embedding vector. No partial vectors are emitted.
        """
        self._stats["stream_embed_calls"] += 1
        t0 = time.monotonic()
        
        # Validate (same as _do_embed)
        if not isinstance(spec.text, str) or not spec.text.strip():
            raise BadRequest("text must be a non-empty string")
        
        if spec.model not in self._supported_models:
            raise ModelNotAvailable(f"Model '{spec.model}' is not supported")
        
        timeout = self._get_timeout(ctx)
        
        try:
            response = await self._client.embed(
                model=spec.model,
                text=spec.text,
                timeout=timeout
            )
        except Exception as e:
            self._stats["error_count"] += 1
            raise self._map_provider_error(e)
        
        vec = response["vector"]
        tokens = response.get("tokens", self._count_tokens_sync(spec.text, spec.model))
        
        ev = EmbeddingVector(
            vector=vec,
            text=spec.text,
            model=spec.model,
            dimensions=len(vec)
        )
        
        # SINGLE CHUNK, is_final=True
        yield EmbedChunk(
            embeddings=[ev],
            is_final=True,
            usage={"tokens": tokens},
            model=spec.model
        )
        
        self._stats["total_texts_embedded"] += 1
        self._stats["total_tokens_processed"] += tokens
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
    
    # ----------------------------------------------------------------------
    # TOKEN COUNTING (Accurate - no approximations)
    # ----------------------------------------------------------------------
    
    async def _do_count_tokens(self, text: str, model: str, *, ctx=None) -> int:
        """MANDATORY: Accurate token counting."""
        self._stats["count_tokens_calls"] += 1
        t0 = time.monotonic()
        
        if model not in self._supported_models:
            raise ModelNotAvailable(f"Model '{model}' is not supported")
        
        timeout = self._get_timeout(ctx)
        
        try:
            count = await self._client.count_tokens(
                model=model,
                text=text,
                timeout=timeout
            )
        except Exception as e:
            self._stats["error_count"] += 1
            raise self._map_provider_error(e)
        
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
        return count
    
    def _count_tokens_sync(self, text: str, model: str) -> int:
        """Synchronous token count for internal use."""
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    
    # ----------------------------------------------------------------------
    # STATS (NO CACHE METRICS - CRITICAL)
    # ----------------------------------------------------------------------
    
    async def _do_get_stats(self, *, ctx=None) -> EmbeddingStats:
        """MANDATORY: Adapter-owned stats ONLY. NO cache metrics."""
        total_ops = (
            self._stats["embed_calls"] +
            self._stats["embed_batch_calls"] +
            self._stats["stream_embed_calls"] +
            self._stats["count_tokens_calls"]
        )
        
        avg_ms = (self._stats["total_processing_time_ms"] / total_ops) if total_ops else 0
        
        # ✅ CORRECT: NO cache_hits, NO cache_misses
        return EmbeddingStats(
            total_requests=total_ops,
            total_texts=self._stats["total_texts_embedded"],
            total_tokens=self._stats["total_tokens_processed"],
            avg_processing_time_ms=avg_ms,
            error_count=self._stats["error_count"]
        )
    
    # ----------------------------------------------------------------------
    # HEALTH
    # ----------------------------------------------------------------------
    
    async def _do_health(self, *, ctx=None) -> Dict[str, Any]:
        """Health check - NO ctx.attrs-driven forcing."""
        try:
            healthy = await self._client.health_check()
            return {
                "ok": healthy,
                "status": "ok" if healthy else "degraded",
                "server": "my-embed-provider",
                "version": "1.0.0",
                "models": {
                    m: {"status": "ready"}
                    for m in self._supported_models
                }
            }
        except Exception:
            return {
                "ok": False,
                "status": "down",
                "server": "my-embed-provider",
                "version": "1.0.0"
            }
    
    # ----------------------------------------------------------------------
    # UTILITIES
    # ----------------------------------------------------------------------
    
    def _get_timeout(self, ctx):
        """Convert deadline to timeout."""
        if ctx is None:
            return None
        rem = ctx.remaining_ms()
        if rem is None or rem <= 0:
            return None
        return rem / 1000.0
    
    def _map_provider_error(self, e: Exception):
        """Map provider errors to canonical Corpus errors."""
        from corpus_sdk.embedding.embedding_base import (
            BadRequest, AuthError, ResourceExhausted,
            TransientNetwork, Unavailable, TextTooLong,
            ModelNotAvailable, DeadlineExceeded
        )
        
        if "rate limit" in str(e).lower():
            return ResourceExhausted("Rate limit exceeded", retry_after_ms=5000)
        if "auth" in str(e).lower() or "key" in str(e).lower():
            return AuthError("Authentication failed")
        if "timeout" in str(e).lower():
            return TransientNetwork("Request timeout")
        if "too long" in str(e).lower() or "length" in str(e).lower():
            return TextTooLong(str(e))
        if "model" in str(e).lower() and "not found" in str(e).lower():
            return ModelNotAvailable(str(e))
        
        return Unavailable(f"Provider error: {type(e).__name__}")
    
    def reset_stats(self):
        """Reset stats (for testing)."""
        self._stats = {k: 0 for k in self._stats}
```

---

## 9. VECTOR ADAPTER: IMPLEMENTATION REQUIREMENTS

### 9.1 Required Methods

```python
from corpus_sdk.vector.vector_base import BaseVectorAdapter
from corpus_sdk.vector.vector_base import (
    VectorCapabilities, QuerySpec, BatchQuerySpec,
    UpsertSpec, DeleteSpec, NamespaceSpec,
    QueryResult, UpsertResult, DeleteResult, NamespaceResult,
    Vector, VectorMatch, VectorID, OperationContext
)
from corpus_sdk.vector.vector_base import (
    BadRequest, AuthError, ResourceExhausted,
    TransientNetwork, Unavailable, NotSupported,
    DeadlineExceeded, DimensionMismatch, IndexNotReady
)

class MyVectorAdapter(BaseVectorAdapter):
    async def _do_capabilities(self) -> VectorCapabilities:
        """REQUIRED: Describe dimensions, metrics, limits."""
        
    async def _do_query(
        self,
        spec: QuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        """REQUIRED: Single vector similarity search."""
        
    async def _do_batch_query(
        self,
        spec: BatchQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> List[QueryResult]:
        """REQUIRED: Batch queries: ALL OR NOTHING atomic."""
        
    async def _do_upsert(
        self,
        spec: UpsertSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        """REQUIRED: Insert/update vectors with namespace authority enforcement."""
        
    async def _do_delete(
        self,
        spec: DeleteSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        """REQUIRED: Delete vectors: IDEMPOTENT, no error on missing."""
        
    async def _do_create_namespace(
        self,
        spec: NamespaceSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        """REQUIRED: Create namespace with dimensions and metric."""
        
    async def _do_delete_namespace(
        self,
        namespace: str,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> NamespaceResult:
        """REQUIRED: Delete namespace and all its vectors."""
        
    async def _do_health(
        self,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> Dict[str, Any]:
        """REQUIRED: Health with per-namespace status."""
```

### 9.2 Namespace Authority: Spec.namespace is Authoritative: MANDATORY

```python
async def _do_upsert(self, spec: UpsertSpec, *, ctx=None) -> UpsertResult:
    """MANDATORY: Enforce namespace authority."""
    
    # spec.namespace is AUTHORITATIVE
    ns = spec.namespace
    
    # ✅ ENFORCE: Vector.namespace MUST match spec.namespace
    for i, v in enumerate(spec.vectors):
        if v.namespace is not None and v.namespace != ns:
            raise BadRequest(
                "vector.namespace must match UpsertSpec.namespace",
                details={
                    "index": i,
                    "spec_namespace": ns,
                    "vector_namespace": v.namespace,
                    "vector_id": str(v.id)
                }
            )
    
    # ❌ NEVER silently correct mismatches
    # v.namespace = ns  # WRONG: do not modify
    
    # ❌ NEVER allow per-item namespace override
    # Process each vector in its own namespace  # WRONG: violates spec
    
    # Proceed with upsert...
```

**RULE:** `UpsertSpec.namespace` is authoritative. If any `Vector.namespace` is provided and does not match, raise `BadRequest` with complete details. Do not silently correct. Do not allow per-vector namespaces.

**SAME RULE APPLIES TO BATCH QUERY:**

```python
async def _do_batch_query(self, spec: BatchQuerySpec, *, ctx=None) -> List[QueryResult]:
    """MANDATORY: Batch namespace authority."""
    
    for i, q in enumerate(spec.queries):
        if q.namespace != spec.namespace:
            raise BadRequest(
                f"query[{i}].namespace must match batch namespace",
                details={
                    "index": i,
                    "batch_namespace": spec.namespace,
                    "query_namespace": q.namespace
                }
            )
```

### 9.3 Include Vectors Contract: [] NOT null: MANDATORY

```python
def _render_matches(self, matches, include_vectors: bool, include_metadata: bool):
    """MANDATORY: include_vectors=False → return [], NOT null, NOT omitted."""
    
    rendered = []
    for match in matches:
        # ✅ CORRECT: include_vectors=False → [] (empty list)
        vector = match.vector
        out_vec = list(vector.vector) if include_vectors else []  # NOT None, NOT omitted
        
        out_meta = dict(vector.metadata) if (include_metadata and vector.metadata) else None
        
        rendered.append(VectorMatch(
            vector=Vector(
                id=vector.id,
                vector=out_vec,  # [] when include_vectors=False
                metadata=out_meta,
                namespace=vector.namespace
            ),
            score=match.score,
            distance=match.distance
        ))
    
    return rendered
```

**RULE:** When `include_vectors=False`, `Vector.vector` MUST be `[]` (empty list). Not `None`. Not omitted. Not zero-filled. Empty list is the canonical "not included" representation.

### 9.4 Filter Dialect Validation: Strict, No Silent Ignore: MANDATORY

```python
def _validate_filter_dialect(self, filter: Optional[Dict], namespace: str):
    """MANDATORY: Validate filter operators before execution."""
    
    if filter is None:
        return
    
    if not isinstance(filter, dict):
        raise BadRequest(
            "filter must be an object",
            details={
                "namespace": namespace,
                "type": type(filter).__name__
            }
        )
    
    for field, condition in filter.items():
        if isinstance(condition, dict):
            # Check for unsupported operators
            unknown_ops = [op for op in condition.keys() if op != "$in"]
            if unknown_ops:
                raise BadRequest(
                    "unsupported filter operator",
                    details={
                        "namespace": namespace,
                        "field": field,
                        "operator": unknown_ops[0],
                        "supported": ["$in"]  # REQUIRED: list of supported operators
                    }
                )
            
            # Validate $in operand
            if "$in" in condition:
                allowed = condition["$in"]
                if not isinstance(allowed, list):
                    raise BadRequest(
                        "invalid '$in' operand: must be list",
                        details={
                            "namespace": namespace,
                            "field": field,
                            "type": type(allowed).__name__
                        }
                    )
```

**RULE:** Unknown filter operators MUST raise `BadRequest` with `supported` list in details. Do NOT silently ignore unsupported operators. Do NOT treat them as "no match".

### 9.5 Filter Operator Error Details: CANONICAL SHAPE: MANDATORY

```python
raise BadRequest(
    "unsupported filter operator",
    details={
        "operator": "$regex",        # REQUIRED
        "field": "title",           # REQUIRED
        "supported": ["$in"],       # REQUIRED: array of supported operators
        "namespace": "docs"         # REQUIRED
    }
)
```

### 9.6 Batch Query Atomicity: All or Nothing: MANDATORY

```python
async def _do_batch_query(self, spec: BatchQuerySpec, *, ctx=None) -> List[QueryResult]:
    """MANDATORY: Batch query is ATOMIC: all or nothing."""
    
    # ✅ VALIDATE ALL queries FIRST
    ns_info = self._get_namespace_info(spec.namespace)
    
    for i, q in enumerate(spec.queries):
        # Validate namespace
        if q.namespace != spec.namespace:
            raise BadRequest(
                f"query[{i}].namespace must match batch namespace",
                details={
                    "index": i,
                    "batch_namespace": spec.namespace,
                    "query_namespace": q.namespace
                }
            )
        
        # Validate filter dialect
        self._validate_filter_dialect(q.filter, spec.namespace)
        
        # Validate dimensions
        if len(q.vector) != ns_info.dimensions:
            raise DimensionMismatch(
                f"query[{i}] vector dimension {len(q.vector)} does not match namespace {ns_info.dimensions}",
                details={
                    "index": i,
                    "expected": ns_info.dimensions,
                    "actual": len(q.vector),
                    "namespace": spec.namespace
                }
            )
    
    # ✅ ONLY after ALL validations pass, execute queries
    results = []
    for q in spec.queries:
        result = await self._execute_single_query(q, spec.namespace, ctx)
        results.append(result)
    
    return results
```

**RULE:** Batch query is ALL OR NOTHING. If any query is invalid, raise error for the ENTIRE batch. Do not return partial results. Do not fall back to per-query execution.

### 9.7 Delete Idempotency: No Error on Missing: MANDATORY

```python
async def _do_delete(self, spec: DeleteSpec, *, ctx=None) -> DeleteResult:
    """MANDATORY: Delete is IDEMPOTENT: no error on missing IDs."""
    
    deleted = 0
    
    if spec.ids:
        for vid in spec.ids:
            key = str(vid)
            if key in self._store[spec.namespace]:
                del self._store[spec.namespace][key]
                deleted += 1
            # ✅ CORRECT: ID not found: continue silently, no error
    
    # ✅ Return deleted_count, NOT attempted_count
    return DeleteResult(
        deleted_count=deleted,  # Actual deletions, not attempts
        failed_count=0,
        failures=[]
    )
```

**RULE:** Delete operations MUST NOT error when IDs do not exist. Return count of ACTUAL deletions, not attempted deletions.

### 9.8 Delete Parameter Rule: IDs XOR Filter: MANDATORY

```python
async def _do_delete(self, spec: DeleteSpec, *, ctx=None) -> DeleteResult:
    """MANDATORY: Must provide either IDs OR filter (not both, not neither)."""
    
    has_ids = bool(spec.ids)
    has_filter = bool(spec.filter)
    
    if has_ids and has_filter:
        raise BadRequest(
            "must provide either ids OR filter, not both",
            details={"namespace": spec.namespace}
        )
    
    if not has_ids and not has_filter:
        raise BadRequest(
            "must provide either ids or filter for deletion",
            details={"namespace": spec.namespace}
        )
    
    # Proceed with deletion...
```

**RULE:** Delete operations MUST accept EITHER `ids` OR `filter`, not both, not neither. This is required by conformance tests.

### 9.9 Distance Metric Strings: EXACT VALUES: MANDATORY

```python
# ✅ CORRECT: Use exact strings from specification
SUPPORTED_METRICS = ("cosine", "euclidean", "dotproduct")

async def _do_capabilities(self) -> VectorCapabilities:
    return VectorCapabilities(
        # ... other fields ...
        supported_metrics=SUPPORTED_METRICS,  # EXACT strings
    )
```

**RULE:** Distance metric strings MUST be exactly `"cosine"`, `"euclidean"`, `"dotproduct"`. No variations. Case-sensitive.

### 9.10 Suggested Batch Reduction: Percentage Semantics: MANDATORY

```python
def _suggested_batch_reduction_percent(self, requested: int, maximum: int) -> Optional[int]:
    """
    MANDATORY: Return PERCENTAGE reduction hint (0-100), not absolute number.
    """
    if requested <= 0 or maximum < 0 or requested <= maximum:
        return None
    
    # ✅ PERCENTAGE calculation
    reduction_pct = int(100 * (requested - maximum) / requested)
    return reduction_pct

async def _do_upsert(self, spec: UpsertSpec, *, ctx=None) -> UpsertResult:
    if len(spec.vectors) > self._max_batch_size:
        reduction_pct = self._suggested_batch_reduction_percent(
            len(spec.vectors), 
            self._max_batch_size
        )
        
        raise BadRequest(
            f"batch size {len(spec.vectors)} exceeds maximum of {self._max_batch_size}",
            details={
                "max_batch_size": self._max_batch_size, 
                "namespace": spec.namespace
            },
            suggested_batch_reduction=reduction_pct  # ✅ PERCENTAGE, not absolute
        )
```

**RULE:** `suggested_batch_reduction` MUST be a PERCENTAGE (0-100). Base uses this to automatically split batches. Do not return absolute numbers.

### 9.11 IndexNotReady: Retry Semantics: MANDATORY

```python
async def _do_query(self, spec: QuerySpec, *, ctx=None) -> QueryResult:
    """MANDATORY: IndexNotReady with retry_after_ms."""
    
    # Check if namespace exists but has no data
    if spec.namespace in self._namespaces and not self._store.get(spec.namespace):
        from corpus_sdk.vector.vector_base import IndexNotReady
        raise IndexNotReady(
            "index not ready (no data in namespace)",
            retry_after_ms=500,  # REQUIRED: always provide retry hint
            details={
                "namespace": spec.namespace  # REQUIRED
            }
        )
```

**RULE:** `IndexNotReady` MUST include `retry_after_ms`. Default 500ms if provider doesn't specify. Always include `namespace` in details.

### 9.12 Namespace Mismatch Error Details: CANONICAL SHAPE: MANDATORY

```python
from corpus_sdk.vector.vector_base import BadRequest

raise BadRequest(
    "vector.namespace must match UpsertSpec.namespace",
    details={
        "index": 2,                    # REQUIRED
        "spec_namespace": "default",   # REQUIRED
        "vector_namespace": "other",   # REQUIRED  
        "vector_id": "vec_123"        # REQUIRED
    }
)
```

### 9.13 Dimension Mismatch Error Details: CANONICAL SHAPE: MANDATORY

```python
from corpus_sdk.vector.vector_base import DimensionMismatch

raise DimensionMismatch(
    f"vector dimension {actual} does not match namespace {expected}",
    details={
        "expected": 384,      # REQUIRED
        "actual": 512,       # REQUIRED
        "namespace": "docs", # REQUIRED
        "vector_id": "vec_123",  # REQUIRED if available
        "index": 3           # REQUIRED for batch operations
    }
)
```

### 9.14 Health Response: Namespace Status: MANDATORY

```python
async def _do_health(self, *, ctx=None) -> Dict[str, Any]:
    """MANDATORY: Health with per-namespace status."""
    
    try:
        healthy = await self._client.health_check()
    except Exception:
        return {
            "ok": False,
            "status": "down",
            "server": "my-vector-provider",
            "version": "1.0.0"
        }
    
    return {
        "ok": healthy,
        "status": "ok" if healthy else "degraded",
        "server": "my-vector-provider",
        "version": "1.0.0",
        "namespaces": {
            ns: {
                "dimensions": info.dimensions,     # REQUIRED
                "metric": info.distance_metric,    # REQUIRED
                "count": len(self._store.get(ns, {})),  # REQUIRED
                "status": "ok"                     # REQUIRED
            }
            for ns, info in self._namespaces.items()
        }
    }
```

**RULE:** Vector health response MUST include per-namespace status with dimensions, metric, count, and status. This is required by conformance tests.

### 9.15 Complete Vector Example (Production Ready)

```python
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import math
import time
from dataclasses import dataclass
from corpus_sdk.vector.vector_base import BaseVectorAdapter
from corpus_sdk.vector.vector_base import (
    VectorCapabilities, QuerySpec, BatchQuerySpec,
    UpsertSpec, DeleteSpec, NamespaceSpec,
    QueryResult, UpsertResult, DeleteResult, NamespaceResult,
    Vector, VectorMatch, VectorID, OperationContext
)
from corpus_sdk.vector.vector_base import (
    BadRequest, AuthError, ResourceExhausted, TransientNetwork,
    Unavailable, NotSupported, DeadlineExceeded,
    DimensionMismatch, IndexNotReady
)

# EXACT metric strings: DO NOT CHANGE
METRIC_COSINE = "cosine"
METRIC_EUCLIDEAN = "euclidean"
METRIC_DOTPRODUCT = "dotproduct"
SUPPORTED_METRICS = (METRIC_COSINE, METRIC_EUCLIDEAN, METRIC_DOTPRODUCT)


class ProductionVectorAdapter(BaseVectorAdapter):
    """
    Production-ready vector adapter with 100% conformance.
    
    BATCH QUERY: Atomic (all or nothing): any invalid query fails entire batch.
    DELETE: Idempotent: no error on missing IDs.
    NAMESPACE: Spec.namespace is authoritative; mismatches raise BadRequest.
    FILTERS: Strict validation: unknown operators raise BadRequest.
    """
    
    def __init__(self, client, **kwargs):
        super().__init__(**kwargs)
        self._client = client
        self._max_dimensions = 2048
        self._max_batch_size = 1000
        self._max_top_k = 1000
        self._max_filter_terms = 10
        
        # namespace -> {id -> Vector}
        self._store: Dict[str, Dict[str, Vector]] = {}
        # namespace -> NamespaceInfo
        self._namespaces: Dict[str, _NamespaceInfo] = {}
        
        # Stats (adapter-owned only)
        self._stats = {
            "query_calls": 0,
            "batch_query_calls": 0,
            "upsert_calls": 0,
            "delete_calls": 0,
            "create_namespace_calls": 0,
            "delete_namespace_calls": 0,
            "total_vectors_upserted": 0,
            "total_vectors_deleted": 0,
            "total_processing_time_ms": 0.0,
            "error_count": 0
        }
    
    # ----------------------------------------------------------------------
    # CAPABILITIES
    # ----------------------------------------------------------------------
    
    async def _do_capabilities(self) -> VectorCapabilities:
        """Advertise true capabilities - HARDCODED, not configurable."""
        return VectorCapabilities(
            server="my-vector-provider",
            version="1.0.0",
            protocol="vector/v1.0",
            max_dimensions=self._max_dimensions,
            supported_metrics=SUPPORTED_METRICS,  # EXACT strings
            supports_namespaces=True,
            supports_metadata_filtering=True,
            supports_batch_operations=True,
            max_batch_size=self._max_batch_size,
            supports_index_management=True,
            idempotent_writes=False,
            supports_multi_tenant=False,
            supports_deadline=True,
            max_top_k=self._max_top_k,
            max_filter_terms=self._max_filter_terms,
            supports_batch_queries=True,
            text_storage_strategy="none"
        )
    
    # ----------------------------------------------------------------------
    # NAMESPACE MANAGEMENT
    # ----------------------------------------------------------------------
    
    async def _do_create_namespace(
        self, spec: NamespaceSpec, *, ctx=None
    ) -> NamespaceResult:
        """Create namespace: idempotent."""
        self._stats["create_namespace_calls"] += 1
        t0 = time.monotonic()
        
        if spec.distance_metric not in SUPPORTED_METRICS:
            raise NotSupported(
                f"distance_metric must be one of: {', '.join(SUPPORTED_METRICS)}"
            )
        
        # Idempotent: if exists, succeed
        if spec.namespace not in self._namespaces:
            self._namespaces[spec.namespace] = _NamespaceInfo(
                dimensions=spec.dimensions,
                distance_metric=spec.distance_metric
            )
            self._store.setdefault(spec.namespace, {})
            created = True
        else:
            created = False
        
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
        
        return NamespaceResult(
            success=True,
            namespace=spec.namespace,
            details={"created": created}
        )
    
    async def _do_delete_namespace(self, namespace: str, *, ctx=None) -> NamespaceResult:
        """Delete namespace: idempotent."""
        self._stats["delete_namespace_calls"] += 1
        t0 = time.monotonic()
        
        existed = namespace in self._namespaces
        self._namespaces.pop(namespace, None)
        self._store.pop(namespace, None)
        
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
        
        return NamespaceResult(
            success=True,
            namespace=namespace,
            details={"existed": existed}
        )
    
    # ----------------------------------------------------------------------
    # QUERY (Single)
    # ----------------------------------------------------------------------
    
    async def _do_query(self, spec: QuerySpec, *, ctx=None) -> QueryResult:
        """Single vector similarity search."""
        self._stats["query_calls"] += 1
        t0 = time.monotonic()
        
        # Validate namespace exists
        if spec.namespace not in self._namespaces:
            raise BadRequest(f"unknown namespace '{spec.namespace}'")
        
        # Validate filter dialect
        self._validate_filter_dialect(spec.filter, spec.namespace)
        
        # Validate dimensions
        ns_info = self._namespaces[spec.namespace]
        if len(spec.vector) != ns_info.dimensions:
            raise DimensionMismatch(
                f"query vector dimension {len(spec.vector)} does not match namespace {ns_info.dimensions}",
                details={
                    "expected": ns_info.dimensions,
                    "actual": len(spec.vector),
                    "namespace": spec.namespace
                }
            )
        
        # Check if index is ready
        if not self._store.get(spec.namespace):
            raise IndexNotReady(
                "index not ready (no data in namespace)",
                retry_after_ms=500,
                details={"namespace": spec.namespace}
            )
        
        timeout = self._get_timeout(ctx)
        
        try:
            response = await self._client.query(
                vector=spec.vector,
                top_k=spec.top_k,
                namespace=spec.namespace,
                filter=self._convert_filter(spec.filter),
                include_metadata=spec.include_metadata,
                include_vectors=spec.include_vectors,
                timeout=timeout
            )
        except Exception as e:
            self._stats["error_count"] += 1
            raise self._map_provider_error(e)
        
        matches = self._render_matches(
            matches=response.matches,
            include_vectors=spec.include_vectors,
            include_metadata=spec.include_metadata
        )
        
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
        
        return QueryResult(
            matches=matches,
            query_vector=spec.vector,
            namespace=spec.namespace,
            total_matches=response.total_matches
        )
    
    # ----------------------------------------------------------------------
    # BATCH QUERY (ATOMIC: All or Nothing)
    # ----------------------------------------------------------------------
    
    async def _do_batch_query(
        self, spec: BatchQuerySpec, *, ctx=None
    ) -> List[QueryResult]:
        """MANDATORY: Batch query is ATOMIC: all or nothing."""
        self._stats["batch_query_calls"] += 1
        t0 = time.monotonic()
        
        # Validate namespace exists
        if spec.namespace not in self._namespaces:
            raise BadRequest(f"unknown namespace '{spec.namespace}'")
        
        ns_info = self._namespaces[spec.namespace]
        
        # ✅ PHASE 1: VALIDATE ALL QUERIES
        for i, q in enumerate(spec.queries):
            # Validate namespace authority
            if q.namespace != spec.namespace:
                raise BadRequest(
                    f"query[{i}].namespace must match batch namespace",
                    details={
                        "index": i,
                        "batch_namespace": spec.namespace,
                        "query_namespace": q.namespace
                    }
                )
            
            # Validate filter dialect
            self._validate_filter_dialect(q.filter, spec.namespace)
            
            # Validate dimensions
            if len(q.vector) != ns_info.dimensions:
                raise DimensionMismatch(
                    f"query[{i}] vector dimension {len(q.vector)} does not match namespace {ns_info.dimensions}",
                    details={
                        "index": i,
                        "expected": ns_info.dimensions,
                        "actual": len(q.vector),
                        "namespace": spec.namespace
                    }
                )
        
        # ✅ PHASE 2: EXECUTE ALL QUERIES (atomic)
        timeout = self._get_timeout(ctx)
        results = []
        
        try:
            responses = await self._client.batch_query(
                queries=[
                    {
                        "vector": q.vector,
                        "top_k": q.top_k,
                        "filter": self._convert_filter(q.filter),
                        "include_metadata": q.include_metadata,
                        "include_vectors": q.include_vectors
                    }
                    for q in spec.queries
                ],
                namespace=spec.namespace,
                timeout=timeout
            )
        except Exception as e:
            self._stats["error_count"] += 1
            raise self._map_provider_error(e)
        
        for i, q in enumerate(spec.queries):
            matches = self._render_matches(
                matches=responses[i].matches,
                include_vectors=q.include_vectors,
                include_metadata=q.include_metadata
            )
            
            results.append(QueryResult(
                matches=matches,
                query_vector=q.vector,
                namespace=spec.namespace,
                total_matches=responses[i].total_matches
            ))
        
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
        return results
    
    # ----------------------------------------------------------------------
    # UPSERT (with Namespace Authority)
    # ----------------------------------------------------------------------
    
    async def _do_upsert(self, spec: UpsertSpec, *, ctx=None) -> UpsertResult:
        """Upsert vectors with namespace authority enforcement."""
        self._stats["upsert_calls"] += 1
        t0 = time.monotonic()
        
        ns = spec.namespace
        
        # Validate namespace exists
        if ns not in self._namespaces:
            raise BadRequest(f"unknown namespace '{ns}'")
        
        # Enforce batch size limit
        if len(spec.vectors) > self._max_batch_size:
            reduction_pct = self._suggested_batch_reduction_percent(
                len(spec.vectors),
                self._max_batch_size
            )
            raise BadRequest(
                f"batch size {len(spec.vectors)} exceeds maximum of {self._max_batch_size}",
                details={"max_batch_size": self._max_batch_size, "namespace": ns},
                suggested_batch_reduction=reduction_pct
            )
        
        # ✅ ENFORCE NAMESPACE AUTHORITY
        for i, v in enumerate(spec.vectors):
            if v.namespace is not None and v.namespace != ns:
                raise BadRequest(
                    "vector.namespace must match UpsertSpec.namespace",
                    details={
                        "index": i,
                        "spec_namespace": ns,
                        "vector_namespace": v.namespace,
                        "vector_id": str(v.id)
                    }
                )
        
        # Validate dimensions
        dims = self._namespaces[ns].dimensions
        for i, v in enumerate(spec.vectors):
            if len(v.vector) != dims:
                raise DimensionMismatch(
                    f"vector dimension {len(v.vector)} does not match namespace {dims}",
                    details={
                        "index": i,
                        "expected": dims,
                        "actual": len(v.vector),
                        "namespace": ns,
                        "vector_id": str(v.id)
                    }
                )
        
        timeout = self._get_timeout(ctx)
        
        try:
            response = await self._client.upsert(
                vectors=[
                    {
                        "id": str(v.id),
                        "vector": v.vector,
                        "metadata": v.metadata
                    }
                    for v in spec.vectors
                ],
                namespace=ns,
                timeout=timeout
            )
        except Exception as e:
            self._stats["error_count"] += 1
            raise self._map_provider_error(e)
        
        # Update local cache
        bucket = self._store.setdefault(ns, {})
        for v in spec.vectors:
            bucket[str(v.id)] = v
        
        self._stats["total_vectors_upserted"] += len(spec.vectors)
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
        
        return UpsertResult(
            upserted_count=len(spec.vectors),
            failed_count=0,
            failures=[]
        )
    
    # ----------------------------------------------------------------------
    # DELETE (Idempotent: No Error on Missing)
    # ----------------------------------------------------------------------
    
    async def _do_delete(self, spec: DeleteSpec, *, ctx=None) -> DeleteResult:
        """MANDATORY: Delete is IDEMPOTENT: no error on missing IDs."""
        self._stats["delete_calls"] += 1
        t0 = time.monotonic()
        
        ns = spec.namespace
        
        # Validate namespace exists
        if ns not in self._namespaces:
            raise BadRequest(f"unknown namespace '{ns}'")
        
        # ✅ Enforce IDs XOR Filter
        has_ids = bool(spec.ids)
        has_filter = bool(spec.filter)
        
        if has_ids and has_filter:
            raise BadRequest(
                "must provide either ids OR filter, not both",
                details={"namespace": ns}
            )
        
        if not has_ids and not has_filter:
            raise BadRequest(
                "must provide either ids or filter for deletion",
                details={"namespace": ns}
            )
        
        # Validate filter if provided
        if has_filter:
            self._validate_filter_dialect(spec.filter, ns)
        
        bucket = self._store.get(ns, {})
        deleted = 0
        
        if has_ids:
            for vid in spec.ids:
                key = str(vid)
                if key in bucket:
                    del bucket[key]
                    deleted += 1
                # ✅ ID not found: continue silently, no error
        
        elif has_filter:
            to_delete = []
            for vid, v in bucket.items():
                if self._filter_match(v.metadata, spec.filter):
                    to_delete.append(vid)
            
            for vid in to_delete:
                del bucket[vid]
                deleted += 1
        
        self._stats["total_vectors_deleted"] += deleted
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
        
        return DeleteResult(
            deleted_count=deleted,  # Actual deletions, not attempts
            failed_count=0,
            failures=[]
        )
    
    # ----------------------------------------------------------------------
    # FILTER VALIDATION (Strict: No Silent Ignore)
    # ----------------------------------------------------------------------
    
    def _validate_filter_dialect(self, filter: Optional[Dict], namespace: str):
        """MANDATORY: Validate filter operators before execution."""
        from corpus_sdk.vector.vector_base import BadRequest
        
        if filter is None:
            return
        
        if not isinstance(filter, dict):
            raise BadRequest(
                "filter must be an object",
                details={
                    "namespace": namespace,
                    "type": type(filter).__name__
                }
            )
        
        for field, condition in filter.items():
            if isinstance(condition, dict):
                # Check for unsupported operators
                unknown_ops = [op for op in condition.keys() if op != "$in"]
                if unknown_ops:
                    raise BadRequest(
                        "unsupported filter operator",
                        details={
                            "namespace": namespace,
                            "field": field,
                            "operator": unknown_ops[0],
                            "supported": ["$in"]  # REQUIRED
                        }
                    )
                
                # Validate $in operand
                if "$in" in condition:
                    allowed = condition["$in"]
                    if not isinstance(allowed, list):
                        raise BadRequest(
                            "invalid '$in' operand: must be list",
                            details={
                                "namespace": namespace,
                                "field": field,
                                "type": type(allowed).__name__
                            }
                        )
    
    def _filter_match(self, metadata: Optional[Dict], filter: Optional[Dict]) -> bool:
        """Match metadata against filter."""
        if not filter:
            return True
        if not metadata:
            return False
        
        for k, v in filter.items():
            if isinstance(v, dict):
                if "$in" in v:
                    if metadata.get(k) not in v["$in"]:
                        return False
            else:
                if metadata.get(k) != v:
                    return False
        return True
    
    def _convert_filter(self, filter: Optional[Dict]) -> Optional[Dict]:
        """Convert Corpus filter to provider-specific filter format."""
        if filter is None:
            return None
        # Implement provider-specific conversion
        return filter
    
    # ----------------------------------------------------------------------
    # HEALTH (with Namespace Status)
    # ----------------------------------------------------------------------
    
    async def _do_health(self, *, ctx=None) -> Dict[str, Any]:
        """MANDATORY: Health with per-namespace status."""
        
        try:
            healthy = await self._client.health_check()
        except Exception:
            return {
                "ok": False,
                "status": "down",
                "server": "my-vector-provider",
                "version": "1.0.0"
            }
        
        return {
            "ok": healthy,
            "status": "ok" if healthy else "degraded",
            "server": "my-vector-provider",
            "version": "1.0.0",
            "namespaces": {
                ns: {
                    "dimensions": info.dimensions,
                    "metric": info.distance_metric,
                    "count": len(self._store.get(ns, {})),
                    "status": "ok" if healthy else "degraded"
                }
                for ns, info in self._namespaces.items()
            }
        }
    
    # ----------------------------------------------------------------------
    # UTILITIES
    # ----------------------------------------------------------------------
    
    def _suggested_batch_reduction_percent(self, requested: int, maximum: int) -> Optional[int]:
        """PERCENTAGE reduction hint, not absolute."""
        if requested <= 0 or maximum < 0 or requested <= maximum:
            return None
        return int(100 * (requested - maximum) / requested)
    
    def _render_matches(self, matches, include_vectors: bool, include_metadata: bool):
        """include_vectors=False → [] (empty list), not null."""
        rendered = []
        for m in matches:
            out_vec = list(m.vector) if include_vectors else []
            out_meta = dict(m.metadata) if (include_metadata and m.metadata) else None
            
            rendered.append(VectorMatch(
                vector=Vector(
                    id=VectorID(m.id),
                    vector=out_vec,
                    metadata=out_meta,
                    namespace=m.namespace
                ),
                score=m.score,
                distance=m.distance
            ))
        return rendered
    
    def _get_timeout(self, ctx):
        if ctx is None:
            return None
        rem = ctx.remaining_ms()
        if rem is None or rem <= 0:
            return None
        return rem / 1000.0
    
    def _map_provider_error(self, e: Exception):
        """Map provider errors to canonical Corpus errors."""
        from corpus_sdk.vector.vector_base import (
            BadRequest, AuthError, ResourceExhausted,
            TransientNetwork, Unavailable, DimensionMismatch,
            IndexNotReady, DeadlineExceeded
        )
        
        if "rate limit" in str(e).lower():
            return ResourceExhausted("Rate limit exceeded", retry_after_ms=5000)
        if "auth" in str(e).lower() or "key" in str(e).lower():
            return AuthError("Authentication failed")
        if "timeout" in str(e).lower():
            return TransientNetwork("Request timeout")
        if "dimension" in str(e).lower():
            return DimensionMismatch(str(e))
        if "not ready" in str(e).lower():
            return IndexNotReady(str(e), retry_after_ms=500)
        
        return Unavailable(f"Provider error: {type(e).__name__}")
    
    def _execute_single_query(self, q, namespace, ctx):
        """Execute a single query (helper for batch)."""
        # Implementation depends on provider
        pass
    
    def _get_namespace_info(self, namespace):
        """Get namespace info."""
        return self._namespaces.get(namespace)


@dataclass
class _NamespaceInfo:
    dimensions: int
    distance_metric: str
```

---

### 9.16 Technical Reference: Similarity & Normalization (MANDATORY)

**Why this matters:** Corpus Protocol conformance tests are mathematically sensitive. If your underlying engine uses a non-standard approximation for similarity, your adapter will fail Behavioral Conformance due to floating-point drift. This section defines the exact mathematical formulas your implementation must satisfy.

---

#### 9.16.1 Cosine Similarity Canonical Form

When `distance_metric` is set to `"cosine"`, the similarity score `S_c` between query vector `A` and document vector `B` **must** resolve to:

```
S_c(A, B) = (A · B) / (||A||₂ × ||B||₂)
```

Where:
- `A · B` = Σᵢ (Aᵢ × Bᵢ)  (dot product)
- `||A||₂` = √(Σᵢ Aᵢ²)      (L2 norm)

**Critical implementation rules:**
- If your provider returns **distance** (e.g., `1 - cosine_similarity`), you **must** convert to similarity before returning: `similarity = 1 - distance`
- If your provider returns **angular distance** (`cosine_distance = 1 - cosine_similarity`), apply same conversion
- If your provider returns **cosine similarity** directly, pass through unchanged
- Never return raw dot product unless vectors are pre-normalized (see §9.16.2)
- If `||A||₂ = 0` or `||B||₂ = 0`, return `0.0` (prevents division by zero)

**Verification test case:**
```python
A = [1.0, 0.0, 0.0]
B = [0.0, 1.0, 0.0]
# Expected cosine similarity: 0.0 (orthogonal vectors)
```

---

#### 9.16.2 L2 Normalization Contract

If your adapter reports `normalizes_at_source = True` in capabilities, the resulting vector `v̂` **must** be the unit vector of `v`, calculated as:

```
v̂ = v / ||v||₂
```

Where:
- `||v||₂` = √(Σᵢ vᵢ²)
- If `||v||₂ = 0`, return zero vector (all components 0.0)

**Critical implementation rules:**
- Normalization **must** happen at the provider level before Corpus receives the vector
- If your provider does **not** normalize at source, set `normalizes_at_source = False` and let the base class handle normalization
- **Never** apply normalization twice (provider + base) as this distorts results and moves vectors off the unit sphere
- Never use L1 normalization (Σ|vᵢ|) as a substitute

**Verification test case:**
```python
v = [3.0, 4.0]           # L2 norm = 5.0
v_hat = [0.6, 0.8]       # Expected normalized vector
```

---

#### 9.16.3 Euclidean Distance Canonical Form

When `distance_metric` is set to `"euclidean"`, the distance `D_e` between vectors **must** be:

```
D_e(A, B) = √(Σᵢ (Aᵢ - Bᵢ)²)
```

This is the **magnitude of the difference vector**: `||A - B||₂`.

**Critical implementation rules:**
- Return **distance**, not squared distance
- Never return negative distances (results are always ≥ 0)
- For zero vectors, distance = norm of the other vector: `D_e(0, B) = ||B||₂`

**Verification test case:**
```python
A = [1.0, 0.0, 0.0]
B = [0.0, 1.0, 0.0]
# Difference vector = [1, -1, 0]
# Sum of squares = 1² + (-1)² + 0² = 2
# Expected Euclidean distance = √2 ≈ 1.4142135623730951
```

---

#### 9.16.4 Dot Product Canonical Form

When `distance_metric` is set to `"dotproduct"`, the score `S_d` **must** be:

```
S_d(A, B) = Σᵢ (Aᵢ × Bᵢ)
```

**Critical implementation rules:**
- Dot product assumes vectors are already appropriately scaled
- If your provider returns cosine similarity when dot product is requested, this **will** fail conformance
- Dot product can be negative; this is expected behavior
- No clamping required (range is (-∞, +∞))

---

#### 9.16.5 Precision Requirements

| Requirement | Value | Consequence if violated |
|-------------|-------|------------------------|
| **Floating Point Precision** | FP32 (single precision) minimum during calculation | Score drift > 0.0001 causes conformance failure |
| **Score Range - Cosine** | Clamp to `[-1.0, 1.0]` | Overflow errors in wire handler |
| **Score Range - Euclidean** | Clamp to `[0.0, +∞)` | Negative distances break downstream systems |
| **Score Range - Dot Product** | No clamping (can be any float) | N/A |
| **Zero Vector Handling - Cosine** | Return `0.0` | Division by zero errors |
| **Zero Vector Handling - Normalization** | Return zero vector (all zeros) | Division by zero errors |
| **Zero Vector Handling - Euclidean** | Return `||other||₂` | Incorrect distance calculations |

---

#### 9.16.6 Conformance Test Vectors

Use these known inputs/outputs to validate your implementation **before** running the full conformance suite:

| Test Case | Input A | Input B | Expected Cosine | Expected Euclidean | Expected Dot |
|-----------|---------|---------|-----------------|-------------------|--------------|
| Orthogonal | `[1,0,0]` | `[0,1,0]` | `0.0` | `√2 ≈ 1.41421356` | `0.0` |
| Identical | `[1,2,3]` | `[1,2,3]` | `1.0` | `0.0` | `14.0` |
| Opposite | `[1,0,0]` | `[-1,0,0]` | `-1.0` | `2.0` | `-1.0` |
| Zero A | `[0,0,0]` | `[1,0,0]` | `0.0` | `1.0` | `0.0` |
| Zero B | `[1,0,0]` | `[0,0,0]` | `0.0` | `1.0` | `0.0` |
| Zero Both | `[0,0,0]` | `[0,0,0]` | `0.0` | `0.0` | `0.0` |
| Scaled | `[2,0,0]` | `[1,0,0]` | `1.0` | `1.0` | `2.0` |

**Implementation verification:** If your adapter passes these seven test cases with precision < 0.0001 drift, you have satisfied the mathematical requirements of the protocol.

---

#### 9.16.7 Common Failure Patterns

| Pattern | Symptom | Root Cause |
|---------|---------|------------|
| **Distance instead of similarity** | Cosine scores always ≤ 0 or inverted | Returning `1 - cosine_similarity` or `cosine_distance` without conversion to similarity |
| **Squared distance** | Euclidean scores too large | Returning Σ(Aᵢ - Bᵢ)² without square root |
| **L1 normalization** | Vectors not unit length | Using Σ\|vᵢ\| instead of √(Σvᵢ²) |
| **Unclamped cosine** | Scores > 1.0 or < -1.0 | Floating-point accumulation drift beyond [-1.0, 1.0] |
| **Provider-side double normalization** | Scores slightly off (0.0001 drift) | Provider normalizes AND base normalizes (moves vectors off unit sphere) |
| **Zero vector division** | Runtime crashes or NaN | Not handling `||v||₂ = 0` case before division |
| **Dot product as cosine** | Scores always between -1 and 1 | Returning `dot(A,B)` when vectors aren't normalized, or returning cosine when dot requested |
| **Incorrect difference vector** | Euclidean distances wrong | Using `A + B` instead of `A - B`, or missing square root |

**If your conformance tests fail by < 0.001:** Check floating-point precision and clamping first. These tiny drifts are almost always precision issues, not logic errors. The most common culprit is **double normalization**—verify your provider isn't normalizing when `normalizes_at_source = True`.

---

#### 9.16.8 Zero Vector Edge Cases (Production Safety)

Zero vectors occur in production (empty documents, failed embeddings, corrupted data). Your adapter **must** handle them gracefully:

| Scenario | Required Behavior |
|----------|-------------------|
| **Query vector is zero, document non-zero** (cosine) | Return 0.0 (cannot determine angle) |
| **Document vector is zero, query non-zero** (cosine) | Return 0.0 |
| **Both vectors zero** (cosine) | Return 0.0 |
| **Normalizing zero vector** | Return zero vector `[0.0, ..., 0.0]` |
| **Query zero, document non-zero** (euclidean) | Return `||document||₂` |
| **Document zero, query non-zero** (euclidean) | Return `||query||₂` |
| **Both zero** (euclidean) | Return 0.0 |
| **Dot product with zero vector** | Return 0.0 |

**Production rule:** Never throw an exception, crash, or return NaN for zero vectors. The system must remain operational even with garbage inputs.

---

## 10. GRAPH ADAPTER: IMPLEMENTATION REQUIREMENTS

### 10.1 Required Methods

```python
from corpus_sdk.graph.graph_base import BaseGraphAdapter
from corpus_sdk.graph.graph_base import (
    GraphCapabilities, GraphID, Node, Edge,
    GraphQuerySpec, GraphTraversalSpec,
    UpsertNodesSpec, UpsertEdgesSpec,
    DeleteNodesSpec, DeleteEdgesSpec,
    BulkVerticesSpec, BulkVerticesResult,
    BatchOperation, BatchResult,
    QueryResult, QueryChunk, TraversalResult,
    GraphSchema, OperationContext
)
from corpus_sdk.graph.graph_base import (
    BadRequest, AuthError, ResourceExhausted,
    TransientNetwork, Unavailable, NotSupported,
    DeadlineExceeded
)

class MyGraphAdapter(BaseGraphAdapter):
    async def _do_capabilities(self) -> GraphCapabilities:
        """REQUIRED: Describe dialects, features, limits."""
    
    async def _do_query(
        self,
        spec: GraphQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> QueryResult:
        """REQUIRED: Unary graph query."""
    
    async def _do_stream_query(
        self,
        spec: GraphQuerySpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> AsyncIterator[QueryChunk]:
        """REQUIRED: Streaming graph query."""
    
    async def _do_bulk_vertices(
        self,
        spec: BulkVerticesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BulkVerticesResult:
        """REQUIRED: Bulk vertex scan with pagination."""
    
    async def _do_batch(
        self,
        ops: List[BatchOperation],
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchResult:
        """REQUIRED: Batch operations with per-op result envelopes."""
    
    async def _do_transaction(
        self,
        operations: List[BatchOperation],
        *,
        ctx: Optional[OperationContext] = None,
    ) -> BatchResult:
        """REQUIRED if supports_transaction=True: Atomic transaction."""
    
    async def _do_traversal(
        self,
        spec: GraphTraversalSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> TraversalResult:
        """REQUIRED if supports_traversal=True: Graph traversal."""
    
    async def _do_get_schema(
        self,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> GraphSchema:
        """REQUIRED if supports_schema=True: Schema retrieval."""
    
    async def _do_upsert_nodes(
        self,
        spec: UpsertNodesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        """REQUIRED: Node upsert."""
    
    async def _do_upsert_edges(
        self,
        spec: UpsertEdgesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> UpsertResult:
        """REQUIRED: Edge upsert."""
    
    async def _do_delete_nodes(
        self,
        spec: DeleteNodesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        """REQUIRED: Node delete (idempotent)."""
    
    async def _do_delete_edges(
        self,
        spec: DeleteEdgesSpec,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> DeleteResult:
        """REQUIRED: Edge delete (idempotent)."""
    
    async def _do_health(
        self,
        *,
        ctx: Optional[OperationContext] = None,
    ) -> Dict[str, Any]:
        """REQUIRED: Health check."""
```

### 10.2 Batch/Transaction Result Envelope: {ok, result}: MANDATORY

```python
async def _do_batch(self, ops: List[BatchOperation], *, ctx=None) -> BatchResult:
    """MANDATORY: Batch results use {ok, result} envelope format."""
    
    results = []
    
    for op in ops:
        try:
            if op.op == "graph.upsert_nodes":
                spec = UpsertNodesSpec(**op.args)
                res = await self._do_upsert_nodes(spec, ctx=ctx)
                # ✅ CORRECT: {ok: true, result: {...}} envelope
                results.append({
                    "ok": True,
                    "result": asdict(res)
                })
            
            elif op.op == "graph.upsert_edges":
                spec = UpsertEdgesSpec(**op.args)
                res = await self._do_upsert_edges(spec, ctx=ctx)
                results.append({
                    "ok": True,
                    "result": asdict(res)
                })
            
            elif op.op == "graph.delete_nodes":
                spec = DeleteNodesSpec(**op.args)
                res = await self._do_delete_nodes(spec, ctx=ctx)
                results.append({
                    "ok": True,
                    "result": asdict(res)
                })
            
            elif op.op == "graph.delete_edges":
                spec = DeleteEdgesSpec(**op.args)
                res = await self._do_delete_edges(spec, ctx=ctx)
                results.append({
                    "ok": True,
                    "result": asdict(res)
                })
            
            elif op.op == "graph.query":
                spec = GraphQuerySpec(**op.args)
                res = await self._do_query(spec, ctx=ctx)
                results.append({
                    "ok": True,
                    "result": {
                        "rows": len(res.records),
                        "dialect": res.dialect
                    }
                })
            
            else:
                # Unknown operation
                results.append({
                    "ok": False,
                    "error": "NotSupported",
                    "code": "NOT_SUPPORTED",
                    "message": f"unknown batch op '{op.op}'"
                })
                
        except Exception as e:
            # Operation-level failure
            results.append({
                "ok": False,
                "error": type(e).__name__,
                "code": getattr(e, "code", None) or type(e).__name__.upper(),
                "message": str(e)
            })
    
    return BatchResult(results=results)

async def _do_transaction(self, operations: List[BatchOperation], *, ctx=None) -> BatchResult:
    """MANDATORY: Transaction uses same envelope format + atomicity signal."""
    
    # Reuse same executor
    results = await self._execute_ops_as_envelopes(operations, ctx)
    
    # Atomicity: success = ALL ops succeeded
    all_ok = all(r.get("ok") for r in results)
    
    return BatchResult(
        results=results,
        success=all_ok,  # Atomic success/failure
        error=None if all_ok else "transaction failed",
        transaction_id=f"tx_{uuid.uuid4().hex[:16]}" if all_ok else None
    )
```

**RULE:** Batch and transaction results MUST use the `{"ok": True/False, "result": {...}, "error": ...}` envelope format. Base expects this for cache invalidation.

### 10.3 Shared Op Executor: Single Kernel for Batch + Transaction: MANDATORY

```python
class MyGraphAdapter(BaseGraphAdapter):
    
    async def _execute_ops_as_envelopes(
        self, 
        ops: List[BatchOperation], 
        ctx: Optional[OperationContext]
    ) -> List[Dict[str, Any]]:
        """
        SHARED OP EXECUTOR: Used by BOTH batch() and transaction().
        Single source of truth for operation execution.
        """
        results = []
        caps = await self._do_capabilities()
        
        for idx, op in enumerate(ops):
            try:
                kind = op.op
                args = dict(op.args or {})
                
                if kind == "graph.upsert_nodes":
                    spec = UpsertNodesSpec(**args)
                    res = await self._do_upsert_nodes(spec, ctx=ctx)
                    results.append({"ok": True, "result": asdict(res)})
                
                elif kind == "graph.upsert_edges":
                    spec = UpsertEdgesSpec(**args)
                    res = await self._do_upsert_edges(spec, ctx=ctx)
                    results.append({"ok": True, "result": asdict(res)})
                
                elif kind == "graph.delete_nodes":
                    spec = DeleteNodesSpec(**args)
                    res = await self._do_delete_nodes(spec, ctx=ctx)
                    results.append({"ok": True, "result": asdict(res)})
                
                elif kind == "graph.delete_edges":
                    spec = DeleteEdgesSpec(**args)
                    res = await self._do_delete_edges(spec, ctx=ctx)
                    results.append({"ok": True, "result": asdict(res)})
                
                elif kind == "graph.query":
                    # ✅ RE-VALIDATE dialect (batch bypasses base)
                    dialect = args.get("dialect")
                    if dialect and caps.supported_query_dialects:
                        if dialect not in caps.supported_query_dialects:
                            raise NotSupported(
                                f"dialect '{dialect}' not supported",
                                details={"supported": caps.supported_query_dialects}
                            )
                    
                    spec = GraphQuerySpec(**args)
                    res = await self._do_query(spec, ctx=ctx)
                    results.append({
                        "ok": True,
                        "result": {
                            "rows": len(res.records),
                            "dialect": res.dialect or dialect
                        }
                    })
                
                else:
                    results.append({
                        "ok": False,
                        "error": "NotSupported",
                        "code": "NOT_SUPPORTED",
                        "message": f"unknown batch op '{kind}'",
                        "index": idx
                    })
                    
            except Exception as e:
                results.append({
                    "ok": False,
                    "error": type(e).__name__,
                    "code": getattr(e, "code", None) or type(e).__name__.upper(),
                    "message": str(e),
                    "index": idx
                })
        
        return results
    
    async def _do_batch(self, ops: List[BatchOperation], *, ctx=None) -> BatchResult:
        """Batch: uses shared executor."""
        results = await self._execute_ops_as_envelopes(ops, ctx)
        return BatchResult(results=results)
    
    async def _do_transaction(self, operations: List[BatchOperation], *, ctx=None) -> BatchResult:
        """Transaction: uses SAME shared executor."""
        results = await self._execute_ops_as_envelopes(operations, ctx)
        
        all_ok = all(r.get("ok") for r in results)
        tx_id = f"tx_{uuid.uuid4().hex[:16]}" if all_ok else None
        
        return BatchResult(
            results=results,
            success=all_ok,
            error=None if all_ok else "transaction failed",
            transaction_id=tx_id
        )
```

**RULE:** You MUST use a single operation execution kernel shared between `_do_batch()` and `_do_transaction()`. Do not duplicate operation logic.

### 10.4 Dialect Validation: TWO Layers: MANDATORY

```python
async def _do_query(self, spec: GraphQuerySpec, *, ctx=None) -> QueryResult:
    """LAYER 1: Validate dialect in _do_query."""
    
    caps = await self._do_capabilities()
    
    if spec.dialect and caps.supported_query_dialects:
        if spec.dialect not in caps.supported_query_dialects:
            raise NotSupported(
                f"dialect '{spec.dialect}' not supported",
                details={
                    "supported_query_dialects": caps.supported_query_dialects
                }
            )
    
    # Proceed with query...

async def _execute_ops_as_envelopes(self, ops, ctx):
    """LAYER 2: RE-VALIDATE dialect in batch/transaction."""
    
    caps = await self._do_capabilities()
    
    for op in ops:
        if op.op == "graph.query":
            args = op.args or {}
            dialect = args.get("dialect")
            
            # ✅ RE-VALIDATE: batch bypasses base validation
            if dialect and caps.supported_query_dialects:
                if dialect not in caps.supported_query_dialects:
                    raise NotSupported(
                        f"dialect '{dialect}' not supported",
                        details={"supported": caps.supported_query_dialects}
                    )
```

**RULE:** You MUST validate dialect in TWO places:
1. In `_do_query()` and `_do_stream_query()` (normal path)
2. AGAIN in batch/transaction op executor (batch bypasses base)

Do not assume base validation occurred for batch operations.

### 10.5 Delete Idempotency: No Error on Missing: MANDATORY

```python
async def _do_delete_nodes(self, spec: DeleteNodesSpec, *, ctx=None) -> DeleteResult:
    """MANDATORY: Delete is IDEMPOTENT: no error on missing IDs."""
    
    deleted = 0
    
    if spec.ids:
        for node_id in spec.ids:
            key = str(node_id)
            if key in self._store[spec.namespace]:
                del self._store[spec.namespace][key]
                deleted += 1
            # ✅ ID not found: continue silently, no error
    
    elif spec.filter:
        # Delete by filter
        to_delete = [
            vid for vid, v in self._store[spec.namespace].items()
            if self._filter_match(v.properties, spec.filter)
        ]
        for vid in to_delete:
            del self._store[spec.namespace][vid]
            deleted += 1
    
    return DeleteResult(
        deleted_count=deleted,  # Actual deletions, not attempts
        failed_count=0,
        failures=[]
    )
```

**RULE:** Delete operations MUST NOT error when IDs do not exist. Return count of ACTUAL deletions, not attempted deletions.

### 10.6 Bulk Vertices Pagination: Cursor Contract: MANDATORY

```python
async def _do_bulk_vertices(
    self, spec: BulkVerticesSpec, *, ctx=None
) -> BulkVerticesResult:
    """MANDATORY: Pagination contract: cursor, has_more."""
    
    total = await self._get_vertex_count(spec.namespace)
    
    # Cursor is opaque to base: can be string offset
    start = int(spec.cursor or 0)
    end = min(start + spec.limit, total)
    
    nodes = []
    for i in range(start, end):
        node = await self._get_vertex_by_index(i, spec.namespace)
        nodes.append(node)
    
    # ✅ REQUIRED fields
    next_cursor = str(end) if end < total else None
    has_more = end < total
    
    return BulkVerticesResult(
        nodes=nodes,
        next_cursor=next_cursor,  # MUST be None when no more
        has_more=has_more         # MUST be bool
    )
```

**RULE:** Bulk vertices MUST return:
- `next_cursor`: string when more results exist, `None` when no more
- `has_more`: boolean indicating if more results exist

Cursor format is opaque to base: can be string offset, token, etc.

### 10.7 Traversal Result Shape: Nodes, Edges, Paths: MANDATORY

```python
async def _do_traversal(
    self, spec: GraphTraversalSpec, *, ctx=None
) -> TraversalResult:
    """MANDATORY: Traversal returns nodes, relationships, paths."""
    
    nodes = []
    edges = []
    paths = []
    
    for start_node in spec.start_nodes:
        # Perform traversal
        result = await self._client.traverse(
            start=start_node,
            max_depth=spec.max_depth,
            direction=spec.direction,
            relationship_types=spec.relationship_types
        )
        
        # ✅ REQUIRED: nodes array
        nodes.extend(result.nodes)
        
        # ✅ REQUIRED: relationships array
        edges.extend(result.relationships)
        
        # ✅ REQUIRED: paths array (sequence of nodes/edges)
        paths.append(result.path)
    
    # Deduplicate nodes while preserving order
    seen = set()
    unique_nodes = []
    for n in nodes:
        if str(n.id) not in seen:
            seen.add(str(n.id))
            unique_nodes.append(n)
    
    return TraversalResult(
        nodes=unique_nodes,
        relationships=edges,
        paths=paths,
        summary={
            "start_nodes": list(spec.start_nodes),
            "max_depth": spec.max_depth,
            "direction": spec.direction,
            "nodes": len(unique_nodes),
            "relationships": len(edges)
        },
        namespace=spec.namespace
    )
```

**RULE:** Traversal result MUST include `nodes`, `relationships`, and `paths` arrays. This is required by conformance tests.

### 10.8 Capabilities Enforcement: Operation Coupling: MANDATORY

```python
async def _do_transaction(self, operations: List[BatchOperation], *, ctx=None) -> BatchResult:
    """MANDATORY: Enforce capabilities before proceeding."""
    
    caps = await self._do_capabilities()
    
    if not caps.supports_transaction:
        raise NotSupported(
            "transactions are not supported by this adapter",
            details={"capability": "supports_transaction"}
        )
    
    # Proceed with transaction...

async def _do_traversal(self, spec: GraphTraversalSpec, *, ctx=None) -> TraversalResult:
    """MANDATORY: Enforce capabilities before proceeding."""
    
    caps = await self._do_capabilities()
    
    if not caps.supports_traversal:
        raise NotSupported(
            "traversal operations are not supported by this adapter",
            details={"capability": "supports_traversal"}
        )
    
    # Proceed with traversal...
```

**RULE:** If `caps.supports_X = False`, `_do_X` MUST raise `NotSupported`. Do not silently no-op.

### 10.9 Capabilities: NO RUNTIME CONFIGURATION: MANDATORY

```python
# ❌ WRONG: NEVER make core capabilities configurable
class MyGraphAdapter(BaseGraphAdapter):
    def __init__(
        self,
        supports_transaction_ops: bool = True,  # ❌ NO
        supports_traversal_ops: bool = True,    # ❌ NO
        **kwargs
    ):
        super().__init__(**kwargs)
        self._supports_transaction = supports_transaction_ops  # ❌ NO
        self._supports_traversal = supports_traversal_ops      # ❌ NO

# ✅ CORRECT: Capabilities are HARDCODED
class MyGraphAdapter(BaseGraphAdapter):
    async def _do_capabilities(self) -> GraphCapabilities:
        return GraphCapabilities(
            # ... other fields ...
            supports_transaction=True,   # HARDCODED: provider supports transactions
            supports_traversal=True,     # HARDCODED: provider supports traversal
        )
```

**RULE:** Core capabilities MUST be hardcoded based on your provider's actual capabilities. Do NOT make them configurable constructor parameters.

### 10.10 Complete Graph Example (Production Ready)

```python
from typing import AsyncIterator, Dict, Any, List, Optional, Tuple
import asyncio
import uuid
import time
from dataclasses import asdict
from corpus_sdk.graph.graph_base import BaseGraphAdapter
from corpus_sdk.graph.graph_base import (
    GraphCapabilities, GraphID, Node, Edge,
    GraphQuerySpec, GraphTraversalSpec,
    UpsertNodesSpec, UpsertEdgesSpec,
    DeleteNodesSpec, DeleteEdgesSpec,
    BulkVerticesSpec, BulkVerticesResult,
    BatchOperation, BatchResult,
    QueryResult, QueryChunk, TraversalResult,
    GraphSchema, OperationContext
)
from corpus_sdk.graph.graph_base import (
    BadRequest, AuthError, ResourceExhausted,
    TransientNetwork, Unavailable, NotSupported,
    DeadlineExceeded
)

class ProductionGraphAdapter(BaseGraphAdapter):
    """
    Production-ready graph adapter with 100% conformance.
    
    DIALECTS: Supported dialects hardcoded in capabilities.
    DELETE: Idempotent: no error on missing IDs.
    BATCH/TRANSACTION: Shared op executor with {ok, result} envelopes.
    CAPABILITIES: Hardcoded, NOT configurable at runtime.
    """
    
    def __init__(self, client, **kwargs):
        super().__init__(**kwargs)
        self._client = client
        
        # HARDCODED capabilities: NOT configurable
        self._supported_dialects = ("cypher", "opencypher")
        self._supports_stream = True
        self._supports_bulk = True
        self._supports_batch = True
        self._supports_schema = True
        self._supports_transaction = True
        self._supports_traversal = True
        self._max_traversal_depth = 10
        self._max_batch_ops = 1000
        
        # In-memory store (replace with real client calls)
        self._store: Dict[str, Dict[str, Node]] = {}
        self._edge_store: Dict[str, Dict[str, Edge]] = {}
        self._namespaces: set = set()
        
        # Stats (adapter-owned only)
        self._stats = {
            "query_calls": 0,
            "stream_query_calls": 0,
            "bulk_vertices_calls": 0,
            "batch_calls": 0,
            "transaction_calls": 0,
            "traversal_calls": 0,
            "get_schema_calls": 0,
            "upsert_nodes_calls": 0,
            "upsert_edges_calls": 0,
            "delete_nodes_calls": 0,
            "delete_edges_calls": 0,
            "total_nodes_upserted": 0,
            "total_edges_upserted": 0,
            "total_nodes_deleted": 0,
            "total_edges_deleted": 0,
            "total_processing_time_ms": 0.0,
            "error_count": 0
        }
    
    # ----------------------------------------------------------------------
    # CAPABILITIES (Hardcoded - NOT configurable)
    # ----------------------------------------------------------------------
    
    async def _do_capabilities(self) -> GraphCapabilities:
        """Advertise true capabilities - NEVER configurable."""
        return GraphCapabilities(
            server="my-graph-provider",
            version="1.0.0",
            protocol="graph/v1.0",
            supported_query_dialects=self._supported_dialects,
            supports_stream_query=self._supports_stream,
            supports_namespaces=True,
            supports_property_filters=True,
            supports_bulk_vertices=self._supports_bulk,
            supports_batch=self._supports_batch,
            supports_schema=self._supports_schema,
            idempotent_writes=False,
            supports_multi_tenant=True,
            supports_deadline=True,
            max_batch_ops=self._max_batch_ops,
            supports_transaction=self._supports_transaction,
            supports_traversal=self._supports_traversal,
            max_traversal_depth=self._max_traversal_depth,
            supports_path_queries=False
        )
    
    # ----------------------------------------------------------------------
    # QUERY (Unary)
    # ----------------------------------------------------------------------
    
    async def _do_query(self, spec: GraphQuerySpec, *, ctx=None) -> QueryResult:
        """Unary graph query with dialect validation."""
        self._stats["query_calls"] += 1
        t0 = time.monotonic()
        
        # ✅ VALIDATE dialect
        if spec.dialect and spec.dialect not in self._supported_dialects:
            raise NotSupported(
                f"dialect '{spec.dialect}' not supported",
                details={
                    "supported_query_dialects": self._supported_dialects
                }
            )
        
        timeout = self._get_timeout(ctx)
        
        try:
            response = await self._client.query(
                dialect=spec.dialect or "cypher",
                query=spec.text,
                params=spec.params or {},
                namespace=spec.namespace,
                timeout=timeout
            )
        except Exception as e:
            self._stats["error_count"] += 1
            raise self._map_provider_error(e)
        
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
        
        return QueryResult(
            records=response.records,
            summary={
                "rows": len(response.records),
                "consumed_ms": response.latency_ms
            },
            dialect=spec.dialect,
            namespace=spec.namespace
        )
    
    # ----------------------------------------------------------------------
    # STREAM QUERY
    # ----------------------------------------------------------------------
    
    async def _do_stream_query(
        self, spec: GraphQuerySpec, *, ctx=None
    ) -> AsyncIterator[QueryChunk]:
        """Streaming graph query."""
        self._stats["stream_query_calls"] += 1
        t0 = time.monotonic()
        
        # Enforce capabilities
        caps = await self._do_capabilities()
        if not caps.supports_stream_query:
            raise NotSupported("stream_query is not supported")
        
        # ✅ VALIDATE dialect
        if spec.dialect and spec.dialect not in self._supported_dialects:
            raise NotSupported(
                f"dialect '{spec.dialect}' not supported",
                details={"supported": self._supported_dialects}
            )
        
        timeout = self._get_timeout(ctx)
        chunk_count = 0
        
        try:
            stream = await self._client.stream_query(
                dialect=spec.dialect or "cypher",
                query=spec.text,
                params=spec.params or {},
                namespace=spec.namespace,
                timeout=timeout
            )
            
            async for chunk in stream:
                chunk_count += 1
                yield QueryChunk(
                    records=chunk.records,
                    is_final=chunk.is_final
                )
                
        except Exception as e:
            self._stats["error_count"] += 1
            raise self._map_provider_error(e)
        
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
    
    # ----------------------------------------------------------------------
    # BULK VERTICES (Pagination Contract)
    # ----------------------------------------------------------------------
    
    async def _do_bulk_vertices(
        self, spec: BulkVerticesSpec, *, ctx=None
    ) -> BulkVerticesResult:
        """Bulk vertex scan with pagination."""
        self._stats["bulk_vertices_calls"] += 1
        t0 = time.monotonic()
        
        caps = await self._do_capabilities()
        if not caps.supports_bulk_vertices:
            raise NotSupported("bulk_vertices is not supported")
        
        timeout = self._get_timeout(ctx)
        
        try:
            response = await self._client.scan_vertices(
                namespace=spec.namespace,
                limit=spec.limit,
                cursor=spec.cursor,
                filter=spec.filter,
                timeout=timeout
            )
        except Exception as e:
            self._stats["error_count"] += 1
            raise self._map_provider_error(e)
        
        nodes = [
            Node(
                id=GraphID(n["id"]),
                labels=tuple(n.get("labels", [])),
                properties=n.get("properties", {}),
                namespace=spec.namespace
            )
            for n in response.vertices
        ]
        
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
        
        # ✅ REQUIRED pagination fields
        return BulkVerticesResult(
            nodes=nodes,
            next_cursor=response.next_cursor,  # None if no more
            has_more=response.has_more         # bool
        )
    
    # ----------------------------------------------------------------------
    # SHARED OP EXECUTOR (Single Kernel)
    # ----------------------------------------------------------------------
    
    async def _execute_ops_as_envelopes(
        self,
        ops: List[BatchOperation],
        ctx: Optional[OperationContext]
    ) -> List[Dict[str, Any]]:
        """SHARED executor for BATCH and TRANSACTION."""
        
        results = []
        caps = await self._do_capabilities()
        
        for idx, op in enumerate(ops):
            try:
                kind = op.op
                args = dict(op.args or {})
                
                if kind == "graph.upsert_nodes":
                    spec = UpsertNodesSpec(**args)
                    res = await self._do_upsert_nodes(spec, ctx=ctx)
                    results.append({"ok": True, "result": asdict(res)})
                
                elif kind == "graph.upsert_edges":
                    spec = UpsertEdgesSpec(**args)
                    res = await self._do_upsert_edges(spec, ctx=ctx)
                    results.append({"ok": True, "result": asdict(res)})
                
                elif kind == "graph.delete_nodes":
                    spec = DeleteNodesSpec(**args)
                    res = await self._do_delete_nodes(spec, ctx=ctx)
                    results.append({"ok": True, "result": asdict(res)})
                
                elif kind == "graph.delete_edges":
                    spec = DeleteEdgesSpec(**args)
                    res = await self._do_delete_edges(spec, ctx=ctx)
                    results.append({"ok": True, "result": asdict(res)})
                
                elif kind == "graph.query":
                    # ✅ RE-VALIDATE dialect (batch bypasses base)
                    dialect = args.get("dialect")
                    if dialect and caps.supported_query_dialects:
                        if dialect not in caps.supported_query_dialects:
                            raise NotSupported(
                                f"dialect '{dialect}' not supported",
                                details={"supported": caps.supported_query_dialects}
                            )
                    
                    spec = GraphQuerySpec(**args)
                    res = await self._do_query(spec, ctx=ctx)
                    results.append({
                        "ok": True,
                        "result": {
                            "rows": len(res.records),
                            "dialect": res.dialect or dialect
                        }
                    })
                
                else:
                    results.append({
                        "ok": False,
                        "error": "NotSupported",
                        "code": "NOT_SUPPORTED",
                        "message": f"unknown batch op '{kind}'",
                        "index": idx
                    })
                    
            except Exception as e:
                results.append({
                    "ok": False,
                    "error": type(e).__name__,
                    "code": getattr(e, "code", None) or type(e).__name__.upper(),
                    "message": str(e),
                    "index": idx
                })
        
        return results
    
    # ----------------------------------------------------------------------
    # BATCH
    # ----------------------------------------------------------------------
    
    async def _do_batch(
        self, ops: List[BatchOperation], *, ctx=None
    ) -> BatchResult:
        """Batch operations: uses shared executor."""
        self._stats["batch_calls"] += 1
        t0 = time.monotonic()
        
        caps = await self._do_capabilities()
        if not caps.supports_batch:
            raise NotSupported("batch is not supported")
        
        results = await self._execute_ops_as_envelopes(ops, ctx)
        
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
        return BatchResult(results=results)
    
    # ----------------------------------------------------------------------
    # TRANSACTION (Atomic)
    # ----------------------------------------------------------------------
    
    async def _do_transaction(
        self, operations: List[BatchOperation], *, ctx=None
    ) -> BatchResult:
        """Transaction: uses SAME shared executor."""
        self._stats["transaction_calls"] += 1
        t0 = time.monotonic()
        
        caps = await self._do_capabilities()
        if not caps.supports_transaction:
            raise NotSupported("transactions are not supported")
        
        results = await self._execute_ops_as_envelopes(operations, ctx)
        
        # Atomicity: success = ALL ops succeeded
        all_ok = all(r.get("ok") for r in results)
        
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
        
        return BatchResult(
            results=results,
            success=all_ok,
            error=None if all_ok else "transaction failed",
            transaction_id=f"tx_{uuid.uuid4().hex[:16]}" if all_ok else None
        )
    
    # ----------------------------------------------------------------------
    # TRAVERSAL
    # ----------------------------------------------------------------------
    
    async def _do_traversal(
        self, spec: GraphTraversalSpec, *, ctx=None
    ) -> TraversalResult:
        """Graph traversal: returns nodes, edges, paths."""
        self._stats["traversal_calls"] += 1
        t0 = time.monotonic()
        
        caps = await self._do_capabilities()
        if not caps.supports_traversal:
            raise NotSupported("traversal is not supported")
        
        timeout = self._get_timeout(ctx)
        
        try:
            response = await self._client.traverse(
                start_nodes=[str(n) for n in spec.start_nodes],
                max_depth=spec.max_depth,
                direction=spec.direction,
                relationship_types=spec.relationship_types,
                namespace=spec.namespace,
                timeout=timeout
            )
        except Exception as e:
            self._stats["error_count"] += 1
            raise self._map_provider_error(e)
        
        # Convert to Node/Edge objects
        nodes = [
            Node(
                id=GraphID(n["id"]),
                labels=tuple(n.get("labels", [])),
                properties=n.get("properties", {}),
                namespace=spec.namespace
            )
            for n in response.nodes
        ]
        
        edges = [
            Edge(
                id=GraphID(e["id"]),
                src=GraphID(e["src"]),
                dst=GraphID(e["dst"]),
                label=e["label"],
                properties=e.get("properties", {}),
                namespace=spec.namespace
            )
            for e in response.edges
        ]
        
        # ✅ REQUIRED: paths array
        paths = response.paths
        
        # Deduplicate nodes
        seen = set()
        unique_nodes = []
        for n in nodes:
            if str(n.id) not in seen:
                seen.add(str(n.id))
                unique_nodes.append(n)
        
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
        
        return TraversalResult(
            nodes=unique_nodes,
            relationships=edges,
            paths=paths,
            summary={
                "start_nodes": list(spec.start_nodes),
                "max_depth": spec.max_depth,
                "direction": spec.direction,
                "nodes": len(unique_nodes),
                "relationships": len(edges)
            },
            namespace=spec.namespace
        )
    
    # ----------------------------------------------------------------------
    # SCHEMA
    # ----------------------------------------------------------------------
    
    async def _do_get_schema(self, *, ctx=None) -> GraphSchema:
        """Retrieve graph schema."""
        self._stats["get_schema_calls"] += 1
        t0 = time.monotonic()
        
        caps = await self._do_capabilities()
        if not caps.supports_schema:
            raise NotSupported("get_schema is not supported")
        
        timeout = self._get_timeout(ctx)
        
        try:
            schema = await self._client.get_schema(timeout=timeout)
        except Exception as e:
            self._stats["error_count"] += 1
            raise self._map_provider_error(e)
        
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
        
        return GraphSchema(
            nodes=schema.node_labels,
            edges=schema.relationship_types,
            metadata={
                "version": schema.version,
                "generated_by": "my-graph-provider"
            }
        )
    
    # ----------------------------------------------------------------------
    # NODE/UPSERT
    # ----------------------------------------------------------------------
    
    async def _do_upsert_nodes(self, spec: UpsertNodesSpec, *, ctx=None) -> UpsertResult:
        """Upsert nodes."""
        self._stats["upsert_nodes_calls"] += 1
        t0 = time.monotonic()
        
        upserted = 0
        failures = []
        
        # Initialize namespace if needed
        if spec.namespace not in self._store:
            self._store[spec.namespace] = {}
            self._namespaces.add(spec.namespace)
        
        for idx, node in enumerate(spec.nodes):
            try:
                # Validate
                if node.labels:
                    if any(not isinstance(l, str) or not l for l in node.labels):
                        raise BadRequest("node.labels must be non-empty strings")
                
                # Upsert
                await self._client.upsert_node(
                    id=str(node.id),
                    labels=node.labels,
                    properties=node.properties or {},
                    namespace=spec.namespace
                )
                
                # Update local cache
                self._store[spec.namespace][str(node.id)] = node
                upserted += 1
                
            except Exception as e:
                failures.append({
                    "index": idx,
                    "id": str(node.id),
                    "error": type(e).__name__,
                    "code": getattr(e, "code", None) or type(e).__name__.upper(),
                    "message": str(e)
                })
                self._stats["error_count"] += 1
        
        self._stats["total_nodes_upserted"] += upserted
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
        
        return UpsertResult(
            upserted_count=upserted,
            failed_count=len(failures),
            failures=failures
        )
    
    # ----------------------------------------------------------------------
    # EDGE/UPSERT
    # ----------------------------------------------------------------------
    
    async def _do_upsert_edges(self, spec: UpsertEdgesSpec, *, ctx=None) -> UpsertResult:
        """Upsert edges."""
        self._stats["upsert_edges_calls"] += 1
        t0 = time.monotonic()
        
        upserted = 0
        failures = []
        
        # Initialize namespace if needed
        if spec.namespace not in self._edge_store:
            self._edge_store[spec.namespace] = {}
            self._namespaces.add(spec.namespace)
        
        for idx, edge in enumerate(spec.edges):
            try:
                # Validate
                if not edge.label:
                    raise BadRequest("edge.label must be non-empty")
                
                # Upsert
                await self._client.upsert_edge(
                    id=str(edge.id),
                    src=str(edge.src),
                    dst=str(edge.dst),
                    label=edge.label,
                    properties=edge.properties or {},
                    namespace=spec.namespace
                )
                
                # Update local cache
                self._edge_store[spec.namespace][str(edge.id)] = edge
                upserted += 1
                
            except Exception as e:
                failures.append({
                    "index": idx,
                    "id": str(edge.id),
                    "error": type(e).__name__,
                    "code": getattr(e, "code", None) or type(e).__name__.upper(),
                    "message": str(e)
                })
                self._stats["error_count"] += 1
        
        self._stats["total_edges_upserted"] += upserted
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
        
        return UpsertResult(
            upserted_count=upserted,
            failed_count=len(failures),
            failures=failures
        )
    
    # ----------------------------------------------------------------------
    # DELETE NODES (Idempotent)
    # ----------------------------------------------------------------------
    
    async def _do_delete_nodes(self, spec: DeleteNodesSpec, *, ctx=None) -> DeleteResult:
        """MANDATORY: Delete is IDEMPOTENT: no error on missing IDs."""
        self._stats["delete_nodes_calls"] += 1
        t0 = time.monotonic()
        
        deleted = 0
        
        if spec.ids:
            for node_id in spec.ids:
                key = str(node_id)
                if spec.namespace in self._store and key in self._store[spec.namespace]:
                    del self._store[spec.namespace][key]
                    deleted += 1
                # ✅ ID not found: continue silently, no error
        
        elif spec.filter:
            # Delete by filter
            if spec.namespace in self._store:
                to_delete = []
                for vid, v in self._store[spec.namespace].items():
                    if self._filter_match(v.properties, spec.filter):
                        to_delete.append(vid)
                
                for vid in to_delete:
                    del self._store[spec.namespace][vid]
                    deleted += 1
        
        self._stats["total_nodes_deleted"] += deleted
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
        
        return DeleteResult(
            deleted_count=deleted,
            failed_count=0,
            failures=[]
        )
    
    # ----------------------------------------------------------------------
    # DELETE EDGES (Idempotent)
    # ----------------------------------------------------------------------
    
    async def _do_delete_edges(self, spec: DeleteEdgesSpec, *, ctx=None) -> DeleteResult:
        """MANDATORY: Delete is IDEMPOTENT: no error on missing IDs."""
        self._stats["delete_edges_calls"] += 1
        t0 = time.monotonic()
        
        deleted = 0
        
        if spec.ids:
            for edge_id in spec.ids:
                key = str(edge_id)
                if spec.namespace in self._edge_store and key in self._edge_store[spec.namespace]:
                    del self._edge_store[spec.namespace][key]
                    deleted += 1
                # ✅ ID not found: continue silently, no error
        
        elif spec.filter:
            # Delete by filter
            if spec.namespace in self._edge_store:
                to_delete = []
                for eid, e in self._edge_store[spec.namespace].items():
                    if self._filter_match(e.properties, spec.filter):
                        to_delete.append(eid)
                
                for eid in to_delete:
                    del self._edge_store[spec.namespace][eid]
                    deleted += 1
        
        self._stats["total_edges_deleted"] += deleted
        self._stats["total_processing_time_ms"] += (time.monotonic() - t0) * 1000
        
        return DeleteResult(
            deleted_count=deleted,
            failed_count=0,
            failures=[]
        )
    
    # ----------------------------------------------------------------------
    # HEALTH
    # ----------------------------------------------------------------------
    
    async def _do_health(self, *, ctx=None) -> Dict[str, Any]:
        """Health check."""
        try:
            healthy = await self._client.health_check()
            return {
                "ok": healthy,
                "status": "ok" if healthy else "degraded",
                "server": "my-graph-provider",
                "version": "1.0.0",
                "namespaces": {
                    ns: "ok" if healthy else "degraded"
                    for ns in self._namespaces
                }
            }
        except Exception:
            return {
                "ok": False,
                "status": "down",
                "server": "my-graph-provider",
                "version": "1.0.0"
            }
    
    # ----------------------------------------------------------------------
    # UTILITIES
    # ----------------------------------------------------------------------
    
    def _get_timeout(self, ctx):
        if ctx is None:
            return None
        rem = ctx.remaining_ms()
        if rem is None or rem <= 0:
            return None
        return rem / 1000.0
    
    def _map_provider_error(self, e: Exception):
        """Map provider errors to canonical Corpus errors."""
        from corpus_sdk.graph.graph_base import (
            BadRequest, AuthError, ResourceExhausted,
            TransientNetwork, Unavailable, NotSupported,
            DeadlineExceeded
        )
        
        if "rate limit" in str(e).lower():
            return ResourceExhausted("Rate limit exceeded", retry_after_ms=5000)
        if "auth" in str(e).lower() or "key" in str(e).lower():
            return AuthError("Authentication failed")
        if "timeout" in str(e).lower():
            return TransientNetwork("Request timeout")
        if "not found" in str(e).lower():
            return BadRequest(str(e))
        if "not supported" in str(e).lower():
            return NotSupported(str(e))
        
        return Unavailable(f"Provider error: {type(e).__name__}")
    
    def _filter_match(self, properties: Optional[Dict], filter: Optional[Dict]) -> bool:
        """Match properties against filter."""
        if not filter:
            return True
        if not properties:
            return False
        
        for k, v in filter.items():
            if properties.get(k) != v:
                return False
        return True
    
    async def _get_vertex_count(self, namespace: str) -> int:
        """Get vertex count for namespace."""
        return len(self._store.get(namespace, {}))
    
    async def _get_vertex_by_index(self, index: int, namespace: str) -> Node:
        """Get vertex by index (for pagination)."""
        items = list(self._store.get(namespace, {}).items())
        if index < len(items):
            return items[index][1]
        raise IndexError("vertex not found")
```

---

## 11. CACHE OWNERSHIP BOUNDARY: CRITICAL

### 11.1 Embedding Stats: NO Cache Metrics: MANDATORY

```python
# ❌ WRONG: DO NOT include cache metrics in adapter stats
async def _do_get_stats(self, *, ctx=None):
    return {
        "total_requests": ...,
        "cache_hits": self._cache_hits,      # ❌ Base owns this
        "cache_misses": self._cache_misses,  # ❌ Base owns this
    }

# ✅ CORRECT: Adapter stats = provider metrics ONLY
async def _do_get_stats(self, *, ctx=None):
    from corpus_sdk.embedding.embedding_base import EmbeddingStats
    return EmbeddingStats(
        total_requests=self._stats["total_ops"],
        total_texts=self._stats["total_texts"],
        total_tokens=self._stats["total_tokens"],
        avg_processing_time_ms=self._stats["avg_ms"],
        error_count=self._stats["errors"]
        # NO cache_hits, NO cache_misses
    )
```

**RULE:** `_do_get_stats()` MUST NOT include `cache_hits`, `cache_misses`, or any stream stats aggregated by base. Base owns these counters. Duplicating them causes double-counting in metrics and conformance failures.

### 11.2 Capabilities Caching: Allowed, With Rules: MANDATORY

```python
class MyLLMAdapter(BaseLLMAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._caps_cache = None
        self._caps_cache_time = 0
        self._caps_cache_ttl_s = 60  # Refresh every 60 seconds
    
    async def _do_capabilities(self) -> LLMCapabilities:
        """Caching allowed, with TTL."""
        
        now = time.time()
        
        # Return cached if fresh
        if self._caps_cache and (now - self._caps_cache_time) < self._caps_cache_ttl_s:
            return self._caps_cache
        
        # Fetch fresh capabilities
        caps = await self._fetch_capabilities_from_provider()
        
        # Update cache
        self._caps_cache = caps
        self._caps_cache_time = now
        
        return caps
```

**RULE:** Caching capabilities is ALLOWED and RECOMMENDED for performance. You MUST:
- Set a reasonable TTL (60 seconds recommended)
- Refresh periodically
- Document caching behavior
- NEVER use stale caps for enforcement decisions

---

## 12. BATCH FAILURE MODE: DECISION MATRIX

**YOU MUST CHOOSE ONE. DOCUMENT YOUR CHOICE. NEVER MAKE CONFIGURABLE.**

| Component | Recommended Mode | When To Use |
|-----------|-----------------|-------------|
| **Embedding** | Collect per-item | Provider supports partial batch; per-item errors don't fail entire batch |
| **Embedding** | Fail-fast | Provider rejects entire batch on any validation error |
| **Vector Upsert** | Fail-fast | Upsert validation errors (dimension, namespace) are request-level; no partial writes |
| **Vector Delete** | Continue on missing | Delete is idempotent; missing IDs are not errors |
| **Graph Batch** | Collect per-item | Batch operations can partially succeed; per-op errors don't fail entire batch |
| **Graph Transaction** | Fail-fast | Atomicity required; any operation failure aborts entire transaction |

**DOCUMENTATION REQUIREMENT:**

```python
class MyEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    BATCH FAILURE MODE: Collect per-item failures.
    
    This adapter processes all items in the batch and collects individual failures.
    The batch operation always succeeds (HTTP 200) with partial results.
    Failures are reported in the `failed_texts` array with index, error, and message.
    
    Rationale: Provider supports partial batch processing and returns per-item status.
    """
```

---

## 13. STREAMING PATTERN: DECISION MATRIX

**YOU MUST CHOOSE ONE. DOCUMENT YOUR CHOICE. NEVER MAKE CONFIGURABLE.**

| Pattern | Description | When To Use |
|---------|-------------|-------------|
| **Single-chunk** | One final chunk with complete vector | Provider returns entire embedding at once; no partial results |
| **Progressive** | Partial vectors growing to full dimension | Provider streams vector components incrementally |
| **Multi-vector** | Multiple complete vectors per chunk | Provider returns multiple embedding variations or batch streaming |

**DOCUMENTATION REQUIREMENT:**

```python
class MyEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    STREAMING PATTERN: Single-chunk.
    
    This adapter emits exactly one chunk with is_final=True containing the complete
    embedding vector. No partial vectors are emitted.
    
    Rationale: Provider returns entire embedding in a single response.
    """
```

---

## 14. PRODUCTION HARDENING: REMOVE ALL MOCK ONLY CODE

### 14.1 Patterns to DELETE Entirely

```python
# ❌ DELETE ALL OF THESE: MOCK-ONLY, NOT FOR PRODUCTION

# 1. Context attribute error injection
ctx.attrs.get("simulate_error")
ctx.attrs.get("fail") 
ctx.attrs.get("sleep_ms")
ctx.attrs.get("health") == "degraded"

# 2. Probabilistic failures
self.failure_rate = 0.0
def _random_failure(self): ...
if self._rng.random() < self.failure_rate: ...

# 3. RNG/randomness
self._rng = random.Random()
self._make_rng()
_seeded_rng()

# 4. Artificial latency simulation
_sleep_random()
simulate_latency = True
await asyncio.sleep(dur_ms / 1000.0)  # Except for rate limiting

# 5. Configurable capabilities
supports_batch=True  # Constructor param: NO
supports_streaming=True  # Constructor param: NO
collect_failures_in_native_batch=True  # Constructor param: NO

# 6. Test/demo code
__main__ block
if __name__ == "__main__":
env var seeding (VECTOR_SEED_DEFAULT)

# 7. Token approximations
_approx_tokens()
len(text.split()) + 2

# 8. Temperature simulation
_apply_temperature()
dup_thresh = 0.05 * t
drop_thresh = 0.10 * t

# 9. Deterministic hashing for IDs
_stable_int()
hashlib.sha256(repr(obj).encode()).hexdigest()
```

### 14.2 Patterns to TRANSFORM

```python
# ✅ KEEP but TRANSFORM for production

# 1. Validation logic → Keep, make strict
# FROM (mock): permissive validation
# TO (prod): strict validation with error details

# 2. Error mapping → Keep, use real provider errors
# FROM (mock): ctx.attrs-driven errors
# TO (prod): Map real provider exceptions, import from domain base

# 3. Batch failure handling → Choose ONE mode, remove config
# FROM (mock): Configurable via flag
# TO (prod): Hardcoded, documented behavior

# 4. Streaming patterns → Choose ONE, remove others
# FROM (mock): Configurable stream_chunk_pattern
# TO (prod): Single pattern, hardcoded

# 5. Capabilities → Hardcode, don't configure
# FROM (mock): Constructor params for supports_*
# TO (prod): Hardcoded based on provider
```

---

## 15. PER DOMAIN IMPLEMENTATION CHECKLISTS

### 15.1 LLM Adapter Checklist

- [ ] **Shared planning path**: `_do_complete()` and `_do_stream()` use same planning function, return identical text
- [ ] **Tool call validation**: `tool_choice` validated against available tools; unknown tool → BadRequest with details
- [ ] **Tool call usage accounting**: Completion_tokens > 0 for tool-calling turns; synthesized from payload if provider returns 0
- [ ] **Stop sequences**: Cut at FIRST occurrence of ANY stop sequence; never include stop sequence in output
- [ ] **Tool call streaming**: Non-final chunks empty (text="", tool_calls=[]); final chunk has tool_calls populated
- [ ] **Capabilities enforcement**: If `caps.supports_X=False`, `_do_X` raises NotSupported
- [ ] **Role validation**: Permissive, not restrictive; only reject roles provider explicitly cannot handle
- [ ] **Token counting**: Accurate (tiktoken or provider API); NO word-count approximations
- [ ] **Deadline propagation**: `ctx.remaining_ms()` converted to provider timeout
- [ ] **Error mapping**: All provider errors map to canonical Corpus errors (imported from `llm_base`) with complete detail schemas
- [ ] **NO mock patterns**: No `ctx.attrs`, no RNG, no `failure_rate`, no `_sleep()`, no configurable capabilities

### 15.2 Embedding Adapter Checklist

- [ ] **Validation placement**: `_do_embed()` validates non-empty string, model support; does NOT assume base validated
- [ ] **Batch failure mode**: CHOOSE ONE (collect per-item OR fail-fast); DOCUMENTED; NOT configurable
- [ ] **Cache stats ownership**: `_do_get_stats()` EXCLUDES `cache_hits`, `cache_misses`; base owns these
- [ ] **Streaming pattern**: CHOOSE ONE (single-chunk, progressive, multi-vector); DOCUMENTED; NOT configurable
- [ ] **Capabilities**: Hardcoded, NOT configurable via constructor (`supports_batch`, `supports_streaming`)
- [ ] **Token counting**: Accurate (provider API or tiktoken); NO `_approx_tokens()`
- [ ] **Truncation/normalization**: Base handles; you only report `normalizes_at_source`
- [ ] **Batch partial failures**: Failure objects include `index`, `error`, `code`, `message`, `metadata` (if available)
- [ ] **NO mock patterns**: No `ctx.attrs`, no RNG, no `failure_rate`, no `_sleep()`, no configurable capabilities
- [ ] **Error imports**: Import error classes from `corpus_sdk.embedding.embedding_base`

### 15.3 Vector Adapter Checklist

- [ ] **Namespace authority**: `UpsertSpec.namespace` is authoritative; mismatches → BadRequest with complete details
- [ ] **Include vectors contract**: `include_vectors=False` → `Vector.vector = []` (empty list), NOT null, NOT omitted
- [ ] **Filter dialect validation**: Unknown operators → BadRequest with `supported` list in details; no silent ignore
- [ ] **Batch query atomicity**: All or nothing; any invalid query → entire batch fails; no partial results
- [ ] **Delete idempotency**: No error on missing IDs; `deleted_count` = actual deletions, NOT attempts
- [ ] **Delete parameter rule**: Must provide either `ids` OR `filter`; not both, not neither
- [ ] **Distance metrics**: EXACT strings `"cosine"`, `"euclidean"`, `"dotproduct"`; no variations
- [ ] **Suggested batch reduction**: PERCENTAGE (0-100), not absolute number; used in `BadRequest`
- [ ] **IndexNotReady**: Always includes `retry_after_ms` (default 500); includes `namespace` in details
- [ ] **Health response**: Includes per-namespace status with `dimensions`, `metric`, `count`, `status`
- [ ] **Error detail schemas**: DimensionMismatch, NamespaceMismatch include ALL required fields (expected, actual, namespace, vector_id, index)
- [ ] **NO mock patterns**: No `ctx.attrs`, no RNG, no `failure_rate`, no `_sleep()`, no `VECTOR_SEED_*`
- [ ] **Error imports**: Import error classes from `corpus_sdk.vector.vector_base`
- [ ] **Mathematical precision**: Passes all 7 test vectors with < 0.0001 drift; handles zero vectors without errors
- [ ] **Normalization flag**: `normalizes_at_source` correctly reflects provider behavior; no double normalization

### 15.4 Graph Adapter Checklist

- [ ] **Batch result envelope**: `{"ok": True/False, "result": {...}, "error": ...}` format; used for all batch/transaction ops
- [ ] **Shared op executor**: Single execution kernel used by BOTH `_do_batch()` and `_do_transaction()`
- [ ] **Dialect validation**: TWO layers — in `_do_query()` AND re-validated in batch executor
- [ ] **Delete idempotency**: No error on missing IDs; `deleted_count` = actual deletions
- [ ] **Bulk vertices pagination**: `next_cursor` (string or None), `has_more` (bool); cursor format opaque
- [ ] **Traversal result shape**: Includes `nodes`, `relationships`, `paths` arrays; nodes deduplicated
- [ ] **Capabilities enforcement**: If `caps.supports_X=False`, `_do_X` raises NotSupported
- [ ] **Capabilities**: Hardcoded, NOT configurable via constructor (`supports_transaction`, `supports_traversal`)
- [ ] **NO mock patterns**: No `ctx.attrs`, no RNG, no `_stable_int()`, no `_sleep()`, no configurable capabilities
- [ ] **Error imports**: Import error classes from `corpus_sdk.graph.graph_base`

---

## 16. COMMON PITFALLS: CONFORMANCE FAILURES

### LLM PITFALLS 

1. **Mismatched complete/stream output**: `_do_complete()` and `_do_stream()` return different text ❌
2. **No tool choice validation**: Accepts `tool_choice` with unknown tool name ❌
3. **Zero completion tokens for tool calls**: Reports 0 tokens for tool-calling turns ❌
4. **Stop sequence at LAST occurrence**: Cuts at last occurrence instead of first ❌
5. **Tool calls in non-final chunks**: Emits tool calls before final chunk ❌
6. **Missing final tool call chunk**: Stream ends without tool_calls in final chunk ❌
7. **Operation without capability**: Implements `_do_stream()` when `supports_streaming=False` ❌
8. **Overly restrictive roles**: Rejects `tool`, `function`, `developer` roles ❌
9. **Word-count token approximation**: Uses `len(text.split())` for token counting ❌
10. **Configurable capabilities**: `supports_streaming` as constructor param ❌
11. **Context attribute error injection**: Reads `ctx.attrs["simulate_error"]` ❌
12. **Temperature simulation**: Manually duplicates/drops tokens ❌
13. **Wrong error imports**: Imports from non-existent `corpus_sdk.llm.exceptions` ❌

### EMBEDDING PITFALLS 

14. **No validation in _do_embed**: Assumes base validated non-empty string ❌
15. **Configurable batch failure mode**: `collect_failures_in_native_batch` flag ❌
16. **Cache stats in _do_get_stats**: Includes `cache_hits`, `cache_misses` ❌
17. **Configurable streaming pattern**: `stream_chunk_pattern` parameter ❌
18. **Configurable capabilities**: `supports_batch`, `supports_streaming` as constructor params ❌
19. **Token approximation**: `_approx_tokens()` word-count hack ❌
20. **Artificial latency**: `_sleep_random()`, `simulate_latency` flag ❌
21. **RNG for vectors**: Seeded random for deterministic vectors ❌
22. **Probabilistic failures**: `failure_rate`, `_random_failure()` ❌
23. **Dummy cache**: In-memory dict cache instead of TTL cache ❌
24. **Missing batch failure details**: Failure objects missing `index`, `code`, `message` ❌
25. **Null failed_texts**: Returns `None` instead of empty array ❌
26. **No model validation**: Accepts unsupported models ❌
27. **Normalizes_at_source misreporting**: Returns normalized vectors but sets `normalizes_at_source=False` ❌
28. **Wrong error imports**: Imports from non-existent `corpus_sdk.embedding.exceptions` ❌

### VECTOR PITFALLS 

29. **No namespace authority**: Allows vector.namespace != spec.namespace ❌
30. **Silent namespace correction**: Overwrites vector.namespace to match spec ❌
31. **include_vectors=False returns null**: Returns `null` or omits field instead of `[]` ❌
32. **Silent filter operator ignore**: Ignores unknown operators instead of raising BadRequest ❌
33. **Missing filter error details**: No `supported` list in error details ❌
34. **Batch query partial execution**: Continues after query failure ❌
35. **Delete errors on missing**: Raises error when ID not found ❌
36. **Delete count = attempted**: Reports `len(ids)` instead of actual deletions ❌
37. **Delete accepts both ids AND filter**: No validation for mutual exclusivity ❌
38. **Wrong metric strings**: Uses `"cosine_sim"`, `"l2"`, `"dot"` instead of canonical strings ❌
39. **Absolute batch reduction**: Returns absolute number instead of percentage ❌
40. **IndexNotReady missing retry_after_ms**: No retry hint provided ❌
41. **Missing namespace in IndexNotReady details**: Omitted ❌
42. **DimensionMismatch missing fields**: Missing `expected`, `actual`, `namespace`, `vector_id`, `index` ❌
43. **NamespaceMismatch missing fields**: Missing `spec_namespace`, `vector_namespace`, `vector_id`, `index` ❌
44. **Health missing namespace status**: No per-namespace metrics ❌
45. **Context attribute failure injection**: Reads `ctx.attrs["fail"]` ❌
46. **Environment test seeding**: `VECTOR_SEED_DEFAULT` env var ❌
47. **Wrong error imports**: Imports from non-existent `corpus_sdk.vector.exceptions` ❌
48. **Cosine similarity math**: Returns `1 - cosine_distance` incorrectly ❌
49. **Euclidean squared distance**: Returns Σ(Aᵢ - Bᵢ)² without square root ❌
50. **Double normalization**: `normalizes_at_source=True` but base normalizes again ❌
51. **Zero vector division**: Crashes on zero vector input ❌
52. **Unclamped cosine scores**: Returns values outside [-1.0, 1.0] ❌

### GRAPH PITFALLS

53. **Wrong batch result format**: Returns raw provider response, not `{"ok": True, "result": ...}` ❌
54. **Duplicated batch/transaction logic**: Separate implementations for batch and transaction ❌
55. **No dialect re-validation in batch**: Assumes base validated, skips check ❌
56. **Delete errors on missing**: Raises error when node/edge not found ❌
57. **Bulk vertices missing pagination fields**: No `next_cursor` or `has_more` ❌
58. **Traversal missing paths**: Returns nodes/edges but no `paths` array ❌
59. **Operation without capability**: Implements `_do_transaction` when `supports_transaction=False` ❌
60. **Configurable capabilities**: `supports_transaction_ops`, `supports_traversal_ops` as constructor params ❌
61. **Deterministic hash IDs**: Uses `_stable_int()` for ID generation ❌
62. **Context attribute error injection**: Reads `ctx.attrs["simulate_error"]` ❌
63. **Artificial latency**: `_sleep()` with fixed delay ❌
64. **Wrong error imports**: Imports from non-existent `corpus_sdk.graph.exceptions` ❌

---

**END OF IMPLEMENTATION GUIDE**
