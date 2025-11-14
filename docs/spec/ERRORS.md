# ERRORS.md

**Corpus Protocol Suite & SDKs — Normalized Error Taxonomy**  
**errors_version:** `1.0`

> This document defines the **normalized error model** used across **Graph**, **LLM**, **Vector**, and **Embedding** adapters. It aligns with the Specification's Common Foundation (§6.3), Error Handling & Resilience (§12), and protocol-specific sections (§7.6 Graph health, §8.5 LLM, §9.5 Vector, §10.3 Embedding).  
> It is **implementation-agnostic** and compatible with any transport (HTTP, gRPC, queues) and any observability backend.

---

## 0) Goals & Non-Goals

### Goals

- One **cross-protocol taxonomy** with consistent semantics.
- Deterministic, machine-readable **retryability** and **hints**.
- Privacy-preserving, SIEM-safe payloads.
- Predictable transport mappings for HTTP/gRPC.
- Stable naming and documented **versioning** rules.

### Non-Goals

- Exhaustive backend-specific error catalogs.
- Mandating specific transports or frameworks.

---

## 1) Normalized Error Classes (Canonical)

```
AdapterError (base)
├─ BadRequest               # 400 client errors (validation, schema, ranges)
├─ AuthError                # 401/403 credentials/permissions
├─ ResourceExhausted        # 429 quotas, rate limits (retry-after)
├─ TransientNetwork         # 502/504 network flaps, gateway timeouts
├─ Unavailable              # 503 backend temporary unavailability/overload
├─ NotSupported             # 501 or 400 for unsupported operation/parameter
└─ DeadlineExceeded         # 504 absolute deadline/budget exhausted
```

### 1.1 Protocol-Specific Subtypes (Full Expansion)

All subtypes MUST inherit from one of the canonical classes above and MUST NOT contradict the parent's retry semantics.

*Note: Some subtypes are shared across protocols (e.g., ProviderQuotaExceeded).*

| Subtype | Parent Class | Protocol(s) | Description |
|---------|--------------|-------------|-------------|
| ModelNotFound | BadRequest | LLM, Embedding | Requested model does not exist / is not configured |
| ModelOverloaded | Unavailable | LLM | Model transiently at capacity |
| PromptTooLong | BadRequest | LLM | Prompt/messages exceed model max context |
| ContentFiltered | BadRequest | LLM, Embedding | Provider content/safety filter triggered |
| SafetyPolicyViolation | BadRequest | LLM | Safety system blocked the request |
| UnsupportedModelFamily | NotSupported | LLM, Embedding | Model family or type unsupported |
| InputFormatError | BadRequest | LLM, Embedding, Graph | Invalid input structure/schema/JSON |
| TaskRejected | Unavailable | LLM | Request rejected due to load/capacity policy |
| ThroughputLimitExceeded | ResourceExhausted | LLM, Vector, Graph | Per-tenant or per-resource throughput limit exceeded |
| LatencySLAExceeded | Unavailable | LLM | Provider cannot meet requested latency target/class |
| TextTooLong | BadRequest | Embedding | Input text exceeds embedding model max and truncate=false |
| EmbeddingDimensionMismatch | BadRequest | Embedding | Expected output dims vs provider dims mismatch |
| ProviderQuotaExceeded | ResourceExhausted | LLM, Embedding, Vector, Graph | Provider-level quota exceeded (tokens, requests, etc.) |
| DimensionMismatch | BadRequest | Vector | Query/upsert vector dims ≠ namespace dims |
| IndexNotReady | Unavailable | Vector, Graph | Index/namespace exists but not yet ready/empty/building |
| NamespaceNotFound | BadRequest | Vector, Graph | Namespace/graph/collection not found |
| FilterSyntaxError | BadRequest | Vector, Graph | Invalid filter/query predicate syntax |
| QueryParseError | BadRequest | Vector, Graph | Query text (dialect) cannot be parsed |
| IndexCorrupt | Unavailable | Vector, Graph | Index corrupted or inconsistent |
| ShardUnavailable | Unavailable | Vector, Graph | Shard/partition temporarily unavailable |
| SchemaValidationError | BadRequest | Graph | Graph schema violation (labels, properties, constraints) |
| VertexNotFound | BadRequest | Graph | Vertex ID not found where required |
| EdgeNotFound | BadRequest | Graph | Edge ID not found where required |

**Stability:** Canonical classes and the above subtype names + meanings are frozen for `errors_version=1.0`. New subtypes MAY be added only if they refine an existing parent class and preserve retry semantics.

---

## 2) Programmatic Error Envelope (Wire-Level)

Adapters SHOULD surface errors using the following machine-readable envelope:

```json
{
  "ok": false,
  "error": "ResourceExhausted",
  "message": "Rate limit exceeded for tenant",
  "code": "RATE_LIMIT",
  "http_status": 429,
  "retry_after_ms": 1200,
  "resource_scope": "rate_limit",
  "throttle_scope": "tenant:acme:llm",
  "suggested_batch_reduction": 50,
  "details": {
    "max_batch_size": 1000,
    "provided_batch_size": 2400
  }
}
```

### 2.1 Normative Rules

- `message` MUST be present and human-readable (no secrets, prompts, vectors, IDs).
- `code` is an adapter-specific short identifier for logs and dashboards. It is not the same as the metrics code label (see §5).
- `retry_after_ms` MUST be non-negative when present; else null or omitted.
- `details` MUST be JSON-serializable and SIEM-safe.
- Subtypes MUST set `error` equal to the subtype name (e.g. `"TextTooLong"`).

---

## 3) Transport Mappings

### 3.1 HTTP (Recommended)

| Canonical Class | HTTP Status |
|-----------------|-------------|
| BadRequest | 400 |
| AuthError | 401 / 403 |
| ResourceExhausted | 429 |
| NotSupported | 501 (or 400 for invalid params) |
| TransientNetwork | 502 / 504 |
| Unavailable | 503 |
| DeadlineExceeded | 504 |

### 3.2 gRPC (Informative)

| Canonical Class | gRPC Code |
|-----------------|-----------|
| BadRequest | INVALID_ARGUMENT |
| AuthError | UNAUTHENTICATED / PERMISSION_DENIED |
| ResourceExhausted | RESOURCE_EXHAUSTED |
| NotSupported | UNIMPLEMENTED |
| TransientNetwork | UNAVAILABLE / DEADLINE_EXCEEDED |
| Unavailable | UNAVAILABLE |
| DeadlineExceeded | DEADLINE_EXCEEDED |

---

## 4) Retry Semantics (Normative)

### 4.1 Canonical Classes

| Class | Retryable? | Client Guidance |
|-------|------------|----------------|
| BadRequest | No | Fix parameters / schema before retry |
| AuthError | No | Refresh credentials/permissions |
| ResourceExhausted | Yes | Backoff; honor retry_after_ms; reduce concurrency/batch |
| TransientNetwork | Yes | Exponential backoff + jitter; consider failover |
| Unavailable | Yes | Backoff; treat as overload; use breakers |
| NotSupported | No | Probe capabilities() and adjust feature/parameter |
| DeadlineExceeded | Conditional | Retry only if deadline increased or work reduced |

### 4.2 Subtypes

| Subtype | Retryable? | Notes |
|---------|------------|-------|
| ModelNotFound | No | Fix model name or configuration |
| ModelOverloaded | Yes | Backoff; try alternative model/family |
| PromptTooLong | No | Shorten prompt or enable truncation (truncate=true) |
| ContentFiltered | No | Sanitize or change content |
| SafetyPolicyViolation | No | Same as ContentFiltered but explicit to safety system |
| UnsupportedModelFamily | No | Use supported family; probe capabilities |
| InputFormatError | No | Fix JSON/field types/role schema |
| TaskRejected | Yes | Backoff; may require lower concurrency or different priority |
| ThroughputLimitExceeded | Yes | Backoff; reduce concurrency/batch; honor retry_after_ms |
| LatencySLAExceeded | Conditional | Retry w/ more relaxed latency or smaller workload |
| TextTooLong | No | Enable truncation or split content |
| EmbeddingDimensionMismatch | No | Fix expected dims / configuration |
| ProviderQuotaExceeded | Yes | Retry after admin intervention or quota reset; honor retry_after_ms |
| DimensionMismatch | No | Align vector dims with namespace |
| IndexNotReady | Yes | Retry after delay; honor retry_after_ms |
| NamespaceNotFound | No | Create namespace or fix name |
| FilterSyntaxError | No | Fix filter syntax |
| QueryParseError | No | Fix query text / dialect |
| IndexCorrupt | Yes | Operational fix; usually requires manual repair |
| ShardUnavailable | Yes | Backoff; failover if possible |
| SchemaValidationError | No | Fix graph schema / labels / edge types |
| VertexNotFound | No | Fix vertex ids or query |
| EdgeNotFound | No | Fix edge ids or query |

---

## 5) Observability & Privacy (SIEM-Safe)

- For metrics (see METRICS.md), the canonical class name or subtype name (e.g., `"Unavailable"`, `"IndexNotReady"`) MUST be used as the metrics code label — not the adapter-level envelope code string.
- Never emit raw prompts, vectors, embeddings, tenant IDs, doc IDs, or arbitrary free-text fields.
- Use tenant hashing where tenant participates in labels or details.
- `details` MUST be low-cardinality and JSON-safe.

---

## 6) Protocol-Specific Supplements (Full Detail)

### 6.1 LLM

LLM adapters MUST map common failures into the taxonomy as follows:

**Validation / Request Errors (BadRequest)**
- **PromptTooLong**
  - When: combined messages + system prompt exceed model's max context.
  - `details` SHOULD include: `{"max_context_length": <int>, "provided_tokens": <int>, "model": "<name>"}`
- **ContentFiltered / SafetyPolicyViolation**
  - When: safety/content filters block the request.
  - No prompts or content must be included in details.
  - `details` may include: `{"policy_section": "safety.v2", "category": "HATE"}`
- **InputFormatError**
  - When: messages schema invalid (unknown role, missing content, bad tools)
  - `details` may include: fields that failed validation (names only).
- **ModelNotFound**
  - When: backend rejects unknown model name.
  - `details`: `{"model": "<requested_model>"}`
- **UnsupportedModelFamily**
  - When: adapter does not support requested model family.

**Capacity / Quota / Latency**
- **ModelOverloaded (Unavailable)**
  - When: model is temporarily at capacity.
  - Should often include `retry_after_ms`.
- **TaskRejected (Unavailable)**
  - When: provider rejects due to overload or scheduler policy.
- **ThroughputLimitExceeded / ProviderQuotaExceeded (ResourceExhausted)**
  - When: per-tenant throughput or provider-level quota hit.
- **LatencySLAExceeded (Unavailable)**
  - When: provider cannot meet requested latency/priority.

**Deadline & Network**
- Use `DeadlineExceeded` for context budgets (adapter-side or provider).
- Use `TransientNetwork` for timeouts / gateway issues.

**Streaming**
- For stream, streaming MUST follow the same classification rules as complete.
- The final error (if any) MUST be surfaced via a final LLMChunk failure and metrics `count_stream_final_outcome` with the normalized error as code.

### 6.2 Embedding

Embedding adapters MUST map failures like:

**Request & Content**
- **TextTooLong (BadRequest)**
  - When: input exceeds model length and truncation disabled.
  - `details`: `{"max_text_length": <int>, "provided_length": <int>}`
- **ContentFiltered (BadRequest)**
  - When: provider refuses embedding due to content policy.
- **InputFormatError (BadRequest)**
  - When: request body malformed, non-string text elements, etc.
- **ModelNotFound / UnsupportedModelFamily (BadRequest/NotSupported)**
  - When: unknown model or unsupported family for embeddings.

**Shape / Dimensions**
- **EmbeddingDimensionMismatch (BadRequest)**
  - When: caller expects a certain output dims but provider yields different dims or adapter config mismatched.
  - `details`: `{"expected_dims": <int>, "actual_dims": <int>, "model": "<name>"}`

**Quota**
- **ProviderQuotaExceeded, ThroughputLimitExceeded (ResourceExhausted)**
  - When hitting token or embed count quotas.

**Batch Semantics**
- When an entire batch is rejected (e.g., too large), use `BadRequest` with `suggested_batch_reduction` and `details.max_batch_size`.
- Per-item failures inside `embed_batch` MUST be represented in `failed_texts` with a reduced shape (see §10).

### 6.3 Vector

Vector adapters MUST:

**Dimension & Namespace**
- **DimensionMismatch (BadRequest)**
  - When: `len(vector) != namespace.dimensions`.
  - `details`: `{"expected": <int>, "provided": <int>, "namespace": "<ns>"}`
- **NamespaceNotFound (BadRequest)**
  - When: reference to unknown namespace / collection.

**Index State**
- **IndexNotReady (Unavailable)**
  - When: namespace exists but index/data is not ready (empty, rebuilding).
  - SHOULD include `retry_after_ms`.
- **IndexCorrupt (Unavailable)**
  - When: index inconsistency or corruption detected.
- **ShardUnavailable (Unavailable or TransientNetwork)**
  - When: a shard/partition is down.
  - Choose `Unavailable` for long-lived issues, `TransientNetwork` for short-lived routing failures.

**Query & Filter**
- **FilterSyntaxError (BadRequest)**
  - When: filter expression cannot be parsed.
- **QueryParseError (BadRequest)**
  - When: query DSL expression invalid (if applicable).

**Quota / Rate / Throughput**
- **ThroughputLimitExceeded, ProviderQuotaExceeded (ResourceExhausted)**
  - When hitting internal QPS or usage quotas.

**Batch Upsert/Delete**
- If entire batch is too large:
  - Use `BadRequest` with `suggested_batch_reduction` and `details.max_batch_size`.
- Per-item failures inside batch upsert/delete MUST be captured in `failures[]` with reduced error records (see §10).

### 6.4 Graph

Graph adapters MUST mirror Vector/LLM level of detail:

**Syntax & Dialect**
- **QueryParseError (BadRequest)**
  - When: graph query (Cypher, openCypher, GQL, etc.) fails to parse.
- **FilterSyntaxError (BadRequest)**
  - When: property filters or query predicates are malformed.
- **NotSupported**
  - When: dialect not supported, query feature not supported, or `_do_capabilities()` indicates no support for a requested op.

**Schema & Entities**
- **SchemaValidationError (BadRequest)**
  - When: labels, property keys, or edge types violate configured schema.
- **VertexNotFound / EdgeNotFound (BadRequest)**
  - When: referenced IDs do not exist where existence is required.
- **NamespaceNotFound (BadRequest)**
  - When: graph/namespace not found.

**Index & Shards**
- **IndexNotReady (Unavailable)**
  - When: index exists but still building; may occur in `bulk_vertices` or query.
  - SHOULD include `retry_after_ms`.
- **IndexCorrupt (Unavailable)**
  - When: internal index corruption is detected.
- **ShardUnavailable (Unavailable/TransientNetwork)**
  - When: specific shard/partition is down.

**Quota / Throughput**
- **ThroughputLimitExceeded, ProviderQuotaExceeded (ResourceExhausted)**
  - When hitting graph engine throughput or quota limits.

**Batch Ops**
- `_do_batch` MUST:
  - Use canonical classes for top-level batch failure (e.g., invalid size → `BadRequest`).
  - Use per-item reduced failure records for each failed operation in the batch (`failures[]`).

**Streaming Queries**
- `stream_query` MUST:
  - Use the same normalization rules as `query`.
  - Emit a single final error if the stream fails, plus one `count_stream_final_outcome` metric with the normalized error/code.

---

## 7) Error Hints (Machine-Readable Mitigations)

Adapters SHOULD attach structured hints:
- `retry_after_ms` — recommended wait time before retry.
- `resource_scope` — one of: `"model" | "token_limit" | "rate_limit" | "memory" | "compute" | "time_budget" | "index" | "shard"`
- `throttle_scope` — bounded identifier for throttling domain (e.g., `"tenant:acme:llm"`, `"tenant:acme:graph"`).
- `suggested_batch_reduction` — percentage [0..100].

**Client rule:**

Clients SHOULD reduce their next batch size as:

```
new_size = ceil(old_size * (100 - suggested_batch_reduction) / 100)
```

`details` SHOULD be used for low-cardinality, structured context (e.g. `"max_batch_size"`, `"max_top_k"`, `"max_text_length"`, `"max_context_length"`).

---

## 8) Normalization Rules (Adapter Implementations)

Adapters MUST map provider-specific failures into the taxonomy:

1. **Classify** into canonical class or allowed subtype.
2. **Set** a clean message (no raw upstream content).
3. **Map** HTTP/gRPC codes per §3.
4. **Attach** hints (`retry_after_ms`, `resource_scope`, `throttle_scope`, `suggested_batch_reduction`) where meaningful.
5. **Populate** `details` with JSON-safe, low-cardinality fields.
6. **Emit** metrics (`observe_operation`, `count_operation`, and `count_stream_final_outcome` for streaming).

**Pseudocode:**

```python
def normalize(provider_exc) -> NormalizedError:
    if provider_exc.code in {"429", "RATE_LIMIT"}:
        return ResourceExhausted(
            message="Rate limit exceeded",
            code="RATE_LIMIT",
            retry_after_ms=provider_exc.retry_ms or 1000,
            resource_scope="rate_limit",
            details={"provider_code": provider_exc.code}
        )
    if provider_exc.code in {"TIMEOUT", "GATEWAY_TIMEOUT"}:
        return TransientNetwork(
            message="Upstream timeout",
            code="GATEWAY_TIMEOUT"
        )
    if provider_exc.code in {"MODEL_NOT_FOUND"}:
        return ModelNotFound(
            message="Model not found",
            code="MODEL_NOT_FOUND",
            details={"model": provider_exc.model}
        )
    # ... additional mappings ...
    return Unavailable(message="Service unavailable", code="UNKNOWN_UPSTREAM")
```

---

## 9) Client Backoff & Breaker Guidance

**Exponential Backoff (Recommended)**
- Base: 100–500 ms
- Factor: ×2
- Cap: 10–30 s
- Override schedule with server-provided `retry_after_ms` when present.

**Deadline-Aware Retry**
- Do not blindly retry `DeadlineExceeded`.
- Retry only after:
  - Increasing deadline, or
  - Reducing work (tokens, batch size, graph complexity).

**Circuit Breakers**
- Trip on flurries of `Unavailable` / `TransientNetwork`.
- Return `Unavailable("circuit open")` while breaker is open.
- Use `retry_after_ms` to dampen traffic when closing.

---

## 10) Partial Failure (Batches)

Batch APIs MUST surface per-item status, not just aggregate failure:

```json
{
  "upserted_count": 42,
  "failed_count": 3,
  "failures": [
    { "id": "doc-17", "error": "DimensionMismatch", "detail": "expected 1536, got 1537" },
    { "id": "doc-19", "error": "BadRequest", "detail": "vector is empty" },
    { "id": "doc-23", "error": "BadRequest", "detail": "metadata not JSON-serializable" }
  ]
}
```

- Per-item failures use a reduced shape (`error`, `detail`, optional `item-code`) — they are not full envelopes.
- If an entire batch is rejected (e.g., too large, rate-limited), the top-level error SHOULD include `suggested_batch_reduction`.

---

## 11) Examples (Per Protocol)

**LLM — ContentFiltered**

```json
{
  "ok": false,
  "error": "ContentFiltered",
  "message": "Input violates content policy",
  "code": "CONTENT_FILTERED",
  "http_status": 400,
  "retry_after_ms": null,
  "resource_scope": "model",
  "details": { "policy_section": "safety.v2" }
}
```

**Embedding — TextTooLong**

```json
{
  "ok": false,
  "error": "TextTooLong",
  "message": "Input exceeds maximum length and truncate=false",
  "code": "TEXT_TOO_LONG",
  "http_status": 400,
  "retry_after_ms": null,
  "resource_scope": "token_limit",
  "details": { "max_text_length": 16000, "provided_length": 24210 }
}
```

**Vector — IndexNotReady**

```json
{
  "ok": false,
  "error": "IndexNotReady",
  "message": "index not ready (namespace initialized but empty)",
  "code": "INDEX_NOT_READY",
  "http_status": 503,
  "retry_after_ms": 2000,
  "resource_scope": "index",
  "details": { "namespace": "acme.docs" }
}
```

**Graph — QueryParseError**

```json
{
  "ok": false,
  "error": "QueryParseError",
  "message": "Failed to parse Cypher query",
  "code": "GRAPH_QUERY_PARSE",
  "http_status": 400,
  "retry_after_ms": null,
  "resource_scope": "model",
  "details": { "dialect": "cypher" }
}
```

---

## 12) Versioning & Deprecation (Errors)

- **errors_version:** `1.0`
- Canonical classes and retry semantics are frozen in 1.0.
- Adding new subtypes is allowed only if:
  - They refine an existing canonical class.
  - They keep retry semantics consistent.
  - They preserve low cardinality.
- **Deprecation:**
  - Mark old subtype as deprecated for ≥1 minor release.
  - Do not reuse names with different semantics.

---

## 13) Compliance Checklist

- [ ] Error uses a canonical class or allowed subtype.
- [ ] `message` present; no content or secrets.
- [ ] Envelope code is short, stable, and distinct from metrics code.
- [ ] `retry_after_ms` is valid or omitted.
- [ ] `resource_scope` provided where meaningful.
- [ ] `suggested_batch_reduction` used for batch overruns.
- [ ] `details` JSON-safe and low-cardinality.
- [ ] Transport mapping conforms to §3.
- [ ] Metrics emitted with canonical class/subtype as code label.
- [ ] Tenant hashing used in telemetry contexts.

---

## 14) FAQ

**Q: Can we emit provider raw error messages in details?**
**A:** Only if scrubbed of prompts, vectors, IDs, and sensitive text. Prefer structured fields.

**Q: When do I choose TransientNetwork vs Unavailable?**
**A:** Use `TransientNetwork` for short-lived network/gateway issues; `Unavailable` for backend capacity/health issues.

**Q: Should DeadlineExceeded include retry_after_ms?**
**A:** Typically no. Clients should adjust deadlines or workload size instead.

---

*End of ERRORS.md (errors_version 1.0)*
