# Normalized Error Taxonomy

**Table of Contents**
- [0) Goals & Non-Goals](#0-goals--non-goals)
- [1) Normalized Error Classes (Canonical)](#1-normalized-error-classes-canonical)
- [2) Programmatic Error Envelope (Wire-Level)](#2-programmatic-error-envelope-wire-level)
- [3) Transport Mappings (Non-Wire)](#3-transport-mappings-non-wire)
- [4) Retry Semantics (Normative)](#4-retry-semantics-normative)
- [5) Observability & Privacy (SIEM-Safe)](#5-observability--privacy-siem-safe)
- [6) Protocol-Specific Guidance (Informative)](#6-protocol-specific-guidance-informative)
- [7) Error Hints (Machine-Readable Mitigations)](#7-error-hints-machine-readable-mitigations)
- [8) Normalization Rules (Adapter Implementations)](#8-normalization-rules-adapter-implementations)
- [9) Client Backoff & Breaker Guidance](#9-client-backoff--breaker-guidance)
- [10) Partial Failure (Batches)](#10-partial-failure-batches)
- [11) Examples (Per Protocol)](#11-examples-per-protocol)
- [12) Versioning & Deprecation (Errors)](#12-versioning--deprecation-errors)
- [13) Compliance Checklist](#13-compliance-checklist)
- [14) FAQ](#14-faq)
- [15) Conformance Testing](#15-conformance-testing)

---

**Errors Version:** `1.0`

> This document defines the **normalized error model** used across **Graph**, **LLM**, **Vector**, and **Embedding** adapters. It aligns with the Specification's Common Foundation (§6.3), Error Handling & Resilience (§12), and protocol-specific sections (§7.6 Graph health, §8.5 LLM, §9.5 Vector, §10.3 Embedding).  

> It is **implementation-agnostic** and compatible with any transport (HTTP, gRPC, queues) and any observability backend.

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

**Canonical Wire Codes (Normative):**
- `BAD_REQUEST`
- `AUTH_ERROR`
- `RESOURCE_EXHAUSTED`
- `TRANSIENT_NETWORK`
- `UNAVAILABLE`
- `NOT_SUPPORTED`
- `DEADLINE_EXCEEDED`

**Stability:** These 7 canonical classes and their wire codes are **frozen** for `errors_version=1.0`. All adapters MUST use these as the basis for error classification.

### 1.1 Optional Subtypes (Informative / Non-Normative)

Adapters MAY provide additional specificity through subtypes, but clients MUST function correctly if they only branch on canonical classes.

| Suggested Subtype | Suggested Subtype Code | Parent Class | Protocol(s) | Description |
|-------------------|------------------------|--------------|-------------|-------------|
| ModelNotFound | MODEL_NOT_FOUND | BadRequest | LLM, Embedding | Requested model does not exist / is not configured |
| ModelOverloaded | MODEL_OVERLOADED | Unavailable | LLM | Model transiently at capacity |
| ContentFiltered | CONTENT_FILTERED | BadRequest | LLM, Embedding | Provider content/safety filter triggered |
| SafetyPolicyViolation | SAFETY_POLICY_VIOLATION | BadRequest | LLM | Safety system blocked the request |
| InputFormatError | INPUT_FORMAT_ERROR | BadRequest | LLM, Embedding, Graph | Invalid input structure/schema/JSON |
| ThroughputLimitExceeded | THROUGHPUT_LIMIT_EXCEEDED | ResourceExhausted | LLM, Vector, Graph | Per-tenant or per-resource throughput limit exceeded |
| TextTooLong | TEXT_TOO_LONG | BadRequest | Embedding | Input text exceeds embedding model max and truncate=false |
| EmbeddingDimensionMismatch | EMBEDDING_DIMENSION_MISMATCH | BadRequest | Embedding | Expected output dims vs provider dims mismatch |
| ProviderQuotaExceeded | PROVIDER_QUOTA_EXCEEDED | ResourceExhausted | LLM, Embedding, Vector, Graph | Provider-level quota exceeded |
| DimensionMismatch | DIMENSION_MISMATCH | BadRequest | Vector | Query/upsert vector dims ≠ namespace dims |
| IndexNotReady | INDEX_NOT_READY | Unavailable | Vector, Graph | Index/namespace exists but not yet ready/empty/building |
| NamespaceNotFound | NAMESPACE_NOT_FOUND | BadRequest | Vector, Graph | Namespace/graph/collection not found |
| QueryParseError | QUERY_PARSE_ERROR | BadRequest | Vector, Graph | Query text (dialect) cannot be parsed |
| SchemaValidationError | SCHEMA_VALIDATION_ERROR | BadRequest | Graph | Graph schema violation |
| VertexNotFound | VERTEX_NOT_FOUND | BadRequest | Graph | Vertex ID not found where required |
| EdgeNotFound | EDGE_NOT_FOUND | BadRequest | Graph | Edge ID not found where required |

**Note:** This table is **informative**, not normative. Subtypes may be added or removed by adapters without affecting wire compatibility.

## 2) Programmatic Error Envelope (Wire-Level)

Adapters MUST surface errors using the following canonical wire envelope. The normative shape is defined in `SCHEMA.md` under `/schemas/common/error_envelope.json` (schema-closed, `additionalProperties: false`).

```json
{
  "ok": false,
  "error": "ResourceExhausted",
  "message": "Rate limit exceeded for tenant",
  "code": "RESOURCE_EXHAUSTED",
  "retry_after_ms": 1200,
  "details": {
    "subtype": "ThroughputLimitExceeded",
    "subtype_code": "THROUGHPUT_LIMIT_EXCEEDED",
    "hints": {
      "resource_scope": "rate_limit",
      "throttle_scope": "tenant:acme:llm",
      "suggested_batch_reduction": 50
    },
    "max_batch_size": 1000,
    "provided_batch_size": 2400,
    "adapter_code": "RATE_LIMIT",
    "provider_code": "429"
  },
  "ms": 15.2
}
```

### 2.1 Normative Rules

- `ok` MUST be `false`
- `error` MUST be one of the **7 canonical class names** (PascalCase): `BadRequest`, `AuthError`, `ResourceExhausted`, `TransientNetwork`, `Unavailable`, `NotSupported`, `DeadlineExceeded`
- `code` MUST be the corresponding **canonical wire code** (ALL_CAPS_SNAKE): `BAD_REQUEST`, `AUTH_ERROR`, `RESOURCE_EXHAUSTED`, `TRANSIENT_NETWORK`, `UNAVAILABLE`, `NOT_SUPPORTED`, `DEADLINE_EXCEEDED`
- `message` MUST be present and human-readable (no secrets, prompts, vectors, IDs)
- `retry_after_ms` MUST be non-negative integer when present; else `null`
- `details` MAY contain:
  - `subtype`: suggested subtype name (PascalCase, optional)
  - `subtype_code`: suggested subtype code (ALL_CAPS_SNAKE, optional)
  - `hints`: object with machine-readable mitigations (excluding `retry_after_ms` which is top-level)
  - Additional structured, SIEM-safe context
- `ms` MUST be present (float, milliseconds elapsed since operation start)
- **No other top-level keys** are allowed (schema `additionalProperties: false`)

## 3) Transport Mappings (Non-Wire)

Transport-layer mappings are recommendations, not part of the wire envelope. The wire code remains unchanged regardless of transport mapping.

### 3.1 HTTP (Recommended)

| Canonical Class | HTTP Status |
|-----------------|-------------|
| BadRequest | 400 |
| AuthError | 401 / 403 |
| ResourceExhausted | 429 |
| NotSupported | 501 (or 400 for invalid params; wire code remains `NOT_SUPPORTED`) |
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

## 4) Retry Semantics (Normative)

### 4.1 Canonical Classes (Normative)

| Class | Retryable? | Client Guidance |
|-------|------------|----------------|
| BadRequest | No | Fix parameters / schema before retry |
| AuthError | No | Refresh credentials/permissions |
| ResourceExhausted | Yes | Backoff; honor retry_after_ms; reduce concurrency/batch |
| TransientNetwork | Yes | Exponential backoff + jitter; consider failover |
| Unavailable | Yes | Backoff; treat as overload; use breakers |
| NotSupported | No | Probe capabilities() and adjust feature/parameter |
| DeadlineExceeded | Conditional | Retry only if deadline increased or work reduced |

**Canonical class retryability is normative** and MUST be respected by all implementations.

### 4.2 Subtype Guidance (Informative)

| Suggested Subtype | Parent Class | Retryable? | Notes |
|-------------------|--------------|------------|-------|
| ModelNotFound | BadRequest | No | Fix model name or configuration |
| ModelOverloaded | Unavailable | Yes | Backoff; try alternative model/family |
| ContentFiltered | BadRequest | No | Sanitize or change content |
| SafetyPolicyViolation | BadRequest | No | Same as ContentFiltered but explicit to safety system |
| InputFormatError | BadRequest | No | Fix JSON/field types/role schema |
| ThroughputLimitExceeded | ResourceExhausted | Yes | Backoff; reduce concurrency/batch; honor retry_after_ms |
| TextTooLong | BadRequest | No | Enable truncation or split content |
| EmbeddingDimensionMismatch | BadRequest | No | Fix expected dims / configuration |
| ProviderQuotaExceeded | ResourceExhausted | Conditional | Retryable only when quota resets / after retry_after_ms |
| DimensionMismatch | BadRequest | No | Align vector dims with namespace |
| IndexNotReady | Unavailable | Yes | Retry after delay; honor retry_after_ms |
| NamespaceNotFound | BadRequest | No | Create namespace or fix name |
| QueryParseError | BadRequest | No | Fix query text / dialect |
| SchemaValidationError | BadRequest | No | Fix graph schema / labels / edge types |
| VertexNotFound | BadRequest | No | Fix vertex ids or query |
| EdgeNotFound | BadRequest | No | Fix edge ids or query |

> **Note:** Subtype-level retry guidance is **informative** only. Client retry behavior MUST be driven by the **canonical class** retryability rules, which are **normative** and binding.

## 5) Observability & Privacy (SIEM-Safe)

- For metrics (see METRICS.md), the **canonical wire code** (ALL_CAPS_SNAKE) MUST be used as the metrics code label.
- Never emit raw prompts, vectors, embeddings, tenant IDs, doc IDs, or arbitrary free-text fields.
- Use tenant hashing where tenant participates in labels or details.
- `details` MUST be low-cardinality and JSON-safe.

## 6) Protocol-Specific Guidance (Informative)

### 6.1 LLM

LLM adapters SHOULD map common failures into the taxonomy as follows:

**BadRequest scenarios:**
- Context length exceeded (suggested subtype: `TextTooLong`, `PromptTooLong`)
- Content filtered by provider
- Invalid input format (messages schema, tool definitions)
- Unknown model requested

**ResourceExhausted scenarios:**
- Rate limits (throughput, token quotas)
- Provider quota exceeded

**Unavailable scenarios:**
- Model overloaded
- Task rejected due to capacity
- Backend temporary unavailability

### 6.2 Embedding

Embedding adapters SHOULD map failures similarly:

**BadRequest scenarios:**
- Text too long (truncate=false)
- Content filtered
- Input format errors
- Dimension mismatches between expected and actual

**ResourceExhausted scenarios:**
- Throughput limits
- Provider quotas

### 6.3 Vector

Vector adapters SHOULD consider:

**BadRequest scenarios:**
- Dimension mismatches
- Unknown namespace
- Invalid filter/query syntax

**Unavailable scenarios:**
- Index not ready (building, empty)
- Shard unavailable
- Index corruption

### 6.4 Graph

Graph adapters SHOULD handle:

**BadRequest scenarios:**
- Query parse errors
- Schema validation failures
- Unknown vertices/edges
- Invalid filter syntax

**Unavailable scenarios:**
- Index not ready
- Shard unavailable

> **Streaming note (normative):** Streaming success frames MUST always be `{ok:true, code:"STREAMING", ms, chunk}`. If an error occurs, the stream MUST terminate by emitting exactly one **standard error envelope** `{ok:false, ...}` (not a streaming success frame), and no further frames may follow.

## 7) Error Hints (Machine-Readable Mitigations)

Adapters SHOULD attach structured hints in `details.hints`:
- `resource_scope` — one of: `"model" | "token_limit" | "rate_limit" | "memory" | "compute" | "time_budget" | "index" | "shard"`
- `throttle_scope` — bounded identifier for throttling domain (e.g., `"tenant:acme:llm"`, `"tenant:acme:graph"`).
- `suggested_batch_reduction` — percentage [0..100].

*Note: `retry_after_ms` is a top-level field, not included in hints.*

**Client rule:**

Clients SHOULD reduce their next batch size as:

```
new_size = ceil(old_size * (100 - suggested_batch_reduction) / 100)
```

## 8) Normalization Rules (Adapter Implementations)

Adapters MUST map provider-specific failures into the canonical taxonomy:

1. **Classify** into one of the 7 canonical classes.
2. **Set** `error` to canonical class name (PascalCase).
3. **Set** `code` to corresponding canonical wire code (ALL_CAPS_SNAKE).
4. **Set** a clean message (no raw upstream content).
5. **Optionally add** subtype information in `details.subtype` and `details.subtype_code`.
6. **Attach** hints in `details.hints` where meaningful.
7. **Populate** `details` with JSON-safe, low-cardinality fields.
8. **Include** `ms` with operation duration.
9. **Emit** metrics with canonical wire code as code label.

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

## 10) Partial Failure (Batches)

Batch APIs MUST surface per-item status, not just aggregate failure:

**Vector Example:**
```json
{
  "upserted_count": 42,
  "failed_count": 3,
  "failures": [
    { "id": "doc-17", "error": "BadRequest", "detail": "dimension mismatch" },
    { "id": "doc-19", "error": "BadRequest", "detail": "vector is empty" }
  ]
}
```

*Note: Per-item error strings are not required to be canonical error names (they are not full envelopes).*

**Embedding Example (schema-enforced):**
```json
{
  "embeddings": [...],
  "failed_texts": [
    {
      "index": 2,
      "text": "[redacted]",
      "error": "BadRequest",
      "code": "BAD_REQUEST",
      "message": "Text exceeds maximum length"
    }
  ]
}
```

## 11) Examples (Per Protocol)

**LLM — ResourceExhausted with subtype**

```json
{
  "ok": false,
  "error": "ResourceExhausted",
  "message": "Rate limit exceeded",
  "code": "RESOURCE_EXHAUSTED",
  "retry_after_ms": 1200,
  "details": {
    "subtype": "ThroughputLimitExceeded",
    "subtype_code": "THROUGHPUT_LIMIT_EXCEEDED",
    "hints": {
      "resource_scope": "rate_limit",
      "throttle_scope": "tenant:acme:llm"
    },
    "adapter_code": "RATE_LIMIT"
  },
  "ms": 45.7
}
```

**Vector — Unavailable with subtype**

```json
{
  "ok": false,
  "error": "Unavailable",
  "message": "Index not ready",
  "code": "UNAVAILABLE",
  "retry_after_ms": 2000,
  "details": {
    "subtype": "IndexNotReady",
    "subtype_code": "INDEX_NOT_READY",
    "namespace": "acme.docs",
    "adapter_code": "INDEX_NOT_READY"
  },
  "ms": 8.1
}
```

**BadRequest (generic)**

```json
{
  "ok": false,
  "error": "BadRequest",
  "message": "Invalid input format",
  "code": "BAD_REQUEST",
  "retry_after_ms": null,
  "details": {
    "adapter_code": "INPUT_FORMAT_ERROR"
  },
  "ms": 3.4
}
```

## 12) Versioning & Deprecation (Errors)

- **errors_version:** `1.0`
- The 7 canonical classes and their wire codes are **frozen** for v1.0.
- Adding new canonical classes or changing retry semantics of existing classes is a **MAJOR** change (wire-breaking).
- Subtypes in `details` are **informative only** and may change without affecting wire compatibility.
- **Deprecation:** When canonical classes must change, follow VERSIONING.md §6 (announce, warn, support window, remove in next MAJOR).

## 13) Compliance Checklist

- [ ] `ok` is `false`
- [ ] `error` is one of 7 canonical class names (PascalCase)
- [ ] `code` is corresponding canonical wire code (ALL_CAPS_SNAKE)
- [ ] `message` present; no content or secrets
- [ ] `retry_after_ms` is valid integer or `null`
- [ ] `ms` present with operation duration
- [ ] `details` JSON-safe and low-cardinality
- [ ] No other top-level keys present
- [ ] Metrics use canonical wire code as code label
- [ ] Tenant hashing used in telemetry contexts

## 14) FAQ

**Q: Can adapters add their own error types?**
**A:** Yes, but only as subtypes in `details`. The top-level `error` and `code` MUST use canonical values.

**Q: What about provider-specific error details?**
**A:** Include in `details.provider_code` or `details.adapter_code`, but keep SIEM-safe.

**Q: How should clients handle subtypes?**
**A:** Clients MAY inspect `details.subtype` for enhanced handling, but MUST handle the canonical class correctly regardless.

**Q: Does this affect protocol versioning?**
**A:** Changes to canonical classes or codes = MAJOR. Subtype changes = MINOR/PATCH (non-breaking).

## 15) Conformance Testing

- Conformance test suites MUST validate:
  - Wire envelope shape matches schema (`additionalProperties: false`)
  - Required fields present (`ok`, `error`, `code`, `message`, `ms`)
  - `error` value from 7 canonical class names
  - `code` value from 7 canonical wire codes
  - `error` and `code` alignment (e.g., `BadRequest` ↔ `BAD_REQUEST`)
  - No raw content in `message` or `details`
- Test tools MAY verify subtype usage in `details` but MUST NOT require it
- Reference implementations MUST include error handling in test coverage
- Subtype usage is optional and not part of conformance requirements
- Per-item error strings in batch failures are not validated as canonical errors

---

*End of ERRORS.md (errors_version 1.0)*