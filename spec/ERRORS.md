# ERRORS.md

**Corpus Protocol Suite & SDKs — Normalized Error Taxonomy**
**errors_version:** `1.0`

> This document defines the **normalized error model** used across **Graph**, **LLM**, **Vector**, and **Embedding** adapters. It aligns with the Specification’s Common Foundation (§6.3), Error Handling and Resilience (§12), and protocol-specific sections (§7.6 Graph health, §8.5 LLM, §9.5 Vector, §10.3 Embedding). It is **implementation-agnostic** and intended for use with any transport (HTTP, gRPC, queues) and any metrics/logging backend.

---

## 0) Goals & Non-Goals

**Goals**

* One **cross-protocol taxonomy** with consistent semantics and stable programmatic fields
* Clear **retryability** rules and **machine-readable hints** for adaptive clients
* Privacy-preserving, SIEM-safe error payloads and observability integration
* Predictable **HTTP** and **gRPC** mappings without exposing backend-specific quirks

**Non-Goals**

* Exhaustive vendor error catalogs
* Mandating a specific transport or framework

---

## 1) Normalized Error Classes (Canonical)

```
AdapterError (base)
├─ BadRequest               # 400 client errors (validation, schema, ranges)
├─ AuthError                # 401/403 authentication/authorization failures
├─ ResourceExhausted        # 429 quotas, rate limits (retry-after)
├─ TransientNetwork         # 502/504 upstream gateway/timeouts; network flaps
├─ Unavailable              # 503 backend temporarily unavailable/overloaded
├─ NotSupported             # 501/400 unsupported operation/parameter
└─ DeadlineExceeded         # 504 deadline/budget exhausted (absolute time)
```

**Protocol-specific subtypes** (used where applicable):

* **Vector (§9.5)**: `DimensionMismatch`, `IndexNotReady`
* **LLM (§8.5)**: `ModelOverloaded` (subtype of `Unavailable`), `ContentFiltered` (subtype of `BadRequest`)
* **Embedding (§10.3)**: `TextTooLong` (subtype of `BadRequest`)

> **Stability:** Class names and meanings are **frozen** for `errors_version=1.0`.

---

## 2) Programmatic Error Shape (Wire-Level Envelope)

All adapters SHOULD surface the following machine-readable fields:

```json
{
  "ok": false,
  "error": "ResourceExhausted",          // canonical class
  "message": "Rate limit exceeded for tenant",
  "code": "RATE_LIMIT",                  // short string; stable within adapter
  "http_status": 429,                    // transport mapping (if HTTP)
  "retry_after_ms": 1200,                // non-negative integer or null
  "resource_scope": "rate_limit",        // e.g., model|token_limit|rate_limit|memory|compute|time_budget
  "throttle_scope": "tenant:acme:llm",   // optional, bounded string
  "suggested_batch_reduction": 50,       // int [0..100], percentage (see §6.3)
  "details": {                           // JSON-serializable, privacy-safe
    "max_batch_size": 1000,
    "provided_batch_size": 2400
  }
}
```

**Rules (Normative):**

* `message` MUST be present and human-readable (no secrets or raw content).
* `code` SHOULD be a short stable string for analytics/dashboards.
* `retry_after_ms` MUST be a **non-negative integer** if present; otherwise `null`.
* `details` MUST be JSON-serializable and **SIEM-safe** (no prompts, vectors, or PII).
* For subtype errors, set `error` to the subtype name (e.g., `"DimensionMismatch"`).

---

## 3) Transport Mappings

### 3.1 HTTP Status (Recommended)

| Class             | HTTP                               |
| ----------------- | ---------------------------------- |
| BadRequest        | 400                                |
| AuthError         | 401/403                            |
| ResourceExhausted | 429                                |
| NotSupported      | 501 (or 400 for invalid parameter) |
| TransientNetwork  | 502/504                            |
| Unavailable       | 503                                |
| DeadlineExceeded  | 504                                |

### 3.2 gRPC Codes (Informative)

| Class             | gRPC Code                           |
| ----------------- | ----------------------------------- |
| BadRequest        | INVALID_ARGUMENT                    |
| AuthError         | UNAUTHENTICATED / PERMISSION_DENIED |
| ResourceExhausted | RESOURCE_EXHAUSTED                  |
| NotSupported      | UNIMPLEMENTED                       |
| TransientNetwork  | UNAVAILABLE / DEADLINE_EXCEEDED     |
| Unavailable       | UNAVAILABLE                         |
| DeadlineExceeded  | DEADLINE_EXCEEDED                   |

---

## 4) Retry Semantics (Normative)

| Class                 | Retryable         | Client Guidance                                                 |
| --------------------- | ----------------- | --------------------------------------------------------------- |
| **BadRequest**        | **No**            | Fix parameters; do not retry as-is                              |
| **AuthError**         | **No**            | Refresh credentials/permissions                                 |
| **ResourceExhausted** | **Yes**           | Backoff; **honor `retry_after_ms`**; reduce concurrency/batch   |
| **TransientNetwork**  | **Yes**           | Exponential backoff + jitter; consider failover                 |
| **Unavailable**       | **Yes**           | Backoff; trip/bias breaker; consider failover                   |
| **NotSupported**      | **No**            | Probe `capabilities()`; switch feature/parameter                |
| **DeadlineExceeded**  | **Conditionally** | Only retry if deadline increased or work reduced (tokens/batch) |
| **DimensionMismatch** | **No**            | Align dimensions to namespace/index                             |
| **IndexNotReady**     | **Yes**           | Retry after `retry_after_ms` (or reasonable backoff)            |
| **ContentFiltered**   | **No**            | Adjust/sanitize content                                         |
| **TextTooLong**       | **No**            | Enable truncation or split input                                |

---

## 5) Observability & Privacy (SIEM-Safe)

* The error **class** (e.g., `Unavailable`) SHALL be used as the **`code` label** in metrics (see `METRICS.md`).
* **Never** log raw prompts, vectors, embeddings, or raw tenant IDs.
* Use **tenant hashing** for any tenant-related labels/fields.
* Keep `details` low-cardinality and scrubbed of secrets.

---

## 6) Protocol-Specific Supplements

### 6.1 Vector (§9.5)

* **DimensionMismatch**

  * **When:** upsert/query vector length differs from namespace dimensions
  * **Retry:** **No**
  * **details:** `{"expected": <int>, "provided": <int>, "namespace": "<ns>"}`

* **IndexNotReady**

  * **When:** namespace exists but index/data not ready (empty or building)
  * **Retry:** **Yes**; respect `retry_after_ms` if present

### 6.2 LLM (§8.5)

* **ModelOverloaded** (subtype of `Unavailable`)

  * **When:** transient capacity pressure
  * **Retry:** **Yes**; consider alternative model/family

* **ContentFiltered** (subtype of `BadRequest`)

  * **When:** provider content policy triggers
  * **Retry:** **No**; sanitize or change prompt

### 6.3 Embedding (§10.3)

* **TextTooLong** (subtype of `BadRequest`)

  * **When:** input exceeds model limit and `truncate=false`
  * **Retry:** **No** (unless enabling truncation or splitting)

---

## 7) Error Hints (Machine-Readable Mitigations)

Adapters SHOULD include structured hints to enable **adaptive clients**:

* `retry_after_ms` — delay before retrying (non-negative int)
* `resource_scope` — `"model" | "token_limit" | "rate_limit" | "memory" | "compute" | "time_budget"`
* `throttle_scope` — bounded string to scope throttling (e.g., `"tenant:acme:llm"`)
* `suggested_batch_reduction` — integer percentage `[0..100]` indicating how much to reduce batch size when re-trying (e.g., `50` means “half your batch”)
* `details` — low-cardinality, JSON-safe additional context (e.g., `"max_batch_size"`, `"max_top_k"`)

**Example (Vector batch too large):**

```json
{
  "ok": false,
  "error": "BadRequest",
  "message": "batch size 2400 exceeds maximum of 1000",
  "code": "BATCH_TOO_LARGE",
  "retry_after_ms": null,
  "resource_scope": "rate_limit",
  "suggested_batch_reduction": 58,
  "details": {"max_batch_size": 1000, "provided_batch_size": 2400}
}
```

---

## 8) Normalization Rules (Adapter Implementation)

Adapters MUST translate provider-specific errors into the canonical taxonomy:

1. **Classify** the error into one of the canonical classes (or protocol-specific subtypes).
2. **Set** a succinct `message` (no secrets, no raw content).
3. **Map** transport code (HTTP/gRPC) per §3.
4. **Attach** hints (retry_after_ms, resource_scope, suggested_batch_reduction, throttle_scope) when meaningful.
5. **Populate** `details` with low-cardinality, JSON-safe context.
6. **Emit** corresponding metrics (`observe_operation` with `code`, and for streams a single `count_stream_final_outcome`).

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
        return TransientNetwork(message="Upstream timeout", code="GATEWAY_TIMEOUT")
    # ...map others...
    return Unavailable(message="Service unavailable", code="UNKNOWN_UPSTREAM")
```

---

## 9) Client Backoff & Breaker Guidance

**Exponential Backoff with Full Jitter (Recommended):**

* Base delay: 100–500 ms
* Factor: ×2
* Cap: 10–30 s
* Respect server `retry_after_ms` if present (override schedule)

**Deadline-aware retry (Conditionally Retryable):**

* For `DeadlineExceeded`, **do not** retry unchanged.
* Retry **only** after increasing deadline **or** reducing work (e.g., `max_tokens`, batch size).

**Breaker Integration:**

* Trip breaker on `Unavailable`/`TransientNetwork` flurries
* Fail fast with `Unavailable("circuit open")`
* Include `retry_after_ms` where possible to prevent thundering herds

---

## 10) Partial Failure (Batches)

Batch APIs MUST report **per-item** status, not just aggregate failure:

```json
{
  "upserted_count": 42,
  "failed_count": 3,
  "failures": [
    {"id": "doc-17", "error": "DimensionMismatch", "detail": "expected 1536, got 1537"},
    {"id": "doc-19", "error": "BadRequest", "detail": "vector is empty"},
    {"id": "doc-23", "error": "BadRequest", "detail": "metadata not JSON-serializable"}
  ]
}
```

Include `suggested_batch_reduction` at the **top-level error** when the entire batch request is too large or rate-limited.

---

## 11) Examples (Per Protocol)

**LLM — content filtered**

```json
{
  "ok": false,
  "error": "ContentFiltered",
  "message": "Input violates content policy",
  "code": "CONTENT_FILTERED",
  "http_status": 400,
  "retry_after_ms": null,
  "resource_scope": "model",
  "details": {"policy_section": "safety.v2"}
}
```

**Vector — index not ready**

```json
{
  "ok": false,
  "error": "IndexNotReady",
  "message": "index not ready (namespace initialized but empty)",
  "code": "INDEX_NOT_READY",
  "http_status": 503,
  "retry_after_ms": 2000,
  "resource_scope": "compute",
  "details": {"namespace": "acme.docs"}
}
```

**Embedding — text too long**

```json
{
  "ok": false,
  "error": "TextTooLong",
  "message": "Input exceeds maximum length and truncate=false",
  "code": "TEXT_TOO_LONG",
  "http_status": 400,
  "retry_after_ms": null,
  "resource_scope": "token_limit",
  "details": {"max_text_length": 16000, "provided_length": 24210}
}
```

---

## 12) Versioning & Deprecation (Errors)

* **errors_version:** `1.0` (this document)
* **Stability:** Class names, meaning, and retry semantics are frozen in 1.0.
* **Adding new subtypes:** Allowed if they **refine** an existing class **without** contradicting retry semantics (e.g., `ModelOverloaded` under `Unavailable`).
* **Deprecation policy:** Mark an old subtype as deprecated for ≥1 minor release before removal. Do **not** reuse names with different semantics.

---

## 13) Compliance Checklist

* [ ] Error belongs to the canonical taxonomy (or allowed subtype)
* [ ] `message` present; no secrets, no raw content
* [ ] `code` short, stable string (optional but recommended)
* [ ] `retry_after_ms` non-negative int or null
* [ ] `resource_scope` present where meaningful
* [ ] `suggested_batch_reduction` provided for batch overruns
* [ ] `details` JSON-safe and low-cardinality
* [ ] Transport mapping applied correctly (HTTP/gRPC)
* [ ] Metrics emitted with `code` label; streaming emits exactly one final outcome metric
* [ ] Tenant hashing applied in all telemetry contexts

---

## 14) FAQ

**Q: Can we emit provider raw error messages in `details`?**
**A:** Not if they contain prompts, vectors, IDs, or sensitive text. Scrub or map to bounded fields.

**Q: How do we choose between `TransientNetwork` and `Unavailable`?**
**A:** Use `TransientNetwork` for gateway/timeouts and clear network pathologies; `Unavailable` for backend overload/capacity issues.

**Q: Should `DeadlineExceeded` include `retry_after_ms`?**
**A:** Usually **no**; the fix is increasing the deadline or reducing work, not waiting.

---

*End of ERRORS.md (errors_version 1.0)*

