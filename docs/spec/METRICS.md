# METRICS

**Table of Contents**
- [0) Scope & Goals](#0-scope--goals)
- [1) Privacy & SIEM-Safe Requirements (Normative)](#1-privacy--siem-safe-requirements-normative)
- [2) Metric Taxonomy & Required Dimensions](#2-metric-taxonomy--required-dimensions)
- [3) Deadline Buckets (Normative, Frozen Set)](#3-deadline-buckets-normative-frozen-set)
- [4) Histogram Guidance (Agnostic)](#4-histogram-guidance-agnostic)
- [5) Optional, Bounded Dimensions](#5-optional-bounded-dimensions)
- [6) Per-Protocol Notes](#6-per-protocol-notes)
- [7) Error Taxonomy Integration](#7-error-taxonomy-integration)
- [8) Versioning & Deprecation Policy (Metrics)](#8-versioning--deprecation-policy-metrics)
- [9) Minimal Emission Rules](#9-minimal-emission-rules)
- [10) Example Semantic Records (Agnostic Pseudo)](#10-example-semantic-records-agnostic-pseudo)
- [11) Compliance Checklist](#11-compliance-checklist)
- [12) Security, Privacy & Audit Notes](#12-security-privacy--audit-notes)
- [13) FAQ (Agnostic)](#13-faq-agnostic)
- [14) Conformance Testing](#14-conformance-testing)
- [15) Aggregation & Sampling Guidance](#15-aggregation--sampling-guidance)

---

**Cross-Protocol Observability (Agnostic)**  
**Metrics Version:** `1.1`

> This document defines an *implementation-agnostic* metrics contract for **Graph**, **LLM**, **Vector**, and **Embedding** adapters. It aligns with the specification's Observability sections (see §13.1–§13.3) and the Common Foundation. Providers, stacks, and exporters are **not** mandated; the shapes below are semantic contracts that any telemetry pipeline can map to.

## 0) Scope & Goals

**Goals**

* Uniform, low-cardinality metrics across all protocol adapters
* Strong privacy guarantees (SIEM-safe) with deterministic **tenant hashing**
* Deadline awareness via **deadline buckets** and **final stream outcome** events
* Stable naming with clear **versioning** and deprecation rules
* Full coverage of all protocol operations defined in `PROTOCOLS.md`

**Non-Goals**

* No provider/vendor coupling (no Prometheus, Datadog, etc. specifics)
* No raw content (prompts, vectors, tenant IDs) in telemetry

## 1) Privacy & SIEM-Safe Requirements (Normative)

**Hard rules (MUST):**

1. **No content in telemetry.** Bodies, prompts, vectors, embeddings, and *raw* tenant identifiers **MUST NOT** appear in metrics or logs.
2. **Tenant hashing.** Report only a deterministic, irreversible `tenant_hash`. Use **HMAC-SHA256** with a deployment-specific secret key (or salted SHA-256 with protected salt), then truncate to first 12 hex characters. The raw tenant value MUST NOT be emitted.
3. **Low cardinality.** All labels/attributes MUST be bounded and documented. Free-text or high-entropy fields are prohibited.

**Recommended (SHOULD):**

* Redact or hash any incidental IDs that could explode cardinality (docs, sessions, users).
* Gate model names behind a capability/extension flag (see §5).

## 2) Metric Taxonomy & Required Dimensions

All adapters (Graph/LLM/Vector/Embedding) MUST emit at least:

### 2.1 Operation Latency (Timer)

* **Instrument:** latency observation (timer/histogram)
* **Name (semantic):** `observe_operation`
* **Dimensions (labels/attrs):**

  * `component` ∈ `{graph|llm|vector|embedding}`
  * `op` — adapter operation (see §2.4 for canonical names)
  * `code` — normalized status (e.g., `OK`, `BAD_REQUEST`, `UNAVAILABLE`, …) from the shared error taxonomy (ALL_CAPS_SNAKE case)
  * `deadline_bucket` — **one of** `"<1s" | "<5s" | "<15s" | "<60s" | ">=60s"` *(see §3; normative set — MUST NOT add values without a version bump)*. **Required only if `ctx.deadline_ms` is present; otherwise omit.**
  * `tenant_hash` — deterministic hash (or omitted if no tenant)
  * Optional **bounded** extras (see §5)

### 2.2 Operation Count (Counter)

* **Instrument:** monotonic counter
* **Name (semantic):** `count_operation`
* **Dimensions:** same as `observe_operation` (above), with extras allowed (bounded)

### 2.3 Final Stream Outcome (Streaming Only)

* **Instrument:** monotonic counter
* **Name (semantic):** `count_stream_final_outcome`
* **Semantics:** Emitted **exactly once per stream** with the terminal outcome code. For successful streams, use `code="OK"`. For errored streams, use the error's wire code (e.g., `DEADLINE_EXCEEDED`, `UNAVAILABLE`).
* **Dimensions:** `component`, `op`, `code`, `tenant_hash` (optional), and **no** content-derived labels.

> Rationale: Streaming can emit many interim events; only a single *terminal* metric provides reliable SLO/SLA-like insight without inflating cardinality.

### 2.4 Canonical Operation Names (Normative)

The `op` dimension MUST be chosen from the following finite protocol-specific sets:

**LLM**
* `capabilities`
* `complete`
* `stream`
* `count_tokens`
* `health`

**Embedding**
* `capabilities`
* `embed`
* `embed_batch`
* `stream_embed`
* `get_stats`
* `count_tokens`
* `health`

**Vector**
* `capabilities`
* `query`
* `batch_query`
* `upsert`
* `delete`
* `create_namespace`
* `delete_namespace`
* `health`

**Graph**
* `capabilities`
* `query`
* `stream_query`
* `upsert_nodes`
* `upsert_edges`
* `delete_nodes`
* `delete_edges`
* `bulk_vertices`
* `batch`
* `get_schema`
* `transaction`
* `traversal`
* `health`

Adapters MUST NOT invent new `op` values under `metrics_version=1.1`. If additional operation types are required in the future, they MUST be introduced alongside a `metrics_version` bump and documented.

## 3) **Deadline Buckets** (Normative, Frozen Set)

Adapters compute remaining time (`ctx.deadline_ms - now`) and place operations into one of:

* `"<1s"`, `"<5s"`, `"<15s"`, `"<60s"`, `">=60s"`

**Rules:**

* These **five** categories are **normative** for `metrics_version=1.1`.
* Clients and exporters **MUST NOT** introduce additional buckets without a **metrics_version** increment.
* If there is no deadline (`ctx.deadline_ms` absent or null), **omit** the `deadline_bucket` dimension entirely.
* **Units:** All time measurements MUST be reported in **milliseconds**.

## 4) Histogram Guidance (Agnostic)

For latency histograms, use a **consistent millisecond** bucket strategy to enable comparable dashboards:

```text
[5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
```

* Treat this as RECOMMENDED. If you adjust buckets, keep them stable within a metrics_version and document them.
* **Units:** All histogram bucket boundaries and latency measurements MUST be in **milliseconds**.

## 5) Optional, Bounded Dimensions

To preserve low cardinality, only the following optional dimensions are recommended:

* `namespace` (Vector/Graph): when the deployment limits the namespace set (e.g., O(10^2) **MUST be bounded by deployment policy**)
* `cached` ∈ `{0,1}` for cache hits on eligible read paths
* `vectors_processed`, `rows`, `batch_size` as numeric sample fields or counters (not labels) where your telemetry system supports numeric fields
* `model` (LLM/Embedding) ONLY when the adapter capability `extensions.tag_model_in_metrics = true` is enabled; otherwise, omit

Do not introduce free-form labels (file names, doc IDs, user IDs, GUIDs, arbitrary strings).

## 6) Per-Protocol Notes

### 6.1 Vector

* **Operations (see §2.4):** `capabilities`, `query`, `batch_query`, `upsert`, `delete`, `create_namespace`, `delete_namespace`, `health`
* **Useful additional numeric counters/fields:**
  * `vectors_upserted`
  * `vectors_deleted`
* **Privacy:** never log vectors, metadata content, or raw IDs in any metrics or logs.

### 6.2 LLM

* **Operations:** `capabilities`, `complete`, `stream`, `count_tokens`, `health`
* **Streaming MUST** emit exactly one `count_stream_final_outcome` per stream with the terminal code.
* **Streaming success outcome:** For streams ending with `is_final: true`, use `code="OK"`. For errored streams, use the error's wire code.
* **Optional numeric fields:**
  * `tokens_processed` (prompt + completion)
  * `prompt_tokens`
  * `completion_tokens`
* These MUST be numeric fields/counters, not labels.

### 6.3 Graph

* **Operations:** `capabilities`, `query`, `stream_query`, `upsert_nodes`, `upsert_edges`, `delete_nodes`, `delete_edges`, `bulk_vertices`, `batch`, `get_schema`, `transaction`, `traversal`, `health`
* `bulk_vertices` SHOULD include numeric fields like `nodes_returned` where supported.
* **Streaming queries** (`stream_query`) MUST also fire a single terminal outcome event via `count_stream_final_outcome`.

### 6.4 Embedding

* **Operations:** `capabilities`, `embed`, `embed_batch`, `stream_embed`, `get_stats`, `count_tokens`, `health`
* **Optional numeric fields:**
  * `texts_embedded`
  * `embeddings_generated`
  * `tokens_processed`
* **Batch operations** SHOULD reflect the number of successful vs failed items via fields and/or protocol-level results; metrics should not encode per-item IDs.

## 7) Error Taxonomy Integration

The `code` label MUST be drawn from the shared error taxonomy defined in the main Corpus specification. Use the **wire code format** (ALL_CAPS_SNAKE case).

**Example code values (non-exhaustive):**
* `"OK"`
* `"BAD_REQUEST"`
* `"AUTH_ERROR"`
* `"RESOURCE_EXHAUSTED"`
* `"TRANSIENT_NETWORK"`
* `"UNAVAILABLE"`
* `"NOT_SUPPORTED"`
* `"DIMENSION_MISMATCH"`
* `"INDEX_NOT_READY"`
* `"DEADLINE_EXCEEDED"`

**Rules:**
* Keep code values finite and aligned with the shared taxonomy.
* Use wire codes exactly as they appear in error envelopes.
* If the main spec introduces a new error class, you MAY:
  * add it to metrics under the same metrics_version only if it does not increase cardinality in an unbounded way, and
  * no existing code values are removed or repurposed.
* If adding new codes would break comparability or cardinality assumptions, bump metrics_version and document the change.

## 8) Versioning & Deprecation Policy (Metrics)

* **metrics_version:** 1.1 (this document)

**Stability guarantees (within a metrics_version):**
* Names and required dimensions (e.g., `observe_operation`, `count_operation`, `deadline_bucket`) are stable.
* Deadline buckets are frozen (see §3).
* Canonical op values are frozen (see §2.4).

**Deprecation policy:**

When changing names, semantics, or buckets:
1. Mark old items deprecated and emit both old and new metrics for at least one minor software release.
2. Remove deprecated items in the next minor release.
3. Bump metrics_version if:
   * bucket sets change,
   * canonical op sets change, or
   * changes break historical comparability.

## 9) Minimal Emission Rules

* Emit one latency observation per operation (`observe_operation`) recording duration and outcome.
* Increment one counter per operation (`count_operation`) with the same labels.
* For streaming operations (e.g., `op="stream"` for LLM, `op="stream_query"` for Graph), increment exactly one `count_stream_final_outcome` after the terminal condition:
  * For successful streams: when `is_final: true` is received, use `code="OK"`
  * For errored streams: when error envelope is received, use the error's wire code
* Never emit intermediate chunk content or high-cardinality identifiers as labels.
* **Sampling:** For high-volume deployments, consider sampling latency observations (e.g., 10% sample rate) but always emit counters fully.

## 10) Example Semantic Records (Agnostic Pseudo)

Note: These are semantic event shapes. Map them to your telemetry framework as appropriate.

**Operation success (LLM complete):**

```json
observe_operation:
  component: "llm"
  op: "complete"
  code: "OK"
  deadline_bucket: "<15s"
  tenant_hash: "7d9f53d2f1ab"
  ms: 412.7

count_operation:
  component: "llm"
  op: "complete"
  code: "OK"
  deadline_bucket: "<15s"
  tenant_hash: "7d9f53d2f1ab"
```

**Streaming terminal outcome (LLM stream success):**

```json
count_stream_final_outcome:
  component: "llm"
  op: "stream"
  code: "OK"
  tenant_hash: "7d9f53d2f1ab"
```

**Streaming terminal outcome (LLM stream error):**

```json
count_stream_final_outcome:
  component: "llm"
  op: "stream"
  code: "DEADLINE_EXCEEDED"
  tenant_hash: "7d9f53d2f1ab"
```

**Vector query (cache hit example, optional field):**

```json
observe_operation:
  component: "vector"
  op: "query"
  code: "OK"
  deadline_bucket: "<5s"
  tenant_hash: "a1c293bc0e11"
  ms: 28.5
  extras:
    cached: 1
    matches: 10

count_operation:
  component: "vector"
  op: "query"
  code: "OK"
  deadline_bucket: "<5s"
  tenant_hash: "a1c293bc0e11"
```

## 11) Compliance Checklist

- [ ] No prompts, vectors, embeddings, or raw tenant IDs in telemetry
- [ ] `tenant_hash` used where tenant is relevant (HMAC-SHA256 with secret or salted hash)
- [ ] `deadline_bucket` ∈ fixed set (`<1s`|`<5s`|`<15s`|`<60s`|`>=60s`) and present only when `ctx.deadline_ms` exists
- [ ] One `observe_operation` + one `count_operation` per operation
- [ ] One `count_stream_final_outcome` per streaming operation with correct terminal code
- [ ] Optional labels are bounded; model tagging gated by capability
- [ ] Histogram buckets consistent with §4 (or documented alternative)
- [ ] `op` values drawn from canonical sets in §2.4
- [ ] `code` values use wire format (ALL_CAPS_SNAKE) from shared error taxonomy
- [ ] Deprecations follow §8
- [ ] All time measurements in milliseconds

## 12) Security, Privacy & Audit Notes

* Treat this metrics contract as part of your privacy program; it enforces data minimization by design.
* Periodically audit telemetry for accidental content leakage (e.g., unexpected attribute growth, free-text values).
* Configure alerting on cardinality explosions (symptom of mislabeling or misuse of labels).
* Ensure tenant hashing uses HMAC-SHA256 with a secure secret key, or salted SHA-256 with protected salt.
* Rotate hashing secrets according to your organization's cryptographic key rotation policy.

## 13) FAQ (Agnostic)

**Q: Can we add a new deadline bucket for sub-millisecond calls?**
**A:** Not under `metrics_version=1.1`. Propose a new version if truly needed.

**Q: Can we tag model names on every call?**
**A:** Only if `extensions.tag_model_in_metrics=true`. Otherwise omit to protect cardinality and privacy.

**Q: Can we emit per-chunk streaming metrics?**
**A:** Allowed for internal debug, but not part of the stable contract. The only required streaming metric is the single final outcome counter (`count_stream_final_outcome`).

**Q: Can we introduce new op values specific to our adapter?**
**A:** Not under `metrics_version=1.1`. Use the canonical sets in §2.4. If you need additional operation types, propose a spec and metrics_version update.

**Q: What if metrics collection is disabled?**
**A:** Adapters MUST still function correctly; metrics emission is optional for functional correctness but required for compliance when enabled.

## 14) Conformance Testing

* Conformance test suites MUST validate:
  - No raw content in emitted metrics
  - Correct `op` values from canonical sets
  - Proper `code` values (wire format)
  - Correct `deadline_bucket` usage (present only with deadlines)
  - Single `count_stream_final_outcome` per stream
* Test tools SHOULD verify cardinality bounds on labels
* Reference implementations MUST include metrics emission in their test coverage

## 15) Aggregation & Sampling Guidance

**Recommended aggregation periods:**
- 1 minute: High-resolution debugging
- 5 minutes: Standard operational dashboards  
- 1 hour: Long-term trend analysis

**Sampling recommendations:**
- **Counters:** Never sample - always emit
- **Latency observations:** Consider sampling (e.g., 10-20%) for high-volume deployments
- **Stream outcomes:** Never sample - always emit

**Cardinality monitoring:**
- Alert when unique label combinations exceed deployment-specific thresholds
- Monitor growth of `namespace`, `model`, and other optional dimensions

---

  *End of METRICS.md (metrics_version 1.0)*

**Change History:**
- **v1.0**: Initial version