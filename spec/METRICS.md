# METRICS.md

**Corpus Protocol Suite & SDKs — Cross-Protocol Observability (Agnostic)**  
**metrics_version:** `1.0`

> This document defines an *implementation-agnostic* metrics contract for **Graph**, **LLM**, **Vector**, and **Embedding** adapters. It aligns with the specification’s Observability sections (see §13.1–§13.3) and the Common Foundation. Providers, stacks, and exporters are **not** mandated; the shapes below are semantic contracts that any telemetry pipeline can map to.

---

## 0) Scope & Goals

**Goals**

* Uniform, low-cardinality metrics across all protocol adapters
* Strong privacy guarantees (SIEM-safe) with deterministic **tenant hashing**
* Deadline awareness via **deadline buckets** and **final stream outcome** events
* Stable naming with clear **versioning** and deprecation rules

**Non-Goals**

* No provider/vendor coupling (no Prometheus, Datadog, etc. specifics)
* No raw content (prompts, vectors, tenant IDs) in telemetry

---

## 1) Privacy & SIEM-Safe Requirements (Normative)

**Hard rules (MUST):**

1. **No content in telemetry.** Bodies, prompts, vectors, embeddings, and *raw* tenant identifiers **MUST NOT** appear in metrics or logs.
2. **Tenant hashing.** Report only a deterministic, irreversible `tenant_hash` (e.g., first 12 chars of SHA-256). The raw tenant value MUST NOT be emitted.
3. **Low cardinality.** All labels/attributes MUST be bounded and documented. Free-text or high-entropy fields are prohibited.

**Recommended (SHOULD):**

* Redact or hash any incidental IDs that could explode cardinality (docs, sessions, users).
* Gate model names behind a capability/extension flag (see §5).

---

## 2) Metric Taxonomy & Required Dimensions

All adapters (Graph/LLM/Vector/Embedding) MUST emit at least:

### 2.1 Operation Latency (Timer)

* **Instrument:** latency observation (timer/histogram)
* **Name (semantic):** `observe_operation`
* **Dimensions (labels/attrs):**

  * `component` ∈ `{graph|llm|vector|embedding}`
  * `op` — adapter operation (see §2.4 for canonical names)
  * `code` — normalized status (e.g., `OK`, `BadRequest`, `Unavailable`, …) from the shared error taxonomy
  * `deadline_bucket` — **one of** `"<1s" | "<5s" | "<15s" | "<60s" | ">=60s"` *(see §3; normative set — MUST NOT add values without a version bump)*
  * `tenant_hash` — deterministic hash (or omitted if no tenant)
  * Optional **bounded** extras (see §5)

### 2.2 Operation Count (Counter)

* **Instrument:** monotonic counter
* **Name (semantic):** `count_operation`
* **Dimensions:** same as `observe_operation` (above), with extras allowed (bounded)

### 2.3 Final Stream Outcome (Streaming Only)

* **Instrument:** monotonic counter
* **Name (semantic):** `count_stream_final_outcome`
* **Semantics:** Emitted **exactly once per stream** with the terminal `code` (e.g., `OK`, `DeadlineExceeded`, `Unavailable`).
* **Dimensions:** `component`, `op`, `code`, `tenant_hash` (optional), and **no** content-derived labels.

> Rationale: Streaming can emit many interim events; only a single *terminal* metric provides reliable SLO/SLA-like insight without inflating cardinality.

### 2.4 Canonical Operation Names (Normative)

The `op` dimension MUST be chosen from the following finite protocol-specific sets:

**LLM**

* `complete`
* `stream`
* `count_tokens`
* `health`

**Embedding**

* `embed`
* `embed_batch`
* `count_tokens`
* `health`

**Vector**

* `query`
* `upsert`
* `delete`
* `create_namespace`
* `delete_namespace`
* `health`

**Graph**

* `query`
* `stream_query`
* `upsert_nodes`
* `upsert_edges`
* `delete_nodes`
* `delete_edges`
* `bulk_vertices`
* `batch`
* `health`

Adapters MUST NOT invent new `op` values under `metrics_version=1.0`. If additional operation types are required in the future, they MUST be introduced alongside a `metrics_version` bump and documented.

---

## 3) **Deadline Buckets** (Normative, Frozen Set)

Adapters compute remaining time (`ctx.deadline_ms - now`) and place operations into one of:

* `"<1s"`, `"<5s"`, `"<15s"`, `"<60s"`, `">=60s"`

**Rules:**

* These **five** categories are **normative** for `metrics_version=1.0`.
* Clients and exporters **MUST NOT** introduce additional buckets without a **metrics_version** increment.
* If there is no deadline, either:
  * omit the `deadline_bucket` label, **or**
  * set a deployment-standard sentinel (e.g., `"none"`). Choose one approach globally and document it in your ops runbook.

---

## 4) Histogram Guidance (Agnostic)

For latency histograms, use a **consistent millisecond** bucket strategy to enable comparable dashboards:

```text
[5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]

	•	Treat this as RECOMMENDED. If you adjust buckets, keep them stable within a metrics_version and document them.

⸻

5) Optional, Bounded Dimensions

To preserve low cardinality, only the following optional dimensions are recommended:
	•	namespace (Vector/Graph): when the deployment limits the namespace set (e.g., O(10^2))
	•	cached ∈ {0,1} for cache hits on eligible read paths
	•	vectors_processed, rows, batch_size as numeric sample fields or counters (not labels) where your telemetry system supports numeric fields
	•	model (LLM/Embedding) ONLY when the adapter capability extensions.tag_model_in_metrics = true is enabled; otherwise, omit

Do not introduce free-form labels (file names, doc IDs, user IDs, GUIDs, arbitrary strings).

⸻

6) Per-Protocol Notes

6.1 Vector
	•	Operations (see §2.4): query, upsert, delete, create_namespace, delete_namespace, health
	•	Useful additional numeric counters/fields:
	•	vectors_upserted
	•	vectors_deleted
	•	Privacy: never log vectors, metadata content, or raw IDs in any metrics or logs.

6.2 LLM
	•	Operations: complete, stream, count_tokens, health
	•	Streaming MUST emit exactly one count_stream_final_outcome per stream with the terminal code.
	•	Optional numeric fields:
	•	tokens_processed (prompt + completion)
	•	prompt_tokens
	•	completion_tokens
These MUST be numeric fields/counters, not labels.

6.3 Graph
	•	Operations: query, stream_query, upsert_nodes, upsert_edges, delete_nodes, delete_edges, bulk_vertices, batch, health
	•	bulk_vertices SHOULD include numeric fields like nodes_returned where supported.
	•	Streaming queries (stream_query) MUST also fire a single terminal outcome event via count_stream_final_outcome.

6.4 Embedding
	•	Operations: embed, embed_batch, count_tokens, health
	•	Optional numeric fields:
	•	texts_embedded
	•	embeddings_generated
	•	tokens_processed
	•	Batch operations SHOULD reflect the number of successful vs failed items via fields and/or protocol-level results; metrics should not encode per-item IDs.

⸻

7) Error Taxonomy Integration

The code label MUST be drawn from the shared error taxonomy defined in the main Corpus specification. This section provides examples, not an alternative source of truth.

Example code values (non-exhaustive):
	•	"OK"
	•	"BadRequest"
	•	"AuthError"
	•	"ResourceExhausted"
	•	"TransientNetwork"
	•	"Unavailable"
	•	"NotSupported"
	•	"DimensionMismatch"
	•	"IndexNotReady"
	•	"DeadlineExceeded"

Rules:
	•	Keep code values finite and aligned with the shared taxonomy.
	•	If the main spec introduces a new error class, you MAY:
	•	add it to metrics under the same metrics_version only if it does not increase cardinality in an unbounded way, and
	•	no existing code values are removed or repurposed.
	•	If adding new codes would break comparability or cardinality assumptions, bump metrics_version and document the change.

⸻

8) Versioning & Deprecation Policy (Metrics)
	•	metrics_version: 1.0 (this document)

Stability guarantees (within a metrics_version):
	•	Names and required dimensions (e.g., observe_operation, count_operation, deadline_bucket) are stable.
	•	Deadline buckets are frozen (see §3).
	•	Canonical op values are frozen (see §2.4).

Deprecation policy:

When changing names, semantics, or buckets:
	1.	Mark old items deprecated and emit both old and new metrics for at least one minor software release.
	2.	Remove deprecated items in the next minor release.
	3.	Bump metrics_version if:
	•	bucket sets change,
	•	canonical op sets change, or
	•	changes break historical comparability.

⸻

9) Minimal Emission Rules
	•	Emit one latency observation per operation (observe_operation) recording duration and outcome.
	•	Increment one counter per operation (count_operation) with the same labels.
	•	For streaming operations (e.g., op="stream" for LLM, op="stream_query" for Graph), increment exactly one count_stream_final_outcome after the terminal condition (success or error).
	•	Never emit intermediate chunk content or high-cardinality identifiers as labels.

⸻

10) Example Semantic Records (Agnostic Pseudo)

Note: These are semantic event shapes. Map them to your telemetry framework as appropriate.

Operation success (LLM complete):

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

Streaming terminal outcome (LLM stream):

count_stream_final_outcome:
  component: "llm"
  op: "stream"
  code: "DeadlineExceeded"
  tenant_hash: "7d9f53d2f1ab"

Vector query (cache hit example, optional field):

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


⸻

11) Compliance Checklist
	•	No prompts, vectors, embeddings, or raw tenant IDs in telemetry
	•	tenant_hash used where tenant is relevant
	•	deadline_bucket ∈ fixed set (<1s|<5s|<15s|<60s|>=60s) or omitted/"none" as per global policy
	•	One observe_operation + one count_operation per operation
	•	One count_stream_final_outcome per streaming operation
	•	Optional labels are bounded; model tagging gated by capability
	•	Histogram buckets consistent with §4 (or documented alternative)
	•	op values drawn from canonical sets in §2.4
	•	code values drawn from the shared error taxonomy (see main spec, §7)
	•	Deprecations follow §8

⸻

12) Security, Privacy & Audit Notes
	•	Treat this metrics contract as part of your privacy program; it enforces data minimization by design.
	•	Periodically audit telemetry for accidental content leakage (e.g., unexpected attribute growth, free-text values).
	•	Configure alerting on cardinality explosions (symptom of mislabeling or misuse of labels).
	•	Ensure tenant hashing is deterministic and irreversible, and that salt/keys (if any) are managed securely.

⸻

13) FAQ (Agnostic)

Q: Can we add a new deadline bucket for sub-millisecond calls?
A: Not under metrics_version=1.0. Propose a new version if truly needed.

Q: Can we tag model names on every call?
A: Only if extensions.tag_model_in_metrics=true. Otherwise omit to protect cardinality and privacy.

Q: Can we emit per-chunk streaming metrics?
A: Allowed for internal debug, but not part of the stable contract. The only required streaming metric is the single final outcome counter (count_stream_final_outcome).

Q: Can we introduce new op values specific to our adapter?
A: Not under metrics_version=1.0. Use the canonical sets in §2.4. If you need additional operation types, propose a spec and metrics_version update.

⸻

End of METRICS.md (metrics_version 1.0)

