# BEHAVIORAL_CONFORMANCE.md

> ✅ **Corpus Protocol (v1.0) — Behavioral Conformance (Runtime Semantics)**
> Components: **LLM** • **Vector** • **Embedding** • **Graph**
> Scope: **adapter logic**, **deadlines**, **errors & retryability**, **streaming semantics**, **observability/SIEM**, **caching/idempotency**, **limits enforcement**
> Out of scope: **schema shape** (see `SCHEMA_CONFORMANCE.md`)

---

## 1) Overview

This document specifies **runtime semantics** an implementation MUST satisfy beyond schema shape. Passing behavioral conformance means your adapter's *behavior* matches the protocol across deadlines, streaming lifecycles, error taxonomy mapping, normalization/truncation rules, caching, observability, and limits enforcement.

* **Status:** Stable / Production-ready
* **Applies To:** `llm/v1.0`, `vector/v1.0`, `embedding/v1.0`, `graph/v1.0`
* **Tests live in your repo:** `tests/<component>/` (authoritative; pass **unmodified**)

This document is the **runtime counterpart** to `SCHEMA_CONFORMANCE.md`.  
Where `SCHEMA_CONFORMANCE.md` defines what the wire **looks like** (data shape), this file defines what your adapter **does** at runtime (behavior). It doubles as the **syllabus** for the behavioral conformance tests and as a map from **spec → tests → implementation behavior**.

---

## 2) Runbook (Makefile-Aligned)

**All components (with coverage)**

```bash
make test-conformance
# or:
make verify

Per-component (with coverage)

make test-llm-conformance
make test-vector-conformance
make test-graph-conformance
make test-embedding-conformance

Fast lanes (no coverage, skip @slow)

make test-fast
make test-fast-llm
make test-fast-vector
make test-fast-graph
make test-fast-embedding

Safety & smoke

make validate-env    # warn if CORPUS_TEST_ENV unset
make safety-check    # block full suite in production
make quick-check     # minimal smoke run
make conformance-report


⸻

3) Scope & Non-Goals

In scope (behavior)
Deadline budgets & failure semantics • Streaming state machine & terminal rules • Error taxonomy mapping + retryability hints • Normalization & truncation rules (Embedding) • Token counting behavior (LLM/Embedding) • Caching & idempotency (Embedding/Vector) • Limits & model support enforcement • Observability/SIEM hygiene.

Out of scope (schema)
JSON Schema validity, $id/$ref, golden message shapes → SCHEMA_CONFORMANCE.md.

⸻

4) Test Authority

The behavioral tests in tests/<component>/ directories are the authoritative specification.

Key test patterns
	•	test_*_basic.py — Core operation semantics
	•	test_*_semantics.py — Behavioral requirements
	•	test_error_*.py — Error handling & retry logic
	•	test_*_validation.py — Input validation rules

Explore these directories to understand requirements. Each test file is designed to be self-documenting.

Related canonical documents:
	•	SCHEMA_CONFORMANCE.md — schema and wire-shape rules
	•	ERRORS.md — canonical error codes and taxonomy
	•	METRICS.md — metric names and required labels

⸻

5) Annotated Test Suite Index (Gold Standard)

This index preserves discoverability and adds triage metadata.
Tags highlight intent; Focus indicates the primary behavioral domain under verification.

5.1 Embedding — tests/embedding/

File	Purpose / Tags	Focus
run_conformance.py	harness / ordering / environment smoke	Test Infrastructure
test_embed_basic.py	core unary; required fields; model support	Core Operations
test_embed_batch_basic.py	batch semantics; per-index failures; ordering	Batch Processing
test_truncation_and_text_length.py	truncate flags; max_text_length	Input Handling
test_normalization_semantics.py	normalize; normalizes_at_source	Output Quality
test_count_tokens_behavior.py	token monotonicity; unicode; model gating	Tokenization
test_cache_and_batch_fallback.py	caching/idempotency; batch→per-item fallback	Performance & Fallbacks
test_error_mapping_retryable.py	taxonomy mapping; retry hints	Error Handling
test_deadline_enforcement.py	budgets; fast-fail; propagation	Timing & Deadlines
test_capabilities_shape.py	caps truthfulness vs behavior	Capability Verification
test_health_report.py	degraded health shape + meaning	Health Monitoring
test_context_siem.py	observability/SIEM; one-observe; no PII	Observability
test_wire_handler.py	canonical envelopes; unknown ops	Protocol Compliance

5.2 Graph — tests/graph/

File	Purpose / Tags	Focus
test_crud_basic.py	vertex/edge CRUD lifecycle	Core Operations
test_query_basic.py	query lifecycle; result coherence	Query Processing
test_streaming_semantics.py	frames; one terminal; no data after terminal	Streaming
test_batch_operations.py	batch + partial success; ack semantics	Batch Processing
test_schema_operations.py	DDL/dialect schema ops behavior	Schema Management
test_dialect_validation.py	dialect flags & validation	Configuration
test_capabilities_shape.py	caps truthfulness	Capability Verification
test_error_mapping_retryable.py	taxonomy mapping	Error Handling
test_deadline_enforcement.py	budgets	Timing & Deadlines
test_health_report.py	health under degradation	Health Monitoring
test_context_siem.py	observability/SIEM	Observability
test_wire_handler.py	canonical envelopes	Protocol Compliance

5.3 LLM — tests/llm/

File	Purpose / Tags	Focus
test_complete_basic.py	completion lifecycle; params honored	Core Operations
test_streaming_semantics.py	data/end/error stream rules	Streaming
test_count_tokens_consistency.py	token monotonicity/consistency	Tokenization
test_sampling_params_validation.py	sampling semantics; validation	Configuration
test_message_validation.py	message/tool-calling surfaces	Input Validation
test_capabilities_shape.py	caps truthfulness	Capability Verification
test_error_mapping_retryable.py	taxonomy mapping	Error Handling
test_deadline_enforcement.py	budgets	Timing & Deadlines
test_health_report.py	degraded health	Health Monitoring
test_context_siem.py	observability/SIEM	Observability
test_wire_handler.py	canonical envelopes	Protocol Compliance

5.4 Vector — tests/vector/

File	Purpose / Tags	Focus
test_upsert_basic.py	upsert lifecycle	Core Operations
test_query_basic.py	query lifecycle; scoring shape	Query Processing
test_dimension_validation.py	dimension caps; consistency	Input Validation
test_filtering_semantics.py	filter correctness	Query Processing
test_delete_operations.py	delete lifecycle	Core Operations
test_namespace_operations.py	namespace create/delete	Resource Management
test_batch_size_limits.py	batch limits	Performance & Limits
test_capabilities_shape.py	caps truthfulness	Capability Verification
test_error_mapping_retryable.py	taxonomy mapping	Error Handling
test_deadline_enforcement.py	budgets	Timing & Deadlines
test_health_report.py	health reporting	Health Monitoring
test_context_siem.py	observability/SIEM	Observability
test_wire_handler.py	canonical envelopes	Protocol Compliance

How to use this index: Start from the file tagged with the behavior you’re debugging (e.g., taxonomy mapping). Use the Focus column to understand the primary domain being verified.

⸻

6) Behavioral Requirements (Normative)

6.1 Deadlines & Budgets
	•	Pre-expired ⇒ fast-fail (no backend work).
	•	Budget propagation across nested calls; never negative.
	•	Short budgets may raise DEADLINE_EXCEEDED; handler surfaces canonical envelope and emits deadline tags.

Condition	Action	Required Signals
deadline_ms <= now	Fail fast	Error DEADLINE_EXCEEDED; no backend RPCs or child spans
deadline_ms elapses mid-op	Abort promptly	Canonical error + observation with deadline_bucket
No deadline	Proceed	No deadline tags


⸻

6.2 Streaming Semantics (LLM, Graph)
	•	Exactly one terminal (end or error) per stream.
	•	No data after terminal; mid-stream error is terminal.
	•	Keep-alives/heartbeats MUST NOT violate pacing/backpressure.
	•	The single terminal frame MUST correspond to exactly one count_stream_final_outcome metric emission (see METRICS.md).

START -> DATA* -> (END | ERROR) -> STOP
            ^           |
            |-----------|
(No DATA after END/ERROR; ERROR is terminal)

Example (JSONL-style stream):

{"type": "data", "delta": {"text": "Hello"}}
{"type": "data", "delta": {"text": " world"}}
{"type": "end"}

Invalid (data after terminal, MUST NOT happen):

{"type": "data", "delta": {"text": "Hello"}}
{"type": "end"}
{"type": "data", "delta": {"text": " again"}}  // ❌ invalid


⸻

6.3 Error Taxonomy & Retryability
	•	Deterministic mapping from provider error → canonical code.
	•	Canonical codes MUST be those defined in ERRORS.md (e.g. BAD_REQUEST, RESOURCE_EXHAUSTED, DEADLINE_EXCEEDED).
	•	Retryable errors include structured hints (e.g., retry_after_ms when available).
	•	Validation errors are non-retryable and include structured validation_errors when applicable.

Class	Canonical examples	Retryable?	Notes
Validation	BAD_REQUEST, TEXT_TOO_LONG, MODEL_NOT_AVAILABLE	No	Include validation_errors where relevant
Resource	RESOURCE_EXHAUSTED	Yes	Prefer retry_after_ms
Transport	UNAVAILABLE, TRANSIENT_NETWORK	Yes	Backoff expected
Deadline	DEADLINE_EXCEEDED	No	Emitted locally
Internal	INTERNAL	No	Don’t mark retryable unless clearly transient

Example: retryable resource error

{
  "error": {
    "code": "RESOURCE_EXHAUSTED",
    "message": "Rate limit exceeded",
    "retryable": true,
    "retry_after_ms": 1200
  }
}

Example: non-retryable validation error

{
  "error": {
    "code": "BAD_REQUEST",
    "message": "input too long",
    "retryable": false,
    "validation_errors": [
      {
        "field": "input[0].text",
        "code": "TEXT_TOO_LONG",
        "max_length": 8192
      }
    ]
  }
}


⸻

6.4 Normalization & Truncation (Embedding)
	•	normalize=true ⇒ vectors approximately unit-norm when supported.
	•	truncate=true ⇒ cap to max_text_length; set truncated=true.
	•	normalizes_at_source=true ⇒ base MUST NOT re-normalize.

⸻

6.5 Token Counting (LLM/Embedding)
	•	Monotonicity: For any two inputs A and B where A is a strict prefix of B under the same model, tokens(B) >= tokens(A) MUST hold.
	•	Unicode-safe: combining marks / surrogate pairs handled safely.
	•	Model-gated: unsupported models return canonical error.

⸻

6.6 Caching & Idempotency
	•	Cache keys are tenant-aware and content-addressed (no raw text).
	•	No cross-tenant bleed.
	•	Batch→per-item fallback preserves ordering and per-index error reporting.

key = hash(tenant_hash, op, model, normalize, params, sha256(text|payload))


⸻

6.7 Limits & Model Support
	•	Enforce declared maxes (context window, vector dims, batch size, namespaces).
	•	Unknown/unsupported model ⇒ MODEL_NOT_AVAILABLE / NOT_SUPPORTED.

⸻

6.8 Observability & SIEM Hygiene
	•	Exactly one observation per op; stable low-cardinality tags.
	•	Never emit raw text, raw vectors, or plain tenant IDs (hash only).
	•	When deadline_ms present, emit deadline_bucket.
	•	Metrics names and required labels are defined in METRICS.md; this section constrains values and SIEM-safety, not metric shapes.

Required tags (illustrative): component, op, server, version, tenant_hash, deadline_bucket?, batch_size?.

⸻

7) Adapter Readiness Checklist
	•	Deadlines: preflight & propagation; budgets never negative
	•	Stream state machine: one terminal; no data after terminal
	•	Error taxonomy mapping deterministic; retry hints where applicable
	•	Embedding normalization & truncation semantics correct
	•	Token counting monotonic & model-gated
	•	Cache keys tenant-aware; no raw text; no cross-tenant hits
	•	Limits enforced (dims, batch, context window, namespaces)
	•	Observability: SIEM-safe, single observe/op, required tags present
	•	Wire handler rejects unknown ops; ignores unknown fields safely
	•	All suites in §5 pass unmodified in CI

⸻

8) Environment Profiles (for stable CI)
	•	Local dev: generous deadlines, fewer workers, verbose traces.
	•	CI default: PYTEST_JOBS=auto, deterministic seeds, isolated caches.
	•	Stress (opt-in): short budgets, backoffs, jitter (@slow).

⸻

9) Reproducibility
	•	Seed any stochastic components; print seed on failure.
	•	Log adapter version + capabilities snapshot at test start.
	•	Scope caches per test or clear between tests.

⸻

10) Deviations & Extensions
	•	Vendor features MAY add behavior without breaking canonical outcomes.
	•	Temporary deviations MUST be documented, flag-gated, and not change canonical results; include removal plan.

⸻

11) Pass/Fail Policy
	•	Pass: All tests in §5 for claimed components pass unmodified.
	•	Fail: Any regression in deadlines, streaming, taxonomy, caching isolation, limits enforcement, or SIEM hygiene.

⸻

12) Conformance Attestation (Template)

✅ Corpus Protocol (v1.0) — Behavioral Conformant
Components: <LLM|Vector|Embedding|Graph>
Commit: <git sha>
CI Run: <link>
Notes: All suites passed unmodified with COV_FAIL_UNDER=<N>.


⸻

13) Maintenance
	•	Keep tests authoritative; this document explains semantics and points to the suites in §5.
	•	Add tests when semantics change; mark heavy tests @slow.
	•	Update Make targets only when adding/removing a component directory.

⸻

Maintainers: Corpus SDK Team
Last Updated: 2025-11-12
Scope: Behavioral semantics only (schema: SCHEMA_CONFORMANCE.md)

