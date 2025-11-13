# Corpus Protocol (v1.0) — Conformance Guide

> **Goal:** Help you get from “my adapter sort of works” to  
> “this thing *provably* conforms to the Corpus Protocol and is certifiable.”

**Audience**

- SDK / adapter implementers (LLM, Embedding, Vector, Graph)
- Vendor platform teams
- Compliance / standards reviewers

**By the end of this guide you can:**

- Understand what **conformance** means in the Corpus world  
- Run the **right test suites** locally and in CI  
- Debug failing tests without guesswork  
- Know what’s **MUST** vs **SHOULD**  
- See how conformance ties into **certification levels**

---

## 1. What “Conformance” Means

### 1.1 Goals of Conformance

When we say an implementation “conforms to the Corpus Protocol,” we mean:

- **Wire compatibility**  
  Your service accepts and returns the canonical Corpus envelopes and schemas.

- **Behavioral guarantees**  
  Deadlines, errors, streaming, batching, and multi-tenancy all behave as the spec describes.

- **Interop across vendors**  
  A router can swap one conformant adapter for another with no surprises.

- **Safety + observability**  
  No unbounded hangs; canonical error codes; SIEM-safe telemetry.

Conformance is not a fuzzy “looks good” judgment. It’s defined as:

> **Pass the documented schema + behavioral test suites for the protocols you implement, unmodified.**

### 1.2 Relationship to Other Docs

This guide is the “how to actually *use* the tests” doc.

It sits next to:

- **`SPECIFICATION.md`**  
  Normative protocol semantics and operation definitions.

- **`SCHEMA_CONFORMANCE.md`**  
  JSON schemas for envelopes and payloads, and the tests that enforce them.

- **`BEHAVIORAL_CONFORMANCE.md`**  
  Semantic expectations: deadlines, streaming rules, error taxonomy, caching, etc.

- **`CERTIFICATION.md`**  
  How conformance test results map to **Platinum / Silver / Development** suite-level certification and per-protocol certifications.

- **`IMPLEMENTATION.md`**  
  Deep dive on base classes, `_do_*` hooks, and runtime semantics.

- **`QUICKSTART_ADAPTERS.md` / `ADAPTER_RECIPES.md`**  
  “Do this to get started” and “patterns & examples” guides.

### 1.3 Scope of This Guide

This document focuses on:

- Which **test suites** exist and what they cover
- How to **run** those suites (locally + CI)
- How to **interpret and debug** failures
- What behavior is **MUST** vs. **SHOULD**
- How to integrate conformance into your **release process**

---

## 2. Conformance Surfaces at a Glance

### 2.1 Protocol Surfaces

Conformance is split across five surfaces:

1. **LLM Protocol**  
   - `llm.complete`, `llm.stream`, `llm.count_tokens`, `llm.capabilities`, `llm.health`

2. **Embedding Protocol**  
   - `embedding.embed`, `embedding.embed_batch`, `embedding.count_tokens`, `embedding.capabilities`, `embedding.health`

3. **Vector Protocol**  
   - `vector.query`, `vector.upsert`, `vector.delete`, namespace ops, `vector.capabilities`, `vector.health`

4. **Graph Protocol**  
   - `graph.query`, `graph.stream_query`, `graph.batch`, `graph.capabilities`, `graph.health`

5. **Schema Conformance**  
   - JSON schema shape for all envelopes and payloads  
   - Golden sample messages that exercise wire compatibility

You can implement any subset (e.g., only LLM). Conformance is measured both:

- **Per protocol**, and  
- **Across the full suite** (for certification).

### 2.2 Conformance Dimensions

The tests cover four major dimensions:

1. **Wire / schema conformance**  
   - Required fields  
   - Types, enums, discriminators  
   - Backwards-compatible contracts

2. **Behavioral semantics**  
   - Deadlines and timeouts  
   - Error taxonomy and retryability  
   - Streaming and batch semantics  
   - Token limits, truncation, normalization

3. **Cross-protocol foundations**  
   - `OperationContext` (deadline, tenant, trace)  
   - Capability discovery  
   - Idempotency and caching behavior

4. **Observability, security, and multi-tenancy**  
   - SIEM-safe logs and metrics  
   - Tenant isolation signals  
   - No PII or raw embeddings in key observability paths

### 2.3 Test Suite Layout (Actual Tree)

The conformance tests are organized roughly as:

```text
tests/
  run_conformance.py        # top-level runner (optional)
  cli.py                    # CLI helpers
  conftest.py               # shared pytest fixtures

  embedding/
    run_conformance.py
    test_cache_and_batch_fallback.py
    test_capabilities_shape.py
    test_context_siem.py
    test_count_tokens_behavior.py
    test_deadline_enforcement.py
    test_embed_basic.py
    test_embed_batch_basic.py
    test_error_mapping_retryable.py
    test_health_report.py
    test_normalization_semantics.py
    test_truncation_and_text_length.py
    test_wire_handler.py

  graph/
    run_conformance.py
    test_batch_operations.py
    test_capabilities_shape.py
    test_context_siem.py
    test_crud_basic.py
    test_deadline_enforcement.py
    test_dialect_validation.py
    test_error_mapping_retryable.py
    test_health_report.py
    test_query_basic.py
    test_schema_operations.py
    test_streaming_semantics.py
    test_wire_handler.py

  llm/
    run_conformance.py
    test_capabilities_shape.py
    test_complete_basic.py
    test_context_siem.py
    test_count_tokens_consistency.py
    test_deadline_enforcement.py
    test_error_mapping_retryable.py
    test_health_report.py
    test_message_validation.py
    test_sampling_params_validation.py
    test_streaming_semantics.py
    test_wire_handler.py

  vector/
    run_conformance.py
    test_batch_size_limits.py
    test_capabilities_shape.py
    test_context_siem.py
    test_deadline_enforcement.py
    test_delete_operations.py
    test_dimension_validation.py
    test_error_mapping_retryable.py
    test_filtering_semantics.py
    test_health_report.py
    test_namespace_operations.py
    test_query_basic.py
    test_upsert_basic.py
    test_wire_handler.py

  golden/
    test_golden_samples.py
    embedding_capabilities_request.json
    embedding_embed_batch_request.json
    embedding_embed_request.json
    embedding_embed_success.j
    embedding_health_request.json
    embedding_partial_success_result.json
    error_envelope_example.json
    graph_stream.ndjson
    graph_stream_error.ndjson
    graph_stream_query_request.json
    llm_complete_request.json
    llm_complete_success.json
    llm_count_tokens_request.json
    llm_count_tokens_success.json
    llm_stream.ndjson
    llm_stream_error.ndjson
    vector_error_dimension_mismatch.json
    vector_query_request.json
    vector_query_success.json

  schema/
    test_schema_lint.py

  utils/
    schema_registry.py
    stream_validator.py
````

High-level:

* `embedding/`, `llm/`, `vector/`, `graph/` — protocol-specific behavioral + wire tests.
* `golden/` — cross-protocol golden messages.
* `schema/` — schema lint / registry tests.
* `utils/` — helpers used by tests (`schema_registry`, stream validator).
* `run_conformance.py` — top-level helper for running the full suite (optional).

---

## 3. Running the Test Suites

### 3.1 Prerequisites

* Python (exact version in your repo; usually 3.10+)

* All test dependencies installed, e.g.:

  ```bash
  pip install -e .[dev,test]
  ```

* Local or test deployment of your adapter(s) running, or in-process fixtures
  (your repo should document which the tests expect).

Common env vars:

* Provider API keys (e.g. `OPENAI_API_KEY`, `MY_VENDOR_KEY`)
* Adapter endpoint URLs (e.g. `EMBEDDING_ENDPOINT`, `LLM_ENDPOINT`)
* Test-mode flags (e.g. `CORPUS_TEST_MODE=1`)

### 3.2 Full “All Up” Run

Depending on how you wire things, you can:

```bash
# Full suite via pytest
pytest tests/ -v

# Or via the top-level runner (if wired that way)
python -m tests.run_conformance
```

You should see:

* Aggregated pass/fail summary
* Per-test details for failures
* Exit code `0` if you’re conformant, non-zero otherwise

### 3.3 Per-Protocol Runs

To iterate faster on one component:

```bash
# LLM only
pytest tests/llm/ -v
# or
python -m tests.llm.run_conformance

# Embedding only
pytest tests/embedding/ -v
# or
python -m tests.embedding.run_conformance

# Vector only
pytest tests/vector/ -v
# or
python -m tests.vector.run_conformance

# Graph only
pytest tests/graph/ -v
# or
python -m tests.graph.run_conformance

# Schema + golden wire tests
pytest tests/schema/ tests/golden/ -v
```

Filter down to specific areas with `-k`:

```bash
pytest tests/embedding/ -k "truncation" -vv
pytest tests/llm/ -k "streaming" -vv
```

### 3.4 CI Integration

Recommended CI layout:

* **Job 1 – Schema + Golden:**
  `pytest tests/schema tests/golden`

* **Job 2 – LLM:**
  `pytest tests/llm`

* **Job 3 – Embedding:**
  `pytest tests/embedding`

* **Job 4 – Vector:**
  `pytest tests/vector`

* **Job 5 – Graph:**
  `pytest tests/graph`

Nice extras:

* Nightly job: `pytest tests/ -v`
* Publish artifacts:

  * HTML coverage
  * JUnit XML
  * A simple “conformance status” markdown or badge

---

## 4. Schema Conformance

### 4.1 What Schema Conformance Validates

Schema conformance ensures that:

* Your service accepts **valid Corpus envelopes** and rejects invalid ones.

* Request / response bodies match the **canonical JSON schemas**:

  * Required vs optional fields
  * Enum values
  * Nested structures (messages, filters, embeddings, matches, errors)

* Golden wire messages continue to round-trip correctly over time.

This is what lets routers and SDKs treat any conformant implementation as “just another Corpus endpoint”.

### 4.2 Test Suites and Files

Schema-level conformance uses:

* `tests/schema/test_schema_lint.py`

  * Validates schema meta-lint and hygiene via the shared `schema_registry`:

    * No broken refs / missing IDs
    * Consistent namespaces
    * Registrations match spec expectations

* `tests/golden/test_golden_samples.py`

  * Uses the JSON / NDJSON payloads in `tests/golden/`:

    * Request/response examples for LLM, Embedding, Vector, Graph
    * Error envelopes (`error_envelope_example.json`)
    * Streaming transcripts (`llm_stream.ndjson`, `graph_stream.ndjson`, etc.)
  * Ensures your wire handlers:

    * Accept canonical request envelopes (e.g. `llm_complete_request.json`)
    * Produce schema-conformant responses (e.g. `llm_complete_success.json`)
    * Handle golden error cases like vector dimension mismatches.

Helpers:

* `tests/utils/schema_registry.py` — shared registry that loads schema definitions.
* `tests/utils/stream_validator.py` — used to validate streaming transcripts in golden tests.

### 4.3 How to Fix Schema Failures

Common issues:

* **Missing fields**
  e.g. missing `model` or `embedding.dimensions`.
  → Check your serialization: are you omitting fields that the schema marks as required?

* **Wrong types**
  e.g. `dimensions` as string instead of integer.
  → Align your DTO / dataclass / Pydantic model with the JSON schema.

* **Unexpected extra fields**
  Some schemas disallow unknown properties.
  → Drop unused extra fields or put them into designated `metadata` / `details` fields.

For golden sample failures:

* Compare the failing payload with the corresponding file in `tests/golden/`.
* Ensure your wire handler is not renaming fields or wrapping/unwrapping in a custom envelope.

---

## 5. Behavioral Conformance

### 5.1 What Behavioral Conformance Covers

Behavioral tests focus on **how** your service behaves, not just the JSON shape:

* **Deadlines & timeouts**

  * Respect `ctx.deadline_ms`, use `ctx.remaining_ms()`, return `DEADLINE_EXCEEDED` correctly.

* **Error taxonomy & mapping**

  * Map provider failures to canonical error classes and error codes.

* **Streaming semantics**

  * Correct ordering and termination rules for streaming ops.

* **Batch behavior & partial failures**

  * Especially for embedding and vector operations.

* **Caching and idempotency**

  * No surprising nondeterminism where the spec requires stability.

The normative expectations live in `BEHAVIORAL_CONFORMANCE.md`.

---

### 5.2 LLM Behavioral Expectations

Representative tests:

* `tests/llm/test_complete_basic.py`
* `tests/llm/test_streaming_semantics.py`
* `tests/llm/test_count_tokens_consistency.py`
* `tests/llm/test_sampling_params_validation.py`
* `tests/llm/test_message_validation.py`
* `tests/llm/test_deadline_enforcement.py`
* `tests/llm/test_error_mapping_retryable.py`
* `tests/llm/test_context_siem.py`
* `tests/llm/test_health_report.py`
* `tests/llm/test_capabilities_shape.py`
* `tests/llm/test_wire_handler.py`

Key behaviors (not exhaustive):

* **Context window enforcement**

  * When `supports_count_tokens=True`, you must:

    * Count prompt + completion tokens.
    * Reject over-limit requests with the correct canonical error.

* **Streaming rules**

  * `llm.stream`:

    * Zero or more partial chunks.
    * Exactly one final chunk where `is_final=True`.
    * No chunks after an error or the final chunk.
  * `test_streaming_semantics.py` verifies state machine correctness.

* **Sampling parameters / validation**

  * `test_sampling_params_validation.py` enforces:

    * Ranges for temperature, top_p, top_k, etc.
    * Rejection of obviously invalid sampling configs.

* **Error mapping & retryability**

  * `test_error_mapping_retryable.py` checks that you map:

    * Rate limits → `RESOURCE_EXHAUSTED`
    * Auth issues → `AUTH_ERROR`
    * Invalid parameters → `BAD_REQUEST`
    * Provider outages → `UNAVAILABLE`
  * And that the “retryable vs non-retryable” dimension is consistent.

* **Deadlines & SIEM-safe context**

  * `test_deadline_enforcement.py` ensures you respect `ctx.deadline_ms`.
  * `test_context_siem.py` ensures you do *not* leak raw tenant IDs or PII into metrics/logs.

---

### 5.3 Embedding Behavioral Expectations

Representative tests:

* `tests/embedding/test_embed_basic.py`
* `tests/embedding/test_embed_batch_basic.py`
* `tests/embedding/test_truncation_and_text_length.py`
* `tests/embedding/test_normalization_semantics.py`
* `tests/embedding/test_cache_and_batch_fallback.py`
* `tests/embedding/test_count_tokens_behavior.py`
* `tests/embedding/test_deadline_enforcement.py`
* `tests/embedding/test_error_mapping_retryable.py`
* `tests/embedding/test_context_siem.py`
* `tests/embedding/test_health_report.py`
* `tests/embedding/test_capabilities_shape.py`
* `tests/embedding/test_wire_handler.py`

Highlights:

* **Truncation semantics** (`test_truncation_and_text_length.py`)

  * If `max_text_length` is set:

    * `truncate=True` → text truncated, `truncated=True` in result.
    * `truncate=False` → raise `TextTooLong` when text is over limit.

* **Normalization semantics** (`test_normalization_semantics.py`)

  * With `normalize=True`:

    * `supports_normalization` must be `True`.
    * Either:

      * Provider returns normalized vectors, or
      * The base normalizes them when `normalizes_at_source=False`.

* **Batch + partial failure behavior** (`test_embed_batch_basic.py`, `test_cache_and_batch_fallback.py`)

  * `BatchEmbedResult.failed_texts` must include:

    * `index`, `text`, `code`, `message` for failures.
  * `test_cache_and_batch_fallback.py` also validates:

    * Fallback behavior when `_do_embed_batch` is not supported or partially fails.
    * That caching and batch fallback remain consistent with spec.

* **Count tokens behavior** (`test_count_tokens_behavior.py`)

  * If `supports_token_counting=True`:

    * Behavior must be consistent across single and batch calls.
    * Over-limit requests must be rejected correctly.

* **Deadlines, errors, SIEM**

  * `test_deadline_enforcement.py` → deadlines.
  * `test_error_mapping_retryable.py` → canonical error mapping.
  * `test_context_siem.py` → metrics/logging hygiene.

---

### 5.4 Vector Behavioral Expectations

Representative tests:

* `tests/vector/test_query_basic.py`
* `tests/vector/test_upsert_basic.py`
* `tests/vector/test_delete_operations.py`
* `tests/vector/test_batch_size_limits.py`
* `tests/vector/test_filtering_semantics.py`
* `tests/vector/test_dimension_validation.py`
* `tests/vector/test_namespace_operations.py`
* `tests/vector/test_deadline_enforcement.py`
* `tests/vector/test_error_mapping_retryable.py`
* `tests/vector/test_context_siem.py`
* `tests/vector/test_health_report.py`
* `tests/vector/test_capabilities_shape.py`
* `tests/vector/test_wire_handler.py`

Core expectations:

* **Dimensions & shapes** (`test_dimension_validation.py`)

  * All vectors must have allowed dimensions ≤ `max_dimensions`.
  * Mismatches trigger `DimensionMismatch` (or equivalent canonical error).

* **Query semantics** (`test_query_basic.py`)

  * Deterministic ordering by score.
  * `top_k` honored.
  * `include_vectors` / `include_metadata` respected.

* **Namespaces + filters**

  * `test_namespace_operations.py`:

    * Creating/deleting namespaces is idempotent.
  * `test_filtering_semantics.py`:

    * Respects `supports_metadata_filtering`.
    * Errors clearly if filter is used when not supported.

* **Batch limits & operations** (`test_batch_size_limits.py`, `test_upsert_basic.py`, `test_delete_operations.py`)

  * `max_batch_size` enforced.
  * Upsert/delete return correct counts and failure information.

* **Deadlines, errors, SIEM**

  * Same pattern: `test_deadline_enforcement.py`, `test_error_mapping_retryable.py`, `test_context_siem.py`, `test_health_report.py`.

---

### 5.5 Graph Behavioral Expectations

Representative tests:

* `tests/graph/test_query_basic.py`
* `tests/graph/test_streaming_semantics.py`
* `tests/graph/test_batch_operations.py`
* `tests/graph/test_crud_basic.py`
* `tests/graph/test_schema_operations.py`
* `tests/graph/test_dialect_validation.py`
* `tests/graph/test_deadline_enforcement.py`
* `tests/graph/test_error_mapping_retryable.py`
* `tests/graph/test_context_siem.py`
* `tests/graph/test_health_report.py`
* `tests/graph/test_capabilities_shape.py`
* `tests/graph/test_wire_handler.py`

Key behaviors:

* **Query vs streaming query** (`test_query_basic.py`, `test_streaming_semantics.py`)

  * Unary `graph.query` returns a complete result.
  * `graph.stream_query` emits:

    * Zero or more row events.
    * Exactly one terminal event (end or error).
    * No events after terminal.

* **Batch semantics** (`test_batch_operations.py`)

  * Partial successes must be explicitly represented.
  * Only successful operations mutate state.
  * Input ordering / IDs preserved.

* **Dialect & schema operations** (`test_dialect_validation.py`, `test_schema_operations.py`)

  * Unsupported dialects → canonical “not supported” error.
  * Schema operations behave predictably and idempotently.

* **CRUD fundamentals** (`test_crud_basic.py`)

  * Basic vertex/edge create/read/update/delete semantics.

* **Deadlines, errors, SIEM**

  * Handled via the same foundational tests: deadline, retryable error mapping, context/metrics hygiene.

---

### 5.6 Cross-Protocol Foundations

Cross-cutting behavior is validated via the per-protocol tests plus:

* `test_context_siem.py` in each protocol:

  * Ensures tenant IDs are hashed.
  * Ensures logs/metrics don’t leak full texts or embeddings where they shouldn’t.

* `test_error_mapping_retryable.py` in each protocol:

  * Ensures canonical error codes and retryability semantics are consistent.

* `test_health_report.py`:

  * Ensures health endpoints expose enough structured info (but not secrets/PII).

The details of **OperationContext**, error taxonomy, and observability constraints live in `BEHAVIORAL_CONFORMANCE.md` and `IMPLEMENTATION.md`.

---

## 6. Certification Levels and Thresholds

> Exact thresholds (Platinum/Silver/Development, per-protocol Gold/Silver/Dev) live in `CERTIFICATION.md`.
> This section is about how **conformance results feed certification**, not the raw numbers.

### 6.1 Suite Levels (Platinum / Silver / Development)

Suite-level certification (all protocols + schema) typically offers:

* **Platinum**

  * Pass essentially **all normative tests** for all claimed protocols.
  * Intended for fully conformant, production-grade implementations.

* **Silver**

  * Pass a large majority of tests across major protocols.
  * Suitable for production usage with limited, well-understood gaps.

* **Development**

  * Pass a minimum bar of tests per protocol.
  * Intended for adapters in active development.

### 6.2 Protocol-Level Certifications

You can also certify **per protocol**:

* LLM V1.0 Gold / Silver / Development
* Embedding V1.0 Gold / Silver / Development
* Vector V1.0 Gold / Silver / Development
* Graph V1.0 Gold / Silver / Development
* Schema V1.0 Gold / Silver / Development

Each has:

* **Gold:** essentially full protocol completion.
* **Silver:** ~80%+ of protocol tests.
* **Development:** ~50%+ of protocol tests.

The exact counts → see `CERTIFICATION.md`.

### 6.3 Mapping Tests → Certification

Roughly:

1. Run the full conformance suite (`pytest tests/` or `python -m tests.run_conformance`).
2. Count tests passed per:

   * Suite
   * Protocol (`llm/`, `embedding/`, `vector/`, `graph/`)
   * Schema/golden (`schema/`, `golden/`)
3. Compare those counts to the tables in `CERTIFICATION.md`.
4. If you meet the thresholds, you can claim:

   * Suite-level certification, and/or
   * Per-protocol certifications.

### 6.4 Stable vs Experimental Areas

Tests fall into buckets:

* **Normative / stable**

  * Tied directly to MUST / MUST NOT sections of `SPECIFICATION.md`, `SCHEMA_CONFORMANCE.md`, `BEHAVIORAL_CONFORMANCE.md`.
  * These don’t change behaviorally within a major protocol version.

* **Advisory / experimental**

  * Exercise SHOULD / SHOULD NOT behaviors.
  * May evolve faster across minor versions.
  * Failing them may limit you to lower certification tiers.

---

## 7. Debugging Failing Tests

### 7.1 Reading Test Output

Pytest output will give you something like:

```text
FAILED tests/embedding/test_truncation_and_text_length.py::test_truncate_true_sets_flag
E   AssertionError: assert result.truncated is True
```

Use:

* `-vv` for full assertion diffs.
* `-k "<pattern>"` to isolate one failure:

```bash
pytest tests/embedding/test_truncation_and_text_length.py::test_truncate_true_sets_flag -vv
```

### 7.2 Common Failure Categories

1. **Schema mismatch**

   * Shapes don’t match what `test_golden_samples.py` or `test_schema_lint.py` expect.
   * Fix your DTO / serializer to align with the schema.

2. **Wrong error code or class**

   * `test_error_mapping_retryable.py` shows you expected `(code, class)` pairs.
   * Update your provider error mapping.

3. **Streaming violations**

   * `test_streaming_semantics.py` in LLM/Graph catches:

     * Multiple final events
     * Data after final/error
     * Missing `is_final` markers.

4. **Deadline misbehavior**

   * `test_deadline_enforcement.py` ensures `ctx.deadline_ms` is respected and you surface `DEADLINE_EXCEEDED` instead of silent hangs.

5. **Context / SIEM issues**

   * `test_context_siem.py` fails if:

     * Tenant IDs are logged raw.
     * PII appears in metrics labels.

### 7.3 Guided Triage Flow

When a test fails:

1. **Open the test file.**
   Tests are deliberately descriptive (function names, docstrings).

2. **Identify the operation.**
   e.g., `embedding.embed_batch`, `vector.query`, `llm.stream`.

3. **Check the spec.**

   * `BEHAVIORAL_CONFORMANCE.md` for semantics.
   * `SCHEMA_CONFORMANCE.md` for payload shapes.

4. **Check your implementation.**

   * Is your adapter following the patterns in `IMPLEMENTATION.md`?
   * Are you re-implementing things the base already handles (e.g., truncation, normalization)?

5. **Log, re-run, iterate.**

   * Add limited, redacted logging.
   * Use `ctx.request_id` to correlate logs and pytest traces.

### 7.4 Debugging Tools and Techniques

Recommendations:

* **Structured logging around `_do_*`**

  * Log fields like:

    * operation
    * model / namespace
    * truncated flags
    * error class / code
  * Never log full embeddings or full user texts.

* **Trace IDs**

  * Propagate `ctx.request_id` and `ctx.traceparent` into upstream SDKs and logs.

* **Minimal reproduction**

  * Use the request envelopes from failing golden / behavioral tests (or re-create them).
  * Send manually via curl or a tiny script to verify behavior outside pytest.

---

## 8. Hard Requirements vs. Flexibility

### 8.1 MUST / MUST NOT Behaviors

Non-negotiable patterns (examples):

* Use canonical **error codes + classes** for common failure types.
* Respect **deadlines** and emit `DEADLINE_EXCEEDED` when appropriate.
* Follow **streaming state machine** rules:

  * Zero or more data events
  * Exactly one terminal event (final chunk or error)
  * No events after terminal.
* Maintain **schema compatibility**:

  * Required fields present
  * Types correct
  * No breaking changes to envelope shape.

Failing these will typically:

* Red-light core tests, and
* Disqualify you from higher certification levels.

### 8.2 SHOULD / SHOULD NOT Behaviors

Strongly recommended but more flexible:

* Use **token counting** to enforce context windows when supported.
* Provide structured `details` in errors where safe.
* Implement robust **standalone** protections if you bypass defaults.
* Expose richer **health** payloads for observability.

Some of these are enforced, but may “only” affect Silver/Development vs Platinum.

### 8.3 Versioning and Compatibility

Conformance is versioned alongside the protocol and test suite:

* **Major version bump** (v1 → v2):

  * May introduce breaking changes and new tests.
  * Requires re-conformance and re-certification.

* **Minor / patch**:

  * Should remain backward-compatible.
  * Tests may be added but not change semantics of existing behaviors.

`CERTIFICATION.md` ties each certification level to:

* Protocol version
* Test suite version

---

## 9. Integrating Conformance into Your Release Process

### 9.1 Local Dev Workflow

For adapter authors:

* **On feature branches**:

  * Run the protocol-specific suite(s) you touched, e.g.:

    ```bash
    pytest tests/embedding/
    ```

* **Before merging to main**:

  * At minimum:

    ```bash
    pytest tests/schema/ tests/golden/
    pytest tests/<protocols-you-changed>/
    ```

### 9.2 CI / CD Requirements

Suggested gating:

* **Required for merge**:

  * Schema + golden tests.
  * Relevant protocol tests (`llm/`, `embedding/`, `vector/`, `graph/`) for changed code.

* **Required for release**:

  * Full suite:

    ```bash
    pytest tests/ -v
    ```

* **Nice-to-have**:

  * Nightly `python -m tests.run_conformance` to catch regressions early.

### 9.3 Tracking Conformance Over Time

Best practices:

* Record conformance runs:

  * Test suite version
  * Commit SHA
  * Date/time
  * Per-protocol pass/fail summary

* Tie certification claims to:

  * Adapter versions
  * Test suite versions
  * Protocol major versions

* When regressions appear:

  * CI should fail visibly.
  * You can decide to:

    * Fix the behavior, or
    * Consciously drop from Platinum → Silver / Development.

---

## 10. Example Conformance Profiles

### 10.1 Fully Conformant Multi-Protocol Adapter

* Implements: LLM, Embedding, Vector, Graph.
* All tests pass in:

  * `schema/`, `golden/`
  * `llm/`, `embedding/`, `vector/`, `graph/`.

Command:

```bash
pytest tests/schema tests/golden tests/llm tests/embedding tests/vector tests/graph -v
```

Certification:

* Suite: Platinum.
* Protocol: Gold for each protocol.

### 10.2 Single-Protocol Adapter (Embedding Only)

* Implements: Embedding only.
* Runs:

  ```bash
  pytest tests/schema tests/golden tests/embedding -v
  ```

Certification:

* Suite: likely Development/Silver depending on thresholds.
* Protocol: Embedding V1.0 Gold/Silver/Development depending on pass count.

### 10.3 In-Development Adapter

* New provider integration, WIP.
* Strategy:

  1. Start with:

     * `test_embed_basic.py`
     * `test_capabilities_shape.py`
     * `test_health_report.py`
  2. Add:

     * Deadline + error mapping tests
     * Truncation / normalization tests
     * Batch / cache tests
  3. Aim for Development tier first, then iterate to Silver/Platinum.

---

## 11. FAQ and Troubleshooting

**Q: Do I need to pass *all* tests to be “conformant”?**
A: For **full (Platinum-style) conformance** for a protocol: effectively yes, for all normative tests in that protocol and schema/golden. Lower levels (Silver/Development) allow gaps; see `CERTIFICATION.md`.

**Q: Can I skip schema tests if I only use the official SDK?**
A: No. Schema tests validate the **wire contract**, which still matters even if you’re using the SDK. They also catch configuration/customization mistakes.

**Q: Why are streaming tests so picky?**
A: Routers and clients rely on precise streaming semantics for resource cleanup and backpressure. Leaky or mis-ordered streams cause real production issues.

**Q: How do I know if a failure is a test bug vs my bug?**
A:

1. Read the failing test in `tests/<protocol>/...`.
2. Cross-check the referenced section in `SPECIFICATION.md` / `BEHAVIORAL_CONFORMANCE.md`.
3. If your behavior clearly matches spec and the test does not, open an issue with:

   * Test name
   * Logs or captured envelopes
   * Your reasoning.

**Q: What happens when the test suite changes?**
A: `CERTIFICATION.md` ties certifications to test suite versions. A new suite or version may require re-running conformance and refreshing certifications.

---

## 12. Appendix

### 12.1 Test Index (Aligned with Current Tree)

> This is a convenience map; the source of truth is the test files themselves.

| Area                         | File                                                 | Protocol  |
| ---------------------------- | ---------------------------------------------------- | --------- |
| LLM capabilities shape       | `tests/llm/test_capabilities_shape.py`               | LLM       |
| LLM basic complete           | `tests/llm/test_complete_basic.py`                   | LLM       |
| LLM streaming semantics      | `tests/llm/test_streaming_semantics.py`              | LLM       |
| LLM token / sampling checks  | `tests/llm/test_sampling_params_validation.py`       | LLM       |
| LLM token consistency        | `tests/llm/test_count_tokens_consistency.py`         | LLM       |
| LLM message validation       | `tests/llm/test_message_validation.py`               | LLM       |
| LLM deadlines & errors       | `tests/llm/test_deadline_enforcement.py`             | LLM       |
| LLM retryable errors         | `tests/llm/test_error_mapping_retryable.py`          | LLM       |
| LLM SIEM / context hygiene   | `tests/llm/test_context_siem.py`                     | LLM       |
| LLM health reporting         | `tests/llm/test_health_report.py`                    | LLM       |
| LLM wire handler             | `tests/llm/test_wire_handler.py`                     | LLM       |
| Embedding basic              | `tests/embedding/test_embed_basic.py`                | Embedding |
| Embedding batch basics       | `tests/embedding/test_embed_batch_basic.py`          | Embedding |
| Embedding truncation         | `tests/embedding/test_truncation_and_text_length.py` | Embedding |
| Embedding normalization      | `tests/embedding/test_normalization_semantics.py`    | Embedding |
| Embedding cache & fallback   | `tests/embedding/test_cache_and_batch_fallback.py`   | Embedding |
| Embedding count tokens       | `tests/embedding/test_count_tokens_behavior.py`      | Embedding |
| Embedding capabilities       | `tests/embedding/test_capabilities_shape.py`         | Embedding |
| Embedding deadlines & errors | `tests/embedding/test_deadline_enforcement.py`       | Embedding |
| Embedding retryable errors   | `tests/embedding/test_error_mapping_retryable.py`    | Embedding |
| Embedding SIEM / context     | `tests/embedding/test_context_siem.py`               | Embedding |
| Embedding health             | `tests/embedding/test_health_report.py`              | Embedding |
| Embedding wire handler       | `tests/embedding/test_wire_handler.py`               | Embedding |
| Vector query basic           | `tests/vector/test_query_basic.py`                   | Vector    |
| Vector upsert basic          | `tests/vector/test_upsert_basic.py`                  | Vector    |
| Vector delete operations     | `tests/vector/test_delete_operations.py`             | Vector    |
| Vector batch limits          | `tests/vector/test_batch_size_limits.py`             | Vector    |
| Vector filtering semantics   | `tests/vector/test_filtering_semantics.py`           | Vector    |
| Vector dimension validation  | `tests/vector/test_dimension_validation.py`          | Vector    |
| Vector namespace ops         | `tests/vector/test_namespace_operations.py`          | Vector    |
| Vector capabilities          | `tests/vector/test_capabilities_shape.py`            | Vector    |
| Vector deadlines & errors    | `tests/vector/test_deadline_enforcement.py`          | Vector    |
| Vector retryable errors      | `tests/vector/test_error_mapping_retryable.py`       | Vector    |
| Vector SIEM / context        | `tests/vector/test_context_siem.py`                  | Vector    |
| Vector health                | `tests/vector/test_health_report.py`                 | Vector    |
| Vector wire handler          | `tests/vector/test_wire_handler.py`                  | Vector    |
| Graph query basics           | `tests/graph/test_query_basic.py`                    | Graph     |
| Graph streaming semantics    | `tests/graph/test_streaming_semantics.py`            | Graph     |
| Graph batch operations       | `tests/graph/test_batch_operations.py`               | Graph     |
| Graph CRUD basics            | `tests/graph/test_crud_basic.py`                     | Graph     |
| Graph schema operations      | `tests/graph/test_schema_operations.py`              | Graph     |
| Graph dialect validation     | `tests/graph/test_dialect_validation.py`             | Graph     |
| Graph capabilities           | `tests/graph/test_capabilities_shape.py`             | Graph     |
| Graph deadlines & errors     | `tests/graph/test_deadline_enforcement.py`           | Graph     |
| Graph retryable errors       | `tests/graph/test_error_mapping_retryable.py`        | Graph     |
| Graph SIEM / context         | `tests/graph/test_context_siem.py`                   | Graph     |
| Graph health                 | `tests/graph/test_health_report.py`                  | Graph     |
| Graph wire handler           | `tests/graph/test_wire_handler.py`                   | Graph     |
| Schema lint & hygiene        | `tests/schema/test_schema_lint.py`                   | Schema    |
| Golden wire samples          | `tests/golden/test_golden_samples.py`                | All       |
| Golden LLM samples           | `tests/golden/llm_*` (JSON + NDJSON)                 | LLM       |
| Golden Embedding samples     | `tests/golden/embedding_*` (JSON)                    | Embedding |
| Golden Vector samples        | `tests/golden/vector_*` (JSON)                       | Vector    |
| Golden Graph samples         | `tests/golden/graph_*` (JSON + NDJSON)               | Graph     |

### 12.2 Glossary

* **Adapter** — Implementation that plugs a provider (OpenAI, Vertex, in-house) into the Corpus Protocol via the SDK base classes.

* **Wire handler** — Component that takes JSON envelopes, calls adapter methods, and returns canonical envelopes.

* **OperationContext (`ctx`)** — Per-request context; carries deadline, tenant, tracing, and attributes.

* **Conformance** — State of passing the normative schema + behavioral test suites for your claimed protocols.

* **Certification** — Formal badge / status derived from conformance test results (see `CERTIFICATION.md`).

### 12.3 Links

* `SPECIFICATION.md`
* `IMPLEMENTATION.md`
* `SCHEMA_CONFORMANCE.md`
* `BEHAVIORAL_CONFORMANCE.md`
* `CERTIFICATION.md`
* `QUICKSTART_ADAPTERS.md`
* `ADAPTER_RECIPES.md`

---

**Maintainers:** Corpus Standards Working Group
**Status:** Living document; aligned with Corpus Protocol Suite v1.0 conformance tests and current `tests/` tree.

```
