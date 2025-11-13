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

- **`QUICK_START.md` / `ADAPTER_RECIPES.md`**  
  “Do this to get started” guides.

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

### 2.3 Test Suite Overview

The conformance tests are organized like:

```text
tests/
  schema/
  golden/
  llm/
  embedding/
  vector/
  graph/
  shared/   # utilities, fixtures
````

High-level:

* `tests/schema/`
  Schema validation for requests / responses.

* `tests/golden/`
  Golden wire messages. Ensures your implementation can handle “known-good” canonical envelopes.

* `tests/llm/`
  LLM-specific behavioral + schema tests.

* `tests/embedding/`
  Embedding-specific behavioral + schema tests.

* `tests/vector/`
  Vector-specific behavioral + schema tests.

* `tests/graph/`
  Graph-specific behavioral + schema tests.

---

## 3. Running the Test Suites

### 3.1 Prerequisites

* Python (exact version in your repo; usually 3.10+)
* All test dependencies installed (e.g. `pip install -e .[dev,test]`)
* Local or test deployment of your adapter(s) running, or in-process fixtures
  (Your repo should document which your tests expect.)

Common env vars:

* Provider API keys (e.g. `OPENAI_API_KEY`, `MY_VENDOR_KEY`)
* Adapter endpoint URLs (e.g. `EMBEDDING_ENDPOINT`, `LLM_ENDPOINT`)
* Test-mode flags (e.g. `CORPUS_TEST_MODE=1`)

Check your repo’s `README` / `Makefile` for specifics.

### 3.2 Local “All Up” Run

Most repos expose a single “run it all” command:

```bash
# Full suite: schema + behavioral for all protocols you implement
make test-conformance
# or
pytest tests/ -m "conformance" -v
```

You should see:

* Aggregated pass/fail summary
* Per-test details for failures
* Exit code `0` if you’re conformant, non-zero otherwise

### 3.3 Per-Protocol Runs

To iterate quickly:

```bash
# Schema-only
pytest tests/schema/ tests/golden/ -v

# LLM only
pytest tests/llm/ -v

# Embedding only
pytest tests/embedding/ -v

# Vector only
pytest tests/vector/ -v

# Graph only
pytest tests/graph/ -v
```

You can also use pytest’s `-k` filter for specific subsets:

```bash
pytest tests/embedding/ -k "truncation" -vv
pytest tests/llm/ -k "streaming" -vv
```

### 3.4 CI Integration

Recommended CI layout:

* **Job 1 – Schema:** `pytest tests/schema tests/golden`
* **Job 2 – LLM:** `pytest tests/llm`
* **Job 3 – Embedding:** `pytest tests/embedding`
* **Job 4 – Vector:** `pytest tests/vector`
* **Job 5 – Graph:** `pytest tests/graph`

Ideas:

* Run **per-protocol** suites on PRs that touch that code path.
* Run the **full suite nightly** (all protocols + schema).
* Publish artifacts:

  * HTML coverage report
  * JUnit XML (for build dashboards)
  * A “conformance status” markdown badge or summary

---

## 4. Schema Conformance

### 4.1 What Schema Conformance Validates

Schema conformance ensures that:

* Your service accepts **valid Corpus envelopes** and rejects invalid ones.
* Request / response bodies match the **canonical JSON schemas**:

  * Required vs optional fields
  * Enum values
  * Nested structures (e.g. messages, filters, embeddings, matches)
* Golden wire messages continue to round-trip correctly over time.

This is what lets routers and SDKs treat any conformant implementation as “just another Corpus endpoint.”

### 4.2 Test Suites and Files

You’ll typically see:

* `tests/schema/test_schemas_valid.py`

  * Ensures that canonical request/response objects match the JSON schema.

* `tests/schema/test_schemas_invalid.py`

  * Ensures invalid variants are rejected as expected.

* `tests/golden/test_golden_samples.py`

  * Sends “golden” envelopes (one for each operation) through your handler and asserts:

    * Success or expected error code
    * Response shape matches schema

### 4.3 How to Fix Schema Failures

Common patterns:

* **Missing fields**

  * e.g. missing `model` or `embedding.dimensions`
    → Check your serialization: are you omitting fields you think are optional but are actually required?

* **Wrong types**

  * e.g. `dimensions` as string instead of integer
    → Align your encoder / Pydantic / dataclass definitions with the schema.

* **Unexpected extra fields**

  * Some schemas disallow unknown properties.
    → Drop unused fields or tuck them into `details` / `metadata` fields where allowed.

If a test points at the golden suite:

* Compare the failing payload with the corresponding **example in `SCHEMA_CONFORMANCE.md`**.
* Make sure your wire handler isn’t doing per-adapter quirks (e.g., renaming fields).

---

## 5. Behavioral Conformance

### 5.1 What Behavioral Conformance Covers

Behavioral tests focus on **how** your service behaves, not just the JSON shape:

* **Deadlines & timeouts**

  * Respect `ctx.deadline_ms` and return `DEADLINE_EXCEEDED` appropriately.

* **Error taxonomy & mapping**

  * Map provider failures to the canonical error classes & codes.

* **Streaming semantics**

  * Correct ordering and termination for streaming ops.

* **Batch behavior & partial failures**

  * Especially for embedding and vector ops.

* **Caching and idempotency**

  * No surprising nondeterminism where the spec requires stability.

The exact expectations live in `BEHAVIORAL_CONFORMANCE.md`.

---

### 5.2 LLM Behavioral Expectations

Key behaviors (examples, not exhaustive):

* **Context window enforcement**

  * If `supports_count_tokens=True`, the base or your adapter must:

    * Count prompt + completion tokens.
    * Reject requests that exceed `max_context_length` with a canonical error.

* **Streaming rules**

  * Exactly one final chunk where `is_final=True`.
  * No chunks after an error or final chunk.
  * Streaming tests will send prompts that yield:

    * Multiple partial chunks
    * Terminal chunk
    * Error mid-stream

* **Error mapping**

  * Rate limits → `RESOURCE_EXHAUSTED`
  * Auth failures → `AUTH_ERROR`
  * Invalid requests → `BAD_REQUEST`
  * Upstream outages → `UNAVAILABLE`

Tests to watch:

* `tests/llm/test_streaming_semantics.py`
* `tests/llm/test_message_validation.py`
* `tests/llm/test_sampling_params_validation.py`

---

### 5.3 Embedding Behavioral Expectations

Highlights:

* **Truncation semantics**

  * If `max_text_length` set:

    * With `truncate=True`: text is truncated, and `truncated=True` in the result.
    * With `truncate=False`: raise `TextTooLong` when over the limit.

* **Normalization semantics**

  * If `normalize=True`:

    * `supports_normalization` must be `True`.
    * Either:

      * Provider returns normalized vectors; or
      * The base normalizes them when `normalizes_at_source=False`.

* **Batch behavior**

  * `BatchEmbedResult.failed_texts` must contain:

    * `index`, `text`, `code`, `message` for failures.
  * Tests verify both “all succeed” and “partial failures” cases.

Look for:

* `tests/embedding/test_truncation_and_text_length.py`
* `tests/embedding/test_normalization_semantics.py`
* `tests/embedding/test_embed_batch_basic.py`

---

### 5.4 Vector Behavioral Expectations

Core expectations:

* **Dimensions**

  * All vectors must match protocol expectations and respect `max_dimensions`.
  * Mismatched dimensions → `DimensionMismatch` / canonical error.

* **Query semantics**

  * Deterministic ordering of matches by score.
  * Correct application of `top_k`.
  * Respect `include_vectors` and `include_metadata`.

* **Namespace & filters**

  * If `supports_metadata_filtering=False`, using `filter` should produce a canonical error.
  * Namespace operations (`create_namespace`, `delete_namespace`) should be idempotent.

Tests to inspect:

* `tests/vector/test_query_basic.py`
* `tests/vector/test_namespace_operations.py`
* `tests/vector/test_dimension_validation.py`

---

### 5.5 Graph Behavioral Expectations

Graph tests focus on:

* **Query vs streaming query**

  * Unary `query` returns a complete result set.
  * `stream_query` returns a sequence of events:

    * Row events
    * Exactly one terminal event (end or error).

* **Batch semantics**

  * Partial successes encoded per operation.
  * No “everything failed” response when some ops succeeded.

* **Dialect & validation**

  * Unknown / unsupported dialects must produce canonical errors.
  * Invalid queries surface as `InvalidQuery` (or equivalent) instead of generic `UNAVAILABLE`.

Look for:

* `tests/graph/test_query_basic.py`
* `tests/graph/test_streaming_semantics.py`
* `tests/graph/test_batch_operations.py`

---

### 5.6 Cross-Protocol Foundations

Shared expectations:

* **OperationContext**

  * The `ctx` object carries:

    * `request_id`, `idempotency_key`
    * `deadline_ms`, `tenant`, `traceparent`
  * Tests ensure:

    * Deadlines are respected.
    * Tenant is used consistently when required.
    * Context is preserved end-to-end.

* **Error codes & retryability**

  * Tests assert specific `code` values and error classes for common failure modes.

* **Observability & security**

  * Some tests indirectly verify that:

    * Tenant IDs are hashed in metrics.
    * Logs / metrics do not leak raw embeddings or large texts in disallowed places.

If a failure references “common foundation,” look into `BEHAVIORAL_CONFORMANCE.md` §Common Patterns and `IMPLEMENTATION.md`’s OperationContext + metrics sections.

---

## 6. Certification Levels and Thresholds

> The exact numbers and thresholds live in `CERTIFICATION.md`.
> This section is about how **conformance results feed certification**, not the specific math.

### 6.1 Suite Levels (Platinum / Silver / Development)

The suite (all protocols + schema) supports:

* **Platinum**

  * Pass essentially **all normative tests** for the protocols you claim.
  * Intended for production-grade, fully conformant implementations.

* **Silver**

  * Pass a large majority of tests across major protocols.
  * Suitable for serious production use with some limited feature gaps.

* **Development**

  * Pass a minimum bar of tests per protocol.
  * Intended for “in active development” adapters.

### 6.2 Protocol-Level Certifications

You can also certify **per protocol**:

* LLM V1.0 Gold / Silver / Development
* Embedding V1.0 Gold / Silver / Development
* Vector V1.0 Gold / Silver / Development
* Graph V1.0 Gold / Silver / Development
* Schema V1.0 Gold / Silver / Development

Each has:

* **Gold:** full protocol test completion
* **Silver:** 80%+ of protocol tests
* **Development:** 50%+ of protocol tests

Exact numbers → see `CERTIFICATION.md`.

### 6.3 Mapping Tests → Certification

The mapping is:

1. Run the full conformance suite (schema + all relevant protocols).
2. Count tests passed per:

   * Suite
   * Protocol
   * Schema
3. Compare counts to the tables in `CERTIFICATION.md`.
4. If you meet the thresholds, you can:

   * Claim the relevant certification level.
   * Use the associated badge.

### 6.4 Stable vs Experimental Areas

Tests fall into two buckets:

* **Normative / stable**

  * Tied directly to MUST / MUST NOT sections of the spec.
  * Will not change behaviorally within a major protocol version without a deprecation path.

* **Advisory / experimental**

  * May exercise SHOULD / SHOULD NOT behaviors.
  * May evolve faster across minor versions.
  * Failing these may not block lower certification levels.

When in doubt: `BEHAVIORAL_CONFORMANCE.md` and `SCHEMA_CONFORMANCE.md` will label tests as “normative” vs “recommended.”

---

## 7. Debugging Failing Tests

### 7.1 Reading Test Output

Pytest output will show something like:

```text
FAILED tests/embedding/test_truncation_and_text_length.py::test_truncate_true_sets_flag
E   AssertionError: assert response["truncated"] is True
```

Use:

* `-vv` for more detail
* `-k <pattern>` to re-run a single failing area, e.g.:

```bash
pytest tests/embedding/test_truncation_and_text_length.py::test_truncate_true_sets_flag -vv
```

### 7.2 Common Failure Categories

1. **Schema mismatch**

   * Fix your wire shapes (see §4).

2. **Wrong error code**

   * Provider error mapping incorrect (see §5.1, §5.6).

3. **Streaming violations**

   * Multiple final chunks
   * Chunks after an error
   * Missing `is_final` flag

4. **Deadline misbehavior**

   * Ignoring `ctx.deadline_ms`
   * Not timing out when expected

5. **Multi-tenant / context issues**

   * Treating `tenant` incorrectly
   * Ignoring `idempotency_key` or `request_id` where tests expect behavior.

### 7.3 Guided Triage Flow

When you hit a failing test:

1. **Open the test file.**
   The test will usually name the exact behavior it expects.

2. **Identify the operation.**
   e.g., `llm.complete`, `embedding.embed_batch`.

3. **Check the relevant spec section.**

   * `BEHAVIORAL_CONFORMANCE.md` for semantics.
   * `SCHEMA_CONFORMANCE.md` for shapes.

4. **Check your adapter implementation.**

   * Does it follow the pattern in `IMPLEMENTATION.md`?
   * Are you re-implementing something the base already handles?

5. **Instrument and re-run.**

   * Add logs (with redaction).
   * Use request IDs from `ctx` to correlate.

### 7.4 Debugging Tools and Techniques

Suggestions:

* **Structured logging around `_do_*`**

  * Log:

    * operation
    * model / namespace
    * truncated flags
    * error classes and codes
  * Never log raw embeddings or full user texts.

* **Trace IDs**

  * Echo `ctx.request_id` into your logs and upstream calls.
  * Include `traceparent` if you propagate distributed tracing.

* **Minimal input reproduction**

  * Take the envelope from the failing test and send it directly via curl or a small script.
  * Compare logs to pytest output.

---

## 8. Hard Requirements vs. Flexibility

### 8.1 MUST / MUST NOT Behaviors

Non-negotiable behaviors (examples):

* Use canonical **error codes + classes** for common failure types.
* Respect **deadlines**, returning `DEADLINE_EXCEEDED` appropriately.
* Follow **streaming state machine**:

  * zero or more data events
  * exactly one terminal event (final chunk or error)
  * nothing after terminal.
* Enforce **schema compatibility**:

  * Required fields
  * Types
  * No incompatible changes.

Failing these will typically:

* Fail core tests, and
* Disqualify you from higher certification levels.

### 8.2 SHOULD / SHOULD NOT Behaviors

These behaviors are strongly recommended but may not be strictly required for lower levels:

* Using **token counting** to enforce context windows when supported.
* Providing detailed error `details` when safe.
* Implementing **standalone** mode protections (circuit breaker, rate limiter) if you bypass the base defaults.
* Returning rich **health** information for observability.

These are still enforced by some tests, but failing them may only affect **Silver/Development** vs **Platinum** standing.

### 8.3 Versioning and Compatibility

Conformance is versioned with the protocol:

* **Major version** (e.g. v1 → v2):

  * May introduce breaking changes and new conformance requirements.
  * Requires re-certification.

* **Minor version / patch**:

  * Should be backward compatible.
  * Tests may gain coverage but not change semantics for existing behaviors.

`CERTIFICATION.md` will specify which protocol + test suite version a certification applies to.

---

## 9. Integrating Conformance into Your Release Process

### 9.1 Local Dev Workflow

Recommended for adapter authors:

* **On feature branches**:

  * Run protocol-specific suites for the component you’re touching:

    * e.g. `pytest tests/embedding/` if you changed an embedding adapter.

* **Before merging to main**:

  * Run at least:

    * `tests/schema/`
    * The protocols modified in the PR.

### 9.2 CI / CD Requirements

Suggested gates:

* **Required** for merge:

  * Schema + relevant protocol tests passing.
* **Required** for release:

  * Full suite passing (all protocols you claim to support).
* **Nice-to-have**:

  * A nightly job running the full suites even if not all protocols are currently enabled in production.

### 9.3 Tracking Conformance Over Time

Best practices:

* Store conformance run results for each release:

  * Test suite version
  * Git SHA
  * Date
  * Pass/fail matrix per protocol

* Tie certification claims to specific:

  * Adapter versions
  * Test suite versions
  * Protocol major versions

* If a future change causes regressions:

  * CI should catch it.
  * You can decide whether to:

    * Fix the regression, or
    * Downgrade certification level.

---

## 10. Example Conformance Profiles

### 10.1 Fully Conformant Adapter (Platinum-style)

* Implements: LLM, Embedding, Vector, Graph
* All schema + behavioral tests pass.
* Certification:

  * Suite: Platinum
  * Protocol: Gold for each component

CI snippet:

```bash
pytest tests/schema tests/golden tests/llm tests/embedding tests/vector tests/graph -v
```

### 10.2 Protocol-Specific Adapter (e.g., Embedding Only)

* Implements: Embedding only
* Runs:

  * `tests/schema/`
  * `tests/golden/`
  * `tests/embedding/`
* Certification:

  * Suite: may qualify at Development/Silver depending on thresholds
  * Protocol: Embedding V1.0 Gold/Silver/Development

### 10.3 In-Development Adapter

* New provider integration in progress
* Strategy:

  * Start by passing simpler behavioral tests (basic operations, health).
  * Gradually enable:

    * Deadline tests
    * Edge-case streaming tests
    * Partial failure tests
* Certification:

  * Target Development level first, then iterate toward Silver/Platinum.

---

## 11. FAQ and Troubleshooting

**Q: Do I need to pass *all* tests to be “conformant”?**
A: For **full, Platinum-style conformance**: effectively yes, for the protocols you claim and all normative tests. Lower certification levels allow some gaps; see `CERTIFICATION.md`.

**Q: Can I skip schema tests if I only use the official SDK?**
A: No. Schema tests verify your **wire contract**, which remains relevant even if you use the SDK. They also catch misconfigurations and customizations.

**Q: Why are streaming tests so strict?**
A: Because routers and clients rely on precise streaming semantics to free resources, apply backpressure correctly, and avoid leaks.

**Q: How do I know if a failure is a test bug vs my bug?**
A:

1. Read the test file and spec section it references.
2. If your behavior matches the spec but the test disagrees, open an issue with:

   * Test name
   * Logs / envelopes
   * Your reasoning.
     Test suites are versioned; genuine issues do get fixed.

**Q: What happens when the test suite changes?**
A: `CERTIFICATION.md` will identify test suite versions. A new test suite may require re-running conformance and potentially re-certification for new behavior.

---

## 12. Appendix

### 12.1 Suggested Test Index (Example)

> The actual index lives next to the tests; this is an example layout.

| Area                    | File / Folder                                        | Protocol  |
| ----------------------- | ---------------------------------------------------- | --------- |
| LLM streaming           | `tests/llm/test_streaming_semantics.py`              | LLM       |
| LLM tokens              | `tests/llm/test_sampling_params_validation.py`       | LLM       |
| Embedding truncation    | `tests/embedding/test_truncation_and_text_length.py` | Embedding |
| Embedding normalization | `tests/embedding/test_normalization_semantics.py`    | Embedding |
| Vector query            | `tests/vector/test_query_basic.py`                   | Vector    |
| Vector dims             | `tests/vector/test_dimension_validation.py`          | Vector    |
| Graph streaming         | `tests/graph/test_streaming_semantics.py`            | Graph     |
| Schema lint             | `tests/schema/test_schema_lint.py`                   | Schema    |
| Golden samples          | `tests/golden/test_golden_samples.py`                | All       |

### 12.2 Glossary

* **Adapter** — Implementation that plugs a provider (OpenAI, Vertex, in-house) into the Corpus Protocol via the base SDK.

* **Wire handler** — Component that takes JSON envelopes, calls adapter methods, and returns canonical envelopes.

* **OperationContext (`ctx`)** — Per-request context; carries deadline, tenant, tracing, etc.

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
**Status:** Living document; aligned with Corpus Protocol Suite v1.0 conformance tests.

```
