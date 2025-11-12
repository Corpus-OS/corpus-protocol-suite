# ADAPTER_GUIDE.md

> ✅ **Corpus Protocol (v1.0) — Adapter Implementation Guide**
> Scope: **runtime behavior** for adapters (LLM • Vector • Embedding • Graph)
> Out of scope: JSON Schema shape (see `SCHEMA_CONFORMANCE.md`)

---

## 1) Purpose & Audience

* **What this is:** A practical guide to implement a Corpus-compatible adapter that passes **behavioral conformance** (runtime semantics).
* **Who it’s for:** Adapter authors and maintainers integrating providers behind the Corpus wire contract.
* **Read alongside:**

  * `BEHAVIORAL_CONFORMANCE.md` (semantic contract & pass criteria)
  * `SCHEMA_CONFORMANCE.md` (schema/wire shape)

---

## 2) Runbook (Makefile-Aligned)

* **All components (with coverage):** `make test-conformance` (or `make verify`)
* **Per component:** `make test-llm-conformance`, `make test-vector-conformance`, `make test-graph-conformance`, `make test-embedding-conformance`
* **Fast lanes (skip @slow, no coverage):** `make test-fast-*`
* **Safety & smoke:** `make validate-env`, `make safety-check`, `make quick-check`
* **Schema smoke:** `make verify-schema` (schema meta-lint + goldens)

---

## 3) Single-File Adapter Layout (Suggested Sections)

Keep one module but **separate concerns** clearly.

1. **Header & Imports**
2. **Config & Constants**

   * limits (batch size, context/window, dims), timeouts, feature flags
3. **OperationContext & Utilities**

   * `deadline_remaining_ms()`, `now_ms()`
   * `hash_tenant()`, `content_hash()` (no raw tenant/text in keys)
4. **Error Mapping (Provider → Canonical)**

   * deterministic mapping, retryability hints, validation errors
5. **Observability (SIEM-safe)**

   * `observe_once(component, op, tags, status, err=None)`
6. **Provider Facade**

   * thin wrappers; isolate SDK peculiarities and retries
7. **Adapter API**

   * LLM: `complete`, `count_tokens`, `capabilities`, `health`
   * Embedding: `embed`, `embed_batch`, `count_tokens`, …
   * Vector: `upsert`, `query`, `delete`, namespaces …
   * Graph: `query`, `stream_query`, batch ops, ids/acks …
8. **Streaming Guardrails**

   * one terminal, no data after terminal, heartbeat policy
9. **Wire Handler**

   * canonical envelopes in/out, unknown op → `NOT_SUPPORTED`
10. **Optional Cache Hooks**

* tenant-aware, content-addressed keys; no raw text

11. **Test Hints**

* docstring pointers to the relevant conformance tests

---

## 4) Canonical Error Taxonomy & Mapping (Authoritative)

**Goals**

* Normalize provider errors into **canonical codes**.
* Attach **retry hints** (`retry_after_ms`) for retryable classes.
* Emit **`validation_errors`** for input/contract issues.

**Concrete mapping example (drop-in):**

```python
# Example error mapping
PROVIDER_TO_CANONICAL = {
    "rate_limit_exceeded": ("RESOURCE_EXHAUSTED", {"retry_after_ms": 5000}),
    "invalid_api_key": ("AUTHENTICATION_ERROR", None),
    "context_length_exceeded": ("BAD_REQUEST", {"validation_errors": [...]})
}

def map_error(provider_error) -> tuple[str, dict | None, bool]:
    """
    Returns (canonical_code, details_dict_or_none, retryable_bool).
    Ensure deterministic mapping; never leak provider internals.
    """
    code, details = PROVIDER_TO_CANONICAL.get(
        provider_error.code, ("INTERNAL", None)
    )
    retryable = code in {"RESOURCE_EXHAUSTED", "UNAVAILABLE", "TRANSIENT_NETWORK"}
    return code, details, retryable
```

**Must-haves**

* Validation class (BAD_REQUEST, TEXT_TOO_LONG, MODEL_NOT_AVAILABLE) → **non-retryable**.
* Resource/transport class (RESOURCE_EXHAUSTED, UNAVAILABLE, TRANSIENT_NETWORK) → **retryable** and include **`retry_after_ms`** when known.
* DEADLINE_EXCEEDED is produced **locally** when budget elapses.

---

## 5) Deadlines & Budgets

* **Preflight fast-fail** when `deadline_ms <= now`.
* Propagate remaining budget to all provider calls; **never negative**.
* Emit `deadline_bucket` tag when a deadline is set.
* Abort promptly and surface `DEADLINE_EXCEEDED` if time elapses mid-op.

---

## 6) Streaming State Machine (LLM, Graph)

**Rules**

* Exactly **one terminal** (`end` or `error`) per stream.
* **No data** after terminal; mid-stream `error` is terminal.
* Heartbeats allowed but must respect pacing limits.

**State**

```
START -> DATA* -> (END | ERROR) -> STOP
            ^           |
            |-----------|
```

---

## 7) Normalization & Truncation (Embedding)

* `normalize=true` → approximately **unit-norm** when supported;
* `normalizes_at_source=true` → **do not** re-normalize downstream;
* `truncate=true` → cap to `max_text_length` and set `truncated=True`;
* `truncate=false` with over-limit → `TEXT_TOO_LONG`.

---

## 8) Token Counting (LLM/Embedding)

* **Monotonic:** longer input must not yield fewer tokens.
* **Unicode-safe** and **model-gated** (unknown model → `MODEL_NOT_AVAILABLE` or `NOT_SUPPORTED`).

---

## 9) Caching & Idempotency (Where Applicable)

* Keys are **tenant-aware** and **content-addressed** (no raw text).
* **No cross-tenant bleed**.
* Batch fallback to per-item preserves input ordering and emits per-index failures.

**Key derivation (example)**

```
key = hash(tenant_hash, op, model, normalize, params, sha256(text|payload))
```

---

## 10) Limits & Model Enforcement

* Enforce declared **batch size**, **context window**, **dimensions**, **namespace** limits.
* Unknown/unsupported model → `MODEL_NOT_AVAILABLE` / `NOT_SUPPORTED`.

---

## 11) Observability & SIEM Hygiene

* **Exactly one** `observe` per op.
* **No raw** text, vectors, or tenant IDs.
* Emit stable, low-cardinality tags: `component`, `op`, `server`, `version`, `tenant_hash`, `deadline_bucket?`, `batch_size?`.

---

## 12) Wire Handler Rules

* Canonical envelopes in/out; ignore unknown fields safely.
* Unknown op → normalized `NOT_SUPPORTED`.
* Ensure `schema_version` populated on success envelopes.

---

## 13) Minimal Adapter Stubs (Illustrative)

* **`capabilities()`** returns truthful limits/flags.
* **`health()`** always stable shape; degraded mode surfaces reason/status.
* Per-op methods perform preflight deadline check, map errors canonically, emit single observation.

*(Keep this section as commented skeletons inline in your real adapter file—no duplication here.)*

---

## 14) When Tests Fail: Diagnostic Flow

**Schema failures?** → Run `make verify-schema`
**Behavioral failures?** → Open the specific test from `BEHAVIORAL_CONFORMANCE.md §5`
**Streaming issues?** → Revisit **§6 Streaming State Machine**
**Error mapping wrong?** → Revisit **§4 Error Taxonomy & Mapping**

---

## 15) Adapter Readiness Checklist

* [ ] Deadline preflight + budget propagation; never negative
* [ ] Streaming: one terminal; no data after terminal; heartbeats sane
* [ ] Error mapping deterministic; retry hints where applicable
* [ ] Embedding normalization/truncation semantics correct
* [ ] Token counting monotonic & model-gated
* [ ] Cache keys tenant-aware; no raw text; no cross-tenant hits
* [ ] Limits enforced (dims, batch, context, namespaces)
* [ ] Observability single-observe; SIEM-safe; required tags
* [ ] Wire handler canonical envelopes; unknown op handled
* [ ] All per-component suites pass unmodified in CI

---

## 16) Environment Profiles (for Stable CI)

* **Local dev:** generous timeouts, minimal parallelism.
* **CI:** `PYTEST_JOBS=auto`, explicit budgets, no network flakiness.
* **Stress (opt-in):** short deadlines, forced backoffs, jitter (`@slow`).

---

## 17) Reproducibility

* Seed randomness; log seed on failure.
* Log adapter version and capabilities snapshot at test start.
* Isolate/clear caches per test.

---

## 18) Deviations & Extensions

* Vendor features may extend behavior **without breaking** canonical outcomes.
* Temporary deviations must be flagged, documented, and tracked to removal.

---

## 19) Pass/Fail Attestation Template

```
✅ Corpus Protocol (v1.0) — Behavioral Conformant
Components: <LLM|Vector|Embedding|Graph>
Commit: <git sha>
CI Run: <link>
Notes: All suites passed unmodified with COV_FAIL_UNDER=<N>.
```

---

## 20) References

* `BEHAVIORAL_CONFORMANCE.md` — semantic contract & pass criteria
* `SCHEMA_CONFORMANCE.md` — schema/wire shape conformance
* `tests/<component>/*.py` — **authoritative** tests
* `Makefile` — standardized run targets

---
