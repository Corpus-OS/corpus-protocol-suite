I’m on it—here’s the **updated, full `ADAPTER_GUIDE.md`** you asked for. It’s focused on **implementing the adapter**, aligned with your Makefile, and merges in the concrete error-mapping example, diagnostic flow, precise Adapter API ops/semantics, and all cross-cutting guarantees. No test lists duplicated.

---

# ADAPTER_GUIDE.md

> ✅ **Corpus Protocol (v1.0) — Adapter Implementation Guide**
> Scope: production-grade **runtime behavior** for adapters (**LLM • Vector • Embedding • Graph**)
> Out of scope: JSON Schema shape (see `SCHEMA_CONFORMANCE.md`)

---

## 1) Purpose & Audience

This guide explains *how to implement* a Corpus-compatible adapter that **passes behavioral conformance**. It turns the normative requirements from `BEHAVIORAL_CONFORMANCE.md` into practical implementation steps, patterns, and drop-in snippets.

**Read alongside:**

* `BEHAVIORAL_CONFORMANCE.md` — semantic contract & pass criteria
* `SCHEMA_CONFORMANCE.md` — schema/wire shape conformance
* `Makefile` — standardized run targets and CI alignment

---

## 2) Runbook (Makefile-Aligned)

**All components (with coverage):**

```bash
make test-conformance
# or:
make verify
```

**Per component:**

```bash
make test-llm-conformance
make test-vector-conformance
make test-graph-conformance
make test-embedding-conformance
```

**Fast lanes (skip @slow, no coverage):**

```bash
make test-fast
make test-fast-llm
make test-fast-vector
make test-fast-graph
make test-fast-embedding
```

**Schema smoke (meta-lint + goldens):**

```bash
make verify-schema
```

**Safety & smoke:**

```bash
make validate-env
make safety-check
make quick-check
make conformance-report
```

Env overrides: `PYTEST_JOBS=auto`, `PYTEST_ARGS="-x --tb=short"`, `COV_FAIL_UNDER=90`, etc.

---

## 3) Recommended Adapter Structure (Single-File or Package)

If you keep a **single module**, separate concerns clearly with sections (same order if possible):

1. **Imports & Header**
2. **Config & Limits** (batch size, context/window, dims, timeouts/TTLs, feature flags)
3. **OperationContext & Utilities**

   * `deadline_remaining_ms()`, `now_ms()`
   * `hash_tenant()`, `content_hash()` (never store raw tenant/text)
4. **Error Mapping (Provider → Canonical)** *(drop-in below)*
5. **Observability (SIEM-safe)**

   * `observe_once(component, op, tags, status, err=None)`
6. **Provider Facade**

   * thin wrappers for SDK/HTTP with backoff/jitter; *no* Corpus types here
7. **Adapter API** *(LLM/Embedding/Vector/Graph ops; see §7)*
8. **Streaming Guardrails** *(one terminal, pacing/heartbeats)*
9. **Wire Handler** (canonical envelopes; unknown op → `NOT_SUPPORTED`)
10. **Optional Cache Hooks** (tenant-aware, content-addressed)
11. **Test Hints** (docstrings link to relevant tests)

---

## 4) Canonical Error Taxonomy & Mapping (Drop-In)

**Goals**

* Normalize provider errors into *canonical codes*
* Attach **retry hints** (`retry_after_ms`) for retryable classes
* Emit **`validation_errors`** for contract problems
* Never leak provider internals

```python
# Example error mapping
PROVIDER_TO_CANONICAL = {
    "rate_limit_exceeded": ("RESOURCE_EXHAUSTED", {"retry_after_ms": 5000}),
    "invalid_api_key": ("AUTHENTICATION_ERROR", None),
    "context_length_exceeded": ("BAD_REQUEST", {"validation_errors": [...]})
}

def map_error(provider_error) -> tuple[str, dict | None, bool]:
    """
    Returns (canonical_code, details_or_none, retryable_bool).
    Deterministic: the same provider error must always map to the same tuple.
    """
    code_key = getattr(provider_error, "code", None) or str(getattr(provider_error, "status", "")) or "unknown"
    code, details = PROVIDER_TO_CANONICAL.get(code_key, ("INTERNAL", None))
    retryable = code in {"RESOURCE_EXHAUSTED", "UNAVAILABLE", "TRANSIENT_NETWORK"}
    return code, details, retryable
```

**Rules**

* **Validation** (`BAD_REQUEST`, `TEXT_TOO_LONG`, `MODEL_NOT_AVAILABLE`) → **non-retryable**; include `validation_errors` when relevant
* **Resource/Transport** (`RESOURCE_EXHAUSTED`, `UNAVAILABLE`, `TRANSIENT_NETWORK`) → **retryable**; include `retry_after_ms` if known
* **Deadline** (`DEADLINE_EXCEEDED`) is produced *locally* when budget elapses

---

## 5) Deadlines & Budgets

* **Preflight**: if `deadline_ms <= now`, fail fast (no backend call) with `DEADLINE_EXCEEDED`
* **Propagation**: pass remaining budget to every provider call; **never negative**
* **Mid-op**: abort promptly if budget elapses; surface canonical error
* **Observability**: when deadline present, emit `deadline_bucket` tag

---

## 6) Streaming State Machine (LLM, Graph)

**Rules**

* Exactly **one terminal** frame (`end` or `error`) per stream
* **No data after terminal**; mid-stream `error` is terminal
* Heartbeats allowed but must respect pacing/backpressure limits

**State**

```
START -> DATA* -> (END | ERROR) -> STOP
            ^           |
            |-----------|
```

---

## 7) Adapter API (Operations & Behavioral Guarantees)

Canonical operation names per component (signatures illustrative). Behaviors below align with the conformance suites.

### 7.1 LLM

| Operation                    | Purpose                | Required Behavior (non-exhaustive)                                                                                                                            |
| ---------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `complete(ctx, request)`     | Generate text          | Enforce deadlines; stream or unary per request; canonical error mapping; one observation; respect sampling params; final usage equals sum of streamed deltas. |
| `count_tokens(ctx, request)` | Tokenize text          | Monotonic counts; unicode-safe; model-gated; deadline-aware; returns non-negative int.                                                                        |
| `capabilities(ctx)`          | Report features/limits | Truthful, stable snapshot; includes server/version/models, max context, supported features.                                                                   |
| `health(ctx)`                | Liveness & readiness   | Stable shape when degraded; include reason/status; no exceptions on partial outages.                                                                          |

**Illustrative shapes (behavior focus):**

```python
# request: { model, messages|prompt, sampling?, response_format?, tools? }
# response (unary): { text, usage:{prompt_tokens, completion_tokens, total_tokens}, warnings? }
# response (stream): data frames then exactly one terminal (end|error)
```

---

### 7.2 Embedding

| Operation                    | Purpose               | Required Behavior (non-exhaustive)                                                                                            |
| ---------------------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `embed(ctx, request)`        | Single text embedding | `truncate=true` caps to max_text_length and sets `truncated`; `normalize=true` → ~unit-norm if supported; enforce dimensions. |
| `embed_batch(ctx, request)`  | Batch embeddings      | Respect `max_batch_size`; preserve input order; per-index failures with metadata; no silent drops.                            |
| `count_tokens(ctx, request)` | Token counting        | Monotonic; unicode-safe; model-gated; deadline-aware.                                                                         |
| `capabilities(ctx)`          | Features/limits       | Truthful flags/limits (normalization, truncation, max_text_length, max_dimensions, batch).                                    |
| `health(ctx)`                | Liveness/readiness    | Stable on degrade; reason/status included.                                                                                    |

**Notes:** No NaN/Inf; dimensions consistent per model; cache keys tenant-aware & content-addressed (no raw text).

---

### 7.3 Vector

| Operation                           | Purpose                    | Required Behavior (non-exhaustive)                                                                            |
| ----------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `upsert(ctx, request)`              | Insert/update vectors      | Enforce dimension/namespace constraints; idempotent per id; preserve batch order; per-item failures reported. |
| `query(ctx, request)`               | Similarity/filtered search | Respect filters; matches have consistent dims; stable scoring semantics; deadline-aware.                      |
| `delete(ctx, request)`              | Remove vectors             | Idempotent deletes; side-effects confined to request scope.                                                   |
| `namespace_create(ctx, request)`    | Create namespace           | Safe to re-create; acknowledge creation.                                                                      |
| `namespace_delete(ctx, request)`    | Delete namespace           | Idempotent; no residuals after success.                                                                       |
| `capabilities(ctx)` / `health(ctx)` | Features & liveness        | Truthful limits (max dims, batch sizes, filter dialect); stable health on partial backend issues.             |

---

### 7.4 Graph

| Operation                                     | Purpose               | Required Behavior (non-exhaustive)                                                                                 |
| --------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `query(ctx, request)`                         | Query graph (unary)   | Deterministic row/entity shapes; enforce limits; canonical warnings.                                               |
| `stream_query(ctx, request)`                  | Query with streaming  | **Exactly one terminal** (`end` or `error`); **no data after terminal**; optional heartbeats within pacing limits. |
| `batch(ctx, request)`                         | Mixed vertex/edge ops | Partial-success accounting correct; side-effects reflect successes only; preserve item order and ids/acks.         |
| `vertex_create / vertex_delete / edge_create` | CRUD primitives       | Idempotent where defined; validation errors non-retryable; consistent ids.                                         |
| `id(ctx, request)` / `ack(ctx, request)`      | ID lookup / Ack       | Stable guarantees around visibility/ack semantics.                                                                 |
| `capabilities(ctx)` / `health(ctx)`           | Features & liveness   | Truthful dialect/limits; health includes reason on degrade.                                                        |

---

### 7.5 Cross-Cutting Guarantees (apply to all ops)

* **Deadlines:** fast-fail pre-expired; propagate remaining budget; never negative; emit `deadline_bucket` when present
* **Errors:** deterministic provider→canonical mapping; retryables include `retry_after_ms`; validation errors are non-retryable with `validation_errors`
* **Streaming:** one terminal; no data after terminal; mid-stream error is terminal
* **Observability/SIEM:** single observation per op; no raw text/vectors/tenant IDs; use `tenant_hash`
* **Limits:** enforce declared batch/context/dim/namespace limits
* **Idempotency & Caching (when applicable):** tenant-aware, content-addressed keys; no cross-tenant bleed; stable batch fallback

---

## 8) Caching & Key Derivation (Where Applicable)

**Key derivation example**

```
key = hash(tenant_hash, op, model, normalize, params, sha256(text|payload))
```

* No raw text or plain tenant IDs in keys
* Partition caches per tenant/environment where possible
* Clear caches between tests or scope per test to avoid leakage

---

## 9) Wire Handler Rules

* Canonical envelopes in/out; unknown fields ignored safely
* **Unknown op** → `NOT_SUPPORTED`
* `schema_version` must be present on success envelopes (SemVer)

---

## 10) Minimal Adapter Stubs (Illustrative Only)

Keep real code in your adapter module; this is a sketch of per-op flow:

```python
def embed(ctx, req):
    # 1) Deadline preflight
    if ctx.deadline_expired():
        return err("DEADLINE_EXCEEDED")

    try:
        # 2) Validate model/support
        ensure_supported_model(req.model)

        # 3) Truncation/normalization semantics
        text = maybe_truncate(req.text, caps.max_text_length, req.truncate)
        vec  = provider.embed(model=req.model, text=text, timeout=ctx.remaining_ms())

        if req.normalize and caps.supports_normalization:
            vec = normalize(vec)

        # 4) Observability (success)
        observe_once("embedding", "embed", tags(ctx, req), status="OK")

        # 5) Return canonical success
        return ok_embedding(vec, truncated=was_truncated(text, req.text))

    except ProviderError as pe:
        code, details, retryable = map_error(pe)
        observe_once("embedding", "embed", tags(ctx, req), status=code, err=str(pe))
        return error_envelope(code, details, retryable)
```

---

## 11) When Tests Fail: Diagnostic Flow

**Schema failures?** → Run `make verify-schema`
**Behavioral failures?** → Open the specific test from `BEHAVIORAL_CONFORMANCE.md §5`
**Streaming issues?** → Review **§6 Streaming State Machine**
**Error mapping wrong?** → Review **§4 Error Taxonomy & Mapping**

---

## 12) Adapter Readiness Checklist

* [ ] Deadline preflight + budget propagation; never negative
* [ ] Streaming: one terminal; no data after terminal; heartbeats within limits
* [ ] Error mapping deterministic; retry hints (`retry_after_ms`) where applicable
* [ ] Embedding truncation/normalization semantics correct
* [ ] Token counting monotonic & model-gated
* [ ] Cache keys tenant-aware; no raw text; no cross-tenant hits
* [ ] Limits enforced (dims, batch, context, namespaces)
* [ ] Observability: single observe/op; SIEM-safe; required tags
* [ ] Wire handler canonical envelopes; unknown op handled
* [ ] All per-component suites pass unmodified in CI

---

## 13) Environment Profiles (Stable CI)

* **Local dev:** generous timeouts, low parallelism
* **CI:** `PYTEST_JOBS=auto`, explicit budgets, no external dependencies
* **Stress (opt-in):** short deadlines, forced backoffs, jitter (`@slow`)

---

## 14) Reproducibility

* Seed randomness and log the seed on failure
* Log adapter version + capabilities snapshot at test start
* Isolate or clear caches between tests

---

## 15) Deviations & Extensions

* Vendor features may extend behavior **without breaking** canonical outcomes
* Temporary deviations must be **flagged**, **documented**, and tracked to removal with a target release

---

## 16) Conformance Attestation (Template)

```
✅ Corpus Protocol (v1.0) — Behavioral Conformant
Components: <LLM|Vector|Embedding|Graph>
Commit: <git sha>
CI Run: <link>
Notes: All suites passed unmodified with COV_FAIL_UNDER=<N>.
```

---

## 17) References

* `BEHAVIORAL_CONFORMANCE.md` — semantic contract & pass criteria
* `SCHEMA_CONFORMANCE.md` — schema/wire shape conformance
* `tests/<component>/*.py` — **authoritative** behavioral tests
* `Makefile` — standardized run targets and safety checks

---

**Maintainers:** Corpus SDK Team
**Last Updated:** 2025-11-12
**Scope:** Adapter runtime behavior; see `SCHEMA_CONFORMANCE.md` for schema shape
