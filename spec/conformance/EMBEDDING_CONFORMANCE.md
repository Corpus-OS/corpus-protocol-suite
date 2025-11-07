Here’s the fixed **Embedding Protocol V1** conformance doc — aligned with Graph/LLM/Vector, using your actual base (`adapter_sdk/embedding_base.py`), mock adapter, wire handler, phases, maintenance, and with the correct test count (**54**). No extra fluff.

---

# Embedding Protocol V1 Conformance Test Coverage

## Overview

This document tracks conformance test coverage for the **Embedding Protocol V1.0** specification as defined in `SPECIFICATION.md §10`. Each test validates normative requirements (MUST/SHOULD) from the specification and the reference implementation in:

* `adapter_sdk/embedding_base.py`
* `examples/embedding/mock_embedding_adapter.py`

This suite constitutes the **official Embedding Protocol V1.0 Reference Conformance Test Suite**. Any implementation (Corpus or third-party) MAY run these tests to verify and publicly claim conformance, provided all referenced tests pass unmodified.

**Protocol Version:** Embedding Protocol V1.0
**Status:** Stable / Production-Ready
**Last Updated:** 2025-01-XX
**Test Location:** `tests/embedding/`

## Conformance Summary

**Overall Coverage: 54/54 tests (100%) ✅**

| Category                 | Tests | Coverage |
| ------------------------ | ----- | -------- |
| Core Operations          | 3/3   | 100% ✅   |
| Capabilities             | 7/7   | 100% ✅   |
| Batch & Partial Failures | 5/5   | 100% ✅   |
| Truncation & Length      | 5/5   | 100% ✅   |
| Normalization Semantics  | 5/5   | 100% ✅   |
| Token Counting           | 3/3   | 100% ✅   |
| Error Handling           | 5/5   | 100% ✅   |
| Deadline Semantics       | 4/4   | 100% ✅   |
| Health Endpoint          | 4/4   | 100% ✅   |
| Observability & Privacy  | 6/6   | 100% ✅   |
| Caching & Idempotency    | 3/3   | 100% ✅   |
| Wire Contract            | 4/4   | 100% ✅   |

---

## Test Files

### `test_capabilities_shape.py`

**Specification:** §6.2, §10.2, §10.5
**Status:** ✅ Complete (7 tests)

Validates EmbeddingCapabilities contract and alignment with `EmbeddingProtocolV1`:

* `test_capabilities_returns_correct_type` — Returns `EmbeddingCapabilities`.
* `test_capabilities_identity_fields` — `server`, `version` non-empty strings.
* `test_capabilities_supported_models_non_empty` — `supported_models` non-empty list.
* `test_capabilities_resource_limits_valid` — `max_batch_size`, `max_text_length`, `max_dimensions` are positive ints or `None`.
* `test_capabilities_boolean_flags` — All boolean flags set and of correct type (`supports_normalization`, `supports_truncation`, etc.).
* `test_capabilities_protocol_alignment` — Matches `embedding/v1.0` semantics from spec.
* `test_capabilities_idempotent_and_multi_tenant_flags` — `idempotent_operations` / `supports_multi_tenant` coherent with base.

---

### `test_embed_basic.py`

**Specification:** §10.3, §10.6
**Status:** ✅ Complete (3 tests)

Core unary behavior:

* `test_embed_returns_embedding_result` — Returns embedding with `EmbeddingVector`, correct `dimensions`, `model`.
* `test_embed_rejects_empty_text_or_model` — Empty `text` or `model` → `BadRequest`.
* `test_embed_requires_supported_model` — Unknown `model` → `ModelNotAvailable`.

---

### `test_embed_batch_partial_failures.py`

**Specification:** §10.3, §10.6, §12.5
**Status:** ✅ Complete (5 tests)

Batch + partial failure semantics:

* `test_embed_batch_success_shape` — Returns `EmbeddingResult`-compatible shape (`embeddings`, `model`, optional `failures`).
* `test_embed_batch_enforces_max_batch_size` — Exceeds `max_batch_size` → `BadRequest` with limit detail.
* `test_embed_batch_per_item_validation` — Bad items reported individually.
* `test_embed_batch_partial_success_failures_indexed` — `failures[index]` matches input positions.
* `test_embed_batch_does_not_drop_failures_silently` — No silent drops; either embedding or recorded failure for each input.

---

### `test_truncation_and_text_length.py`

**Specification:** §10.2, §10.6, §12.1
**Status:** ✅ Complete (5 tests)

Truncation + `max_text_length` semantics wired to `BaseEmbeddingAdapter`:

* `test_truncate_true_shortens_but_succeeds` — Over-limit with `truncate=True` → truncated embedding, no `TextTooLong`.
* `test_truncate_false_raises_text_too_long` — Over-limit with `truncate=False` → `TextTooLong`.
* `test_truncation_respects_capabilities_max_text_length` — Uses `caps.max_text_length` as source of truth.
* `test_truncation_flag_exposed_in_result` — `EmbedResult`/`BatchEmbedResult` reflects truncation.
* `test_no_text_leak_in_truncation_errors` — `TextTooLong.details` SIEM-safe, no full text.

---

### `test_normalization_semantics.py`

**Specification:** §10.2, §10.5, §10.6
**Status:** ✅ Complete (5 tests)

Normalization behavior consistent with `supports_normalization` and `normalizes_at_source`:

* `test_normalize_true_produces_unit_norm` — When supported, `normalize=True` → L2 ≈ 1.0.
* `test_normalize_false_no_forced_unit_norm` — `normalize=False` → no forced normalization.
* `test_normalization_not_supported_raises` — `normalize=True` when `supports_normalization=False` → `NotSupported`.
* `test_normalizes_at_source_flag_respected` — If `normalizes_at_source=True`, vectors already unit norm; base does not double-normalize.
* `test_batch_normalization_consistent` — `embed_batch` normalization semantics match single `embed`.

---

### `test_count_tokens_consistency.py`

**Specification:** §10.3, §10.5
**Status:** ✅ Complete (3 tests)

Token counting behavior:

* `test_count_tokens_monotonic` — Longer text → token count ≥ shorter text.
* `test_count_tokens_requires_supported_model` — Unknown model → `ModelNotAvailable`.
* `test_count_tokens_not_supported_flag` — If `supports_token_counting=False` → `NotSupported`.

---

### `test_error_mapping_retryable.py`

**Specification:** §6.3, §10.4, §12.1–§12.5
**Status:** ✅ Complete (5 tests)

Embedding-specific error taxonomy:

* `test_text_too_long_maps_correctly` — `TextTooLong` → `TEXT_TOO_LONG`, non-retryable.
* `test_model_not_available_maps_correctly` — → `MODEL_NOT_AVAILABLE`.
* `test_retryable_errors_have_retry_after_ms` — `ResourceExhausted`, `Unavailable`, `TransientNetwork` expose hints.
* `test_deadline_exceeded_maps_correctly` — `DeadlineExceeded` → `DEADLINE_EXCEEDED`, conditional retry semantics.
* `test_partial_failure_codes_in_failures` — Batch failures use normalized `error`/`code`.

---

### `test_deadline_enforcement.py`

**Specification:** §6.1, §10.3, §10.6, §12.1
**Status:** ✅ Complete (4 tests)

Deadline behavior via `OperationContext` + `BaseEmbeddingAdapter`:

* `test_deadline_budget_nonnegative` — Remaining budget helper never negative.
* `test_preexpired_deadline_fails_fast_embed` — Pre-expired → `DeadlineExceeded` before backend work.
* `test_embed_respects_deadline` — Short deadline may trigger `DeadlineExceeded`.
* `test_embed_batch_respects_deadline` — Batch path also enforces deadlines.

---

### `test_health_report.py`

**Specification:** §10.3, §6.2
**Status:** ✅ Complete (4 tests)

Health endpoint contract:

* `test_health_shape` — Returns `{ok, server, version, models}`.
* `test_health_ok_flags_boolean` — `ok` is bool.
* `test_health_models_mapping` — `models` map includes supported models.
* `test_health_consistent_on_error` — Shape stable even when degraded/unavailable.

---

### `test_context_siem.py`

**Specification:** §6.4, §13, §15 (Embedding)
**Status:** ✅ Complete (6 tests)

SIEM-safe metrics and context propagation using `MetricsSink`:

* `test_context_propagates_to_metrics_siem_safe` — `observe` called with `component="embedding"`, correct `op`.
* `test_tenant_hashed_never_raw` — Only tenant hash appears; raw tenant never logged.
* `test_no_text_in_metrics` — No raw `text`/`texts`/vectors in `extra`.
* `test_metrics_emitted_on_error_path` — Errors still produce `observe` + error counters.
* `test_batch_metrics_include_batch_size` — `batch_size` recorded for batch ops.
* `test_deadline_bucket_tagged_when_present` — `deadline_bucket` emitted when `deadline_ms` set.

---

### `test_caching_and_idempotency.py`

**Specification:** §10.6, §11.6, Base embedding cache behavior
**Status:** ✅ Complete (3 tests)

Cache + idempotency semantics for `BaseEmbeddingAdapter` / `InMemoryTTLCache`:

* `test_embed_cache_hit_for_identical_request` — Same `(tenant, model, text, normalize)` → cache hit observed.
* `test_cache_key_uses_hash_not_raw_text` — Cache keys use SHA-256 digest; no raw text.
* `test_idempotency_does_not_break_semantics` — Repeated calls with same inputs yield consistent outputs.

---

### `test_wire_handler_envelopes.py`

**Specification:** §4.1, §4.1.6, §10.3, §10.6
**Status:** ✅ Complete (4 tests)

`WireEmbeddingHandler` conformance to canonical envelopes:

* `test_handle_embed_success_envelope` — `{op:"embedding.embed",...}` → `ok:true`, `code:"OK"`, `result` shape per `EmbeddingResult`.
* `test_handle_embed_batch_success_envelope` — Proper mapping for `embedding.embed_batch` with `texts`.
* `test_handle_count_tokens_and_health_envelopes` — Correct for `embedding.count_tokens`, `embedding.health`.
* `test_unknown_op_returns_not_supported` — Unknown `op` → `NotSupported` normalized error envelope.

---

## Specification Mapping

### §10.3 Operations — Complete Coverage

#### `capabilities()`

| Requirement                         | Test File                  | Status |
| ----------------------------------- | -------------------------- | ------ |
| Returns EmbeddingCapabilities       | test_capabilities_shape.py | ✅      |
| Declares server/version/models      | test_capabilities_shape.py | ✅      |
| Limits & flags wired per spec       | test_capabilities_shape.py | ✅      |
| Protocol alignment (embedding/v1.0) | test_capabilities_shape.py | ✅      |

#### `embed()`

| Requirement                         | Test File                          | Status |
| ----------------------------------- | ---------------------------------- | ------ |
| Validates non-empty text/model      | test_embed_basic.py                | ✅      |
| Model must be supported             | test_embed_basic.py                | ✅      |
| Respects truncate & max_text_length | test_truncation_and_text_length.py | ✅      |
| Supports optional normalization     | test_normalization_semantics.py    | ✅      |
| Deadline / context honored          | test_deadline_enforcement.py       | ✅      |

#### `embed_batch()`

| Requirement                    | Test File                            | Status |
| ------------------------------ | ------------------------------------ | ------ |
| Validates texts non-empty      | test_embed_batch_partial_failures.py | ✅      |
| Enforces max_batch_size        | test_embed_batch_partial_failures.py | ✅      |
| Per-item validation & failures | test_embed_batch_partial_failures.py | ✅      |
| Truncation semantics           | test_truncation_and_text_length.py   | ✅      |
| Normalization semantics        | test_normalization_semantics.py      | ✅      |

#### `count_tokens()`

| Requirement                     | Test File                        | Status |
| ------------------------------- | -------------------------------- | ------ |
| Non-negative int                | test_count_tokens_consistency.py | ✅      |
| Monotonic behavior              | test_count_tokens_consistency.py | ✅      |
| Model must be supported / gated | test_count_tokens_consistency.py | ✅      |

#### `health()`

| Requirement                      | Test File             | Status |
| -------------------------------- | --------------------- | ------ |
| Returns ok/server/version/models | test_health_report.py | ✅      |
| Stable shape even when degraded  | test_health_report.py | ✅      |

### §10.4 Errors — Complete Coverage

| Error Type        | Test File                                                                              | Status |
| ----------------- | -------------------------------------------------------------------------------------- | ------ |
| TextTooLong       | test_truncation_and_text_length.py, test_error_mapping_retryable.py                    | ✅      |
| ModelNotAvailable | test_embed_basic.py, test_count_tokens_consistency.py, test_error_mapping_retryable.py | ✅      |
| DeadlineExceeded  | test_deadline_enforcement.py                                                           | ✅      |
| Retryable vs non- | test_error_mapping_retryable.py                                                        | ✅      |
| Partial failures  | test_embed_batch_partial_failures.py                                                   | ✅      |

### §10.5 Capabilities — Complete Coverage

All required capability fields, normalization/truncation flags, token counting, multi-tenant and deadline support are asserted in:

* `test_capabilities_shape.py`
* Cross-checked implicitly via behavior tests above.

### §10.6 Semantics — Complete Coverage

* Truncation rules: `test_truncation_and_text_length.py`
* Normalization rules: `test_normalization_semantics.py`
* Partial-failure encoding: `test_embed_batch_partial_failures.py`
* Deadline behavior: `test_deadline_enforcement.py`
* Observability & privacy: `test_context_siem.py`
* Caching semantics: `test_caching_and_idempotency.py`
* Wire contract: `test_wire_handler_envelopes.py`

---

## Running Tests

### All Embedding conformance tests

```bash
pytest tests/embedding/ -v
```

### By category

```bash
# Core operations & capabilities
pytest tests/embedding/test_capabilities_shape.py \
       tests/embedding/test_embed_basic.py \
       tests/embedding/test_embed_batch_partial_failures.py -v

# Truncation & normalization & tokens
pytest tests/embedding/test_truncation_and_text_length.py \
       tests/embedding/test_normalization_semantics.py \
       tests/embedding/test_count_tokens_consistency.py -v

# Error handling & deadlines & health
pytest tests/embedding/test_error_mapping_retryable.py \
       tests/embedding/test_deadline_enforcement.py \
       tests/embedding/test_health_report.py -v

# Observability, caching, wire contract
pytest tests/embedding/test_context_siem.py \
       tests/embedding/test_caching_and_idempotency.py \
       tests/embedding/test_wire_handler_envelopes.py -v
```

### With coverage report

```bash
pytest tests/embedding/ --cov=adapter_sdk.embedding_base --cov-report=html
```

---

## Adapter Compliance Checklist

Use this when implementing or validating a new **Embedding adapter** against `EmbeddingProtocolV1` + `BaseEmbeddingAdapter`.

### ✅ Phase 1: Core Operations (3/3)

* [x] `capabilities()` returns `EmbeddingCapabilities` with required fields.
* [x] `embed()` returns valid embeddings with correct dimensions/model.
* [x] `embed_batch()` and `count_tokens()` implemented or explicitly `NotSupported` per caps.

### ✅ Phase 2: Validation, Truncation & Normalization (10/10)

* [x] Reject empty `text` / `model`.
* [x] Enforce `supported_models`.
* [x] Enforce `max_batch_size` when present.
* [x] Enforce `max_text_length` with `truncate` semantics.
* [x] `TextTooLong` when `truncate=false`.
* [x] `normalize=true` only if `supports_normalization`.
* [x] Respect `normalizes_at_source`.
* [x] Ensure dimensions set correctly on all embeddings.
* [x] `count_tokens` consistent & model-gated.
* [x] No NaN/Inf or invalid vectors.

### ✅ Phase 3: Error Handling & Partial Failures (9/9)

* [x] Map provider errors to normalized codes (`TextTooLong`, `ModelNotAvailable`, etc.).
* [x] Do not treat validation errors as retryable.
* [x] Provide `retry_after_ms` for retryable errors when available.
* [x] Use `EmbeddingResult.failures` / item failures for batch.
* [x] No silent drops in `embed_batch`.
* [x] `DeadlineExceeded` on exhausted budgets.
* [x] Honor `NotSupported` for unsupported features.
* [x] Preserve SIEM-safe `details`.
* [x] Follow §12.5 partial-failure semantics.

### ✅ Phase 4: Observability & Privacy (6/6)

* [x] Use `component="embedding"` in metrics.
* [x] Emit exactly one `observe` per op.
* [x] Never log raw text, embeddings, or tenant IDs.
* [x] Use `tenant_hash`, `deadline_bucket`, `batch_size` as low-cardinality tags.
* [x] Emit error counters on failure.
* [x] Ensure wire+logs SIEM-safe per §13, §15.

### ✅ Phase 5: Deadlines, Caching & Wire Contract (8/8)

* [x] Respect `OperationContext.deadline_ms` with preflight checks.
* [x] Use `DeadlineExceeded` when time budget elapses.
* [x] If caching, key by `(tenant_hash, model, normalize, sha256(text))`, no raw text.
* [x] Idempotent behavior for repeat-identical unary requests.
* [x] `WireEmbeddingHandler` implements `embedding.*` ops with canonical envelopes.
* [x] Unknown fields ignored; unknown ops → `NotSupported`.
* [x] Error envelopes use normalized `code`/`error`.
* [x] Compatible with `{ "protocol": "embedding/v1.0" }` contract.

---

## Conformance Badge

```text
✅ Embedding Protocol V1.0 - 100% Conformant
   54/54 tests passing (12 test files)

   ✅ Core Operations: 3/3 (100%)
   ✅ Capabilities: 7/7 (100%)
   ✅ Batch & Partial Failures: 5/5 (100%)
   ✅ Truncation & Length: 5/5 (100%)
   ✅ Normalization: 5/5 (100%)
   ✅ Token Counting: 3/3 (100%)
   ✅ Error Handling: 5/5 (100%)
   ✅ Deadline: 4/4 (100%)
   ✅ Health: 4/4 (100%)
   ✅ Observability & Privacy: 6/6 (100%)
   ✅ Caching & Idempotency: 3/3 (100%)
   ✅ Wire Contract: 4/4 (100%)

   Status: Production Ready
```

---

## Maintenance

### Adding New Tests

1. Create file: `tests/embedding/test_<feature>_<aspect>.py`.
2. Add SPDX header and spec references (`§10.x`, `§12.x`, `§13.x`, etc.).
3. Use `pytestmark = pytest.mark.asyncio` for async tests.
4. Update this document’s **Conformance Summary**, **Test Files**, and **Mapping**.
5. Keep counts accurate; do not repurpose existing tests without updating references.

### Updating for Specification Changes

1. Review `SPECIFICATION.md` Appendix F for Embedding-related changes.
2. Add or adjust tests for any new normative behavior.
3. Bump documented protocol version if required.
4. Update the conformance badge and checklist to match.

## Related Documentation

* `../../SPECIFICATION.md` — Corpus SDK Specification (§10 Embedding Protocol).
* `corpus_sdk/embedding/embedding_base.py` — Reference base implementation.
* `examples/embedding/mock_embedding_adapter.py` — Mock adapter used in tests.
* `../README.md` — Test harness and contribution guidelines.

---
