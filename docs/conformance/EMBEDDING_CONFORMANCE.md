# Embedding Protocol Conformance Test Coverage

**Table of Contents**
- [Overview](#overview)
- [Conformance Summary](#conformance-summary)
- [Test Files](#test-files)
- [Specification Mapping](#specification-mapping)
- [Running Tests](#running-tests)
- [Adapter Compliance Checklist](#adapter-compliance-checklist)
- [Conformance Badge](#conformance-badge)
- [Maintenance](#maintenance)

---

## Overview

This document tracks conformance test coverage for the **Embedding Protocol V1.0** specification as defined in `SPECIFICATION.md §10`. Each test validates normative requirements (MUST/SHOULD) from the specification and the reference implementation in:

This suite constitutes the **official Embedding Protocol V1.0 Reference Conformance Test Suite**. Any implementation (Corpus or third-party) MAY run these tests to verify and publicly claim conformance, provided all referenced tests pass unmodified.

**Protocol Version:** Embedding Protocol V1.0
**Status:** Stable / Production-Ready
**Last Updated:** 2025-01-XX
**Test Location:** `tests/embedding/`

## Conformance Summary

**Overall Coverage: 75/75 tests (100%) ✅**

| Category                 | Tests | Coverage |
| ------------------------ | ----- | -------- |
| Core Operations          | 19/19 | 100% ✅   |
| Capabilities             | 8/8   | 100% ✅   |
| Batch & Partial Failures | 6/6   | 100% ✅   |
| Truncation & Length      | 5/5   | 100% ✅   |
| Normalization Semantics  | 5/5   | 100% ✅   |
| Token Counting           | 6/6   | 100% ✅   |
| Error Handling           | 5/5   | 100% ✅   |
| Deadline Semantics       | 4/4   | 100% ✅   |
| Health Endpoint          | 4/4   | 100% ✅   |
| Observability & Privacy  | 6/6   | 100% ✅   |
| Caching & Idempotency    | 3/3   | 100% ✅   |
| Wire Contract            | 16/16 | 100% ✅   |

> Note: Categories are logical groupings. Individual tests may satisfy multiple normative requirements.

---

## Test Files

### `test_capabilities_shape.py`

**Specification:** §6.2, §10.2, §10.5
**Status:** ✅ Complete (8 tests)

Validates EmbeddingCapabilities contract and alignment with `EmbeddingProtocolV1`:

* `test_capabilities_returns_correct_type` — Ensures `capabilities()` returns an `EmbeddingCapabilities` instance.
* `test_capabilities_identity_fields` — Verifies `server` and `version` are present and non-empty.
* `test_capabilities_supported_models_non_empty_tuple` — Asserts `supported_models` is a non-empty tuple of non-empty strings.
* `test_capabilities_resource_limits_valid` — Checks `max_batch_size`, `max_text_length`, `max_dimensions` are positive when set.
* `test_capabilities_feature_flags_boolean` — Ensures key capability flags are present and strictly boolean.
* `test_capabilities_truncation_mode_valid` — Confirms `truncation_mode` is one of the allowed values.
* `test_capabilities_max_dimensions_consistent_with_models` — Sanity-checks `max_dimensions` consistency.
* `test_capabilities_idempotent` — Ensures repeated `capabilities()` calls return stable, identical results.

---

### `test_embed_basic.py`

**Specification:** §10.3, §10.6
**Status:** ✅ Complete (7 tests)

Core unary behavior:

* `test_embed_returns_embed_result_and_vector` — Valid embed returns an embedding with populated vector, dimensions, and model.
* `test_embed_requires_non_empty_text` — Empty `text` is rejected with `BadRequest`.
* `test_embed_requires_non_empty_model` — Empty `model` is rejected with `BadRequest`.
* `test_embed_unknown_model_raises_model_not_available` — Unsupported model raises `ModelNotAvailable`.
* `test_embed_truncates_when_allowed` — Over-length input with `truncate=True` is truncated and succeeds.
* `test_embed_rejects_when_truncate_false_and_too_long` — Over-length input with `truncate=False` raises `TextTooLong`.
* `test_embed_normalize_flag_produces_unit_vector` — `normalize=True` yields approximately unit-norm embeddings when supported.

---

### `test_embed_batch_basic.py`

**Specification:** §10.3, §10.6, §12.5
**Status:** ✅ Complete (6 tests)

Batch behavior and partial failures:

* `test_embed_batch_returns_batch_result` — Valid batch returns `BatchEmbedResult` with embeddings aligned to inputs.
* `test_embed_batch_requires_non_empty_model` — Empty `model` for batch raises `BadRequest`.
* `test_embed_batch_requires_non_empty_texts` — Empty `texts` list raises `BadRequest`.
* `test_embed_batch_respects_max_batch_size` — Exceeding `max_batch_size` raises `BadRequest`.
* `test_embed_batch_unknown_model_raises_model_not_available` — Unknown model in batch raises `ModelNotAvailable`.
* `test_embed_batch_partial_failure_reporting_on_fallback` — Fallback path reports per-item failures with index + error metadata.

---

### `test_truncation_and_text_length.py`

**Specification:** §10.2, §10.6, §12.1
**Status:** ✅ Complete (5 tests)

Truncation + `max_text_length` semantics:

* `test_embed_truncates_when_allowed_and_sets_flag` — Over-limit input with `truncate=True` is truncated to `max_text_length` and sets `truncated=True`.
* `test_embed_raises_when_truncation_disallowed` — Over-limit input with `truncate=False` raises `TextTooLong`.
* `test_batch_truncates_all_when_allowed` — Batch with long texts and `truncate=True` truncates each item consistently.
* `test_batch_oversize_without_truncation_raises` — Batch with over-limit text and `truncate=False` raises `TextTooLong`.
* `test_short_texts_unchanged` — Inputs under the limit pass through unchanged with `truncated=False`.

---

### `test_normalization_semantics.py`

**Specification:** §10.2, §10.5, §10.6
**Status:** ✅ Complete (5 tests)

Normalization behavior:

* `test_single_embed_normalize_true_produces_unit_vector` — `normalize=True` produces unit-length vector when supported.
* `test_single_embed_normalize_false_not_forced_unit_norm` — `normalize=False` does not artificially enforce unit norm.
* `test_batch_embed_normalize_true_all_unit_vectors` — Batch with `normalize=True` yields unit-norm vectors per item.
* `test_normalization_not_supported_raises` — If `supports_normalization=False`, using `normalize=True` raises `NotSupported`.
* `test_normalizes_at_source_respected_no_double_normalization` — `normalizes_at_source=True` is honored without extra normalization in base.

---

### `test_count_tokens_behavior.py`

**Specification:** §10.3, §10.5
**Status:** ✅ Complete (6 tests)

Token counting behavior:

* `test_count_tokens_returns_non_negative_int` — `count_tokens` returns a non-negative integer.
* `test_count_tokens_monotonic_with_respect_to_length` — Longer input never yields fewer tokens than shorter input.
* `test_count_tokens_empty_string_zero_or_minimal` — Empty string returns 0 or minimal overhead.
* `test_count_tokens_unicode_safe` — Unicode input is handled safely with non-negative counts.
* `test_count_tokens_unknown_model_raises_model_not_available` — Unknown model raises `ModelNotAvailable`.
* `test_count_tokens_not_supported_raises_not_supported` — If capabilities disable token counting, `count_tokens` raises `NotSupported`.

---

### `test_error_mapping_retryable.py`

**Specification:** §6.3, §10.4, §12.1–§12.5
**Status:** ✅ Complete (5 tests)

Embedding-specific error taxonomy:

* `test_text_too_long_maps_correctly` — `TextTooLong` maps to `TEXT_TOO_LONG` and is non-retryable.
* `test_model_not_available_maps_correctly` — Unsupported models map to `MODEL_NOT_AVAILABLE` without retry hints.
* `test_retryable_errors_have_retry_after_ms` — `ResourceExhausted`, `Unavailable`, `TransientNetwork` expose well-formed `retry_after_ms`.
* `test_deadline_exceeded_maps_correctly` — Pre-expired deadlines raise `DeadlineExceeded` with `DEADLINE_EXCEEDED` code.
* `test_partial_failure_codes_in_failures` — Batch failures emit normalized `code`/`error` entries in `details.failures`.

---

### `test_deadline_enforcement.py`

**Specification:** §6.1, §10.3, §10.6, §12.1
**Status:** ✅ Complete (4 tests)

Deadline behavior via `OperationContext`:

* `test_deadline_budget_nonnegative` — Remaining budget helper never reports negative values.
* `test_preexpired_deadline_fails_fast_embed` — Pre-expired deadlines cause immediate `DeadlineExceeded` before backend work.
* `test_embed_respects_deadline` — Short deadlines on `embed()` are honored and may raise `DeadlineExceeded`.
* `test_embed_batch_respects_deadline` — Short deadlines on `embed_batch()` are honored and may raise `DeadlineExceeded`.

---

### `test_health_report.py`

**Specification:** §10.3, §6.2
**Status:** ✅ Complete (4 tests)

Health endpoint contract:

* `test_health_shape` — Health returns required structural fields (e.g. `ok`, `server`, `version`).
* `test_health_ok_flags_boolean` — `ok` is strictly boolean.
* `test_health_models_mapping` — Health exposes a models map aligned with supported models.
* `test_health_consistent_on_error` — Health response shape remains stable even when backend is degraded.

---

### `test_context_siem.py`

**Specification:** §6.4, §13, §15 (Embedding)
**Status:** ✅ Complete (6 tests)

SIEM-safe metrics and context propagation:

* `test_context_propagates_to_metrics_siem_safe` — `observe` called with `component="embedding"` and correct op for successful calls.
* `test_tenant_hashed_never_raw` — Tenant identifiers appear only in hashed/derived form; raw tenant is never logged.
* `test_no_text_in_metrics` — No raw text/texts/vectors/embeddings appear in metrics extras.
* `test_metrics_emitted_on_error_path` — Error paths still emit observations and error counters.
* `test_batch_metrics_include_batch_size` — Batch operations record batch size metadata in metrics.
* `test_deadline_bucket_tagged_when_present` — `deadline_bucket`/deadline tags are emitted when `deadline_ms` is set.

---

### `test_cache_and_batch_fallback.py`

**Specification:** §10.3, §10.6, §12.5
**Status:** ✅ Complete (3 tests)

Caching & batch fallback behavior:

* `test_embed_cache_respected_in_standalone_mode` — Identical embed calls hit cache instead of duplicating backend work.
* `test_embed_cache_respects_tenant_isolation` — Cache keys are tenant-aware; no cross-tenant reuse.
* `test_embed_batch_fallback_uses_per_item_and_reports_failures` — When batch is unsupported, per-item fallback works and reports failures.

---

### `test_wire_handler.py`

**Specification:** §4.1, §4.1.6, §10.3, §10.6
**Status:** ✅ Complete (16 tests)

`WireEmbeddingHandler` canonical envelopes:

* `test_capabilities_envelope_success` — `embedding.capabilities` envelope returns `ok:true`, `code:"OK"`, and `result`.
* `test_embed_envelope_success` — `embedding.embed` envelope returns `ok:true` and embedding result in canonical shape.
* `test_embed_batch_envelope_success` — `embedding.embed_batch` envelope returns `ok:true` with correct `embeddings` list.
* `test_count_tokens_envelope_success` — `embedding.count_tokens` envelope returns `ok:true` with integer `result`.
* `test_health_envelope_success` — `embedding.health` envelope returns `ok:true` with proper `result`.
* `test_missing_op_rejected_with_bad_request` — Missing `op` yields normalized `BAD_REQUEST`/`NOT_SUPPORTED`.
* `test_unknown_op_rejected_with_not_supported` — Unknown `op` yields `NOT_SUPPORTED` envelope.
* `test_embed_missing_required_fields_yields_bad_request` — Missing `text` or `model` yields `BAD_REQUEST`.
* `test_embed_unknown_model_maps_model_not_available` — Unknown model maps to `MODEL_NOT_AVAILABLE`/`NOT_SUPPORTED`.
* `test_embed_batch_missing_texts_yields_bad_request` — Missing `texts` yields `BAD_REQUEST`.
* `test_embed_batch_unknown_model_maps_model_not_available` — Unknown model in batch maps correctly.
* `test_count_tokens_unknown_model_maps_model_not_available` — Unknown model in `count_tokens` maps correctly.
* `test_error_envelope_includes_message_and_type` — Adapter `BadRequest` surfaces canonical error envelope with message.
* `test_text_too_long_maps_to_text_too_long_code_when_exposed` — Adapter `TextTooLong` maps to `TEXT_TOO_LONG`/`BAD_REQUEST`.
* `test_embed_context_roundtrip_and_context_plumbing` — Verifies `OperationContext` fields are correctly constructed and passed into the adapter.
* `test_unexpected_exception_maps_to_unavailable` — Verifies unexpected exceptions are normalized to `UNAVAILABLE` envelopes.

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

| Requirement                    | Test File                          | Status |
| ------------------------------ | ---------------------------------- | ------ |
| Validates texts non-empty      | test_embed_batch_basic.py          | ✅      |
| Enforces max_batch_size        | test_embed_batch_basic.py          | ✅      |
| Per-item validation & failures | test_embed_batch_basic.py          | ✅      |
| Truncation semantics           | test_truncation_and_text_length.py | ✅      |
| Normalization semantics        | test_normalization_semantics.py    | ✅      |

#### `count_tokens()`

| Requirement                     | Test File                     | Status |
| ------------------------------- | ----------------------------- | ------ |
| Non-negative int                | test_count_tokens_behavior.py | ✅      |
| Monotonic behavior              | test_count_tokens_behavior.py | ✅      |
| Model must be supported / gated | test_count_tokens_behavior.py | ✅      |

#### `health()`

| Requirement                      | Test File             | Status |
| -------------------------------- | --------------------- | ------ |
| Returns ok/server/version/models | test_health_report.py | ✅      |
| Stable shape even when degraded  | test_health_report.py | ✅      |

### §10.4 Errors — Complete Coverage

| Error Type        | Test File                                                                           | Status |
| ----------------- | ----------------------------------------------------------------------------------- | ------ |
| TextTooLong       | test_truncation_and_text_length.py, test_error_mapping_retryable.py                 | ✅      |
| ModelNotAvailable | test_embed_basic.py, test_count_tokens_behavior.py, test_error_mapping_retryable.py | ✅      |
| DeadlineExceeded  | test_deadline_enforcement.py                                                        | ✅      |
| Retryable vs non- | test_error_mapping_retryable.py                                                     | ✅      |
| Partial failures  | test_embed_batch_basic.py                                                           | ✅      |

### §10.5 Capabilities — Complete Coverage

All required capability fields, normalization/truncation flags, token counting, multi-tenant and deadline support are asserted in:

* `test_capabilities_shape.py`
* Cross-checked implicitly via behavior tests above.

### §10.6 Semantics — Complete Coverage

* Truncation rules: `test_truncation_and_text_length.py`
* Normalization rules: `test_normalization_semantics.py`
* Partial-failure encoding: `test_embed_batch_basic.py`
* Deadline behavior: `test_deadline_enforcement.py`
* Observability & privacy: `test_context_siem.py`
* Caching semantics: `test_cache_and_batch_fallback.py`
* Wire contract: `test_wire_handler.py`

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
       tests/embedding/test_embed_batch_basic.py -v

# Truncation & normalization & tokens
pytest tests/embedding/test_truncation_and_text_length.py \
       tests/embedding/test_normalization_semantics.py \
       tests/embedding/test_count_tokens_behavior.py -v

# Error handling & deadlines & health
pytest tests/embedding/test_error_mapping_retryable.py \
       tests/embedding/test_deadline_enforcement.py \
       tests/embedding/test_health_report.py -v

# Observability, caching, wire contract
pytest tests/embedding/test_context_siem.py \
       tests/embedding/test_cache_and_batch_fallback.py \
       tests/embedding/test_wire_handler.py -v
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
   75/75 tests passing (12 test files)

   ✅ Core Operations: 19/19 (100%)
   ✅ Capabilities: 8/8 (100%)
   ✅ Batch & Partial Failures: 6/6 (100%)
   ✅ Truncation & Length: 5/5 (100%)
   ✅ Normalization: 5/5 (100%)
   ✅ Token Counting: 6/6 (100%)
   ✅ Error Handling: 5/5 (100%)
   ✅ Deadline: 4/4 (100%)
   ✅ Health: 4/4 (100%)
   ✅ Observability & Privacy: 6/6 (100%)
   ✅ Caching & Idempotency: 3/3 (100%)
   ✅ Wire Contract: 16/16 (100%)

   Status: Production Ready
```

---

## Maintenance

### Adding New Tests

* `../../SPECIFICATION.md` - Full protocol specification (§10 Vector Protocol)
* `../../ERRORS.md` - Error taxonomy reference
* `../../METRICS.md` - Observability guidelines
* `../README.md` - General testing guidelines

### Updating for Specification Changes

1. Review `SPECIFICATION.md` Appendix F for Embedding-related changes.
2. Add or adjust tests for any new normative behavior.
3. Bump documented protocol version if required.
4. Update the conformance badge and checklist to match.

---

**Last Updated:** 2025-01-XX
**Maintained By:** Corpus SDK Team
**Status:** 100% V1.0 Conformant - Production Ready