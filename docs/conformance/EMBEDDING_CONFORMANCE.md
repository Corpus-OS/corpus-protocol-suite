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
**Last Updated:** 2026-01-19
**Test Location:** `tests/embedding/`

## Conformance Summary

**Overall Coverage: 135/135 tests (100%) ✅**

| Category                 | Tests | Coverage |
| ------------------------ | ----- | -------- |
| Core Operations          | 21/21 | 100% ✅   |
| Capabilities             | 15/15 | 100% ✅   |
| Batch & Partial Failures | 10/10 | 100% ✅   |
| Truncation & Length      | 12/12 | 100% ✅   |
| Normalization Semantics  | 10/10 | 100% ✅   |
| Token Counting           | 10/10 | 100% ✅   |
| Error Handling           | 10/10 | 100% ✅   |
| Deadline Semantics       | 7/7   | 100% ✅   |
| Health Endpoint          | 10/10 | 100% ✅   |
| Observability & Privacy  | 8/8   | 100% ✅   |
| Caching & Idempotency    | 13/13 | 100% ✅   |
| Wire Contract            | 19/19 | 100% ✅   |

> Note: Categories are logical groupings. Individual tests may satisfy multiple normative requirements.

---

## Test Files

### `test_capabilities_shape.py`

**Specification:** §6.2, §10.2, §10.5
**Status:** ✅ Complete (15 tests)

Validates EmbeddingCapabilities contract and alignment with `EmbeddingProtocolV1`:

* `test_capabilities_returns_correct_type` — Ensures `capabilities()` returns an `EmbeddingCapabilities` instance.
* `test_capabilities_identity_fields` — Verifies `server` and `version` are present and non-empty.
* `test_capabilities_supported_models_non_empty_tuple` — Asserts `supported_models` is a non-empty tuple of non-empty strings.
* `test_capabilities_resource_limits_valid` — Checks `max_batch_size`, `max_text_length`, `max_dimensions` are positive when set.
* `test_capabilities_feature_flags_boolean` — Ensures key capability flags are present and strictly boolean.
* `test_capabilities_truncation_mode_valid` — Confirms `truncation_mode` is one of the allowed values.
* `test_capabilities_max_dimensions_consistent_with_models` — Sanity-checks `max_dimensions` consistency.
* `test_capabilities_idempotent` — Ensures repeated `capabilities()` calls return stable, identical results.
* `test_capabilities_serializable_structure` — Verifies JSON serializability.
* `test_capabilities_protocol_version` — Checks protocol version matches expected format.
* `test_capabilities_supported_models_accurate` — Validates models list accuracy.
* `test_capabilities_max_batch_size_respected` — Verifies batch size enforcement.
* `test_capabilities_max_text_length_respected` — Validates text length enforcement.
* `test_capabilities_match_operational_behavior_batch` — Ensures batch behavior matches capabilities.
* `test_capabilities_match_operational_behavior_normalization` — Ensures normalization behavior matches capabilities.
* `test_capabilities_streaming_flag_present` — Validates streaming capability flag.
* `test_capabilities_cache_flag_accurate` — Verifies cache capability accuracy.

---

### `test_embed_basic.py`

**Specification:** §10.3, §10.6
**Status:** ✅ Complete (11 tests)

Core unary behavior:

* `test_core_ops_embed_returns_valid_embedding_structure` — Valid embed returns an embedding with populated vector, dimensions, and model.
* `test_core_ops_embed_requires_valid_text` — Invalid text is rejected with appropriate error.
* `test_core_ops_embed_requires_valid_model` — Invalid model is rejected with appropriate error.
* `test_core_ops_embed_unknown_model_clear_error` — Unsupported model raises `ModelNotAvailable`.
* `test_core_ops_embed_truncation_behavior_matches_capabilities` — Over-length input with `truncate=True` is truncated and succeeds.
* `test_core_ops_embed_normalization_produces_unit_vectors` — `normalize=True` yields approximately unit-norm embeddings when supported.
* `test_core_ops_embed_normalization_unsupported_raises_clear_error` — Using `normalize=True` without support raises `NotSupported`.
* `test_core_ops_embed_vector_quality_and_consistency` — Ensures vector quality and consistency across calls.
* `test_core_ops_embed_special_character_handling` — Validates special character handling.
* `test_core_ops_embed_context_propagation` — Verifies context propagation.
* `test_core_ops_embed_dimensions_consistent_with_capabilities` — Ensures dimensions match capabilities.

---

### `test_embed_batch_basic.py`

**Specification:** §10.3, §10.6, §12.5
**Status:** ✅ Complete (10 tests)

Batch behavior and partial failures:

* `test_batch_partial_returns_batch_result` — Valid batch returns `BatchEmbedResult` with embeddings aligned to inputs.
* `test_batch_partial_requires_non_empty_model` — Empty `model` for batch raises `BadRequest`.
* `test_batch_partial_requires_non_empty_texts` — Empty `texts` list raises `BadRequest`.
* `test_batch_partial_respects_max_batch_size` — Exceeding `max_batch_size` raises `BadRequest`.
* `test_batch_partial_unknown_model_raises_model_not_available` — Unknown model in batch raises `ModelNotAvailable`.
* `test_batch_partial_partial_failure_reporting` — Partial failures are properly reported with indices and error metadata.
* `test_batch_partial_single_item_works` — Single-item batch works correctly.
* `test_batch_partial_ordering_preserved` — Output ordering matches input ordering.
* `test_batch_partial_empty_strings_handled_consistently` — Empty strings are handled consistently.
* `test_batch_partial_not_supported_raises_clear_error` — Batch operation raises `NotSupported` when unsupported.

---

### `test_truncation_and_text_length.py`

**Specification:** §10.2, §10.6, §12.1
**Status:** ✅ Complete (12 tests)

Truncation + `max_text_length` semantics:

* `test_truncation_embed_truncates_when_allowed_and_sets_flag` — Over-limit input with `truncate=True` is truncated to `max_text_length` and sets `truncated=True`.
* `test_truncation_embed_raises_when_truncation_disallowed` — Over-limit input with `truncate=False` raises `TextTooLong`.
* `test_truncation_batch_truncates_all_when_allowed` — Batch with long texts and `truncate=True` truncates each item consistently.
* `test_truncation_batch_oversize_without_truncation_raises` — Batch with over-limit text and `truncate=False` raises `TextTooLong`.
* `test_truncation_short_texts_unchanged` — Inputs under the limit pass through unchanged with `truncated=False`.
* `test_truncation_exact_length_text_handled` — Text at exact length limit is handled correctly.
* `test_truncation_batch_mixed_lengths_with_truncation` — Batch with mixed-length texts handles truncation correctly.
* `test_truncation_unicode_text_truncation` — Unicode text truncation works correctly.
* `test_truncation_truncation_boundary_consistency` — Ensures consistent truncation boundaries.
* `test_truncation_truncation_mode_behavior` — Validates truncation mode behavior.
* `test_truncation_empty_string_handled` — Empty string handling.
* `test_truncation_whitespace_only_text` — Whitespace-only text handling.

---

### `test_normalization_semantics.py`

**Specification:** §10.2, §10.5, §10.6
**Status:** ✅ Complete (10 tests)

Normalization behavior:

* `test_normalization_single_embed_normalize_true_produces_unit_vector` — `normalize=True` produces unit-length vector when supported.
* `test_normalization_single_embed_normalize_false_not_forced_unit_norm` — `normalize=False` does not artificially enforce unit norm.
* `test_normalization_batch_embed_normalize_true_all_unit_vectors` — Batch with `normalize=True` yields unit-norm vectors per item.
* `test_normalization_not_supported_raises_clear_error` — If `supports_normalization=False`, using `normalize=True` raises `NotSupported`.
* `test_normalization_normalizes_at_source_respected` — `normalizes_at_source=True` is honored without extra normalization in base.
* `test_normalization_consistency_across_calls` — Normalization is consistent across multiple calls.
* `test_normalization_different_texts_different_vectors` — Different texts produce different normalized vectors.
* `test_normalization_small_vectors_handled` — Small vectors are handled correctly during normalization.
* `test_normalization_batch_mixed_normalization` — Batch with mixed normalization flags works correctly.
* `test_normalization_different_texts_different_vectors` — Different texts produce different vectors.

---

### `test_count_tokens_behavior.py`

**Specification:** §10.3, §10.5
**Status:** ✅ Complete (10 tests)

Token counting behavior:

* `test_token_counting_returns_non_negative_int` — `count_tokens` returns a non-negative integer.
* `test_token_counting_monotonic_with_text_length` — Longer input never yields fewer tokens than shorter input.
* `test_token_counting_empty_string_handling` — Empty string returns 0 or minimal overhead.
* `test_token_counting_unicode_boundary_cases` — Unicode input is handled safely with non-negative counts.
* `test_token_counting_unknown_model_raises_model_not_available` — Unknown model raises `ModelNotAvailable`.
* `test_token_counting_invalid_input_raises_bad_request` — Invalid input raises `BadRequest`.
* `test_token_counting_various_whitespace_handling` — Various whitespace is handled correctly.
* `test_token_counting_consistent_for_identical_inputs` — Identical inputs yield identical token counts.
* `test_token_counting_support_matches_capabilities` — Token counting availability matches capabilities.
* `test_token_counting_context_propagation` — Context is properly propagated.

---

### `test_error_mapping_retryable.py`

**Specification:** §6.3, §10.4, §12.1–§12.5
**Status:** ✅ Complete (10 tests)

Embedding-specific error taxonomy:

* `test_error_handling_text_too_long_maps_correctly` — `TextTooLong` maps to `TEXT_TOO_LONG` and is non-retryable.
* `test_error_handling_model_not_available_maps_correctly` — Unsupported models map to `MODEL_NOT_AVAILABLE` without retry hints.
* `test_error_handling_bad_request_validation` — Validation errors map to `BAD_REQUEST`.
* `test_error_handling_not_supported_clear_messages` — `NotSupported` provides clear error messages.
* `test_error_handling_deadline_exceeded_maps_correctly` — Pre-expired deadlines raise `DeadlineExceeded` with `DEADLINE_EXCEEDED` code.
* `test_error_handling_batch_partial_failure_codes` — Batch failures emit normalized `code`/`error` entries in `details.failures`.
* `test_error_handling_retryable_errors_have_retry_after_ms` — `ResourceExhausted`, `Unavailable`, `TransientNetwork` expose well-formed `retry_after_ms`.
* `test_error_handling_error_inheritance_hierarchy` — Error inheritance hierarchy is correct.
* `test_error_handling_context_preserved_in_errors` — Context is preserved in error details.
* `test_error_handling_text_too_long_maps_correctly` — Text length errors are correctly mapped.

---

### `test_deadline_enforcement.py`

**Specification:** §6.1, §10.3, §10.6, §12.1
**Status:** ✅ Complete (7 tests)

Deadline behavior via `OperationContext`:

* `test_deadline_budget_calculation_accurate` — Remaining budget helper calculates accurately.
* `test_deadline_preexpired_deadline_fails_fast_embed` — Pre-expired deadlines cause immediate `DeadlineExceeded` before backend work.
* `test_deadline_embed_respects_very_short_deadline` — Short deadlines on `embed()` are honored and may raise `DeadlineExceeded`.
* `test_deadline_batch_partial_completion_before_deadline` — Batch operations respect deadlines and may complete partially.
* `test_deadline_metrics_include_buckets` — Deadline metrics include proper bucket tagging.
* `test_deadline_sequential_operations_respect_deadline` — Sequential operations respect individual deadlines.
* `test_deadline_exceeded_has_clear_error_message` — `DeadlineExceeded` provides clear error messages.

---

### `test_health_report.py`

**Specification:** §10.3, §6.2
**Status:** ✅ Complete (10 tests)

Health endpoint contract:

* `test_health_returns_required_fields` — Health returns required structural fields (e.g. `ok`, `server`, `version`).
* `test_health_ok_is_boolean` — `ok` is strictly boolean.
* `test_health_models_dict_shape` — Health exposes a models map aligned with supported models.
* `test_health_server_version_strings` — Server and version fields are properly formatted strings.
* `test_health_shape_consistent_on_error_like_response` — Health response shape remains stable even when backend is degraded.
* `test_health_models_includes_supported_models` — Models map includes all supported models.
* `test_health_context_propagation` — Context is properly propagated to health endpoint.
* `test_health_siem_safe_no_sensitive_data` — Health endpoint is SIEM-safe with no sensitive data.
* `test_health_performance_reasonable` — Health endpoint performance is reasonable.
* `test_health_idempotent` — Health endpoint is idempotent.

---

### `test_context_siem.py`

**Specification:** §6.4, §13, §15 (Embedding)
**Status:** ✅ Complete (8 tests)

SIEM-safe metrics and context propagation:

* `test_observability_context_propagates_to_metrics` — `observe` called with `component="embedding"` and correct op for successful calls.
* `test_observability_tenant_hashed_never_raw` — Tenant identifiers appear only in hashed/derived form; raw tenant is never logged.
* `test_observability_no_sensitive_data_in_metrics` — No raw text/texts/vectors/embeddings appear in metrics extras.
* `test_observability_metrics_emitted_on_error_path` — Error paths still emit observations and error counters.
* `test_observability_batch_metrics_include_accurate_counts` — Batch operations record accurate batch size metadata in metrics.
* `test_observability_deadline_metrics_include_bucket_tags` — `deadline_bucket`/deadline tags are emitted when `deadline_ms` is set.
* `test_observability_metrics_include_operation_specific_tags` — Operation-specific tags are included in metrics.
* `test_observability_errors_total_counter_incremented_on_failure` — Error counters are incremented on failures.

---

### `test_cache_and_batch_fallback.py`

**Specification:** §10.3, §10.6, §12.5
**Status:** ✅ Complete (13 tests)

Caching & batch fallback behavior:

* `test_cache_hits_and_misses_tracked` — Cache hits and misses are properly tracked.
* `test_cache_tenant_isolation` — Cache keys are tenant-aware; no cross-tenant reuse.
* `test_cache_model_isolation` — Cache respects model isolation.
* `test_cache_normalization_isolation` — Cache respects normalization flag.
* `test_cache_observable_behavior` — Cache behavior is observable via metrics.
* `test_batch_fallback_or_native_behavior` — Batch fallback works when native batch is unsupported.
* `test_batch_handles_invalid_texts` — Batch handles invalid texts with proper error reporting.
* `test_batch_ordering_preserved` — Output ordering is preserved in batch operations.
* `test_batch_metadata_propagation` — Metadata is properly propagated in batch operations.
* `test_batch_size_limit_enforced` — Batch size limits are enforced.
* `test_batch_empty_text_handling` — Empty texts are handled correctly.
* `test_cache_and_batch_independence` — Cache and batch operations work independently.
* `test_batch_cache_integration_positive` — Cache integration works with batch operations.

---

### `test_wire_handler.py`

**Specification:** §4.1, §4.1.6, §10.3, §10.6
**Status:** ✅ Complete (19 tests)

`WireEmbeddingHandler` canonical envelopes:

* `test_wire_contract_capabilities_envelope_success` — `embedding.capabilities` envelope returns `ok:true`, `code:"OK"`, and `result`.
* `test_wire_contract_embed_envelope_success` — `embedding.embed` envelope returns `ok:true` and embedding result in canonical shape.
* `test_wire_contract_embed_batch_envelope_success` — `embedding.embed_batch` envelope returns `ok:true` with correct `embeddings` list.
* `test_wire_contract_count_tokens_envelope_success` — `embedding.count_tokens` envelope returns `ok:true` with integer `result`.
* `test_wire_contract_health_envelope_success` — `embedding.health` envelope returns `ok:true` with proper `result`.
* `test_wire_contract_missing_op_rejected_with_bad_request` — Missing `op` yields normalized `BAD_REQUEST`.
* `test_wire_contract_unknown_op_rejected_with_not_supported` — Unknown `op` yields `NOT_SUPPORTED` envelope.
* `test_wire_contract_embed_missing_required_fields_yields_bad_request` — Missing `text` or `model` yields `BAD_REQUEST`.
* `test_wire_contract_embed_unknown_model_maps_model_not_available` — Unknown model maps to `MODEL_NOT_AVAILABLE`.
* `test_wire_contract_embed_batch_missing_texts_yields_bad_request` — Missing `texts` yields `BAD_REQUEST`.
* `test_wire_contract_embed_batch_empty_texts_list_yields_bad_request` — Empty `texts` list yields `BAD_REQUEST`.
* `test_wire_contract_embed_batch_unknown_model_maps_model_not_available` — Unknown model in batch maps correctly.
* `test_wire_contract_count_tokens_unknown_model_maps_model_not_available` — Unknown model in `count_tokens` maps correctly.
* `test_wire_contract_error_envelope_includes_message_and_type` — Adapter errors surface canonical error envelope with message.
* `test_wire_contract_text_too_long_maps_to_text_too_long_code_when_exposed` — `TextTooLong` maps to `TEXT_TOO_LONG`.
* `test_wire_contract_unexpected_exception_maps_to_unavailable` — Unexpected exceptions are normalized to `UNAVAILABLE` envelopes.
* `test_wire_contract_invalid_envelope_structure_rejected` — Invalid envelope structure is rejected.
* `test_wire_contract_batch_invalid_texts_type_rejected` — Invalid texts type in batch is rejected.
* `test_wire_contract_embed_context_roundtrip_and_context_plumbing` — Verifies `OperationContext` fields are correctly constructed and passed into the adapter.

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
| Streaming flag presence             | test_capabilities_shape.py | ✅      |
| Cache flag accuracy                 | test_capabilities_shape.py | ✅      |

#### `embed()`

| Requirement                         | Test File                          | Status |
| ----------------------------------- | ---------------------------------- | ------ |
| Validates non-empty text/model      | test_embed_basic.py                | ✅      |
| Model must be supported             | test_embed_basic.py                | ✅      |
| Respects truncate & max_text_length | test_truncation_and_text_length.py | ✅      |
| Supports optional normalization     | test_normalization_semantics.py    | ✅      |
| Deadline / context honored          | test_deadline_enforcement.py       | ✅      |
| Vector quality & consistency        | test_embed_basic.py                | ✅      |
| Special character handling          | test_embed_basic.py                | ✅      |
| Context propagation                 | test_embed_basic.py                | ✅      |

#### `embed_batch()`

| Requirement                    | Test File                          | Status |
| ------------------------------ | ---------------------------------- | ------ |
| Validates texts non-empty      | test_embed_batch_basic.py          | ✅      |
| Enforces max_batch_size        | test_embed_batch_basic.py          | ✅      |
| Per-item validation & failures | test_embed_batch_basic.py          | ✅      |
| Truncation semantics           | test_truncation_and_text_length.py | ✅      |
| Normalization semantics        | test_normalization_semantics.py    | ✅      |
| Single-item batch              | test_embed_batch_basic.py          | ✅      |
| Ordering preservation          | test_embed_batch_basic.py          | ✅      |
| Empty string handling          | test_embed_batch_basic.py          | ✅      |

#### `count_tokens()`

| Requirement                     | Test File                     | Status |
| ------------------------------- | ----------------------------- | ------ |
| Non-negative int                | test_count_tokens_behavior.py | ✅      |
| Monotonic behavior              | test_count_tokens_behavior.py | ✅      |
| Model must be supported / gated | test_count_tokens_behavior.py | ✅      |
| Invalid input handling          | test_count_tokens_behavior.py | ✅      |
| Whitespace handling             | test_count_tokens_behavior.py | ✅      |
| Consistency for identical inputs| test_count_tokens_behavior.py | ✅      |
| Capability matching             | test_count_tokens_behavior.py | ✅      |

#### `health()`

| Requirement                      | Test File             | Status |
| -------------------------------- | --------------------- | ------ |
| Returns ok/server/version/models | test_health_report.py | ✅      |
| Stable shape even when degraded  | test_health_report.py | ✅      |
| Boolean ok flag                  | test_health_report.py | ✅      |
| Models mapping                   | test_health_report.py | ✅      |
| Context propagation              | test_health_report.py | ✅      |
| SIEM safety                      | test_health_report.py | ✅      |
| Performance                      | test_health_report.py | ✅      |
| Idempotency                      | test_health_report.py | ✅      |

### §10.4 Errors — Complete Coverage

| Error Type        | Test File                                                                           | Status |
| ----------------- | ----------------------------------------------------------------------------------- | ------ |
| TextTooLong       | test_truncation_and_text_length.py, test_error_mapping_retryable.py                 | ✅      |
| ModelNotAvailable | test_embed_basic.py, test_count_tokens_behavior.py, test_error_mapping_retryable.py | ✅      |
| DeadlineExceeded  | test_deadline_enforcement.py, test_error_mapping_retryable.py                       | ✅      |
| Retryable vs non- | test_error_mapping_retryable.py                                                     | ✅      |
| Partial failures  | test_embed_batch_basic.py, test_error_mapping_retryable.py                          | ✅      |
| BadRequest        | test_error_mapping_retryable.py                                                     | ✅      |
| NotSupported      | test_error_mapping_retryable.py                                                     | ✅      |

### §10.5 Capabilities — Complete Coverage

All required capability fields, normalization/truncation flags, token counting, multi-tenant and deadline support are asserted in:

* `test_capabilities_shape.py` (15 comprehensive tests)
* Cross-checked implicitly via behavior tests above.

### §10.6 Semantics — Complete Coverage

* Truncation rules: `test_truncation_and_text_length.py` (12 tests)
* Normalization rules: `test_normalization_semantics.py` (10 tests)
* Partial-failure encoding: `test_embed_batch_basic.py` (10 tests)
* Deadline behavior: `test_deadline_enforcement.py` (7 tests)
* Observability & privacy: `test_context_siem.py` (8 tests)
* Caching semantics: `test_cache_and_batch_fallback.py` (13 tests)
* Wire contract: `test_wire_handler.py` (19 tests)

---

## Running Tests

### All Embedding conformance tests

```bash
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter pytest tests/embedding/ -v
```

### By category

```bash
# Core operations & capabilities
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter pytest \
  tests/embedding/test_capabilities_shape.py \
  tests/embedding/test_embed_basic.py \
  tests/embedding/test_embed_batch_basic.py -v

# Truncation & normalization & tokens
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter pytest \
  tests/embedding/test_truncation_and_text_length.py \
  tests/embedding/test_normalization_semantics.py \
  tests/embedding/test_count_tokens_behavior.py -v

# Error handling & deadlines & health
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter pytest \
  tests/embedding/test_error_mapping_retryable.py \
  tests/embedding/test_deadline_enforcement.py \
  tests/embedding/test_health_report.py -v

# Observability, caching, wire contract
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter pytest \
  tests/embedding/test_context_siem.py \
  tests/embedding/test_cache_and_batch_fallback.py \
  tests/embedding/test_wire_handler.py -v
```

### With coverage report

```bash
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/embedding/ --cov=corpus_sdk.embedding --cov-report=html
```

---

## Adapter Compliance Checklist

Use this when implementing or validating a new **Embedding adapter** against `EmbeddingProtocolV1` + `BaseEmbeddingAdapter`.

### ✅ Phase 1: Core Operations (4/4)

* [x] `capabilities()` returns `EmbeddingCapabilities` with all required fields (15 tests).
* [x] `embed()` returns valid embeddings with correct dimensions/model (11 tests).
* [x] `embed_batch()` implements batch operations with partial failure reporting (10 tests).
* [x] `count_tokens()` implemented or explicitly `NotSupported` per caps (10 tests).

### ✅ Phase 2: Validation, Truncation & Normalization (14/14)

* [x] Reject empty/invalid `text` / `model`.
* [x] Enforce `supported_models` accurately.
* [x] Enforce `max_batch_size` when present.
* [x] Enforce `max_text_length` with `truncate` semantics.
* [x] `TextTooLong` when `truncate=false`.
* [x] `normalize=true` only if `supports_normalization`.
* [x] Respect `normalizes_at_source`.
* [x] Ensure dimensions set correctly on all embeddings.
* [x] `count_tokens` consistent & model-gated.
* [x] No NaN/Inf or invalid vectors.
* [x] Handle special characters and unicode.
* [x] Preserve ordering in batch operations.
* [x] Handle empty strings correctly.
* [x] Validate text length boundaries.

### ✅ Phase 3: Error Handling & Partial Failures (13/13)

* [x] Map provider errors to normalized codes (`TextTooLong`, `ModelNotAvailable`, etc.).
* [x] Do not treat validation errors as retryable.
* [x] Provide `retry_after_ms` for retryable errors when available.
* [x] Use `EmbeddingResult.failures` / item failures for batch.
* [x] No silent drops in `embed_batch`.
* [x] `DeadlineExceeded` on exhausted budgets.
* [x] Honor `NotSupported` for unsupported features.
* [x] Preserve SIEM-safe `details`.
* [x] Follow §12.5 partial-failure semantics.
* [x] Error inheritance hierarchy correct.
* [x] Context preserved in error details.
* [x] Clear error messages for all error types.
* [x] Proper error code mapping.

### ✅ Phase 4: Observability & Privacy (8/8)

* [x] Use `component="embedding"` in metrics.
* [x] Emit exactly one `observe` per op.
* [x] Never log raw text, embeddings, or tenant IDs.
* [x] Use `tenant_hash`, `deadline_bucket`, `batch_size` as low-cardinality tags.
* [x] Emit error counters on failure.
* [x] Include operation-specific tags in metrics.
* [x] Track cache hits and misses.
* [x] Ensure wire+logs SIEM-safe per §13, §15.

### ✅ Phase 5: Deadlines, Caching & Wire Contract (19/19)

* [x] Respect `OperationContext.deadline_ms` with preflight checks.
* [x] Use `DeadlineExceeded` when time budget elapses.
* [x] If caching, key by `(tenant_hash, model, normalize, sha256(text))`, no raw text.
* [x] Idempotent behavior for repeat-identical unary requests.
* [x] `WireEmbeddingHandler` implements `embedding.*` ops with canonical envelopes.
* [x] Unknown fields ignored; unknown ops → `NotSupported`.
* [x] Error envelopes use normalized `code`/`error`.
* [x] Compatible with `{ "protocol": "embedding/v1.0" }` contract.
* [x] Handle invalid envelope structures.
* [x] Validate required fields in wire requests.
* [x] Propagate context through wire handler.
* [x] Handle unexpected exceptions gracefully.
* [x] Support streaming capability flag.
* [x] Cache isolation by tenant/model/normalization.
* [x] Batch fallback when native batch unsupported.
* [x] Deadline metrics include bucket tags.
* [x] Sequential operations respect deadlines.
* [x] Batch partial completion before deadline.

---

## Conformance Badge

```text
✅ Embedding Protocol V1.0 - 100% Conformant
   135/135 tests passing (12 test files)

   ✅ Core Operations: 21/21 (100%)
   ✅ Capabilities: 15/15 (100%)
   ✅ Batch & Partial Failures: 10/10 (100%)
   ✅ Truncation & Length: 12/12 (100%)
   ✅ Normalization: 10/10 (100%)
   ✅ Token Counting: 10/10 (100%)
   ✅ Error Handling: 10/10 (100%)
   ✅ Deadline Semantics: 7/7 (100%)
   ✅ Health Endpoint: 10/10 (100%)
   ✅ Observability & Privacy: 8/8 (100%)
   ✅ Caching & Idempotency: 13/13 (100%)
   ✅ Wire Contract: 19/19 (100%)

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

**Last Updated:** 2026-01-19
**Maintained By:** Corpus SDK Team
**Status:** 100% V1.0 Conformant - Production Ready (135/135 tests)
