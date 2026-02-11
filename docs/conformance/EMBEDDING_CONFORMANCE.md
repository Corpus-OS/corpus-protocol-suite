# Embedding Protocol Conformance Test Coverage

**Table of Contents**
- [Overview](#overview)
- [Conformance Summary](#conformance-summary)
- [Test Files](#test-files)
- [Specification Mapping](#specification-mapping)
- [Running Tests](#running-tests)
- [Adapter Compliance Checklist](#adapter-compliance-checklist)
- [Conformance Badge](#conformance-badge)

---

## Overview

This document tracks conformance test coverage for the **Embedding Protocol V1.0** specification as defined in `SPECIFICATION.md §10`. Each test validates normative requirements (MUST/SHOULD) from the specification and shared behavior from the common foundation (errors, deadlines, observability, privacy).

This suite constitutes the **official Embedding Protocol V1.0 Reference Conformance Test Suite**. Any implementation (Corpus or third-party) MAY run these tests to verify and publicly claim conformance, provided all referenced tests pass unmodified.

**Protocol Version:** Embedding Protocol V1.0  
**Status:** Stable / Production-Ready  
**Last Updated:** 2026-02-10  
**Test Location:** `tests/embedding/`  
**Performance:** 4.17s total (30.9ms/test average)

## Conformance Summary

**Overall Coverage: 135/135 tests (100%) ✅**

📊 **Total Tests:** 135/135 passing (100%)  
⚡ **Execution Time:** 4.17s (30.9ms/test avg)  
🏆 **Certification:** Platinum (100%)

| Category | Tests | Coverage | Status |
|----------|-------|-----------|---------|
| **Core Operations** | 12/12 | 100% ✅ | Production Ready |
| **Cache & Batch Integration** | 13/13 | 100% ✅ | Production Ready |
| **Capabilities Discovery** | 15/15 | 100% ✅ | Production Ready |
| **Token Counting** | 9/9 | 100% ✅ | Production Ready |
| **Deadline Semantics** | 6/6 | 100% ✅ | Production Ready |
| **Embedding Validation** | 10/10 | 100% ✅ | Production Ready |
| **Batch Operations** | 10/10 | 100% ✅ | Production Ready |
| **Error Handling** | 9/9 | 100% ✅ | Production Ready |
| **Health Endpoint** | 10/10 | 100% ✅ | Production Ready |
| **Normalization Semantics** | 9/9 | 100% ✅ | Production Ready |
| **Truncation & Text Length** | 12/12 | 100% ✅ | Production Ready |
| **Observability & Privacy** | 8/8 | 100% ✅ | Production Ready |
| **Wire Envelopes & Routing** | 12/12 | 100% ✅ | Production Ready |
| **Total** | **135/135** | **100% ✅** | **🏆 Platinum Certified** |

### Performance Characteristics
- **Test Execution:** 4.17 seconds total runtime
- **Average Per Test:** 30.9 milliseconds
- **Cache Efficiency:** 0 cache hits, 135 misses (cache size: 135)
- **Parallel Ready:** Optimized for parallel execution with `pytest -n auto`

### Test Infrastructure
- **Mock Adapter:** `tests.mock.mock_embedding_adapter:MockEmbeddingAdapter` - Deterministic mock for Embedding operations
- **Testing Framework:** pytest 9.0.2 with comprehensive plugin support
- **Environment:** Python 3.10.19 on Darwin
- **Strict Mode:** Off (permissive testing)

### Certification Levels
- 🏆 **Platinum:** 135/135 tests (100%) with comprehensive coverage
- 🥇 **Gold:** 108+ tests (80%+ coverage)
- 🥈 **Silver:** 81+ tests (60%+ coverage)
- 🔬 **Development:** 67+ tests (50%+ coverage)

---

## Test Files

### `test_cache_and_batch_fallback.py`

**Specification:** §10.3 Operations, §11.6 Caching, §12.5 Partial Failure Contracts, §16.3 Caching Strategies  
**Status:** ✅ Complete (13 tests)

Validates cache integration and batch fallback behavior:

* `test_cache_hits_and_misses_tracked` - Cache hits and misses tracked correctly (§16.3)
* `test_cache_tenant_isolation` - Cache respects tenant isolation (§14.1)
* `test_cache_model_isolation` - Cache respects model isolation (§10.3)
* `test_cache_normalization_isolation` - Cache respects normalization isolation (§10.3)
* `test_cache_observable_behavior` - Cache behavior is observable (§13.1)
* `test_batch_fallback_or_native_behavior` - Batch fallback or native behavior validation (§10.3)
* `test_batch_handles_invalid_texts` - Batch handles invalid texts gracefully (§12.5)
* `test_batch_ordering_preserved` - Batch preserves input ordering (§10.3)
* `test_batch_metadata_propagation` - Batch metadata propagation validation (§10.3)
* `test_batch_size_limit_enforced` - Batch size limits enforced (§10.3)
* `test_batch_empty_text_handling` - Empty text handling in batch (§10.3)
* `test_cache_and_batch_independence` - Cache and batch independence validation (§11.6)
* `test_batch_cache_integration_positive` - Batch and cache integration positive test (§11.6)

### `test_capabilities_shape.py`

**Specification:** §10.5 Capabilities, §6.2 Capability Discovery, §10.2 Data Types  
**Status:** ✅ Complete (15 tests)

Tests all aspects of capability discovery for `embedding.capabilities`:

* `test_capabilities_returns_correct_type` - Returns EmbeddingCapabilities instance (§10.2)
* `test_capabilities_identity_fields` - `server`/`version` are non-empty strings (§6.2)
* `test_capabilities_supported_models_non_empty_tuple` - `supported_models` is non-empty tuple (§10.5)
* `test_capabilities_resource_limits_valid` - Resource limits are valid values (§10.5)
* `test_capabilities_feature_flags_boolean` - All feature flags are boolean types (§10.5)
* `test_capabilities_truncation_mode_valid` - Truncation mode validation (§10.5)
* `test_capabilities_max_dimensions_consistent_with_models` - Max dimensions consistent with models (§10.5)
* `test_capabilities_idempotent` - Multiple calls return consistent results (§6.2)
* `test_capabilities_serializable_structure` - Capabilities are JSON serializable (§4.2.1)
* `test_capabilities_protocol_version` - Protocol version validation (§4.2.2)
* `test_capabilities_supported_models_accurate` - Supported models list is accurate (§10.5)
* `test_capabilities_max_batch_size_respected` - Max batch size respected in operations (§10.3)
* `test_capabilities_max_text_length_respected` - Max text length respected in operations (§10.3)
* `test_capabilities_match_operational_behavior_batch` - Batch capability matches operational behavior (§10.5)
* `test_capabilities_match_operational_behavior_normalization` - Normalization capability matches behavior (§10.5)
* `test_capabilities_streaming_flag_present` - Streaming flag presence validation (§10.5)
* `test_capabilities_cache_flag_accurate` - Cache flag accuracy validation (§11.6)

### `test_context_siem.py`

**Specification:** §13 Observability and Monitoring, §15 Privacy Considerations  
**Status:** ✅ Complete (8 tests) ⭐ Critical

Validates SIEM-safe observability:

* `test_observability_context_propagates_to_metrics` - Context propagates to metrics (§13.1)
* `test_observability_tenant_hashed_never_raw` - Tenant identifiers hashed, never raw (§13.1, §15)
* `test_observability_no_sensitive_data_in_metrics` - No sensitive data in metrics (§13.1, §15)
* `test_observability_metrics_emitted_on_error_path` - Metrics emitted on error paths (§13.1)
* `test_observability_batch_metrics_include_accurate_counts` - Batch metrics include accurate counts (§13.1)
* `test_observability_deadline_metrics_include_bucket_tags` - Deadline metrics include bucket tags (§13.1)
* `test_observability_metrics_include_operation_specific_tags` - Metrics include operation-specific tags (§13.1)
* `test_observability_errors_total_counter_incremented_on_failure` - Error counters incremented on failure (§13.1)

### `test_count_tokens_behavior.py`

**Specification:** §10.3 Operations, §10.5 Capabilities  
**Status:** ✅ Complete (9 tests)

Validates token counting behavior for `embedding.count_tokens`:

* `test_token_counting_returns_non_negative_int` - Returns non-negative integer (§10.3)
* `test_token_counting_context_propagation` - Context propagation validation (§6.1)
* `test_token_counting_monotonic_with_text_length` - Monotonic with text length (§10.3)
* `test_token_counting_empty_string_handling` - Empty string handling (§10.3)
* `test_token_counting_unicode_boundary_cases` - Unicode boundary cases (§10.3)
* `test_token_counting_consistent_for_identical_inputs` - Consistent for identical inputs (§10.3)
* `test_token_counting_unknown_model_raises_model_not_available` - Unknown model raises ModelNotAvailable (§10.4)
* `test_token_counting_invalid_input_raises_bad_request` - Invalid input raises BadRequest (§10.4)
* `test_token_counting_various_whitespace_handling` - Various whitespace handling (§10.3)
* `test_token_counting_support_matches_capabilities` - Support matches capabilities (§10.5)

### `test_deadline_enforcement.py`

**Specification:** §6.1 Operation Context, §12.1 Retry Semantics  
**Status:** ✅ Complete (6 tests)

Validates deadline behavior:

* `test_deadline_budget_calculation_accurate` - Budget calculation accuracy (§6.1)
* `test_deadline_preexpired_deadline_fails_fast_embed` - Pre-expired deadline fails fast (§12.1)
* `test_deadline_embed_respects_very_short_deadline` - Embed respects short deadline (§12.1)
* `test_deadline_batch_partial_completion_before_deadline` - Batch partial completion before deadline (§12.1)
* `test_deadline_metrics_include_buckets` - Deadline metrics include buckets (§13.1)
* `test_deadline_sequential_operations_respect_deadline` - Sequential operations respect deadline (§12.1)
* `test_deadline_exceeded_has_clear_error_message` - DeadlineExceeded has clear error message (§12.4)

### `test_embed_basic.py`

**Specification:** §10.3 Operations, §10.5 Capabilities, §10.6 Semantics  
**Status:** ✅ Complete (10 tests)

Validates basic embedding contract for `embedding.embed`:

* `test_core_ops_embed_returns_valid_embedding_structure` - Returns valid EmbeddingVector structure (§10.2)
* `test_core_ops_embed_requires_valid_text` - Requires valid text (§10.3)
* `test_core_ops_embed_requires_valid_model` - Requires valid model (§10.3)
* `test_core_ops_embed_unknown_model_clear_error` - Unknown model clear error (§10.4)
* `test_core_ops_embed_truncation_behavior_matches_capabilities` - Truncation behavior matches capabilities (§10.5)
* `test_core_ops_embed_normalization_produces_unit_vectors` - Normalization produces unit vectors (§10.6)
* `test_core_ops_embed_normalization_unsupported_raises_clear_error` - Normalization unsupported raises error (§10.4)
* `test_core_ops_embed_vector_quality_and_consistency` - Vector quality and consistency (§10.6)
* `test_core_ops_embed_special_character_handling` - Special character handling (§10.3)
* `test_core_ops_embed_context_propagation` - Context propagation (§6.1)
* `test_core_ops_embed_dimensions_consistent_with_capabilities` - Dimensions consistent with capabilities (§10.5)

### `test_embed_batch_basic.py`

**Specification:** §10.3 Operations, §10.5 Capabilities, §12.5 Partial Failure Contracts  
**Status:** ✅ Complete (10 tests)

Validates batch embedding contract for `embedding.embed_batch`:

* `test_batch_partial_returns_batch_result` - Returns BatchResult structure (§10.2)
* `test_batch_partial_requires_non_empty_model` - Requires non-empty model (§10.3)
* `test_batch_partial_requires_non_empty_texts` - Requires non-empty texts list (§10.3)
* `test_batch_partial_respects_max_batch_size` - Respects max batch size (§10.3)
* `test_batch_partial_unknown_model_raises_model_not_available` - Unknown model raises ModelNotAvailable (§10.4)
* `test_batch_partial_partial_failure_reporting` - Partial failure reporting (§12.5)
* `test_batch_partial_single_item_works` - Single item works (§10.3)
* `test_batch_partial_ordering_preserved` - Input ordering preserved (§10.3)
* `test_batch_partial_empty_strings_handled_consistently` - Empty strings handled consistently (§10.3)
* `test_batch_partial_not_supported_raises_clear_error` - Not supported raises clear error (§10.4)

### `test_error_mapping_retryable.py`

**Specification:** §10.4 Errors, §6.3 Error Taxonomy, §12.1 Retry Semantics, §12.4 Error Mapping Table  
**Status:** ✅ Complete (9 tests)

Validates error classification:

* `test_error_handling_text_too_long_maps_correctly` - TextTooLong maps correctly (§10.4)
* `test_error_handling_model_not_available_maps_correctly` - ModelNotAvailable maps correctly (§10.4)
* `test_error_handling_bad_request_validation` - BadRequest validation (§6.3)
* `test_error_handling_not_supported_clear_messages` - NotSupported clear messages (§6.3)
* `test_error_handling_deadline_exceeded_maps_correctly` - DeadlineExceeded maps correctly (§12.4)
* `test_error_handling_batch_partial_failure_codes` - Batch partial failure codes (§12.5)
* `test_error_handling_retryable_errors_have_retry_after_ms` - Retryable errors have retry_after_ms (§12.1)
* `test_error_handling_error_inheritance_hierarchy` - Error inheritance hierarchy (§6.3)
* `test_error_handling_context_preserved_in_errors` - Context preserved in errors (§6.1)

### `test_health_report.py`

**Specification:** §10.3 Operations, §6.4 Observability Interfaces, §10.5 Capabilities  
**Status:** ✅ Complete (10 tests)

Validates health endpoint contract for `embedding.health`:

* `test_health_returns_required_fields` - Returns required fields (§10.3)
* `test_health_ok_is_boolean` - `ok` is boolean (§10.3)
* `test_health_models_dict_shape` - Models dictionary shape (§10.3)
* `test_health_server_version_strings` - Server/version strings (§6.4)
* `test_health_shape_consistent_on_error_like_response` - Shape consistent on error-like response (§6.4)
* `test_health_models_includes_supported_models` - Models includes supported models (§10.5)
* `test_health_context_propagation` - Context propagation (§6.1)
* `test_health_siem_safe_no_sensitive_data` - SIEM-safe, no sensitive data (§13.1, §15)
* `test_health_performance_reasonable` - Performance reasonable (§16.1)
* `test_health_idempotent` - Idempotent calls (§6.2)

### `test_normalization_semantics.py`

**Specification:** §10.6 Semantics, §10.5 Capabilities  
**Status:** ✅ Complete (9 tests)

Validates normalization semantics:

* `test_normalization_single_embed_normalize_true_produces_unit_vector` - Normalize true produces unit vector (§10.6)
* `test_normalization_single_embed_normalize_false_not_forced_unit_norm` - Normalize false not forced unit norm (§10.6)
* `test_normalization_batch_embed_normalize_true_all_unit_vectors` - Batch normalize true all unit vectors (§10.6)
* `test_normalization_not_supported_raises_clear_error` - Not supported raises clear error (§10.4)
* `test_normalization_normalizes_at_source_respected` - Normalizes at source respected (§10.5)
* `test_normalization_consistency_across_calls` - Consistency across calls (§10.6)
* `test_normalization_different_texts_different_vectors` - Different texts produce different vectors (§10.6)
* `test_normalization_small_vectors_handled` - Small vectors handled correctly (§10.6)
* `test_normalization_batch_mixed_normalization` - Batch mixed normalization handling (§10.6)

### `test_truncation_and_text_length.py`

**Specification:** §10.3 Operations, §10.5 Capabilities, §10.4 Errors  
**Status:** ✅ Complete (12 tests)

Validates truncation and text length handling:

* `test_truncation_embed_truncates_when_allowed_and_sets_flag` - Truncates when allowed and sets flag (§10.3)
* `test_truncation_embed_raises_when_truncation_disallowed` - Raises when truncation disallowed (§10.4)
* `test_truncation_batch_truncates_all_when_allowed` - Batch truncates all when allowed (§10.3)
* `test_truncation_batch_oversize_without_truncation_raises` - Batch oversize without truncation raises (§10.4)
* `test_truncation_short_texts_unchanged` - Short texts unchanged (§10.3)
* `test_truncation_exact_length_text_handled` - Exact length text handled (§10.3)
* `test_truncation_batch_mixed_lengths_with_truncation` - Batch mixed lengths with truncation (§10.3)
* `test_truncation_unicode_text_truncation` - Unicode text truncation (§10.3)
* `test_truncation_truncation_boundary_consistency` - Truncation boundary consistency (§10.3)
* `test_truncation_truncation_mode_behavior` - Truncation mode behavior (§10.5)
* `test_truncation_empty_string_handled` - Empty string handled (§10.3)
* `test_truncation_whitespace_only_text` - Whitespace-only text handling (§10.3)

### `test_wire_handler.py`

**Specification:** §4.2 Wire-First Canonical Form, §4.2.6 Operation Registry, §10 Embedding Protocol, §6.1 Operation Context, §6.3 Error Taxonomy, §12.4 Error Mapping Table  
**Status:** ✅ Complete (12 tests)

Validates `WireEmbeddingHandler` wire-level contract:

* `test_wire_contract_capabilities_envelope_success` — `embedding.capabilities` success envelope (§4.2.1)
* `test_wire_contract_embed_envelope_success` — `embedding.embed` success envelope (§4.2.1)
* `test_wire_contract_embed_batch_envelope_success` — `embedding.embed_batch` success envelope (§4.2.1)
* `test_wire_contract_count_tokens_envelope_success` — `embedding.count_tokens` success envelope (§4.2.1)
* `test_wire_contract_health_envelope_success` — `embedding.health` success envelope (§4.2.1)
* `test_wire_contract_embed_context_roundtrip_and_context_plumbing` — Context propagation (§6.1)
* `test_wire_contract_missing_op_rejected_with_bad_request` — Missing op rejected (§4.2.1)
* `test_wire_contract_unknown_op_rejected_with_not_supported` — Unknown op rejected (§4.2.6)
* `test_wire_contract_embed_missing_required_fields_yields_bad_request` — Missing required fields (§4.2.1)
* `test_wire_contract_embed_unknown_model_maps_model_not_available` — Unknown model maps to ModelNotAvailable (§10.4)
* `test_wire_contract_embed_batch_missing_texts_yields_bad_request` — Batch missing texts (§4.2.1)
* `test_wire_contract_embed_batch_empty_texts_list_yields_bad_request` — Batch empty texts list (§4.2.1)
* `test_wire_contract_embed_batch_unknown_model_maps_model_not_available` — Batch unknown model (§10.4)
* `test_wire_contract_count_tokens_unknown_model_maps_model_not_available` — Count tokens unknown model (§10.4)
* `test_wire_contract_error_envelope_includes_message_and_type` — Error envelope includes message and type (§12.4)
* `test_wire_contract_text_too_long_maps_to_text_too_long_code_when_exposed` — TextTooLong maps correctly (§10.4)
* `test_wire_contract_unexpected_exception_maps_to_unavailable` — Unexpected exception maps to Unavailable (§6.3)
* `test_wire_contract_invalid_envelope_structure_rejected` — Invalid envelope structure rejected (§4.2.1)
* `test_wire_contract_batch_invalid_texts_type_rejected` — Batch invalid texts type rejected (§4.2.1)

---

## Specification Mapping

### §10.3 Operations - Complete Coverage

#### `embedding.capabilities()` (§10.3, §10.5)

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Returns EmbeddingCapabilities | `test_capabilities_shape.py` | ✅ |
| Identity fields non-empty | `test_capabilities_shape.py` | ✅ |
| Supported models non-empty | `test_capabilities_shape.py` | ✅ |
| Resource limits valid | `test_capabilities_shape.py` | ✅ |
| Feature flags boolean | `test_capabilities_shape.py` | ✅ |
| Idempotent calls | `test_capabilities_shape.py` | ✅ |
| JSON serializable | `test_capabilities_shape.py` | ✅ |
| Protocol version validation | `test_capabilities_shape.py` | ✅ |

#### `embedding.embed()` (§10.3)

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Returns EmbeddingVector | `test_embed_basic.py` | ✅ |
| Requires valid text | `test_embed_basic.py` | ✅ |
| Requires valid model | `test_embed_basic.py` | ✅ |
| Truncation behavior matches capabilities | `test_embed_basic.py`, `test_truncation_and_text_length.py` | ✅ |
| Normalization produces unit vectors | `test_embed_basic.py`, `test_normalization_semantics.py` | ✅ |
| Vector quality and consistency | `test_embed_basic.py` | ✅ |
| Special character handling | `test_embed_basic.py` | ✅ |
| Context propagation | `test_embed_basic.py` | ✅ |
| Dimensions consistent with capabilities | `test_embed_basic.py` | ✅ |
| Deadline enforcement | `test_deadline_enforcement.py` | ✅ |

#### `embedding.embed_batch()` (§10.3, §12.5)

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Returns BatchResult | `test_embed_batch_basic.py` | ✅ |
| Requires non-empty model | `test_embed_batch_basic.py` | ✅ |
| Requires non-empty texts | `test_embed_batch_basic.py` | ✅ |
| Respects max batch size | `test_embed_batch_basic.py` | ✅ |
| Partial failure reporting | `test_embed_batch_basic.py` | ✅ |
| Input ordering preserved | `test_embed_batch_basic.py` | ✅ |
| Empty strings handled consistently | `test_embed_batch_basic.py` | ✅ |
| Cache integration | `test_cache_and_batch_fallback.py` | ✅ |
| Batch fallback behavior | `test_cache_and_batch_fallback.py` | ✅ |
| Deadline enforcement | `test_deadline_enforcement.py` | ✅ |

#### `embedding.count_tokens()` (§10.3)

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Returns non-negative integer | `test_count_tokens_behavior.py` | ✅ |
| Monotonic with text length | `test_count_tokens_behavior.py` | ✅ |
| Empty string handling | `test_count_tokens_behavior.py` | ✅ |
| Consistent for identical inputs | `test_count_tokens_behavior.py` | ✅ |
| Unicode boundary cases | `test_count_tokens_behavior.py` | ✅ |
| Support matches capabilities | `test_count_tokens_behavior.py` | ✅ |
| Context propagation | `test_count_tokens_behavior.py` | ✅ |
| Deadline enforcement | `test_deadline_enforcement.py` | ✅ |

#### `embedding.health()` (§10.3, §6.4)

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Returns required fields | `test_health_report.py` | ✅ |
| `ok` is boolean | `test_health_report.py` | ✅ |
| Models dictionary shape | `test_health_report.py` | ✅ |
| Server/version strings | `test_health_report.py` | ✅ |
| Shape consistent | `test_health_report.py` | ✅ |
| Models includes supported models | `test_health_report.py` | ✅ |
| Context propagation | `test_health_report.py` | ✅ |
| SIEM-safe | `test_health_report.py` | ✅ |
| Idempotent | `test_health_report.py` | ✅ |

---

### §10.4 Errors - Complete Coverage

| Error Type | Test File | Status |
|------------|-----------|--------|
| TextTooLong | `test_error_mapping_retryable.py`, `test_truncation_and_text_length.py` | ✅ |
| ModelNotAvailable | `test_error_mapping_retryable.py`, `test_embed_basic.py`, `test_embed_batch_basic.py` | ✅ |
| NotSupported | `test_error_mapping_retryable.py`, `test_embed_batch_basic.py`, `test_normalization_semantics.py` | ✅ |
| BadRequest | `test_error_mapping_retryable.py`, `test_count_tokens_behavior.py` | ✅ |
| DeadlineExceeded | `test_error_mapping_retryable.py`, `test_deadline_enforcement.py` | ✅ |

---

### §10.5 Capabilities - Complete Coverage

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Supported models list | `test_capabilities_shape.py` | ✅ |
| Max batch size | `test_capabilities_shape.py` | ✅ |
| Max text length | `test_capabilities_shape.py` | ✅ |
| Max dimensions | `test_capabilities_shape.py` | ✅ |
| Supports normalization | `test_capabilities_shape.py` | ✅ |
| Normalizes at source | `test_capabilities_shape.py` | ✅ |
| Supports truncation | `test_capabilities_shape.py` | ✅ |
| Supports token counting | `test_capabilities_shape.py` | ✅ |
| Supports deadline | `test_capabilities_shape.py` | ✅ |
| Idempotent writes | `test_capabilities_shape.py` | ✅ |
| Supports multi-tenant | `test_capabilities_shape.py` | ✅ |
| Cache support flags | `test_capabilities_shape.py` | ✅ |

---

### §10.6 Semantics - Complete Coverage

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Normalization produces unit vectors | `test_normalization_semantics.py` | ✅ |
| Normalization not forced when false | `test_normalization_semantics.py` | ✅ |
| Consistency across calls | `test_normalization_semantics.py` | ✅ |
| Different texts produce different vectors | `test_normalization_semantics.py` | ✅ |
| Small vectors handled correctly | `test_normalization_semantics.py` | ✅ |
| Batch mixed normalization | `test_normalization_semantics.py` | ✅ |

---

### §13 Observability - Complete Coverage

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Tenant never logged raw | `test_context_siem.py` | ✅ |
| Tenant hashed in metrics | `test_context_siem.py` | ✅ |
| No sensitive data in metrics | `test_context_siem.py` | ✅ |
| Metrics on error path | `test_context_siem.py` | ✅ |
| Batch metrics include accurate counts | `test_context_siem.py` | ✅ |
| Deadline metrics include buckets | `test_context_siem.py` | ✅ |
| Operation-specific tags | `test_context_siem.py` | ✅ |
| Error counters incremented | `test_context_siem.py` | ✅ |

---

### §6.1 Context & Deadlines - Complete Coverage

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Budget calculation accuracy | `test_deadline_enforcement.py` | ✅ |
| Pre-expired deadline fails fast | `test_deadline_enforcement.py` | ✅ |
| Embed respects short deadline | `test_deadline_enforcement.py` | ✅ |
| Batch partial completion before deadline | `test_deadline_enforcement.py` | ✅ |
| Sequential operations respect deadline | `test_deadline_enforcement.py` | ✅ |
| Clear error messages | `test_deadline_enforcement.py` | ✅ |

---

### §4.2 Wire Protocol - Complete Coverage

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Embedding operation routing | `test_wire_handler.py` | ✅ |
| Error envelope normalization | `test_wire_handler.py` | ✅ |
| Context propagation | `test_wire_handler.py` | ✅ |
| Unknown operation handling | `test_wire_handler.py` | ✅ |
| Missing required keys mapping | `test_wire_handler.py` | ✅ |
| Context and args validation | `test_wire_handler.py` | ✅ |
| Missing required fields mapping | `test_wire_handler.py` | ✅ |
| Error envelope structure | `test_wire_handler.py` | ✅ |

---

## Running Tests

### All Embedding conformance tests (4.17s typical)
```bash
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter pytest tests/embedding/ -v
```

### Performance Optimized Runs
```bash
# Parallel execution (recommended for CI/CD) - ~2.2s
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter pytest tests/embedding/ -n auto

# With detailed timing report
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter pytest tests/embedding/ --durations=10

# Fast mode (skip slow markers)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter pytest tests/embedding/ -k "not slow"
```

### By category with timing estimates
```bash
# Core operations (~0.8s)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter pytest \
  tests/embedding/test_embed_basic.py \
  tests/embedding/test_embed_batch_basic.py \
  tests/embedding/test_health_report.py -v

# Cache & normalization (~0.7s)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter pytest \
  tests/embedding/test_cache_and_batch_fallback.py \
  tests/embedding/test_normalization_semantics.py -v

# Capabilities & validation (~0.6s)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter pytest \
  tests/embedding/test_capabilities_shape.py \
  tests/embedding/test_truncation_and_text_length.py -v

# Token counting & errors (~0.5s)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter pytest \
  tests/embedding/test_count_tokens_behavior.py \
  tests/embedding/test_error_mapping_retryable.py -v

# Infrastructure (~0.6s)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter pytest \
  tests/embedding/test_deadline_enforcement.py \
  tests/embedding/test_context_siem.py -v

# Wire handler (~0.4s)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter pytest \
  tests/embedding/test_wire_handler.py -v
```

### With Coverage Report
```bash
# Basic coverage (4.8s typical)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/embedding/ --cov=corpus_sdk.embedding --cov-report=html

# Minimal coverage (4.5s typical)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/embedding/ --cov=corpus_sdk.embedding --cov-report=term-missing

# CI/CD optimized (parallel + coverage) - ~2.5s
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/embedding/ -n auto --cov=corpus_sdk.embedding --cov-report=xml
```

### Adapter-Agnostic Usage
To validate a **third-party** or custom Embedding Protocol implementation:

1. Implement the Embedding Protocol V1.0 interface as defined in `SPECIFICATION.md §10`
2. Provide a small adapter/fixture that binds these tests to your implementation
3. Run the full `tests/embedding/` suite
4. If all 135 tests pass unmodified, you can accurately claim:
   **"Embedding Protocol V1.0 - 100% Conformant (Corpus Reference Suite)"**

### With Makefile Integration
```bash
# Run all Embedding tests (4.17s typical)
make test-embedding

# Run Embedding tests with coverage (4.8s typical)
make test-embedding-coverage

# Run Embedding tests in parallel (2.2s typical)
make test-embedding-parallel

# Run specific categories
make test-embedding-core      # Core operations
make test-embedding-cache     # Cache & batch integration
make test-embedding-norm      # Normalization semantics
make test-embedding-errors    # Error handling
make test-embedding-wire      # Wire handler
```

---

## Adapter Compliance Checklist

Use this checklist when implementing or validating a new Embedding adapter:

### ✅ Phase 1: Core Operations (5/5)
* [x] `embedding.capabilities()` returns valid `EmbeddingCapabilities` with all fields (§10.5)
* [x] `embedding.embed()` returns valid `EmbeddingVector` with proper structure (§10.3)
* [x] `embedding.embed_batch()` returns `BatchResult` with partial failure support (§10.3, §12.5)
* [x] `embedding.count_tokens()` returns non-negative integer with proper behavior (§10.3)
* [x] `embedding.health()` returns proper health status with models dictionary (§10.3)

### ✅ Phase 2: Validation & Semantics (15/15)
* [x] Reject unknown models with `ModelNotAvailable` (§10.4)
* [x] Validate text input requirements (§10.3)
* [x] Respect truncation settings and capabilities (§10.3, §10.5)
* [x] Normalization produces unit vectors when supported (§10.6)
* [x] Reject normalization when unsupported (§10.4)
* [x] Ensure vector quality and consistency (§10.6)
* [x] Handle special characters correctly (§10.3)
* [x] Preserve input ordering in batch operations (§10.3)
* [x] Handle empty strings consistently (§10.3)
* [x] Enforce max batch size limits (§10.3)
* [x] Support token counting when capability declared (§10.5)
* [x] Provide clear error messages (§12.4)
* [x] Context propagation through all operations (§6.1)
* [x] Dimensions consistent with capabilities (§10.5)
* [x] Empty texts list validation (§10.3)

### ✅ Phase 3: Error Handling & Semantics (9/9)
* [x] Map `TextTooLong` correctly when truncation disabled (§10.4)
* [x] Map `ModelNotAvailable` for unknown models (§10.4)
* [x] Map `NotSupported` for unsupported features (§6.3)
* [x] Map `BadRequest` for validation errors (§6.3)
* [x] Map `DeadlineExceeded` on expired budgets (§12.1, §12.4)
* [x] Batch partial failure codes and reporting (§12.5)
* [x] Retryable errors include `retry_after_ms` when applicable (§12.1)
* [x] Proper error inheritance hierarchy (§6.3)
* [x] Context preserved in errors (§6.1)

### ✅ Phase 4: Observability & Privacy (8/8)
* [x] Use `component="embedding"` in metrics (§13.1)
* [x] Tenant hashed, never logged raw (§13.1, §15)
* [x] No sensitive data in metrics or logs (§13.1, §15)
* [x] Metrics emitted on error paths (§13.1)
* [x] Batch metrics include accurate counts (§13.1)
* [x] Deadline metrics include bucket tags (§13.1)
* [x] Operation-specific tags in metrics (§13.1)
* [x] Error counters incremented on failure (§13.1)

### ✅ Phase 5: Deadlines, Caching & Wire Contract (18/18)
* [x] Respect `OperationContext.deadline_ms` with preflight checks (§6.1)
* [x] Use `DeadlineExceeded` when time budget elapses mid-operation (§12.1)
* [x] Cache hits and misses tracked correctly (§16.3)
* [x] Cache respects tenant isolation (§14.1)
* [x] Cache respects model isolation (§10.3)
* [x] Cache respects normalization isolation (§10.3)
* [x] Cache behavior is observable (§13.1)
* [x] Batch fallback or native behavior validation (§10.3)
* [x] Batch and cache integration works correctly (§11.6)
* [x] `WireEmbeddingHandler` implements all `embedding.*` ops with canonical envelopes (§4.2.6)
* [x] Unknown fields ignored; unknown ops → `NotSupported` (§4.2.5, §4.2.6)
* [x] Error envelopes use normalized `code`/`error` structure (§6.3)
* [x] Proper wire envelope shapes for all operations (§4.2.1)
* [x] Context propagation through wire handler (§6.1)
* [x] Success envelopes for all operations (§4.2.1)
* [x] Missing/invalid operation error handling (§4.2.6)
* [x] Embedding adapter error normalization (§6.3)

---

## Conformance Badge

```text
🏆 EMBEDDING PROTOCOL V1.0 - PLATINUM CERTIFIED
   135/135 conformance tests passing (100%)

   📊 Total Tests: 135/135 passing (100%)
   ⚡ Execution Time: 4.17s (30.9ms/test avg)
   🏆 Certification: Platinum (100%)

   ✅ Core Operations: 12/12 (100%) - §10.3
   ✅ Cache & Batch Integration: 13/13 (100%) - §10.3, §11.6, §16.3
   ✅ Capabilities Discovery: 15/15 (100%) - §10.5, §6.2
   ✅ Token Counting: 9/9 (100%) - §10.3
   ✅ Deadline Semantics: 6/6 (100%) - §6.1, §12.1
   ✅ Embedding Validation: 10/10 (100%) - §10.3
   ✅ Batch Operations: 10/10 (100%) - §10.3, §12.5
   ✅ Error Handling: 9/9 (100%) - §10.4, §6.3, §12.4
   ✅ Health Endpoint: 10/10 (100%) - §10.3, §6.4
   ✅ Normalization Semantics: 9/9 (100%) - §10.6
   ✅ Truncation & Text Length: 12/12 (100%) - §10.3, §10.5
   ✅ Observability & Privacy: 8/8 (100%) - §13, §15
   ✅ Wire Envelopes & Routing: 12/12 (100%) - §4.2

   Status: Production Ready 🏆 Platinum Certified
```

**Badge Suggestion:**
[![Corpus Embedding Protocol](https://img.shields.io/badge/CorpusEmbedding%20Protocol-Platinum%20Certified-brightgreen)](./embedding_conformance_report.json)

**Performance Benchmark:**
```text
Execution Time: 4.17s total (30.9ms/test average)
Cache Efficiency: 0 hits, 135 misses (cache size: 135)
Parallel Ready: Yes (optimized for pytest-xdist)
Memory Footprint: Minimal (deterministic mocks)
Specification Coverage: 100% of §10 requirements
Test Files: 12 comprehensive modules
```

**Last Updated:** 2026-02-10  
**Maintained By:** Corpus SDK Team  
**Test Suite:** `tests/embedding/` (12 test files)  
**Specification Version:** V1.0.0 §10  
**Status:** 100% V1.0 Conformant - Platinum Certified (135/135 tests, 4.17s runtime)

---
