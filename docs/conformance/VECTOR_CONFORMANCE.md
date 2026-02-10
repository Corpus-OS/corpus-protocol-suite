# Vector Protocol Conformance Test Coverage

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

This document tracks conformance test coverage for the **Vector Protocol V1.0** specification as defined in `SPECIFICATION.md §9`. Each test validates normative requirements (MUST/SHOULD) from the specification and shared behavior from the common foundation (errors, deadlines, observability, privacy).

This suite constitutes the **official Vector Protocol V1.0 Reference Conformance Test Suite**. Any implementation (Corpus or third-party) MAY run these tests to verify and publicly claim conformance, provided all referenced tests pass unmodified.

**Protocol Version:** Vector Protocol V1.0
**Status:** Stable / Production-Ready
**Last Updated:** 2026-01-19
**Test Location:** `tests/vector/`

## Conformance Summary

**Overall Coverage: 108/108 tests (100%) ✅**

| Category                 | Tests | Coverage |
| ------------------------ | ----- | -------- |
| Core Operations          | 12/12 | 100% ✅   |
| Capabilities             | 9/9   | 100% ✅   |
| Namespace Management     | 10/10 | 100% ✅   |
| Upsert Operations        | 8/8   | 100% ✅   |
| Query Operations         | 12/12 | 100% ✅   |
| Delete Operations        | 8/8   | 100% ✅   |
| Filtering Semantics      | 7/7   | 100% ✅   |
| Dimension Validation     | 6/6   | 100% ✅   |
| Error Handling           | 12/12 | 100% ✅   |
| Deadline Semantics       | 5/5   | 100% ✅   |
| Health Endpoint          | 6/6   | 100% ✅   |
| Observability & Privacy  | 6/6   | 100% ✅   |
| Batch Size Limits        | 6/6   | 100% ✅   |
| Wire Envelopes & Routing | 13/13 | 100% ✅   |

**Certification Levels:**
- 🏆 **Gold:** 108/108 tests (100%)
- 🥈 **Silver:** 86+ tests (80%+)
- 🔬 **Development:** 54+ tests (50%+)

---

## Test Files

### `test_capabilities_shape.py`

**Specification:** §9.2, §6.2 - Capabilities Discovery
**Status:** ✅ Complete (9 tests)

Tests all aspects of capability discovery:

* `test_capabilities_capabilities_returns_correct_type` - Returns `VectorCapabilities` dataclass instance
* `test_capabilities_identity_fields` - `server` / `version` are non-empty strings
* `test_capabilities_supported_metrics` - Non-empty tuple with valid distance metrics
* `test_capabilities_resource_limits_positive` - `max_dimensions`, `max_top_k`, `max_batch_size` are positive or `None`
* `test_capabilities_feature_flags_boolean` - All feature flags are booleans
* `test_capabilities_idempotent_calls` - Multiple calls return consistent results
* `test_capabilities_all_required_fields_present` - All required fields present and valid
* `test_capabilities_text_storage_strategy_enum_and_text_limits` - Text storage strategy enum validation
* `test_capabilities_supports_batch_queries_flag_boolean` - Batch queries flag boolean validation

### `test_namespace_operations.py`

**Specification:** §9.3, §9.4 - Namespace Management
**Status:** ✅ Complete (10 tests)

Validates namespace lifecycle:

* `test_namespace_create_namespace_returns_success` - Namespace creation succeeds with valid spec
* `test_namespace_namespace_requires_positive_dimensions` - Dimensions must be positive
* `test_namespace_namespace_requires_valid_distance_metric` - Metric validated against §9.4 and capabilities
* `test_namespace_health_exposes_namespaces_dict` - Health/namespaces exposure returns a dictionary
* `test_namespace_delete_namespace_idempotent` - Deleting a non-existent namespace succeeds (idempotent)
* `test_namespace_namespace_isolation` - Vectors in different namespaces are isolated
* `test_namespace_ops_respect_supports_index_management_flag` - Index management flag respected
* `test_namespace_query_rejects_namespace_when_supports_namespaces_false` - Namespace support validation
* `test_namespace_upsert_rejects_vector_namespace_mismatch` - Namespace-vector mismatch validation
* `test_namespace_batch_query_rejects_query_namespace_mismatch` - Batch query namespace validation

### `test_upsert_basic.py`

**Specification:** §9.3, §9.5, §12.5 - Upsert Operations & Partial Failures
**Status:** ✅ Complete (8 tests)

Validates upsert contract and partial-failure semantics:

* `test_upsert_upsert_returns_result_with_counts` - Returns `upserted_count`, `failed_count`
* `test_upsert_validates_dimensions` - Per-item dimension mismatches reported explicitly
* `test_upsert_validates_namespace_exists_or_behavior_documented` - Unknown namespace handled via validation or explicit semantics
* `test_upsert_requires_non_empty_vectors` - Rejects empty vectors list
* `test_upsert_partial_failure_reporting` - Partial failures follow §12.5: successful items committed; failed items reported per-index
* `test_upsert_rejects_vector_namespace_mismatch` - Vector-namespace mismatch validation
* `test_upsert_respects_max_batch_size_if_published` - Batch size limit enforcement
* `test_upsert_text_not_supported_when_text_storage_strategy_none` - Text storage strategy validation

### `test_query_basic.py`

**Specification:** §9.3, §9.2 - Query Operations
**Status:** ✅ Complete (12 tests)

Validates query contract:

* `test_query_query_returns_vector_matches` - Returns list of `VectorMatch` instances
* `test_query_validates_dimensions` - Dimension mismatches raise `DimensionMismatch`
* `test_query_top_k_must_be_positive` - `top_k` must be > 0
* `test_query_respects_max_top_k` - `top_k` bounded by `capabilities.max_top_k`
* `test_query_results_sorted_by_score_desc` - Results sorted descending by score
* `test_query_include_flags_respected` - `include_vectors` / `include_metadata` respected
* `test_query_include_vectors_false_returns_list_type` - `include_vectors` flag type validation
* `test_query_include_metadata_false_allows_none_or_empty` - `include_metadata` flag behavior
* `test_query_respects_supports_metadata_filtering_capability` - Metadata filtering capability validation
* `test_query_unknown_namespace_behavior_consistent_with_contract` - Unknown namespace behavior consistency
* `test_query_does_not_require_exact_score_values` - Score value flexibility

### `test_delete_operations.py`

**Specification:** §9.3, §12.5 - Delete Operations
**Status:** ✅ Complete (8 tests)

Validates delete contract:

* `test_delete_delete_by_ids_returns_counts` - Delete by IDs returns counts
* `test_delete_delete_by_filter_returns_counts` - Delete by filter returns counts
* `test_delete_requires_ids_or_filter` - Requires IDs or filter
* `test_delete_idempotent_for_missing_ids` - Deleting non-existent IDs succeeds (idempotent)
* `test_delete_delete_result_structure` - Result includes `deleted_count`, `failed_count`, `failures`
* `test_delete_filter_not_supported_raises_notsupported_if_capability_false` - Filter support capability validation
* `test_delete_batch_ids_respects_supports_batch_operations` - Batch operations support validation
* `test_delete_exceed_max_batch_size_raises_badrequest_when_declared` - Batch size limit validation

### `test_filtering_semantics.py`

**Specification:** §9.3 - Metadata Filtering
**Status:** ✅ Complete (7 tests)

Validates filtering behavior:

* `test_filtering_query_filter_equality` - Basic equality filters work in queries
* `test_filtering_delete_filter_equality` - Equality filters work in deletes
* `test_filtering_filter_requires_mapping_type` - Filter type validation
* `test_filtering_filter_respects_capabilities_support` - Filter capability validation
* `test_filtering_filter_empty_results_ok` - Empty results are valid and correctly encoded
* `test_filtering_unknown_operator_rejected_or_accepted_consistently` - Unknown operator handling
* `test_filtering_filter_complexity_enforced_if_caps_max_filter_terms_declared` - Filter complexity enforcement

### `test_dimension_validation.py`

**Specification:** §9.5, §12.4 - Vector-Specific Errors
**Status:** ✅ Complete (6 tests)

Validates dimension checking and error semantics:

* `test_dimension_validation_dimension_mismatch_on_upsert` - Upsert reports `DimensionMismatch` per-item
* `test_dimension_validation_dimension_mismatch_on_query` - Query raises `DimensionMismatch` on bad query vector
* `test_dimension_validation_dimension_mismatch_error_attributes` - Error includes expected/actual dimensions
* `test_dimension_validation_dimension_mismatch_non_retryable` - `DimensionMismatch` is non-retryable (no `retry_after_ms`)
* `test_dimension_validation_exact_namespace_dimension_mismatch` - Namespace-specific dimension validation
* `test_dimension_validation_dimension_mismatch_asdict_is_json_serializable` - Error JSON serializability

### `test_deadline_enforcement.py`

**Specification:** §6.1, §12.1, §12.4 - Deadline Semantics
**Status:** ✅ Complete (5 tests)

Validates deadline behavior:

* `test_deadline_deadline_budget_nonnegative` - Budget computation never negative
* `test_deadline_deadline_exceeded_on_expired_budget` - `DeadlineExceeded` on expired budget
* `test_deadline_preflight_deadline_check_on_upsert` - Pre-flight validation of `deadline_ms`
* `test_deadline_query_respects_deadline_mid_operation` - Query checks and enforces deadline during execution

### `test_error_mapping_retryable.py`

**Specification:** §6.3, §9.5, §12.1, §12.4 - Error Handling
**Status:** ✅ Complete (12 tests)

Validates error classification and mapping to the shared taxonomy:

* `test_error_handling_retryable_errors_with_hints` - Retryable errors with hints
* `test_error_handling_index_not_ready_retryable` - Index not ready retryable validation
* `test_error_handling_dimension_mismatch_non_retryable_flag` - Dimension mismatch non-retryable flag
* `test_error_handling_error_has_siem_safe_details` - Error SIEM safety
* `test_error_handling_retry_after_preserved_when_raised_resource_exhausted` - Retry after preservation
* `test_error_handling_retry_after_preserved_when_raised_unavailable` - Unavailable retry after
* `test_error_handling_retry_after_preserved_when_raised_index_not_ready` - Index not ready retry after
* `test_error_handling_retry_after_preserved_when_raised_transient_network` - Transient network retry after
* `test_error_handling_bad_request_on_invalid_top_k` - Bad request on invalid top_k
* `test_error_handling_retry_after_field_exists_on_adapter_errors` - Retry after field existence
* `test_error_handling_upsert_bad_request_message_siem_safe` - Upsert error SIEM safety
* `test_wire_retry_after_propagates_in_error_envelope` - Wire retry after propagation

### `test_health_report.py`

**Specification:** §9.3, §6.4 - Health Endpoint
**Status:** ✅ Complete (6 tests)

Validates health endpoint contract:

* `test_health_health_returns_required_fields` - Returns `ok`, `server`, `version`
* `test_health_health_includes_namespaces` - Namespaces dictionary present (Vector-specific)
* `test_health_status_ok_bool` - Status/flags use valid and documented forms
* `test_health_shape_consistent_on_error` - Shape remains consistent on degraded/error states
* `test_health_identity_fields_stable` - Identity fields stability
* `test_health_identity_fields_nonempty_strings` - Identity fields non-empty validation

### `test_context_siem.py`

**Specification:** §13.1-§13.3, §15, §6.1 - Observability & Privacy
**Status:** ✅ Complete (6 tests) ⭐ Critical

Validates SIEM-safe observability:

* `test_observability_context_propagates_to_metrics_siem_safe` - Context propagated without leaking PII
* `test_observability_tenant_hashed_never_raw` - Tenant identifiers hashed; never logged raw
* `test_observability_no_vector_data_in_metrics` - No raw vectors in metrics/logs
* `test_observability_metrics_emitted_on_error_path` - Error paths still respect privacy rules
* `test_observability_query_metrics_include_namespace` - Namespace attached as low-cardinality tag
* `test_observability_upsert_metrics_include_vector_count` - Upsert metrics include aggregate counts only

### `test_batch_size_limits.py`

**Specification:** §9.3, §12.5 - Batch Size & Partial Failures
**Status:** ✅ Complete (6 tests)

Validates batch size and partial-failure behavior:

* `test_batch_limits_upsert_respects_max_batch_size` - Enforces `capabilities.max_batch_size`
* `test_batch_limits_batch_size_exceeded_includes_suggestion` - Oversized batches include `suggested_batch_reduction`
* `test_batch_limits_partial_failure_reporting_shape` - Partial failure reporting shape validation
* `test_batch_limits_batch_operations_atomic_per_vector` - Per-vector atomicity: one item's failure does not corrupt others
* `test_batch_limits_delete_respects_max_batch_size_or_supports_batch_operations` - Delete batch size validation
* `test_batch_limits_batch_query_respects_supports_batch_queries` - Batch query capability validation

### `test_wire_handler.py`

**Specification:** §4.1, §4.1.6, §6.1, §6.3, §9.3, §11.2, §13 - Wire Envelopes & Routing
**Status:** ✅ Complete (13 tests)

Validates wire-level contract and mapping:

* `test_wire_contract_capabilities_success_envelope` - `vector.capabilities` success envelope shape and protocol identity
* `test_wire_contract_query_roundtrip_and_context_plumbing` - `vector.query` envelope, ctx → `OperationContext` plumbing, and result mapping
* `test_wire_contract_upsert_delete_namespace_health_envelopes` - `vector.upsert` / `vector.delete` / namespace ops / `vector.health` success envelopes
* `test_wire_contract_delete_namespace_operation` - Delete namespace operation validation
* `test_wire_contract_unknown_op_maps_to_not_supported` - Unknown `op` mapped to `NotSupported` with normalized error envelope
* `test_wire_contract_maps_vector_adapter_error_to_normalized_envelope` - Adapter error envelopes include normalized `code`, `error`, and human-readable `message`
* `test_wire_contract_maps_unexpected_exception_to_unavailable` - Unexpected exceptions mapped to `UNAVAILABLE` per common taxonomy
* `test_wire_contract_missing_or_invalid_op_maps_to_bad_request` - Missing or invalid `op` mapped to `BadRequest` with normalized error envelope
* `test_wire_contract_maps_not_supported_adapter_error` - Adapter-raised `NotSupported` mapped to `NOT_SUPPORTED` wire code
* `test_wire_contract_error_envelope_includes_message_and_type` - Error envelope message and type validation
* `test_wire_contract_query_missing_required_fields_maps_to_bad_request` - `vector.query` with missing required fields mapped to `BadRequest`
* `test_wire_strict_requires_ctx_and_args_keys` - Strict mode validation
* `test_wire_strict_ctx_and_args_must_be_objects` - Context and args object validation
* `test_wire_query_include_flags_type_validation` - Query include flags type validation
* `test_wire_error_envelope_has_required_fields` - Error envelope required fields validation

---

## Specification Mapping

### §9.3 Operations - Complete Coverage

#### create_namespace()

| Requirement               | Test File                    | Status |
| ------------------------- | ---------------------------- | ------ |
| Returns success result    | test_namespace_operations.py | ✅      |
| Validates dimensions > 0  | test_namespace_operations.py | ✅      |
| Validates distance metric | test_namespace_operations.py | ✅      |
| Namespace isolation       | test_namespace_operations.py | ✅      |
| Index management flag     | test_namespace_operations.py | ✅      |

#### delete_namespace()

| Requirement         | Test File                    | Status |
| ------------------- | ---------------------------- | ------ |
| Idempotent deletion | test_namespace_operations.py | ✅      |
| Namespace support   | test_namespace_operations.py | ✅      |

#### upsert()

| Requirement                        | Test File                                       | Status |
| ---------------------------------- | ----------------------------------------------- | ------ |
| Returns result with counts         | test_upsert_basic.py                            | ✅      |
| Validates dimensions               | test_upsert_basic.py                            | ✅      |
| Validates namespace                | test_upsert_basic.py                            | ✅      |
| Per-item failure reporting (§12.5) | test_upsert_basic.py, test_batch_size_limits.py | ✅      |
| Respects max_batch_size            | test_batch_size_limits.py                       | ✅      |
| Text storage strategy              | test_upsert_basic.py                            | ✅      |
| Namespace-vector match             | test_upsert_basic.py                            | ✅      |

#### query()

| Requirement                       | Test File                                         | Status |
| --------------------------------- | ------------------------------------------------- | ------ |
| Returns `VectorMatch` list        | test_query_basic.py                               | ✅      |
| Validates dimensions              | test_query_basic.py, test_dimension_validation.py | ✅      |
| `top_k > 0`                       | test_query_basic.py                               | ✅      |
| `top_k ≤ max_top_k`               | test_query_basic.py                               | ✅      |
| Sorted by score (desc)            | test_query_basic.py                               | ✅      |
| `include_vectors` flag respected  | test_query_basic.py                               | ✅      |
| `include_metadata` flag respected | test_query_basic.py                               | ✅      |
| Metadata filtering behavior       | test_filtering_semantics.py                       | ✅      |
| Pre-search filtering semantics    | test_filtering_semantics.py                       | ✅      |
| Deadline enforcement              | test_deadline_enforcement.py                      | ✅      |
| Batch query support               | test_batch_size_limits.py                         | ✅      |
| Namespace behavior                | test_query_basic.py                               | ✅      |

#### delete()

| Requirement         | Test File                 | Status |
| ------------------- | ------------------------- | ------ |
| Delete by IDs       | test_delete_operations.py | ✅      |
| Delete by filter    | test_delete_operations.py | ✅      |
| Returns counts      | test_delete_operations.py | ✅      |
| Idempotent          | test_delete_operations.py | ✅      |
| Validates namespace | test_delete_operations.py | ✅      |
| Batch operations    | test_delete_operations.py | ✅      |
| Filter support      | test_delete_operations.py | ✅      |

#### health()

| Requirement                | Test File             | Status |
| -------------------------- | --------------------- | ------ |
| Returns dict               | test_health_report.py | ✅      |
| Contains ok (bool)         | test_health_report.py | ✅      |
| Contains server (str)      | test_health_report.py | ✅      |
| Contains version (str)     | test_health_report.py | ✅      |
| Contains namespaces (dict) | test_health_report.py | ✅      |
| Identity fields stable     | test_health_report.py | ✅      |
| Shape consistency          | test_health_report.py | ✅      |

### §9.2 Capabilities - Complete Coverage

| Requirement                  | Test File                  | Status |
| ---------------------------- | -------------------------- | ------ |
| Returns `VectorCapabilities` | test_capabilities_shape.py | ✅      |
| Identity fields non-empty    | test_capabilities_shape.py | ✅      |
| `supported_metrics` tuple    | test_capabilities_shape.py | ✅      |
| Resource limits valid        | test_capabilities_shape.py | ✅      |
| All feature flags boolean    | test_capabilities_shape.py | ✅      |
| Idempotent calls             | test_capabilities_shape.py | ✅      |
| All fields present           | test_capabilities_shape.py | ✅      |
| Text storage strategy        | test_capabilities_shape.py | ✅      |
| Batch queries flag           | test_capabilities_shape.py | ✅      |

### §9.4 Distance Metrics - Complete Coverage

| Requirement              | Test File                    | Status |
| ------------------------ | ---------------------------- | ------ |
| Known metrics advertised | test_capabilities_shape.py   | ✅      |
| Unknown metrics rejected | test_namespace_operations.py | ✅      |

### §9.5 Vector-Specific Errors - Complete Coverage

| Error Type        | Semantics                                                            | Test File                                                     | Status |
| ----------------- | -------------------------------------------------------------------- | ------------------------------------------------------------- | ------ |
| DimensionMismatch | Raised on dimension mismatch; **non-retryable**; no `retry_after_ms` | test_dimension_validation.py, test_error_mapping_retryable.py | ✅      |
| IndexNotReady     | Retryable; may include `retry_after_ms` hint                         | test_error_mapping_retryable.py                               | ✅      |

### §12 Error Handling & Partial Failures - Complete Coverage

| Requirement                                | Test File                                                                          | Status |
| ------------------------------------------ | ---------------------------------------------------------------------------------- | ------ |
| `BadRequest` on invalid parameters         | test_query_basic.py, test_namespace_operations.py, test_error_mapping_retryable.py | ✅      |
| `NotSupported` on unknown metrics/features | test_namespace_operations.py                                                       | ✅      |
| `ResourceExhausted` classification         | test_error_mapping_retryable.py                                                    | ✅      |
| `Unavailable` classification               | test_error_mapping_retryable.py                                                    | ✅      |
| `DeadlineExceeded` mapping                 | test_deadline_enforcement.py                                                       | ✅      |
| `DimensionMismatch` (non-retryable)        | test_dimension_validation.py, test_error_mapping_retryable.py                      | ✅      |
| `IndexNotReady` (retryable)                | test_error_mapping_retryable.py                                                    | ✅      |
| Partial failures per §12.5                 | test_upsert_basic.py, test_batch_size_limits.py                                    | ✅      |
| Retry after preservation                   | test_error_mapping_retryable.py                                                    | ✅      |
| SIEM-safe error details                    | test_error_mapping_retryable.py                                                    | ✅      |

### §13 Observability - Complete Coverage

| Requirement                            | Test File            | Status |
| -------------------------------------- | -------------------- | ------ |
| Tenant never logged raw                | test_context_siem.py | ✅      |
| Tenant hashed in metrics               | test_context_siem.py | ✅      |
| No vector content in metrics           | test_context_siem.py | ✅      |
| Metrics on error path                  | test_context_siem.py | ✅      |
| Namespace tagged in metrics            | test_context_siem.py | ✅      |
| Vector counts as low-cardinality stats | test_context_siem.py | ✅      |

### §15 Privacy - Complete Coverage

| Requirement             | Test File            | Status |
| ----------------------- | -------------------- | ------ |
| No PII in telemetry     | test_context_siem.py | ✅      |
| Hash tenant identifiers | test_context_siem.py | ✅      |
| No raw vectors in logs  | test_context_siem.py | ✅      |

### §6.1 Context & Deadlines - Complete Coverage

| Requirement             | Test File                    | Status |
| ----------------------- | ---------------------------- | ------ |
| Budget computation      | test_deadline_enforcement.py | ✅      |
| Pre-flight validation   | test_deadline_enforcement.py | ✅      |
| Operation timeout       | test_deadline_enforcement.py | ✅      |
| Query respects deadline | test_deadline_enforcement.py | ✅      |

---

## Running Tests

### All Vector conformance tests
```bash
CORPUS_ADAPTER=tests.mock.mock_vector_adapter:MockVectorAdapter pytest tests/vector/ -v
```

### By category
```bash
# Core operations & namespaces
CORPUS_ADAPTER=tests.mock.mock_vector_adapter:MockVectorAdapter pytest \
  tests/vector/test_namespace_operations.py \
  tests/vector/test_upsert_basic.py \
  tests/vector/test_query_basic.py \
  tests/vector/test_delete_operations.py \
  tests/vector/test_health_report.py -v

# Validation & filtering
CORPUS_ADAPTER=tests.mock.mock_vector_adapter:MockVectorAdapter pytest \
  tests/vector/test_dimension_validation.py \
  tests/vector/test_filtering_semantics.py \
  tests/vector/test_batch_size_limits.py -v

# Infrastructure & observability
CORPUS_ADAPTER=tests.mock.mock_vector_adapter:MockVectorAdapter pytest \
  tests/vector/test_capabilities_shape.py \
  tests/vector/test_deadline_enforcement.py \
  tests/vector/test_context_siem.py -v

# Error handling
CORPUS_ADAPTER=tests.mock.mock_vector_adapter:MockVectorAdapter pytest \
  tests/vector/test_error_mapping_retryable.py -v

# Wire handler
CORPUS_ADAPTER=tests.mock.mock_vector_adapter:MockVectorAdapter pytest \
  tests/vector/test_wire_handler.py -v
```

### With coverage report
```bash
CORPUS_ADAPTER=tests.mock.mock_vector_adapter:MockVectorAdapter \
  pytest tests/vector/ --cov=corpus_sdk.vector --cov-report=html
```

### Adapter-Agnostic Usage
To validate a **third-party** or custom Vector Protocol implementation:

1. Implement the Vector Protocol V1.0 interface as defined in `SPECIFICATION.md §9`
2. Provide a small adapter/fixture that binds these tests to your implementation
3. Run the full `tests/vector/` suite
4. If all 108 tests pass unmodified, you can accurately claim:
   **"Vector Protocol V1.0 - 100% Conformant (Corpus Reference Suite)"**

---

## Adapter Compliance Checklist

Use this checklist when implementing or validating a new Vector adapter:

### ✅ Phase 1: Core Operations (12/12)
* [x] `capabilities()` returns valid `VectorCapabilities` with all fields
* [x] `create_namespace()` validates dimensions and metrics with index management flag
* [x] `delete_namespace()` is idempotent with namespace support validation
* [x] `upsert()` returns result with `upserted_count` / `failed_count`
* [x] `upsert()` validates dimensions per-item
* [x] `upsert()` reports per-item failures with proper shape
* [x] `upsert()` respects text storage strategy
* [x] `query()` returns sorted `VectorMatch` list
* [x] `query()` validates dimensions and namespace
* [x] `query()` enforces `top_k > 0` and respects `max_top_k`
* [x] `query()` supports `include_vectors` and `include_metadata` flags
* [x] `delete()` by IDs and filter works with proper counts

### ✅ Phase 2: Validation & Filtering (15/15)
* [x] Dimensions validated on upsert and query
* [x] Distance metric validated against capabilities
* [x] `top_k` validated positive and against max
* [x] Namespace existence checked and behavior consistent
* [x] Metadata filters work in queries with capability validation
* [x] Metadata filters work in deletes with proper support
* [x] Pre-search filtering semantics honored/documented
* [x] Batch size limits enforced with suggestions
* [x] Empty filter results handled correctly
* [x] Unknown operator handling consistent
* [x] Filter complexity enforced if declared
* [x] Namespace-vector mismatch validation
* [x] Batch query capability validation
* [x] Text storage strategy validation
* [x] Namespace support flag validation

### ✅ Phase 3: Error Handling (12/12)
* [x] `BadRequest` on invalid parameters with SIEM-safe messages
* [x] `NotSupported` on unknown metrics/features
* [x] `ResourceExhausted` classified correctly with retry_after
* [x] `Unavailable` classified correctly with retry_after
* [x] `DeadlineExceeded` on timeout
* [x] `DimensionMismatch` on upsert and query (non-retryable)
* [x] `IndexNotReady` is retryable with retry_after
* [x] Per-item failures reported with indices (§12.5)
* [x] Batch size exceeded includes suggestion
* [x] Error details include relevant context (namespace, dimensions)
* [x] Retry after preserved across error types
* [x] Wire retry after propagation

### ✅ Phase 4: Observability & Privacy (6/6)
* [x] Never logs raw tenant IDs
* [x] Hashes tenant in metrics
* [x] No vector content in metrics/logs
* [x] Emits metrics on error paths
* [x] Namespace tagged in metrics
* [x] Vector count included as safe aggregates

### ✅ Phase 5: Deadline Enforcement (5/5)
* [x] Budget computation correct
* [x] Pre-flight deadline check
* [x] Operations respect deadline
* [x] Queries respect deadline mid-operation
* [x] Deadline exceeded error proper

### ✅ Phase 6: Wire Contract (13/13)
* [x] `WireVectorHandler` implements all `vector.*` ops
* [x] Success envelopes have correct `{ok, code, ms, result}` shape
* [x] Error envelopes normalize to `{ok=false, code, error, message, details}`
* [x] OperationContext properly constructed from wire `ctx`
* [x] Unknown fields ignored in requests
* [x] Unknown operations map to `NotSupported`
* [x] Unexpected exceptions map to `UNAVAILABLE`
* [x] Missing required fields handled with `BAD_REQUEST`
* [x] Strict mode validation works
* [x] Query include flags type validation
* [x] Error envelope required fields validation
* [x] Delete namespace operation validation
* [x] NotSupported error propagation

---

## Conformance Badge

```text
✅ Vector Protocol V1.0 - 100% Conformant
   108/108 tests passing (13 test files)

   ✅ Core Operations: 12/12 (100%)
   ✅ Capabilities: 9/9 (100%)
   ✅ Namespace Management: 10/10 (100%)
   ✅ Upsert Operations: 8/8 (100%)
   ✅ Query Operations: 12/12 (100%)
   ✅ Delete Operations: 8/8 (100%)
   ✅ Filtering Semantics: 7/7 (100%)
   ✅ Dimension Validation: 6/6 (100%)
   ✅ Error Handling: 12/12 (100%)
   ✅ Deadline Semantics: 5/5 (100%)
   ✅ Health Endpoint: 6/6 (100%)
   ✅ Observability & Privacy: 6/6 (100%)
   ✅ Batch Size Limits: 6/6 (100%)
   ✅ Wire Envelopes & Routing: 13/13 (100%)

   Status: Production Ready 🏆 Gold Certified
```

**Badge Suggestion:**
[![Corpus Vector Protocol](https://img.shields.io/badge/CorpusVector%20Protocol-100%25%20Conformant-brightgreen)](./vector_conformance_report.json)

**Last Updated:** 2026-01-19  
**Maintained By:** Corpus SDK Team  
**Status:** 100% V1.0 Conformant - Production Ready (108/108 tests)

---
