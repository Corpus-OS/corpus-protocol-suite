# Vector Protocol Conformance Test Coverage

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

This document tracks conformance test coverage for the **Vector Protocol V1.0** specification as defined in `SPECIFICATION.md §9`. Each test validates normative requirements (MUST/SHOULD) from the specification and shared behavior from the common foundation (errors, deadlines, observability, privacy).

This suite constitutes the **official Vector Protocol V1.0 Reference Conformance Test Suite**. Any implementation (Corpus or third-party) MAY run these tests to verify and publicly claim conformance, provided all referenced tests pass unmodified.

**Protocol Version:** Vector Protocol V1.0
**Status:** Pre-Release
**Last Updated:** 2025-01-XX
**Test Location:** `tests/vector/`

## Conformance Summary

**Overall Coverage: 72/72 tests (100%) ✅**

| Category                 | Tests | Coverage |
| ------------------------ | ----- | -------- |
| Core Operations          | 7/7   | 100% ✅   |
| Capabilities             | 7/7   | 100% ✅   |
| Namespace Management     | 6/6   | 100% ✅   |
| Upsert Operations        | 5/5   | 100% ✅   |
| Query Operations         | 6/6   | 100% ✅   |
| Delete Operations        | 5/5   | 100% ✅   |
| Filtering Semantics      | 5/5   | 100% ✅   |
| Dimension Validation     | 4/4   | 100% ✅   |
| Error Handling           | 6/6   | 100% ✅   |
| Deadline Semantics       | 4/4   | 100% ✅   |
| Health Endpoint          | 4/4   | 100% ✅   |
| Observability & Privacy  | 6/6   | 100% ✅   |
| Batch Size Limits        | 4/4   | 100% ✅   |
| Wire Envelopes & Routing | 10/10 | 100% ✅   |

---

## Test Files

### `test_capabilities_shape.py`

**Specification:** §9.2, §6.2 - Capabilities Discovery
**Status:** ✅ Complete (7 tests)

Tests all aspects of capability discovery:

* `test_capabilities_returns_correct_type` - Returns `VectorCapabilities` dataclass instance
* `test_capabilities_identity_fields` - `server` / `version` are non-empty strings
* `test_capabilities_supported_metrics` - Non-empty tuple with valid distance metrics
* `test_capabilities_resource_limits` - `max_dimensions`, `max_top_k`, `max_batch_size` are positive or `None`
* `test_capabilities_feature_flags_are_boolean` - All feature flags are booleans
* `test_capabilities_idempotency` - Multiple calls return consistent results
* `test_capabilities_all_fields_present` - All required fields present and valid

### `test_namespace_operations.py`

**Specification:** §9.3, §9.4 - Namespace Management
**Status:** ✅ Complete (6 tests)

Validates namespace lifecycle:

* `test_create_namespace_returns_success` - Namespace creation succeeds with valid spec
* `test_namespace_requires_dimensions` - Dimensions must be positive
* `test_namespace_requires_valid_distance_metric` - Metric validated against §9.4 and capabilities
* `test_list_namespaces_returns_dict` - Health/namespaces exposure returns a dictionary
* `test_delete_namespace_idempotent` - Deleting a non-existent namespace succeeds (idempotent)
* `test_namespace_isolation` - Vectors in different namespaces are isolated

### `test_upsert_basic.py`

**Specification:** §9.3, §9.5, §12.5 - Upsert Operations & Partial Failures
**Status:** ✅ Complete (5 tests)

Validates upsert contract and partial-failure semantics:

* `test_upsert_returns_result_with_counts` - Returns `upserted_count`, `failed_count`
* `test_upsert_validates_dimensions` - Per-item dimension mismatches reported explicitly
* `test_upsert_validates_namespace_exists` - Unknown namespace handled via validation or explicit semantics
* `test_upsert_result_structure` - Result shape includes failures list when applicable
* `test_upsert_partial_failure_reporting` - Partial failures follow §12.5: successful items committed; failed items reported per-index

### `test_query_basic.py`

**Specification:** §9.3, §9.2 - Query Operations
**Status:** ✅ Complete (6 tests)

Validates query contract:

* `test_query_returns_vector_matches` - Returns list of `VectorMatch` instances
* `test_query_validates_dimensions` - Dimension mismatches raise `DimensionMismatch`
* `test_query_top_k_positive` - `top_k` must be > 0
* `test_query_top_k_respects_max` - `top_k` bounded by `capabilities.max_top_k`
* `test_query_returns_sorted_by_score` - Results sorted descending by score
* `test_query_include_flags_work` - `include_vectors` / `include_metadata` respected

### `test_delete_operations.py`

**Specification:** §9.3, §12.5 - Delete Operations
**Status:** ✅ Complete (5 tests)

Validates delete contract:

* `test_delete_by_ids_returns_counts` - Delete by IDs returns counts
* `test_delete_by_filter_returns_counts` - Delete by filter returns counts
* `test_delete_validates_namespace_exists` - Namespace validation behavior
* `test_delete_idempotent` - Deleting non-existent IDs succeeds (idempotent)
* `test_delete_result_structure` - Result includes `deleted_count`, `failed_count`, `failures`

### `test_filtering_semantics.py`

**Specification:** §9.3 - Metadata Filtering
**Status:** ✅ Complete (5 tests)

Validates filtering behavior:

* `test_query_filter_equality` - Basic equality filters work in queries
* `test_query_filter_pre_search` - Verifies filters applied before search when supported (or documented behavior)
* `test_delete_filter_equality` - Equality filters work in deletes
* `test_filter_requires_metadata_support` - Enforces `supports_metadata_filtering` from capabilities
* `test_filter_empty_results_ok` - Empty results are valid and correctly encoded

### `test_dimension_validation.py`

**Specification:** §9.5, §12.4 - Vector-Specific Errors
**Status:** ✅ Complete (4 tests)

Validates dimension checking and error semantics:

* `test_dimension_mismatch_on_upsert` - Upsert reports `DimensionMismatch` per-item
* `test_dimension_mismatch_on_query` - Query raises `DimensionMismatch` on bad query vector
* `test_dimension_mismatch_error_attributes` - Error includes expected/actual dimensions
* `test_dimension_mismatch_non_retryable` - `DimensionMismatch` is non-retryable (no `retry_after_ms`)

### `test_deadline_enforcement.py`

**Specification:** §6.1, §12.1, §12.4 - Deadline Semantics
**Status:** ✅ Complete (4 tests)

Validates deadline behavior:

* `test_deadline_budget_nonnegative` - Budget computation never negative
* `test_deadline_exceeded_on_expired_budget` - `DeadlineExceeded` on expired budget
* `test_preflight_deadline_check` - Pre-flight validation of `deadline_ms`
* `test_query_respects_deadline` - Query checks and enforces deadline during execution

### `test_error_mapping_retryable.py`

**Specification:** §6.3, §9.5, §12.1, §12.4 - Error Handling
**Status:** ✅ Complete (6 tests)

Validates error classification and mapping to the shared taxonomy:

* `test_retryable_errors_with_hints`
* `test_error_includes_namespace_field`
* `test_dimension_mismatch_non_retryable`
* `test_bad_request_on_invalid_top_k`
* `test_index_not_ready_retryable`
* (Additional coverage for normalized mapping semantics)

### `test_health_report.py`

**Specification:** §9.3, §6.4 - Health Endpoint
**Status:** ✅ Complete (4 tests)

Validates health endpoint contract:

* `test_health_returns_required_fields` - Returns `ok`, `server`, `version`
* `test_health_includes_namespaces` - Namespaces dictionary present (Vector-specific)
* `test_health_status_is_valid_enum` - Status/flags use valid and documented forms
* `test_health_consistent_on_error` - Shape remains consistent on degraded/error states

### `test_context_siem.py`

**Specification:** §13.1-§13.3, §15, §6.1 - Observability & Privacy
**Status:** ✅ Complete (6 tests) ⭐ Critical

Validates SIEM-safe observability:

* `test_context_propagates_to_metrics_siem_safe` - Context propagated without leaking PII
* `test_tenant_hashed_never_raw` - Tenant identifiers hashed; never logged raw
* `test_no_vector_data_in_metrics` - No raw vectors in metrics/logs
* `test_metrics_emitted_on_error_path` - Error paths still respect privacy rules
* `test_query_metrics_include_namespace` - Namespace attached as low-cardinality tag
* `test_upsert_metrics_include_vector_count` - Upsert metrics include aggregate counts only

### `test_batch_size_limits.py`

**Specification:** §9.3, §12.5 - Batch Size & Partial Failures
**Status:** ✅ Complete (4 tests)

Validates batch size and partial-failure behavior:

* `test_upsert_respects_max_batch_size` - Enforces `capabilities.max_batch_size`
* `test_batch_size_exceeded_includes_suggestion` - Oversized batches include `suggested_batch_reduction`
* `test_partial_failure_reporting` - Per-item failures reported with indices (per §12.5)
* `test_batch_operations_atomic_per_vector` - Per-vector atomicity: one item’s failure does not corrupt others

### `test_wire_handler_envelopes.py`

**Specification:** §4.1, §4.1.6, §6.1, §6.3, §9.3, §11.2, §13 - Wire Envelopes & Routing
**Status:** ✅ Complete (10 tests)

Validates wire-level contract and mapping:

* `vector.capabilities` success envelope shape and protocol identity
* `vector.query` envelope, ctx → `OperationContext` plumbing, and result mapping
* `vector.upsert` / `vector.delete` / namespace ops / `vector.health` success envelopes
* Unknown `op` mapped to `NotSupported` with normalized error envelope
* Missing or invalid `op` mapped to `BadRequest` with normalized error envelope
* Adapter-raised `NotSupported` mapped to `NOT_SUPPORTED` wire code
* Adapter error envelopes include normalized `code`, `error`, and human-readable `message`
* `vector.query` with missing required fields mapped to `BadRequest`
* `VectorAdapterError` mapped to canonical error envelope (`code`, `error`, `message`, `details`)
* Unexpected exceptions mapped to `UNAVAILABLE` per common taxonomy

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

#### delete_namespace()

| Requirement         | Test File                    | Status |
| ------------------- | ---------------------------- | ------ |
| Idempotent deletion | test_namespace_operations.py | ✅      |

#### upsert()

| Requirement                        | Test File                                       | Status |
| ---------------------------------- | ----------------------------------------------- | ------ |
| Returns result with counts         | test_upsert_basic.py                            | ✅      |
| Validates dimensions               | test_upsert_basic.py                            | ✅      |
| Validates namespace                | test_upsert_basic.py                            | ✅      |
| Per-item failure reporting (§12.5) | test_upsert_basic.py, test_batch_size_limits.py | ✅      |
| Respects max_batch_size            | test_batch_size_limits.py                       | ✅      |

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

#### delete()

| Requirement         | Test File                 | Status |
| ------------------- | ------------------------- | ------ |
| Delete by IDs       | test_delete_operations.py | ✅      |
| Delete by filter    | test_delete_operations.py | ✅      |
| Returns counts      | test_delete_operations.py | ✅      |
| Idempotent          | test_delete_operations.py | ✅      |
| Validates namespace | test_delete_operations.py | ✅      |

#### health()

| Requirement                | Test File             | Status |
| -------------------------- | --------------------- | ------ |
| Returns dict               | test_health_report.py | ✅      |
| Contains ok (bool)         | test_health_report.py | ✅      |
| Contains server (str)      | test_health_report.py | ✅      |
| Contains version (str)     | test_health_report.py | ✅      |
| Contains namespaces (dict) | test_health_report.py | ✅      |

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
pytest tests/vector/ -v
```

### By category

```bash
# Core operations & namespaces
pytest tests/vector/test_namespace_operations.py \
       tests/vector/test_upsert_basic.py \
       tests/vector/test_query_basic.py \
       tests/vector/test_delete_operations.py \
       tests/vector/test_health_report.py -v

# Validation & filtering
pytest tests/vector/test_dimension_validation.py \
       tests/vector/test_filtering_semantics.py \
       tests/vector/test_batch_size_limits.py -v

# Infrastructure & observability
pytest tests/vector/test_capabilities_shape.py \
       tests/vector/test_deadline_enforcement.py \
       tests/vector/test_context_siem.py -v

# Error handling
pytest tests/vector/test_error_mapping_retryable.py -v
```

### With coverage report

```bash
pytest tests/vector/ --cov=corpus_sdk.vector --cov-report=html
```

---

## Adapter Compliance Checklist

Use this checklist when implementing or validating a new Vector adapter:

### ✅ Phase 1: Core Operations (17/17)

* [x] `capabilities()` returns valid `VectorCapabilities`
* [x] `create_namespace()` validates dimensions and metrics
* [x] `delete_namespace()` is idempotent
* [x] `upsert()` returns result with `upserted_count` / `failed_count`
* [x] `upsert()` validates dimensions per-item
* [x] `upsert()` reports per-item failures
* [x] `query()` returns sorted `VectorMatch` list
* [x] `query()` validates dimensions
* [x] `query()` enforces `top_k > 0`
* [x] `query()` respects `max_top_k`
* [x] `query()` `include_vectors` flag works
* [x] `query()` `include_metadata` flag works
* [x] `delete()` by IDs works
* [x] `delete()` by filter works
* [x] `delete()` is idempotent
* [x] `health()` returns required fields
* [x] `health()` includes namespaces

### ✅ Phase 2: Validation (11/11)

* [x] Dimensions validated on upsert
* [x] Dimensions validated on query
* [x] Distance metric validated
* [x] `top_k` validated positive
* [x] `top_k` validated against max
* [x] Namespace existence checked or clearly defined
* [x] Metadata filters work (query)
* [x] Metadata filters work (delete)
* [x] Pre-search filtering semantics honored/documented
* [x] Batch size limits enforced
* [x] Empty filter results handled correctly

### ✅ Phase 3: Error Handling (12/12)

* [x] `BadRequest` on invalid parameters
* [x] `NotSupported` on unknown metrics/features
* [x] `ResourceExhausted` classified correctly
* [x] `Unavailable` classified correctly
* [x] `DeadlineExceeded` on timeout
* [x] `DimensionMismatch` on upsert
* [x] `DimensionMismatch` on query
* [x] `DimensionMismatch` is non-retryable
* [x] `IndexNotReady` is retryable
* [x] Per-item failures reported (§12.5)
* [x] Batch size exceeded includes suggestion
* [x] Error details include relevant context (e.g., namespace)

### ✅ Phase 4: Observability (6/6)

* [x] Never logs raw tenant IDs
* [x] Hashes tenant in metrics
* [x] No vector content in metrics/logs
* [x] Emits metrics on error paths
* [x] Namespace tagged in metrics
* [x] Vector count included as safe aggregates

### ✅ Phase 5: Deadline Enforcement (4/4)

* [x] Budget computation correct
* [x] Pre-flight deadline check
* [x] Operations respect deadline
* [x] Queries respect deadline

---

## Conformance Badge

```text
✅ Vector Protocol V1.0 - 100% Conformant
   72/72 tests passing (13 test files)

   ✅ Core Operations: 7/7 (100%)
   ✅ Capabilities: 7/7 (100%)
   ✅ Namespace Management: 6/6 (100%)
   ✅ Upsert Operations: 5/5 (100%)
   ✅ Query Operations: 6/6 (100%)
   ✅ Delete Operations: 5/5 (100%)
   ✅ Filtering Semantics: 5/5 (100%)
   ✅ Dimension Validation: 4/4 (100%)
   ✅ Error Handling: 6/6 (100%)
   ✅ Deadline Semantics: 4/4 (100%)
   ✅ Health Endpoint: 4/4 (100%)
   ✅ Observability & Privacy: 6/6 (100%)
   ✅ Batch Size Limits: 4/4 (100%)
   ✅ Wire Envelopes & Routing: 10/10 (100%)

   Status: Production Ready
```

---

## Maintenance

### Adding New Tests

1. Create test file: `test_<feature>_<aspect>.py`
2. Add SPDX license header and a docstring with relevant spec references
3. Use `pytestmark = pytest.mark.asyncio` for async tests
4. Update this `CONFORMANCE.md` with new coverage details
5. Update the conformance summary and badge

### Updating for Specification Changes

1. Review `SPECIFICATION.md` changelog (Appendix F)
2. Identify new or changed requirements for §9 / shared sections
3. Add or update tests accordingly
4. Update protocol version and date in this document
5. Update the conformance badge and mapping tables

## Related Documentation

* `../../SPECIFICATION.md` - Full protocol specification (§9 Vector Protocol)
* `../../ERRORS.md` - Error taxonomy reference
* `../../METRICS.md` - Observability guidelines
* `../README.md` - General testing guidelines

---

**Last Updated:** 2025-01-XX
**Maintained By:** Corpus SDK Team
**Status:** 100% V1.0 Conformant - Production Ready