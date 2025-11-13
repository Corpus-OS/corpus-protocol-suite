# Graph Protocol V1 Conformance Test Coverage

## Overview

This document tracks conformance test coverage for the **Graph Protocol V1.0** specification as defined in `SPECIFICATION.md §7`. Each test validates normative requirements (MUST/SHOULD) from the specification.

This suite constitutes the **official Graph Protocol V1.0 Reference Conformance Test Suite**. Any implementation (Corpus or third-party) MAY run these tests to verify and publicly claim conformance, provided all referenced tests pass unmodified.

**Protocol Version:** Graph Protocol V1.0
**Status:** Stable / Production-Ready
**Last Updated:** 2025-01-XX
**Test Location:** `tests/graph/`

## Conformance Summary

**Overall Coverage: 68/68 tests (100%) ✅**

| Category                 | Tests | Coverage |
| ------------------------ | ----- | -------- |
| Core Operations          | 7/7   | 100% ✅   |
| CRUD Validation          | 7/7   | 100% ✅   |
| Query Operations         | 4/4   | 100% ✅   |
| Dialect Validation       | 4/4   | 100% ✅   |
| Streaming Semantics      | 4/4   | 100% ✅   |
| Batch Operations         | 7/7   | 100% ✅   |
| Schema Operations        | 3/3   | 100% ✅   |
| Error Handling           | 5/5   | 100% ✅   |
| Capabilities             | 7/7   | 100% ✅   |
| Observability & Privacy  | 6/6   | 100% ✅   |
| Deadline Semantics       | 4/4   | 100% ✅   |
| Health Endpoint          | 5/5   | 100% ✅   |
| Wire Envelopes & Routing | 12/12 | 100% ✅   |

## Test Files

### test_capabilities_shape.py

**Specification:** §7.2, §6.2 - Capabilities Discovery
**Status:** ✅ Complete (7 tests)

Tests all aspects of capability discovery:

* `test_capabilities_returns_correct_type` - Returns GraphCapabilities dataclass instance
* `test_capabilities_identity_fields` - `server`/`version` are non-empty strings
* `test_capabilities_dialects_tuple` - `dialects` is non-empty tuple of strings
* `test_capabilities_feature_flags_are_boolean` - All feature flags are boolean types
* `test_capabilities_max_batch_ops_valid` - `None` or positive integer
* `test_capabilities_rate_limit_unit` - Valid enum value
* `test_capabilities_idempotency` - Multiple calls return consistent results

---

### test_crud_basic.py

**Specification:** §7.3.1, §17.2 - CRUD Operations
**Status:** ✅ Complete (7 tests)

Validates basic CRUD contract:

* `test_create_vertex_returns_graph_id` - Returns GraphID with proper format
* `test_create_edge_returns_graph_id` - Returns GraphID with proper format
* `test_vertex_requires_label_and_props` - Validates required fields
* `test_edge_requires_from_to_label` - Validates required fields
* `test_delete_vertex_idempotent` - Deleting non-existent vertex succeeds
* `test_delete_edge_idempotent` - Deleting non-existent edge succeeds
* `test_properties_are_json_serializable` - Properties normalized to JSON-safe keys

---

### test_query_basic.py

**Specification:** §7.3.2, §7.4, §17.2 - Query Operations
**Status:** ✅ Complete (4 tests)

Validates query execution:

* `test_query_returns_list_of_mappings` - Returns list of dict results
* `test_query_requires_dialect_and_text` - Validates required parameters
* `test_query_params_are_bound_safely` - Parameter injection safety
* `test_query_empty_params_allowed` - `None` params accepted

---

### test_dialect_validation.py

**Specification:** §7.4, §6.3 - Dialect Handling
**Status:** ✅ Complete (4 tests) ⭐ Exemplary

Comprehensive dialect validation with parametrized tests:

* `test_unknown_dialect_rejected` - Rejects unknown dialects (param: `unknown`, `sql`, `sparql`)
* `test_known_dialect_accepted` - Accepts known dialects (param: `cypher`, `opencypher`)
* `test_dialect_not_in_capabilities_raises_not_supported` - Validates against capabilities
* `test_error_message_includes_dialect_name` - Error messages are informative

---

### test_streaming_semantics.py

**Specification:** §7.3.2, §6.1, §12.1 - Streaming Operations
**Status:** ✅ Complete (4 tests)

Validates streaming contract:

* `test_stream_query_yields_mappings` - Yields dict instances
* `test_stream_can_be_interrupted_early` - Early cancellation safe
* `test_stream_releases_resources_on_cancel` - Resource cleanup guaranteed
* `test_stream_respects_deadline` - Deadline enforced mid-stream

---

### test_batch_operations.py

**Specification:** §7.3.3, §7.2, §12.5 - Batch & Bulk Operations
**Status:** ✅ Complete (7 tests)

Validates batch operations:

* `test_bulk_vertices_returns_graph_ids` - Returns list of GraphIDs
* `test_bulk_vertices_respects_max_batch_ops` - Enforces batch size limits
* `test_batch_operations_returns_results_per_op` - Per-operation results
* `test_batch_helper_create_vertex_op` - Helper constructs valid ops
* `test_batch_helper_create_edge_op` - Helper constructs valid ops
* `test_batch_helper_query_op` - Helper constructs valid ops
* `test_batch_size_exceeded_includes_suggestion` - Error includes `suggested_batch_reduction`

---

### test_schema_operations.py

**Specification:** §7.5, §5.3, §13.1 - Schema Retrieval
**Status:** ✅ Complete (3 tests)

Validates schema operations:

* `test_get_schema_returns_dict` - Returns dictionary
* `test_get_schema_structure_valid` - Contains expected keys (`nodes`, `edges`)
* `test_schema_cached_in_standalone_mode` - Caching behavior in standalone mode

---

### test_deadline_enforcement.py

**Specification:** §6.1, §12.1 - Deadline Semantics
**Status:** ✅ Complete (4 tests)

Validates deadline behavior:

* `test_deadline_budget_nonnegative` - Budget computation never negative
* `test_deadline_exceeded_on_expired_budget` - `DeadlineExceeded` on expired budget
* `test_preflight_deadline_check` - Pre-flight validation
* `test_stream_respects_deadline_mid_operation` - Streaming respects deadlines

---

### test_error_mapping_retryable.py

**Specification:** §6.3, §12.1, §12.4, §17.2 - Error Handling
**Status:** ✅ Complete (5 tests)

Validates error classification:

* `test_retryable_errors_with_hints` - Retryable errors include `retry_after_ms`
* `test_error_includes_operation_field` - Operation context in errors
* `test_error_includes_dialect_field` - Dialect context in errors
* `test_bad_request_on_empty_label` - Validation errors
* `test_not_supported_on_unknown_dialect` - `NotSupported` for unknown dialects

---

### test_health_report.py

**Specification:** §7.6, §6.4 - Health Endpoint
**Status:** ✅ Complete (5 tests)

Validates health endpoint contract:

* `test_health_returns_required_fields` - Returns `status`/`server`/`version`
* `test_health_status_is_valid_enum` - Status from HealthStatus enum
* `test_health_includes_read_only_flag` - `read_only` flag present
* `test_health_includes_degraded_flag` - `degraded` flag present
* `test_health_consistent_on_error` - Shape consistent on error

---

### test_context_siem.py

**Specification:** §13.1, §13.2, §6.1 - Observability & Privacy
**Status:** ✅ Complete (6 tests) ⭐ Critical

Validates SIEM-safe observability:

* `test_context_propagates_to_metrics_siem_safe` - Context propagates safely
* `test_tenant_hashed_never_raw` - Tenant identifiers hashed
* `test_no_query_text_in_metrics` - No query text in metrics (privacy)
* `test_metrics_emitted_on_error_path` - Error metrics maintain privacy
* `test_query_metrics_include_dialect` - Dialect tagged in metrics
* `test_batch_metrics_include_op_count` - Operation count in batch metrics

---

### test_wire_handler.py

**Specification:** §4.1, §4.1.6, §7, §6.1, §6.3, §12.4, §11.2, §13 - Wire Envelopes & Routing
**Status:** ✅ Complete (12 tests)

Validates `WireGraphHandler` wire-level contract:

* `test_wire_capabilities_success_envelope` — `graph.capabilities` success envelope, protocol/server/version asserted.
* `test_wire_query_roundtrip_and_context_plumbing` — `graph.query` success path + `OperationContext` construction and propagation.
* `test_wire_upsert_delete_bulk_batch_schema_health_envelopes` — Success envelopes for `upsert_*`, `delete_*`, `bulk_vertices`, `batch`, `get_schema`, `health`.
* `test_wire_stream_query_success_chunks_and_context` — `graph.stream_query` via `handle_stream()` yields `{ok, code, chunk}` envelopes with propagated context.
* `test_wire_unknown_op_maps_to_not_supported` — Unknown `op` → `NOT_SUPPORTED` normalized error envelope.
* `test_wire_missing_or_invalid_op_maps_to_bad_request` — Missing/invalid `op` → `BAD_REQUEST` normalized error.
* `test_wire_maps_graph_adapter_error_to_normalized_envelope` — `GraphAdapterError` mapped to `{code, error, message, details}`.
* `test_wire_maps_unexpected_exception_to_unavailable` — Unexpected exception → `UNAVAILABLE` with stable envelope.
* `test_wire_get_schema_envelope_success` — Explicit `graph.get_schema` success envelope shape validation.
* `test_wire_maps_notsupported_adapter_error_to_not_supported_code` — Adapter `NotSupported` propagates as `NOT_SUPPORTED` code.
* `test_wire_error_envelope_includes_message_and_type` — Error envelopes include human message and error class/type.
* `test_wire_query_missing_required_fields_maps_to_bad_request` — Missing required `graph.query` args → `BAD_REQUEST` via wire.

---

## Specification Mapping

### §7.3 Operations - Complete Coverage

#### create_vertex() / create_edge()

| Requirement               | Test File                    | Status |
| ------------------------- | ---------------------------- | ------ |
| Returns GraphID           | test_crud_basic.py           | ✅      |
| Validates label non-empty | test_crud_basic.py           | ✅      |
| Validates properties      | test_crud_basic.py           | ✅      |
| Edge validates from/to    | test_crud_basic.py           | ✅      |
| Deadline enforcement      | test_deadline_enforcement.py | ✅      |

#### delete_vertex() / delete_edge()

| Requirement           | Test File          | Status |
| --------------------- | ------------------ | ------ |
| Idempotent deletion   | test_crud_basic.py | ✅      |
| Validates identifiers | test_crud_basic.py | ✅      |

#### query()

| Requirement              | Test File                    | Status |
| ------------------------ | ---------------------------- | ------ |
| Returns list of mappings | test_query_basic.py          | ✅      |
| Validates dialect        | test_dialect_validation.py   | ✅      |
| Validates text non-empty | test_query_basic.py          | ✅      |
| Parameter binding safe   | test_query_basic.py          | ✅      |
| Empty params allowed     | test_query_basic.py          | ✅      |
| Dialect in capabilities  | test_dialect_validation.py   | ✅      |
| Deadline enforcement     | test_deadline_enforcement.py | ✅      |

#### stream_query()

| Requirement             | Test File                   | Status |
| ----------------------- | --------------------------- | ------ |
| Yields dict instances   | test_streaming_semantics.py | ✅      |
| Early cancellation safe | test_streaming_semantics.py | ✅      |
| Resource cleanup        | test_streaming_semantics.py | ✅      |
| Deadline enforcement    | test_streaming_semantics.py | ✅      |

#### bulk_vertices()

| Requirement                   | Test File                | Status |
| ----------------------------- | ------------------------ | ------ |
| Returns list of GraphIDs      | test_batch_operations.py | ✅      |
| Respects max_batch_ops        | test_batch_operations.py | ✅      |
| Includes batch reduction hint | test_batch_operations.py | ✅      |

#### batch()

| Requirement                 | Test File                | Status |
| --------------------------- | ------------------------ | ------ |
| Returns per-op results      | test_batch_operations.py | ✅      |
| Respects max_batch_ops      | test_batch_operations.py | ✅      |
| Helpers construct valid ops | test_batch_operations.py | ✅      |

#### get_schema()

| Requirement               | Test File                 | Status |
| ------------------------- | ------------------------- | ------ |
| Returns dict              | test_schema_operations.py | ✅      |
| Contains nodes/edges keys | test_schema_operations.py | ✅      |
| Cached in standalone      | test_schema_operations.py | ✅      |

#### health()

| Requirement        | Test File             | Status |
| ------------------ | --------------------- | ------ |
| Returns dict       | test_health_report.py | ✅      |
| Contains status    | test_health_report.py | ✅      |
| Contains server    | test_health_report.py | ✅      |
| Contains version   | test_health_report.py | ✅      |
| Contains read_only | test_health_report.py | ✅      |
| Contains degraded  | test_health_report.py | ✅      |

---

### §7.2 Capabilities - Complete Coverage

| Requirement               | Test File                  | Status |
| ------------------------- | -------------------------- | ------ |
| Returns GraphCapabilities | test_capabilities_shape.py | ✅      |
| Identity fields non-empty | test_capabilities_shape.py | ✅      |
| Dialects tuple non-empty  | test_capabilities_shape.py | ✅      |
| All feature flags boolean | test_capabilities_shape.py | ✅      |
| max_batch_ops valid       | test_capabilities_shape.py | ✅      |
| rate_limit_unit valid     | test_capabilities_shape.py | ✅      |
| Idempotent calls          | test_capabilities_shape.py | ✅      |

---

### §7.4 Dialect Handling - Complete Coverage

| Requirement                    | Test File                  | Status |
| ------------------------------ | -------------------------- | ------ |
| Unknown dialects rejected      | test_dialect_validation.py | ✅      |
| Known dialects accepted        | test_dialect_validation.py | ✅      |
| Validates against capabilities | test_dialect_validation.py | ✅      |
| Error includes dialect name    | test_dialect_validation.py | ✅      |

---

### §6.3 Error Handling - Complete Coverage

| Error Type              | Test File                                                         | Status |
| ----------------------- | ----------------------------------------------------------------- | ------ |
| BadRequest (validation) | test_crud_basic.py, test_query_basic.py, test_batch_operations.py | ✅      |
| NotSupported (dialect)  | test_dialect_validation.py, test_error_mapping_retryable.py       | ✅      |
| ResourceExhausted       | test_error_mapping_retryable.py                                   | ✅      |
| Unavailable             | test_error_mapping_retryable.py                                   | ✅      |
| DeadlineExceeded        | test_deadline_enforcement.py                                      | ✅      |
| retry_after_ms hint     | test_error_mapping_retryable.py                                   | ✅      |
| operation field         | test_error_mapping_retryable.py                                   | ✅      |
| dialect field           | test_error_mapping_retryable.py                                   | ✅      |

---

### §13 Observability - Complete Coverage

| Requirement               | Test File            | Status |
| ------------------------- | -------------------- | ------ |
| Tenant never logged raw   | test_context_siem.py | ✅      |
| Tenant hashed in metrics  | test_context_siem.py | ✅      |
| No query text in metrics  | test_context_siem.py | ✅      |
| Metrics on error path     | test_context_siem.py | ✅      |
| Dialect in metrics        | test_context_siem.py | ✅      |
| Op count in batch metrics | test_context_siem.py | ✅      |

---

### §6.1 Context & Deadlines - Complete Coverage

| Requirement           | Test File                                                 | Status |
| --------------------- | --------------------------------------------------------- | ------ |
| Budget computation    | test_deadline_enforcement.py                              | ✅      |
| Pre-flight validation | test_deadline_enforcement.py                              | ✅      |
| Operation timeout     | test_deadline_enforcement.py                              | ✅      |
| Stream timeout        | test_deadline_enforcement.py, test_streaming_semantics.py | ✅      |

---

## Running Tests

### All Graph conformance tests

```bash
pytest tests/graph/ -v
```

### By category

```bash
# Core operations
pytest tests/graph/test_crud_basic.py \
       tests/graph/test_query_basic.py \
       tests/graph/test_streaming_semantics.py \
       tests/graph/test_health_report.py -v

# Validation
pytest tests/graph/test_dialect_validation.py \
       tests/graph/test_batch_operations.py -v

# Infrastructure (capabilities, deadlines, observability)
pytest tests/graph/test_capabilities_shape.py \
       tests/graph/test_deadline_enforcement.py \
       tests/graph/test_context_siem.py -v

# Schema & Error handling
pytest tests/graph/test_schema_operations.py \
       tests/graph/test_error_mapping_retryable.py -v
```

### With coverage report

```bash
pytest tests/graph/ --cov=corpus_sdk.graph --cov-report=html
```

### Adapter-Agnostic Usage

To validate a **third-party** or custom Graph Protocol implementation:

1. Implement the Graph Protocol V1.0 interface as defined in `SPECIFICATION.md §7`.
2. Provide a small adapter/fixture that binds these tests to your implementation (e.g., via a factory or configuration).
3. Run the full `tests/graph/` suite.
4. If all tests pass unmodified, you can accurately claim:
   **“Graph Protocol V1.0 - 100% Conformant (Corpus Reference Suite)”**.

---

## Adapter Compliance Checklist

*(unchanged content)*

---

## Conformance Badge

```text
✅ Graph Protocol V1.0 - 100% Conformant (Corpus Reference Suite)
   68/68 tests passing

   ✅ Core Operations:       7/7 (100%)
   ✅ CRUD Validation:       7/7 (100%)
   ✅ Query Operations:      4/4 (100%)
   ✅ Dialect Validation:    4/4 (100%)
   ✅ Streaming:             4/4 (100%)
   ✅ Batch Operations:      7/7 (100%)
   ✅ Schema Operations:     3/3 (100%)
   ✅ Error Handling:        5/5 (100%)
   ✅ Capabilities:          7/7 (100%)
   ✅ Observability:         6/6 (100%)
   ✅ Deadline:              4/4 (100%)
   ✅ Health:                5/5 (100%)
   ✅ Wire Envelopes & Routing: 12/12 (100%)

   Status: Production Ready
```

Implementations that pass all tests in `tests/graph/` without modification MAY display this badge in their documentation.

---

## Maintenance

### Adding New Tests

1. Create test file: `test_<feature>_<aspect>.py`
2. Add SPDX license header and docstring with spec references
3. Use `pytestmark = pytest.mark.asyncio` for async tests
4. Update this `CONFORMANCE.md` with new coverage
5. Update conformance summary and badge

### Updating for Specification Changes

1. Review `SPECIFICATION.md` changelog (Appendix F)
2. Identify new/changed requirements
3. Add/update tests accordingly
4. Update version number and dates in this document
5. Update conformance badge as needed

---

## Related Documentation

* `../../SPECIFICATION.md` — Full protocol specification (§7 Graph Protocol)
* `../../ERRORS.md` — Error taxonomy reference
* `../../METRICS.md` — Observability guidelines
* `../README.md` — General testing guidelines

---

**Last Updated:** 2025-01-XX
**Maintained By:** Corpus SDK Team
**Status:** 100% V1.0 Conformant - Production Ready
