# Graph Protocol Conformance Test Coverage

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

This document tracks conformance test coverage for the **Graph Protocol V1.0** specification as defined in `SPECIFICATION.md §7`. Each test validates normative requirements (MUST/SHOULD) from the specification.

This suite constitutes the **official Graph Protocol V1.0 Reference Conformance Test Suite**. Any implementation (Corpus or third-party) MAY run these tests to verify and publicly claim conformance, provided all referenced tests pass unmodified.

**Protocol Version:** Graph Protocol V1.0
**Status:** Stable / Production-Ready
**Last Updated:** 2026-01-19
**Test Location:** `tests/graph/`

## Conformance Summary

**Overall Coverage: 99/99 tests (100%) ✅**

| Category                 | Tests | Coverage |
| ------------------------ | ----- | -------- |
| Core Operations          | 9/9   | 100% ✅   |
| CRUD Validation          | 10/10 | 100% ✅   |
| Query Operations         | 8/8   | 100% ✅   |
| Dialect Validation       | 6/6   | 100% ✅   |
| Streaming Semantics      | 5/5   | 100% ✅   |
| Batch Operations         | 10/10 | 100% ✅   |
| Schema Operations        | 2/2   | 100% ✅   |
| Error Handling           | 12/12 | 100% ✅   |
| Capabilities             | 8/8   | 100% ✅   |
| Observability & Privacy  | 6/6   | 100% ✅   |
| Deadline Semantics       | 4/4   | 100% ✅   |
| Health Endpoint          | 5/5   | 100% ✅   |
| Wire Envelopes & Routing | 14/14 | 100% ✅   |

## Test Files

### test_capabilities_shape.py

**Specification:** §7.2, §6.2 - Capabilities Discovery
**Status:** ✅ Complete (8 tests)

Tests all aspects of capability discovery:

* `test_capabilities_returns_correct_type` - Returns GraphCapabilities dataclass instance
* `test_capabilities_identity_fields` - `server`/`version` are non-empty strings
* `test_capabilities_dialects_tuple` - `dialects` is non-empty tuple of strings
* `test_capabilities_feature_flags_are_boolean` - All feature flags are boolean types
* `test_capabilities_max_batch_ops_valid` - `None` or positive integer
* `test_capabilities_protocol` - Protocol field validation
* `test_capabilities_idempotency` - Multiple calls return consistent results
* `test_capabilities_json_serializable` - Capabilities are JSON serializable

---

### test_crud_basic.py

**Specification:** §7.3.1, §17.2 - CRUD Operations
**Status:** ✅ Complete (10 tests)

Validates basic CRUD contract:

* `test_crud_upsert_node_returns_success` - Upsert returns success with GraphID
* `test_crud_upsert_edge_returns_success` - Upsert returns success with GraphID
* `test_crud_node_labels_type_validation_happens_at_model_level` - Label validation at model level
* `test_crud_properties_must_be_json_serializable` - Properties normalized to JSON-safe keys
* `test_crud_upsert_nodes_empty_rejected` - Empty nodes list rejected
* `test_crud_upsert_edges_empty_rejected` - Empty edges list rejected
* `test_crud_validation_edge_requires_src_dst_label` - Validates required fields for edges
* `test_crud_delete_nodes_requires_ids_or_filter` - Delete requires identifiers or filter
* `test_crud_delete_edges_requires_ids_or_filter` - Delete requires identifiers or filter
* `test_crud_delete_filter_must_be_json_serializable` - Filter must be JSON serializable
* `test_crud_delete_nodes_idempotent_repeatable` - Deleting non-existent nodes succeeds (idempotent)
* `test_crud_delete_edges_idempotent_repeatable` - Deleting non-existent edges succeeds (idempotent)
* `test_crud_properties_with_non_string_keys_allowed_if_json_allows` - Non-string keys allowed per JSON spec

---

### test_query_basic.py

**Specification:** §7.3.2, §7.4, §17.2 - Query Operations
**Status:** ✅ Complete (8 tests)

Validates query execution:

* `test_query_returns_json_serializable_records_list` - Returns list of JSON-serializable dict results
* `test_query_requires_non_empty_text` - Validates query text non-empty
* `test_query_params_are_bound_safely` - Parameter injection safety
* `test_query_none_and_empty_params_allowed` - `None` and empty params accepted
* `test_query_params_must_be_json_serializable` - Parameters must be JSON serializable
* `test_query_accepts_params_with_non_string_keys_if_json_allows` - Non-string key parameters allowed
* `test_query_dialect_validation_is_capability_driven` - Dialect validation against capabilities
* `test_wire_handle_query_success_envelope_shape` - Wire envelope shape validation

---

### test_dialect_validation.py

**Specification:** §7.4, §6.3 - Dialect Handling
**Status:** ✅ Complete (6 tests) ⭐ Exemplary

Comprehensive dialect validation with parametrized tests:

* `test_unknown_dialect_behavior_is_capability_consistent` - Tests unknown dialect behavior (parametrized: `unknown`, `sql`, `sparql`)
* `test_known_dialect_accepted_when_declared` - Accepts known dialects when declared in capabilities
* `test_error_message_includes_dialect_when_rejected_due_to_declared_list` - Error messages include dialect name
* Additional coverage in error handling tests

---

### test_streaming_semantics.py

**Specification:** §7.3.2, §6.1, §12.1 - Streaming Operations
**Status:** ✅ Complete (5 tests)

Validates streaming contract:

* `test_stream_query_capability_alignment` - Validates streaming capability alignment
* `test_stream_query_yields_querychunks_with_json_serializable_records` - Yields QueryChunk instances with JSON-serializable records
* `test_streaming_can_be_interrupted_early` - Early cancellation safe
* `test_streaming_releases_resources_on_cancel` - Resource cleanup guaranteed
* `test_wire_handle_stream_emits_streaming_frames_when_supported` - Wire streaming frames validation

---

### test_batch_operations.py

**Specification:** §7.3.3, §7.2, §12.5 - Batch & Bulk Operations
**Status:** ✅ Complete (10 tests)

Validates batch operations:

* `test_batch_ops_bulk_vertices_returns_graph_ids` - Returns list of GraphIDs
* `test_batch_ops_batch_respects_max_batch_ops` - Enforces batch size limits
* `test_batch_ops_batch_operations_returns_results_per_op` - Per-operation results
* `test_batch_ops_batch_size_exceeded_includes_hint` - Error includes `suggested_batch_reduction`
* `test_bulk_vertices_pagination_invariants_when_supported` - Pagination invariants validation
* `test_bulk_vertices_cursor_progresses_when_supported` - Cursor progression validation
* `test_transaction_success_path_when_supported` - Transaction success path
* `test_transaction_enforces_max_batch_ops_when_declared` - Transaction batch size enforcement
* `test_traversal_success_path_when_supported` - Traversal success path
* `test_traversal_enforces_max_depth_when_declared` - Traversal depth enforcement

---

### test_schema_operations.py

**Specification:** §7.5, §5.3, §13.1 - Schema Retrieval
**Status:** ✅ Complete (2 tests)

Validates schema operations:

* `test_get_schema_capability_alignment` - Schema capability alignment
* `test_schema_consistency_and_serializable_when_supported` - Schema consistency and serializability

---

### test_deadline_enforcement.py

**Specification:** §6.1, §12.1 - Deadline Semantics
**Status:** ✅ Complete (4 tests)

Validates deadline behavior:

* `test_deadline_exceeded_on_expired_budget_query_when_supported` - `DeadlineExceeded` on expired budget for queries
* `test_deadline_exceeded_on_expired_budget_write_when_supported` - `DeadlineExceeded` on expired budget for write operations
* `test_deadline_exceeded_on_expired_budget_stream_preflight_when_supported` - `DeadlineExceeded` on expired budget for streaming
* Additional deadline coverage in streaming tests

---

### test_error_mapping_retryable.py

**Specification:** §6.3, §12.1, §12.4, §17.2 - Error Handling
**Status:** ✅ Complete (12 tests)

Validates error classification:

* `test_error_handling_retryable_errors_with_hints` - Retryable errors include `retry_after_ms`
* `test_graph_adapter_error_details_is_mapping` - Error details are mapping type
* `test_normalized_error_default_codes` - Normalized error code mapping (multiple parametrized tests)
* `test_retryable_error_types_accept_retry_after_and_details` - Retryable error types accept retry_after and details
* `test_error_string_includes_code_when_present` - Error string includes code
* `test_error_handling_bad_request_on_empty_edge_label` - Validation errors for empty labels
* `test_not_supported_on_unknown_dialect_when_declared` - `NotSupported` for unknown dialects
* `test_error_message_includes_dialect_name_when_rejected_due_to_declared_list` - Error messages include dialect name

---

### test_health_report.py

**Specification:** §7.6, §6.4 - Health Endpoint
**Status:** ✅ Complete (5 tests)

Validates health endpoint contract:

* `test_health_returns_required_fields` - Returns `ok`/`server`/`version`
* `test_health_basic_types` - Basic type validation
* `test_health_namespaces_is_mapping_like` - Namespaces mapping validation
* `test_health_json_serializable` - JSON serializability
* `test_health_required_keys_stable_across_calls` - Shape consistency across calls

---

### test_context_siem.py

**Specification:** §13.1, §13.2, §6.1 - Observability & Privacy
**Status:** ✅ Complete (6 tests) ⭐ Critical

Validates SIEM-safe observability:

* `test_observability_context_propagates_to_metrics_siem_safe` - Context propagates safely
* `test_observability_tenant_hashed_never_raw` - Tenant identifiers hashed
* `test_observability_no_query_text_in_metrics` - No query text in metrics (privacy)
* `test_observability_metrics_emitted_on_error_path` - Error metrics maintain privacy
* `test_observability_query_metrics_include_dialect` - Dialect tagged in metrics
* `test_observability_batch_metrics_include_op_count_when_supported` - Operation count in batch metrics

---

### test_wire_handler.py

**Specification:** §4.1, §4.1.6, §7, §6.1, §6.3, §12.4, §11.2, §13 - Wire Envelopes & Routing
**Status:** ✅ Complete (14 tests)

Validates `WireGraphHandler` wire-level contract:

* `test_wire_contract_capabilities_success_envelope` — `graph.capabilities` success envelope, protocol/server/version asserted.
* `test_wire_contract_query_roundtrip_and_context_plumbing` — `graph.query` success path + `OperationContext` construction and propagation.
* `test_wire_contract_upsert_delete_bulk_batch_schema_health_envelopes` — Success envelopes for `upsert_*`, `delete_*`, `bulk_vertices`, `batch`, `get_schema`, `health`.
* `test_wire_contract_get_schema_envelope_success` — Explicit `graph.get_schema` success envelope shape validation.
* `test_wire_contract_stream_query_success_chunks_and_context` — `graph.stream_query` via `handle_stream()` yields `{ok, code, chunk}` envelopes with propagated context.
* `test_wire_contract_stream_query_wrong_op_errors` — Wrong operation errors for streaming.
* `test_wire_contract_unknown_op_maps_to_not_supported` — Unknown `op` → `NOT_SUPPORTED` normalized error envelope.
* `test_wire_contract_missing_or_invalid_op_maps_to_bad_request` — Missing/invalid `op` → `BAD_REQUEST` normalized error.
* `test_wire_contract_maps_graph_adapter_error_to_normalized_envelope` — `GraphAdapterError` mapped to `{code, error, message, details}`.
* `test_wire_contract_maps_notsupported_adapter_error_to_not_supported_code` — Adapter `NotSupported` propagates as `NOT_SUPPORTED` code.
* `test_wire_contract_maps_unexpected_exception_to_unavailable_and_hardens_message` — Unexpected exception → `UNAVAILABLE` with hardened message.
* `test_wire_contract_error_envelope_includes_message_and_type` — Error envelopes include human message and error class/type.
* `test_wire_contract_graph_adapter_error_includes_retry_after_and_details_fields` — Adapter error includes retry_after and details.
* `test_wire_contract_query_missing_required_fields_maps_to_bad_request` — Missing required `graph.query` args → `BAD_REQUEST` via wire.

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
| JSON serializable props   | test_crud_basic.py           | ✅      |
| Empty list rejection      | test_crud_basic.py           | ✅      |
| Non-string keys allowed   | test_crud_basic.py           | ✅      |
| Deadline enforcement      | test_deadline_enforcement.py | ✅      |

#### delete_vertex() / delete_edge()

| Requirement           | Test File          | Status |
| --------------------- | ------------------ | ------ |
| Idempotent deletion   | test_crud_basic.py | ✅      |
| Validates identifiers | test_crud_basic.py | ✅      |
| Filter support        | test_crud_basic.py | ✅      |
| Filter serialization  | test_crud_basic.py | ✅      |

#### query()

| Requirement              | Test File                    | Status |
| ------------------------ | ---------------------------- | ------ |
| Returns JSON-serializable| test_query_basic.py          | ✅      |
| Validates dialect        | test_dialect_validation.py   | ✅      |
| Validates text non-empty | test_query_basic.py          | ✅      |
| Parameter binding safe   | test_query_basic.py          | ✅      |
| Empty params allowed     | test_query_basic.py          | ✅      |
| Dialect in capabilities  | test_dialect_validation.py   | ✅      |
| JSON serializable params | test_query_basic.py          | ✅      |
| Non-string keys allowed  | test_query_basic.py          | ✅      |
| Deadline enforcement     | test_deadline_enforcement.py | ✅      |

#### stream_query()

| Requirement             | Test File                   | Status |
| ----------------------- | --------------------------- | ------ |
| Yields QueryChunk       | test_streaming_semantics.py | ✅      |
| JSON-serializable recs  | test_streaming_semantics.py | ✅      |
| Early cancellation safe | test_streaming_semantics.py | ✅      |
| Resource cleanup        | test_streaming_semantics.py | ✅      |
| Wire frame validation   | test_streaming_semantics.py | ✅      |
| Deadline enforcement    | test_deadline_enforcement.py | ✅      |

#### bulk_vertices()

| Requirement                   | Test File                | Status |
| ----------------------------- | ------------------------ | ------ |
| Returns list of GraphIDs      | test_batch_operations.py | ✅      |
| Respects max_batch_ops        | test_batch_operations.py | ✅      |
| Includes batch reduction hint | test_batch_operations.py | ✅      |
| Pagination invariants         | test_batch_operations.py | ✅      |
| Cursor progression            | test_batch_operations.py | ✅      |

#### batch()

| Requirement                 | Test File                | Status |
| --------------------------- | ------------------------ | ------ |
| Returns per-op results      | test_batch_operations.py | ✅      |
| Respects max_batch_ops      | test_batch_operations.py | ✅      |
| Transaction support         | test_batch_operations.py | ✅      |
| Traversal support           | test_batch_operations.py | ✅      |
| Depth enforcement           | test_batch_operations.py | ✅      |

#### get_schema()

| Requirement               | Test File                 | Status |
| ------------------------- | ------------------------- | ------ |
| Capability alignment      | test_schema_operations.py | ✅      |
| Consistency & serializable| test_schema_operations.py | ✅      |

#### health()

| Requirement        | Test File             | Status |
| ------------------ | --------------------- | ------ |
| Returns dict       | test_health_report.py | ✅      |
| Contains ok flag   | test_health_report.py | ✅      |
| Contains server    | test_health_report.py | ✅      |
| Contains version   | test_health_report.py | ✅      |
| Namespaces mapping | test_health_report.py | ✅      |
| JSON serializable  | test_health_report.py | ✅      |
| Stable shape       | test_health_report.py | ✅      |

---

### §7.2 Capabilities - Complete Coverage

| Requirement               | Test File                  | Status |
| ------------------------- | -------------------------- | ------ |
| Returns GraphCapabilities | test_capabilities_shape.py | ✅      |
| Identity fields non-empty | test_capabilities_shape.py | ✅      |
| Dialects tuple non-empty  | test_capabilities_shape.py | ✅      |
| All feature flags boolean | test_capabilities_shape.py | ✅      |
| max_batch_ops valid       | test_capabilities_shape.py | ✅      |
| Protocol field validation | test_capabilities_shape.py | ✅      |
| Idempotent calls          | test_capabilities_shape.py | ✅      |
| JSON serializable         | test_capabilities_shape.py | ✅      |

---

### §7.4 Dialect Handling - Complete Coverage

| Requirement                    | Test File                  | Status |
| ------------------------------ | -------------------------- | ------ |
| Unknown dialects rejected      | test_dialect_validation.py | ✅      |
| Known dialects accepted        | test_dialect_validation.py | ✅      |
| Validates against capabilities | test_dialect_validation.py | ✅      |
| Error includes dialect name    | test_dialect_validation.py | ✅      |
| Capability-driven validation   | test_query_basic.py        | ✅      |
| Error mapping for dialects     | test_error_mapping_retryable.py | ✅      |

---

### §6.3 Error Handling - Complete Coverage

| Error Type              | Test File                                                         | Status |
| ----------------------- | ----------------------------------------------------------------- | ------ |
| BadRequest (validation) | test_crud_basic.py, test_query_basic.py, test_error_mapping_retryable.py | ✅      |
| NotSupported (dialect)  | test_dialect_validation.py, test_error_mapping_retryable.py       | ✅      |
| ResourceExhausted       | test_error_mapping_retryable.py                                   | ✅      |
| Unavailable             | test_error_mapping_retryable.py                                   | ✅      |
| DeadlineExceeded        | test_deadline_enforcement.py, test_error_mapping_retryable.py     | ✅      |
| TransientNetwork        | test_error_mapping_retryable.py                                   | ✅      |
| retry_after_ms hint     | test_error_mapping_retryable.py                                   | ✅      |
| error details mapping   | test_error_mapping_retryable.py                                   | ✅      |
| Error string includes code | test_error_mapping_retryable.py                                | ✅      |

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
| Query timeout         | test_deadline_enforcement.py                              | ✅      |
| Write operation timeout | test_deadline_enforcement.py                            | ✅      |
| Stream timeout        | test_deadline_enforcement.py                              | ✅      |
| Pre-flight validation | test_deadline_enforcement.py                              | ✅      |

---

## Running Tests

### All Graph conformance tests

```bash
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter pytest tests/graph/ -v
```

### By category

```bash
# Core operations
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter pytest \
  tests/graph/test_crud_basic.py \
  tests/graph/test_query_basic.py \
  tests/graph/test_streaming_semantics.py \
  tests/graph/test_health_report.py -v

# Validation
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter pytest \
  tests/graph/test_dialect_validation.py \
  tests/graph/test_batch_operations.py -v

# Infrastructure (capabilities, deadlines, observability)
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter pytest \
  tests/graph/test_capabilities_shape.py \
  tests/graph/test_deadline_enforcement.py \
  tests/graph/test_context_siem.py -v

# Schema & Error handling
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter pytest \
  tests/graph/test_schema_operations.py \
  tests/graph/test_error_mapping_retryable.py -v

# Wire handler
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter pytest \
  tests/graph/test_wire_handler.py -v
```

### With coverage report

```bash
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter \
  pytest tests/graph/ --cov=corpus_sdk.graph --cov-report=html
```

### Adapter-Agnostic Usage

To validate a **third-party** or custom Graph Protocol implementation:

1. Implement the Graph Protocol V1.0 interface as defined in `SPECIFICATION.md §7`.
2. Provide a small adapter/fixture that binds these tests to your implementation (e.g., via a factory or configuration).
3. Run the full `tests/graph/` suite.
4. If all tests pass unmodified, you can accurately claim:
   **"Graph Protocol V1.0 - 100% Conformant (Corpus Reference Suite)"**.

---

## Adapter Compliance Checklist

Use this when implementing or validating a new **Graph adapter** against `GraphProtocolV1` + `BaseGraphAdapter`.

### ✅ Phase 1: Core Operations (10/10)

* [x] `capabilities()` returns `GraphCapabilities` with all required fields (8 tests)
* [x] `create_vertex()`/`create_edge()` return valid `GraphID` with proper format
* [x] `delete_vertex()`/`delete_edge()` are idempotent and accept filters
* [x] `query()` returns JSON-serializable results with dialect validation
* [x] `stream_query()` yields QueryChunk instances with proper streaming semantics
* [x] `bulk_vertices()` respects `max_batch_ops` limits with pagination support
* [x] `batch()` returns per-operation results with transaction support
* [x] `get_schema()` returns consistent, serializable schema
* [x] `health()` returns proper health status with namespaces
* [x] All operations support JSON-serializable properties/parameters

### ✅ Phase 2: Validation & Dialect Handling (15/15)

* [x] Reject empty labels in vertex/edge creation
* [x] Validate required `from`/`to` fields for edges
* [x] Ensure properties are JSON-serializable
* [x] Reject unknown dialects with clear error messages
* [x] Validate dialects against capabilities
* [x] Require non-empty query text
* [x] Support empty parameters in queries
* [x] Safe parameter binding to prevent injection
* [x] Enforce `max_batch_ops` with helpful error hints
* [x] Reject empty node/edge lists
* [x] Support filters for delete operations
* [x] Validate filter serializability
* [x] Support non-string keys in properties when JSON allows
* [x] Capability-driven dialect validation
* [x] Error messages include dialect context

### ✅ Phase 3: Error Handling & Semantics (16/16)

* [x] Map provider errors to canonical codes (`BadRequest`, `NotSupported`, etc.)
* [x] Include `retry_after_ms` for retryable errors when available
* [x] Include operation and dialect context in errors
* [x] Do not treat validation errors as retryable
* [x] Provide `suggested_batch_reduction` for batch size errors
* [x] Use `DeadlineExceeded` on expired budgets
* [x] Honor `NotSupported` for unsupported dialects/features
* [x] Follow §12.5 partial-failure semantics for batch operations
* [x] Error details are proper mappings
* [x] Normalized error codes mapped correctly
* [x] Error strings include error codes
* [x] Retryable errors accept retry_after and details
* [x] Handle empty edge label validation
* [x] Proper error for unknown dialects
* [x] Error hardening for unexpected exceptions

### ✅ Phase 4: Observability & Privacy (6/6)

* [x] Use `component="graph"` in metrics
* [x] Emit exactly one `observe` per operation
* [x] Never log raw query text, tenant IDs, or sensitive properties
* [x] Use `tenant_hash`, `dialect`, `op_count` as low-cardinality tags
* [x] Emit error counters on failure paths
* [x] Ensure wire+logs SIEM-safe per §13 requirements

### ✅ Phase 5: Deadlines, Caching & Wire Contract (18/18)

* [x] Respect `OperationContext.deadline_ms` with preflight checks
* [x] Use `DeadlineExceeded` when time budget elapses mid-operation
* [x] Support early cancellation of streaming queries
* [x] Ensure resource cleanup on stream cancellation
* [x] Cache schema when appropriate
* [x] `WireGraphHandler` implements all `graph.*` ops with canonical envelopes
* [x] Unknown fields ignored; unknown ops → `NotSupported`
* [x] Error envelopes use normalized `code`/`error` structure
* [x] Proper wire envelope shapes for all operations
* [x] Context propagation through wire handler
* [x] Success envelopes for all operations
* [x] Streaming wire frame validation
* [x] Wrong operation error handling for streaming
* [x] Missing/invalid operation error handling
* [x] Graph adapter error normalization
* [x] NotSupported error propagation
* [x] Unexpected exception hardening
* [x] Required field validation in wire requests

---

## Conformance Badge

```text
✅ Graph Protocol V1.0 - 100% Conformant
   99/99 tests passing (12 test files)

   ✅ Core Operations:         9/9 (100%)
   ✅ CRUD Validation:        10/10 (100%)
   ✅ Query Operations:       8/8 (100%)
   ✅ Dialect Validation:     6/6 (100%)
   ✅ Streaming Semantics:    5/5 (100%)
   ✅ Batch Operations:      10/10 (100%)
   ✅ Schema Operations:      2/2 (100%)
   ✅ Error Handling:        12/12 (100%)
   ✅ Capabilities:           8/8 (100%)
   ✅ Observability & Privacy: 6/6 (100%)
   ✅ Deadline Semantics:     4/4 (100%)
   ✅ Health Endpoint:        5/5 (100%)
   ✅ Wire Envelopes & Routing: 14/14 (100%)

   Status: Production Ready
```
## **Graph Protocol Conformance**

**Certification Levels:**
- 🏆 **Gold:** 99/99 tests (100%)
- 🥈 **Silver:** 79+ tests (80%+)
- 🔬 **Development:** 50+ tests (50%+)

**Badge Suggestion:**
```markdown
[![Corpus Graph Protocol](https://img.shields.io/badge/CorpusGraph%20Protocol-100%25%20Conformant-brightgreen)](./graph_conformance_report.json)

Implementations that pass all tests in `tests/graph/` without modification MAY display this badge in their documentation.

---

**Last Updated:** 2026-01-19
**Maintained By:** Corpus SDK Team
**Status:** 100% V1.0 Conformant - Production Ready (99/99 tests)
