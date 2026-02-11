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

This document tracks conformance test coverage for the **Graph Protocol V1.0** specification as defined in `SPECIFICATION.md §7`. Each test validates normative requirements (MUST/SHOULD) from the specification and shared behavior from the common foundation (errors, deadlines, observability, privacy).

This suite constitutes the **official Graph Protocol V1.0 Reference Conformance Test Suite**. Any implementation (Corpus or third-party) MAY run these tests to verify and publicly claim conformance, provided all referenced tests pass unmodified.

**Protocol Version:** Graph Protocol V1.0  
**Status:** Stable / Production-Ready  
**Last Updated:** 2026-02-10  
**Test Location:** `tests/graph/`  
**Performance:** 0.64s total (6.5ms/test average)

## Conformance Summary

**Overall Coverage: 99/99 tests (100%) ✅**

📊 **Total Tests:** 99/99 passing (100%)  
⚡ **Execution Time:** 0.64s (6.5ms/test avg)  
🏆 **Certification:** Platinum (100%)

| Category | Tests | Coverage | Status |
|----------|-------|-----------|---------|
| **Core Operations** | 9/9 | 100% ✅ | Production Ready |
| **CRUD Validation** | 10/10 | 100% ✅ | Production Ready |
| **Query Operations** | 8/8 | 100% ✅ | Production Ready |
| **Dialect Validation** | 6/6 | 100% ✅ | Production Ready |
| **Streaming Semantics** | 5/5 | 100% ✅ | Production Ready |
| **Batch Operations** | 10/10 | 100% ✅ | Production Ready |
| **Schema Operations** | 2/2 | 100% ✅ | Production Ready |
| **Error Handling** | 12/12 | 100% ✅ | Production Ready |
| **Capabilities** | 8/8 | 100% ✅ | Production Ready |
| **Observability & Privacy** | 6/6 | 100% ✅ | Production Ready |
| **Deadline Semantics** | 4/4 | 100% ✅ | Production Ready |
| **Health Endpoint** | 5/5 | 100% ✅ | Production Ready |
| **Wire Envelopes & Routing** | 14/14 | 100% ✅ | Production Ready |
| **Total** | **99/99** | **100% ✅** | **🏆 Platinum Certified** |

### Performance Characteristics
- **Test Execution:** 0.64 seconds total runtime
- **Average Per Test:** 6.5 milliseconds
- **Cache Efficiency:** 0 cache hits, 99 misses (cache size: 99)
- **Parallel Ready:** Optimized for parallel execution with `pytest -n auto`

### Test Infrastructure
- **Mock Adapter:** `tests.mock.mock_graph_adapter:MockGraphAdapter` - Deterministic mock for Graph operations
- **Testing Framework:** pytest 9.0.2 with comprehensive plugin support
- **Environment:** Python 3.10.19 on Darwin
- **Strict Mode:** Off (permissive testing)

## **Graph Protocol Certification**

- 🏆 **Platinum:** 99/99 tests (100% comprehensive conformance)
- 🥇 **Gold:** 99 tests (100% protocol mastery)
- 🥈 **Silver:** 80+ tests (80%+ integration-ready)
- 🔬 **Development:** 50+ tests (50%+ early development)

---

## Test Files

### `test_capabilities_shape.py`

**Specification:** §7.2 Data Types, §6.2 Capability Discovery  
**Status:** ✅ Complete (8 tests)

Tests all aspects of capability discovery for `graph.capabilities`:

* `test_capabilities_returns_correct_type` - Returns GraphCapabilities dataclass instance (§7.2)
* `test_capabilities_identity_fields` - `server`/`version` are non-empty strings (§6.2)
* `test_capabilities_dialects_tuple` - `dialects` is non-empty tuple of strings (§7.4)
* `test_capabilities_feature_flags_are_boolean` - All feature flags are boolean types (§6.2)
* `test_capabilities_max_batch_ops_valid` - `None` or positive integer (§7.2)
* `test_capabilities_protocol` - Protocol field validation (§4.2.2)
* `test_capabilities_idempotency` - Multiple calls return consistent results (§6.2)
* `test_capabilities_json_serializable` - Capabilities are JSON serializable (§4.2.1)

### `test_crud_basic.py`

**Specification:** §7.3.1 Node/Edge CRUD, §17.2 Validation  
**Status:** ✅ Complete (10 tests)

Validates basic CRUD contract for `graph.upsert_nodes` and `graph.upsert_edges`:

* `test_crud_upsert_node_returns_success` - `graph.upsert_nodes` returns success with GraphID (§7.3.1)
* `test_crud_upsert_edge_returns_success` - `graph.upsert_edges` returns success with GraphID (§7.3.1)
* `test_crud_node_labels_type_validation_happens_at_model_level` - Label validation at model level (§7.3.1)
* `test_crud_properties_must_be_json_serializable` - Properties normalized to JSON-safe keys (§17.2)
* `test_crud_upsert_nodes_empty_rejected` - Empty nodes list rejected for `graph.upsert_nodes` (§17.2)
* `test_crud_upsert_edges_empty_rejected` - Empty edges list rejected for `graph.upsert_edges` (§17.2)
* `test_crud_validation_edge_requires_src_dst_label` - Validates required fields for edges (§7.3.1)
* `test_crud_delete_nodes_requires_ids_or_filter` - `graph.delete_nodes` requires identifiers or filter (§7.3.1)
* `test_crud_delete_edges_requires_ids_or_filter` - `graph.delete_edges` requires identifiers or filter (§7.3.1)
* `test_crud_delete_filter_must_be_json_serializable` - Filter must be JSON serializable (§17.2)
* `test_crud_delete_nodes_idempotent_repeatable` - Deleting non-existent nodes succeeds (idempotent) (§11.4)
* `test_crud_delete_edges_idempotent_repeatable` - Deleting non-existent edges succeeds (idempotent) (§11.4)
* `test_crud_properties_with_non_string_keys_allowed_if_json_allows` - Non-string keys allowed per JSON spec (§4.2.1)

### `test_query_basic.py`

**Specification:** §7.3.2 Queries, §7.4 Dialects, §17.2 Validation  
**Status:** ✅ Complete (8 tests)

Validates query execution for `graph.query`:

* `test_query_returns_json_serializable_records_list` - Returns list of JSON-serializable dict results (§7.3.2)
* `test_query_requires_non_empty_text` - Validates query text non-empty (§17.2)
* `test_query_params_are_bound_safely` - Parameter injection safety (§14.4)
* `test_query_none_and_empty_params_allowed` - `None` and empty params accepted (§7.3.2)
* `test_query_params_must_be_json_serializable` - Parameters must be JSON serializable (§17.2)
* `test_query_accepts_params_with_non_string_keys_if_json_allows` - Non-string key parameters allowed (§4.2.1)
* `test_query_dialect_validation_is_capability_driven` - Dialect validation against capabilities (§7.4)
* `test_wire_handle_query_success_envelope_shape` - Wire envelope shape validation for `graph.query` (§4.2.1)

### `test_dialect_validation.py`

**Specification:** §7.4 Dialects, §6.3 Error Handling  
**Status:** ✅ Complete (6 tests) ⭐ Exemplary

Comprehensive dialect validation with parametrized tests for `graph.query` and `graph.stream_query`:

* `test_unknown_dialect_behavior_is_capability_consistent` - Tests unknown dialect behavior (parametrized: `unknown`, `sql`, `sparql`) (§7.4)
* `test_known_dialect_accepted_when_declared` - Accepts known dialects when declared in capabilities (§7.4)
* `test_error_message_includes_dialect_when_rejected_due_to_declared_list` - Error messages include dialect name (§12.4)
* Additional coverage in error handling tests

### `test_streaming_semantics.py`

**Specification:** §7.3.2 Queries, §4.2.3 Streaming Frames, §6.1 Operation Context  
**Status:** ✅ Complete (5 tests)

Validates streaming contract for `graph.stream_query`:

* `test_stream_query_capability_alignment` - Validates streaming capability alignment (§7.2)
* `test_stream_query_yields_querychunks_with_json_serializable_records` - Yields QueryChunk instances with JSON-serializable records (§4.2.3)
* `test_streaming_can_be_interrupted_early` - Early cancellation safe (§11.5)
* `test_streaming_releases_resources_on_cancel` - Resource cleanup guaranteed (§11.5)
* `test_wire_handle_stream_emits_streaming_frames_when_supported` - Wire streaming frames validation for `graph.stream_query` (§4.2.3)

### `test_batch_operations.py`

**Specification:** §7.3.3 Batch Operations, §7.2 Data Types, §12.5 Partial Failure Contracts  
**Status:** ✅ Complete (10 tests)

Validates batch operations for `graph.bulk_vertices` and `graph.batch`:

* `test_batch_ops_bulk_vertices_returns_graph_ids` - `graph.bulk_vertices` returns list of GraphIDs (§7.3.3)
* `test_batch_ops_batch_respects_max_batch_ops` - `graph.batch` enforces batch size limits (§7.2)
* `test_batch_ops_batch_operations_returns_results_per_op` - `graph.batch` returns per-operation results (§7.3.3)
* `test_batch_ops_batch_size_exceeded_includes_hint` - Error includes `suggested_batch_reduction` (§12.1)
* `test_bulk_vertices_pagination_invariants_when_supported` - Pagination invariants validation for `graph.bulk_vertices` (§11.5)
* `test_bulk_vertices_cursor_progresses_when_supported` - Cursor progression validation for `graph.bulk_vertices` (§11.5)
* `test_transaction_success_path_when_supported` - Transaction success path for `graph.batch` (§7.3.3)
* `test_transaction_enforces_max_batch_ops_when_declared` - Transaction batch size enforcement for `graph.batch` (§7.2)
* `test_traversal_success_path_when_supported` - Traversal success path for `graph.batch` (§7.3.3)
* `test_traversal_enforces_max_depth_when_declared` - Traversal depth enforcement for `graph.batch` (§7.2)

### `test_schema_operations.py`

**Specification:** §7.5 Schema Operations, §5.3 Implementation Profiles, §13.1 Metrics Taxonomy  
**Status:** ✅ Complete (2 tests)

Validates schema operations for `graph.get_schema`:

* `test_get_schema_capability_alignment` - Schema capability alignment (§7.5)
* `test_schema_consistency_and_serializable_when_supported` - Schema consistency and serializability (§7.5)

### `test_deadline_enforcement.py`

**Specification:** §6.1 Operation Context, §12.1 Retry Semantics  
**Status:** ✅ Complete (4 tests)

Validates deadline behavior across all graph operations:

* `test_deadline_exceeded_on_expired_budget_query_when_supported` - `DeadlineExceeded` on expired budget for `graph.query` (§6.1, §12.1)
* `test_deadline_exceeded_on_expired_budget_write_when_supported` - `DeadlineExceeded` on expired budget for `graph.upsert_nodes`/`graph.upsert_edges` (§6.1, §12.1)
* `test_deadline_exceeded_on_expired_budget_stream_preflight_when_supported` - `DeadlineExceeded` on expired budget for `graph.stream_query` (§6.1, §12.1)
* Additional deadline coverage in streaming tests

### `test_error_mapping_retryable.py`

**Specification:** §6.3 Error Taxonomy, §12.1 Retry Semantics, §12.4 Error Mapping Table, §17.2 Validation  
**Status:** ✅ Complete (12 tests)

Validates error classification for all graph operations:

* `test_error_handling_retryable_errors_with_hints` - Retryable errors include `retry_after_ms` (§12.1)
* `test_graph_adapter_error_details_is_mapping` - Error details are mapping type (§6.3)
* `test_normalized_error_default_codes` - Normalized error code mapping (multiple parametrized tests) (§12.4)
* `test_retryable_error_types_accept_retry_after_and_details` - Retryable error types accept retry_after and details (§6.3)
* `test_error_string_includes_code_when_present` - Error string includes code (§12.4)
* `test_error_handling_bad_request_on_empty_edge_label` - Validation errors for empty labels in `graph.upsert_edges` (§17.2)
* `test_not_supported_on_unknown_dialect_when_declared` - `NotSupported` for unknown dialects in `graph.query` (§7.4)
* `test_error_message_includes_dialect_name_when_rejected_due_to_declared_list` - Error messages include dialect name (§12.4)

### `test_health_report.py`

**Specification:** §7.6 Health, §6.4 Observability Interfaces  
**Status:** ✅ Complete (5 tests)

Validates health endpoint contract for `graph.health`:

* `test_health_returns_required_fields` - Returns `ok`/`server`/`version` (§7.6)
* `test_health_basic_types` - Basic type validation (§7.6)
* `test_health_namespaces_is_mapping_like` - Namespaces mapping validation (§7.6)
* `test_health_json_serializable` - JSON serializability (§4.2.1)
* `test_health_required_keys_stable_across_calls` - Shape consistency across calls (§6.4)

### `test_context_siem.py`

**Specification:** §13.1 Metrics Taxonomy, §13.2 Structured Logging, §6.1 Operation Context  
**Status:** ✅ Complete (6 tests) ⭐ Critical

Validates SIEM-safe observability for all graph operations:

* `test_observability_context_propagates_to_metrics_siem_safe` - Context propagates safely (§13.1)
* `test_observability_tenant_hashed_never_raw` - Tenant identifiers hashed (§13.1, §15)
* `test_observability_no_query_text_in_metrics` - No query text in metrics (privacy) (§13.1, §15)
* `test_observability_metrics_emitted_on_error_path` - Error metrics maintain privacy (§13.1)
* `test_observability_query_metrics_include_dialect` - Dialect tagged in metrics (§13.1)
* `test_observability_batch_metrics_include_op_count_when_supported` - Operation count in batch metrics (§13.1)

### `test_wire_handler.py`

**Specification:** §4.2 Wire-First Canonical Form, §4.2.6 Operation Registry, §7 Graph Protocol, §6.1 Operation Context, §6.3 Error Taxonomy, §12.4 Error Mapping Table, §11.2 Consistent Observability, §13 Observability and Monitoring  
**Status:** ✅ Complete (14 tests)

Validates `WireGraphHandler` wire-level contract for all registered operations:

* `test_wire_contract_capabilities_success_envelope` — `graph.capabilities` success envelope, protocol/server/version asserted (§4.2.1)
* `test_wire_contract_query_roundtrip_and_context_plumbing` — `graph.query` success path + `OperationContext` construction and propagation (§6.1)
* `test_wire_contract_upsert_delete_bulk_batch_schema_health_envelopes` — Success envelopes for `graph.upsert_nodes`, `graph.upsert_edges`, `graph.delete_nodes`, `graph.delete_edges`, `graph.bulk_vertices`, `graph.batch`, `graph.get_schema`, `graph.health` (§4.2.1)
* `test_wire_contract_get_schema_envelope_success` — Explicit `graph.get_schema` success envelope shape validation (§4.2.1)
* `test_wire_contract_stream_query_success_chunks_and_context` — `graph.stream_query` via `handle_stream()` yields `{ok, code, chunk}` envelopes with propagated context (§4.2.3)
* `test_wire_contract_stream_query_wrong_op_errors` — Wrong operation errors for streaming (§4.2.6)
* `test_wire_contract_unknown_op_maps_to_not_supported` — Unknown `op` → `NOT_SUPPORTED` normalized error envelope (§4.2.6)
* `test_wire_contract_missing_or_invalid_op_maps_to_bad_request` — Missing/invalid `op` → `BAD_REQUEST` normalized error (§4.2.6)
* `test_wire_contract_requires_ctx_and_args_and_they_must_be_mappings` — Context and args validation (§4.2.1)
* `test_wire_contract_query_missing_required_fields_maps_to_bad_request` — Missing required `graph.query` args → `BAD_REQUEST` via wire (§4.2.1, §17.2)
* `test_wire_contract_maps_graph_adapter_error_to_normalized_envelope` — `GraphAdapterError` mapped to `{code, error, message, details}` (§6.3)
* `test_wire_contract_maps_notsupported_adapter_error_to_not_supported_code` — Adapter `NotSupported` propagates as `NOT_SUPPORTED` code (§6.3)
* `test_wire_contract_error_envelope_includes_message_and_type` — Error envelopes include human message and error class/type (§12.4)
* `test_wire_contract_graph_adapter_error_includes_retry_after_and_details_fields` — Adapter error includes retry_after and details (§6.3)
* `test_wire_contract_maps_unexpected_exception_to_unavailable_and_hardens_message` — Unexpected exception → `UNAVAILABLE` with hardened message (§6.3)

---

## Specification Mapping

### §7.3 Operations - Complete Coverage

#### `graph.upsert_nodes()` / `graph.upsert_edges()` (§7.3.1)

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Returns GraphID | `test_crud_basic.py` | ✅ |
| Validates label non-empty | `test_crud_basic.py` | ✅ |
| Validates properties | `test_crud_basic.py` | ✅ |
| Edge validates from/to | `test_crud_basic.py` | ✅ |
| JSON serializable props | `test_crud_basic.py` | ✅ |
| Empty list rejection | `test_crud_basic.py` | ✅ |
| Non-string keys allowed | `test_crud_basic.py` | ✅ |
| Deadline enforcement | `test_deadline_enforcement.py` | ✅ |

#### `graph.delete_nodes()` / `graph.delete_edges()` (§7.3.1)

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Idempotent deletion | `test_crud_basic.py` | ✅ |
| Validates identifiers | `test_crud_basic.py` | ✅ |
| Filter support | `test_crud_basic.py` | ✅ |
| Filter serialization | `test_crud_basic.py` | ✅ |

#### `graph.query()` (§7.3.2)

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Returns JSON-serializable | `test_query_basic.py` | ✅ |
| Validates dialect | `test_dialect_validation.py` | ✅ |
| Validates text non-empty | `test_query_basic.py` | ✅ |
| Parameter binding safe | `test_query_basic.py` | ✅ |
| Empty params allowed | `test_query_basic.py` | ✅ |
| Dialect in capabilities | `test_dialect_validation.py` | ✅ |
| JSON serializable params | `test_query_basic.py` | ✅ |
| Non-string keys allowed | `test_query_basic.py` | ✅ |
| Deadline enforcement | `test_deadline_enforcement.py` | ✅ |

#### `graph.stream_query()` (§7.3.2, §4.2.3)

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Yields QueryChunk | `test_streaming_semantics.py` | ✅ |
| JSON-serializable recs | `test_streaming_semantics.py` | ✅ |
| Early cancellation safe | `test_streaming_semantics.py` | ✅ |
| Resource cleanup | `test_streaming_semantics.py` | ✅ |
| Wire frame validation | `test_streaming_semantics.py` | ✅ |
| Deadline enforcement | `test_deadline_enforcement.py` | ✅ |

#### `graph.bulk_vertices()` (§7.3.3)

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Returns list of GraphIDs | `test_batch_operations.py` | ✅ |
| Respects max_batch_ops | `test_batch_operations.py` | ✅ |
| Includes batch reduction hint | `test_batch_operations.py` | ✅ |
| Pagination invariants | `test_batch_operations.py` | ✅ |
| Cursor progression | `test_batch_operations.py` | ✅ |

#### `graph.batch()` (§7.3.3)

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Returns per-op results | `test_batch_operations.py` | ✅ |
| Respects max_batch_ops | `test_batch_operations.py` | ✅ |
| Transaction support | `test_batch_operations.py` | ✅ |
| Traversal support | `test_batch_operations.py` | ✅ |
| Depth enforcement | `test_batch_operations.py` | ✅ |

#### `graph.get_schema()` (§7.5)

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Capability alignment | `test_schema_operations.py` | ✅ |
| Consistency & serializable | `test_schema_operations.py` | ✅ |

#### `graph.health()` (§7.6)

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Returns dict | `test_health_report.py` | ✅ |
| Contains ok flag | `test_health_report.py` | ✅ |
| Contains server | `test_health_report.py` | ✅ |
| Contains version | `test_health_report.py` | ✅ |
| Namespaces mapping | `test_health_report.py` | ✅ |
| JSON serializable | `test_health_report.py` | ✅ |
| Stable shape | `test_health_report.py` | ✅ |

---

### §7.2 Data Types - Complete Coverage

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Returns GraphCapabilities | `test_capabilities_shape.py` | ✅ |
| Identity fields non-empty | `test_capabilities_shape.py` | ✅ |
| Dialects tuple non-empty | `test_capabilities_shape.py` | ✅ |
| All feature flags boolean | `test_capabilities_shape.py` | ✅ |
| max_batch_ops valid | `test_capabilities_shape.py` | ✅ |
| Protocol field validation | `test_capabilities_shape.py` | ✅ |
| Idempotent calls | `test_capabilities_shape.py` | ✅ |
| JSON serializable | `test_capabilities_shape.py` | ✅ |

---

### §7.4 Dialect Handling - Complete Coverage

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Unknown dialects rejected | `test_dialect_validation.py` | ✅ |
| Known dialects accepted | `test_dialect_validation.py` | ✅ |
| Validates against capabilities | `test_dialect_validation.py` | ✅ |
| Error includes dialect name | `test_dialect_validation.py` | ✅ |
| Capability-driven validation | `test_query_basic.py` | ✅ |
| Error mapping for dialects | `test_error_mapping_retryable.py` | ✅ |

---

### §6.3 Error Taxonomy - Complete Coverage

| Error Type | Test File | Status |
|------------|-----------|--------|
| BadRequest (validation) | `test_crud_basic.py`, `test_query_basic.py`, `test_error_mapping_retryable.py` | ✅ |
| NotSupported (dialect) | `test_dialect_validation.py`, `test_error_mapping_retryable.py` | ✅ |
| ResourceExhausted | `test_error_mapping_retryable.py` | ✅ |
| Unavailable | `test_error_mapping_retryable.py` | ✅ |
| DeadlineExceeded | `test_deadline_enforcement.py`, `test_error_mapping_retryable.py` | ✅ |
| TransientNetwork | `test_error_mapping_retryable.py` | ✅ |
| retry_after_ms hint | `test_error_mapping_retryable.py` | ✅ |
| error details mapping | `test_error_mapping_retryable.py` | ✅ |
| Error string includes code | `test_error_mapping_retryable.py` | ✅ |

---

### §13 Observability - Complete Coverage

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Tenant never logged raw | `test_context_siem.py` | ✅ |
| Tenant hashed in metrics | `test_context_siem.py` | ✅ |
| No query text in metrics | `test_context_siem.py` | ✅ |
| Metrics on error path | `test_context_siem.py` | ✅ |
| Dialect in metrics | `test_context_siem.py` | ✅ |
| Op count in batch metrics | `test_context_siem.py` | ✅ |

---

### §6.1 Context & Deadlines - Complete Coverage

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Query timeout | `test_deadline_enforcement.py` | ✅ |
| Write operation timeout | `test_deadline_enforcement.py` | ✅ |
| Stream timeout | `test_deadline_enforcement.py` | ✅ |
| Pre-flight validation | `test_deadline_enforcement.py` | ✅ |

---

### §4.2 Wire Protocol - Complete Coverage
*Note: Complete wire protocol coverage is in the separate wire conformance suite*

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Graph operation routing | `test_wire_handler.py` | ✅ |
| Error envelope normalization | `test_wire_handler.py` | ✅ |
| Context propagation | `test_wire_handler.py` | ✅ |
| Unknown operation handling | `test_wire_handler.py` | ✅ |
| Streaming envelope handling | `test_wire_handler.py` | ✅ |
| Missing required keys mapping | `test_wire_handler.py` | ✅ |
| Context and args object validation | `test_wire_handler.py` | ✅ |
| Query missing required fields mapping | `test_wire_handler.py` | ✅ |

---

## Running Tests

### All Graph conformance tests (0.64s typical)
```bash
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter pytest tests/graph/ -v
```

### Performance Optimized Runs
```bash
# Parallel execution (recommended for CI/CD) - ~0.35s
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter pytest tests/graph/ -n auto

# With detailed timing report
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter pytest tests/graph/ --durations=10

# Fast mode (skip slow markers)
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter pytest tests/graph/ -k "not slow"
```

### By category with timing estimates
```bash
# Core operations & CRUD (~0.25s)
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter pytest \
  tests/graph/test_crud_basic.py \
  tests/graph/test_query_basic.py \
  tests/graph/test_health_report.py -v

# Dialect & streaming (~0.15s)
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter pytest \
  tests/graph/test_dialect_validation.py \
  tests/graph/test_streaming_semantics.py -v

# Batch & schema operations (~0.12s)
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter pytest \
  tests/graph/test_batch_operations.py \
  tests/graph/test_schema_operations.py -v

# Infrastructure & capabilities (~0.12s)
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter pytest \
  tests/graph/test_capabilities_shape.py \
  tests/graph/test_deadline_enforcement.py \
  tests/graph/test_context_siem.py -v

# Error handling (~0.10s)
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter pytest \
  tests/graph/test_error_mapping_retryable.py -v

# Wire handler (~0.15s)
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter pytest \
  tests/graph/test_wire_handler.py -v
```

### With Coverage Report
```bash
# Basic coverage (0.8s typical)
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter \
  pytest tests/graph/ --cov=corpus_sdk.graph --cov-report=html

# Minimal coverage (0.7s typical)
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter \
  pytest tests/graph/ --cov=corpus_sdk.graph --cov-report=term-missing

# CI/CD optimized (parallel + coverage) - ~0.45s
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter \
  pytest tests/graph/ -n auto --cov=corpus_sdk.graph --cov-report=xml
```

### Adapter-Agnostic Usage
To validate a **third-party** or custom Graph Protocol implementation:

1. Implement the Graph Protocol V1.0 interface as defined in `SPECIFICATION.md §7`
2. Provide a small adapter/fixture that binds these tests to your implementation
3. Run the full `tests/graph/` suite
4. If all 99 tests pass unmodified, you can accurately claim:
   **"Graph Protocol V1.0 - 100% Conformant (Corpus Reference Suite)"**

### With Makefile Integration
```bash
# Run all Graph tests (0.64s typical)
make test-graph

# Run Graph tests with coverage (0.8s typical)
make test-graph-coverage

# Run Graph tests in parallel (0.35s typical)
make test-graph-parallel

# Run specific categories
make test-graph-core      # Core operations
make test-graph-crud      # CRUD validation
make test-graph-query     # Query operations
make test-graph-batch     # Batch operations
make test-graph-errors    # Error handling
make test-graph-wire      # Wire handler
```

---

## Adapter Compliance Checklist

Use this checklist when implementing or validating a new Graph adapter:

### ✅ Phase 1: Core Operations (11/11)
* [x] `graph.capabilities()` returns valid `GraphCapabilities` with all fields (§7.2)
* [x] `graph.upsert_nodes()` returns valid `GraphID` with proper format (§7.3.1)
* [x] `graph.upsert_edges()` returns valid `GraphID` with proper format (§7.3.1)
* [x] `graph.delete_nodes()` are idempotent and accept filters (§7.3.1, §11.4)
* [x] `graph.delete_edges()` are idempotent and accept filters (§7.3.1, §11.4)
* [x] `graph.query()` returns JSON-serializable results with dialect validation (§7.3.2)
* [x] `graph.stream_query()` yields QueryChunk instances with proper streaming semantics (§7.3.2, §4.2.3)
* [x] `graph.bulk_vertices()` respects `max_batch_ops` limits with pagination support (§7.3.3)
* [x] `graph.batch()` returns per-operation results with transaction support (§7.3.3)
* [x] `graph.get_schema()` returns consistent, serializable schema (§7.5)
* [x] `graph.health()` returns proper health status with namespaces (§7.6)

### ✅ Phase 2: Validation & Dialect Handling (15/15)
* [x] Reject empty labels in `graph.upsert_nodes`/`graph.upsert_edges` (§17.2)
* [x] Validate required `from`/`to` fields for `graph.upsert_edges` (§7.3.1)
* [x] Ensure properties are JSON-serializable in all operations (§17.2)
* [x] Reject unknown dialects with clear error messages in `graph.query` (§7.4)
* [x] Validate dialects against capabilities in `graph.query` (§7.4)
* [x] Require non-empty query text in `graph.query` (§17.2)
* [x] Support empty parameters in `graph.query` (§7.3.2)
* [x] Safe parameter binding to prevent injection in `graph.query` (§14.4)
* [x] Enforce `max_batch_ops` with helpful error hints in `graph.batch` (§7.2, §12.1)
* [x] Reject empty node/edge lists in `graph.upsert_nodes`/`graph.upsert_edges` (§17.2)
* [x] Support filters for `graph.delete_nodes`/`graph.delete_edges` (§7.3.1)
* [x] Validate filter serializability in delete operations (§17.2)
* [x] Support non-string keys in properties when JSON allows (§4.2.1)
* [x] Capability-driven dialect validation in `graph.query` (§7.4)
* [x] Error messages include dialect context in `graph.query` errors (§12.4)

### ✅ Phase 3: Error Handling & Semantics (16/16)
* [x] Map provider errors to canonical codes (`BadRequest`, `NotSupported`, etc.) (§6.3)
* [x] Include `retry_after_ms` for retryable errors when available (§12.1)
* [x] Include operation and dialect context in errors (§12.4)
* [x] Do not treat validation errors as retryable (§12.1)
* [x] Provide `suggested_batch_reduction` for batch size errors (§12.1)
* [x] Use `DeadlineExceeded` on expired budgets (§6.1, §12.1)
* [x] Honor `NotSupported` for unsupported dialects/features (§7.4)
* [x] Follow §12.5 partial-failure semantics for batch operations
* [x] Error details are proper mappings (§6.3)
* [x] Normalized error codes mapped correctly (§12.4)
* [x] Error strings include error codes (§12.4)
* [x] Retryable errors accept retry_after and details (§6.3)
* [x] Handle empty edge label validation in `graph.upsert_edges` (§17.2)
* [x] Proper error for unknown dialects in `graph.query` (§7.4)
* [x] Error hardening for unexpected exceptions (§6.3)

### ✅ Phase 4: Observability & Privacy (6/6)
* [x] Use `component="graph"` in metrics (§13.1)
* [x] Emit exactly one `observe` per operation (§13.1)
* [x] Never log raw query text, tenant IDs, or sensitive properties (§13.1, §15)
* [x] Use `tenant_hash`, `dialect`, `op_count` as low-cardinality tags (§13.1)
* [x] Emit error counters on failure paths (§13.1)
* [x] Ensure wire+logs SIEM-safe per §13 requirements

### ✅ Phase 5: Deadlines, Caching & Wire Contract (18/18)
* [x] Respect `OperationContext.deadline_ms` with preflight checks (§6.1)
* [x] Use `DeadlineExceeded` when time budget elapses mid-operation (§12.1)
* [x] Support early cancellation of `graph.stream_query` (§11.5)
* [x] Ensure resource cleanup on `graph.stream_query` cancellation (§11.5)
* [x] Cache schema when appropriate (§16.3)
* [x] `WireGraphHandler` implements all `graph.*` ops with canonical envelopes (§4.2.6)
* [x] Unknown fields ignored; unknown ops → `NotSupported` (§4.2.5, §4.2.6)
* [x] Error envelopes use normalized `code`/`error` structure (§6.3)
* [x] Proper wire envelope shapes for all operations (§4.2.1)
* [x] Context propagation through wire handler (§6.1)
* [x] Success envelopes for all operations (§4.2.1)
* [x] Streaming wire frame validation for `graph.stream_query` (§4.2.3)
* [x] Wrong operation error handling for streaming (§4.2.6)
* [x] Missing/invalid operation error handling (§4.2.6)
* [x] Graph adapter error normalization (§6.3)
* [x] NotSupported error propagation (§6.3)
* [x] Unexpected exception hardening (§6.3)
* [x] Required field validation in wire requests (§4.2.1)

---

## Conformance Badge

```text
🏆 GRAPH PROTOCOL V1.0 - PLATINUM CERTIFIED
   99/99 conformance tests passing (100%)

   📊 Total Tests: 99/99 passing (100%)
   ⚡ Execution Time: 0.64s (6.5ms/test avg)
   🏆 Certification: Platinum (100%)

   ✅ Core Operations: 11/11 (100%) - §7.3
   ✅ CRUD Validation: 10/10 (100%) - §7.3.1, §17.2
   ✅ Query Operations: 8/8 (100%) - §7.3.2
   ✅ Dialect Validation: 6/6 (100%) - §7.4
   ✅ Streaming Semantics: 5/5 (100%) - §7.3.2, §4.2.3
   ✅ Batch Operations: 10/10 (100%) - §7.3.3
   ✅ Schema Operations: 2/2 (100%) - §7.5
   ✅ Error Handling: 12/12 (100%) - §6.3, §12.1, §12.4
   ✅ Capabilities: 8/8 (100%) - §7.2, §6.2
   ✅ Observability & Privacy: 6/6 (100%) - §13.1, §13.2, §15
   ✅ Deadline Semantics: 4/4 (100%) - §6.1, §12.1
   ✅ Health Endpoint: 5/5 (100%) - §7.6, §6.4
   ✅ Wire Envelopes & Routing: 14/14 (100%) - §4.2

   Status: Production Ready 🏆 Platinum Certified
```

**Badge Suggestion:**
[![Corpus Graph Protocol](https://img.shields.io/badge/CorpusGraph%20Protocol-Platinum%20Certified-brightgreen)](./graph_conformance_report.json)

**Performance Benchmark:**
```text
Execution Time: 0.64s total (6.5ms/test average)
Cache Efficiency: 0 hits, 99 misses (cache size: 99)
Parallel Ready: Yes (optimized for pytest-xdist)
Memory Footprint: Minimal (deterministic mocks)
Specification Coverage: 100% of §7 requirements
Test Files: 12 comprehensive modules
```

**Last Updated:** 2026-02-10  
**Maintained By:** Corpus SDK Team  
**Test Suite:** `tests/graph/` (12 test files)  
**Specification Version:** V1.0.0 §7  
**Status:** 100% V1.0 Conformant - Platinum Certified (99/99 tests, 0.64s runtime)

---
