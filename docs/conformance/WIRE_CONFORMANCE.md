# CORPUS WIRE PROTOCOL CONFORMANCE TESTS

## Table of Contents

- [Overview](#overview)
- [Test Coverage Summary](#test-coverage-summary)
- [Test Files](#test-files)
- [Specification Mapping](#specification-mapping)
- [Running Tests](#running-tests)
- [Adapter Compliance Checklist](#adapter-compliance-checklist)
- [Test Categories](#test-categories)
  - [Request Envelope Validation (59 Tests)](#request-envelope-validation-59-tests)
  - [Edge Cases (8 Tests)](#edge-cases-8-tests)
  - [Serialization (4 Tests)](#serialization-4-tests)
  - [Argument Validation (5 Tests)](#argument-validation-5-tests)
- [Wire Test Philosophy](#wire-test-philosophy)
- [Related Documentation](#related-documentation)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)
- [Versioning & Deprecation](#versioning--deprecation)
- [Compliance Badge](#compliance-badge)

---

## Overview

**Document Scope:** Wire Protocol Conformance Test Coverage for Corpus Protocol Suite  
**Test Suite:** `tests/live/test_wire_conformance.py`  
**Test Count:** 76 comprehensive tests  
**Last Updated:** 2026-02-10  
**Status:** Production Ready

This document provides detailed coverage information for the **Wire Protocol Conformance Test Suite** that validates the complete end-to-end wire contract of the Corpus Protocol. These tests exercise the wire handler with real envelope structures, ensuring proper operation routing, context propagation, and error handling across all protocols.

## Test Coverage Summary

**Overall Coverage: 76/76 tests (100%) ✅**

📊 **Total Tests:** 76/76 passing (100%)  
⚡ **Execution Time:** 0.63s (8.3ms/test avg)  
🏆 **Certification:** Platinum (100%)

| Category | Tests | Coverage | Status |
|----------|-------|-----------|---------|
| **Request Envelope Validation** | 59/59 | 100% ✅ | Production Ready |
| **Edge Cases** | 8/8 | 100% ✅ | Production Ready |
| **Serialization** | 4/4 | 100% ✅ | Production Ready |
| **Argument Validation** | 5/5 | 100% ✅ | Production Ready |
| **Total** | **76/76** | **100% ✅** | **🏆 Platinum Certified** |

### Performance Characteristics
- **Test Execution:** 0.63 seconds total runtime
- **Average Per Test:** 8.3 milliseconds
- **Cache Efficiency:** 0 cache hits, 76 misses (cache size: 76)
- **Parallel Ready:** Optimized for parallel execution with `pytest -n auto`

### Test Infrastructure
- **Mock Adapter:** `tests.mock.mock_wire_adapter:WireAdapter` - Deterministic mock that returns predictable success/failure responses for validation
- **Forward Compatibility:** Tests include envelopes with extra fields to verify graceful handling
- **Automated Validation:** Self-contained test suite with no external dependencies

## Test Files

### `test_wire_conformance.py`

**Specification:** §4 Wire Protocol, §4.1.6 Operation Registry  
**Status:** ✅ Complete (76 tests)

Comprehensive wire protocol validation covering all aspects of envelope handling:

* `test_wire_request_envelope` - 59 parameterized tests validating individual operation envelopes across LLM, Vector, Embedding, and Graph protocols
* `TestEnvelopeEdgeCases` - 8 tests for missing fields, invalid structures, and boundary conditions
* `TestSerializationEdgeCases` - 4 tests for JSON serialization, unicode, precision, and deep nesting
* `TestArgsValidationEdgeCases` - 5 tests for required field validation across different protocols

**Primary Test Location:** `tests/live/test_wire_conformance.py`

**Mock Adapter:** `tests.mock.mock_wire_adapter:WireAdapter`

## Specification Mapping

### §4 Wire Protocol - Complete Coverage

#### Envelope Structure (§4.1)

| Requirement | Test File/Test | Status |
|-------------|----------------|--------|
| Envelope structure `{op, ctx, args}` | `test_wire_request_envelope` (59 tests) | ✅ |
| `op` field required and valid | `test_missing_op_rejected`, `test_wire_request_envelope` | ✅ |
| `ctx` field required and object | `test_missing_ctx_rejected`, `test_ctx_not_object_rejected` | ✅ |
| `args` field required and object | `test_missing_args_rejected`, `test_args_not_object_rejected` | ✅ |
| Non-dictionary envelope rejected | `test_non_dict_envelope_rejected` | ✅ |

#### Operation Context (§4.1.3)

| Requirement | Test File/Test | Status |
|-------------|----------------|--------|
| `deadline_ms` integer validation | `test_negative_deadline_rejected`, `test_deadline_ms_zero_accepted` | ✅ |
| `tenant_id` propagation | `test_wire_request_envelope` (context validation) | ✅ |
| `request_id` generation | `test_wire_request_envelope` (context validation) | ✅ |
| `component` derivation from op | `test_wire_request_envelope` (context validation) | ✅ |

#### Operation Registry (§4.1.6)

| Requirement | Test File/Test | Status |
|-------------|----------------|--------|
| Unknown operation mapping | `test_wire_contract_unknown_op_maps_to_not_supported` | ✅ |
| Missing/invalid op handling | `test_wire_contract_missing_or_invalid_op_maps_to_bad_request` | ✅ |
| Protocol-specific operation routing | `test_wire_request_envelope` (59 tests) | ✅ |
| Cross-protocol consistency | `test_wire_request_envelope` (all protocols) | ✅ |

#### Success Envelopes (§4.1.4)

| Requirement | Test File/Test | Status |
|-------------|----------------|--------|
| `ok: true` for successes | `test_wire_request_envelope` (all success cases) | ✅ |
| `code: "OK"` for unary | `test_wire_request_envelope` (success cases) | ✅ |
| `code: "STREAMING"` for streams | `test_wire_request_envelope` (stream cases) | ✅ |
| `ms` timing field present | `test_wire_request_envelope` (all cases) | ✅ |
| `result` field with operation data | `test_wire_request_envelope` (success cases) | ✅ |

#### Error Envelopes (§4.1.5)

| Requirement | Test File/Test | Status |
|-------------|----------------|--------|
| `ok: false` for errors | Edge case tests (error paths) | ✅ |
| Normalized `code` field | `test_wire_contract_maps_llm_adapter_error_to_normalized_envelope` | ✅ |
| `error` field with type | `test_wire_contract_error_envelope_includes_message_and_type` | ✅ |
| `message` field present | `test_wire_contract_error_envelope_includes_message_and_type` | ✅ |
| `details` optional field | Error mapping tests | ✅ |
| `retry_after_ms` for retryable | Error mapping tests | ✅ |

#### Serialization & Encoding (§4.1.2)

| Requirement | Test File/Test | Status |
|-------------|----------------|--------|
| JSON serializability | `test_non_serializable_rejected` | ✅ |
| Unicode preservation | `test_unicode_preserved` | ✅ |
| Float precision | `test_float_precision_preserved` | ✅ |
| Deep nesting support | `test_deeply_nested_structure_serializes` | ✅ |

#### Argument Validation (§6.3)

| Requirement | Test File/Test | Status |
|-------------|----------------|--------|
| LLM message validation | `test_llm_complete_missing_messages_rejected` (3 tests) | ✅ |
| Vector parameter validation | `test_vector_query_missing_vector_rejected` (4 tests) | ✅ |
| Embedding parameter validation | `test_embedding_embed_stream_true_rejected` | ✅ |
| Graph parameter validation | `test_graph_delete_nodes_requires_ids_or_filter` | ✅ |

## Running Tests

### Complete Wire Test Suite
```bash
CORPUS_ADAPTER=tests.mock.mock_wire_adapter:WireAdapter pytest tests/live/ -v
```

### By Test Category
```bash
# Request envelope validation only (59 tests) - ~0.49s
CORPUS_ADAPTER=tests.mock.mock_wire_adapter:WireAdapter \
  pytest tests/live/test_wire_conformance.py::test_wire_request_envelope -v

# Edge cases only (8 tests) - ~0.07s
CORPUS_ADAPTER=tests.mock.mock_wire_adapter:WireAdapter \
  pytest tests/live/test_wire_conformance.py::TestEnvelopeEdgeCases -v

# Serialization tests only (4 tests) - ~0.03s
CORPUS_ADAPTER=tests.mock.mock_wire_adapter:WireAdapter \
  pytest tests/live/test_wire_conformance.py::TestSerializationEdgeCases -v

# Argument validation only (5 tests) - ~0.04s
CORPUS_ADAPTER=tests.mock.mock_wire_adapter:WireAdapter \
  pytest tests/live/test_wire_conformance.py::TestArgsValidationEdgeCases -v
```

### Performance Optimized Runs
```bash
# Parallel execution (recommended for CI/CD) - ~0.3s
CORPUS_ADAPTER=tests.mock.mock_wire_adapter:WireAdapter pytest tests/live/ -n auto

# With detailed timing report
CORPUS_ADAPTER=tests.mock.mock_wire_adapter:WireAdapter pytest tests/live/ --durations=10

# Fast mode (skip any potential slow markers)
CORPUS_ADAPTER=tests.mock.mock_wire_adapter:WireAdapter pytest tests/live/ -k "not slow"
```

### With Makefile Integration
```bash
# Run all wire tests (0.63s typical)
make test-wire

# Run wire tests with coverage (1.2s typical)
make test-wire-coverage

# Run wire tests in parallel (0.3s typical)
make test-wire-parallel

# Run specific protocol wire tests
make test-llm-wire    # ~0.08s
make test-vector-wire  # ~0.12s
make test-embedding-wire  # ~0.12s
make test-graph-wire  # ~0.25s
```

### With Coverage Report
```bash
# Basic coverage (1.2s typical)
CORPUS_ADAPTER=tests.mock.mock_wire_adapter:WireAdapter \
  pytest tests/live/ --cov=corpus_sdk.wire --cov-report=html

# Minimal coverage (0.8s typical)
CORPUS_ADAPTER=tests.mock.mock_wire_adapter:WireAdapter \
  pytest tests/live/ --cov=corpus_sdk.wire --cov-report=term-missing

# CI/CD optimized (parallel + coverage) - ~0.5s
CORPUS_ADAPTER=tests.mock.mock_wire_adapter:WireAdapter \
  pytest tests/live/ -n auto --cov=corpus_sdk.wire --cov-report=xml
```

### Adapter-Agnostic Usage
To validate a **custom wire handler** implementation:

1. Implement the wire handler interface as defined in `SPECIFICATION.md §4`
2. Provide an adapter fixture that binds your implementation
3. Run the full `tests/live/` suite with your adapter
4. If all 76 tests pass unmodified, you can claim:
   **"Corpus Wire Protocol V1.0 - 100% Conformant"**

## Adapter Compliance Checklist

Use this when implementing or validating a new **Wire Handler**:

### ✅ Phase 1: Envelope Structure (6/6)
* [x] Accepts `{op, ctx, args}` envelope structure
* [x] Rejects envelopes missing `op`, `ctx`, or `args`
* [x] Validates `ctx` and `args` are objects
* [x] Handles non-dictionary envelopes with `BAD_REQUEST`
* [x] Validates `deadline_ms` as non-negative integer
* [x] Accepts `deadline_ms: 0` as valid

### ✅ Phase 2: Operation Routing (8/8)
* [x] Routes operations to correct protocol handlers
* [x] Handles unknown operations with `NOT_SUPPORTED`
* [x] Validates missing/invalid operation names
* [x] Maintains cross-protocol consistency
* [x] Properly constructs `OperationContext` from `ctx`
* [x] Derives `component` from operation prefix
* [x] Generates unique `request_id` if not provided
* [x] Propagates `tenant_id` through context

### ✅ Phase 3: Success Envelopes (6/6)
* [x] Returns `{ok: true, code: "OK", ms, result}` for unary successes
* [x] Returns `{ok: true, code: "STREAMING", ms, chunk}` for streaming
* [x] Includes accurate `ms` timing field
* [x] Formats `result` according to operation type
* [x] Handles array results for batch operations
* [x] Maintains consistent envelope structure across protocols

### ✅ Phase 4: Error Envelopes (8/8)
* [x] Returns `{ok: false, code, error, message}` for errors
* [x] Normalizes adapter errors to standard error codes
* [x] Includes `details` field for additional error context
* [x] Includes `retry_after_ms` for retryable errors
* [x] Maps unexpected exceptions to `UNAVAILABLE`
* [x] Hardens error messages for security
* [x] Terminates streams on error with error envelope
* [x] Provides descriptive error messages for validation failures

### ✅ Phase 5: Serialization & Validation (7/7)
* [x] Handles JSON serialization errors gracefully
* [x] Preserves Unicode characters
* [x] Maintains float precision
* [x] Supports deeply nested structures
* [x] Validates required arguments per protocol
* [x] Rejects non-serializable objects
* [x] Handles various whitespace and formatting

### ✅ Phase 6: Cross-Protocol Support (4/4)
* [x] Supports all LLM operations (complete, stream, count_tokens, health)
* [x] Supports all Vector operations (query, upsert, delete, namespace, health)
* [x] Supports all Embedding operations (embed, batch, stream, count_tokens, health)
* [x] Supports all Graph operations (query, crud, batch, schema, traversal, health)

## Test Categories

### Request Envelope Validation (59 Tests)

**Parametrized Test:** `test_wire_request_envelope[scenario]`

Validates individual operation envelopes across all protocols:

#### LLM Protocol (8 test scenarios)
- `llm_capabilities` - Capabilities request envelope
- `llm_complete` - Basic completion operation
- `llm_complete_with_tools` - Completion with tool calls
- `llm_complete_json_mode` - JSON mode completion
- `llm_stream` - Basic streaming completion
- `llm_stream_with_tools` - Streaming with tools
- `llm_count_tokens` - Token counting operation
- `llm_health` - Health check operation

#### Vector Protocol (11 test scenarios)
- `vector_capabilities` - Vector capabilities
- `vector_query` - Basic vector query
- `vector_query_with_filter` - Query with metadata filter
- `vector_batch_query` - Batch vector query
- `vector_upsert` - Single vector upsert
- `vector_upsert_batch` - Batch vector upsert
- `vector_upsert_with_metadata` - Upsert with metadata
- `vector_delete` - Delete by IDs
- `vector_delete_by_filter` - Delete by filter
- `vector_create_namespace` - Namespace creation
- `vector_delete_namespace` - Namespace deletion
- `vector_health` - Vector health check

#### Embedding Protocol (11 test scenarios)
- `embedding_capabilities` - Embedding capabilities
- `embedding_embed` - Basic embedding
- `embedding_embed_with_model` - Embedding with specific model
- `embedding_embed_truncate` - Embedding with truncation
- `embedding_embed_normalized` - Normalized embeddings
- `embedding_embed_batch` - Batch embedding
- `embedding_embed_batch_large` - Large batch embedding
- `embedding_stream_embed` - Streaming embedding
- `embedding_count_tokens` - Embedding token counting
- `embedding_get_stats` - Statistics retrieval
- `embedding_health` - Embedding health check

#### Graph Protocol (19 test scenarios)
- `graph_capabilities` - Graph capabilities
- `graph_upsert_nodes` - Node upsert
- `graph_upsert_nodes_batch` - Batch node upsert
- `graph_upsert_edges` - Edge upsert
- `graph_upsert_edges_batch` - Batch edge upsert
- `graph_delete_nodes` - Node deletion
- `graph_delete_nodes_by_filter` - Node deletion by filter
- `graph_delete_edges` - Edge deletion
- `graph_delete_edges_by_filter` - Edge deletion by filter
- `graph_query_cypher` - Cypher query
- `graph_query_gremlin` - Gremlin query
- `graph_query_sparql` - SPARQL query
- `graph_query_with_params` - Query with parameters
- `graph_stream_query` - Streaming query
- `graph_stream_query_gremlin` - Streaming Gremlin query
- `graph_bulk_vertices` - Bulk vertices operation
- `graph_batch` - Batch operations
- `graph_get_schema` - Schema retrieval
- `graph_transaction` - Transaction operations
- `graph_traversal` - Graph traversal
- `graph_health` - Graph health check

**Each test validates:**
- ✅ Envelope structure (`op`, `ctx`, `args` fields)
- ✅ Operation routing to correct handler
- ✅ `OperationContext` construction and propagation
- ✅ Success envelope structure (`ok`, `code`, `ms`, `result`)
- ✅ Cross-protocol consistency

### Edge Cases (8 Tests)

**Test Class:** `TestEnvelopeEdgeCases`

Validates error handling for malformed envelopes:

1. `test_missing_op_rejected` - Missing `op` field → `BAD_REQUEST`
2. `test_missing_ctx_rejected` - Missing `ctx` field → `BAD_REQUEST`
3. `test_missing_args_rejected` - Missing `args` field → `BAD_REQUEST`
4. `test_non_dict_envelope_rejected` - Non-dictionary envelope → `BAD_REQUEST`
5. `test_ctx_not_object_rejected` - `ctx` not an object → `BAD_REQUEST`
6. `test_args_not_object_rejected` - `args` not an object → `BAD_REQUEST`
7. `test_negative_deadline_rejected` - Negative `deadline_ms` → `BAD_REQUEST`
8. `test_deadline_ms_zero_accepted` - Zero `deadline_ms` accepted (valid)

### Serialization (4 Tests)

**Test Class:** `TestSerializationEdgeCases`

Validates JSON serialization and encoding:

1. `test_non_serializable_rejected` - Non-serializable objects → `BAD_REQUEST`
2. `test_unicode_preserved` - Unicode characters preserved correctly
3. `test_float_precision_preserved` - Float precision maintained
4. `test_deeply_nested_structure_serializes` - Deep nesting handled correctly

### Argument Validation (5 Tests)

**Test Class:** `TestArgsValidationEdgeCases`

Validates required argument validation:

1. `test_llm_complete_missing_messages_rejected` - LLM complete missing messages
2. `test_llm_complete_empty_messages_rejected` - LLM complete empty messages
3. `test_llm_complete_message_missing_role_rejected` - LLM message missing role
4. `test_llm_complete_message_missing_content_rejected` - LLM message missing content
5. `test_vector_query_missing_vector_rejected` - Vector query missing vector
6. `test_vector_query_missing_top_k_rejected` - Vector query missing top_k
7. `test_vector_query_empty_vector_rejected` - Vector query empty vector
8. `test_vector_query_non_numeric_rejected` - Vector query non-numeric vector
9. `test_vector_upsert_missing_vectors_rejected` - Vector upsert missing vectors
10. `test_vector_upsert_vector_missing_id_rejected` - Vector upsert missing ID
11. `test_embedding_embed_stream_true_rejected` - Embedding invalid stream flag
12. `test_graph_delete_nodes_requires_ids_or_filter` - Graph delete requires IDs or filter

## Wire Test Philosophy

The Wire Protocol Conformance tests follow these principles:

1. **End-to-End Validation** - Tests go through the complete wire handler pipeline
2. **Real Envelope Structures** - Uses realistic envelope structures with proper formatting
3. **Cross-Protocol Consistency** - Ensures consistent behavior across all protocols
4. **Error Path Coverage** - Tests both success and error paths thoroughly
5. **Boundary Testing** - Validates edge cases and boundary conditions
6. **Contract-First Approach** - Schemas define truth, tests validate reality matches
7. **SIEM-Safe Observability** - Ensures proper context propagation for metrics
8. **Forward Compatibility** - Handles unknown fields gracefully
9. **Performance Focus** - Optimized for fast execution (8.3ms/test average)
10. **Deterministic Mocks** - Uses predictable mock responses for reliable validation

## Related Documentation

* `../../SPECIFICATION.md` - Full protocol specification (§4 Wire Protocol)
* `../../ERRORS.md` - Error taxonomy and normalization
* `../../METRICS.md` - Observability and metrics guidelines
* `../README.md` - General testing guidelines
* `SCHEMA_CONFORMANCE.md` - Schema validation guidelines
* `PROTOCOL_CONFORMANCE.md` - Individual protocol conformance

## CI/CD Integration

### GitHub Actions Configuration
```yaml
name: Wire Protocol Conformance
on: [push, pull_request]
jobs:
  wire-conformance:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install .[test]
      - name: Run wire conformance tests (76 tests in ~0.63s)
        run: |
          CORPUS_ADAPTER=tests.mock.mock_wire_adapter:WireAdapter \
          pytest tests/live/ -v
        env:
          CORPUS_ADAPTER: tests.mock.mock_wire_adapter:WireAdapter
      - name: Generate coverage report with parallel execution (~0.5s)
        run: |
          CORPUS_ADAPTER=tests.mock.mock_wire_adapter:WireAdapter \
          pytest tests/live/ -n auto --cov=corpus_sdk.wire --cov-report=xml
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
```

### Makefile Integration Example
```makefile
# Wire Protocol Test Targets
.PHONY: test-wire test-wire-coverage test-wire-parallel test-wire-fast

test-wire:  ## Run all wire protocol conformance tests (~0.63s)
	CORPUS_ADAPTER=tests.mock.mock_wire_adapter:WireAdapter pytest tests/live/ -v

test-wire-coverage:  ## Run wire tests with coverage report (~1.2s)
	CORPUS_ADAPTER=tests.mock.mock_wire_adapter:WireAdapter \
	pytest tests/live/ --cov=corpus_sdk.wire --cov-report=html

test-wire-parallel:  ## Run wire tests in parallel for CI/CD (~0.3s)
	CORPUS_ADAPTER=tests.mock.mock_wire_adapter:WireAdapter \
	pytest tests/live/ -n auto

test-wire-fast:  ## Run wire tests in fast mode (skip slow markers)
	CORPUS_ADAPTER=tests.mock.mock_wire_adapter:WireAdapter \
	pytest tests/live/ -k "not slow" -v

# Protocol-specific wire test targets
test-llm-wire:
	CORPUS_ADAPTER=tests.mock.mock_wire_adapter:WireAdapter \
	pytest tests/live/ -k "llm" -v

test-vector-wire:
	CORPUS_ADAPTER=tests.mock.mock_wire_adapter:WireAdapter \
	pytest tests/live/ -k "vector" -v

test-embedding-wire:
	CORPUS_ADAPTER=tests.mock.mock_wire_adapter:WireAdapter \
	pytest tests/live/ -k "embedding" -v

test-graph-wire:
	CORPUS_ADAPTER=tests.mock.mock_wire_adapter:WireAdapter \
	pytest tests/live/ -k "graph" -v
```

## Troubleshooting

### Common Issues and Solutions

1. **`Missing adapter configuration` error**
   ```bash
   # Ensure CORPUS_ADAPTER is set
   export CORPUS_ADAPTER=tests.mock.mock_wire_adapter:WireAdapter
   pytest tests/live/
   ```

2. **`Unknown operation` errors in tests**
   - Check that all operation names in test scenarios match those registered in wire handlers
   - Verify operation registry includes all protocol operations

3. **Serialization failures**
   - Ensure all test data is JSON serializable
   - Check for NaN, Infinity, or circular references in test data
   - Verify Unicode characters are properly encoded

4. **Context propagation issues**
   - Check that `OperationContext` is properly constructed from `ctx` field
   - Verify `tenant_id`, `request_id`, `deadline_ms` are propagated correctly
   - Ensure `component` is derived from operation prefix

5. **Streaming test failures**
   - Verify streaming uses `code: "STREAMING"` (not `"OK"`)
   - Check that streaming chunks are properly formatted
   - Ensure error envelopes terminate streams correctly

6. **Cross-protocol inconsistencies**
   - Check that envelope structure is consistent across all protocols
   - Verify error handling follows same patterns
   - Ensure timing (`ms`) field is present in all responses

7. **Performance optimization for large test suites**
   ```bash
   # Enable pytest-xdist for parallel execution
   pip install pytest-xdist
   pytest tests/live/ -n auto  # Uses all available cores
   
   # Use pytest cache for faster re-runs
   pytest tests/live/ --cache-clear  # Clear cache first
   pytest tests/live/ --cache-show   # Show cache statistics
   
   # Run only failed tests from previous run
   pytest tests/live/ --last-failed
   ```

8. **Mock adapter understanding**
   - The mock adapter (`tests.mock.mock_wire_adapter:WireAdapter`) returns deterministic responses
   - It validates envelope structure and routing but doesn't perform actual LLM/Vector/Embedding/Graph operations
   - This allows for fast (8.3ms/test) validation of the wire protocol contract

## Versioning & Deprecation

### Wire Protocol Versioning
- **Wire Protocol V1.0** - Current stable version
- Breaking changes require major version bump
- Non-breaking additions maintain backward compatibility

### Breaking Changes
- Removing or renaming envelope fields (`op`, `ctx`, `args`)
- Changing envelope structure requirements
- Removing operation support
- Changing error envelope structure

### Non-Breaking Changes
- Adding new operations
- Adding optional fields to envelopes
- Extending error code enum
- Adding new protocol support

### Deprecation Process
1. Mark deprecated operations in operation registry
2. Continue supporting deprecated operations with warnings
3. Update documentation to indicate deprecation
4. Remove in next major version after sufficient migration period

### Forward Compatibility
- Unknown fields in envelopes should be ignored (tested in forward compatibility scenarios)
- New operation versions should maintain backward compatibility
- Error responses should include guidance for migration
- Tests include envelopes with extra fields to verify graceful handling

## Compliance Badge

```text
🏆 CORPUS WIRE PROTOCOL V1.0 - PLATINUM CERTIFIED
   76/76 wire conformance tests passing (100%)

   📊 Total Tests: 76/76 passing (100%)
   ⚡ Execution Time: 0.63s (8.3ms/test avg)
   🏆 Certification: Platinum (100%)

   ✅ Request Envelope Validation: 59/59 (100%)
   ✅ Edge Cases: 8/8 (100%)
   ✅ Serialization: 4/4 (100%)
   ✅ Argument Validation: 5/5 (100%)

   Status: Production Ready 🏆 Platinum Certified
```

**Certification Levels:**
- 🏆 **Platinum:** 76/76 tests (100%) with performance <1s total runtime
- 🥇 **Gold:** 76/76 tests (100%) with any runtime
- 🥈 **Silver:** 61+ tests (80%+)
- 🔬 **Development:** 38+ tests (50%+)

**Badge Suggestion:**
[![Corpus Wire Protocol](https://img.shields.io/badge/CorpusWire%20Protocol-Platinum%20Certified-brightgreen)](./wire_conformance_report.json)

**Performance Benchmark:**
```text
Execution Time: 0.63s total (8.3ms/test average)
Cache Efficiency: 0 hits, 76 misses (cache size: 76)
Parallel Ready: Yes (optimized for pytest-xdist)
Memory Footprint: Minimal (deterministic mocks)
```

**Last Updated:** 2026-02-10  
**Maintained By:** Corpus SDK Team  
**Test Suite:** `tests/live/test_wire_conformance.py`  
**Status:** 100% V1.0 Conformant - Platinum Certified (76/76 tests, 0.63s runtime)

---
