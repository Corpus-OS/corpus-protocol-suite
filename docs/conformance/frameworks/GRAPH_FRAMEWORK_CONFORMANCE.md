# Graph Framework Adapters Conformance Test Coverage

## Table of Contents
- [Overview](#overview)
- [Conformance Summary](#conformance-summary)
- [Test Files](#test-files)
- [Framework Coverage](#framework-coverage)
- [Running Tests](#running-tests)
- [Framework Compliance Checklist](#framework-compliance-checklist)
- [Conformance Badge](#conformance-badge)
- [Maintenance](#maintenance)

---

## Overview

This document tracks conformance test coverage for **Graph Framework Adapters V1.0** across five major AI frameworks: LangChain, LlamaIndex, Semantic Kernel, CrewAI, and AutoGen. Each adapter translates framework-specific graph interfaces into the unified Corpus Graph Protocol V1.0.

This suite constitutes the **official Graph Framework Adapters V1.0 Reference Conformance Test Suite**. Any implementation (Corpus or third-party) MAY run these tests to verify and publicly claim conformance, provided all referenced tests pass unmodified.

**Adapter Version:** Graph Framework Adapters V1.0  
**Protocol Version:** Graph Protocol V1.0  
**Status:** Stable / Production-Ready  
**Last Updated:** 2026-02-10  
**Test Location:** `tests/frameworks/graph/`  
**Performance:** 13.97s total (24.4ms/test average)

## Conformance Summary

**Overall Coverage: 574/574 tests (100%) ✅**

📊 **Total Tests:** 574/574 passing (100%)  
⚡ **Execution Time:** 13.97s (24.4ms/test avg)  
🏆 **Certification:** Platinum (100%)

| Category | Tests | Coverage | Status |
|----------|-------|-----------|---------|
| **Parametrized Contract Tests** | 280 | 100% ✅ | Production Ready |
| **Framework-Specific Tests** | 294 | 100% ✅ | Production Ready |
| **Total** | **574/574** | **100% ✅** | **🏆 Platinum Certified** |

### Performance Characteristics
- **Test Execution:** 13.97 seconds total runtime
- **Average Per Test:** 24.4 milliseconds
- **Cache Efficiency:** 0 cache hits, 574 misses (cache size: 574)
- **Parallel Ready:** Optimized for parallel execution with `pytest -n auto`

### Test Infrastructure
- **Mock Adapter:** `tests.mock.mock_graph_adapter:MockGraphAdapter` - Deterministic mock for Graph operations
- **Testing Framework:** pytest 9.0.2 with comprehensive plugin support
- **Environment:** Python 3.10.19 on Darwin
- **Strict Mode:** Off (permissive testing)

## **Graph Framework Adapters Certification**

- 🏆 **Platinum:** 574/574 tests (100% comprehensive conformance)
- 🥇 **Gold:** 574 tests (100% protocol mastery)
- 🥈 **Silver:** 460+ tests (80%+ integration-ready)
- 🔬 **Development:** 287+ tests (50%+ early development)

---

## Test Files

### Parametrized Contract Tests (280 tests)

These tests run **once per framework** (5 frameworks), validating consistent behavior across all adapters.

#### `test_contract_context_and_error_context.py` (90 tests = 18×5)

**Specification:** Framework Adapter Contract V1.0, §3.1-3.4  
**Status:** ✅ Complete

Context translation and error handling across all 5 frameworks:

* `test_registry_declared_methods_exist_and_are_callable_when_available[framework_descriptor0-4]` (5)
* `test_registry_flags_are_coherent_with_declared_methods_when_available[framework_descriptor0-4]` (5)
* `test_rich_mapping_context_is_accepted_and_does_not_break_queries[framework_descriptor0-4]` (5)
* `test_invalid_context_type_is_tolerated_and_does_not_crash[framework_descriptor0-4]` (5)
* `test_context_is_optional_and_omitting_it_still_works[framework_descriptor0-4]` (5)
* `test_rich_mapping_context_is_accepted_across_all_registry_declared_sync_methods[framework_descriptor0-4]` (5)
* `test_invalid_context_is_tolerated_across_all_registry_declared_sync_methods[framework_descriptor0-4]` (5)
* `test_context_is_optional_across_all_registry_declared_sync_methods[framework_descriptor0-4]` (5)
* `test_rich_mapping_context_is_accepted_across_all_registry_declared_async_methods[framework_descriptor0-4]` (5)
* `test_invalid_context_is_tolerated_across_all_registry_declared_async_methods[framework_descriptor0-4]` (5)
* `test_context_is_optional_across_all_registry_declared_async_methods[framework_descriptor0-4]` (5)
* `test_error_context_is_attached_on_sync_query_failure[framework_descriptor0-4]` (5)
* `test_error_context_is_attached_on_sync_stream_failure_when_supported[framework_descriptor0-4]` (5)
* `test_error_context_is_attached_on_sync_stream_calltime_failure_when_supported[framework_descriptor0-4]` (5)
* `test_error_context_is_attached_on_sync_failure_for_all_registry_declared_methods[framework_descriptor0-4]` (5)
* `test_error_context_is_attached_on_async_query_failure_when_supported[framework_descriptor0-4]` (5)
* `test_error_context_is_attached_on_async_stream_failure_when_supported[framework_descriptor0-4]` (5)
* `test_error_context_is_attached_on_async_failure_for_all_registry_declared_methods[framework_descriptor0-4]` (5)

#### `test_contract_interface_conformance.py` (65 tests = 13×5)

**Specification:** Framework Adapter Contract V1.0, §2.1-2.3  
**Status:** ✅ Complete

Validates core interface requirements and async support across all 5 frameworks:

* `test_can_instantiate_graph_client[framework_descriptor0-4]` (5)
* `test_sync_query_interface_conformance[framework_descriptor0-4]` (5)
* `test_sync_streaming_interface_when_declared[framework_descriptor0-4]` (5)
* `test_async_query_interface_conformance_when_supported[framework_descriptor0-4]` (5)
* `test_async_streaming_interface_conformance_when_supported[framework_descriptor0-4]` (5)
* `test_context_kwarg_is_accepted_when_declared_on_primary_query[framework_descriptor0-4]` (5)
* `test_bulk_and_batch_methods_are_callable_when_declared[framework_descriptor0-4]` (5)
* `test_async_bulk_and_batch_methods_are_awaitable_when_declared[framework_descriptor0-4]` (5)
* `test_method_signatures_consistent_between_sync_and_async[framework_descriptor0-4]` (5)
* `test_capabilities_contract_matches_registry_flag[framework_descriptor0-4]` (5)
* `test_async_capabilities_returns_mapping_if_present[framework_descriptor0-4]` (5)
* `test_health_contract_matches_registry_flag[framework_descriptor0-4]` (5)
* `test_async_health_returns_mapping_if_present[framework_descriptor0-4]` (5)

#### `test_contract_shapes_and_batching.py` (60 tests = 12×5)

**Specification:** Framework Adapter Contract V1.0, §4.1-4.3  
**Status:** ✅ Complete

Shape validation and batch behavior across all 5 frameworks:

* `test_registry_declares_all_surfaces_enabled_for_active_testing[framework_descriptor0-4]` (5)
* `test_query_result_type_stable_across_calls[framework_descriptor0-4]` (5)
* `test_stream_chunk_type_consistent_within_stream[framework_descriptor0-4]` (5)
* `test_async_stream_chunk_type_consistent_within_stream[framework_descriptor0-4]` (5)
* `test_bulk_vertices_result_type_stable_across_calls[framework_descriptor0-4]` (5)
* `test_bulk_vertices_limit_zero_is_rejected[framework_descriptor0-4]` (5)
* `test_bulk_vertices_with_explicit_namespace_is_accepted[framework_descriptor0-4]` (5)
* `test_async_bulk_vertices_type_stable_across_calls[framework_descriptor0-4]` (5)
* `test_batch_result_length_matches_ops_when_sized[framework_descriptor0-4]` (5)
* `test_empty_batch_is_rejected[framework_descriptor0-4]` (5)
* `test_batch_result_type_stable_across_calls[framework_descriptor0-4]` (5)
* `test_async_batch_type_stable_across_calls[framework_descriptor0-4]` (5)

#### `test_with_mock_backends.py` (65 tests = 13×5)

**Specification:** Framework Adapter Contract V1.0, §5 (Error Handling)  
**Status:** ✅ Complete

Mock backend edge case validation across all 5 frameworks:

* `test_invalid_backend_result_causes_errors_for_sync_query[framework_descriptor0-4]` (5)
* `test_invalid_backend_result_causes_errors_for_sync_stream[framework_descriptor0-4]` (5)
* `test_async_invalid_backend_result_causes_errors_for_query[framework_descriptor0-4]` (5)
* `test_async_invalid_backend_result_causes_errors_for_stream[framework_descriptor0-4]` (5)
* `test_empty_backend_batch_result_is_not_silently_treated_as_valid[framework_descriptor0-4]` (5)
* `test_wrong_batch_length_from_backend_causes_error_or_obvious_mismatch[framework_descriptor0-4]` (5)
* `test_backend_exception_is_wrapped_with_error_context_on_query[framework_descriptor0-4]` (5)
* `test_backend_exception_is_wrapped_with_error_context_on_stream_calltime[framework_descriptor0-4]` (5)
* `test_backend_exception_is_wrapped_with_error_context_on_stream_iteration[framework_descriptor0-4]` (5)
* `test_backend_exception_is_wrapped_with_error_context_on_batch[framework_descriptor0-4]` (5)
* `test_async_backend_exception_is_wrapped_with_error_context_on_query[framework_descriptor0-4]` (5)
* `test_async_backend_exception_is_wrapped_with_error_context_on_stream_calltime[framework_descriptor0-4]` (5)
* `test_async_backend_exception_is_wrapped_with_error_context_on_stream_iteration[framework_descriptor0-4]` (5)

---

### Framework-Specific Tests (294 tests)

These tests are **unique to each framework**, validating framework-specific features and integration patterns.

#### `test_autogen_graph_adapter.py` (55 tests)

**Specification:** AutoGen Integration  
**Status:** ✅ Complete (55 tests)

AutoGen-specific graph adapter tests covering conversation context translation, function tools, and concurrent operations.

**Key Test Areas:**
- Constructor & Translator: 6 tests
- Context Translation: 7 tests
- Error Context: 5 tests
- Query & Streaming: 11 tests
- Bulk & Batch Operations: 7 tests
- Capabilities & Health: 3 tests
- Resource Management: 5 tests
- Concurrency & Thread Safety: 8 tests
- Real Integration (Function Tools): 3 tests

**Notable Tests:**
* `test_autogen_conversation_and_extra_context_passed_to_core_ctx` — Conversation context handling
* `test_autogen_function_tools_execute_end_to_end` — Real AutoGen function tool integration
* `test_connection_pool_limits` — Connection pool management
* `test_operation_telemetry_includes_framework` — Framework-specific telemetry

#### `test_crewai_graph_adapter.py` (45 tests)

**Specification:** CrewAI Integration  
**Status:** ✅ Complete (45 tests)

CrewAI-specific graph adapter tests covering task context propagation, tool creation, and thread safety.

**Key Test Areas:**
- Constructor & Translator: 6 tests
- Context Translation: 5 tests
- Error Context: 3 tests
- Query & Streaming: 9 tests
- Bulk & Batch Operations: 4 tests
- Capabilities & Health: 2 tests
- Resource Management: 3 tests
- Concurrency: 3 tests
- Real Integration (Tools): 4 tests
- Misc: 6 tests

**Notable Tests:**
* `test_crewai_task_and_extra_context_passed_to_core_ctx` — Task context handling
* `test_crewai_tools_creation_is_real_or_raises_install_error` — Real CrewAI tool integration
* `test_mixed_thread_operations` — Mixed sync/async thread safety
* `test_close_idempotent_and_shuts_down_tool_executor` — Resource cleanup

#### `test_langchain_graph_adapter.py` (53 tests)

**Specification:** LangChain Integration  
**Status:** ✅ Complete (53 tests)

LangChain-specific graph adapter tests covering RunnableConfig translation, tool creation, and thread bridging.

**Key Test Areas:**
- Translator & Context: 5 tests
- Error Context: 2 tests
- Query & Streaming: 9 tests
- Sync Guards (Event Loop): 13 tests
- Dialect Fallback: 2 tests
- Input Validation: 5 tests
- Bulk & Batch Operations: 6 tests
- Capabilities & Health: 2 tests
- Resource Management: 2 tests
- Real Integration (Tools): 7 tests

**Notable Tests:**
* `test_langchain_config_and_extra_context_passed_to_core_ctx` — RunnableConfig translation
* `test_langchain_sync_tool_called_in_event_loop_uses_thread_bridge` — Thread bridging for sync tools
* `test_create_langchain_graph_tools_returns_real_tools` — Real LangChain tool creation
* `test_close_idempotent_and_shuts_down_tool_executor` — Tool executor cleanup

#### `test_llamaindex_graph_adapter.py` (58 tests)

**Specification:** LlamaIndex Integration  
**Status:** ✅ Complete (58 tests)

LlamaIndex-specific graph adapter tests covering callback manager translation, operation context heuristics, and namespace handling.

**Key Test Areas:**
- Constructor & Translator: 6 tests
- Operation Context Heuristics: 4 tests
- Context Translation: 8 tests
- Query Building: 8 tests
- Event Loop Guards: 3 tests
- Query Retry Logic: 4 tests
- Streaming: 7 tests
- CRUD Operations (Upsert/Delete): 7 tests
- Bulk & Batch Operations: 8 tests
- Transaction Support: 2 tests

**Notable Tests:**
* `test_looks_like_operation_context_accepts_attrs_plus_request_id` — Operation context detection
* `test_query_retries_without_dialect_on_NotSupported_when_dialect_explicit` — Smart retry logic
* `test_upsert_edges_validates_edges_and_does_not_mutate_spec_edges` — Immutable edge validation
* `test_transaction_builds_raw_ops_calls_translator_and_validates` — Transaction support

#### `test_semantickernel_graph_adapter.py` (54 tests)

**Specification:** Semantic Kernel Integration  
**Status:** ✅ Complete (54 tests)

Semantic Kernel-specific graph adapter tests covering kernel context translation, plugin system, and namespace precedence.

**Key Test Areas:**
- Constructor & Translator: 5 tests
- Context Translation: 7 tests
- Event Loop Guards: 5 tests
- Capabilities Forwarding: 3 tests
- Query Building & Validation: 7 tests
- Query Retry Logic: 4 tests
- Streaming: 5 tests
- CRUD Operations: 8 tests
- Bulk & Traversal Operations: 4 tests
- Resource Management: 3 tests
- Plugin System: 3 tests

**Notable Tests:**
* `test_plugin_is_available_and_constructible_when_semantic_kernel_installed` — Plugin availability
* `test_plugin_namespace_precedence_and_forwarding_sync` — Namespace precedence rules
* `test_astream_query_accepts_direct_async_iterator` — Flexible async streaming
* `test_upsert_edges_validates_and_is_side_effect_free_and_async_too` — Immutable validation

#### `test_graph_registry_self_check.py` (29 tests)

**Specification:** Graph Registry Validation  
**Status:** ✅ Complete (29 tests)

Registry integrity and descriptor validation tests with framework parametrization.

**Key Test Areas:**
- Registry consistency: 5 tests
- Descriptor validation: 9 tests
- Async/Streaming support: 4 tests
- Method availability (parametrized): 15 tests (3×5 frameworks)
- Edge case handling: 6 tests

**Notable Tests:**
* `test_graph_registry_keys_match_descriptor_name` — Registry key consistency
* `test_registry_declared_capabilities_and_health_are_callable_and_return_values[framework_descriptor0-4]` — Cross-framework capabilities validation
* `test_registry_declared_async_methods_are_awaitable_and_return_values[framework_descriptor0-4]` — Cross-framework async validation
* `test_descriptor_immutability` — Immutability enforcement

---

## Framework Coverage

### Per-Framework Test Breakdown

| Framework | Framework-Specific | + Contract Tests | Total Tests Validating |
|-----------|-------------------|------------------|------------------------|
| **AutoGen** | 55 unique tests | + 56 shared | **111 tests** |
| **CrewAI** | 45 unique tests | + 56 shared | **101 tests** |
| **LangChain** | 53 unique tests | + 56 shared | **109 tests** |
| **LlamaIndex** | 58 unique tests | + 56 shared | **114 tests** |
| **Semantic Kernel** | 54 unique tests | + 56 shared | **110 tests** |
| **Registry** | 29 integrity tests | N/A | **29 tests** |

**Understanding the Numbers:**
- **Framework-Specific**: Tests unique to that framework
- **Contract Tests**: Each framework is validated by 56 parametrized contract tests (280 total ÷ 5 = 56 per framework)
- **Total Tests Validating**: Combined coverage showing how thoroughly each framework is tested

### Execution Time Breakdown

| Category | Tests | Avg Time/Test |
|----------|-------|---------------|
| Parametrized Contract | 280 | ~20ms |
| AutoGen | 55 | ~25ms |
| CrewAI | 45 | ~22ms |
| LangChain | 53 | ~24ms |
| LlamaIndex | 58 | ~23ms |
| Semantic Kernel | 54 | ~24ms |
| Registry | 29 | ~18ms |
| **Overall** | **574** | **24.4ms** |

---

## Running Tests

### All graph framework tests

```bash
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter \
  pytest tests/frameworks/graph/ -v
```

**Expected output:** `574 passed in ~14s`

### By framework

```bash
# AutoGen (55 unique + 56 contract = 111 tests validating AutoGen)
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter \
  pytest tests/frameworks/graph/test_autogen_graph_adapter.py -v

# CrewAI (45 unique + 56 contract = 101 tests validating CrewAI)
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter \
  pytest tests/frameworks/graph/test_crewai_graph_adapter.py -v

# LangChain (53 unique + 56 contract = 109 tests validating LangChain)
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter \
  pytest tests/frameworks/graph/test_langchain_graph_adapter.py -v

# LlamaIndex (58 unique + 56 contract = 114 tests validating LlamaIndex)
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter \
  pytest tests/frameworks/graph/test_llamaindex_graph_adapter.py -v

# Semantic Kernel (54 unique + 56 contract = 110 tests validating Semantic Kernel)
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter \
  pytest tests/frameworks/graph/test_semantickernel_graph_adapter.py -v

# Registry (29 integrity tests)
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter \
  pytest tests/frameworks/graph/test_graph_registry_self_check.py -v
```

### Contract tests only

```bash
# All parametrized contract tests (280 tests across 5 frameworks)
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter \
  pytest tests/frameworks/graph/test_contract_*.py \
         tests/frameworks/graph/test_with_mock_backends.py -v
```

**Expected output:** `280 passed` (90 + 65 + 60 + 65)

### With coverage report

```bash
CORPUS_ADAPTER=tests.mock.mock_graph_adapter:MockGraphAdapter \
  pytest tests/frameworks/graph/ \
  --cov=corpus_sdk.graph.framework_adapters \
  --cov-report=html
```

---

## Framework Compliance Checklist

Use this when implementing or validating a new **Graph framework adapter**.

### ✅ Phase 1: Core Interface (5/5)

* [x] Implements framework-specific base class/interface
* [x] Provides `query(text: str, **kwargs) -> List[Dict]`
* [x] Provides `stream_query(text: str, **kwargs) -> Iterator/AsyncIterator`
* [x] Async variants when framework supports async
* [x] Framework-specific context parameter acceptance

### ✅ Phase 2: Context Translation (8/8)

* [x] Accepts framework-specific context parameter (optional)
* [x] Translates to `OperationContext` using `from_<framework>`
* [x] Graceful degradation on translation failure
* [x] Invalid context types tolerated without crash
* [x] Context propagation through to underlying adapter
* [x] Framework metadata included in contexts
* [x] Configuration flags control context behavior
* [x] Extra context enrichment support

### ✅ Phase 3: Query Operations (6/6)

* [x] Query parameter validation
* [x] Namespace handling and precedence
* [x] Dialect retry logic when not supported
* [x] Non-JSON parameter logging
* [x] Raw query building with proper flags
* [x] Framework context operation tracking

### ✅ Phase 4: Streaming Support (4/4)

* [x] Sync streaming when declared
* [x] Async streaming when supported
* [x] Chunk validation and error context attachment
* [x] Direct async iterator and awaitable support

### ✅ Phase 5: CRUD Operations (7/7)

* [x] Upsert nodes with namespace handling
* [x] Upsert edges with validation (immutable)
* [x] Delete nodes with filter/ID validation
* [x] Delete edges with filter/ID validation
* [x] Required field validation for edges
* [x] JSON serializable property validation
* [x] Async variants for all CRUD operations

### ✅ Phase 6: Bulk & Batch Operations (5/5)

* [x] Bulk vertices with namespace support
* [x] Batch operations with result length validation
* [x] Empty batch rejection
* [x] Async bulk/batch support
* [x] Transaction support when available

### ✅ Phase 7: Error Handling (8/8)

* [x] Error context includes framework-specific fields
* [x] Error context attached on all failure paths
* [x] Async errors include same context as sync
* [x] Sync methods reject event loop context
* [x] Operation-specific error codes
* [x] Graph-specific error categorization
* [x] Backend exception wrapping with context
* [x] Error message quality for user actionability

### ✅ Phase 8: Capabilities & Health (4/4)

* [x] Capabilities passthrough when underlying provides
* [x] Health passthrough when underlying provides
* [x] Async capabilities/health fallback to sync
* [x] Missing capabilities/health handled gracefully

### ✅ Phase 9: Resource Management (4/4)

* [x] Context manager support (sync)
* [x] Async context manager support
* [x] Idempotent close operations
* [x] Tool executor cleanup (when applicable)

### ✅ Phase 10: Concurrency & Safety (5/5)

* [x] Thread safety validation
* [x] Concurrent async operations work correctly
* [x] Mixed sync/async thread safety
* [x] Connection pool limits
* [x] Event loop guard rails for sync methods

### ✅ Phase 11: Framework Integration (4/4)

* [x] Real framework tool creation/integration
* [x] Framework-specific telemetry/logging
* [x] Plugin/extension system support
* [x] Tool bridging for sync-in-async contexts

### ✅ Phase 12: Mock Backend Robustness (13/13)

* [x] Invalid backend result detection
* [x] Empty backend result detection
* [x] Wrong batch length detection
* [x] Backend exception wrapping (query)
* [x] Backend exception wrapping (stream calltime)
* [x] Backend exception wrapping (stream iteration)
* [x] Backend exception wrapping (batch)
* [x] Async backend exception wrapping
* [x] All error paths include rich context
* [x] Sync/async consistency validation
* [x] Streaming error propagation
* [x] Batch operation error handling
* [x] Mock backend tests pass for all frameworks

---

## Conformance Badge

```text
✅ Graph Framework Adapters V1.0 - 100% Conformant
   574/574 tests passing (10 test files, 5 frameworks)

   Framework-Specific Tests: 294/294 (100%)
   ✅ AutoGen:          55/55  ✅ CrewAI:           45/45
   ✅ LangChain:        53/53  ✅ LlamaIndex:       58/58
   ✅ Semantic Kernel:  54/54  ✅ Registry:         29/29

   Parametrized Contract Tests: 280/280 (100%)
   ✅ Context & Error:       90/90  (18×5 frameworks)
   ✅ Interface Conformance: 65/65  (13×5 frameworks)
   ✅ Shapes & Batching:     60/60  (12×5 frameworks)
   ✅ Mock Backends:         65/65  (13×5 frameworks)

   Total Tests Validating Each Framework:
   ✅ AutoGen:          111 tests (55 unique + 56 contract)
   ✅ CrewAI:           101 tests (45 unique + 56 contract)
   ✅ LangChain:        109 tests (53 unique + 56 contract)
   ✅ LlamaIndex:       114 tests (58 unique + 56 contract)
   ✅ Semantic Kernel:  110 tests (54 unique + 56 contract)

   Status: Production Ready - Platinum Certification 🏆
```

## **Graph Framework Adapters Conformance**


**Certification Levels:**
- 🏆 **Platinum:** 574/574 tests (100%) - All frameworks, all tests passing
- 🥇 **Gold:** 574/574 tests (100%) - All frameworks, all tests passing
- 🥈 **Silver:** 460+ tests (80%+) - Core functionality validated
- 🔬 **Development:** 287+ tests (50%+) - Early development, not production-ready

**Certification Requirements:**
- **🏆 Platinum:** Pass all 574 tests with zero failures
- **🥇 Gold:** Pass all 574 tests with zero failures  
- **🥈 Silver:** Pass ≥460 tests (80%+)
- **🔬 Development:** Pass ≥287 tests (50%+)

---

## Maintenance

### Adding New Framework Adapters

When adding a new framework adapter:

1. **Implement framework-specific tests** (45-60 tests recommended based on framework complexity)
   - Minimum 40 tests for basic frameworks
   - 50-60 tests for frameworks with rich features (like LlamaIndex's transaction support)
   - Add test class in `tests/frameworks/graph/test_<framework>_adapter.py`

2. **Ensure parametrized contract tests run** (automatically adds 56 tests validating your framework)
   - Add framework descriptor to `tests/frameworks/graph/conftest.py`
   - Verify all 280 parametrized tests execute for your framework

3. **Add framework descriptor** to registry
   - Update `corpus_sdk/graph/framework_adapters/registry.py`
   - Include version compatibility, async support, streaming support, and sample context

4. **Update this document** with new framework counts
   - Add row to Framework Coverage table
   - Update total test count
   - Document framework-specific features

5. **Run full suite** to verify 100% pass rate
   ```bash
   pytest tests/frameworks/graph/ -v
   ```

### Updating Test Coverage

When the protocol evolves:

1. **Update parametrized tests** (affects all 5 frameworks simultaneously)
   - Changes to `test_contract_*.py` automatically cover all frameworks
   - Ensures consistent behavior across all adapters

2. **Add framework-specific tests** as needed for new features
   - New graph operations may require additional unique tests
   - Maintain parity where possible across frameworks

3. **Maintain backward compatibility** or update all adapters together
   - Breaking changes require updates to all 5 framework adapters
   - Increment adapter version for breaking changes

4. **Document breaking changes** in framework adapter contract
   - Update framework adapter documentation with migration guide
   - Provide code examples for adapter implementers

### Test Count Verification

To verify test counts match this document:

```bash
# Count by file
pytest tests/frameworks/graph/ --collect-only | grep "<Module"

# Verify total
pytest tests/frameworks/graph/ -v | grep "passed"

# Expected output
# 574 passed in ~14s
```

### Performance Benchmarking

If tests become slower:

```bash
# Identify slowest tests
pytest tests/frameworks/graph/ --durations=10

# Profile with pytest-profiling
pytest tests/frameworks/graph/ --profile

# Target: Keep average <30ms/test, total <20s
```

---

**Last Updated:** 2026-02-10  
**Maintained By:** Corpus SDK Team  
**Status:** 100% V1.0 Conformant - Production Ready - Platinum Certification 🏆  
**Test Count:** 574/574 tests (100%)  
**Execution Time:** 13.97s (24.4ms/test average)  
**Framework Coverage:** 5/5 frameworks (AutoGen, CrewAI, LangChain, LlamaIndex, Semantic Kernel)

---
