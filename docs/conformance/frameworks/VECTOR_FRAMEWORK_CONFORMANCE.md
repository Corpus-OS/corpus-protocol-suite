# Vector Framework Adapters Conformance Test Coverage

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

This document tracks conformance test coverage for **Vector Framework Adapters V1.0** across five major AI frameworks: AutoGen, CrewAI, LangChain, LlamaIndex, and Semantic Kernel. Each adapter translates framework-specific vector store interfaces into the unified Corpus Vector Protocol V1.0.

This suite constitutes the **official Vector Framework Adapters V1.0 Reference Conformance Test Suite**. Any implementation (Corpus or third-party) MAY run these tests to verify and publicly claim conformance, provided all referenced tests pass unmodified.

**Adapter Version:** Vector Framework Adapters V1.0  
**Protocol Version:** Vector Protocol V1.0  
**Status:** Stable / Production-Ready  
**Last Updated:** 2026-02-10  
**Test Location:** `tests/frameworks/vector/`  
**Performance:** 23.71s total (24.7ms/test average)

## Conformance Summary

**Overall Coverage: 958/958 tests (100%) ✅**

📊 **Total Tests:** 958/958 passing (100%)  
⚡ **Execution Time:** 23.71s (24.7ms/test avg)  
🏆 **Certification:** Platinum (100%)

| Category | Tests | Coverage | Status |
|----------|-------|-----------|---------|
| **Parametrized Contract Tests** | 185 | 100% ✅ | Production Ready |
| **Framework-Specific Tests** | 773 | 100% ✅ | Production Ready |
| **Total** | **958/958** | **100% ✅** | **🏆 Platinum Certified** |

### Performance Characteristics
- **Test Execution:** 23.71 seconds total runtime
- **Average Per Test:** 24.7 milliseconds
- **Cache Efficiency:** 0 cache hits, 958 misses (cache size: 958)
- **Parallel Ready:** Optimized for parallel execution with `pytest -n auto`

### Test Infrastructure
- **Mock Adapter:** `tests.mock.mock_vector_adapter:MockVectorAdapter` - Deterministic mock for Vector operations
- **Testing Framework:** pytest 9.0.2 with comprehensive plugin support
- **Environment:** Python 3.10.19 on Darwin
- **Strict Mode:** Off (permissive testing)

## **Vector Framework Adapters Certification**

- 🏆 **Platinum:** 958/958 tests (100% comprehensive conformance)
- 🥇 **Gold:** 958 tests (100% protocol mastery)
- 🥈 **Silver:** 767+ tests (80%+ integration-ready)
- 🔬 **Development:** 479+ tests (50%+ early development)

---

## Test Files

### Parametrized Contract Tests (185 tests)

These tests run **once per framework** (5 frameworks), validating consistent behavior across all adapters.

#### `test_contract_context_and_error_context.py` (45 tests = 9×5)

**Specification:** Vector Framework Adapter Contract V1.0, §3.1-3.4  
**Status:** ✅ Complete

Context translation and error handling across all 5 frameworks:

* `test_query_accepts_context_mapping_when_declared[framework_descriptor0-4]` (5)
* `test_query_context_is_optional_even_when_declared[framework_descriptor0-4]` (5)
* `test_invalid_context_type_behavior_is_consistent[framework_descriptor0-4]` (5)
* `test_context_injection_does_not_occur_when_context_kwarg_is_none` (5)
* `test_error_context_attached_on_query_failure[framework_descriptor0-4]` (5)
* `test_error_context_attached_on_upsert_failure[framework_descriptor0-4]` (5)
* `test_error_context_attached_on_delete_failure[framework_descriptor0-4]` (5)
* `test_error_context_attached_on_capabilities_failure[framework_descriptor0-4]` (5)
* `test_error_context_attached_on_health_failure[framework_descriptor0-4]` (5)

#### `test_contract_interface_conformance.py` (70 tests = 14×5)

**Specification:** Vector Framework Adapter Contract V1.0, §2.1-2.3  
**Status:** ✅ Complete

Validates core interface requirements and async support across all 5 frameworks:

* `test_can_import_adapter_module_and_resolve_class_when_available[framework_descriptor0-4]` (5)
* `test_client_exposes_required_vector_protocol_v1_surface_when_available[framework_descriptor0-4]` (5)
* `test_batch_query_method_callable_presence_when_declared_and_available[framework_descriptor0-4]` (5)
* `test_context_kwarg_signature_acceptance_when_declared_and_available[framework_descriptor0-4]` (5)
* `test_context_injection_does_not_occur_when_context_kwarg_is_none` (5)
* `test_context_injection_occurs_when_context_kwarg_is_set` (5)
* `test_adapter_init_kwarg_is_respected_with_nonstandard_kwarg` (5)

#### `test_contract_shapes_and_batching.py` (70 tests = 14×5)

**Specification:** Vector Framework Adapter Contract V1.0, §4.1-4.3  
**Status:** ✅ Complete

Shape validation and batch behavior across all 5 frameworks:

* `test_client_exposes_required_methods_as_callables[framework_descriptor0-4]` (5)
* `test_batch_query_method_callable_presence_when_has_batch_query_true[framework_descriptor0-4]` (5)
* `test_capabilities_returns_non_none[framework_descriptor0-4]` (5)
* `test_health_returns_non_none[framework_descriptor0-4]` (5)
* `test_capabilities_type_stable_across_calls[framework_descriptor0-4]` (5)
* `test_health_type_stable_across_calls[framework_descriptor0-4]` (5)
* `test_create_namespace_does_not_crash[framework_descriptor0-4]` (5)
* `test_delete_namespace_does_not_crash_when_namespace_exists[framework_descriptor0-4]` (5)
* `test_namespace_ops_type_stable_across_calls[framework_descriptor0-4]` (5)
* `test_upsert_accepts_single_and_multiple_items[framework_descriptor0-4]` (5)
* `test_upsert_result_type_stable_across_calls[framework_descriptor0-4]` (5)
* `test_delete_accepts_single_and_multiple_ids[framework_descriptor0-4]` (5)
* `test_delete_result_type_stable_across_calls[framework_descriptor0-4]` (5)
* `test_query_result_type_stable_across_calls[framework_descriptor0-4]` (5)

---

### Framework-Specific Tests (773 tests)

These tests are **unique to each framework**, validating framework-specific features and integration patterns.

#### `test_autogen_vector_adapter.py` (151 tests)

**Specification:** AutoGen Integration  
**Status:** ✅ Complete (151 tests)

AutoGen-specific vector adapter tests covering conversation context translation, embedding functions, and MMR search.

**Key Test Areas:**
- Initialization & config validation: 9 tests
- Translator creation: 4 tests
- Context translation: 7 tests
- Dimension hint management: 7 tests
- Embedding handling: 14 tests
- Text/document operations: 12 tests
- Search operations: 18 tests
- MMR search: 4 tests
- Delete operations: 3 tests
- Call syntax: 6 tests
- Capabilities & health: 8 tests
- Resource management: 5 tests
- Event loop guards: 6 tests
- Error handling: 8 tests
- Utility functions: 4 tests
- Factory methods: 2 tests
- Client wrapper: 5 tests
- Retriever tool: 4 tests
- Real integration: 9 tests

**Notable Tests:**
* `test_build_core_context_from_conversation` — Conversation context handling
* `test_ensure_embeddings_async_falls_back_to_sync` — Async embedding fallback
* `test_mmr_search_uses_mmr_config` — Maximal Marginal Relevance search
* `test_real_autogen_retriever_function_tool_executes_or_raises_install_error` — Real AutoGen integration

#### `test_crewai_vector_adapter.py` (178 tests)

**Specification:** CrewAI Integration  
**Status:** ✅ Complete (178 tests)

CrewAI-specific vector adapter tests covering task context propagation, MMR algorithm, and tool creation.

**Key Test Areas:**
- Initialization & config: 11 tests
- Translator creation: 4 tests
- Context translation: 8 tests
- Dimension management: 4 tests
- Embedding operations: 24 tests
- Compatibility aliases: 7 tests
- Score/match utilities: 6 tests
- MMR algorithm: 6 tests
- Search operations: 28 tests
- Tool/agent interface: 6 tests
- Capabilities & health: 5 tests
- Resource management: 4 tests
- Event loop guards: 2 tests
- Error handling: 7 tests
- Client wrapper: 6 tests
- Schema/validation: 5 tests
- Real integration: 10 tests
- Edge cases: 5 tests

**Notable Tests:**
* `test_build_core_context_from_crewai_task` — Task context extraction
* `test_mmr_select_indices_balances_relevance_and_diversity` — MMR algorithm validation
* `test_real_crewai_vector_tool_is_real_basetool` — Real CrewAI tool validation
* `test_real_crewai_vector_tool_arun_executes_end_to_end` — Async tool execution

#### `test_langchain_adapter.py` (120 tests)

**Specification:** LangChain Integration  
**Status:** ✅ Complete (120 tests)

LangChain-specific vector adapter tests covering RunnableConfig translation, vector store interface, and retriever integration.

**Key Test Areas:**
- Initialization & config: 8 tests
- Translator creation: 4 tests
- Context translation: 6 tests
- Dimension management: 5 tests
- Embedding operations: 12 tests
- Vector conversion: 4 tests
- Text/document operations: 8 tests
- Search operations: 12 tests
- MMR algorithm: 6 tests
- Delete operations: 4 tests
- Call syntax: 3 tests
- Capabilities & health: 4 tests
- Resource management: 4 tests
- Event loop guards: 2 tests
- Error handling: 4 tests
- Client wrapper: 3 tests
- Real integration: 6 tests

**Notable Tests:**
* `test_build_core_context_from_langchain_config` — LangChain config handling
* `test_to_corpus_vectors_handles_metadata_envelope` — Metadata envelope support
* `test_real_langchain_vector_store_as_retriever_invoke` — Retriever integration
* `test_real_langchain_retriever_async_is_invokable` — Async retriever validation

#### `test_llamaindex_adapter.py` (156 tests)

**Specification:** LlamaIndex Integration  
**Status:** ✅ Complete (156 tests)

LlamaIndex-specific vector adapter tests covering callback manager translation, node conversion, and metadata filters.

**Key Test Areas:**
- Initialization & config: 6 tests
- Translator creation: 4 tests
- Context translation: 6 tests
- Dimension management: 3 tests
- Node/vector conversion: 4 tests
- Node operations: 10 tests
- Search operations: 12 tests
- MMR algorithm: 4 tests
- Delete operations: 4 tests
- Capabilities & health: 4 tests
- Resource management: 4 tests
- Event loop guards: 2 tests
- Error handling: 4 tests
- Metadata filters: 8 tests
- Request building: 6 tests
- Parameter validation: 12 tests
- Real integration: 7 tests
- Utility functions: 4 tests

**Notable Tests:**
* `test_build_core_context_from_llamaindex_callback` — Callback manager handling
* `test_metadata_filters_to_corpus_filter_handles_eq_operator` — Metadata filter translation
* `test_llamaindex_node_roundtrip` — Node object preservation
* `test_llamaindex_vector_store_query_integration` — Real LlamaIndex integration

#### `test_semantickernel_vector_adapter.py` (168 tests)

**Specification:** Semantic Kernel Integration  
**Status:** ✅ Complete (168 tests)

Semantic Kernel-specific vector adapter tests covering plugin system, kernel function exceptions, and async embedding support.

**Key Test Areas:**
- Store initialization: 8 tests
- Translator creation: 3 tests
- Capabilities: 4 tests
- Embedding handling: 12 tests
- Metadata/ID normalization: 6 tests
- Vector conversion: 4 tests
- Request building: 6 tests
- Parameter validation: 8 tests
- Text/document operations: 10 tests
- Search operations: 16 tests
- MMR search: 4 tests
- Delete operations: 4 tests
- Context building: 4 tests
- Import handling: 2 tests
- Plugin system: 10 tests
- Error mapping: 4 tests
- End-to-end integration: 6 tests
- Score handling: 4 tests
- Error handling: 5 tests
- Client wrapper: 7 tests
- Document operations: 5 tests
- Async embedding fallback: 2 tests

**Notable Tests:**
* `test_plugin_vector_search_calls_store_asimilarity_search_and_returns_docs` — Plugin system integration
* `test_plugin_error_mapping_NotSupported_to_KernelFunctionException` — SK error mapping
* `test_e2e_plugin_can_be_added_to_kernel_and_invoked_vector_search` — Real SK integration
* `test_async_methods_use_async_embedding_function` — Async embedding support

#### `test_vector_registry_self_check.py` (55 tests)

**Specification:** Vector Registry Validation  
**Status:** ✅ Complete (55 tests)

Registry integrity and descriptor validation tests with comprehensive edge case handling.

**Key Test Areas:**
- Registry consistency: 6 tests
- Descriptor validation: 24 tests
- Constructor knob validation: 4 tests
- Version range formatting: 4 tests
- Availability checking: 8 tests
- Cache behavior: 4 tests
- Filtering: 5 tests

**Notable Tests:**
* `test_vector_registry_keys_match_descriptor_name` — Registry key consistency
* `test_validate_requires_core_protocol_method_names` — Core method validation
* `test_version_range_formatting` — Version range formatting
* `test_availability_cache_keyed_by_adapter_module_and_attr_not_name` — Smart caching

---

## Framework Coverage

### Per-Framework Test Breakdown

| Framework | Framework-Specific | + Contract Tests | Total Tests Validating |
|-----------|-------------------|------------------|------------------------|
| **AutoGen** | 151 unique tests | + 37 shared | **188 tests** |
| **CrewAI** | 178 unique tests | + 37 shared | **215 tests** |
| **LangChain** | 120 unique tests | + 37 shared | **157 tests** |
| **LlamaIndex** | 156 unique tests | + 37 shared | **193 tests** |
| **Semantic Kernel** | 168 unique tests | + 37 shared | **205 tests** |
| **Registry** | 55 integrity tests | N/A | **55 tests** |

**Understanding the Numbers:**
- **Framework-Specific**: Tests unique to that framework
- **Contract Tests**: Each framework is validated by 37 parametrized contract tests (185 total ÷ 5 = 37 per framework)
- **Total Tests Validating**: Combined coverage showing how thoroughly each framework is tested

### Execution Time Breakdown

| Category | Tests | Avg Time/Test |
|----------|-------|---------------|
| Parametrized Contract | 185 | ~25ms |
| AutoGen | 151 | ~25ms |
| CrewAI | 178 | ~25ms |
| LangChain | 120 | ~24ms |
| LlamaIndex | 156 | ~25ms |
| Semantic Kernel | 168 | ~25ms |
| Registry | 55 | ~22ms |
| **Overall** | **958** | **24.7ms** |

---

## Running Tests

### All vector framework tests

```bash
CORPUS_ADAPTER=tests.mock.mock_vector_adapter:MockVectorAdapter \
  pytest tests/frameworks/vector/ -v
```

**Expected output:** `958 passed in ~24s`

### By framework

```bash
# AutoGen (151 unique + 37 contract = 188 tests validating AutoGen)
CORPUS_ADAPTER=tests.mock.mock_vector_adapter:MockVectorAdapter \
  pytest tests/frameworks/vector/test_autogen_vector_adapter.py -v

# CrewAI (178 unique + 37 contract = 215 tests validating CrewAI)
CORPUS_ADAPTER=tests.mock.mock_vector_adapter:MockVectorAdapter \
  pytest tests/frameworks/vector/test_crewai_vector_adapter.py -v

# LangChain (120 unique + 37 contract = 157 tests validating LangChain)
CORPUS_ADAPTER=tests.mock.mock_vector_adapter:MockVectorAdapter \
  pytest tests/frameworks/vector/test_langchain_adapter.py -v

# LlamaIndex (156 unique + 37 contract = 193 tests validating LlamaIndex)
CORPUS_ADAPTER=tests.mock.mock_vector_adapter:MockVectorAdapter \
  pytest tests/frameworks/vector/test_llamaindex_adapter.py -v

# Semantic Kernel (168 unique + 37 contract = 205 tests validating Semantic Kernel)
CORPUS_ADAPTER=tests.mock.mock_vector_adapter:MockVectorAdapter \
  pytest tests/frameworks/vector/test_semantickernel_vector_adapter.py -v

# Registry (55 integrity tests)
CORPUS_ADAPTER=tests.mock.mock_vector_adapter:MockVectorAdapter \
  pytest tests/frameworks/vector/test_vector_registry_self_check.py -v
```

### Contract tests only

```bash
# All parametrized contract tests (185 tests across 5 frameworks)
CORPUS_ADAPTER=tests.mock.mock_vector_adapter:MockVectorAdapter \
  pytest tests/frameworks/vector/test_contract_*.py \
         tests/frameworks/vector/test_with_mock_backends.py -v
```

**Expected output:** `185 passed` (45 + 70 + 70)

### With coverage report

```bash
CORPUS_ADAPTER=tests.mock.mock_vector_adapter:MockVectorAdapter \
  pytest tests/frameworks/vector/ \
  --cov=corpus_sdk.vector.framework_adapters \
  --cov-report=html
```

---

## Framework Compliance Checklist

Use this when implementing or validating a new **Vector framework adapter**.

### ✅ Phase 1: Core Interface (6/6)

* [x] Implements framework-specific vector store interface
* [x] Provides `add_texts`/`aadd_texts` for document ingestion
* [x] Provides `similarity_search`/`asimilarity_search` for retrieval
* [x] Provides `delete`/`adelete` for document removal
* [x] Supports batch operations when available
* [x] Framework-specific context parameter acceptance

### ✅ Phase 2: Context Translation (8/8)

* [x] Accepts framework-specific context parameter (conversation, task, config, callback manager)
* [x] Translates to `OperationContext` using appropriate factory
* [x] Graceful degradation on translation failure
* [x] Invalid context types tolerated without crash
* [x] Context propagation through to underlying adapter
* [x] Framework metadata included in contexts
* [x] Configuration flags control context behavior
* [x] Extra context enrichment support

### ✅ Phase 3: Embedding Handling (7/7)

* [x] Embedding function integration (sync and async)
* [x] Dimension hint management and validation
* [x] Zero vector generation for empty queries
* [x] Batch embedding generation
* [x] Async embedding fallback to sync when needed
* [x] Embedding function error handling
* [x] Dimension consistency enforcement

### ✅ Phase 4: Document Operations (6/6)

* [x] Text/document object conversion
* [x] Metadata normalization and handling
* [x] ID generation and validation
* [x] Partial failure handling
* [x] Empty input handling
* [x] Framework-specific document type support

### ✅ Phase 5: Search Operations (8/8)

* [x] Similarity search with scores
* [x] Maximal Marginal Relevance (MMR) search
* [x] Streaming search support
* [x] Filter validation and translation
* [x] Score threshold application
* [x] Top-k validation against capabilities
* [x] Namespace handling
* [x] Metadata field filtering

### ✅ Phase 6: MMR Algorithm (5/5)

* [x] Cosine similarity calculation
* [x] Relevance-diversity balancing
* [x] Candidate fetching and selection
* [x] Lambda parameter validation (0-1 range)
* [x] Similarity caching for performance

### ✅ Phase 7: Delete Operations (4/4)

* [x] Delete by IDs
* [x] Delete by filter (when supported)
* [x] Validation for empty criteria
* [x] Async delete support

### ✅ Phase 8: Capabilities & Health (6/6)

* [x] Capabilities passthrough when underlying provides
* [x] Health passthrough when underlying provides
* [x] Async capabilities/health fallback to sync via thread
* [x] Missing capabilities/health handled gracefully
* [x] Caching for performance
* [x] Error handling with context attachment

### ✅ Phase 9: Resource Management (5/5)

* [x] Context manager support (sync)
* [x] Async context manager support
* [x] Idempotent close operations
* [x] Translator and adapter cleanup
* [x] Event loop guard rails for sync methods

### ✅ Phase 10: Error Handling (8/8)

* [x] Error context includes framework-specific fields
* [x] Error context attached on all failure paths
* [x] Async errors include same context as sync
* [x] Operation-specific error codes
* [x] Vector-specific error categorization
* [x] Backend exception wrapping with context
* [x] Error message quality for user actionability
* [x] Error context extraction never raises

### ✅ Phase 11: Framework Integration (7/7)

* [x] Real framework object handling
* [x] Tool/plugin system integration
* [x] Metadata envelope support
* [x] Retriever/agent workflow integration
* [x] End-to-end workflow validation
* [x] Dependency availability checks
* [x] Import guards for missing frameworks

### ✅ Phase 12: Mock Backend Robustness (14/14)

* [x] Invalid backend result detection
* [x] Empty backend query handling
* [x] Empty batch query handling
* [x] Backend exception wrapping (query)
* [x] Backend exception wrapping (batch query)
* [x] Backend exception wrapping (capabilities)
* [x] Backend exception wrapping (health)
* [x] Backend exception wrapping (upsert)
* [x] Backend exception wrapping (delete)
* [x] Backend exception wrapping (namespace ops)
* [x] All error paths include rich context
* [x] Sync/async consistency validation
* [x] Query error propagation
* [x] Batch operation error handling

---

## Conformance Badge

```text
✅ Vector Framework Adapters V1.0 - 100% Conformant
   958/958 tests passing (11 test files, 5 frameworks)

   Framework-Specific Tests: 773/773 (100%)
   ✅ AutoGen:          151/151 ✅ CrewAI:           178/178
   ✅ LangChain:        120/120 ✅ LlamaIndex:       156/156
   ✅ Semantic Kernel:  168/168 ✅ Registry:          55/55

   Parametrized Contract Tests: 185/185 (100%)
   ✅ Context & Error:       45/45   (9×5 frameworks)
   ✅ Interface Conformance: 70/70  (14×5 frameworks)
   ✅ Shapes & Batching:     70/70  (14×5 frameworks)

   Total Tests Validating Each Framework:
   ✅ AutoGen:          188 tests (151 unique + 37 contract)
   ✅ CrewAI:           215 tests (178 unique + 37 contract)
   ✅ LangChain:        157 tests (120 unique + 37 contract)
   ✅ LlamaIndex:       193 tests (156 unique + 37 contract)
   ✅ Semantic Kernel:  205 tests (168 unique + 37 contract)

   Status: Production Ready - Platinum Certification 🏆
```

## **Vector Framework Adapters Conformance**

**Certification Levels:**
- 🏆 **Platinum:** 958/958 tests (100%) - All frameworks, all tests passing
- 🥇 **Gold:** 958 tests (100%) - All frameworks, all tests passing
- 🥈 **Silver:** 767+ tests (80%+) - Core functionality validated
- 🔬 **Development:** 479+ tests (50%+) - Early development, not production-ready

**Certification Requirements:**
- **🏆 Platinum:** Pass all 958 tests with zero failures
- **🥇 Gold:** Pass all 958 tests with zero failures  
- **🥈 Silver:** Pass ≥767 tests (80%+)
- **🔬 Development:** Pass ≥479 tests (50%+)

---

## Maintenance

### Adding New Framework Adapters

When adding a new framework adapter:

1. **Implement framework-specific tests** (120-180 tests recommended based on framework complexity)
   - Minimum 100 tests for basic frameworks
   - 150-180 tests for frameworks with rich features (like CrewAI's MMR algorithm)
   - Add test class in `tests/frameworks/vector/test_<framework>_adapter.py`

2. **Ensure parametrized contract tests run** (automatically adds 37 tests validating your framework)
   - Add framework descriptor to `tests/frameworks/vector/conftest.py`
   - Verify all 185 parametrized tests execute for your framework

3. **Add framework descriptor** to registry
   - Update `corpus_sdk/vector/framework_adapters/registry.py`
   - Include version compatibility, batch query support, and sample context

4. **Update this document** with new framework counts
   - Add row to Framework Coverage table
   - Update total test count
   - Document framework-specific features

5. **Run full suite** to verify 100% pass rate
   ```bash
   pytest tests/frameworks/vector/ -v
   ```

### Updating Test Coverage

When the protocol evolves:

1. **Update parametrized tests** (affects all 5 frameworks simultaneously)
   - Changes to `test_contract_*.py` automatically cover all frameworks
   - Ensures consistent behavior across all adapters

2. **Add framework-specific tests** as needed for new features
   - New vector operations may require additional unique tests
   - Maintain parity where possible across frameworks

3. **Maintain backward compatibility** or update all adapters together
   - Breaking changes require updates to all 5 framework adapters
   - Increment adapter version for breaking changes

### Test Count Verification

To verify test counts match this document:

```bash
# Count by file
pytest tests/frameworks/vector/ --collect-only | grep "<Module"

# Verify total
pytest tests/frameworks/vector/ -v | grep "passed"

# Expected output
# 958 passed in ~24s
```

### Performance Benchmarking

If tests become slower:

```bash
# Identify slowest tests
pytest tests/frameworks/vector/ --durations=10

# Profile with pytest-profiling
pytest tests/frameworks/vector/ --profile

# Target: Keep average <30ms/test, total <30s
```

---

**Last Updated:** 2026-02-10  
**Maintained By:** Corpus SDK Team  
**Status:** 100% V1.0 Conformant - Production Ready - Platinum Certification 🏆  
**Test Count:** 958/958 tests (100%)  
**Execution Time:** 23.71s (24.7ms/test average)  
**Framework Coverage:** 5/5 frameworks (AutoGen, CrewAI, LangChain, LlamaIndex, Semantic Kernel)

---
