# Embedding Framework Adapters Conformance Test Coverage

**Quick Stats**
- 📊 **Total Tests:** 418/418 passing (100%)
- ⚡ **Execution Time:** 22.42s (53.6ms/test avg)
- 🎯 **Frameworks:** 5 (AutoGen, CrewAI, LangChain, LlamaIndex, Semantic Kernel)
- 🏆 **Certification:** Platinum (100%)

---

**Table of Contents**
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

This document tracks conformance test coverage for **Embedding Framework Adapters V1.0** across five major AI frameworks: LangChain, LlamaIndex, Semantic Kernel, CrewAI, and AutoGen. Each adapter translates framework-specific embedding interfaces into the unified Corpus Embedding Protocol V1.0.

This suite constitutes the **official Embedding Framework Adapters V1.0 Reference Conformance Test Suite**. Any implementation (Corpus or third-party) MAY run these tests to verify and publicly claim conformance, provided all referenced tests pass unmodified.

**Adapter Version:** Embedding Framework Adapters V1.0  
**Protocol Version:** Embedding Protocol V1.0  
**Status:** Stable / Production-Ready  
**Last Updated:** 2026-01-30  
**Test Location:** `tests/frameworks/embedding/`

## Conformance Summary

**Overall Coverage: 418/418 tests (100%) ✅**

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_autogen_adapter.py` | 42 | 100% ✅ |
| `test_contract_context_and_error_context.py` | 40 (8×5) | 100% ✅ |
| `test_contract_interface_conformance.py` | 65 (13×5) | 100% ✅ |
| `test_contract_shapes_and_batching.py` | 50 (10×5) | 100% ✅ |
| `test_crewai_adapter.py` | 39 | 100% ✅ |
| `test_embedding_registry_self_check.py` | 14 | 100% ✅ |
| `test_langchain_adapter.py` | 41 | 100% ✅ |
| `test_llamaindex_adapter.py` | 38 | 100% ✅ |
| `test_semantickernel_adapter.py` | 54 | 100% ✅ |
| `test_with_mock_backends.py` | 35 (7×5) | 100% ✅ |

**Test Distribution:**

| Category | Tests | Description |
|----------|-------|-------------|
| **Parametrized Contract Tests** | 190 | Shared tests run across all 5 frameworks |
| **Framework-Specific Tests** | 228 | Unique tests per framework + registry |
| **Total** | **418** | All tests passing |

**Parametrized Contract Breakdown:**
- Context & Error Handling: 40 tests (8 functions × 5 frameworks)
- Interface Conformance: 65 tests (13 functions × 5 frameworks)
- Shapes & Batching: 50 tests (10 functions × 5 frameworks)
- Mock Backend Edge Cases: 35 tests (7 functions × 5 frameworks)

**Framework-Specific Breakdown:**
- AutoGen: 42 tests
- CrewAI: 39 tests
- LangChain: 41 tests
- LlamaIndex: 38 tests
- Semantic Kernel: 54 tests
- Registry Self-Check: 14 tests

---

## Test Files

### Parametrized Contract Tests (190 tests)

These tests run **once per framework** (5 frameworks), validating consistent behavior across all adapters.

#### `test_contract_interface_conformance.py` (65 tests = 13×5)

**Specification:** Framework Adapter Contract V1.0, §2.1-2.3  
**Status:** ✅ Complete

Validates core interface requirements and async support across all 5 frameworks:

* `test_can_instantiate_framework_adapter[framework_descriptor0-4]` (5)
* `test_async_methods_exist_when_supports_async_true[framework_descriptor0-4]` (5)
* `test_sync_embedding_interface_conformance[framework_descriptor0-4]` (5)
* `test_single_element_batch[framework_descriptor0-4]` (5)
* `test_empty_batch_handling[framework_descriptor0-4]` (5)
* `test_async_embedding_interface_conformance[framework_descriptor0-4]` (5)
* `test_context_kwarg_is_accepted_when_declared[framework_descriptor0-4]` (5)
* `test_embedding_dimension_when_required[framework_descriptor0-4]` (5)
* `test_alias_methods_exist_and_behave_consistently_when_declared[framework_descriptor0-4]` (5)
* `test_capabilities_contract_if_declared[framework_descriptor0-4]` (5)
* `test_capabilities_async_contract_if_declared[framework_descriptor0-4]` (5)
* `test_health_contract_if_declared[framework_descriptor0-4]` (5)
* `test_health_async_contract_if_declared[framework_descriptor0-4]` (5)

#### `test_contract_context_and_error_context.py` (40 tests = 8×5)

**Specification:** Framework Adapter Contract V1.0, §3.1-3.4  
**Status:** ✅ Complete

Context translation and error handling across all 5 frameworks:

* `test_rich_mapping_context_is_accepted_and_does_not_break_embeddings[framework_descriptor0-4]` (5)
* `test_invalid_context_type_is_tolerated_and_does_not_crash[framework_descriptor0-4]` (5)
* `test_context_is_optional_and_omitting_it_still_works[framework_descriptor0-4]` (5)
* `test_alias_methods_exist_and_behave_consistently_when_declared[framework_descriptor0-4]` (5)
* `test_error_context_is_attached_on_sync_batch_failure[framework_descriptor0-4]` (5)
* `test_error_context_is_attached_on_sync_query_failure[framework_descriptor0-4]` (5)
* `test_error_context_is_attached_on_async_batch_failure_when_supported[framework_descriptor0-4]` (5)
* `test_error_context_is_attached_on_async_query_failure_when_supported[framework_descriptor0-4]` (5)

#### `test_contract_shapes_and_batching.py` (50 tests = 10×5)

**Specification:** Framework Adapter Contract V1.0, §4.1-4.3  
**Status:** ✅ Complete

Shape validation and batch behavior across all 5 frameworks:

* `test_batch_output_row_count_matches_input_length[framework_descriptor0-4]` (5)
* `test_all_rows_have_consistent_dimension[framework_descriptor0-4]` (5)
* `test_query_vector_dimension_matches_batch_rows[framework_descriptor0-4]` (5)
* `test_single_element_batch_matches_query_shape[framework_descriptor0-4]` (5)
* `test_mixed_empty_and_nonempty_texts_preserve_batch_length[framework_descriptor0-4]` (5)
* `test_duplicate_texts_produce_identical_rows_within_same_batch[framework_descriptor0-4]` (5)
* `test_large_batch_shape_is_respected[framework_descriptor0-4]` (5)
* `test_batch_is_order_preserving_for_duplicates[framework_descriptor0-4]` (5)
* `test_async_batch_shape_matches_sync_when_supported[framework_descriptor0-4]` (5)
* `test_async_large_batch_shape_is_respected[framework_descriptor0-4]` (5)

#### `test_with_mock_backends.py` (35 tests = 7×5)

**Specification:** Framework Adapter Contract V1.0, §5 (Error Handling)  
**Status:** ✅ Complete

Mock backend edge case validation across all 5 frameworks:

* `test_invalid_translator_shape_causes_errors_for_batch_and_query[framework_descriptor0-4]` (5)
* `test_async_invalid_translator_shape_causes_errors_when_supported[framework_descriptor0-4]` (5)
* `test_empty_translator_result_is_not_silently_treated_as_valid_embedding[framework_descriptor0-4]` (5)
* `test_translator_returning_wrong_row_count_causes_errors_or_obvious_mismatch[framework_descriptor0-4]` (5)
* `test_translator_exception_is_wrapped_with_error_context_on_batch[framework_descriptor0-4]` (5)
* `test_translator_exception_is_wrapped_with_error_context_on_query[framework_descriptor0-4]` (5)
* `test_async_translator_exception_is_wrapped_with_error_context_when_supported[framework_descriptor0-4]` (5)

---

### Framework-Specific Tests (228 tests)

These tests are **unique to each framework**, validating framework-specific features and integration patterns.

#### `test_autogen_adapter.py` (42 tests)

**Specification:** AutoGen Integration  
**Status:** ✅ Complete (42 tests)

AutoGen-specific adapter tests covering ChromaDB integration, AutoGen context translation, and vector memory helpers.

**Key Test Areas:**
- Constructor & Configuration: 10 tests
- Input Validation: 5 tests
- Core Operations: 5 tests
- Error Context: 4 tests
- Capabilities & Health: 4 tests
- Resource Management: 2 tests
- Concurrency: 2 tests
- Real Integration (ChromaDB): 10 tests

**Notable Tests:**
* `test_real_autogen_chromadb_memory_roundtrip_uses_corpus_embeddings` — ChromaDB roundtrip validation
* `test_real_autogen_chromadb_persistence_reload_roundtrip` — Persistence across sessions
* `test_real_autogen_chromadb_collection_isolation_same_persistence_path` — Collection isolation
* `test_create_vector_memory_configures_chroma_with_custom_embedding_function` — Helper configuration

#### `test_crewai_adapter.py` (39 tests)

**Specification:** CrewAI Integration  
**Status:** ✅ Complete (39 tests)

CrewAI-specific adapter tests covering agent context propagation, task-aware batching, and crew integration.

**Key Test Areas:**
- Constructor & Configuration: 12 tests
- Core Operations: 10 tests
- Capabilities & Health: 4 tests
- Real Integration (Agents/Crews): 7 tests
- Registration: 5 tests
- Concurrency: 2 tests

**Notable Tests:**
* `test_enable_agent_context_propagation_flag_controls_operation_context_propagation` — Agent context control
* `test_task_aware_batching_sets_batch_strategy` — Task-aware batching configuration
* `TestCrewAIIntegration::test_crew_with_multiple_agents_sharing_embedder` — Multi-agent workflows
* `test_register_with_crewai_attaches_embedder_to_agents` — Agent registration

#### `test_langchain_adapter.py` (41 tests)

**Specification:** LangChain Integration  
**Status:** ✅ Complete (41 tests)

LangChain-specific adapter tests covering Pydantic validation, RunnableConfig translation, and LCEL chain integration.

**Key Test Areas:**
- Pydantic & Constructor: 7 tests
- Context Translation (RunnableConfig): 10 tests
- Input Validation & Error Context: 9 tests
- Core Operations: 6 tests
- Capabilities & Health: 4 tests
- Concurrency: 2 tests
- Real Integration (LCEL): 3 tests

**Notable Tests:**
* `test_pydantic_rejects_adapter_without_embed` — Pydantic validation
* `test_runnable_config_passed_to_context_translation` — RunnableConfig translation
* `TestLangChainIntegration::test_embeddings_work_in_runnable_chain` — LCEL chains
* `test_enable_operation_context_propagation_flag_controls_operation_context` — Context propagation control

#### `test_llamaindex_adapter.py` (38 tests)

**Specification:** LlamaIndex Integration  
**Status:** ✅ Complete (38 tests)

LlamaIndex-specific adapter tests covering embedding dimension detection, strict text validation, and node ID handling.

**Key Test Areas:**
- Constructor & Configuration: 14 tests
- Core Operations: 7 tests
- Error Context: 6 tests
- Capabilities & Health: 5 tests
- Resource Management: 2 tests
- Concurrency: 2 tests
- Real Integration: 3 tests

**Notable Tests:**
* `test_embedding_dimension_reads_from_adapter_when_available` — Auto dimension detection
* `test_get_text_embeddings_rejects_non_string_items_when_strict` — Strict validation
* `test_embedding_error_context_truncates_node_ids` — Node ID truncation
* `TestLlamaIndexIntegration::test_configure_llamaindex_embeddings_registers_settings_best_effort` — Settings registration

#### `test_semantickernel_adapter.py` (54 tests)

**Specification:** Semantic Kernel Integration  
**Status:** ✅ Complete (54 tests)

Semantic Kernel-specific adapter tests covering SK configuration, generate_embedding aliases, and kernel service registration.

**Key Test Areas:**
- Constructor & Configuration: 13 tests
- Context Translation: 8 tests
- Input Validation: 7 tests
- Core Operations: 9 tests
- Capabilities & Health: 6 tests
- Resource Management: 2 tests
- Concurrency: 2 tests
- Registration: 5 tests
- Real Integration (Kernel): 3 tests

**Notable Tests:**
* `test_sk_config_type_validation` — SK-specific config validation
* `test_generate_embeddings_rejects_non_string_items` — SK alias validation
* `test_register_with_semantic_kernel_uses_add_service_when_available` — Kernel registration
* `TestSemanticKernelIntegration::test_embeddings_work_in_sk_pipelines` — Pipeline integration

#### `test_embedding_registry_self_check.py` (14 tests)

**Specification:** Framework Registry Validation  
**Status:** ✅ Complete (14 tests)

Registry integrity and descriptor validation tests.

**Key Test Areas:**
- Registry key consistency
- Descriptor validation
- Version detection
- Availability checks
- Async method consistency
- Immutability validation
- Edge case handling

**Notable Tests:**
* `test_embedding_registry_keys_match_descriptor_name` — Key consistency
* `test_embedding_registry_descriptors_validate_cleanly` — Descriptor validation
* `test_descriptor_immutability` — Immutability enforcement
* `test_descriptor_validation_edge_cases` — Edge case handling

---

## Framework Coverage

### Per-Framework Test Breakdown

| Framework | Framework-Specific | + Contract Tests | Total Tests Validating |
|-----------|-------------------|------------------|------------------------|
| **AutoGen** | 42 unique tests | + 38 shared | **80 tests** |
| **CrewAI** | 39 unique tests | + 38 shared | **77 tests** |
| **LangChain** | 41 unique tests | + 38 shared | **79 tests** |
| **LlamaIndex** | 38 unique tests | + 38 shared | **76 tests** |
| **Semantic Kernel** | 54 unique tests | + 38 shared | **92 tests** |
| **Registry** | 14 integrity tests | N/A | **14 tests** |

**Understanding the Numbers:**
- **Framework-Specific**: Tests unique to that framework (e.g., AutoGen's ChromaDB integration)
- **Contract Tests**: Each framework is validated by 38 parametrized contract tests (190 total ÷ 5 = 38 per framework)
- **Total Tests Validating**: Combined coverage showing how thoroughly each framework is tested

**Why 418 total tests, not 404?**
The 190 parametrized tests run across all 5 frameworks simultaneously, so they're counted once (not 5 times) in the total. The math: 228 framework-specific + 190 parametrized = **418 total**.

### Execution Time Breakdown

| Category | Tests | Avg Time/Test |
|----------|-------|---------------|
| Parametrized Contract | 190 | ~50ms |
| AutoGen (ChromaDB integration) | 42 | ~80ms |
| CrewAI | 39 | ~45ms |
| LangChain | 41 | ~40ms |
| LlamaIndex | 38 | ~42ms |
| Semantic Kernel | 54 | ~48ms |
| Registry | 14 | ~35ms |
| **Overall** | **418** | **53.6ms** |

---

## Running Tests

### All embedding framework tests

```bash
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/frameworks/embedding/ -v
```

**Expected output:** `418 passed in ~22s`

### By framework

```bash
# AutoGen (42 unique + 38 contract = 80 tests validating AutoGen)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/frameworks/embedding/test_autogen_adapter.py -v

# CrewAI (39 unique + 38 contract = 77 tests validating CrewAI)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/frameworks/embedding/test_crewai_adapter.py -v

# LangChain (41 unique + 38 contract = 79 tests validating LangChain)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/frameworks/embedding/test_langchain_adapter.py -v

# LlamaIndex (38 unique + 38 contract = 76 tests validating LlamaIndex)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/frameworks/embedding/test_llamaindex_adapter.py -v

# Semantic Kernel (54 unique + 38 contract = 92 tests validating Semantic Kernel)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/frameworks/embedding/test_semantickernel_adapter.py -v

# Registry (14 integrity tests)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/frameworks/embedding/test_embedding_registry_self_check.py -v
```

### Contract tests only

```bash
# All parametrized contract tests (190 tests across 5 frameworks)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/frameworks/embedding/test_contract_*.py \
         tests/frameworks/embedding/test_with_mock_backends.py -v
```

**Expected output:** `190 passed` (40 + 65 + 50 + 35)

### With coverage report

```bash
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/frameworks/embedding/ \
  --cov=corpus_sdk.embedding.framework_adapters \
  --cov-report=html
```

---

## Framework Compliance Checklist

Use this when implementing or validating a new **Embedding framework adapter**.

### ✅ Phase 1: Core Interface (5/5)

* [x] Implements framework-specific base class/interface
* [x] Provides `embed_documents(texts: List[str]) -> List[List[float]]`
* [x] Provides `embed_query(text: str) -> List[float]`
* [x] Async variants when framework supports async
* [x] Dimension property/method when required by framework

### ✅ Phase 2: Context Translation (7/7)

* [x] Accepts framework-specific context parameter (optional)
* [x] Translates to `OperationContext` using `from_<framework>`
* [x] Graceful degradation on translation failure
* [x] Invalid context types tolerated without crash
* [x] Context propagation through to underlying adapter
* [x] Framework metadata included in contexts
* [x] Configuration flags control context behavior

### ✅ Phase 3: Input Validation (5/5)

* [x] Rejects non-string items in batch operations
* [x] Rejects non-string query text
* [x] Clear error messages for validation failures
* [x] Async methods have same validation as sync
* [x] Empty batch handling consistent

### ✅ Phase 4: Error Handling (6/6)

* [x] Error context includes framework-specific fields
* [x] Error context attached on all failure paths
* [x] Async errors include same context as sync
* [x] Sync methods reject event loop context
* [x] Error messages are actionable for users
* [x] Error inheritance hierarchy correct

### ✅ Phase 5: Shape & Batching (10/10)

* [x] Batch output length matches input length
* [x] All vectors have consistent dimension
* [x] Query dimension matches batch dimension
* [x] Single-element batch matches query shape
* [x] Empty/non-empty text mix preserves batch length
* [x] Duplicate texts produce identical vectors
* [x] Large batch shape respected
* [x] Batch preserves input order
* [x] Async batch shape matches sync
* [x] Async large batch handled correctly

### ✅ Phase 6: Capabilities & Health (4/4)

* [x] Capabilities passthrough when underlying provides
* [x] Health passthrough when underlying provides
* [x] Async capabilities/health fallback to sync
* [x] Missing capabilities/health handled gracefully

### ✅ Phase 7: Resource Management (3/3)

* [x] Context manager support (sync)
* [x] Async context manager support
* [x] Proper resource cleanup on close

### ✅ Phase 8: Concurrency (2/2)

* [x] Thread safety validated
* [x] Concurrent async operations work correctly

### ✅ Phase 9: Framework Integration (3/3)

* [x] Real framework usage tests pass
* [x] Framework-specific features work correctly
* [x] Error propagation provides actionable messages

### ✅ Phase 10: Mock Backend Robustness (7/7)

* [x] Invalid translator shapes detected
* [x] Empty translator results detected
* [x] Wrong row counts detected
* [x] Translator exceptions wrapped with context
* [x] Async translator exceptions wrapped
* [x] All error paths include rich context
* [x] Mock backend tests pass for all frameworks

---

## Conformance Badge

```text
✅ Embedding Framework Adapters V1.0 - 100% Conformant
   418/418 tests passing (10 test files, 5 frameworks)

   Framework-Specific Tests: 228/228 (100%)
   ✅ AutoGen:          42/42  ✅ CrewAI:           39/39
   ✅ LangChain:        41/41  ✅ LlamaIndex:       38/38
   ✅ Semantic Kernel:  54/54  ✅ Registry:         14/14

   Parametrized Contract Tests: 190/190 (100%)
   ✅ Context & Error:       40/40  (8×5 frameworks)
   ✅ Interface Conformance: 65/65  (13×5 frameworks)
   ✅ Shapes & Batching:     50/50  (10×5 frameworks)
   ✅ Mock Backends:         35/35  (7×5 frameworks)

   Total Tests Validating Each Framework:
   ✅ AutoGen:          80 tests (42 unique + 38 contract)
   ✅ CrewAI:           77 tests (39 unique + 38 contract)
   ✅ LangChain:        79 tests (41 unique + 38 contract)
   ✅ LlamaIndex:       76 tests (38 unique + 38 contract)
   ✅ Semantic Kernel:  92 tests (54 unique + 38 contract)

   Status: Production Ready - Platinum Certification 🏆
```

## **Embedding Framework Adapters Conformance**

**Certification Levels:**
- 🏆 **Platinum:** 418/418 tests (100%) - All frameworks, all tests passing
- 🥇 **Gold:** 334+ tests (80%+) - Most functionality validated
- 🥈 **Silver:** 209+ tests (50%+) - Core functionality validated
- 🔬 **Development:** <50% - Early development, not production-ready

**Certification Requirements:**
- **Platinum:** Pass all 418 tests with zero failures
- **Gold:** Pass ≥334 tests (all parametrized + ≥60% framework-specific)
- **Silver:** Pass ≥209 tests (all parametrized + ≥10% framework-specific)
- **Development:** Pass <50% of tests

---

## Maintenance

### Adding New Framework Adapters

When adding a new framework adapter:

1. **Implement framework-specific tests** (35-55 tests recommended based on framework complexity)
   - Minimum 30 tests for basic frameworks
   - 40-50 tests for frameworks with rich features (like AutoGen's ChromaDB)
   - Add test class in `tests/frameworks/embedding/test_<framework>_adapter.py`

2. **Ensure parametrized contract tests run** (automatically adds 38 tests validating your framework)
   - Add framework descriptor to `tests/frameworks/embedding/conftest.py`
   - Verify all 190 parametrized tests execute for your framework

3. **Add framework descriptor** to registry
   - Update `corpus_sdk/embedding/framework_adapters/registry.py`
   - Include version compatibility, async support, and sample context

4. **Update this document** with new framework counts
   - Add row to Framework Coverage table
   - Update total test count
   - Document framework-specific features

5. **Run full suite** to verify 100% pass rate
   ```bash
   pytest tests/frameworks/embedding/ -v
   ```

### Updating Test Coverage

When the protocol evolves:

1. **Update parametrized tests** (affects all 5 frameworks simultaneously)
   - Changes to `test_contract_*.py` automatically cover all frameworks
   - Ensures consistent behavior across all adapters

2. **Add framework-specific tests** as needed for new features
   - New framework capabilities may require additional unique tests
   - Maintain parity where possible (e.g., if LangChain adds streaming, add to others)

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
pytest tests/frameworks/embedding/ --collect-only | grep "<Module"

# Verify total
pytest tests/frameworks/embedding/ -v | grep "passed"

# Expected output
# 418 passed in ~22s
```

### Performance Benchmarking

If tests become slower:

```bash
# Identify slowest tests
pytest tests/frameworks/embedding/ --durations=10

# Profile with pytest-profiling
pytest tests/frameworks/embedding/ --profile

# Target: Keep average <100ms/test, total <30s
```

---

**Last Updated:** 2026-01-30  
**Maintained By:** Corpus SDK Team  
**Status:** 100% V1.0 Conformant - Production Ready - Platinum Certification 🏆  
**Test Count:** 418/418 tests (100%)  
**Execution Time:** 22.42s (53.6ms/test average)  
**Framework Coverage:** 5/5 frameworks (AutoGen, CrewAI, LangChain, LlamaIndex, Semantic Kernel)
