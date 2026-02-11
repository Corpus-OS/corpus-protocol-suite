# LLM Framework Adapters Conformance Test Coverage

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

This document tracks conformance test coverage for **LLM Framework Adapters V1.0** across five major AI frameworks: AutoGen, CrewAI, LangChain, LlamaIndex, and Semantic Kernel. Each adapter translates framework-specific LLM interfaces into the unified Corpus LLM Protocol V1.0.

This suite constitutes the **official LLM Framework Adapters V1.0 Reference Conformance Test Suite**. Any implementation (Corpus or third-party) MAY run these tests to verify and publicly claim conformance, provided all referenced tests pass unmodified.

**Adapter Version:** LLM Framework Adapters V1.0  
**Protocol Version:** LLM Protocol V1.0  
**Status:** Stable / Production-Ready  
**Last Updated:** 2026-02-10  
**Test Location:** `tests/frameworks/llm/`  
**Performance:** 20.50s total (32.8ms/test average)

## Conformance Summary

**Overall Coverage: 624/624 tests (100%) ✅**

📊 **Total Tests:** 624/624 passing (100%)  
⚡ **Execution Time:** 20.50s (32.8ms/test avg)  
🏆 **Certification:** Platinum (100%)

| Category | Tests | Coverage | Status |
|----------|-------|-----------|---------|
| **Parametrized Contract Tests** | 210 | 100% ✅ | Production Ready |
| **Framework-Specific Tests** | 414 | 100% ✅ | Production Ready |
| **Total** | **624/624** | **100% ✅** | **🏆 Platinum Certified** |

### Performance Characteristics
- **Test Execution:** 20.50 seconds total runtime
- **Average Per Test:** 32.8 milliseconds
- **Cache Efficiency:** 0 cache hits, 624 misses (cache size: 624)
- **Parallel Ready:** Optimized for parallel execution with `pytest -n auto`

### Test Infrastructure
- **Mock Adapter:** `tests.mock.mock_llm_adapter:MockLLMAdapter` - Deterministic mock for LLM operations
- **Testing Framework:** pytest 9.0.2 with comprehensive plugin support
- **Environment:** Python 3.10.19 on Darwin
- **Strict Mode:** Off (permissive testing)

## **LLM Framework Adapters Certification**

- 🏆 **Platinum:** 624/624 tests (100% comprehensive conformance)
- 🥇 **Gold:** 624 tests (100% protocol mastery)
- 🥈 **Silver:** 500+ tests (80%+ integration-ready)
- 🔬 **Development:** 312+ tests (50%+ early development)

---

## Test Files

### Parametrized Contract Tests (210 tests)

These tests run **once per framework** (5 frameworks), validating consistent behavior across all adapters.

#### `test_contract_context_and_error_context.py` (60 tests = 12×5)

**Specification:** LLM Framework Adapter Contract V1.0, §3.1-3.4  
**Status:** ✅ Complete

Context translation and error handling across all 5 frameworks:

* `test_registry_declared_methods_exist_and_are_callable_when_available[framework_descriptor0-4]` (5)
* `test_registry_flags_are_coherent_with_declared_methods_when_available[framework_descriptor0-4]` (5)
* `test_rich_mapping_context_is_accepted_across_all_registry_declared_sync_methods[framework_descriptor0-4]` (5)
* `test_invalid_context_is_tolerated_across_all_registry_declared_sync_methods[framework_descriptor0-4]` (5)
* `test_context_is_optional_across_all_registry_declared_sync_methods[framework_descriptor0-4]` (5)
* `test_rich_mapping_context_is_accepted_across_all_registry_declared_async_methods[framework_descriptor0-4]` (5)
* `test_invalid_context_is_tolerated_across_all_registry_declared_async_methods[framework_descriptor0-4]` (5)
* `test_context_is_optional_across_all_registry_declared_async_methods[framework_descriptor0-4]` (5)
* `test_error_context_is_attached_on_sync_failure_for_all_registry_declared_methods[framework_descriptor0-4]` (5)
* `test_error_context_is_attached_on_sync_stream_calltime_failure_when_supported[framework_descriptor0-4]` (5)
* `test_error_context_is_attached_on_async_failure_for_all_registry_declared_methods[framework_descriptor0-4]` (5)

#### `test_contract_interface_conformance.py` (85 tests = 17×5)

**Specification:** LLM Framework Adapter Contract V1.0, §2.1-2.3  
**Status:** ✅ Complete

Validates core interface requirements and async support across all 5 frameworks:

* `test_can_instantiate_llm_client[framework_descriptor0-4]` (5)
* `test_registry_flags_are_coherent_with_declared_methods_when_available[framework_descriptor0-4]` (5)
* `test_async_methods_exist_when_supports_async_true[framework_descriptor0-4]` (5)
* `test_sync_completion_interface_conformance_when_declared[framework_descriptor0-4]` (5)
* `test_sync_streaming_interface_when_method_declared[framework_descriptor0-4]` (5)
* `test_sync_streaming_via_kwarg_when_declared[framework_descriptor0-4]` (5)
* `test_async_completion_interface_conformance_when_declared[framework_descriptor0-4]` (5)
* `test_async_streaming_interface_when_method_declared[framework_descriptor0-4]` (5)
* `test_async_streaming_via_kwarg_when_declared[framework_descriptor0-4]` (5)
* `test_context_kwarg_is_accepted_when_declared[framework_descriptor0-4]` (5)
* `test_method_signatures_consistent_between_sync_and_async_when_comparable[framework_descriptor0-4]` (5)
* `test_token_count_contract_matches_registry_flag[framework_descriptor0-4]` (5)
* `test_async_token_count_returns_int_when_declared[framework_descriptor0-4]` (5)
* `test_capabilities_contract_matches_registry_flag[framework_descriptor0-4]` (5)
* `test_async_capabilities_returns_mapping_if_present[framework_descriptor0-4]` (5)
* `test_health_contract_matches_registry_flag[framework_descriptor0-4]` (5)
* `test_async_health_returns_mapping_if_present[framework_descriptor0-4]` (5)

#### `test_contract_shapes_and_batching.py` (65 tests = 13×5)

**Specification:** LLM Framework Adapter Contract V1.0, §4.1-4.3  
**Status:** ✅ Complete

Shape validation and type stability across all 5 frameworks:

* `test_sync_completion_result_type_stable_across_calls[framework_descriptor0-4]` (5)
* `test_async_completion_result_type_stable_across_calls_when_supported[framework_descriptor0-4]` (5)
* `test_sync_and_async_completion_result_types_match_when_both_declared[framework_descriptor0-4]` (5)
* `test_stream_chunk_type_consistent_within_stream_when_supported[framework_descriptor0-4]` (5)
* `test_async_stream_chunk_type_consistent_within_stream_when_supported[framework_descriptor0-4]` (5)
* `test_sync_stream_first_chunk_type_stable_across_calls_when_supported[framework_descriptor0-4]` (5)
* `test_async_stream_first_chunk_type_stable_across_calls_when_supported[framework_descriptor0-4]` (5)
* `test_stream_first_chunk_type_matches_between_sync_and_async_when_both_declared[framework_descriptor0-4]` (5)
* `test_streaming_surface_is_resolvable_when_supports_streaming_true[framework_descriptor0-4]` (5)
* `test_token_count_type_stable_across_calls_when_supported[framework_descriptor0-4]` (5)
* `test_async_token_count_type_stable_across_calls_when_supported[framework_descriptor0-4]` (5)
* `test_async_token_count_type_matches_sync_when_both_declared[framework_descriptor0-4]` (5)

---

### Framework-Specific Tests (414 tests)

These tests are **unique to each framework**, validating framework-specific features and integration patterns.

#### `test_autogen_adapter.py` (58 tests)

**Specification:** AutoGen Integration  
**Status:** ✅ Complete (58 tests)

AutoGen-specific LLM adapter tests covering conversation context translation, function tools, and streaming operations.

**Key Test Areas:**
- Initialization & config validation: 8 tests
- Context translation from AutoGen conversation: 11 tests
- LLM completion operations: 14 tests
- Error handling & context attachment: 6 tests
- Token counting: 6 tests
- Capabilities & health: 6 tests
- Resource management: 3 tests
- End-to-end integration: 4 tests

**Notable Tests:**
* `test_create_builds_operation_context_from_conversation` — Conversation context handling
* `test_create_handles_tool_calls_in_response` — Tool call response processing
* `test_client_supports_direct_call_syntax` — AutoGen-style invocation
* `test_e2e_autogen_core_wrapper_can_be_constructed` — Real AutoGen integration

#### `test_crewai_llm_adapter.py` (68 tests)

**Specification:** CrewAI Integration  
**Status:** ✅ Complete (68 tests)

CrewAI-specific LLM adapter tests covering task context propagation, message normalization, and thread safety.

**Key Test Areas:**
- Initialization & config: 6 tests
- Context translation from CrewAI task/agent: 8 tests
- Message normalization: 7 tests
- Completion operations: 10 tests
- Streaming operations: 4 tests
- Error handling: 4 tests
- Token counting: 4 tests
- Capabilities & health: 4 tests
- Resource management: 3 tests
- Event loop guards: 2 tests
- End-to-end integration: 6 tests

**Notable Tests:**
* `test_complete_builds_operation_context_from_task` — Task context handling
* `test_complete_accepts_crewai_message_object` — CrewAI message type support
* `test_complete_normalizes_mixed_message_list` — Message list normalization
* `test_e2e_crewai_accepts_real_task_object` — Real CrewAI task integration

#### `test_langchain_llm_adapter.py` (103 tests)

**Specification:** LangChain Integration  
**Status:** ✅ Complete (103 tests)

LangChain-specific LLM adapter tests covering RunnableConfig translation, callback handling, and thread bridging.

**Key Test Areas:**
- Import guards & initialization: 6 tests
- Message normalization: 8 tests
- Context handling: 8 tests
- Generate operations (sync/async): 24 tests
- Streaming operations: 12 tests
- Callback handling: 6 tests
- Health & capabilities: 8 tests
- Token counting: 10 tests
- Error handling: 6 tests
- End-to-end integration: 15 tests

**Notable Tests:**
* `test_framework_ctx_contains_expected_keys` — LangChain context structure
* `test_stream_calls_sync_callbacks_in_order` — Callback execution order
* `test_astream_early_break_closes_underlying_generator` — Resource cleanup
* `test_e2e_tools_forwarded` — Real LangChain tool integration

#### `test_llamaindex_llm_adapter.py` (76 tests)

**Specification:** LlamaIndex Integration  
**Status:** ✅ Complete (76 tests)

LlamaIndex-specific LLM adapter tests covering callback manager translation, metadata exposure, and chat response building.

**Key Test Areas:**
- Initialization & config: 8 tests
- Metadata exposure: 4 tests
- Context translation from callback manager: 6 tests
- Message conversion: 6 tests
- Chat operations (sync/async): 12 tests
- Streaming operations: 4 tests
- Complete operations (for non-chat): 4 tests
- Error handling: 4 tests
- Token counting: 4 tests
- Health & capabilities: 6 tests
- Event loop guards: 2 tests
- Integration validation: 6 tests

**Notable Tests:**
* `test_metadata_exposes_context_window` — Model metadata exposure
* `test_to_translator_messages_extracts_content_from_blocks` — Complex message extraction
* `test_build_chat_response_includes_usage_dict` — Response building with usage
* `test_llamaindex_callback_manager_creates_operation_context` — Real callback integration

#### `test_semantic_kernel_llm_adapter.py` (84 tests)

**Specification:** Semantic Kernel Integration  
**Status:** ✅ Complete (84 tests)

Semantic Kernel-specific LLM adapter tests covering kernel context translation, chat message content handling, and execution settings.

**Key Test Areas:**
- Initialization & config: 8 tests
- SK inheritance validation: 4 tests
- Context translation from execution settings: 8 tests
- Message conversion: 8 tests
- Chat operations (sync/async): 12 tests
- Streaming operations: 8 tests
- Error handling: 6 tests
- Token counting: 4 tests
- Health & capabilities: 6 tests
- Resource management: 4 tests
- Event loop guards: 4 tests
- Batch operations: 2 tests
- Integration validation: 10 tests

**Notable Tests:**
* `test_semantic_kernel_chatcompletionbase_inheritance` — SK base class validation
* `test_to_translator_messages_extracts_role_and_author_role` — Dual role handling
* `test_get_streaming_chat_message_contents_yields_chunks` — Batch streaming
* `test_semantic_kernel_chathistory_object_works` — Real SK chat history

#### `test_llm_registry_self_check.py` (25 tests)

**Specification:** LLM Registry Validation  
**Status:** ✅ Complete (25 tests)

Registry integrity and descriptor validation tests with framework parametrization.

**Key Test Areas:**
- Registry consistency: 8 tests
- Descriptor validation: 8 tests
- Async/Streaming support: 6 tests
- Edge case handling: 3 tests

**Notable Tests:**
* `test_llm_registry_keys_match_descriptor_name` — Registry key consistency
* `test_streaming_flag_method_consistency` — Streaming support validation
* `test_descriptor_immutability` — Immutability enforcement
* `test_register_llm_framework_descriptor` — Descriptor registration

---

## Framework Coverage

### Per-Framework Test Breakdown

| Framework | Framework-Specific | + Contract Tests | Total Tests Validating |
|-----------|-------------------|------------------|------------------------|
| **AutoGen** | 58 unique tests | + 42 shared | **100 tests** |
| **CrewAI** | 68 unique tests | + 42 shared | **110 tests** |
| **LangChain** | 103 unique tests | + 42 shared | **145 tests** |
| **LlamaIndex** | 76 unique tests | + 42 shared | **118 tests** |
| **Semantic Kernel** | 84 unique tests | + 42 shared | **126 tests** |
| **Registry** | 25 integrity tests | N/A | **25 tests** |

**Understanding the Numbers:**
- **Framework-Specific**: Tests unique to that framework
- **Contract Tests**: Each framework is validated by 42 parametrized contract tests (210 total ÷ 5 = 42 per framework)
- **Total Tests Validating**: Combined coverage showing how thoroughly each framework is tested

### Execution Time Breakdown

| Category | Tests | Avg Time/Test |
|----------|-------|---------------|
| Parametrized Contract | 210 | ~30ms |
| AutoGen | 58 | ~32ms |
| CrewAI | 68 | ~31ms |
| LangChain | 103 | ~33ms |
| LlamaIndex | 76 | ~32ms |
| Semantic Kernel | 84 | ~33ms |
| Registry | 25 | ~28ms |
| **Overall** | **624** | **32.8ms** |

---

## Running Tests

### All LLM framework tests

```bash
CORPUS_ADAPTER=tests.mock.mock_llm_adapter:MockLLMAdapter \
  pytest tests/frameworks/llm/ -v
```

**Expected output:** `624 passed in ~20s`

### By framework

```bash
# AutoGen (58 unique + 42 contract = 100 tests validating AutoGen)
CORPUS_ADAPTER=tests.mock.mock_llm_adapter:MockLLMAdapter \
  pytest tests/frameworks/llm/test_autogen_adapter.py -v

# CrewAI (68 unique + 42 contract = 110 tests validating CrewAI)
CORPUS_ADAPTER=tests.mock.mock_llm_adapter:MockLLMAdapter \
  pytest tests/frameworks/llm/test_crewai_llm_adapter.py -v

# LangChain (103 unique + 42 contract = 145 tests validating LangChain)
CORPUS_ADAPTER=tests.mock.mock_llm_adapter:MockLLMAdapter \
  pytest tests/frameworks/llm/test_langchain_llm_adapter.py -v

# LlamaIndex (76 unique + 42 contract = 118 tests validating LlamaIndex)
CORPUS_ADAPTER=tests.mock.mock_llm_adapter:MockLLMAdapter \
  pytest tests/frameworks/llm/test_llamaindex_llm_adapter.py -v

# Semantic Kernel (84 unique + 42 contract = 126 tests validating Semantic Kernel)
CORPUS_ADAPTER=tests.mock.mock_llm_adapter:MockLLMAdapter \
  pytest tests/frameworks/llm/test_semantic_kernel_llm_adapter.py -v

# Registry (25 integrity tests)
CORPUS_ADAPTER=tests.mock.mock_llm_adapter:MockLLMAdapter \
  pytest tests/frameworks/llm/test_llm_registry_self_check.py -v
```

### Contract tests only

```bash
# All parametrized contract tests (210 tests across 5 frameworks)
CORPUS_ADAPTER=tests.mock.mock_llm_adapter:MockLLMAdapter \
  pytest tests/frameworks/llm/test_contract_*.py \
         tests/frameworks/llm/test_with_mock_backends.py -v
```

**Expected output:** `210 passed` (60 + 85 + 65)

### With coverage report

```bash
CORPUS_ADAPTER=tests.mock.mock_llm_adapter:MockLLMAdapter \
  pytest tests/frameworks/llm/ \
  --cov=corpus_sdk.llm.framework_adapters \
  --cov-report=html
```

---

## Framework Compliance Checklist

Use this when implementing or validating a new **LLM framework adapter**.

### ✅ Phase 1: Core Interface (6/6)

* [x] Implements framework-specific base class/interface
* [x] Provides sync completion method (`create`, `complete`, `generate`, `chat`, etc.)
* [x] Provides streaming method when supported
* [x] Async variants when framework supports async
* [x] Framework-specific context parameter acceptance
* [x] Direct call syntax support when applicable

### ✅ Phase 2: Context Translation (8/8)

* [x] Accepts framework-specific context parameter (conversation, task, callback manager, etc.)
* [x] Translates to `OperationContext` using appropriate factory
* [x] Graceful degradation on translation failure
* [x] Invalid context types tolerated without crash
* [x] Context propagation through to underlying adapter
* [x] Framework metadata included in contexts
* [x] Configuration flags control context behavior
* [x] Extra context enrichment support

### ✅ Phase 3: Message Handling (7/7)

* [x] Framework message object conversion
* [x] Message normalization for mixed types
* [x] Empty message validation when enabled
* [x] Role extraction from framework objects
* [x] Content extraction from complex message structures
* [x] Support for framework-specific message types
* [x] Batch message processing when needed

### ✅ Phase 4: LLM Operations (8/8)

* [x] Parameter forwarding (model, temperature, max_tokens)
* [x] Stop sequence handling
* [x] System message forwarding
* [x] Tools and tool choice forwarding
* [x] Sampling parameters support
* [x] Response building with framework types
* [x] Usage statistics inclusion
* [x] Finish reason handling

### ✅ Phase 5: Streaming Support (6/6)

* [x] Sync streaming when declared
* [x] Async streaming when supported
* [x] Chunk validation and error context attachment
* [x] Callback execution for streaming
* [x] Resource cleanup on early termination
* [x] Batch streaming when applicable

### ✅ Phase 6: Token Counting (5/5)

* [x] Token counting method when supported
* [x] Async token counting variant
* [x] Multiple input type handling (string, message objects, lists)
* [x] Model parameter forwarding
* [x] Error handling for invalid responses

### ✅ Phase 7: Error Handling (7/7)

* [x] Error context includes framework-specific fields
* [x] Error context attached on all failure paths
* [x] Async errors include same context as sync
* [x] Operation-specific error codes
* [x] LLM-specific error categorization
* [x] Backend exception wrapping with context
* [x] Stream iteration error propagation

### ✅ Phase 8: Capabilities & Health (6/6)

* [x] Capabilities passthrough when underlying provides
* [x] Health passthrough when underlying provides
* [x] Async capabilities/health fallback to sync via thread
* [x] Missing capabilities/health handled gracefully
* [x] Metadata exposure (context window, is_chat_model, etc.)
* [x] Framework version reporting

### ✅ Phase 9: Resource Management (5/5)

* [x] Context manager support (sync)
* [x] Async context manager support
* [x] Idempotent close operations
* [x] Connection pool management when applicable
* [x] Event loop guard rails for sync methods

### ✅ Phase 10: Framework Integration (6/6)

* [x] Real framework object handling
* [x] Framework-specific telemetry/logging
* [x] Plugin/extension system support
* [x] End-to-end workflow validation
* [x] Dependency availability checks
* [x] Import guards for missing frameworks

### ✅ Phase 11: Mock Backend Robustness (13/13)

* [x] Invalid backend result detection
* [x] Empty backend result detection
* [x] Wrong response type detection
* [x] Backend exception wrapping (invoke)
* [x] Backend exception wrapping (stream calltime)
* [x] Backend exception wrapping (stream iteration)
* [x] Async backend exception wrapping
* [x] Token counting error handling
* [x] All error paths include rich context
* [x] Sync/async consistency validation
* [x] Streaming error propagation
* [x] Mock backend tests pass for all frameworks

---

## Conformance Badge

```text
✅ LLM Framework Adapters V1.0 - 100% Conformant
   624/624 tests passing (10 test files, 5 frameworks)

   Framework-Specific Tests: 414/414 (100%)
   ✅ AutoGen:          58/58  ✅ CrewAI:           68/68
   ✅ LangChain:       103/103 ✅ LlamaIndex:       76/76
   ✅ Semantic Kernel:  84/84  ✅ Registry:         25/25

   Parametrized Contract Tests: 210/210 (100%)
   ✅ Context & Error:       60/60  (12×5 frameworks)
   ✅ Interface Conformance: 85/85  (17×5 frameworks)
   ✅ Shapes & Batching:     65/65  (13×5 frameworks)

   Total Tests Validating Each Framework:
   ✅ AutoGen:          100 tests (58 unique + 42 contract)
   ✅ CrewAI:           110 tests (68 unique + 42 contract)
   ✅ LangChain:        145 tests (103 unique + 42 contract)
   ✅ LlamaIndex:       118 tests (76 unique + 42 contract)
   ✅ Semantic Kernel:  126 tests (84 unique + 42 contract)

   Status: Production Ready - Platinum Certification 🏆
```

## **LLM Framework Adapters Conformance**

**Certification Levels:**
- 🏆 **Platinum:** 624/624 tests (100%) - All frameworks, all tests passing
- 🥇 **Gold:** 624/624 tests (100%) - All frameworks, all tests passing
- 🥈 **Silver:** 500+ tests (80%+) - Core functionality validated
- 🔬 **Development:** 312+ tests (50%+) - Early development, not production-ready

**Certification Requirements:**
- **🏆 Platinum:** Pass all 624 tests with zero failures
- **🥇 Gold:** Pass all 624 tests with zero failures  
- **🥈 Silver:** Pass ≥500 tests (80%+)
- **🔬 Development:** Pass ≥312 tests (50%+)

---

## Maintenance

### Adding New Framework Adapters

When adding a new framework adapter:

1. **Implement framework-specific tests** (60-100 tests recommended based on framework complexity)
   - Minimum 50 tests for basic frameworks
   - 70-100 tests for frameworks with rich features (like LangChain's callback system)
   - Add test class in `tests/frameworks/llm/test_<framework>_adapter.py`

2. **Ensure parametrized contract tests run** (automatically adds 42 tests validating your framework)
   - Add framework descriptor to `tests/frameworks/llm/conftest.py`
   - Verify all 210 parametrized tests execute for your framework

3. **Add framework descriptor** to registry
   - Update `corpus_sdk/llm/framework_adapters/registry.py`
   - Include version compatibility, async support, streaming support, and sample context

4. **Update this document** with new framework counts
   - Add row to Framework Coverage table
   - Update total test count
   - Document framework-specific features

5. **Run full suite** to verify 100% pass rate
   ```bash
   pytest tests/frameworks/llm/ -v
   ```

### Updating Test Coverage

When the protocol evolves:

1. **Update parametrized tests** (affects all 5 frameworks simultaneously)
   - Changes to `test_contract_*.py` automatically cover all frameworks
   - Ensures consistent behavior across all adapters

2. **Add framework-specific tests** as needed for new features
   - New LLM operations may require additional unique tests
   - Maintain parity where possible across frameworks

3. **Maintain backward compatibility** or update all adapters together
   - Breaking changes require updates to all 5 framework adapters
   - Increment adapter version for breaking changes

### Test Count Verification

To verify test counts match this document:

```bash
# Count by file
pytest tests/frameworks/llm/ --collect-only | grep "<Module"

# Verify total
pytest tests/frameworks/llm/ -v | grep "passed"

# Expected output
# 624 passed in ~20s
```

### Performance Benchmarking

If tests become slower:

```bash
# Identify slowest tests
pytest tests/frameworks/llm/ --durations=10

# Profile with pytest-profiling
pytest tests/frameworks/llm/ --profile

# Target: Keep average <40ms/test, total <25s
```

---

**Last Updated:** 2026-02-10  
**Maintained By:** Corpus SDK Team  
**Status:** 100% V1.0 Conformant - Production Ready - Platinum Certification 🏆  
**Test Count:** 624/624 tests (100%)  
**Execution Time:** 20.50s (32.8ms/test average)  
**Framework Coverage:** 5/5 frameworks (AutoGen, CrewAI, LangChain, LlamaIndex, Semantic Kernel)

---
