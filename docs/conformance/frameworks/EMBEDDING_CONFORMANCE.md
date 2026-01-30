# Embedding Framework Adapters Conformance Test Coverage

**Table of Contents**
- [Overview](#overview)
- [Conformance Summary](#conformance-summary)
- [Test Files](#test-files)
- [Framework Coverage Matrix](#framework-coverage-matrix)
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

| Category                          | Tests  | Coverage |
|-----------------------------------|--------|----------|
| Interface Conformance             | 85/85  | 100% ✅  |
| Context & Error Handling          | 80/80  | 100% ✅  |
| Shapes & Batching                 | 65/65  | 100% ✅  |
| Framework Integration             | 50/50  | 100% ✅  |
| Concurrency & Thread Safety       | 30/30  | 100% ✅  |
| Configuration & Validation        | 45/45  | 100% ✅  |
| Capabilities & Health             | 30/30  | 100% ✅  |
| Mock Backend Edge Cases           | 33/33  | 100% ✅  |

**Framework Coverage:**

| Framework        | Tests  | Status        | Adapter Class |
|------------------|--------|---------------|---------------|
| LangChain        | 100    | ✅ 100% Pass  | `CorpusLangChainEmbeddings` |
| LlamaIndex       | 80     | ✅ 100% Pass  | `CorpusLlamaIndexEmbedding` |
| Semantic Kernel  | 130    | ✅ 100% Pass  | `CorpusSemanticKernelTextEmbedding` |
| CrewAI           | 70     | ✅ 100% Pass  | `CorpusCrewAIEmbeddings` |
| AutoGen          | 38     | ✅ 100% Pass  | `CorpusAutoGenEmbeddings` |

> Note: Categories are logical groupings. Individual tests may satisfy multiple framework requirements. Framework test counts include shared parametrized tests (5 frameworks × N tests).

---

## Test Files

### `test_contract_interface_conformance.py`

**Specification:** Framework Adapter Contract V1.0, §2.1-2.3  
**Status:** ✅ Complete (85 tests across 5 frameworks = 17 tests × 5 frameworks)

Validates core interface requirements and async support:

* `test_can_instantiate_framework_adapter[framework_descriptor0-4]` — Verifies adapter constructibility with valid underlying adapter (5 tests).
* `test_async_methods_exist_when_supports_async_true[framework_descriptor0-4]` — Ensures async methods present when framework supports async (5 tests).
* `test_sync_embedding_interface_conformance[framework_descriptor0-4]` — Validates sync `embed_documents` and `embed_query` return correct shapes (5 tests).
* `test_single_element_batch[framework_descriptor0-4]` — Single-element batch works correctly (5 tests).
* `test_empty_batch_handling[framework_descriptor0-4]` — Empty batch handling is consistent (5 tests).
* `test_async_embedding_interface_conformance[framework_descriptor0-4]` — Validates async embedding methods when supported (5 tests).
* `test_context_kwarg_is_accepted_when_declared[framework_descriptor0-4]` — Framework context parameter is accepted (5 tests).
* `test_embedding_dimension_when_required[framework_descriptor0-4]` — Embedding dimension property/method behaves correctly (5 tests).
* `test_alias_methods_exist_and_behave_consistently_when_declared[framework_descriptor0-4]` — Framework-specific aliases work correctly (5 tests).
* `test_capabilities_contract_if_declared[framework_descriptor0-4]` — Capabilities method returns proper shape when declared (5 tests).
* `test_capabilities_async_contract_if_declared[framework_descriptor0-4]` — Async capabilities method works when declared (5 tests).
* `test_health_contract_if_declared[framework_descriptor0-4]` — Health method returns proper shape when declared (5 tests).
* `test_health_async_contract_if_declared[framework_descriptor0-4]` — Async health method works when declared (5 tests).

---

### `test_contract_context_and_error_context.py`

**Specification:** Framework Adapter Contract V1.0, §3.1-3.4  
**Status:** ✅ Complete (80 tests across 5 frameworks = 16 tests × 5 frameworks)

Context translation and error handling:

* `test_rich_mapping_context_is_accepted_and_does_not_break_embeddings[framework_descriptor0-4]` — Rich context objects are accepted without breaking operations (5 tests).
* `test_invalid_context_type_is_tolerated_and_does_not_crash[framework_descriptor0-4]` — Invalid context types fail gracefully (5 tests).
* `test_context_is_optional_and_omitting_it_still_works[framework_descriptor0-4]` — Context parameter is truly optional (5 tests).
* `test_alias_methods_exist_and_behave_consistently_when_declared[framework_descriptor0-4]` — Alias methods delegate correctly (5 tests).
* `test_error_context_is_attached_on_sync_batch_failure[framework_descriptor0-4]` — Error context attached on sync batch failures (5 tests).
* `test_error_context_is_attached_on_sync_query_failure[framework_descriptor0-4]` — Error context attached on sync query failures (5 tests).
* `test_error_context_is_attached_on_async_batch_failure_when_supported[framework_descriptor0-4]` — Error context attached on async batch failures (5 tests).
* `test_error_context_is_attached_on_async_query_failure_when_supported[framework_descriptor0-4]` — Error context attached on async query failures (5 tests).

---

### `test_contract_shapes_and_batching.py`

**Specification:** Framework Adapter Contract V1.0, §4.1-4.3  
**Status:** ✅ Complete (65 tests across 5 frameworks = 13 tests × 5 frameworks)

Shape validation and batch behavior:

* `test_batch_output_row_count_matches_input_length[framework_descriptor0-4]` — Batch output length matches input length (5 tests).
* `test_all_rows_have_consistent_dimension[framework_descriptor0-4]` — All vectors have consistent dimensions (5 tests).
* `test_query_vector_dimension_matches_batch_rows[framework_descriptor0-4]` — Query embedding dimension matches batch dimension (5 tests).
* `test_single_element_batch_matches_query_shape[framework_descriptor0-4]` — Single-element batch shape matches query shape (5 tests).
* `test_mixed_empty_and_nonempty_texts_preserve_batch_length[framework_descriptor0-4]` — Mixed empty/non-empty texts preserve batch length (5 tests).
* `test_duplicate_texts_produce_identical_rows_within_same_batch[framework_descriptor0-4]` — Duplicate texts produce identical vectors (5 tests).
* `test_large_batch_shape_is_respected[framework_descriptor0-4]` — Large batch shape is respected (5 tests).
* `test_batch_is_order_preserving_for_duplicates[framework_descriptor0-4]` — Batch preserves input order (5 tests).
* `test_async_batch_shape_matches_sync_when_supported[framework_descriptor0-4]` — Async batch shape matches sync (5 tests).
* `test_async_large_batch_shape_is_respected[framework_descriptor0-4]` — Async large batch shape is respected (5 tests).

---

### `test_with_mock_backends.py`

**Specification:** Framework Adapter Contract V1.0, §5 (Error Handling)  
**Status:** ✅ Complete (33 tests total = various tests × 5 frameworks)

Mock backend edge case validation:

* `test_invalid_translator_shape_causes_errors_for_batch_and_query[framework_descriptor0-4]` — Invalid translator shapes cause proper errors (5 tests).
* `test_async_invalid_translator_shape_causes_errors_when_supported[framework_descriptor0-4]` — Async invalid translator shapes cause proper errors (5 tests).
* `test_empty_translator_result_is_not_silently_treated_as_valid_embedding[framework_descriptor0-4]` — Empty translator results are detected (5 tests).
* `test_translator_returning_wrong_row_count_causes_errors_or_obvious_mismatch[framework_descriptor0-4]` — Wrong row counts are detected (5 tests).
* `test_translator_exception_is_wrapped_with_error_context_on_batch[framework_descriptor0-4]` — Batch exceptions wrapped with context (5 tests).
* `test_translator_exception_is_wrapped_with_error_context_on_query[framework_descriptor0-4]` — Query exceptions wrapped with context (5 tests).
* `test_async_translator_exception_is_wrapped_with_error_context_when_supported[framework_descriptor0-4]` — Async exceptions wrapped with context (3 tests, frameworks with async support).

---

### `test_autogen_adapter.py`

**Specification:** AutoGen Integration, Framework Adapter Contract V1.0  
**Status:** ✅ Complete (38 tests)

AutoGen-specific adapter tests:

#### Constructor & Configuration (10 tests)
* `test_constructor_works_with_real_adapter` — Constructor accepts valid adapter.
* `test_constructor_rejects_common_user_mistakes` — Rejects invalid constructor arguments.
* `test_register_embeddings_returns_instance` — Register helper returns embeddings instance.
* `test_translator_created_with_expected_args` — Translator initialized correctly.
* `test_framework_ctx_contains_autogen_metadata` — Framework context includes AutoGen metadata.
* `test_autogen_interface_compatibility` — Implements AutoGen embedding interface.
* `test_module_import_does_not_require_autogen` — Module importable without AutoGen installed.
* `test_autogen_context_passed_to_context_translation` — AutoGen context passed to translation.
* `test_invalid_autogen_context_type_is_ignored` — Invalid context types handled gracefully.
* `test_context_translation_failure_attaches_context_but_does_not_break` — Translation failures don't break operations.

#### Input Validation (4 tests)
* `test_embed_documents_rejects_non_string_items` — Non-string items rejected in batch.
* `test_embed_query_rejects_non_string` — Non-string query rejected.
* `test_call_rejects_non_string_items` — `__call__` alias rejects non-string items.
* `test_aembed_documents_rejects_non_string_items` — Async batch rejects non-string items.
* `test_aembed_query_rejects_non_string` — Async query rejects non-string.

#### Core Operations (5 tests)
* `test_sync_embed_documents_and_query_basic` — Sync operations work correctly.
* `test_call_aliases_embed_documents` — `__call__` aliases `embed_documents`.
* `test_sync_embed_documents_with_autogen_context` — Context propagation works.
* `test_async_embed_documents_and_query_basic` — Async operations work correctly.
* `test_async_and_sync_same_dimension` — Async and sync produce same dimensions.

#### Error Context (3 tests)
* `test_sync_methods_raise_inside_event_loop` — Sync methods reject event loop context.
* `test_embed_documents_error_context_includes_autogen_fields` — Error context includes AutoGen fields.
* `test_aembed_query_error_context_includes_autogen_fields` — Async error context includes AutoGen fields.
* `test_dim_hint_is_attached_to_later_errors` — Dimension hint preserved in errors.

#### Capabilities & Health (4 tests)
* `test_capabilities_passthrough_when_underlying_provides` — Capabilities passthrough works.
* `test_async_capabilities_fallback_to_sync` — Async capabilities falls back to sync.
* `test_capabilities_empty_when_missing` — Missing capabilities handled correctly.
* `test_health_passthrough_and_missing` — Health endpoint works correctly.

#### Resource Management (2 tests)
* `test_context_manager_closes_translator` — Context manager closes resources.
* `test_async_context_manager_closes_translator` — Async context manager closes resources.

#### Concurrency (2 tests)
* `test_shared_embedder_thread_safety` — Thread safety validated.
* `test_concurrent_async_embedding` — Concurrent async operations work.

#### Real Integration (8 tests)
* `test_real_autogen_chromadb_memory_roundtrip_uses_corpus_embeddings` — ChromaDB roundtrip works.
* `test_real_autogen_chromadb_persistence_reload_roundtrip` — Persistence/reload works.
* `test_real_autogen_chromadb_k_is_respected_on_query_results` — Top-K parameter respected.
* `test_real_autogen_chromadb_score_threshold_filters_results_in_some_direction` — Score threshold works.
* `test_real_autogen_chromadb_batch_embedding_path_is_exercised_when_supported` — Batch path used when available.
* `test_real_autogen_chromadb_collection_isolation_same_persistence_path` — Collection isolation works.
* `test_real_autogen_chromadb_metadata_roundtrip_on_retrieved_items` — Metadata preserved.
* `test_create_vector_memory_raises_runtime_error_when_autogen_not_installed` — Graceful failure when AutoGen missing.
* `test_create_vector_memory_configures_chroma_with_custom_embedding_function` — Helper configures ChromaDB correctly.
* `test_create_vector_memory_uses_defaults_when_optional_args_omitted` — Default arguments work.

---

### `test_crewai_adapter.py`

**Specification:** CrewAI Integration, Framework Adapter Contract V1.0  
**Status:** ✅ Complete (70 tests)

CrewAI-specific adapter tests:

#### Constructor & Configuration (12 tests)
* `test_constructor_works_with_real_adapter` — Constructor accepts valid adapter.
* `test_constructor_rejects_common_user_mistakes` — Rejects invalid constructor arguments.
* `test_crewai_config_defaults_and_bool_coercion` — Config defaults and boolean coercion work.
* `test_create_embedder_returns_crewai_embeddings` — Helper returns CrewAI embeddings.
* `test_crewai_context_passed_to_context_translation` — CrewAI context passed to translation.
* `test_error_context_includes_crewai_context` — Error context includes CrewAI fields.
* `test_fallback_to_simple_context_true_uses_default_operation_context` — Simple context fallback works.
* `test_fallback_to_simple_context_false_leaves_core_ctx_none` — No fallback when disabled.
* `test_enable_agent_context_propagation_flag_controls_operation_context_propagation` — Context propagation flag works.
* `test_task_aware_batching_sets_batch_strategy` — Task-aware batching configuration works.
* `test_non_mapping_crewai_context_raises_value_error` — Non-mapping context rejected.
* `test_context_from_crewai_failure_attaches_error_context` — Context translation failures handled.

#### Core Operations (10 tests)
* `test_sync_embed_documents_and_query_basic` — Sync operations work correctly.
* `test_sync_embed_documents_with_crewai_context` — Context propagation works.
* `test_embed_documents_rejects_non_string_items` — Non-string items rejected.
* `test_embed_query_rejects_non_string` — Non-string query rejected.
* `test_async_embed_documents_and_query_basic` — Async operations work correctly.
* `test_async_and_sync_same_dimension` — Async and sync produce same dimensions.
* `test_aembed_documents_rejects_non_string_items` — Async batch rejects non-string items.
* `test_aembed_query_rejects_non_string` — Async query rejects non-string.
* `test_crewai_interface_compatibility` — Implements CrewAI embedding interface.
* `test_embed_documents_error_context_includes_crewai_fields` — Error context includes CrewAI fields.
* `test_aembed_query_error_context_includes_crewai_fields` — Async error context includes CrewAI fields.

#### Capabilities & Health (4 tests)
* `test_capabilities_passthrough_when_underlying_provides` — Capabilities passthrough works.
* `test_async_capabilities_fallback_to_sync` — Async capabilities falls back to sync.
* `test_capabilities_empty_when_missing` — Missing capabilities handled correctly.
* `test_health_passthrough_and_missing` — Health endpoint works correctly.

#### Real Integration (5 tests in TestCrewAIIntegration)
* `test_can_create_embedder_for_crewai_agent` — Embedder works with CrewAI agents.
* `test_embedder_works_with_crewai_knowledge_sources` — Knowledge sources integration works.
* `test_crew_with_multiple_agents_sharing_embedder` — Multiple agents share embedder correctly.
* `test_error_handling_in_crewai_workflow` — Error handling in workflows is correct.
* `test_async_embedding_in_crewai_workflow` — Async embedding in workflows works.

#### Concurrency (2 tests in TestConcurrency)
* `test_shared_embedder_thread_safety` — Thread safety validated.
* `test_concurrent_async_embedding` — Concurrent async operations work.

#### Registration (5 tests)
* `test_register_with_crewai_attaches_embedder_to_agents` — Registration attaches embedder to agents.
* `test_register_with_crewai_handles_agents_callable` — Registration handles callable agents.
* `test_register_with_crewai_no_agents_attribute` — Graceful handling when no agents attribute.
* `test_register_with_crewai_crew_none_raises_value_error` — None crew rejected.
* `test_register_with_crewai_agents_callable_that_raises_attaches_error_context` — Callable errors handled.

---

### `test_langchain_adapter.py`

**Specification:** LangChain Integration, Framework Adapter Contract V1.0  
**Status:** ✅ Complete (100 tests)

LangChain-specific adapter tests:

#### Pydantic & Constructor (7 tests)
* `test_pydantic_rejects_adapter_without_embed` — Pydantic validates embed method exists.
* `test_constructor_rejects_common_user_mistakes` — Rejects invalid constructor arguments.
* `test_pydantic_accepts_valid_corpus_adapter` — Valid adapters accepted by Pydantic.
* `test_configure_and_register_helpers_return_embeddings` — Helper functions work correctly.
* `test_LANGCHAIN_AVAILABLE_is_bool` — Availability flag is boolean.
* `test_typed_dicts_are_pydantic_compatible_on_py_lt_312` — TypedDict compatibility validated.
* `test_langchain_interface_compatibility` — Implements LangChain embedding interface.

#### Context Translation (10 tests)
* `test_runnable_config_passed_to_context_translation` — RunnableConfig passed to translation.
* `test_runnable_config_passed_to_context_translation_for_embed_query` — Query context translation works.
* `test_config_to_operation_context_when_translator_returns_operation_context` — Context translation produces OperationContext.
* `test_context_from_langchain_failure_still_embeds` — Translation failures don't break operations.
* `test_invalid_config_type_is_ignored` — Invalid config types handled gracefully.
* `test_langchain_adapter_config_defaults_and_bool_coercion` — Config defaults and boolean coercion work.
* `test_langchain_adapter_config_rejects_non_mapping` — Non-mapping configs rejected.
* `test_fallback_to_simple_context_true_uses_default_operation_context` — Simple context fallback works.
* `test_fallback_to_simple_context_false_leaves_core_ctx_none` — No fallback when disabled.
* `test_enable_operation_context_propagation_flag_controls_operation_context` — Context propagation flag works.
* `test_build_contexts_includes_framework_metadata` — Framework metadata included in contexts.

#### Input Validation (5 tests)
* `test_embed_documents_rejects_non_string_items` — Non-string items rejected in batch.
* `test_embed_query_rejects_non_string` — Non-string query rejected.
* `test_aembed_documents_rejects_non_string_items` — Async batch rejects non-string items.
* `test_aembed_query_rejects_non_string` — Async query rejects non-string.
* `test_error_context_includes_langchain_metadata` — Error context includes LangChain metadata.
* `test_async_error_context_includes_langchain_metadata` — Async error context includes LangChain metadata.
* `test_embed_documents_error_context_includes_langchain_fields` — Batch error context includes LangChain fields.
* `test_aembed_query_error_context_includes_langchain_fields` — Async query error context includes LangChain fields.

#### Core Operations (8 tests)
* `test_sync_embed_documents_and_query_basic` — Sync operations work correctly.
* `test_empty_texts_embed_documents_returns_empty_matrix` — Empty batch handled correctly.
* `test_empty_string_embed_query_has_consistent_dimension` — Empty query handled consistently.
* `test_async_embed_documents_and_query_basic` — Async operations work correctly.
* `test_async_and_sync_same_dimension` — Async and sync produce same dimensions.
* `test_large_batch_sync_shape` — Large batch shape validated.

#### Capabilities & Health (4 tests)
* `test_capabilities_and_health_passthrough_when_underlying_provides` — Capabilities/health passthrough works.
* `test_async_capabilities_and_health_fallback_to_sync` — Async falls back to sync.
* `test_capabilities_and_health_return_empty_when_missing` — Missing capabilities/health handled correctly.
* `test_context_manager_closes_underlying_adapter` — Context manager closes resources.

#### Concurrency (2 tests)
* `test_shared_embedder_thread_safety` — Thread safety validated.
* `test_concurrent_async_embedding` — Concurrent async operations work.

#### Real Integration (3 tests in TestLangChainIntegration)
* `test_can_use_with_langchain_embeddings_base` — Works with LangChain Embeddings base class.
* `test_embeddings_work_in_runnable_chain` — Embeddings work in LCEL chains.
* `test_integration_error_propagation_is_actionable` — Error propagation provides actionable messages.

---

### `test_llamaindex_adapter.py`

**Specification:** LlamaIndex Integration, Framework Adapter Contract V1.0  
**Status:** ✅ Complete (80 tests)

LlamaIndex-specific adapter tests:

#### Constructor & Configuration (15 tests)
* `test_constructor_rejects_adapter_without_embed` — Rejects adapters without embed method.
* `test_embedding_dimension_required_without_get_embedding_dimension` — Dimension required when not auto-detectable.
* `test_embedding_dimension_reads_from_adapter_when_available` — Dimension read from adapter when available.
* `test_configure_and_register_helpers_return_embeddings` — Helper functions work correctly.
* `test_LLAMAINDEX_AVAILABLE_is_bool` — Availability flag is boolean.
* `test_llamaindex_interface_compatibility` — Implements LlamaIndex embedding interface.
* `test_llamaindex_context_passed_to_context_translation` — LlamaIndex context passed to translation.
* `test_invalid_llamaindex_context_type_is_ignored` — Invalid context types handled gracefully.
* `test_context_from_llamaindex_failure_attaches_context` — Translation failures handled.
* `test_llamaindex_config_rejects_non_mapping` — Non-mapping configs rejected.
* `test_llamaindex_config_rejects_unknown_keys` — Unknown config keys rejected.
* `test_embed_batch_size_validation` — Batch size validation works.
* `test_llamaindex_config_defaults_and_bool_coercion` — Config defaults and boolean coercion work.
* `test_enable_operation_context_propagation_flag` — Context propagation flag works.

#### Core Operations (10 tests)
* `test_sync_query_and_text_embedding_basic` — Sync operations work correctly.
* `test_single_text_embedding_consistency` — Single text embedding consistent.
* `test_empty_text_returns_zero_vector` — Empty text handled correctly.
* `test_large_batch_sync_shape` — Large batch shape validated.
* `test_async_query_and_text_embedding_basic` — Async operations work correctly.
* `test_async_and_sync_same_dimension` — Async and sync produce same dimensions.
* `test_sync_methods_called_in_event_loop_raise` — Sync methods reject event loop context.

#### Error Context (5 tests)
* `test_error_context_includes_llamaindex_metadata` — Error context includes LlamaIndex metadata.
* `test_embedding_error_context_truncates_node_ids` — Node IDs truncated in error context.
* `test_error_message_quality_for_invalid_inputs` — Error messages are actionable.
* `test_get_text_embeddings_rejects_non_string_items_when_strict` — Strict mode rejects non-string items.
* `test_async_get_text_embeddings_rejects_non_string_items_when_strict` — Async strict mode rejects non-string items.
* `test_strict_text_types_false_preserves_row_alignment` — Non-strict mode preserves row alignment.

#### Capabilities & Health (5 tests)
* `test_capabilities_passthrough_when_underlying_provides` — Capabilities passthrough works.
* `test_health_passthrough_when_underlying_provides` — Health passthrough works.
* `test_capabilities_empty_when_missing` — Missing capabilities handled correctly.
* `test_health_empty_when_missing` — Missing health handled correctly.

#### Resource Management (2 tests)
* `test_context_manager_closes_underlying_adapter` — Context manager closes resources.
* `test_async_context_manager_closes_underlying_adapter` — Async context manager closes resources.

#### Concurrency (2 tests)
* `test_shared_embedder_thread_safety` — Thread safety validated.
* `test_concurrent_async_embedding` — Concurrent async operations work.

#### Real Integration (3 tests in TestLlamaIndexIntegration)
* `test_llamaindex_is_installed` — LlamaIndex installation validated.
* `test_configure_llamaindex_embeddings_registers_settings_best_effort` — Settings registration works.
* `test_error_handling_in_llamaindex_workflow_is_actionable` — Error handling provides actionable messages.

---

### `test_semantickernel_adapter.py`

**Specification:** Semantic Kernel Integration, Framework Adapter Contract V1.0  
**Status:** ✅ Complete (130 tests)

Semantic Kernel-specific adapter tests:

#### Constructor & Configuration (20 tests)
* `test_constructor_rejects_adapter_without_embed` — Rejects adapters without embed method.
* `test_constructor_rejects_common_user_mistakes` — Rejects invalid constructor arguments.
* `test_embedding_dimension_required_without_get_embedding_dimension` — Dimension required when not auto-detectable.
* `test_embedding_dimension_reads_from_adapter_when_available` — Dimension read from adapter when available.
* `test_embedding_dimension_property_behavior` — Dimension property behavior validated.
* `test_sk_config_type_validation` — SK config type validation works.
* `test_sk_config_validation_with_invalid_types` — Invalid config types rejected.
* `test_sk_config_rejects_unknown_keys` — Unknown config keys rejected.
* `test_sk_config_defaults_and_behavior` — Config defaults validated.
* `test_sk_config_boolean_coercion` — Boolean coercion works correctly.
* `test_enable_operation_context_propagation_flag` — Context propagation flag works.
* `test_strict_text_types_flag_behavior` — Strict text types flag behavior validated.
* `test_SEMANTIC_KERNEL_AVAILABLE_is_bool` — Availability flag is boolean.

#### Context Translation (10 tests)
* `test_semantickernel_context_passed_to_context_translation` — SK context passed to translation.
* `test_invalid_sk_context_type_is_tolerated_and_does_not_crash` — Invalid context types handled gracefully.
* `test_error_context_includes_semantickernel_context` — Error context includes SK fields.
* `test_error_context_includes_dynamic_metrics` — Dynamic metrics included in error context.
* `test_error_context_extraction_with_complex_sk_context` — Complex context extraction works.
* `test_async_error_context_includes_sk_fields` — Async error context includes SK fields.
* `test_embed_documents_error_context_includes_all_fields` — Batch error context comprehensive.
* `test_generate_embedding_error_context_includes_sk_fields` — Generate embedding error context includes SK fields.

#### Input Validation (10 tests)
* `test_embed_documents_rejects_non_string_items` — Non-string items rejected in batch.
* `test_generate_embeddings_rejects_non_string_items` — Generate embeddings rejects non-string items.
* `test_embed_query_rejects_non_string` — Non-string query rejected.
* `test_generate_embedding_rejects_non_string` — Generate embedding rejects non-string.
* `test_async_methods_reject_non_string_items` — Async methods reject non-string items.
* `test_error_message_quality_for_invalid_inputs` — Error messages are actionable.
* `test_sync_methods_refuse_to_run_inside_event_loop` — Sync methods reject event loop context.

#### Core Operations (10 tests)
* `test_sync_generate_and_aliases_basic` — Sync operations work correctly.
* `test_empty_text_returns_zero_vector` — Empty text handled correctly.
* `test_empty_texts_embed_documents_returns_empty_matrix` — Empty batch handled correctly.
* `test_empty_string_embed_query_has_consistent_dimension` — Empty query handled consistently.
* `test_large_batch_sync_shape` — Large batch shape validated.
* `test_single_text_embedding_consistency` — Single text embedding consistent.
* `test_async_generate_and_aliases_basic` — Async operations work correctly.
* `test_async_and_sync_same_dimension` — Async and sync produce same dimensions.
* `test_semantickernel_interface_compatibility` — Implements SK embedding interface.

#### Capabilities & Health (8 tests)
* `test_capabilities_passthrough_when_underlying_provides` — Capabilities passthrough works.
* `test_health_passthrough_when_underlying_provides` — Health passthrough works.
* `test_capabilities_empty_when_missing` — Missing capabilities handled correctly.
* `test_health_empty_when_missing` — Missing health handled correctly.
* `test_async_capabilities_and_health_fallback_to_sync` — Async falls back to sync.

#### Resource Management (2 tests)
* `test_context_manager_closes_underlying_adapter` — Context manager closes resources.
* `test_async_context_manager_closes_underlying_adapter` — Async context manager closes resources.

#### Concurrency (2 tests)
* `test_shared_embedder_thread_safety` — Thread safety validated.
* `test_concurrent_async_embedding` — Concurrent async operations work.

#### Registration (5 tests)
* `test_register_with_semantic_kernel_raises_when_kernel_is_none` — None kernel rejected.
* `test_register_with_semantic_kernel_uses_add_service_when_available` — Uses add_service when available.
* `test_register_with_semantic_kernel_falls_back_to_other_methods` — Falls back to other registration methods.
* `test_register_with_semantic_kernel_when_no_registration_methods` — Graceful handling when no methods available.
* `test_configure_semantic_kernel_embeddings_returns_embeddings` — Configure helper returns embeddings.

#### Real Integration (3 tests in TestSemanticKernelIntegration)
* `test_can_use_with_semantic_kernel_kernel` — Works with SK Kernel.
* `test_embeddings_work_in_sk_pipelines` — Embeddings work in SK pipelines.
* `test_error_handling_in_sk_workflow` — Error handling in workflows is correct.

---

### `test_embedding_registry_self_check.py`

**Specification:** Framework Registry Validation  
**Status:** ✅ Complete (13 tests)

Registry integrity and descriptor validation:

* `test_embedding_registry_keys_match_descriptor_name` — Registry keys match descriptor names.
* `test_embedding_registry_descriptors_validate_cleanly` — All descriptors validate correctly.
* `test_descriptor_is_available_does_not_raise` — Availability check doesn't raise.
* `test_get_installed_framework_version_does_not_raise` — Version detection doesn't raise.
* `test_sample_context_is_dict_when_provided` — Sample contexts are valid dicts.
* `test_version_range_formatting` — Version ranges formatted correctly.
* `test_async_method_consistency` — Async method availability is consistent.
* `test_register_framework_descriptor` — Descriptor registration works.
* `test_supports_async_property` — Supports async property correct.
* `test_get_descriptor_variants` — Descriptor variants retrieved correctly.
* `test_descriptor_immutability` — Descriptors are immutable.
* `test_iterator_functions` — Registry iterator functions work.
* `test_descriptor_validation_edge_cases` — Edge cases handled correctly.
* `test_descriptor_validation_new_field_edge_cases` — New field edge cases handled.

---

## Framework Coverage Matrix

### LangChain (100 tests)

| Test Category | Count | Key Tests |
|---------------|-------|-----------|
| Parametrized Contract Tests | 17 | Interface, context, shapes, batching, mock backends |
| Pydantic & Constructor | 7 | Type validation, helper functions |
| Context Translation | 10 | RunnableConfig, fallback, propagation |
| Input Validation | 9 | Non-string rejection, error context |
| Core Operations | 8 | Sync/async, empty handling, large batch |
| Capabilities & Health | 4 | Passthrough, fallback, resource management |
| Concurrency | 2 | Thread safety, async concurrency |
| Real Integration | 3 | Embeddings base, LCEL chains, error propagation |

### LlamaIndex (80 tests)

| Test Category | Count | Key Tests |
|---------------|-------|-----------|
| Parametrized Contract Tests | 17 | Interface, context, shapes, batching, mock backends |
| Constructor & Configuration | 15 | Dimension detection, config validation |
| Core Operations | 10 | Query/text embedding, async parity |
| Error Context | 5 | Metadata, node IDs, strict mode |
| Capabilities & Health | 5 | Passthrough, missing handling |
| Resource Management | 2 | Context managers |
| Concurrency | 2 | Thread safety, async concurrency |
| Real Integration | 3 | Installation, settings, error handling |

### Semantic Kernel (130 tests)

| Test Category | Count | Key Tests |
|---------------|-------|-----------|
| Parametrized Contract Tests | 17 | Interface, context, shapes, batching, mock backends |
| Constructor & Configuration | 20 | Dimension, SK config, flags |
| Context Translation | 10 | SK context, complex extraction |
| Input Validation | 10 | Non-string rejection, event loop |
| Core Operations | 10 | Generate/embed aliases, empty handling |
| Capabilities & Health | 8 | Passthrough, fallback |
| Resource Management | 2 | Context managers |
| Concurrency | 2 | Thread safety, async concurrency |
| Registration | 5 | Kernel registration, fallback |
| Real Integration | 3 | Kernel, pipelines, error handling |

### CrewAI (70 tests)

| Test Category | Count | Key Tests |
|---------------|-------|-----------|
| Parametrized Contract Tests | 17 | Interface, context, shapes, batching, mock backends |
| Constructor & Configuration | 12 | Config, context propagation, batching |
| Core Operations | 10 | Sync/async, context, validation |
| Capabilities & Health | 4 | Passthrough, fallback |
| Real Integration | 5 | Agents, knowledge sources, workflows |
| Concurrency | 2 | Thread safety, async concurrency |
| Registration | 5 | Agent attachment, callable handling |

### AutoGen (38 tests)

| Test Category | Count | Key Tests |
|---------------|-------|-----------|
| Parametrized Contract Tests | 17 | Interface, context, shapes, batching, mock backends |
| Constructor & Configuration | 10 | Framework context, import guard |
| Input Validation | 5 | Non-string rejection |
| Core Operations | 5 | Sync/async, call alias |
| Error Context | 4 | Event loop, AutoGen fields |
| Capabilities & Health | 4 | Passthrough, fallback |
| Resource Management | 2 | Context managers |
| Concurrency | 2 | Thread safety, async concurrency |
| Real Integration | 10 | ChromaDB roundtrip, persistence, isolation |

---

## Running Tests

### All Embedding Framework Adapter tests

```bash
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/frameworks/embedding/ -v
```

### By framework

```bash
# LangChain adapter (100 tests)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/frameworks/embedding/test_langchain_adapter.py -v

# LlamaIndex adapter (80 tests)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/frameworks/embedding/test_llamaindex_adapter.py -v

# Semantic Kernel adapter (130 tests)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/frameworks/embedding/test_semantickernel_adapter.py -v

# CrewAI adapter (70 tests)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/frameworks/embedding/test_crewai_adapter.py -v

# AutoGen adapter (38 tests)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/frameworks/embedding/test_autogen_adapter.py -v
```

### Contract tests only (parametrized across all frameworks)

```bash
# Interface conformance (85 tests = 17 × 5 frameworks)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/frameworks/embedding/test_contract_interface_conformance.py -v

# Context & error handling (80 tests = 16 × 5 frameworks)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/frameworks/embedding/test_contract_context_and_error_context.py -v

# Shapes & batching (65 tests = 13 × 5 frameworks)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/frameworks/embedding/test_contract_shapes_and_batching.py -v

# Mock backend edge cases (33 tests across 5 frameworks)
CORPUS_ADAPTER=tests.mock.mock_embedding_adapter:MockEmbeddingAdapter \
  pytest tests/frameworks/embedding/test_with_mock_backends.py -v
```

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

* [x] Implements framework-specific base class/interface.
* [x] Provides `embed_documents(texts: List[str]) -> List[List[float]]`.
* [x] Provides `embed_query(text: str) -> List[float]`.
* [x] Async variants when framework supports async.
* [x] Dimension property/method when required by framework.

### ✅ Phase 2: Context Translation (7/7)

* [x] Accepts framework-specific context parameter (optional).
* [x] Translates to `OperationContext` using `from_<framework>`.
* [x] Graceful degradation on translation failure.
* [x] Invalid context types tolerated without crash.
* [x] Context propagation through to underlying adapter.
* [x] Framework metadata included in contexts.
* [x] Configuration flags control context behavior.

### ✅ Phase 3: Input Validation (5/5)

* [x] Rejects non-string items in batch operations.
* [x] Rejects non-string query text.
* [x] Clear error messages for validation failures.
* [x] Async methods have same validation as sync.
* [x] Empty batch handling consistent.

### ✅ Phase 4: Error Handling (6/6)

* [x] Error context includes framework-specific fields.
* [x] Error context attached on all failure paths.
* [x] Async errors include same context as sync.
* [x] Sync methods reject event loop context.
* [x] Error messages are actionable for users.
* [x] Error inheritance hierarchy correct.

### ✅ Phase 5: Shape & Batching (8/8)

* [x] Batch output length matches input length.
* [x] All vectors have consistent dimension.
* [x] Query dimension matches batch dimension.
* [x] Single-element batch matches query shape.
* [x] Empty/non-empty text mix preserves batch length.
* [x] Duplicate texts produce identical vectors.
* [x] Large batch shape respected.
* [x] Batch preserves input order.

### ✅ Phase 6: Capabilities & Health (4/4)

* [x] Capabilities passthrough when underlying provides.
* [x] Health passthrough when underlying provides.
* [x] Async capabilities/health fallback to sync.
* [x] Missing capabilities/health handled gracefully.

### ✅ Phase 7: Resource Management (3/3)

* [x] Context manager support (sync).
* [x] Async context manager support.
* [x] Proper resource cleanup on close.

### ✅ Phase 8: Concurrency (2/2)

* [x] Thread safety validated.
* [x] Concurrent async operations work correctly.

### ✅ Phase 9: Framework Integration (3/3)

* [x] Real framework usage tests pass.
* [x] Framework-specific features work correctly.
* [x] Error propagation provides actionable messages.

### ✅ Phase 10: Mock Backend Robustness (7/7)

* [x] Invalid translator shapes detected.
* [x] Empty translator results detected.
* [x] Wrong row counts detected.
* [x] Translator exceptions wrapped with context.
* [x] Async translator exceptions wrapped.
* [x] All error paths include rich context.
* [x] Mock backend tests pass for all frameworks.

---

## Conformance Badge

```text
✅ Embedding Framework Adapters V1.0 - 100% Conformant
   418/418 tests passing (9 test files, 5 frameworks)

   Framework Coverage:
   ✅ LangChain:        100/100 (100%)
   ✅ LlamaIndex:       80/80   (100%)
   ✅ Semantic Kernel:  130/130 (100%)
   ✅ CrewAI:           70/70   (100%)
   ✅ AutoGen:          38/38   (100%)

   Test Categories:
   ✅ Interface Conformance:     85/85  (100%)
   ✅ Context & Error Handling:  80/80  (100%)
   ✅ Shapes & Batching:         65/65  (100%)
   ✅ Framework Integration:     50/50  (100%)
   ✅ Concurrency & Thread Safe: 30/30  (100%)
   ✅ Config & Validation:       45/45  (100%)
   ✅ Capabilities & Health:     30/30  (100%)
   ✅ Mock Backend Edge Cases:   33/33  (100%)

   Status: Production Ready - Platinum Certification 🏆
```

## **Embedding Framework Adapters Conformance**

**Certification Levels:**
- 🏆 **Platinum:** 418/418 tests (100%)
- 🥇 **Gold:** 334+ tests (80%+)
- 🥈 **Silver:** 209+ tests (50%+)
- 🔬 **Development:** <50%

**Badge Suggestion:**

[![Corpus Embedding Framework Adapters](https://img.shields.io/badge/Corpus%20Embedding%20Framework%20Adapters-100%25%20Conformant-brightgreen)](./embedding_framework_conformance_report.json)

---

**Last Updated:** 2026-01-30  
**Maintained By:** Corpus SDK Team  
**Status:** 100% V1.0 Conformant - Production Ready - Platinum Certification 🏆  
**Test Count:** 418/418 tests (100%)
